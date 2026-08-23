#!/usr/bin/env python3
# ============================================================
# RSNA KNEE ABNORMALITY DETECTION
# FINAL W3 SUBMISSION / INFERENCE SCRIPT
#
# Produces TWO competition-ready submissions in one inference run:
#
#   A) PRIMARY / CONTROLLED
#      W3.0 5-fold ensemble for all 12 labels
#
#   B) EXPLORATORY HYBRID
#      W3.0 ensemble for 10 labels
#      W3.1 ensemble for:
#          - Effusion
#          - Fracture
#
# IMPORTANT:
# - The hybrid is post-hoc/exploratory because Effusion/Fracture
#   were chosen after examining W3.1 OOF deltas.
# - submission.csv is intentionally the PRIMARY W3.0 submission.
#
# Inference pipeline:
#
#   test DICOMs
#      -> exact W3.0 preprocessing
#      -> frozen ImageNet ResNet18
#      -> cache [series, 16, 512] FP16 features once
#      -> load W3.0 folds 1..5
#      -> load W3.1 folds 1..5
#      -> mean probabilities across folds
#      -> validate against sample_submission.csv
#
# MODES
# -----
# 1) Final inference (default):
#
#    python rsna_w3_final_submission.py --mode infer
#
# 2) Export a compact Kaggle Dataset bundle from the CURRENT
#    development session:
#
#    python rsna_w3_final_submission.py --mode export_bundle
#
#    This copies only:
#      - 5 W3.0 head checkpoints
#      - 5 W3.1 head checkpoints
#      - ImageNet ResNet18 state_dict
#      - small result/verification JSONs when available
#
#    into:
#      /kaggle/working/rsna_w3_submission_bundle/
#
#    For a Kaggle code-submission rerun, create/attach a Kaggle
#    Dataset from that folder. The inference mode auto-discovers
#    the attached bundle under /kaggle/input.
#
# No W2/W2.3 files are needed at test inference time.
# No train feature cache is needed at test inference time.
# ============================================================

from __future__ import annotations

import os
import gc
import json
import time
import math
import shutil
import argparse
import hashlib
from pathlib import Path
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

import pydicom

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "RSNA_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)

TEST_CSV = DATA_ROOT / "test.csv"
TEST_SERIES_CSV = DATA_ROOT / "test_series.csv"
TEST_SERIES_ROOT = DATA_ROOT / "test_series"
SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"

WORK_ROOT = Path(
    os.environ.get(
        "RSNA_SUBMISSION_WORK_ROOT",
        "/kaggle/working/rsna_w3_submission",
    )
)

TEST_FEATURE_CACHE_ROOT = WORK_ROOT / "test_feature_cache"

RESULT_ROOT = WORK_ROOT / "results"

BUNDLE_EXPORT_ROOT = Path(
    os.environ.get(
        "RSNA_BUNDLE_EXPORT_ROOT",
        "/kaggle/working/rsna_w3_submission_bundle",
    )
)

for path in [
    WORK_ROOT,
    TEST_FEATURE_CACHE_ROOT,
    RESULT_ROOT,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# Exact W3.0 preprocessing/model constants
# ------------------------------------------------------------

IMAGE_SIZE = 224
SLICES_PER_SERIES = 16
FEATURE_DIM = 512
SERIES_META_DIM = 9
NUM_LABELS = 12
NUM_FOLDS = 5
FINAL_EPOCH = 12

USE_AMP = True
USE_SERIES_METADATA = False

FEATURE_CACHE_VERSION = "w3_0_v4_resnet18_amp_fp16_v1"

TEST_FEATURE_CACHE_VERSION = (
    "w3_final_test_" + FEATURE_CACHE_VERSION + "_dicom_repair_v2"
)

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

HYBRID_W31_LABELS = [
    "Effusion",
    "Fracture",
]

UID_COLUMN = "StudyInstanceUID"


# ------------------------------------------------------------
# Speed
# ------------------------------------------------------------

PREPROCESS_THREADS = max(
    1,
    int(
        os.environ.get(
            "RSNA_TEST_PREPROCESS_THREADS",
            "4",
        )
    ),
)

ENCODER_BATCH_SIZE = max(
    16,
    int(
        os.environ.get(
            "RSNA_TEST_ENCODER_BATCH_SIZE",
            "256",
        )
    ),
)

INFERENCE_BATCH_SIZE = max(
    1,
    int(
        os.environ.get(
            "RSNA_TEST_INFERENCE_BATCH_SIZE",
            "16",
        )
    ),
)

INFERENCE_NUM_WORKERS = max(
    0,
    int(
        os.environ.get(
            "RSNA_TEST_INFERENCE_WORKERS",
            "2",
        )
    ),
)

USE_MULTI_GPU_ENCODER = os.environ.get(
    "RSNA_USE_MULTI_GPU_ENCODER",
    "1",
).strip() not in {
    "0",
    "false",
    "False",
}


# ------------------------------------------------------------
# Model verification references
# ------------------------------------------------------------

EXPECTED_W30_AUROC = 0.545918
EXPECTED_W31_CUDA_AUROC = 0.523402

VERIFY_METRIC_TOLERANCE = 0.003


# ------------------------------------------------------------
# Explicit overrides
# ------------------------------------------------------------

EXPLICIT_W30_CHECKPOINT_ROOT = os.environ.get(
    "RSNA_W30_CHECKPOINT_ROOT",
    "",
).strip()

EXPLICIT_W31_CHECKPOINT_ROOT = os.environ.get(
    "RSNA_W31_CHECKPOINT_ROOT",
    "",
).strip()

EXPLICIT_RESNET18_WEIGHTS = os.environ.get(
    "RSNA_RESNET18_WEIGHTS",
    "",
).strip()


# ============================================================
# 2. DEVICE / GENERIC HELPERS
# ============================================================

GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

DEVICE = torch.device("cuda:0" if GPU_COUNT > 0 else "cpu")

ENCODER_DEVICE_IDS = (
    list(
        range(
            min(
                GPU_COUNT,
                2,
            )
        )
    )
    if (USE_MULTI_GPU_ENCODER and GPU_COUNT >= 2)
    else []
)


def autocast_context():

    if USE_AMP and DEVICE.type == "cuda":

        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )

    return nullcontext()


def elapsed_string(
    seconds: float,
) -> str:

    seconds = max(
        0.0,
        float(seconds),
    )

    h = int(seconds // 3600)

    m = int((seconds % 3600) // 60)

    s = int(seconds % 60)

    return f"{h:02d}:" f"{m:02d}:" f"{s:02d}"


def human_bytes(
    value: int,
) -> str:

    size = float(value)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if size < 1024.0:

            return f"{size:.2f} {unit}"

        size /= 1024.0

    return f"{size:.2f} PB"


def atomic_torch_save(
    obj: Any,
    destination: Path,
) -> None:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = destination.with_suffix(destination.suffix + ".tmp")

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        destination,
    )


# ============================================================
# 3. CHECKPOINT / BUNDLE DISCOVERY
# ============================================================


def checkpoint_set_complete(
    root: Path,
) -> bool:

    return all(
        (root / f"fold_{fold}_epoch_{FINAL_EPOCH}.pt").exists()
        for fold in range(
            1,
            NUM_FOLDS + 1,
        )
    )


def checkpoint_kind(
    checkpoint_path: Path,
) -> Optional[str]:

    try:

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        config = payload.get(
            "config",
            {},
        )

        experiment = str(
            config.get(
                "experiment",
                "",
            )
        ).lower()

        pipeline = str(
            config.get(
                "pipeline",
                "",
            )
        ).lower()

        if "w3.1" in experiment or "weak_supervision" in experiment:

            return "w3_1"

        if "cached_frozen_resnet18_features" in pipeline:

            return "w3_0"

    except Exception:

        return None

    return None


def candidate_checkpoint_roots() -> List[Path]:

    candidates: List[Path] = []

    # Current development session.
    candidates.extend(
        [
            Path("/kaggle/working/rsna_w3_0/checkpoints"),
            Path("/kaggle/working/rsna_w3_1/checkpoints"),
        ]
    )

    # Recommended attached bundle layout.
    for base in [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]:

        if not base.exists():
            continue

        try:

            for fold1_path in base.rglob(f"fold_1_epoch_{FINAL_EPOCH}.pt"):

                candidates.append(fold1_path.parent)

        except Exception:

            pass

    # De-duplicate while preserving order.
    return list(dict.fromkeys(candidates))


def discover_checkpoint_root(
    kind: str,
) -> Path:

    if kind not in {
        "w3_0",
        "w3_1",
    }:

        raise ValueError(kind)

    explicit = (
        EXPLICIT_W30_CHECKPOINT_ROOT if kind == "w3_0" else EXPLICIT_W31_CHECKPOINT_ROOT
    )

    if explicit:

        root = Path(explicit)

        if not checkpoint_set_complete(root):

            raise FileNotFoundError(
                f"Explicit {kind} checkpoint root is incomplete:\n" f"{root}"
            )

        first = root / f"fold_1_epoch_{FINAL_EPOCH}.pt"

        detected = checkpoint_kind(first)

        if detected != kind:

            raise RuntimeError(
                f"Explicit root {root} appears to be "
                f"{detected!r}, expected {kind!r}."
            )

        return root

    for root in candidate_checkpoint_roots():

        if not checkpoint_set_complete(root):

            continue

        first = root / f"fold_1_epoch_{FINAL_EPOCH}.pt"

        detected = checkpoint_kind(first)

        if detected == kind:

            return root

    raise FileNotFoundError(
        f"Could not auto-discover a complete {kind} "
        "checkpoint directory.\n"
        "Either keep the current development outputs under "
        "/kaggle/working, attach the exported submission bundle, "
        f"or set the {'RSNA_W30_CHECKPOINT_ROOT' if kind == 'w3_0' else 'RSNA_W31_CHECKPOINT_ROOT'} "
        "environment variable."
    )


def find_result_json_near_checkpoint(
    checkpoint_root: Path,
    filename: str,
) -> Optional[Path]:

    candidates = [
        checkpoint_root.parent / "results" / filename,
        checkpoint_root.parent.parent / "results" / filename,
        checkpoint_root.parent / filename,
    ]

    for path in candidates:

        if path.exists():

            return path

    # Bundle may store verification files under verification/.
    bundle_parent = checkpoint_root.parent

    for path in [
        bundle_parent / "verification" / filename,
        bundle_parent.parent / "verification" / filename,
    ]:

        if path.exists():

            return path

    return None


def verify_development_metrics(
    w30_root: Path,
    w31_root: Path,
) -> Dict[
    str,
    Any,
]:

    result: Dict[
        str,
        Any,
    ] = {
        "w3_0": "verification_file_not_found",
        "w3_1": "verification_file_not_found",
    }

    w30_json = find_result_json_near_checkpoint(
        w30_root,
        "gold_reproduction_summary.json",
    )

    if w30_json is not None:

        with open(
            w30_json,
            "r",
            encoding="utf-8",
        ) as handle:

            payload = json.load(handle)

        auc = float(payload["pooled_oof"]["macro_AUROC"])

        result["w3_0"] = {
            "path": str(w30_json),
            "macro_AUROC": auc,
        }

        if abs(auc - EXPECTED_W30_AUROC) > VERIFY_METRIC_TOLERANCE:

            raise RuntimeError(
                "W3.0 verification metric does not match "
                "the selected final run. "
                f"Found {auc:.6f}, expected approximately "
                f"{EXPECTED_W30_AUROC:.6f}."
            )

    w31_json = find_result_json_near_checkpoint(
        w31_root,
        "W3_1_COMPARISON.json",
    )

    if w31_json is not None:

        with open(
            w31_json,
            "r",
            encoding="utf-8",
        ) as handle:

            payload = json.load(handle)

        auc = float(payload["w3_1"]["macro_AUROC"])

        result["w3_1"] = {
            "path": str(w31_json),
            "macro_AUROC": auc,
        }

        if abs(auc - EXPECTED_W31_CUDA_AUROC) > VERIFY_METRIC_TOLERANCE:

            raise RuntimeError(
                "W3.1 verification metric does not match "
                "the selected CUDA final run. "
                f"Found {auc:.6f}, expected approximately "
                f"{EXPECTED_W31_CUDA_AUROC:.6f}. "
                "This check prevents accidentally submitting the "
                "earlier CPU W3.1 checkpoints."
            )

    return result


# ============================================================
# 4. RESNET18 WEIGHTS DISCOVERY
# ============================================================

RESNET18_BUNDLE_FILENAME = "resnet18_imagenet1k_v1_state_dict.pth"

OFFICIAL_RESNET18_FILENAME = "resnet18-f37072fd.pth"


def discover_resnet18_state_dict_file() -> Optional[Path]:

    candidates: List[Path] = []

    if EXPLICIT_RESNET18_WEIGHTS:

        candidates.append(Path(EXPLICIT_RESNET18_WEIGHTS))

    candidates.extend(
        [
            Path("/root/.cache/torch/hub/checkpoints") / OFFICIAL_RESNET18_FILENAME,
            Path("/kaggle/working") / RESNET18_BUNDLE_FILENAME,
        ]
    )

    for base in [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]:

        if not base.exists():
            continue

        for filename in [
            RESNET18_BUNDLE_FILENAME,
            OFFICIAL_RESNET18_FILENAME,
        ]:

            try:

                for path in base.rglob(filename):

                    candidates.append(path)

            except Exception:

                pass

    for path in list(dict.fromkeys(candidates)):

        if path.exists():

            return path

    return None


def load_resnet18_backbone() -> nn.Module:
    """
    Load the exact torchvision ResNet18 ImageNet weights used in W3.0.

    Priority:
      1) explicit/local/attached state_dict
      2) torchvision DEFAULT (works if already cached or internet allowed)
    """

    state_path = discover_resnet18_state_dict_file()

    if state_path is not None:

        print(
            "ResNet18 weights file:",
            state_path,
        )

        backbone = resnet18(weights=None)

        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        # Handle common wrappers.
        if (
            isinstance(
                state,
                dict,
            )
            and "state_dict" in state
            and isinstance(
                state["state_dict"],
                dict,
            )
        ):

            state = state["state_dict"]

        if not isinstance(
            state,
            dict,
        ):

            raise RuntimeError(f"Unsupported ResNet18 state file: {state_path}")

        cleaned = {}

        for key, value in state.items():

            clean_key = str(key)

            for prefix in [
                "module.",
                "backbone.",
            ]:

                if clean_key.startswith(prefix):

                    clean_key = clean_key[len(prefix) :]

            cleaned[clean_key] = value

        missing, unexpected = backbone.load_state_dict(
            cleaned,
            strict=False,
        )

        if missing or unexpected:

            raise RuntimeError(
                "Attached ResNet18 state_dict is not the expected "
                "torchvision ResNet18 model.\n"
                f"Missing: {missing}\n"
                f"Unexpected: {unexpected}"
            )

        return backbone

    print(
        "No attached ResNet18 state file found; "
        "trying torchvision ResNet18_Weights.DEFAULT."
    )

    try:

        return resnet18(weights=ResNet18_Weights.DEFAULT)

    except Exception as exc:

        raise RuntimeError(
            "Could not load ImageNet ResNet18 weights. "
            "For Kaggle scoring with internet disabled, first run "
            "--mode export_bundle in the development session and "
            "attach that bundle as a Kaggle Dataset."
        ) from exc


# ============================================================
# 5. TEST TABLE / SUBMISSION SCHEMA VALIDATION
# ============================================================


def load_test_tables() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    for path in [
        TEST_CSV,
        TEST_SERIES_CSV,
        TEST_SERIES_ROOT,
        SAMPLE_SUBMISSION_CSV,
    ]:

        if not path.exists():

            raise FileNotFoundError(path)

    test_df = pd.read_csv(TEST_CSV)

    series_df = pd.read_csv(TEST_SERIES_CSV)

    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    for df, name in [
        (
            test_df,
            "test.csv",
        ),
        (
            series_df,
            "test_series.csv",
        ),
        (
            sample_df,
            "sample_submission.csv",
        ),
    ]:

        if UID_COLUMN not in (df.columns):

            raise RuntimeError(f"{name} has no {UID_COLUMN} column.")

        df[UID_COLUMN] = df[UID_COLUMN].astype(str)

    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    sample_label_columns = [
        column for column in (sample_df.columns) if column != UID_COLUMN
    ]

    if (
        set(sample_label_columns) != set(LABEL_COLUMNS)
        or len(sample_label_columns) != NUM_LABELS
    ):

        raise RuntimeError(
            "sample_submission.csv label schema mismatch.\n"
            f"Expected: {LABEL_COLUMNS}\n"
            f"Found   : {sample_label_columns}"
        )

    if sample_df[UID_COLUMN].duplicated().any():

        raise RuntimeError("sample_submission.csv has duplicate study UIDs.")

    if test_df[UID_COLUMN].duplicated().any():

        raise RuntimeError("test.csv has duplicate study UIDs.")

    sample_uids = sample_df[UID_COLUMN].tolist()

    test_uids = test_df[UID_COLUMN].tolist()

    if set(sample_uids) != set(test_uids):

        raise RuntimeError("test.csv and sample_submission.csv UID sets differ.")

    series_uid_set = set(series_df[UID_COLUMN])

    missing_series = [uid for uid in sample_uids if uid not in (series_uid_set)]

    if missing_series:

        raise RuntimeError(
            f"{len(missing_series)} test studies have no "
            "test_series.csv rows. "
            f"Example: {missing_series[0]}"
        )

    return (
        test_df,
        series_df,
        sample_df,
    )


# ============================================================
# 6. EXACT W3.0 DICOM PREPROCESSING
# ============================================================


def get_scalar_slice_position(
    ds,
) -> Optional[float]:

    orientation = getattr(
        ds,
        "ImageOrientationPatient",
        None,
    )

    position = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    if orientation is None or position is None:

        return None

    try:

        row = np.asarray(
            orientation[:3],
            dtype=np.float64,
        )

        col = np.asarray(
            orientation[3:],
            dtype=np.float64,
        )

        normal = np.cross(
            row,
            col,
        )

        position_array = np.asarray(
            position,
            dtype=np.float64,
        )

        return float(
            np.dot(
                position_array,
                normal,
            )
        )

    except Exception:

        return None


def robust_series_normalize(
    images: np.ndarray,
) -> np.ndarray:

    finite = images[np.isfinite(images)]

    if finite.size == 0:

        return np.zeros_like(
            images,
            dtype=np.float32,
        )

    low = np.percentile(
        finite,
        1.0,
    )

    high = np.percentile(
        finite,
        99.0,
    )

    if high <= low:

        return np.zeros_like(
            images,
            dtype=np.float32,
        )

    images = np.nan_to_num(
        images,
        nan=low,
        posinf=high,
        neginf=low,
    )

    images = np.clip(
        images,
        low,
        high,
    )

    images = (images - low) / (high - low)

    return images.astype(np.float32)


def resize_and_pad(
    image: torch.Tensor,
    size: int,
) -> torch.Tensor:

    if image.ndim != 3:

        raise ValueError("Expected [1,H,W], " f"got {image.shape}")

    _, h, w = image.shape

    scale = min(
        size
        / max(
            h,
            1,
        ),
        size
        / max(
            w,
            1,
        ),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(
            new_h,
            new_w,
        ),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    canvas = torch.zeros(
        1,
        size,
        size,
        dtype=resized.dtype,
    )

    top = (size - new_h) // 2

    left = (size - new_w) // 2

    canvas[
        :,
        top : top + new_h,
        left : left + new_w,
    ] = resized

    return canvas


def decode_dicom_pixel_array(
    ds,
    dicom_path: str,
) -> np.ndarray:
    """
    Robust decoder copied from the W3.0 fixed-v2 cache path.

    Normal:
        ds.pixel_array

    Repair 1:
        native/uncompressed NumberOfFrames mismatch.

    Repair 2:
        native MONOCHROME single-frame BitsAllocated mismatch.

    Source bytes are never changed, padded, or truncated.
    Compressed transfer syntaxes are never guessed.
    """

    try:

        return ds.pixel_array

    except ValueError as first_error:

        transfer_syntax = None

        try:

            transfer_syntax = ds.file_meta.TransferSyntaxUID

        except Exception:

            transfer_syntax = None

        is_compressed = False

        try:

            if transfer_syntax is not None:

                is_compressed = bool(transfer_syntax.is_compressed)

        except Exception:

            is_compressed = False

        if is_compressed or "PixelData" not in ds:

            raise

        original_number_of_frames = getattr(
            ds,
            "NumberOfFrames",
            None,
        )

        original_bits_allocated = getattr(
            ds,
            "BitsAllocated",
            None,
        )

        original_samples_per_pixel = getattr(
            ds,
            "SamplesPerPixel",
            None,
        )

        original_high_bit = getattr(
            ds,
            "HighBit",
            None,
        )

        def restore_metadata() -> None:

            if original_number_of_frames is None:

                if "NumberOfFrames" in ds:

                    del ds.NumberOfFrames

            else:

                ds.NumberOfFrames = original_number_of_frames

            if original_bits_allocated is not None:

                ds.BitsAllocated = original_bits_allocated

            if original_samples_per_pixel is not None:

                ds.SamplesPerPixel = original_samples_per_pixel

            if original_high_bit is not None:

                ds.HighBit = original_high_bit

        try:

            rows = int(ds.Rows)

            columns = int(ds.Columns)

            samples = int(
                getattr(
                    ds,
                    "SamplesPerPixel",
                    1,
                )
                or 1
            )

            bits_allocated = int(ds.BitsAllocated)

            bits_stored = int(
                getattr(
                    ds,
                    "BitsStored",
                    bits_allocated,
                )
                or bits_allocated
            )

            photometric = str(
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    "",
                )
            ).upper()

            actual_bytes = len(ds.PixelData)

            if rows <= 0 or columns <= 0 or samples <= 0 or actual_bytes <= 0:

                raise first_error

            # --------------------------------------------------
            # Repair 1: NumberOfFrames only
            # --------------------------------------------------

            if bits_allocated > 0 and bits_allocated % 8 == 0:

                bytes_per_frame = rows * columns * samples * (bits_allocated // 8)

                usable_bytes = actual_bytes

                if (
                    bytes_per_frame > 0
                    and usable_bytes % bytes_per_frame != 0
                    and usable_bytes > 0
                    and (usable_bytes - 1) % bytes_per_frame == 0
                ):

                    usable_bytes -= 1

                if (
                    bytes_per_frame > 0
                    and usable_bytes > 0
                    and usable_bytes % bytes_per_frame == 0
                ):

                    inferred_frames = usable_bytes // bytes_per_frame

                    declared_frames = int(
                        getattr(
                            ds,
                            "NumberOfFrames",
                            1,
                        )
                        or 1
                    )

                    if inferred_frames >= 1 and inferred_frames != declared_frames:

                        ds.NumberOfFrames = int(inferred_frames)

                        repaired = ds.pixel_array

                        if repaired.ndim == 2:

                            print(
                                "[DICOM-REPAIR] "
                                f"{dicom_path} | "
                                "NumberOfFrames "
                                f"{declared_frames} -> "
                                f"{inferred_frames} | "
                                f"BitsAllocated="
                                f"{bits_allocated} | "
                                f"PixelData="
                                f"{actual_bytes} bytes"
                            )

                            return repaired

                        restore_metadata()

            # --------------------------------------------------
            # Repair 2: infer single-frame native storage width
            # --------------------------------------------------

            restore_metadata()

            if not photometric.startswith("MONOCHROME"):

                raise first_error

            base_pixels = rows * columns * samples

            usable_bytes = actual_bytes

            if (
                usable_bytes % base_pixels != 0
                and usable_bytes > 0
                and (usable_bytes - 1) % base_pixels == 0
            ):

                usable_bytes -= 1

            if base_pixels <= 0 or usable_bytes <= 0 or usable_bytes % base_pixels != 0:

                raise first_error

            inferred_bytes_per_sample = usable_bytes // base_pixels

            inferred_bits_allocated = inferred_bytes_per_sample * 8

            if (
                inferred_bits_allocated
                not in {
                    8,
                    16,
                    32,
                    64,
                }
                or bits_stored > inferred_bits_allocated
            ):

                raise first_error

            declared_frames = int(
                getattr(
                    ds,
                    "NumberOfFrames",
                    1,
                )
                or 1
            )

            ds.NumberOfFrames = 1

            ds.BitsAllocated = int(inferred_bits_allocated)

            if bits_stored > 0 and (
                original_high_bit is None
                or int(original_high_bit) >= inferred_bits_allocated
            ):

                ds.HighBit = bits_stored - 1

            repaired = ds.pixel_array

            if repaired.ndim != 2:

                restore_metadata()

                raise first_error

            print(
                "[DICOM-REPAIR] "
                f"{dicom_path} | "
                f"NumberOfFrames "
                f"{declared_frames} -> 1 | "
                f"BitsAllocated "
                f"{bits_allocated} -> "
                f"{inferred_bits_allocated} | "
                f"BitsStored={bits_stored} | "
                f"Rows={rows} Cols={columns} "
                f"Samples={samples} | "
                f"PixelData={actual_bytes} bytes"
            )

            return repaired

        except Exception as repair_error:

            restore_metadata()

            if repair_error is first_error:

                raise

            raise RuntimeError(
                "DICOM pixel decode failed and safe native "
                "metadata repair was not possible. "
                f"Path: {dicom_path}. "
                f"Rows={getattr(ds, 'Rows', None)}, "
                f"Cols={getattr(ds, 'Columns', None)}, "
                f"SamplesPerPixel="
                f"{getattr(ds, 'SamplesPerPixel', None)}, "
                f"BitsAllocated="
                f"{getattr(ds, 'BitsAllocated', None)}, "
                f"BitsStored="
                f"{getattr(ds, 'BitsStored', None)}, "
                f"HighBit="
                f"{getattr(ds, 'HighBit', None)}, "
                f"NumberOfFrames="
                f"{getattr(ds, 'NumberOfFrames', None)}, "
                f"Photometric="
                f"{getattr(ds, 'PhotometricInterpretation', None)}, "
                f"PixelDataBytes="
                f"{len(ds.PixelData) if 'PixelData' in ds else None}. "
                f"Original error: {first_error}. "
                f"Repair error: {repair_error}"
            ) from first_error


def load_preprocessed_series_images(
    study_uid: str,
    series_uid: str,
) -> torch.Tensor:
    """
    Exact W3.0 series preprocessing + fixed-v2 bad-DICOM fallback.

    Returns:
        [16, 3, 224, 224] float32
    """

    series_dir = TEST_SERIES_ROOT / study_uid / series_uid

    dcm_paths = sorted(series_dir.glob("*.dcm"))

    if not dcm_paths:

        raise FileNotFoundError("No DICOM files found:\n" f"{series_dir}")

    headers: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for path in dcm_paths:

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )

        headers.append(
            {
                "path": str(path),
                "position": get_scalar_slice_position(ds),
                "instance": getattr(
                    ds,
                    "InstanceNumber",
                    0,
                ),
            }
        )

    all_spatial = all(item["position"] is not None for item in headers)

    if all_spatial:

        headers.sort(key=lambda x: x["position"])

    else:

        headers.sort(
            key=lambda x: float(x["instance"] if x["instance"] is not None else 0)
        )

    n = len(headers)

    selected_indices = (
        np.linspace(
            0,
            n - 1,
            SLICES_PER_SERIES,
        )
        .round()
        .astype(int)
    )

    raw_images: List[np.ndarray] = []

    for target_index in selected_indices:

        candidate_indices = [int(target_index)]

        for distance in range(
            1,
            n,
        ):

            left = int(target_index) - distance

            right = int(target_index) + distance

            if left >= 0:

                candidate_indices.append(left)

            if right < n:

                candidate_indices.append(right)

            if left < 0 and right >= n:

                break

        decoded = False
        last_error = None

        for candidate_index in candidate_indices:

            item = headers[candidate_index]

            try:

                ds = pydicom.dcmread(
                    item["path"],
                    force=True,
                )

                image = decode_dicom_pixel_array(
                    ds,
                    item["path"],
                )

                # Competition data are expected to be one 2-D image
                # per DICOM file. Permit a singleton frame dimension.
                if image.ndim == 3 and image.shape[0] == 1:

                    image = image[0]

                if image.ndim != 2:

                    raise RuntimeError(
                        "Expected a single 2-D MRI slice, " f"got shape {image.shape}."
                    )

                image = image.astype(np.float32)

                slope = float(
                    getattr(
                        ds,
                        "RescaleSlope",
                        1.0,
                    )
                )

                intercept = float(
                    getattr(
                        ds,
                        "RescaleIntercept",
                        0.0,
                    )
                )

                image = image * slope + intercept

                if candidate_index != int(target_index):

                    print(
                        "[DICOM-SUBSTITUTE] "
                        f"study={study_uid} "
                        f"series={series_uid} | "
                        f"selected_index="
                        f"{int(target_index)} "
                        f"-> nearest_valid_index="
                        f"{candidate_index} | "
                        f"path={item['path']}"
                    )

                raw_images.append(image)

                decoded = True

                break

            except Exception as exc:

                last_error = exc

        if not decoded:

            raise RuntimeError(
                "No decodable DICOM slice remained in series "
                f"{series_uid} for selected index "
                f"{int(target_index)}."
            ) from last_error

    volume = np.stack(
        raw_images,
        axis=0,
    )

    volume = robust_series_normalize(volume)

    processed: List[torch.Tensor] = []

    for image in volume:

        tensor = torch.from_numpy(image).unsqueeze(0)

        tensor = resize_and_pad(
            tensor,
            IMAGE_SIZE,
        )

        tensor = tensor.repeat(
            3,
            1,
            1,
        )

        processed.append(tensor)

    images = torch.stack(
        processed,
        dim=0,
    ).contiguous()

    # W3.0 intentionally emulated the V4 image-cache quantization.
    images = images.half().float()

    return images


# ============================================================
# 7. FROZEN RESNET18 FEATURE ENCODER
# ============================================================


class FrozenResNet18Encoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
    ):

        super().__init__()

        self.feature_dim = backbone.fc.in_features

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.register_buffer(
            "mean",
            torch.tensor(
                [
                    0.485,
                    0.456,
                    0.406,
                ],
                dtype=torch.float32,
            ).view(
                1,
                3,
                1,
                1,
            ),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [
                    0.229,
                    0.224,
                    0.225,
                ],
                dtype=torch.float32,
            ).view(
                1,
                3,
                1,
                1,
            ),
        )

        for parameter in self.backbone.parameters():

            parameter.requires_grad = False

        self.backbone.eval()

    def train(
        self,
        mode: bool = True,
    ):

        super().train(mode)

        self.backbone.eval()

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = (x - self.mean) / self.std

        with torch.no_grad():

            features = self.backbone(x).flatten(1)

        return features


def encode_feature_batches(
    images: torch.Tensor,
    encoder: nn.Module,
) -> torch.Tensor:

    outputs: List[torch.Tensor] = []

    with torch.inference_mode():

        for start in range(
            0,
            images.shape[0],
            ENCODER_BATCH_SIZE,
        ):

            end = min(
                images.shape[0],
                start + ENCODER_BATCH_SIZE,
            )

            chunk = images[start:end].to(
                DEVICE,
                non_blocking=True,
            )

            with autocast_context():

                feature_chunk = encoder(chunk)

            outputs.append(
                feature_chunk.detach().to(
                    "cpu",
                    dtype=torch.float16,
                )
            )

            del chunk
            del feature_chunk

    return torch.cat(
        outputs,
        dim=0,
    ).contiguous()


# ============================================================
# 8. TEST FEATURE CACHE
# ============================================================


def make_series_lookup(
    series_df: pd.DataFrame,
) -> Dict[
    str,
    pd.DataFrame,
]:

    lookup: Dict[
        str,
        pd.DataFrame,
    ] = {}

    for study_uid, group in series_df.groupby(UID_COLUMN):

        lookup[str(study_uid)] = group.sort_values("SeriesInstanceUID").reset_index(
            drop=True
        )

    return lookup


def test_feature_cache_path(
    study_uid: str,
) -> Path:

    digest = hashlib.md5(str(study_uid).encode("utf-8")).hexdigest()

    return TEST_FEATURE_CACHE_ROOT / f"{digest}.pt"


def test_cache_is_valid(
    path: Path,
    study_uid: str,
) -> bool:

    if not path.exists():

        return False

    try:

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if payload.get("cache_version") != TEST_FEATURE_CACHE_VERSION:

            return False

        if str(payload.get("study_uid")) != str(study_uid):

            return False

        features = payload.get("features")

        if (
            not isinstance(
                features,
                torch.Tensor,
            )
            or features.ndim != 3
            or features.shape[1] != SLICES_PER_SERIES
            or features.shape[2] != FEATURE_DIM
        ):

            return False

        return True

    except Exception:

        return False


def encode_and_save_test_study(
    study_uid: str,
    series_lookup: Dict[
        str,
        pd.DataFrame,
    ],
    encoder: nn.Module,
    executor: ThreadPoolExecutor,
) -> Dict[
    str,
    Any,
]:

    path = test_feature_cache_path(study_uid)

    if test_cache_is_valid(
        path,
        study_uid,
    ):

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        return {
            UID_COLUMN: study_uid,
            "SeriesCount": int(payload["features"].shape[0]),
            "CacheBytes": int(path.stat().st_size),
            "Status": "reused",
        }

    if study_uid not in (series_lookup):

        raise RuntimeError("No test_series.csv rows for " f"study {study_uid}.")

    rows = series_lookup[study_uid]

    series_uids = rows["SeriesInstanceUID"].astype(str).tolist()

    tasks = [
        (
            study_uid,
            series_uid,
        )
        for series_uid in (series_uids)
    ]

    try:

        image_list = list(
            executor.map(
                lambda pair: load_preprocessed_series_images(
                    pair[0],
                    pair[1],
                ),
                tasks,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed while preprocessing TEST study "
            f"{study_uid}. "
            f"Series list: {series_uids}"
        ) from exc

    images = torch.stack(
        image_list,
        dim=0,
    )

    s, k, c, h, w = images.shape

    flat_images = images.reshape(
        s * k,
        c,
        h,
        w,
    )

    flat_features = encode_feature_batches(
        flat_images,
        encoder,
    )

    features = flat_features.reshape(
        s,
        k,
        -1,
    ).contiguous()

    if features.shape[2] != FEATURE_DIM:

        raise RuntimeError(
            "Expected ResNet18 feature dim 512, " f"got {tuple(features.shape)}."
        )

    payload = {
        "cache_version": TEST_FEATURE_CACHE_VERSION,
        "study_uid": study_uid,
        "series_uids": series_uids,
        "features": features,
        "feature_dtype": "float16",
        "image_size": IMAGE_SIZE,
        "slices_per_series": SLICES_PER_SERIES,
        "backbone": "ResNet18 ImageNet1K V1",
    }

    atomic_torch_save(
        payload,
        path,
    )

    record = {
        UID_COLUMN: study_uid,
        "SeriesCount": int(s),
        "CacheBytes": int(path.stat().st_size),
        "Status": "created",
    }

    del image_list
    del images
    del flat_images
    del flat_features
    del features
    del payload

    return record


def build_test_feature_cache(
    study_uids: List[str],
    series_lookup: Dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    print("\n")
    print("=" * 88)
    print("BUILDING / RESUMING TEST FEATURE CACHE")
    print("=" * 88)

    missing = [
        uid
        for uid in (study_uids)
        if not test_cache_is_valid(
            test_feature_cache_path(uid),
            uid,
        )
    ]

    print(f"Test studies requested : " f"{len(study_uids)}")

    print(f"Already cached         : " f"{len(study_uids) - len(missing)}")

    print(f"Need encoding          : " f"{len(missing)}")

    print(f"Preprocess threads     : " f"{PREPROCESS_THREADS}")

    print(f"Encoder batch size     : " f"{ENCODER_BATCH_SIZE}")

    print(f"CUDA GPUs              : " f"{GPU_COUNT}")

    if not missing:

        records = []

        for uid in study_uids:

            path = test_feature_cache_path(uid)

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            records.append(
                {
                    UID_COLUMN: uid,
                    "SeriesCount": int(payload["features"].shape[0]),
                    "CacheBytes": int(path.stat().st_size),
                    "Status": "reused",
                }
            )

        manifest = pd.DataFrame(records)

        manifest.to_csv(
            RESULT_ROOT / "test_feature_manifest.csv",
            index=False,
        )

        return manifest

    if DEVICE.type != "cuda":

        print(
            "WARNING: test feature extraction is running on CPU. "
            "This is valid but will be much slower."
        )

    backbone = load_resnet18_backbone()

    encoder_base = FrozenResNet18Encoder(backbone).to(DEVICE)

    encoder_base.eval()

    if len(ENCODER_DEVICE_IDS) >= 2:

        encoder: nn.Module = nn.DataParallel(
            encoder_base,
            device_ids=ENCODER_DEVICE_IDS,
            output_device=ENCODER_DEVICE_IDS[0],
            dim=0,
        )

        print("Frozen ResNet18 encoder: " f"DataParallel GPUs " f"{ENCODER_DEVICE_IDS}")

    else:

        encoder = encoder_base

        print(
            "Frozen ResNet18 encoder:",
            DEVICE,
        )

    records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    started = time.time()

    with ThreadPoolExecutor(max_workers=PREPROCESS_THREADS) as executor:

        for index, uid in enumerate(
            study_uids,
            start=1,
        ):

            record = encode_and_save_test_study(
                uid,
                series_lookup,
                encoder,
                executor,
            )

            records.append(record)

            if index <= 5 or index % 25 == 0 or index == len(study_uids):

                elapsed = time.time() - started

                rate = index / max(
                    elapsed,
                    1e-6,
                )

                remaining = (len(study_uids) - index) / max(
                    rate,
                    1e-6,
                )

                print(
                    f"  {index:5d}/"
                    f"{len(study_uids):5d} "
                    f"({100.0 * index / len(study_uids):5.1f}%) "
                    f"elapsed={elapsed_string(elapsed)} "
                    f"ETA={elapsed_string(remaining)} "
                    f"last={record['Status']}"
                )

    del encoder
    del encoder_base
    del backbone

    gc.collect()

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

    manifest = pd.DataFrame(records)

    manifest.to_csv(
        RESULT_ROOT / "test_feature_manifest.csv",
        index=False,
    )

    total_bytes = int(manifest["CacheBytes"].sum())

    print(
        "Test feature cache size:",
        human_bytes(total_bytes),
    )

    return manifest


# ============================================================
# 9. TEST FEATURE DATASET
# ============================================================


def load_test_cached_features(
    study_uid: str,
) -> torch.Tensor:

    path = test_feature_cache_path(study_uid)

    if not test_cache_is_valid(
        path,
        study_uid,
    ):

        raise RuntimeError("Invalid/missing test feature cache for " f"{study_uid}.")

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return payload["features"].float().contiguous()


class TestFeatureDataset(Dataset):
    def __init__(
        self,
        study_uids: List[str],
    ):

        self.study_uids = [str(uid) for uid in (study_uids)]

    def __len__(
        self,
    ) -> int:

        return len(self.study_uids)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[
        str,
        Any,
    ]:

        uid = self.study_uids[index]

        return {
            "study_uid": uid,
            "features": load_test_cached_features(uid),
        }


def collate_test_features(
    batch: List[
        Dict[
            str,
            Any,
        ]
    ],
) -> Dict[
    str,
    Any,
]:

    b = len(batch)

    max_series = max(item["features"].shape[0] for item in (batch))

    features = torch.zeros(
        b,
        max_series,
        SLICES_PER_SERIES,
        FEATURE_DIM,
        dtype=torch.float32,
    )

    metadata = torch.zeros(
        b,
        max_series,
        SERIES_META_DIM,
        dtype=torch.float32,
    )

    series_mask = torch.zeros(
        b,
        max_series,
        dtype=torch.bool,
    )

    uids: List[str] = []

    for i, item in enumerate(batch):

        s = item["features"].shape[0]

        features[
            i,
            :s,
        ] = item["features"]

        series_mask[
            i,
            :s,
        ] = True

        uids.append(item["study_uid"])

    return {
        "study_uid": uids,
        "features": features,
        "metadata": metadata,
        "series_mask": series_mask,
    }


# ============================================================
# 10. EXACT W3.0/W3.1 HEAD ARCHITECTURE
# ============================================================


class SliceAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):

        super().__init__()

        hidden = max(
            64,
            embedding_dim // 2,
        )

        self.score = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        scores = self.score(x).squeeze(-1)

        attention = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * attention.unsqueeze(-1),
            dim=1,
        )

        return (
            pooled,
            attention,
        )


class SeriesAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):

        super().__init__()

        hidden = max(
            64,
            embedding_dim // 2,
        )

        self.score = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        scores = self.score(x).squeeze(-1)

        scores = scores.float()

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        attention = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * attention.unsqueeze(-1),
            dim=1,
        )

        return (
            pooled,
            attention,
        )


class CachedV4MetadataAblationModel(nn.Module):
    def __init__(
        self,
        metadata_dim: int = SERIES_META_DIM,
        num_labels: int = NUM_LABELS,
    ):

        super().__init__()

        visual_dim = FEATURE_DIM

        self.embedding_dim = 256

        self.visual_projection = nn.Sequential(
            nn.Linear(
                visual_dim,
                self.embedding_dim,
            ),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )

        self.metadata_projection = nn.Sequential(
            nn.Linear(
                metadata_dim,
                64,
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(
                64,
                128,
            ),
            nn.GELU(),
        )

        self.series_fusion = nn.Sequential(
            nn.Linear(
                self.embedding_dim + 128,
                self.embedding_dim,
            ),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.slice_attention = SliceAttention(self.embedding_dim)

        self.series_attention = SeriesAttention(self.embedding_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.15),
            nn.Linear(
                self.embedding_dim,
                num_labels,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        metadata: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:

        b, s, k, d = features.shape

        flat_features = features.reshape(
            b * s * k,
            d,
        )

        flat_features = self.visual_projection(flat_features)

        slice_features = flat_features.reshape(
            b * s,
            k,
            self.embedding_dim,
        )

        (
            series_visual,
            _,
        ) = self.slice_attention(slice_features)

        series_visual = series_visual.reshape(
            b,
            s,
            self.embedding_dim,
        )

        if USE_SERIES_METADATA:

            metadata_input = metadata

        else:

            metadata_input = torch.zeros_like(metadata)

        metadata_features = self.metadata_projection(metadata_input)

        combined = torch.cat(
            [
                series_visual,
                metadata_features,
            ],
            dim=-1,
        )

        series_features = self.series_fusion(combined)

        (
            study_embedding,
            _,
        ) = self.series_attention(
            series_features,
            series_mask,
        )

        logits = self.classifier(study_embedding)

        return logits


# ============================================================
# 11. LOAD 5+5 FOLD MODELS
# ============================================================


def load_head_ensemble(
    checkpoint_root: Path,
    expected_kind: str,
) -> List[nn.Module]:

    models: List[nn.Module] = []

    for fold in range(
        1,
        NUM_FOLDS + 1,
    ):

        path = checkpoint_root / f"fold_{fold}_epoch_{FINAL_EPOCH}.pt"

        if not path.exists():

            raise FileNotFoundError(path)

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            int(
                payload.get(
                    "fold",
                    -1,
                )
            )
            != fold
        ):

            raise RuntimeError(f"Checkpoint fold mismatch: {path}")

        if (
            int(
                payload.get(
                    "epoch",
                    -1,
                )
            )
            != FINAL_EPOCH
        ):

            raise RuntimeError(f"Checkpoint epoch mismatch: {path}")

        detected = checkpoint_kind(path)

        if detected != expected_kind:

            raise RuntimeError(
                f"Checkpoint kind mismatch for {path}: "
                f"{detected!r} vs expected {expected_kind!r}"
            )

        model = CachedV4MetadataAblationModel().to(DEVICE)

        state = payload["model_state_dict"]

        model.load_state_dict(
            state,
            strict=True,
        )

        model.eval()

        models.append(model)

    print(
        f"Loaded {len(models)} {expected_kind} "
        f"fold models from:\n  {checkpoint_root}"
    )

    return models


# ============================================================
# 12. TEST INFERENCE
# ============================================================


@torch.inference_mode()
def predict_test(
    study_uids: List[str],
    w30_models: List[nn.Module],
    w31_models: List[nn.Module],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    dataset = TestFeatureDataset(study_uids)

    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_test_features,
        persistent_workers=(INFERENCE_NUM_WORKERS > 0),
    )

    n = len(study_uids)

    w30_fold_probs = np.zeros(
        (
            NUM_FOLDS,
            n,
            NUM_LABELS,
        ),
        dtype=np.float32,
    )

    w31_fold_probs = np.zeros(
        (
            NUM_FOLDS,
            n,
            NUM_LABELS,
        ),
        dtype=np.float32,
    )

    cursor = 0
    started = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        features = batch["features"].to(
            DEVICE,
            non_blocking=True,
        )

        metadata = batch["metadata"].to(
            DEVICE,
            non_blocking=True,
        )

        series_mask = batch["series_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        b = features.shape[0]

        with autocast_context():

            for fold_idx, model in enumerate(w30_models):

                logits = model(
                    features,
                    metadata,
                    series_mask,
                )

                probs = torch.sigmoid(logits).float().cpu().numpy()

                w30_fold_probs[
                    fold_idx,
                    cursor : cursor + b,
                    :,
                ] = probs

            for fold_idx, model in enumerate(w31_models):

                logits = model(
                    features,
                    metadata,
                    series_mask,
                )

                probs = torch.sigmoid(logits).float().cpu().numpy()

                w31_fold_probs[
                    fold_idx,
                    cursor : cursor + b,
                    :,
                ] = probs

        cursor += b

        if batch_index <= 3 or batch_index % 10 == 0 or cursor == n:

            elapsed = time.time() - started

            rate = cursor / max(
                elapsed,
                1e-6,
            )

            remaining = (n - cursor) / max(
                rate,
                1e-6,
            )

            print(
                f"Inference: "
                f"{cursor:5d}/{n:5d} "
                f"({100.0 * cursor / n:5.1f}%) "
                f"elapsed={elapsed_string(elapsed)} "
                f"ETA={elapsed_string(remaining)}"
            )

    if cursor != n:

        raise RuntimeError(f"Inference row count mismatch: " f"{cursor} vs {n}")

    w30_mean = w30_fold_probs.mean(axis=0)

    w31_mean = w31_fold_probs.mean(axis=0)

    return (
        w30_mean,
        w31_mean,
        w30_fold_probs,
        w31_fold_probs,
    )


# ============================================================
# 13. SUBMISSION BUILD / VALIDATION
# ============================================================


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    name: str,
) -> None:

    if submission.columns.tolist() != (sample.columns.tolist()):

        raise RuntimeError(
            f"{name}: column schema/order differs from " "sample_submission.csv."
        )

    if submission[UID_COLUMN].tolist() != (sample[UID_COLUMN].tolist()):

        raise RuntimeError(f"{name}: StudyInstanceUID order differs from sample.")

    if submission[UID_COLUMN].duplicated().any():

        raise RuntimeError(f"{name}: duplicate study UIDs.")

    values = submission[LABEL_COLUMNS].values.astype(np.float64)

    if not np.isfinite(values).all():

        raise RuntimeError(f"{name}: non-finite probability detected.")

    if values.min() < 0.0 or values.max() > 1.0:

        raise RuntimeError(f"{name}: probabilities outside [0,1].")


def make_submission_from_probs(
    sample: pd.DataFrame,
    probs: np.ndarray,
) -> pd.DataFrame:

    if probs.shape != (
        len(sample),
        NUM_LABELS,
    ):

        raise RuntimeError(f"Prediction shape mismatch: {probs.shape}")

    submission = sample.copy()

    for idx, label in enumerate(LABEL_COLUMNS):

        submission[label] = probs[
            :,
            idx,
        ]

    # Restore sample column order exactly.
    return submission[sample.columns.tolist()]


def write_prediction_audit(
    study_uids: List[str],
    w30_mean: np.ndarray,
    w31_mean: np.ndarray,
    hybrid: np.ndarray,
    w30_fold_probs: np.ndarray,
    w31_fold_probs: np.ndarray,
) -> None:

    audit = pd.DataFrame({UID_COLUMN: study_uids})

    for idx, label in enumerate(LABEL_COLUMNS):

        audit[f"W3_0__{label}"] = w30_mean[
            :,
            idx,
        ]

        audit[f"W3_1__{label}"] = w31_mean[
            :,
            idx,
        ]

        audit[f"HYBRID__{label}"] = hybrid[
            :,
            idx,
        ]

        audit[f"W3_0_foldSD__{label}"] = w30_fold_probs[
            :,
            :,
            idx,
        ].std(axis=0)

        audit[f"W3_1_foldSD__{label}"] = w31_fold_probs[
            :,
            :,
            idx,
        ].std(axis=0)

    audit.to_csv(
        RESULT_ROOT / "test_prediction_audit.csv",
        index=False,
    )


# ============================================================
# 14. BUNDLE EXPORT MODE
# ============================================================


def export_bundle() -> None:
    """
    Create a compact folder suitable for turning into a Kaggle Dataset.

    Run this in the development session where W3.0/W3.1 outputs exist.
    """

    print("=" * 88)
    print("EXPORTING W3 SUBMISSION BUNDLE")
    print("=" * 88)

    w30_root = discover_checkpoint_root("w3_0")

    w31_root = discover_checkpoint_root("w3_1")

    verification = verify_development_metrics(
        w30_root,
        w31_root,
    )

    if BUNDLE_EXPORT_ROOT.exists():

        shutil.rmtree(BUNDLE_EXPORT_ROOT)

    w30_out = BUNDLE_EXPORT_ROOT / "w3_0_checkpoints"

    w31_out = BUNDLE_EXPORT_ROOT / "w3_1_checkpoints"

    verification_out = BUNDLE_EXPORT_ROOT / "verification"

    for path in [
        w30_out,
        w31_out,
        verification_out,
    ]:

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    copied_files: List[str] = []

    for fold in range(
        1,
        NUM_FOLDS + 1,
    ):

        src = w30_root / f"fold_{fold}_epoch_{FINAL_EPOCH}.pt"

        dst = w30_out / src.name

        shutil.copy2(
            src,
            dst,
        )

        copied_files.append(str(dst.relative_to(BUNDLE_EXPORT_ROOT)))

        src = w31_root / f"fold_{fold}_epoch_{FINAL_EPOCH}.pt"

        dst = w31_out / src.name

        shutil.copy2(
            src,
            dst,
        )

        copied_files.append(str(dst.relative_to(BUNDLE_EXPORT_ROOT)))

    # Save an explicit exact torchvision state_dict into the bundle.
    backbone = load_resnet18_backbone()

    resnet_out = BUNDLE_EXPORT_ROOT / RESNET18_BUNDLE_FILENAME

    torch.save(
        backbone.state_dict(),
        resnet_out,
    )

    copied_files.append(str(resnet_out.relative_to(BUNDLE_EXPORT_ROOT)))

    del backbone
    gc.collect()

    # Copy small verification files when available.
    for checkpoint_root, filename in [
        (
            w30_root,
            "gold_reproduction_summary.json",
        ),
        (
            w31_root,
            "W3_1_COMPARISON.json",
        ),
    ]:

        src = find_result_json_near_checkpoint(
            checkpoint_root,
            filename,
        )

        if src is not None:

            dst = verification_out / filename

            shutil.copy2(
                src,
                dst,
            )

            copied_files.append(str(dst.relative_to(BUNDLE_EXPORT_ROOT)))

    manifest = {
        "purpose": "RSNA Knee W3 final inference bundle",
        "w3_0_checkpoint_source": str(w30_root),
        "w3_1_checkpoint_source": str(w31_root),
        "verification": verification,
        "expected_w3_0_macro_auc": EXPECTED_W30_AUROC,
        "expected_w3_1_cuda_macro_auc": EXPECTED_W31_CUDA_AUROC,
        "hybrid_w3_1_labels": HYBRID_W31_LABELS,
        "files": copied_files,
    }

    with open(
        BUNDLE_EXPORT_ROOT / "bundle_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
            allow_nan=True,
        )

    total_bytes = sum(
        path.stat().st_size
        for path in (BUNDLE_EXPORT_ROOT.rglob("*"))
        if path.is_file()
    )

    print("\nBundle created:")
    print(BUNDLE_EXPORT_ROOT)

    print(
        "Bundle size:",
        human_bytes(total_bytes),
    )

    print(
        "\nCreate/attach a Kaggle Dataset from this folder "
        "before the final hidden-test notebook run."
    )


# ============================================================
# 15. INFERENCE MODE
# ============================================================


def run_inference() -> None:

    started = time.time()

    print("=" * 88)
    print("RSNA W3 FINAL SUBMISSION INFERENCE")
    print("=" * 88)

    print(f"Device                : {DEVICE}")

    print(f"CUDA GPU count        : {GPU_COUNT}")

    print(f"Encoder GPUs          : {ENCODER_DEVICE_IDS}")

    print(f"Data root             : {DATA_ROOT}")

    print(f"Output                : {WORK_ROOT}")

    print("Primary submission    : W3.0 5-fold ensemble")

    print("Exploratory hybrid    : W3.1 for " + ", ".join(HYBRID_W31_LABELS))

    # --------------------------------------------------------
    # Discover and verify model artifacts first.
    # --------------------------------------------------------

    w30_checkpoint_root = discover_checkpoint_root("w3_0")

    w31_checkpoint_root = discover_checkpoint_root("w3_1")

    print("\nW3.0 checkpoints:")
    print(w30_checkpoint_root)

    print("\nW3.1 checkpoints:")
    print(w31_checkpoint_root)

    verification = verify_development_metrics(
        w30_checkpoint_root,
        w31_checkpoint_root,
    )

    print("\nDevelopment metric verification:")
    print(
        json.dumps(
            verification,
            indent=2,
            allow_nan=True,
        )
    )

    # --------------------------------------------------------
    # Load test schema.
    # --------------------------------------------------------

    (
        test_df,
        test_series_df,
        sample_df,
    ) = load_test_tables()

    study_uids = sample_df[UID_COLUMN].astype(str).tolist()

    print(f"\nTest studies          : " f"{len(study_uids)}")

    print(f"Test series           : " f"{len(test_series_df)}")

    series_lookup = make_series_lookup(test_series_df)

    # --------------------------------------------------------
    # Feature cache.
    # --------------------------------------------------------

    manifest = build_test_feature_cache(
        study_uids,
        series_lookup,
    )

    if len(manifest) != len(study_uids):

        raise RuntimeError("Test feature manifest row count mismatch.")

    if not all(
        test_cache_is_valid(
            test_feature_cache_path(uid),
            uid,
        )
        for uid in (study_uids)
    ):

        raise RuntimeError("Test feature cache is incomplete after build.")

    # --------------------------------------------------------
    # Load all 10 heads once.
    # --------------------------------------------------------

    w30_models = load_head_ensemble(
        w30_checkpoint_root,
        expected_kind="w3_0",
    )

    w31_models = load_head_ensemble(
        w31_checkpoint_root,
        expected_kind="w3_1",
    )

    # --------------------------------------------------------
    # Predict.
    # --------------------------------------------------------

    (
        w30_mean,
        w31_mean,
        w30_fold_probs,
        w31_fold_probs,
    ) = predict_test(
        study_uids,
        w30_models,
        w31_models,
    )

    if not (np.isfinite(w30_mean).all() and np.isfinite(w31_mean).all()):

        raise RuntimeError("Non-finite test predictions.")

    # --------------------------------------------------------
    # Primary W3.0.
    # --------------------------------------------------------

    submission_w30 = make_submission_from_probs(
        sample_df,
        w30_mean,
    )

    validate_submission(
        submission_w30,
        sample_df,
        "W3.0 primary",
    )

    # --------------------------------------------------------
    # Exploratory hybrid.
    # --------------------------------------------------------

    hybrid = w30_mean.copy()

    for label in HYBRID_W31_LABELS:

        idx = LABEL_COLUMNS.index(label)

        hybrid[
            :,
            idx,
        ] = w31_mean[
            :,
            idx,
        ]

    submission_hybrid = make_submission_from_probs(
        sample_df,
        hybrid,
    )

    validate_submission(
        submission_hybrid,
        sample_df,
        "W3.0/W3.1 hybrid",
    )

    # --------------------------------------------------------
    # Write files.
    # --------------------------------------------------------

    primary_path = WORK_ROOT / "submission_w3_0_primary.csv"

    hybrid_path = WORK_ROOT / "submission_w3_0_hybrid_effusion_fracture.csv"

    kaggle_default_path = (
        Path("/kaggle/working/submission.csv")
        if Path("/kaggle/working").exists()
        else WORK_ROOT / "submission.csv"
    )

    submission_w30.to_csv(
        primary_path,
        index=False,
    )

    submission_hybrid.to_csv(
        hybrid_path,
        index=False,
    )

    # Deliberately make the defensible W3.0 model the default Kaggle
    # submission artifact.
    submission_w30.to_csv(
        kaggle_default_path,
        index=False,
    )

    write_prediction_audit(
        study_uids,
        w30_mean,
        w31_mean,
        hybrid,
        w30_fold_probs,
        w31_fold_probs,
    )

    # Fold-level compressed audit.
    np.savez_compressed(
        RESULT_ROOT / "test_fold_predictions.npz",
        study_uids=np.asarray(
            study_uids,
            dtype=object,
        ),
        labels=np.asarray(
            LABEL_COLUMNS,
            dtype=object,
        ),
        w3_0_fold_probabilities=w30_fold_probs,
        w3_1_fold_probabilities=w31_fold_probs,
    )

    # Basic distribution audit.
    distribution_rows = []

    for idx, label in enumerate(LABEL_COLUMNS):

        for model_name, matrix in [
            (
                "W3.0",
                w30_mean,
            ),
            (
                "W3.1",
                w31_mean,
            ),
            (
                "HYBRID",
                hybrid,
            ),
        ]:

            values = matrix[
                :,
                idx,
            ]

            distribution_rows.append(
                {
                    "Model": model_name,
                    "Label": label,
                    "Mean": float(np.mean(values)),
                    "Std": float(np.std(values)),
                    "Min": float(np.min(values)),
                    "P05": float(
                        np.quantile(
                            values,
                            0.05,
                        )
                    ),
                    "Median": float(np.median(values)),
                    "P95": float(
                        np.quantile(
                            values,
                            0.95,
                        )
                    ),
                    "Max": float(np.max(values)),
                }
            )

    distribution_df = pd.DataFrame(distribution_rows)

    distribution_df.to_csv(
        RESULT_ROOT / "test_prediction_distribution.csv",
        index=False,
    )

    manifest_json = {
        "primary_submission": str(primary_path),
        "kaggle_default_submission": str(kaggle_default_path),
        "exploratory_hybrid_submission": str(hybrid_path),
        "primary_model": "W3.0 5-fold mean probability ensemble",
        "hybrid_model": (
            "W3.0 5-fold ensemble except "
            "Effusion and Fracture from W3.1 "
            "5-fold ensemble"
        ),
        "hybrid_is_post_hoc_exploratory": True,
        "test_studies": len(study_uids),
        "test_series": len(test_series_df),
        "w3_0_checkpoint_root": str(w30_checkpoint_root),
        "w3_1_checkpoint_root": str(w31_checkpoint_root),
        "development_metric_verification": verification,
        "device": str(DEVICE),
        "gpu_count": GPU_COUNT,
        "runtime_seconds": time.time() - started,
    }

    with open(
        RESULT_ROOT / "submission_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest_json,
            handle,
            indent=2,
            allow_nan=True,
        )

    # --------------------------------------------------------
    # Final console.
    # --------------------------------------------------------

    print("\n")
    print("=" * 88)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 88)

    print("\nPRIMARY / Kaggle submission.csv:")
    print(kaggle_default_path)

    print("\nPrimary W3.0 copy:")
    print(primary_path)

    print("\nExploratory Effusion+Fracture hybrid:")
    print(hybrid_path)

    print(
        "\nBoth submissions passed exact sample_submission "
        "schema/order/UID/range/finite validation."
    )

    print("\nPrediction means by label:")

    quick = pd.DataFrame(
        {
            "Label": LABEL_COLUMNS,
            "W3_0_Mean": [
                float(
                    w30_mean[
                        :,
                        idx,
                    ].mean()
                )
                for idx in range(NUM_LABELS)
            ],
            "Hybrid_Mean": [
                float(
                    hybrid[
                        :,
                        idx,
                    ].mean()
                )
                for idx in range(NUM_LABELS)
            ],
        }
    )

    print(quick.to_string(index=False))

    print(
        "\nTotal runtime:",
        elapsed_string(time.time() - started),
    )

    print(
        "\nIMPORTANT: submission.csv is W3.0 primary. "
        "The hybrid is a separate exploratory file."
    )


# ============================================================
# 16. CLI
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="RSNA Knee W3 final submission inference",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "infer",
            "export_bundle",
        ],
        default="infer",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.mode == "export_bundle":

        export_bundle()

    elif args.mode == "infer":

        run_inference()

    else:

        raise ValueError(args.mode)


if __name__ == "__main__":

    main()
