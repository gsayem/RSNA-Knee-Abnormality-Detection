#!/usr/bin/env python3
# ============================================================
# RSNA KNEE ABNORMALITY DETECTION - W3.0
#
# Frozen-ResNet18 Feature Cache + Gold-Only V4 Reproduction
#
# PURPOSE
# -------
# We are now optimizing for fast iteration toward weakly supervised
# MRI training/submission.
#
# W3.0 makes ONE execution-pipeline change:
#
#   V4:
#       DICOM -> 16 sampled slices/series -> frozen ResNet18
#       -> trainable projection/attention head
#
#   W3.0:
#       DICOM -> 16 sampled slices/series -> frozen ResNet18
#       -> CACHE 512-D slice features ONCE
#       -> same trainable projection/attention head
#
# The frozen image encoder is mathematically outside the trainable
# optimization problem, so caching its output lets W3.1/W3.x avoid
# repeatedly decoding DICOM and running ResNet18 every epoch/fold.
#
# IMPORTANT CONTROL:
# Before using this cache for pseudo-supervised training, W3.0
# reproduces the gold-only V4 experiment using the exact:
#   - 58 gold studies
#   - 5-fold greedy multilabel split, seed 42
#   - expected controlled fold checksum
#   - frozen ResNet18 ImageNet encoder
#   - trainable visual projection / slice attention
#   - zero-metadata ablation branch
#   - series fusion / series attention
#   - 12-label classifier
#   - pos_weight BCE
#   - AdamW lr=1e-3, wd=1e-4
#   - 12 fixed epochs
#   - CosineAnnealingLR eta_min=1e-6
#   - effective batch/gradient accumulation = 1
#
# DEFAULT "all" MODE IS ORDERED FOR SPEED/SAFETY:
#   1) cache only 58 gold studies first;
#   2) immediately run gold-only cached-feature reproduction;
#   3) then cache the remaining 4,349 studies for W3.1.
#
# That way a reproduction problem is visible early instead of after
# waiting for all 4,407 studies to be cached.
#
# MODES
# -----
#   W30_MODE=all          (default)
#   W30_MODE=cache_gold
#   W30_MODE=reproduce
#   W30_MODE=cache_all
#
# This script trains NO pseudo-supervised MRI model.
# ============================================================

from __future__ import annotations

import os
import gc
import json
import math
import time
import random
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
from torchvision.models import resnet18, ResNet18_Weights

import pydicom

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "W30_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"

WORK_ROOT = Path(
    os.environ.get(
        "W30_WORK_ROOT",
        "/kaggle/working/rsna_w3_0",
    )
)

FEATURE_CACHE_ROOT = WORK_ROOT / "feature_cache"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"
RESULT_ROOT = WORK_ROOT / "results"

for path in [
    WORK_ROOT,
    FEATURE_CACHE_ROOT,
    CHECKPOINT_ROOT,
    RESULT_ROOT,
]:
    path.mkdir(parents=True, exist_ok=True)


MODE = (
    os.environ.get(
        "W30_MODE",
        "all",
    )
    .strip()
    .lower()
)

VALID_MODES = {
    "all",
    "cache_gold",
    "reproduce",
    "cache_all",
}

if MODE not in VALID_MODES:
    raise ValueError(f"W30_MODE must be one of {sorted(VALID_MODES)}, got {MODE!r}")


# ------------------------------------------------------------
# Exact V4 controlled settings
# ------------------------------------------------------------

IMAGE_SIZE = 224
SLICES_PER_SERIES = 16
SERIES_META_DIM = 9
NUM_LABELS = 12

BATCH_SIZE = 1
TRAIN_NUM_WORKERS = int(os.environ.get("W30_TRAIN_NUM_WORKERS", "2"))

NUM_FOLDS = 5
RANDOM_SEED = 42
NUM_EPOCHS = 12

HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
COSINE_ETA_MIN = 1e-6
USE_AMP = True

USE_SERIES_METADATA = False
METADATA_ABLATION_MODE = "zero_metadata_before_encoder_same_architecture"


# ------------------------------------------------------------
# Feature-cache speed settings
# ------------------------------------------------------------

PREPROCESS_THREADS = max(
    1,
    int(
        os.environ.get(
            "W30_PREPROCESS_THREADS",
            "4",
        )
    ),
)

ENCODER_BATCH_SIZE = max(
    16,
    int(
        os.environ.get(
            "W30_ENCODER_BATCH_SIZE",
            "256",
        )
    ),
)

USE_MULTI_GPU_ENCODER = os.environ.get(
    "W30_USE_MULTI_GPU",
    "1",
).strip() not in {"0", "false", "False"}

# V4 writes preprocessed image tensors as float16 and on subsequent
# reads converts them back to float32. Most V4 optimization steps
# therefore see half-quantized input images. Reproducing this
# quantization before feature extraction makes cached features more
# faithful to the actual V4 execution path.
EMULATE_V4_IMAGE_CACHE_QUANTIZATION = True

FEATURE_CACHE_DTYPE = torch.float16
# Keep this version unchanged after the DICOM decoder robustness fix.
# Already-created feature tensors remain valid and should be reused.
FEATURE_CACHE_VERSION = "w3_0_v4_resnet18_amp_fp16_v1"

# If a previous V4/V1 cache exists in the same Kaggle session, use
# it to avoid decoding those series again. These caches contain
# preprocessed sampled images, NOT labels.
OPTIONAL_SERIES_CACHE_ROOTS = [
    Path("/kaggle/working/rsna_v4/series_cache"),
    Path("/kaggle/working/rsna_v1_1/series_cache"),
    Path("/kaggle/working/rsna_v1/series_cache"),
]


# ------------------------------------------------------------
# Controlled fold audit
# ------------------------------------------------------------

EXPECTED_V4_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)


# ------------------------------------------------------------
# Known V4 pooled OOF reference values
# ------------------------------------------------------------
#
# These are diagnostic references only. W3.0 does NOT optimize
# against them or select an epoch based on them.
#
EXPECTED_V4_OOF_MACRO_AUROC = 0.54577
EXPECTED_V4_OOF_MACRO_AP = 0.43250
EXPECTED_V4_OOF_MACRO_F1 = 0.40716

REPRO_AUROC_WARN_TOL = float(
    os.environ.get(
        "W30_REPRO_AUROC_WARN_TOL",
        "0.03",
    )
)

REPRO_AP_WARN_TOL = float(
    os.environ.get(
        "W30_REPRO_AP_WARN_TOL",
        "0.05",
    )
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

UID_COLUMN = "StudyInstanceUID"


# ============================================================
# 2. REPRODUCIBILITY / DEVICE
# ============================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(RANDOM_SEED)

# Avoid CPU oversubscription when several series are preprocessed in
# parallel. GPU training remains unaffected.
try:
    torch.set_num_threads(
        max(
            1,
            int(
                os.environ.get(
                    "W30_TORCH_CPU_THREADS",
                    "2",
                )
            ),
        )
    )
except Exception:
    pass


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


# ============================================================
# 3. GENERIC HELPERS
# ============================================================


def safe_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:

    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")

        return float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    except Exception:
        return float("nan")


def safe_ap(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:

    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")

        return float(
            average_precision_score(
                y_true,
                y_score,
            )
        )

    except Exception:
        return float("nan")


def autocast_context():

    if USE_AMP and DEVICE.type == "cuda":

        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )

    return nullcontext()


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


def elapsed_string(
    seconds: float,
) -> str:

    seconds = max(
        0.0,
        float(seconds),
    )

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    secs = int(seconds % 60)

    return f"{hours:02d}:" f"{minutes:02d}:" f"{secs:02d}"


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
# 4. LOAD DATA / GOLD COHORT
# ============================================================


def load_tables() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    if not TRAIN_SERIES_CSV.exists():
        raise FileNotFoundError(TRAIN_SERIES_CSV)

    if not TRAIN_SERIES_ROOT.exists():
        raise FileNotFoundError(TRAIN_SERIES_ROOT)

    train_df = pd.read_csv(TRAIN_CSV)

    series_df = pd.read_csv(TRAIN_SERIES_CSV)

    train_df[UID_COLUMN] = train_df[UID_COLUMN].astype(str)

    series_df[UID_COLUMN] = series_df[UID_COLUMN].astype(str)

    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    gold_df = (
        train_df[train_df[LABEL_COLUMNS].notna().all(axis=1)]
        .copy()
        .sort_values(UID_COLUMN)
        .reset_index(drop=True)
    )

    if len(train_df) != 4407:

        print("WARNING: expected 4,407 train studies; " f"found {len(train_df)}.")

    if len(gold_df) != 58:

        raise RuntimeError(
            "Expected exactly 58 fully labeled studies, " f"found {len(gold_df)}."
        )

    return (
        train_df,
        series_df,
        gold_df,
    )


# ============================================================
# 5. EXACT V4 FOLD SPLIT
# ============================================================


def greedy_multilabel_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    n_samples, n_labels = y.shape

    fold_assignments = -np.ones(
        n_samples,
        dtype=int,
    )

    label_frequency = y.sum(axis=0) + 1e-8

    sample_rarity = np.zeros(
        n_samples,
        dtype=np.float64,
    )

    for i in range(n_samples):

        positive_labels = np.where(y[i] > 0)[0]

        if len(positive_labels) == 0:

            sample_rarity[i] = 0.0

        else:

            sample_rarity[i] = float(np.sum(1.0 / label_frequency[positive_labels]))

    tie_noise = rng.random(n_samples) * 1e-6

    order = np.argsort(-sample_rarity - tie_noise)

    fold_label_counts = np.zeros(
        (
            n_splits,
            n_labels,
        ),
        dtype=np.float64,
    )

    fold_sizes = np.zeros(
        n_splits,
        dtype=int,
    )

    desired_fold_label_counts = label_frequency / n_splits

    for sample_idx in order:

        sample = y[sample_idx]

        positive_labels = np.where(sample > 0)[0]

        scores: List[float] = []

        for fold in range(n_splits):

            label_score = 0.0

            if len(positive_labels) > 0:

                ratios = fold_label_counts[
                    fold,
                    positive_labels,
                ] / (desired_fold_label_counts[positive_labels] + 1e-8)

                label_score = float(np.mean(ratios))

            size_score = fold_sizes[fold] / max(
                1,
                math.ceil(n_samples / n_splits),
            )

            scores.append(label_score + 0.05 * size_score)

        best_fold = int(np.argmin(scores))

        fold_assignments[sample_idx] = best_fold

        fold_sizes[best_fold] += 1

        fold_label_counts[best_fold] += sample

    return fold_assignments


def fold_assignment_sha256(
    assignments: pd.DataFrame,
) -> str:

    ordered = assignments.sort_values(UID_COLUMN).reset_index(drop=True)

    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(
            ordered[UID_COLUMN].astype(str),
            ordered["OuterFold"].astype(int),
        )
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_and_verify_folds(
    gold_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    pd.DataFrame,
]:

    y = gold_df[LABEL_COLUMNS].values.astype(np.int64)

    fold_zero = greedy_multilabel_folds(
        y,
        n_splits=NUM_FOLDS,
        seed=RANDOM_SEED,
    )

    fold_df = gold_df[
        [
            UID_COLUMN,
            *LABEL_COLUMNS,
        ]
    ].copy()

    fold_df["OuterFold"] = fold_zero + 1

    fold_df = fold_df[
        [
            UID_COLUMN,
            "OuterFold",
            *LABEL_COLUMNS,
        ]
    ]

    fold_hash = fold_assignment_sha256(
        fold_df[
            [
                UID_COLUMN,
                "OuterFold",
            ]
        ]
    )

    print(
        "Outer fold SHA256:",
        fold_hash,
    )

    print(
        "Expected V4 SHA256:",
        EXPECTED_V4_FOLD_SHA256,
    )

    if fold_hash != EXPECTED_V4_FOLD_SHA256:

        raise RuntimeError(
            "W3.0 outer folds do not match the " "controlled V4/W2.3 split."
        )

    fold_df.to_csv(
        RESULT_ROOT / "00_outer_fold_assignments.csv",
        index=False,
    )

    return (
        fold_zero,
        fold_df,
    )


# ============================================================
# 6. EXACT V4 DICOM PREPROCESSING
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


def v4_series_cache_key(
    study_uid: str,
    series_uid: str,
    root: Path,
) -> Path:

    key = f"{study_uid}_{series_uid}"

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()

    return root / f"{digest}.pt"


def try_load_existing_series_cache(
    study_uid: str,
    series_uid: str,
) -> Optional[torch.Tensor]:

    for root in OPTIONAL_SERIES_CACHE_ROOTS:

        if not root.exists():
            continue

        path = v4_series_cache_key(
            study_uid,
            series_uid,
            root,
        )

        if not path.exists():
            continue

        try:

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            images = payload["images"]

            if (
                images.ndim == 4
                and images.shape[0] == SLICES_PER_SERIES
                and images.shape[1] == 3
                and images.shape[2] == IMAGE_SIZE
                and images.shape[3] == IMAGE_SIZE
            ):

                return images.float().contiguous()

        except Exception:

            continue

    return None


def decode_dicom_pixel_array(
    ds,
    dicom_path: str,
) -> np.ndarray:
    """
    Decode one native MRI DICOM slice robustly.

    Normal path:
        ds.pixel_array

    Repair path 1:
        Correct NumberOfFrames when current native pixel geometry
        proves an exact different frame count.

    Repair path 2:
        For MONOCHROME native PixelData, infer a single-frame storage
        width directly from:
            len(PixelData) / (Rows * Columns * SamplesPerPixel)
        and repair BitsAllocated + NumberOfFrames in-memory when that
        inferred width is an exact standard width (8/16/32/64 bits)
        and is compatible with BitsStored.

    The source DICOM is never modified.
    Pixel bytes are never padded or truncated.
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

        # Snapshot metadata so each repair attempt starts cleanly.
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
            # Repair 1: NumberOfFrames only.
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
            # Repair 2:
            # infer single-frame native storage width.
            #
            # This is deliberately limited to MONOCHROME MRI-like
            # images because W3 expects each .dcm to be one 2-D slice.
            # --------------------------------------------------
            restore_metadata()

            if not (photometric.startswith("MONOCHROME")):
                raise first_error

            candidate_samples = samples

            base_pixels = rows * columns * candidate_samples

            usable_bytes = actual_bytes

            # Permit only a single DICOM padding byte.
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

            # W3/V4 treats each DICOM file as one 2-D slice.
            ds.NumberOfFrames = 1
            ds.BitsAllocated = int(inferred_bits_allocated)

            # HighBit should be compatible with BitsStored.
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
    Exact V4 image preprocessing.

    Returns:
        [16, 3, 224, 224] float32

    When no old series cache exists, W3.0 intentionally applies
    float16 -> float32 quantization after preprocessing to emulate
    the cached-image path used by V4 during nearly all optimization
    steps.
    """

    old_cached = try_load_existing_series_cache(
        study_uid,
        series_uid,
    )

    if old_cached is not None:

        return old_cached

    series_dir = TRAIN_SERIES_ROOT / study_uid / series_uid

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

    selected_headers = [headers[i] for i in (selected_indices)]

    raw_images: List[np.ndarray] = []

    for target_index in selected_indices:

        # Try the intended uniformly sampled slice first. If that
        # specific file is genuinely corrupt even after safe metadata
        # repair, search outward for the nearest decodable slice in
        # the same series. This prevents a single bad file from
        # killing a 4,407-study cache build.
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
                ).astype(np.float32)

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

    if EMULATE_V4_IMAGE_CACHE_QUANTIZATION:

        images = images.half().float()

    return images


# ============================================================
# 7. FROZEN RESNET18 FEATURE ENCODER
# ============================================================


class FrozenResNet18Encoder(nn.Module):
    def __init__(
        self,
    ):

        super().__init__()

        try:

            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        except Exception as exc:

            raise RuntimeError(
                "W3.0 requires ImageNet-pretrained "
                "ResNet18 weights. Could not load them. "
                "Do not continue with random weights."
            ) from exc

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


# ============================================================
# 8. STUDY FEATURE CACHE
# ============================================================


def feature_cache_path(
    study_uid: str,
) -> Path:

    digest = hashlib.md5(str(study_uid).encode("utf-8")).hexdigest()

    return FEATURE_CACHE_ROOT / f"{digest}.pt"


def cached_study_is_valid(
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

        if payload.get("cache_version") != FEATURE_CACHE_VERSION:

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
            or features.shape[2] != 512
        ):

            return False

        return True

    except Exception:

        return False


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


def encode_feature_batches(
    images: torch.Tensor,
    encoder: nn.Module,
) -> torch.Tensor:
    """
    images:
        [N, 3, 224, 224] CPU float32

    returns:
        [N, 512] CPU float16
    """

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
                    dtype=FEATURE_CACHE_DTYPE,
                )
            )

            del chunk
            del feature_chunk

    return torch.cat(
        outputs,
        dim=0,
    ).contiguous()


def encode_and_save_study(
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

    cache_path = feature_cache_path(study_uid)

    if cached_study_is_valid(
        cache_path,
        study_uid,
    ):

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        return {
            UID_COLUMN: study_uid,
            "SeriesCount": int(payload["features"].shape[0]),
            "SliceCount": int(
                payload["features"].shape[0] * payload["features"].shape[1]
            ),
            "FeatureDim": int(payload["features"].shape[2]),
            "CachePath": str(cache_path),
            "CacheBytes": int(cache_path.stat().st_size),
            "Status": "reused",
        }

    if study_uid not in (series_lookup):

        raise RuntimeError("No train_series.csv rows for " f"study {study_uid}.")

    rows = series_lookup[study_uid]

    series_uids = rows["SeriesInstanceUID"].astype(str).tolist()

    tasks = [
        (
            study_uid,
            series_uid,
        )
        for series_uid in (series_uids)
    ]

    # executor.map preserves task order, so series order remains
    # identical to V4's SeriesInstanceUID sort.
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
            "Failed while preprocessing study "
            f"{study_uid}. Series list: {series_uids}"
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

    if features.shape[2] != 512:

        raise RuntimeError(
            "Expected ResNet18 feature dim 512, " f"got {features.shape}."
        )

    payload = {
        "cache_version": FEATURE_CACHE_VERSION,
        "study_uid": study_uid,
        "series_uids": series_uids,
        "features": features,
        "feature_dtype": "float16",
        "image_size": IMAGE_SIZE,
        "slices_per_series": SLICES_PER_SERIES,
        "backbone": "ResNet18_Weights.DEFAULT",
        "image_cache_quantization_emulated": EMULATE_V4_IMAGE_CACHE_QUANTIZATION,
    }

    atomic_torch_save(
        payload,
        cache_path,
    )

    del image_list
    del images
    del flat_images
    del flat_features
    del features
    del payload

    return {
        UID_COLUMN: study_uid,
        "SeriesCount": int(s),
        "SliceCount": int(s * k),
        "FeatureDim": 512,
        "CachePath": str(cache_path),
        "CacheBytes": int(cache_path.stat().st_size),
        "Status": "created",
    }


def build_feature_cache(
    study_uids: Iterable[str],
    series_lookup: Dict[
        str,
        pd.DataFrame,
    ],
    scope_name: str,
) -> pd.DataFrame:

    ordered_uids = [str(uid) for uid in (study_uids)]

    # preserve user-provided order but remove duplicates
    ordered_uids = list(dict.fromkeys(ordered_uids))

    print("\n")
    print("=" * 88)
    print(f"BUILDING FEATURE CACHE: " f"{scope_name}")
    print("=" * 88)
    print(f"Studies requested     : " f"{len(ordered_uids)}")
    print(f"Preprocess threads    : " f"{PREPROCESS_THREADS}")
    print(f"Encoder batch size    : " f"{ENCODER_BATCH_SIZE}")
    print(f"CUDA GPUs             : " f"{GPU_COUNT}")

    missing_uids = [
        uid
        for uid in (ordered_uids)
        if not cached_study_is_valid(
            feature_cache_path(uid),
            uid,
        )
    ]

    print(f"Already cached        : " f"{len(ordered_uids) - len(missing_uids)}")
    print(f"Need encoding         : " f"{len(missing_uids)}")

    encoder: Optional[nn.Module] = None

    if missing_uids:

        encoder_base = FrozenResNet18Encoder().to(DEVICE)

        encoder_base.eval()

        if len(ENCODER_DEVICE_IDS) >= 2:

            encoder = nn.DataParallel(
                encoder_base,
                device_ids=ENCODER_DEVICE_IDS,
                output_device=ENCODER_DEVICE_IDS[0],
                dim=0,
            )

            print(
                "Frozen ResNet18 cache encoder: "
                f"DataParallel GPUs "
                f"{ENCODER_DEVICE_IDS}"
            )

        else:

            encoder = encoder_base

            print("Frozen ResNet18 cache encoder: " f"{DEVICE}")

    records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=PREPROCESS_THREADS) as executor:

        for index, study_uid in enumerate(
            ordered_uids,
            start=1,
        ):

            if encoder is None:

                # Every requested study was already cached.
                cache_path = feature_cache_path(study_uid)

                payload = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )

                record = {
                    UID_COLUMN: study_uid,
                    "SeriesCount": int(payload["features"].shape[0]),
                    "SliceCount": int(
                        payload["features"].shape[0] * payload["features"].shape[1]
                    ),
                    "FeatureDim": int(payload["features"].shape[2]),
                    "CachePath": str(cache_path),
                    "CacheBytes": int(cache_path.stat().st_size),
                    "Status": "reused",
                }

            else:

                record = encode_and_save_study(
                    study_uid,
                    series_lookup,
                    encoder,
                    executor,
                )

            records.append(record)

            if index <= 5 or index % 50 == 0 or index == len(ordered_uids):

                elapsed = time.time() - start_time

                rate = index / max(
                    elapsed,
                    1e-6,
                )

                remaining = (len(ordered_uids) - index) / max(
                    rate,
                    1e-6,
                )

                print(
                    f"  {index:5d}/"
                    f"{len(ordered_uids):5d} "
                    f"({100.0 * index / len(ordered_uids):5.1f}%) "
                    f"elapsed={elapsed_string(elapsed)} "
                    f"ETA={elapsed_string(remaining)} "
                    f"last={record['Status']}"
                )

    if encoder is not None:

        del encoder

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    gc.collect()

    manifest = pd.DataFrame(records)

    manifest_path = RESULT_ROOT / f"feature_manifest_{scope_name}.csv"

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    total_bytes = int(manifest["CacheBytes"].sum())

    print(f"\n{scope_name} feature cache size: " f"{human_bytes(total_bytes)}")

    return manifest


# ============================================================
# 9. CACHED-FEATURE GOLD DATASET
# ============================================================


class CachedFeatureStudyDataset(Dataset):
    def __init__(
        self,
        study_uids: List[str],
        gold_df: pd.DataFrame,
    ):

        self.study_uids = [str(uid) for uid in (study_uids)]

        self.gold_df = gold_df.set_index(UID_COLUMN)

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

        study_uid = self.study_uids[index]

        path = feature_cache_path(study_uid)

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if payload.get("cache_version") != FEATURE_CACHE_VERSION:

            raise RuntimeError("Feature cache version mismatch: " f"{path}")

        # V4 image encoder output under CUDA autocast is float16.
        # Collation below intentionally places these values into
        # float32 tensors before the trainable head, mirroring the
        # V4 pattern where cached images were converted to float32
        # and the trainable path then ran under autocast.
        features = payload["features"].float().contiguous()

        series_count = features.shape[0]

        labels = torch.tensor(
            self.gold_df.loc[
                study_uid,
                LABEL_COLUMNS,
            ].values.astype(np.float32),
            dtype=torch.float32,
        )

        # V4 preserves the metadata branch but zeros all study-
        # specific metadata before metadata_projection.
        metadata = torch.zeros(
            series_count,
            SERIES_META_DIM,
            dtype=torch.float32,
        )

        return {
            "study_uid": study_uid,
            "features": features,
            "metadata": metadata,
            "series_mask": torch.ones(
                series_count,
                dtype=torch.bool,
            ),
            "labels": labels,
        }


def cached_study_collate_fn(
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

    batch_size = len(batch)

    max_series = max(item["features"].shape[0] for item in (batch))

    num_slices = batch[0]["features"].shape[1]

    feature_dim = batch[0]["features"].shape[2]

    features = torch.zeros(
        batch_size,
        max_series,
        num_slices,
        feature_dim,
        dtype=torch.float32,
    )

    metadata = torch.zeros(
        batch_size,
        max_series,
        SERIES_META_DIM,
        dtype=torch.float32,
    )

    series_mask = torch.zeros(
        batch_size,
        max_series,
        dtype=torch.bool,
    )

    labels = torch.stack([item["labels"] for item in (batch)])

    study_uids: List[str] = []

    for b, item in enumerate(batch):

        s = item["features"].shape[0]

        features[
            b,
            :s,
        ] = item["features"]

        metadata[
            b,
            :s,
        ] = item["metadata"]

        series_mask[
            b,
            :s,
        ] = True

        study_uids.append(item["study_uid"])

    return {
        "study_uid": study_uids,
        "features": features,
        "metadata": metadata,
        "series_mask": series_mask,
        "labels": labels,
    }


# ============================================================
# 10. EXACT V4 TRAINABLE HEAD
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


def consume_v4_backbone_initialization_rng() -> None:
    """
    V4 constructs a torchvision ResNet18 inside every fold before
    constructing the trainable projection/attention modules.

    Even though pretrained weights overwrite the ResNet parameters,
    construction first consumes torch RNG for parameter initialization.

    W3.0 has no live backbone inside its trainable model. Constructing
    and immediately discarding weights=None ResNet18 here consumes the
    same architecture-initialization RNG so downstream head
    initialization and DataLoader/dropout RNG remain as close as
    possible to V4.
    """

    dummy = resnet18(weights=None)

    del dummy


class CachedV4MetadataAblationModel(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        num_labels: int,
        align_v4_rng: bool = True,
    ):

        super().__init__()

        if align_v4_rng:

            consume_v4_backbone_initialization_rng()

        visual_dim = 512
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
    ) -> Dict[
        str,
        torch.Tensor,
    ]:
        """
        features:
            [B, S, K, 512]
        """

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
            slice_attention,
        ) = self.slice_attention(slice_features)

        series_visual = series_visual.reshape(
            b,
            s,
            self.embedding_dim,
        )

        slice_attention = slice_attention.reshape(
            b,
            s,
            k,
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
            series_attention,
        ) = self.series_attention(
            series_features,
            series_mask,
        )

        logits = self.classifier(study_embedding)

        return {
            "logits": logits,
            "study_embedding": study_embedding,
            "slice_attention": slice_attention,
            "series_attention": series_attention,
        }


# ============================================================
# 11. GOLD TRAINING HELPERS
# ============================================================


def make_pos_weight(
    y_train: np.ndarray,
) -> torch.Tensor:

    positive = y_train.sum(axis=0)

    negative = y_train.shape[0] - positive

    weights = negative / np.maximum(
        positive,
        1,
    )

    weights = np.clip(
        weights,
        1.0,
        5.0,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[
    Dict[
        str,
        float,
    ],
    np.ndarray,
    np.ndarray,
]:

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets: List[np.ndarray] = []

    all_probs: List[np.ndarray] = []

    for batch in loader:

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

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        if training:

            optimizer.zero_grad(set_to_none=True)

        with autocast_context():

            outputs = model(
                features,
                metadata,
                series_mask,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits,
                labels,
            )

        if training:

            if scaler is not None:

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probabilities)

        all_targets.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_prob = np.concatenate(
        all_probs,
        axis=0,
    )

    metrics: Dict[
        str,
        float,
    ] = {
        "loss": total_loss
        / max(
            total_samples,
            1,
        )
    }

    aucs: List[float] = []

    aps: List[float] = []

    for idx, label in enumerate(LABEL_COLUMNS):

        auc = safe_auc(
            y_true[
                :,
                idx,
            ],
            y_prob[
                :,
                idx,
            ],
        )

        ap = safe_ap(
            y_true[
                :,
                idx,
            ],
            y_prob[
                :,
                idx,
            ],
        )

        metrics[f"{label}_AUROC"] = auc

        metrics[f"{label}_AP"] = ap

        if np.isfinite(auc):

            aucs.append(auc)

        if np.isfinite(ap):

            aps.append(ap)

    metrics["macro_AUROC"] = float(np.mean(aucs)) if aucs else float("nan")

    metrics["macro_AP"] = float(np.mean(aps)) if aps else float("nan")

    try:

        y_pred = (y_prob >= 0.5).astype(np.int32)

        metrics["macro_F1"] = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

    except Exception:

        metrics["macro_F1"] = float("nan")

    return (
        metrics,
        y_true,
        y_prob,
    )


# ============================================================
# 12. GOLD-ONLY CACHED-FEATURE REPRODUCTION
# ============================================================


def pooled_oof_summary(
    oof_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    Dict[
        str,
        float,
    ],
]:

    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    aucs: List[float] = []

    aps: List[float] = []

    true_columns: List[np.ndarray] = []

    prob_columns: List[np.ndarray] = []

    for label in LABEL_COLUMNS:

        y_true = oof_df[f"{label}_true"].values

        y_prob = oof_df[f"{label}_prob"].values

        auc = safe_auc(
            y_true,
            y_prob,
        )

        ap = safe_ap(
            y_true,
            y_prob,
        )

        rows.append(
            {
                "Label": label,
                "PositiveCount": int(y_true.sum()),
                "AUROC": auc,
                "AveragePrecision": ap,
            }
        )

        if np.isfinite(auc):
            aucs.append(auc)

        if np.isfinite(ap):
            aps.append(ap)

        true_columns.append(y_true)

        prob_columns.append(y_prob)

    y_true_matrix = np.stack(
        true_columns,
        axis=1,
    )

    y_prob_matrix = np.stack(
        prob_columns,
        axis=1,
    )

    y_pred_matrix = (y_prob_matrix >= 0.5).astype(np.int32)

    summary = {
        "macro_AUROC": float(np.mean(aucs)),
        "macro_AP": float(np.mean(aps)),
        "macro_F1": float(
            f1_score(
                y_true_matrix,
                y_pred_matrix,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return (
        pd.DataFrame(rows),
        summary,
    )


def compare_with_existing_v4(
    w30_oof: pd.DataFrame,
) -> Optional[
    Dict[
        str,
        Any,
    ]
]:

    candidates = [
        Path("/kaggle/working/rsna_v4/results/oof_predictions.csv"),
        Path("/kaggle/working/rsna_v4/oof_predictions.csv"),
    ]

    v4_path = next(
        (path for path in candidates if path.exists()),
        None,
    )

    if v4_path is None:
        return None

    v4 = pd.read_csv(v4_path)

    v4[UID_COLUMN] = v4[UID_COLUMN].astype(str)

    left = w30_oof.sort_values(UID_COLUMN).reset_index(drop=True)

    right = v4.sort_values(UID_COLUMN).reset_index(drop=True)

    if left[UID_COLUMN].tolist() != right[UID_COLUMN].tolist():

        return {
            "v4_path": str(v4_path),
            "status": "UID_MISMATCH",
        }

    fold_match = None

    if "Fold" in left.columns and "Fold" in right.columns:

        fold_match = bool(
            np.array_equal(
                left["Fold"].values,
                right["Fold"].values,
            )
        )

    left_probs = np.concatenate(
        [left[f"{label}_prob"].values for label in LABEL_COLUMNS]
    )

    right_probs = np.concatenate(
        [right[f"{label}_prob"].values for label in LABEL_COLUMNS]
    )

    mae = float(np.mean(np.abs(left_probs - right_probs)))

    if np.std(left_probs) > 0 and np.std(right_probs) > 0:

        corr = float(
            np.corrcoef(
                left_probs,
                right_probs,
            )[
                0,
                1,
            ]
        )

    else:

        corr = float("nan")

    return {
        "v4_path": str(v4_path),
        "status": "COMPARED",
        "fold_match": fold_match,
        "probability_MAE": mae,
        "probability_Pearson": corr,
    }


def run_gold_reproduction(
    gold_df: pd.DataFrame,
    fold_ids: np.ndarray,
) -> Dict[
    str,
    Any,
]:

    print("\n")
    print("=" * 88)
    print("W3.0 GOLD-ONLY CACHED-FEATURE REPRODUCTION")
    print("=" * 88)

    # The cache-building phase constructed a pretrained ResNet and
    # therefore consumed torch RNG. Reset here so the controlled
    # training RNG starts from the same top-level seed as V4.
    seed_everything(RANDOM_SEED)

    y_all = gold_df[LABEL_COLUMNS].values.astype(np.int64)

    study_uids = gold_df[UID_COLUMN].astype(str).tolist()

    missing_cache = [
        uid
        for uid in (study_uids)
        if not cached_study_is_valid(
            feature_cache_path(uid),
            uid,
        )
    ]

    if missing_cache:

        raise RuntimeError(
            "Gold feature cache incomplete. " f"Missing {len(missing_cache)} studies."
        )

    oof_records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    fold_summaries: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    epoch_oof_records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    training_started = time.time()

    for fold in range(NUM_FOLDS):

        print("\n")
        print("#" * 88)
        print(f"FOLD {fold + 1}/" f"{NUM_FOLDS}")
        print("#" * 88)

        train_indices = np.where(fold_ids != fold)[0]

        val_indices = np.where(fold_ids == fold)[0]

        train_uids = [study_uids[i] for i in (train_indices)]

        val_uids = [study_uids[i] for i in (val_indices)]

        train_y = y_all[train_indices]

        pos_weight = make_pos_weight(train_y)

        train_dataset = CachedFeatureStudyDataset(
            train_uids,
            gold_df,
        )

        val_dataset = CachedFeatureStudyDataset(
            val_uids,
            gold_df,
        )

        # Deliberately create loaders BEFORE model construction,
        # matching V4's order.
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=TRAIN_NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            collate_fn=cached_study_collate_fn,
            persistent_workers=(TRAIN_NUM_WORKERS > 0),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=TRAIN_NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            collate_fn=cached_study_collate_fn,
            persistent_workers=(TRAIN_NUM_WORKERS > 0),
        )

        model = CachedV4MetadataAblationModel(
            metadata_dim=SERIES_META_DIM,
            num_labels=NUM_LABELS,
            align_v4_rng=True,
        ).to(DEVICE)

        trainable_parameters = [
            parameter for parameter in (model.parameters()) if parameter.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=HEAD_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=COSINE_ETA_MIN,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            )
            if DEVICE.type == "cuda"
            else None
        )

        history: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        fold_epoch_records: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        for epoch in range(
            1,
            NUM_EPOCHS + 1,
        ):

            train_metrics, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer=optimizer,
                scaler=scaler,
            )

            with torch.no_grad():

                (
                    val_metrics,
                    y_true,
                    y_prob,
                ) = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    optimizer=None,
                    scaler=None,
                )

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "fold": fold + 1,
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_macro_AUROC": val_metrics["macro_AUROC"],
                "val_macro_AP": val_metrics["macro_AP"],
                "val_macro_F1": val_metrics["macro_F1"],
            }

            history.append(row)

            print(
                f"Fold {fold + 1} "
                f"Epoch {epoch:02d}/{NUM_EPOCHS}: "
                f"loss={row['train_loss']:.4f} "
                f"val_auc={row['val_macro_AUROC']:.4f} "
                f"val_ap={row['val_macro_AP']:.4f} "
                f"val_f1={row['val_macro_F1']:.4f}"
            )

            for row_idx, study_uid in enumerate(val_uids):

                prediction_record: Dict[
                    str,
                    Any,
                ] = {
                    UID_COLUMN: study_uid,
                    "Fold": fold + 1,
                    "Epoch": epoch,
                }

                for label_idx, label in enumerate(LABEL_COLUMNS):

                    prediction_record[f"{label}_true"] = int(
                        y_true[
                            row_idx,
                            label_idx,
                        ]
                    )

                    prediction_record[f"{label}_prob"] = float(
                        y_prob[
                            row_idx,
                            label_idx,
                        ]
                    )

                fold_epoch_records.append(prediction_record)

            if epoch == NUM_EPOCHS:

                atomic_torch_save(
                    {
                        "fold": fold + 1,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": {
                            "pipeline": "cached_frozen_resnet18_features",
                            "feature_cache_version": FEATURE_CACHE_VERSION,
                            "image_size": IMAGE_SIZE,
                            "slices_per_series": SLICES_PER_SERIES,
                            "series_meta_dim": SERIES_META_DIM,
                            "use_series_metadata": False,
                            "metadata_ablation_mode": METADATA_ABLATION_MODE,
                            "head_learning_rate": HEAD_LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "num_epochs": NUM_EPOCHS,
                            "cosine_eta_min": COSINE_ETA_MIN,
                        },
                    },
                    CHECKPOINT_ROOT / f"fold_{fold + 1}_epoch_{NUM_EPOCHS}.pt",
                )

        pd.DataFrame(history).to_csv(
            RESULT_ROOT / f"fold_{fold + 1}_history.csv",
            index=False,
        )

        pd.DataFrame(fold_epoch_records).to_csv(
            RESULT_ROOT / f"fold_{fold + 1}_epoch_predictions.csv",
            index=False,
        )

        epoch_oof_records.extend(fold_epoch_records)

        # Final epoch model remains in memory.
        with torch.no_grad():

            (
                final_metrics,
                y_true_final,
                y_prob_final,
            ) = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                scaler=None,
            )

        for row_idx, study_uid in enumerate(val_uids):

            record: Dict[
                str,
                Any,
            ] = {
                UID_COLUMN: study_uid,
                "Fold": fold + 1,
            }

            for label_idx, label in enumerate(LABEL_COLUMNS):

                record[f"{label}_true"] = int(
                    y_true_final[
                        row_idx,
                        label_idx,
                    ]
                )

                record[f"{label}_prob"] = float(
                    y_prob_final[
                        row_idx,
                        label_idx,
                    ]
                )

            oof_records.append(record)

        fold_summary: Dict[
            str,
            Any,
        ] = {
            "fold": fold + 1,
            "final_epoch": NUM_EPOCHS,
            "final_macro_AUROC": final_metrics["macro_AUROC"],
            "final_macro_AP": final_metrics["macro_AP"],
            "final_macro_F1": final_metrics["macro_F1"],
        }

        for label in LABEL_COLUMNS:

            fold_summary[f"{label}_AUROC"] = final_metrics[f"{label}_AUROC"]

            fold_summary[f"{label}_AP"] = final_metrics[f"{label}_AP"]

        fold_summaries.append(fold_summary)

        print(
            f"Fold {fold + 1} final: "
            f"AUROC={fold_summary['final_macro_AUROC']:.4f} "
            f"AP={fold_summary['final_macro_AP']:.4f} "
            f"F1={fold_summary['final_macro_F1']:.4f}"
        )

        del model
        del optimizer
        del scheduler
        del criterion
        del scaler
        del train_loader
        del val_loader
        del train_dataset
        del val_dataset

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    oof_df = pd.DataFrame(oof_records)

    fold_metrics_df = pd.DataFrame(fold_summaries)

    epoch_oof_df = pd.DataFrame(epoch_oof_records)

    oof_df.to_csv(
        RESULT_ROOT / "oof_predictions.csv",
        index=False,
    )

    fold_metrics_df.to_csv(
        RESULT_ROOT / "fold_metrics.csv",
        index=False,
    )

    epoch_oof_df.to_csv(
        RESULT_ROOT / "epoch_oof_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Complete OOF curve, exactly as in V4
    # --------------------------------------------------------

    epoch_summary_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_df = epoch_oof_df[epoch_oof_df["Epoch"] == epoch].copy()

        if len(epoch_df) != len(gold_df):

            raise RuntimeError(
                f"Epoch {epoch}: expected "
                f"{len(gold_df)} OOF rows, "
                f"found {len(epoch_df)}."
            )

        (
            _,
            epoch_summary,
        ) = pooled_oof_summary(epoch_df)

        epoch_summary_rows.append(
            {
                "Epoch": epoch,
                "OOF_Macro_AUROC": epoch_summary["macro_AUROC"],
                "OOF_Macro_AP": epoch_summary["macro_AP"],
                "OOF_Macro_F1": epoch_summary["macro_F1"],
            }
        )

    epoch_summary_df = pd.DataFrame(epoch_summary_rows)

    epoch_summary_df.to_csv(
        RESULT_ROOT / "epoch_oof_summary.csv",
        index=False,
    )

    (
        per_label_df,
        pooled_summary,
    ) = pooled_oof_summary(oof_df)

    per_label_df.to_csv(
        RESULT_ROOT / "oof_per_label_metrics.csv",
        index=False,
    )

    v4_comparison = compare_with_existing_v4(oof_df)

    auc_delta = pooled_summary["macro_AUROC"] - EXPECTED_V4_OOF_MACRO_AUROC

    ap_delta = pooled_summary["macro_AP"] - EXPECTED_V4_OOF_MACRO_AP

    f1_delta = pooled_summary["macro_F1"] - EXPECTED_V4_OOF_MACRO_F1

    reproduction_pass = bool(
        abs(auc_delta) <= REPRO_AUROC_WARN_TOL and abs(ap_delta) <= REPRO_AP_WARN_TOL
    )

    result = {
        "pooled_oof": pooled_summary,
        "expected_v4_pooled_oof": {
            "macro_AUROC": EXPECTED_V4_OOF_MACRO_AUROC,
            "macro_AP": EXPECTED_V4_OOF_MACRO_AP,
            "macro_F1": EXPECTED_V4_OOF_MACRO_F1,
        },
        "delta_vs_reference": {
            "macro_AUROC": auc_delta,
            "macro_AP": ap_delta,
            "macro_F1": f1_delta,
        },
        "diagnostic_reproduction_gate_pass": reproduction_pass,
        "diagnostic_tolerances": {
            "AUROC": REPRO_AUROC_WARN_TOL,
            "AP": REPRO_AP_WARN_TOL,
        },
        "direct_existing_v4_comparison": v4_comparison,
        "training_seconds": time.time() - training_started,
    }

    with open(
        RESULT_ROOT / "gold_reproduction_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
            allow_nan=True,
        )

    print("\n")
    print("=" * 88)
    print("W3.0 GOLD REPRODUCTION RESULT")
    print("=" * 88)

    print(epoch_summary_df.to_string(index=False))

    print("\nFinal pooled OOF:")
    print(
        f"  AUROC : "
        f"{pooled_summary['macro_AUROC']:.6f} "
        f"(V4 ref {EXPECTED_V4_OOF_MACRO_AUROC:.6f}, "
        f"delta {auc_delta:+.6f})"
    )
    print(
        f"  AP    : "
        f"{pooled_summary['macro_AP']:.6f} "
        f"(V4 ref {EXPECTED_V4_OOF_MACRO_AP:.6f}, "
        f"delta {ap_delta:+.6f})"
    )
    print(
        f"  F1    : "
        f"{pooled_summary['macro_F1']:.6f} "
        f"(V4 ref {EXPECTED_V4_OOF_MACRO_F1:.6f}, "
        f"delta {f1_delta:+.6f})"
    )

    print(
        "\nDiagnostic reproduction gate: "
        + ("PASS" if reproduction_pass else "WARNING")
    )

    if v4_comparison is not None:

        print("\nDirect comparison with existing V4 OOF:")

        print(
            json.dumps(
                v4_comparison,
                indent=2,
                allow_nan=True,
            )
        )

    return result


# ============================================================
# 13. CACHE MANIFEST / FINAL AUDIT
# ============================================================


def build_full_cache_manifest(
    train_df: pd.DataFrame,
    series_lookup: Dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for study_uid in train_df[UID_COLUMN].astype(str).tolist():

        path = feature_cache_path(study_uid)

        if not cached_study_is_valid(
            path,
            study_uid,
        ):

            records.append(
                {
                    UID_COLUMN: study_uid,
                    "Cached": False,
                    "SeriesCount": len(series_lookup.get(study_uid, [])),
                    "CacheBytes": 0,
                    "CachePath": str(path),
                }
            )

            continue

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        records.append(
            {
                UID_COLUMN: study_uid,
                "Cached": True,
                "SeriesCount": int(payload["features"].shape[0]),
                "CacheBytes": int(path.stat().st_size),
                "CachePath": str(path),
            }
        )

    df = pd.DataFrame(records)

    df.to_csv(
        RESULT_ROOT / "feature_manifest_all_4407.csv",
        index=False,
    )

    return df


def write_config(
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_hash: str,
) -> None:

    config = {
        "mode": MODE,
        "data_root": str(DATA_ROOT),
        "work_root": str(WORK_ROOT),
        "feature_cache_root": str(FEATURE_CACHE_ROOT),
        "train_studies": int(len(train_df)),
        "gold_studies": int(len(gold_df)),
        "image_size": IMAGE_SIZE,
        "slices_per_series": SLICES_PER_SERIES,
        "feature_dim": 512,
        "feature_cache_dtype": "float16",
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "emulate_v4_image_cache_quantization": EMULATE_V4_IMAGE_CACHE_QUANTIZATION,
        "backbone": "ResNet18_Weights.DEFAULT",
        "backbone_frozen": True,
        "metadata_used": USE_SERIES_METADATA,
        "metadata_ablation_mode": METADATA_ABLATION_MODE,
        "num_folds": NUM_FOLDS,
        "random_seed": RANDOM_SEED,
        "fold_sha256": fold_hash,
        "expected_fold_sha256": EXPECTED_V4_FOLD_SHA256,
        "fold_sha256_match": fold_hash == EXPECTED_V4_FOLD_SHA256,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "cosine_eta_min": COSINE_ETA_MIN,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": 1,
        "amp": USE_AMP,
        "gpu_count": GPU_COUNT,
        "encoder_device_ids": ENCODER_DEVICE_IDS,
        "encoder_batch_size": ENCODER_BATCH_SIZE,
        "preprocess_threads": PREPROCESS_THREADS,
        "labels": LABEL_COLUMNS,
    }

    with open(
        RESULT_ROOT / "w3_0_config.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            config,
            handle,
            indent=2,
        )


# ============================================================
# 14. MAIN
# ============================================================


def main() -> None:

    total_started = time.time()

    print("=" * 88)
    print("RSNA W3.0 - CACHED RESNET18 + GOLD V4 REPRODUCTION")
    print("=" * 88)

    print(f"Mode               : {MODE}")
    print(f"Data root          : {DATA_ROOT}")
    print(f"Work root          : {WORK_ROOT}")
    print(f"Primary device     : {DEVICE}")
    print(f"CUDA GPU count     : {GPU_COUNT}")
    print(f"Encoder GPUs       : {ENCODER_DEVICE_IDS}")
    print(f"Encoder batch      : {ENCODER_BATCH_SIZE}")
    print(f"Preprocess threads : {PREPROCESS_THREADS}")

    (
        train_df,
        series_df,
        gold_df,
    ) = load_tables()

    (
        fold_ids,
        fold_df,
    ) = build_and_verify_folds(gold_df)

    fold_hash = fold_assignment_sha256(
        fold_df[
            [
                UID_COLUMN,
                "OuterFold",
            ]
        ]
    )

    write_config(
        train_df,
        gold_df,
        fold_hash,
    )

    series_lookup = make_series_lookup(series_df)

    all_uids = train_df[UID_COLUMN].astype(str).tolist()

    gold_uids = gold_df[UID_COLUMN].astype(str).tolist()

    gold_uid_set = set(gold_uids)

    unlabeled_uids = [uid for uid in (all_uids) if uid not in (gold_uid_set)]

    print(f"Train studies       : " f"{len(all_uids)}")
    print(f"Gold / unlabeled    : " f"{len(gold_uids)} / " f"{len(unlabeled_uids)}")

    # --------------------------------------------------------
    # CACHE GOLD FIRST
    # --------------------------------------------------------

    if MODE in {
        "all",
        "cache_gold",
    }:

        build_feature_cache(
            gold_uids,
            series_lookup,
            scope_name="gold_58",
        )

    # --------------------------------------------------------
    # GOLD REPRODUCTION IMMEDIATELY AFTER GOLD CACHE
    # --------------------------------------------------------

    reproduction_result = None

    if MODE in {
        "all",
        "reproduce",
    }:

        reproduction_result = run_gold_reproduction(
            gold_df,
            fold_ids,
        )

    # --------------------------------------------------------
    # FULL 4,407-STUDY CACHE
    # --------------------------------------------------------
    #
    # In "all" mode gold is already cached, so this section only
    # encodes missing studies and resumes safely after interruption.
    # --------------------------------------------------------

    if MODE in {
        "all",
        "cache_all",
    }:

        build_feature_cache(
            all_uids,
            series_lookup,
            scope_name="all_4407",
        )

    full_manifest = build_full_cache_manifest(
        train_df,
        series_lookup,
    )

    cached_n = int(full_manifest["Cached"].sum())

    total_cache_bytes = int(
        full_manifest.loc[
            full_manifest["Cached"],
            "CacheBytes",
        ].sum()
    )

    summary = {
        "mode": MODE,
        "fold_sha256": fold_hash,
        "fold_sha256_match": fold_hash == EXPECTED_V4_FOLD_SHA256,
        "cached_studies": cached_n,
        "expected_train_studies": int(len(train_df)),
        "full_cache_complete": cached_n == len(train_df),
        "cache_bytes": total_cache_bytes,
        "cache_human": human_bytes(total_cache_bytes),
        "gold_reproduction": reproduction_result,
        "total_seconds": time.time() - total_started,
    }

    with open(
        RESULT_ROOT / "W3_0_SUMMARY.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            allow_nan=True,
        )

    print("\n")
    print("=" * 88)
    print("W3.0 COMPLETE")
    print("=" * 88)

    print(f"Cached studies      : " f"{cached_n}/" f"{len(train_df)}")

    print(f"Feature cache size  : " f"{human_bytes(total_cache_bytes)}")

    print(f"Total runtime       : " f"{elapsed_string(time.time() - total_started)}")

    print(f"\nResults:\n" f"{RESULT_ROOT}")

    print(f"\nFeature cache:\n" f"{FEATURE_CACHE_ROOT}")

    if MODE == "all" and cached_n == len(train_df):

        print("\nREADY FOR W3.1: " "all 4,407 study features are cached.")

    elif MODE == "all":

        print("\nWARNING: full feature cache is incomplete.")


if __name__ == "__main__":
    main()
