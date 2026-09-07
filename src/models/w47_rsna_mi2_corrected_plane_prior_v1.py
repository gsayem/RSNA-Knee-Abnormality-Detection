# %%writefile w47_rsna_mi2_corrected_plane_prior_v1.py
#!/usr/bin/env python3
"""
W47 | MedImageInsight Corrected Plane Prior v1
================================================

Single-variable production ablation against W45.

WHAT CHANGES
------------
Only the label-aware plane prior mapping is corrected to match the actual W43
cache plane IDs:
    Axial=0, Coronal=1, Sagittal=2

W44/W45 authored the prior rows in:
    Sagittal, Coronal, Axial

but indexed them directly using the W43 IDs. W47 reorders the same numerical
anatomical priors to:
    Axial, Coronal, Sagittal

No prior strength is changed; only the column mapping is corrected.

WHAT DOES NOT CHANGE
--------------------
* MedImageInsight model and weight SHA256.
* W44 frozen MI2 train embeddings and semantic scores.
* Center-slice replicated-RGB input policy.
* W43 physical MRI geometry / 5-stack-per-series hidden-test preprocessing.
* Hidden dimension 192 and metadata dimensions.
* Label-aware attention and pathology semantic pooling.
* W40 production teacher: 4,349 studies / 32,027 selected cells.
* Pseudo pretraining: 5 epochs, batch 48, LR 3e-4.
* Gold adaptation: all 58 studies, 10 epochs, batch 16, LR 8e-5.
* AdamW weight decay 1e-3 and gradient clip 3.0.
* Production seeds 45101..45105 and pseudo seed 44100.
* Frozen large MI2 encoder.
* Five-seed arithmetic probability mean.
* No calibration.
* Hidden test uses MRI only; reports are not used.

IMPORTANT
---------
W45 checkpoints MUST NOT be reused for W47 inference because they were trained
with the historical plane-prior mismatch. W47 trains new lightweight head
checkpoints from the existing frozen W44 MI2 embedding cache.

Standalone project rule:
* This file does not import, call, execute, runpy, or subprocess another project .py.
* No nn.DataParallel.
* Multiprocessing may spawn this same file/process for MI2 extraction workers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image

W47_SCRIPT_VERSION = "w47_mi2_corrected_plane_prior_v1"
W47_EXPERIMENT = "W47 | MedImageInsight Corrected Plane Prior v1"
W47_OUTPUT_DIR = "rsna_w47_mi2_corrected_plane_prior_v1"

UID_COLUMN = "StudyInstanceUID"
SERIES_UID_COLUMN = "SeriesInstanceUID"
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
LABELS = LABEL_COLUMNS
NUM_LABELS = len(LABEL_COLUMNS)

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_SELECTED_CELLS = 32027
EXPECTED_TRAIN_MI2_TOKENS = 121855
EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

W47_TRAIN_CACHE_VERSION = "w44_frozen_medical_embedding_cache_v1"
W47_TRAIN_INPUT_MODE = "center_slice_replicated_rgb"
W47_MI2_WEIGHT_SHA256 = (
    "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"
)
W47_MI2_DIM = 1024
W47_MI2_NATIVE_SIZE = 480
W47_MI2_BATCH = 4
W47_MI2_CACHE_VERSION = "w47_corrected_plane_mi2_test_embedding_cache_v1"
W47_MI2_MAX_TOKENS = 64

W47_HIDDEN_DIM = 192
W47_PLANE_EMBED_DIM = 16
W47_BINARY_META_DIM = 8
W47_POSITION_DIM = 32
W47_HEAD_DROPOUT = 0.18

# Actual W43 plane IDs used in both frozen train cache and hidden-test cache.
W47_W43_PLANE_TO_INDEX = {"Axial": 0, "Coronal": 1, "Sagittal": 2}

# W47 corrected plane prior.
# Columns are explicitly [Axial, Coronal, Sagittal].
# These are the SAME values originally authored for W44/W45, only reordered
# from [Sagittal, Coronal, Axial] -> [Axial, Coronal, Sagittal].
W47_PLANE_PRIOR_ORDER = ["Axial", "Coronal", "Sagittal"]
W47_PLANE_PRIOR = torch.tensor(
    [
        [0.25, 0.80, 1.00],  # ACL
        [0.20, 1.00, 0.55],  # MCL
        [0.25, 0.90, 1.00],  # Medial Meniscus
        [0.25, 0.90, 1.00],  # Lateral Meniscus
        [0.35, 1.00, 0.55],  # Medial OA
        [0.35, 1.00, 0.55],  # Lateral OA
        [1.00, 0.40, 0.75],  # PF OA
        [0.90, 0.45, 1.00],  # Effusion
        [0.90, 0.50, 0.95],  # Synovitis
        [0.30, 0.35, 1.00],  # Baker's
        [0.85, 0.85, 0.85],  # Contusion
        [0.85, 0.85, 0.85],  # Fracture
    ],
    dtype=torch.float32,
)
W47_PLANE_PRIOR_LOG = torch.log(W47_PLANE_PRIOR.clamp_min(1e-3))

# Historical W45 matrix, kept only for explicit status/audit comparison.
W47_W45_HISTORICAL_PRIOR = torch.tensor(
    [
        [1.00, 0.80, 0.25],
        [0.55, 1.00, 0.20],
        [1.00, 0.90, 0.25],
        [1.00, 0.90, 0.25],
        [0.55, 1.00, 0.35],
        [0.55, 1.00, 0.35],
        [0.75, 0.40, 1.00],
        [1.00, 0.45, 0.90],
        [0.95, 0.50, 0.90],
        [1.00, 0.35, 0.30],
        [0.85, 0.85, 0.85],
        [0.85, 0.85, 0.85],
    ],
    dtype=torch.float32,
)

W47_STACKS_PER_SERIES = 5
W47_STACK_OFFSETS = (-1, 0, +1)
W47_CACHE_IMAGE_SIZE = 224
W47_GEOMETRY_PLANE_CONFIDENCE = 0.80
W47_PLANE_TARGET_AXES = {
    "Axial": ("P", "L"),
    "Coronal": ("I", "L"),
    "Sagittal": ("I", "P"),
}
W47_LPS_VECTOR = {
    "L": np.asarray([+1.0, 0.0, 0.0], dtype=np.float64),
    "R": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    "P": np.asarray([0.0, +1.0, 0.0], dtype=np.float64),
    "A": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "S": np.asarray([0.0, 0.0, +1.0], dtype=np.float64),
    "I": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
}

PSEUDO_SEED = 44100
PRODUCTION_SEEDS = [45101, 45102, 45103, 45104, 45105]
PSEUDO_EPOCHS = 5
GOLD_ADAPT_EPOCHS = 10
PSEUDO_BATCH = 48
GOLD_BATCH = 16
INFERENCE_BATCH = 64
PSEUDO_LR = 3e-4
GOLD_LR = 8e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 3.0

EXPECTED_W40_PROB_SHA256 = (
    "c1c12cdc966f16cef1fe0cf0c68ea9a928ffb29ac53d6325ed433f69ba73e203"
)
EXPECTED_W40_WEIGHT_SHA256 = (
    "6d202381aefad9034d0701476cb1312dfdce38297fdcd906315500c8d8c304f2"
)
EXPECTED_W40_MASK_SHA256 = (
    "4a41a877d7db708dd5a02eb72d86e50d51d4de2fe992a348f0d984a84aa7938f"
)

_W47_SHA_CACHE: Dict[str, str] = {}


W47_PROMPTS = {
    "ACL": {
        "positive": [
            "knee MRI showing anterior cruciate ligament tear",
            "MRI of the knee with ACL rupture or disruption",
            "abnormal torn anterior cruciate ligament on knee MRI",
        ],
        "negative": [
            "knee MRI showing an intact anterior cruciate ligament",
            "normal ACL on knee MRI without tear",
            "preserved anterior cruciate ligament on MRI",
        ],
    },
    "MCL": {
        "positive": [
            "knee MRI showing medial collateral ligament injury or tear",
            "MRI of the knee with MCL sprain or disruption",
            "abnormal medial collateral ligament on knee MRI",
        ],
        "negative": [
            "knee MRI showing an intact medial collateral ligament",
            "normal MCL on knee MRI without injury",
            "preserved medial collateral ligament on MRI",
        ],
    },
    "Medial Meniscus": {
        "positive": [
            "knee MRI showing a medial meniscus tear",
            "abnormal torn medial meniscus on knee MRI",
            "MRI of the knee with medial meniscal tear",
        ],
        "negative": [
            "knee MRI showing an intact medial meniscus",
            "normal medial meniscus without tear on MRI",
            "preserved medial meniscus on knee MRI",
        ],
    },
    "Lateral Meniscus": {
        "positive": [
            "knee MRI showing a lateral meniscus tear",
            "abnormal torn lateral meniscus on knee MRI",
            "MRI of the knee with lateral meniscal tear",
        ],
        "negative": [
            "knee MRI showing an intact lateral meniscus",
            "normal lateral meniscus without tear on MRI",
            "preserved lateral meniscus on knee MRI",
        ],
    },
    "Medial OA": {
        "positive": [
            "knee MRI showing medial compartment osteoarthritis",
            "medial tibiofemoral osteoarthritis on knee MRI",
            "degenerative osteoarthritis of the medial knee compartment",
        ],
        "negative": [
            "knee MRI without medial compartment osteoarthritis",
            "preserved medial tibiofemoral compartment without osteoarthritis",
            "no medial compartment degenerative osteoarthritis on knee MRI",
        ],
    },
    "Lateral OA": {
        "positive": [
            "knee MRI showing lateral compartment osteoarthritis",
            "lateral tibiofemoral osteoarthritis on knee MRI",
            "degenerative osteoarthritis of the lateral knee compartment",
        ],
        "negative": [
            "knee MRI without lateral compartment osteoarthritis",
            "preserved lateral tibiofemoral compartment without osteoarthritis",
            "no lateral compartment degenerative osteoarthritis on knee MRI",
        ],
    },
    "PF OA": {
        "positive": [
            "knee MRI showing patellofemoral osteoarthritis",
            "patellofemoral compartment degenerative osteoarthritis on knee MRI",
            "MRI showing patellar or trochlear osteoarthritis",
        ],
        "negative": [
            "knee MRI without patellofemoral osteoarthritis",
            "preserved patellofemoral joint without degenerative osteoarthritis",
            "no patellofemoral osteoarthritis on knee MRI",
        ],
    },
    "Effusion": {
        "positive": [
            "knee MRI showing joint effusion",
            "MRI of the knee with increased joint fluid effusion",
            "large or moderate knee joint effusion on MRI",
        ],
        "negative": [
            "knee MRI without joint effusion",
            "no significant knee joint fluid on MRI",
            "knee MRI showing no effusion",
        ],
    },
    "Synovitis": {
        "positive": [
            "knee MRI showing synovitis",
            "inflamed thickened synovium on knee MRI",
            "MRI findings of knee synovial inflammation",
        ],
        "negative": [
            "knee MRI without synovitis",
            "no synovial inflammation on knee MRI",
            "normal synovium without synovitis on MRI",
        ],
    },
    "Baker's": {
        "positive": [
            "knee MRI showing a Baker cyst",
            "popliteal Baker's cyst on knee MRI",
            "fluid filled Baker cyst in the posterior knee on MRI",
        ],
        "negative": [
            "knee MRI without a Baker cyst",
            "no popliteal cyst on knee MRI",
            "posterior knee MRI without Baker's cyst",
        ],
    },
    "Contusion": {
        "positive": [
            "knee MRI showing bone contusion or bone bruise",
            "bone marrow edema pattern from osseous contusion on knee MRI",
            "MRI of the knee with traumatic bone contusion",
        ],
        "negative": [
            "knee MRI without bone contusion",
            "no traumatic bone marrow edema or bone bruise on knee MRI",
            "normal marrow without osseous contusion on knee MRI",
        ],
    },
    "Fracture": {
        "positive": [
            "knee MRI showing an acute fracture",
            "fracture line or osseous fracture on knee MRI",
            "MRI of the knee with bone fracture",
        ],
        "negative": [
            "knee MRI without fracture",
            "no acute osseous fracture on knee MRI",
            "intact knee bones without fracture on MRI",
        ],
    },
}


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        if not numeric.isin([0, 1]).all():
            raise ValueError("Expected binary 0/1 values.")
        return numeric.astype(int).astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n", "", "nan", "none"}
    unknown = set(lowered.unique()) - truthy - falsy
    if unknown:
        raise ValueError(f"Unexpected boolean values: {sorted(unknown)[:10]}")
    return lowered.isin(truthy)


def decode_dicom_pixel_array(ds, dicom_path: str) -> np.ndarray:
    """
    Robust native-DICOM repair path retained from W3.0 fixed-v2.
    Pixel bytes are never altered. Compressed transfer syntaxes are
    never guessed or rewritten.
    """

    try:
        return ds.pixel_array

    except ValueError as first_error:
        try:
            transfer_syntax = ds.file_meta.TransferSyntaxUID
        except Exception:
            transfer_syntax = None

        try:
            is_compressed = (
                bool(transfer_syntax.is_compressed)
                if transfer_syntax is not None
                else False
            )
        except Exception:
            is_compressed = False

        if is_compressed or "PixelData" not in ds:
            raise

        original_number_of_frames = getattr(ds, "NumberOfFrames", None)
        original_bits_allocated = getattr(ds, "BitsAllocated", None)
        original_samples_per_pixel = getattr(ds, "SamplesPerPixel", None)
        original_high_bit = getattr(ds, "HighBit", None)

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
            samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
            bits_allocated = int(ds.BitsAllocated)
            bits_stored = int(
                getattr(ds, "BitsStored", bits_allocated) or bits_allocated
            )
            photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
            actual_bytes = len(ds.PixelData)

            if rows <= 0 or columns <= 0 or samples <= 0 or actual_bytes <= 0:
                raise first_error

            # Repair 1: NumberOfFrames mismatch for native data.
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
                    declared_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)

                    if inferred_frames >= 1 and inferred_frames != declared_frames:
                        ds.NumberOfFrames = int(inferred_frames)
                        repaired = ds.pixel_array

                        if repaired.ndim == 2:
                            return repaired

                        restore_metadata()

            # Repair 2: infer native MONOCHROME storage width.
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

            inferred_bits_allocated = (usable_bytes // base_pixels) * 8

            if (
                inferred_bits_allocated not in {8, 16, 32, 64}
                or bits_stored > inferred_bits_allocated
            ):
                raise first_error

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

            return repaired

        except Exception as repair_error:
            restore_metadata()

            if repair_error is first_error:
                raise

            raise RuntimeError(
                "DICOM pixel decode failed and safe native metadata repair "
                f"was not possible: {dicom_path}. Original={first_error}; "
                f"repair={repair_error}"
            ) from first_error


def _w47_log(message: str = "") -> None:
    print(message, flush=True)


def _w47_sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    key = str(path.resolve())
    if key in _W47_SHA_CACHE:
        return _W47_SHA_CACHE[key]
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    value = h.hexdigest()
    _W47_SHA_CACHE[key] = value
    return value


def _w47_dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _w47_project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _w47_is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def _w47_output_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if _w47_is_kaggle():
        return Path("/kaggle/working") / W47_OUTPUT_DIR
    return _w47_project_root() / "output" / "results" / W47_OUTPUT_DIR


def _w47_data_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        _w47_project_root() / "input",
    ]
    for root in candidates:
        if all(
            (root / name).exists()
            for name in [
                "test.csv",
                "test_series.csv",
                "sample_submission.csv",
                "test_series",
            ]
        ):
            return root.resolve()
    return candidates[0].resolve()


def _w47_bounded_glob(root: Path, patterns: Sequence[str]) -> List[Path]:
    output: List[Path] = []
    if not root.exists():
        return output
    for pattern in patterns:
        try:
            output.extend(root.glob(pattern))
        except Exception:
            pass
    return output


def _w47_mi2_root_usable(root: Path) -> bool:
    return bool(
        root.is_dir()
        and (root / "MedImageInsight" / "UniCLModel.py").is_file()
        and (root / "2024.09.27" / "config.yaml").is_file()
        and (
            root / "2024.09.27" / "vision_model" / "medimageinsigt-v1.0.0.pt"
        ).is_file()
        and (root / "2024.09.27" / "language_model" / "clip_tokenizer_4.16.2").is_dir()
    )


def _w47_discover_mi2_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _w47_mi2_root_usable(root):
            raise FileNotFoundError(f"Invalid MedImageInsight root: {root}")
        return root

    candidates = [
        Path("/kaggle/input/datasets/isayem/models/MedImageInsights"),
        _w47_project_root() / "models" / "MedImageInsights",
    ]
    kroot = Path("/kaggle/input")
    candidates += _w47_bounded_glob(
        kroot,
        [
            "MedImageInsights",
            "*/MedImageInsights",
            "*/*/MedImageInsights",
            "*/*/*/MedImageInsights",
            "datasets/*/*/MedImageInsights",
            "datasets/*/*/*/MedImageInsights",
        ],
    )
    valid = sorted(
        {p.resolve() for p in candidates if _w47_mi2_root_usable(p)},
        key=lambda p: (len(str(p)), str(p)),
    )
    if not valid:
        raise FileNotFoundError(
            "MedImageInsight assets were not found. Pass --mi2-root."
        )
    return valid[0]


def _w47_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def _w47_prompt_lists() -> Tuple[List[List[str]], List[List[str]]]:
    positive, negative = [], []
    for label in LABEL_COLUMNS:
        positive.append(W47_PROMPTS[label]["positive"])
        negative.append(W47_PROMPTS[label]["negative"])
    return positive, negative


def _w47_load_mi2_model(root: Path, device: torch.device):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from MedImageInsight.UniCLModel import build_unicl_model
    from MedImageInsight.Utils.Arguments import load_opt_from_config_files
    from MedImageInsight.ImageDataLoader import build_transforms
    from MedImageInsight.LangEncoder import build_tokenizer

    model_dir = root / "2024.09.27"
    config_path = model_dir / "config.yaml"
    vision_path = model_dir / "vision_model" / "medimageinsigt-v1.0.0.pt"
    tokenizer_path = model_dir / "language_model" / "clip_tokenizer_4.16.2"

    opt = load_opt_from_config_files([str(config_path)])
    opt["LANG_ENCODER"]["PRETRAINED_TOKENIZER"] = str(tokenizer_path)
    opt["UNICL_MODEL"]["PRETRAINED"] = str(vision_path)
    try:
        opt["IMAGE_ENCODER"]["SPEC"]["ENABLE_CHECKPOINT"] = False
    except Exception:
        pass
    opt["VERBOSE"] = False

    preprocess = build_transforms(opt, False)
    model = build_unicl_model(opt)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = build_tokenizer(opt["LANG_ENCODER"])
    context_length = int(opt["LANG_ENCODER"]["CONTEXT_LENGTH"])
    return model, preprocess, tokenizer, context_length, vision_path


def _w47_mi2_text_prototypes(
    model, tokenizer, context_length: int, device: torch.device
):
    pos_sets, neg_sets = _w47_prompt_lists()

    def encode(prompts: List[str]) -> torch.Tensor:
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=context_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            x = model.encode_text(tokens)
        return _w47_normalize_rows(x)

    pos_proto, neg_proto = [], []
    for pos, neg in zip(pos_sets, neg_sets):
        pos_proto.append(_w47_normalize_rows(encode(pos).mean(dim=0, keepdim=True))[0])
        neg_proto.append(_w47_normalize_rows(encode(neg).mean(dim=0, keepdim=True))[0])
    return torch.stack(pos_proto), torch.stack(neg_proto)


def _w47_runtime_autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    major, _minor = torch.cuda.get_device_capability(device.index or 0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _w47_encode_images_mi2(
    model, preprocess, images: List[Image.Image], device: torch.device, batch_size: int
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            part = images[start : start + batch_size]
            batch = torch.stack([preprocess(img) for img in part]).to(device)
            with _w47_runtime_autocast(device):
                features = model.encode_image(batch)
            features = _w47_normalize_rows(features).cpu()
            if not torch.isfinite(features).all():
                raise RuntimeError("MedImageInsight produced non-finite embeddings")
            chunks.append(features)
            del batch, features
    return torch.cat(chunks, dim=0)


def _w47_unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero orientation vector")
    return vector / norm


def _w47_array_axis_vectors(iop) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(iop, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected six IOP values, got {values.shape}")
    return _w47_unit_vector(values[3:6]), _w47_unit_vector(values[:3])


def _w47_geometry_plane(iop) -> Tuple[str, float]:
    axis0, axis1 = _w47_array_axis_vectors(iop)
    normal = _w47_unit_vector(np.cross(axis1, axis0))
    scores = {
        "Axial": abs(float(normal[2])),
        "Coronal": abs(float(normal[1])),
        "Sagittal": abs(float(normal[0])),
    }
    plane = max(scores, key=scores.get)
    return plane, float(scores[plane])


def _w47_orientation_spec(iop, target_plane: str) -> Dict[str, Any]:
    target_letters = W47_PLANE_TARGET_AXES[target_plane]
    target0 = W47_LPS_VECTOR[target_letters[0]]
    target1 = W47_LPS_VECTOR[target_letters[1]]
    current0, current1 = _w47_array_axis_vectors(iop)
    identity_score = abs(float(np.dot(current0, target0))) + abs(
        float(np.dot(current1, target1))
    )
    transpose_score = abs(float(np.dot(current1, target0))) + abs(
        float(np.dot(current0, target1))
    )
    transpose = transpose_score > identity_score
    if transpose:
        new0, new1 = current1.copy(), current0.copy()
    else:
        new0, new1 = current0.copy(), current1.copy()
    flip0 = float(np.dot(new0, target0)) < 0
    if flip0:
        new0 *= -1.0
    flip1 = float(np.dot(new1, target1)) < 0
    if flip1:
        new1 *= -1.0
    return {"transpose": bool(transpose), "flip0": bool(flip0), "flip1": bool(flip1)}


def _w47_apply_orientation(
    image: np.ndarray, spec: Optional[Mapping[str, Any]]
) -> np.ndarray:
    result = np.asarray(image)
    if spec is None:
        return np.ascontiguousarray(result)
    if bool(spec["transpose"]):
        result = result.T
    if bool(spec["flip0"]):
        result = np.flip(result, axis=0)
    if bool(spec["flip1"]):
        result = np.flip(result, axis=1)
    return np.ascontiguousarray(result)


def _w47_scalar_slice_position(ds) -> Optional[float]:
    orientation = getattr(ds, "ImageOrientationPatient", None)
    position = getattr(ds, "ImagePositionPatient", None)
    if orientation is None or position is None:
        return None
    try:
        row = np.asarray(orientation[:3], dtype=np.float64)
        col = np.asarray(orientation[3:], dtype=np.float64)
        normal = np.cross(row, col)
        return float(np.dot(np.asarray(position, dtype=np.float64), normal))
    except Exception:
        return None


def _w47_read_series_headers(
    series_root: Path, study_uid: str, series_uid: str
) -> List[Dict[str, Any]]:
    series_dir = series_root / str(study_uid) / str(series_uid)
    paths = sorted(series_dir.glob("*.dcm"))
    records = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            iop = getattr(ds, "ImageOrientationPatient", None)
            records.append(
                {
                    "path": str(path),
                    "position": _w47_scalar_slice_position(ds),
                    "instance": getattr(ds, "InstanceNumber", 0),
                    "iop": list(iop) if iop is not None else None,
                }
            )
        except Exception:
            records.append(
                {"path": str(path), "position": None, "instance": 0, "iop": None}
            )
    if records and all(r["position"] is not None for r in records):
        records.sort(key=lambda r: r["position"])
    else:

        def key(r):
            try:
                return float(r["instance"])
            except Exception:
                return 0.0

        records.sort(key=key)
    return records


def _w47_normalized_slice_position(records: List[Dict[str, Any]], index: int) -> float:
    positions = [r["position"] for r in records]
    if positions and all(v is not None for v in positions):
        values = np.asarray(positions, dtype=np.float64)
        low, high = float(np.min(values)), float(np.max(values))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return float(2.0 * ((values[index] - low) / (high - low)) - 1.0)
    if len(records) <= 1:
        return 0.0
    return float(2.0 * (index / (len(records) - 1)) - 1.0)


def _w47_resize_uint8(
    image: np.ndarray, size: int = W47_CACHE_IMAGE_SIZE
) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if max(image.shape) >= size else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(image.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )[0, 0]
    return resized.clamp(0, 255).round().byte().numpy()


def _w47_robust_triplet_to_uint8(images: Sequence[np.ndarray]) -> np.ndarray:
    if len(images) != 3:
        raise ValueError("2.5D triplet must contain exactly three slices")
    stack = np.stack([np.asarray(x, dtype=np.float32) for x in images], axis=0)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        raise RuntimeError("MRI triplet contains no finite pixels")
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        scaled = np.zeros_like(stack, dtype=np.uint8)
    else:
        scaled = np.clip((stack - low) / (high - low), 0.0, 1.0)
        scaled = np.round(scaled * 255.0).astype(np.uint8)
    return np.stack([_w47_resize_uint8(channel) for channel in scaled], axis=0)


def _w47_decode_record(
    records: List[Dict[str, Any]],
    target_index: int,
    orientation_spec: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, int]:
    n = len(records)
    candidates = [target_index]
    for radius in range(1, min(8, n)):
        candidates.extend([target_index - radius, target_index + radius])
    last_error = None
    for index in candidates:
        if index < 0 or index >= n:
            continue
        path = records[index]["path"]
        try:
            ds = pydicom.dcmread(path, force=True)
            image = decode_dicom_pixel_array(ds, path)
            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]
            if image.ndim != 2:
                raise RuntimeError(f"Expected 2-D MRI slice, got {image.shape}")
            image = image.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            image = image * slope + intercept
            spec = orientation_spec
            if spec is None:
                iop = getattr(ds, "ImageOrientationPatient", None)
                if iop is not None:
                    plane, confidence = _w47_geometry_plane(iop)
                    if confidence >= W47_GEOMETRY_PLANE_CONFIDENCE:
                        spec = _w47_orientation_spec(iop, plane)
            image = _w47_apply_orientation(image, spec)
            return image, index
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"No decodable slice near target={target_index}. Last={last_error}"
    )


def _w47_series_centers(n_slices: int) -> List[int]:
    if n_slices <= 0:
        return []
    if n_slices == 1:
        return [0]
    quantiles = np.linspace(0.10, 0.90, W47_STACKS_PER_SERIES)
    indices = [int(round(q * (n_slices - 1))) for q in quantiles]
    unique = []
    for index in indices:
        index = int(np.clip(index, 0, n_slices - 1))
        if index not in unique:
            unique.append(index)
    return unique


def _w47_build_study_center_images(
    series_root: Path, study_uid: str, study_series: pd.DataFrame
) -> Dict[str, Any]:
    study_series = study_series.copy()
    study_series["_w47_plane_idx"] = study_series["Anatomical_Plane"].map(
        W47_W43_PLANE_TO_INDEX
    )
    study_series = study_series.sort_values(
        ["_w47_plane_idx", "_fluid", "_fs", SERIES_UID_COLUMN],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    centers_rgb: List[Image.Image] = []
    plane_ids, fluid_ids, fs_ids, positions = [], [], [], []

    for _, row in study_series.iterrows():
        series_uid = str(row[SERIES_UID_COLUMN])
        metadata_plane = str(row["Anatomical_Plane"])
        fluid = int(row["_fluid"])
        fs = int(row["_fs"])
        records = _w47_read_series_headers(series_root, study_uid, series_uid)
        if not records:
            continue

        valid_iop = next((r["iop"] for r in records if r["iop"] is not None), None)
        plane_used = metadata_plane
        orientation_spec = None
        if valid_iop is not None:
            geometry_plane, confidence = _w47_geometry_plane(valid_iop)
            if (
                geometry_plane != metadata_plane
                and confidence >= W47_GEOMETRY_PLANE_CONFIDENCE
            ):
                plane_used = geometry_plane
            orientation_spec = _w47_orientation_spec(valid_iop, plane_used)

        for center in _w47_series_centers(len(records)):
            triplet = []
            actual_center = center
            try:
                for offset in W47_STACK_OFFSETS:
                    target = int(np.clip(center + offset, 0, len(records) - 1))
                    image, actual = _w47_decode_record(
                        records, target, orientation_spec
                    )
                    triplet.append(image)
                    if offset == 0:
                        actual_center = actual
                stack = _w47_robust_triplet_to_uint8(triplet)
                center_uint8 = np.asarray(stack[1], dtype=np.uint8)
                rgb = np.repeat(center_uint8[:, :, None], 3, axis=2)
                centers_rgb.append(Image.fromarray(rgb, mode="RGB"))
                plane_ids.append(
                    int(
                        W47_W43_PLANE_TO_INDEX.get(
                            plane_used, W47_W43_PLANE_TO_INDEX[metadata_plane]
                        )
                    )
                )
                fluid_ids.append(fluid)
                fs_ids.append(fs)
                positions.append(
                    float(_w47_normalized_slice_position(records, actual_center))
                )
            except Exception:
                continue

    if not centers_rgb:
        raise RuntimeError(f"Study {study_uid} produced no MI2 center images")

    return {
        "images": centers_rgb,
        "plane": np.asarray(plane_ids, dtype=np.int8),
        "fluid": np.asarray(fluid_ids, dtype=np.int8),
        "fat_suppression": np.asarray(fs_ids, dtype=np.int8),
        "slice_position": np.asarray(positions, dtype=np.float32),
    }


def _w47_mi2_cache_path(cache_root: Path, uid: str) -> Path:
    return (
        cache_root
        / "studies"
        / f"{hashlib.md5(str(uid).encode('utf-8')).hexdigest()}.npz"
    )


def _w47_atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def _w47_mi2_cache_usable(path: Path, uid: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as x:
            if str(x["cache_version"].item()) != W47_MI2_CACHE_VERSION:
                return False
            if str(x["study_uid"].item()) != str(uid):
                return False
            if str(x["encoder_sha256"].item()) != W47_MI2_WEIGHT_SHA256:
                return False
            emb = x["embeddings"]
            sem = x["semantic"]
            if emb.ndim != 2 or emb.shape[1] != W47_MI2_DIM or emb.dtype != np.float16:
                return False
            if sem.shape != (len(emb), NUM_LABELS) or sem.dtype != np.float16:
                return False
            if len(emb) <= 0:
                return False
            for key in ("plane", "fluid", "fat_suppression", "slice_position"):
                if len(x[key]) != len(emb):
                    return False
        return True
    except Exception:
        return False


def _w47_prepare_test_series(
    data_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_df = pd.read_csv(data_root / "test.csv")
    series_df = pd.read_csv(data_root / "test_series.csv")
    sample_df = pd.read_csv(data_root / "sample_submission.csv")
    for frame in (test_df, series_df, sample_df):
        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
    series_df[SERIES_UID_COLUMN] = series_df[SERIES_UID_COLUMN].astype(str)
    series_df["_fluid"] = coerce_bool_series(series_df["Fluid_Sensitive"]).astype(int)
    series_df["_fs"] = coerce_bool_series(series_df["Fat_Suppression"]).astype(int)
    sample_uids = sample_df[UID_COLUMN].astype(str).tolist()
    if set(sample_uids) != set(test_df[UID_COLUMN].astype(str)):
        raise RuntimeError("W47 test/sample UID sets differ")
    if set([c for c in sample_df.columns if c != UID_COLUMN]) != set(LABEL_COLUMNS):
        raise RuntimeError("W47 sample_submission label set mismatch")
    return test_df, series_df, sample_df


def _w47_mi2_worker(
    worker_id: int,
    gpu_id: int,
    uid_shard: Sequence[str],
    data_root_s: str,
    mi2_root_s: str,
    cache_root_s: str,
    batch_size: int,
) -> Dict[str, Any]:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    data_root = Path(data_root_s)
    mi2_root = Path(mi2_root_s)
    cache_root = Path(cache_root_s)
    _test, series_df, _sample = _w47_prepare_test_series(data_root)
    groups = {
        str(uid): group.copy()
        for uid, group in series_df[
            series_df[UID_COLUMN].isin(list(uid_shard))
        ].groupby(UID_COLUMN)
    }

    model, preprocess, tokenizer, context_length, vision_path = _w47_load_mi2_model(
        mi2_root, device
    )
    # Parent process validates the 2.3-GB vision checkpoint SHA once before
    # spawning workers. Do not re-hash the same file independently on each GPU.
    pos_proto, neg_proto = _w47_mi2_text_prototypes(
        model, tokenizer, context_length, device
    )
    pos_cpu, neg_cpu = pos_proto.float().cpu(), neg_proto.float().cpu()

    ok, failed = 0, []
    started = time.time()
    _w47_log("=" * 100)
    _w47_log(f"W47 MI2 WORKER {worker_id} | device={device} | studies={len(uid_shard)}")
    _w47_log("=" * 100)

    for index, uid in enumerate(uid_shard, start=1):
        path = _w47_mi2_cache_path(cache_root, uid)
        if _w47_mi2_cache_usable(path, uid):
            ok += 1
            continue
        try:
            if uid not in groups:
                raise RuntimeError("No test_series rows")
            built = _w47_build_study_center_images(
                data_root / "test_series", uid, groups[uid]
            )
            features = _w47_encode_images_mi2(
                model, preprocess, built["images"], device, batch_size
            )
            semantic = features.float() @ pos_cpu.T - features.float() @ neg_cpu.T
            if semantic.shape != (len(features), NUM_LABELS):
                raise RuntimeError(f"semantic shape mismatch {semantic.shape}")
            _w47_atomic_npz(
                path,
                cache_version=np.asarray(W47_MI2_CACHE_VERSION),
                study_uid=np.asarray(uid),
                encoder_sha256=np.asarray(W47_MI2_WEIGHT_SHA256),
                input_mode=np.asarray("center_slice_replicated_rgb"),
                embeddings=features.numpy().astype(np.float16),
                semantic=semantic.numpy().astype(np.float16),
                plane=built["plane"],
                fluid=built["fluid"],
                fat_suppression=built["fat_suppression"],
                slice_position=built["slice_position"].astype(np.float16),
            )
            ok += 1
        except Exception as exc:
            failed.append({UID_COLUMN: uid, "error": repr(exc)})
        if index <= 3 or index % 50 == 0 or index == len(uid_shard):
            rate = index / max(time.time() - started, 1e-9) * 60.0
            _w47_log(
                f"worker={worker_id} {index:4d}/{len(uid_shard)} ok={ok:4d} fail={len(failed):3d} rate={rate:5.1f} studies/min"
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"worker": worker_id, "gpu": gpu_id, "ok": ok, "failed": failed}


def _w47_mi2_cache_summary(cache_root: Path, sample_df: pd.DataFrame) -> Dict[str, Any]:
    usable, tokens, missing, invalid = 0, 0, [], []
    for uid in sample_df[UID_COLUMN].astype(str):
        path = _w47_mi2_cache_path(cache_root, uid)
        if not path.is_file():
            if len(missing) < 10:
                missing.append(uid)
            continue
        if not _w47_mi2_cache_usable(path, uid):
            if len(invalid) < 10:
                invalid.append(uid)
            continue
        with np.load(path, allow_pickle=False) as x:
            tokens += int(len(x["embeddings"]))
        usable += 1
    return {
        "root": str(cache_root),
        "cache_version": W47_MI2_CACHE_VERSION,
        "usable_studies": usable,
        "expected_studies": int(len(sample_df)),
        "total_tokens": tokens,
        "complete": usable == len(sample_df),
        "missing_examples": missing,
        "invalid_examples": invalid,
    }


def _w47_extract_mi2_test(
    data_root: Path, mi2_root: Path, cache_root: Path, batch_size: int
) -> Dict[str, Any]:
    _test, _series, sample_df = _w47_prepare_test_series(data_root)
    uids = sample_df[UID_COLUMN].astype(str).tolist()
    remaining = [
        u
        for u in uids
        if not _w47_mi2_cache_usable(_w47_mi2_cache_path(cache_root, u), u)
    ]
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < 1:
        raise RuntimeError("W47 MI2 hidden-test extraction requires CUDA")

    _w47_log("=" * 100)
    _w47_log(f"{W47_EXPERIMENT} | MI2 HIDDEN TEST EXTRACTION")
    _w47_log("=" * 100)
    _w47_log(f"Test studies         : {len(uids)}")
    _w47_log(f"Already cached       : {len(uids)-len(remaining)}")
    _w47_log(f"Need extraction      : {len(remaining)}")
    _w47_log(f"MI2 batch            : {batch_size}")
    _w47_log(f"Visible CUDA GPUs    : {visible}")

    if remaining:
        n_workers = min(visible, len(remaining))
        shards = [remaining[i::n_workers] for i in range(n_workers)]
        if n_workers == 1:
            results = [
                _w47_mi2_worker(
                    0,
                    0,
                    shards[0],
                    str(data_root),
                    str(mi2_root),
                    str(cache_root),
                    batch_size,
                )
            ]
        else:
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()

            # Nested targets are not spawn-picklable on all Python versions.
            # Therefore use the module-level proxy defined below.
            processes = []
            for worker_id in range(n_workers):
                p = ctx.Process(
                    target=_w47_mi2_worker_proxy,
                    args=(
                        worker_id,
                        worker_id,
                        shards[worker_id],
                        str(data_root),
                        str(mi2_root),
                        str(cache_root),
                        batch_size,
                        queue,
                    ),
                )
                p.start()
                processes.append(p)
            received = {}
            for _ in processes:
                worker_id, result, error = queue.get()
                received[worker_id] = (result, error)
            for p in processes:
                p.join()
            errors = [
                received[i][1] for i in sorted(received) if received[i][1] is not None
            ]
            exitcodes = [p.exitcode for p in processes]
            if errors or any(code != 0 for code in exitcodes):
                raise RuntimeError(
                    f"W47 MI2 worker failure: errors={errors}, exitcodes={exitcodes}"
                )
            results = [received[i][0] for i in sorted(received)]
        failures = [row for result in results for row in result["failed"]]
        if failures:
            pd.DataFrame(failures).to_csv(
                cache_root.parent / "mi2_test_failures.csv", index=False
            )
            raise RuntimeError(f"MI2 extraction failed for {len(failures)} studies")

    summary = _w47_mi2_cache_summary(cache_root, sample_df)
    if not summary["complete"]:
        raise RuntimeError(f"W47 MI2 cache incomplete: {summary}")
    return summary


def _w47_mi2_worker_proxy(
    worker_id, gpu_id, shard, data_root_s, mi2_root_s, cache_root_s, bsz, queue
):
    try:
        result = _w47_mi2_worker(
            worker_id, gpu_id, shard, data_root_s, mi2_root_s, cache_root_s, bsz
        )
        queue.put((worker_id, result, None))
    except Exception as exc:
        queue.put((worker_id, None, repr(exc)))
        raise


def _w47_position_features(position: torch.Tensor) -> torch.Tensor:
    p = position
    return torch.stack(
        [
            p,
            torch.sin(math.pi * p),
            torch.cos(math.pi * p),
            torch.sin(2 * math.pi * p),
            torch.cos(2 * math.pi * p),
            torch.sin(4 * math.pi * p),
            torch.cos(4 * math.pi * p),
            p * p,
            torch.ones_like(p),
        ],
        dim=-1,
    )


class W47MI2Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(W47_MI2_DIM)
        self.projection = nn.Linear(W47_MI2_DIM, W47_HIDDEN_DIM)
        self.plane_embedding = nn.Embedding(3, W47_PLANE_EMBED_DIM)
        self.fluid_embedding = nn.Embedding(2, W47_BINARY_META_DIM)
        self.fs_embedding = nn.Embedding(2, W47_BINARY_META_DIM)
        self.position_projection = nn.Linear(9, W47_POSITION_DIM)
        self.meta_projection = nn.Linear(
            W47_PLANE_EMBED_DIM + 2 * W47_BINARY_META_DIM + W47_POSITION_DIM,
            W47_HIDDEN_DIM,
        )
        self.token_norm = nn.LayerNorm(W47_HIDDEN_DIM)
        self.queries = nn.Parameter(torch.randn(NUM_LABELS, W47_HIDDEN_DIM) * 0.02)
        self.dropout = nn.Dropout(W47_HEAD_DROPOUT)

    def forward(
        self, branch: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = branch["embeddings"]
        mask = branch["mask"]
        plane = branch["plane"].clamp(0, 2)
        fluid = branch["fluid"].clamp(0, 1)
        fs = branch["fat_suppression"].clamp(0, 1)
        pos = branch["slice_position"]
        visual = self.projection(self.input_norm(x))
        meta = torch.cat(
            [
                self.plane_embedding(plane),
                self.fluid_embedding(fluid),
                self.fs_embedding(fs),
                self.position_projection(_w47_position_features(pos)),
            ],
            dim=-1,
        )
        tokens = self.token_norm(visual + self.meta_projection(meta))
        tokens = self.dropout(tokens)
        scores = torch.einsum("bth,lh->blt", tokens, self.queries) / math.sqrt(
            W47_HIDDEN_DIM
        )
        plane_prior = W47_PLANE_PRIOR_LOG.to(scores.device)
        scores = scores + plane_prior[:, plane].permute(1, 0, 2)
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        attention = torch.softmax(scores.float(), dim=-1).to(tokens.dtype)
        pooled = torch.einsum("blt,bth->blh", attention, tokens)
        semantic_by_label = branch["semantic"].permute(0, 2, 1)
        semantic_pooled = (attention.float() * semantic_by_label.float()).sum(dim=-1)
        return pooled, semantic_pooled


class W47MI2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = W47MI2Branch()
        self.classifier_weight = nn.Parameter(
            torch.randn(NUM_LABELS, W47_HIDDEN_DIM + 1) * 0.02
        )
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        visual, semantic = self.branch(batch)
        features = torch.cat([visual, semantic.unsqueeze(-1).to(visual.dtype)], dim=-1)
        logits = (features * self.classifier_weight[None, :, :]).sum(dim=-1)
        return logits + self.classifier_bias[None, :]


def _w47_select_balanced_indices(
    plane: np.ndarray, n: int, max_tokens: int
) -> np.ndarray:
    if n <= max_tokens:
        return np.arange(n, dtype=np.int64)
    selected: List[int] = []
    per_plane = max(1, max_tokens // 3)
    for plane_id in (0, 1, 2):
        idx = np.where(plane == plane_id)[0]
        if len(idx):
            take = min(per_plane, len(idx))
            chosen = np.linspace(0, len(idx) - 1, take).round().astype(int)
            selected.extend(idx[chosen].tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < max_tokens:
        used = set(selected)
        remainder = np.array([i for i in range(n) if i not in used], dtype=int)
        need = min(max_tokens - len(selected), len(remainder))
        if need:
            chosen = np.linspace(0, len(remainder) - 1, need).round().astype(int)
            selected.extend(remainder[chosen].tolist())
    return np.asarray(sorted(selected[:max_tokens]), dtype=np.int64)


class W47MI2TestStore:
    def __init__(self, cache_root: Path, ordered_uids: Sequence[str]):
        self.uids = [str(u) for u in ordered_uids]
        self.records = []
        for index, uid in enumerate(self.uids, start=1):
            path = _w47_mi2_cache_path(cache_root, uid)
            if not _w47_mi2_cache_usable(path, uid):
                raise RuntimeError(f"Missing/incompatible MI2 test cache: {uid}")
            with np.load(path, allow_pickle=False) as x:
                plane = x["plane"].astype(np.int64)
                selected = _w47_select_balanced_indices(
                    plane, len(plane), W47_MI2_MAX_TOKENS
                )
                self.records.append(
                    {
                        "embeddings": x["embeddings"][selected].copy(),
                        "semantic": x["semantic"][selected].copy(),
                        "plane": x["plane"][selected].astype(np.int8),
                        "fluid": x["fluid"][selected].astype(np.int8),
                        "fat_suppression": x["fat_suppression"][selected].astype(
                            np.int8
                        ),
                        "slice_position": x["slice_position"][selected].astype(
                            np.float16
                        ),
                    }
                )
            if index % 250 == 0 or index == len(self.uids):
                _w47_log(f"  MI2 test cache loaded {index}/{len(self.uids)}")

    def make_batch(
        self, indices: Sequence[int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        records = [self.records[int(i)] for i in indices]
        b = len(records)
        max_t = max(len(r["embeddings"]) for r in records)
        embeddings = torch.zeros(b, max_t, W47_MI2_DIM, dtype=torch.float32)
        semantic = torch.zeros(b, max_t, NUM_LABELS, dtype=torch.float32)
        mask = torch.zeros(b, max_t, dtype=torch.bool)
        plane = torch.zeros(b, max_t, dtype=torch.long)
        fluid = torch.zeros(b, max_t, dtype=torch.long)
        fs = torch.zeros(b, max_t, dtype=torch.long)
        pos = torch.zeros(b, max_t, dtype=torch.float32)
        for bi, rec in enumerate(records):
            n = len(rec["embeddings"])
            embeddings[bi, :n] = torch.from_numpy(rec["embeddings"].astype(np.float32))
            semantic[bi, :n] = torch.from_numpy(rec["semantic"].astype(np.float32))
            mask[bi, :n] = True
            plane[bi, :n] = torch.from_numpy(rec["plane"].astype(np.int64))
            fluid[bi, :n] = torch.from_numpy(rec["fluid"].astype(np.int64))
            fs[bi, :n] = torch.from_numpy(rec["fat_suppression"].astype(np.int64))
            pos[bi, :n] = torch.from_numpy(rec["slice_position"].astype(np.float32))
        return {
            "embeddings": embeddings.to(device),
            "semantic": semantic.to(device),
            "mask": mask.to(device),
            "plane": plane.to(device),
            "fluid": fluid.to(device),
            "fat_suppression": fs.to(device),
            "slice_position": pos.to(device),
        }


def _w47_validate_mi2_assets(root: Path) -> Dict[str, Any]:
    vision = root / "2024.09.27" / "vision_model" / "medimageinsigt-v1.0.0.pt"
    sha = _w47_sha256_file(vision)
    return {
        "root": str(root),
        "vision_weight": str(vision),
        "vision_weight_size_gb": vision.stat().st_size / (1024**3),
        "sha256": sha,
        "expected_sha256": W47_MI2_WEIGHT_SHA256,
        "sha_match": sha == W47_MI2_WEIGHT_SHA256,
        "native_input_size": W47_MI2_NATIVE_SIZE,
        "embedding_dim": W47_MI2_DIM,
        "complete": _w47_mi2_root_usable(root) and sha == W47_MI2_WEIGHT_SHA256,
    }


def _w47_dicom_preflight(data_root: Path, max_series: int = 64) -> Dict[str, Any]:
    series = pd.read_csv(data_root / "test_series.csv")
    series[UID_COLUMN] = series[UID_COLUMN].astype(str)
    series[SERIES_UID_COLUMN] = series[SERIES_UID_COLUMN].astype(str)
    if len(series) > max_series:
        indices = np.unique(np.linspace(0, len(series) - 1, max_series, dtype=int))
        series = series.iloc[indices]
    syntax_repr = {}
    missing = []
    for row in series.itertuples(index=False):
        uid = str(getattr(row, UID_COLUMN))
        suid = str(getattr(row, SERIES_UID_COLUMN))
        paths = sorted((data_root / "test_series" / uid / suid).glob("*.dcm"))
        if not paths:
            missing.append({UID_COLUMN: uid, SERIES_UID_COLUMN: suid})
            continue
        path = paths[0]
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            ts = str(
                getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "UNKNOWN")
            )
            syntax_repr.setdefault(ts, str(path))
        except Exception as exc:
            missing.append({"path": str(path), "error": repr(exc)})
    decoded = {}
    for ts, path in syntax_repr.items():
        try:
            ds = pydicom.dcmread(path, force=True)
            arr = decode_dicom_pixel_array(ds, path)
            decoded[ts] = {
                "decoded": True,
                "shape": list(np.asarray(arr).shape),
                "error": None,
            }
        except Exception as exc:
            decoded[ts] = {"decoded": False, "shape": None, "error": repr(exc)}
    overall = bool(
        syntax_repr and not missing and all(v["decoded"] for v in decoded.values())
    )
    return {
        "sampled_series": int(len(series)),
        "transfer_syntaxes": list(syntax_repr),
        "decode_results": decoded,
        "missing_or_header_failures": missing[:10],
        "overall_pass": overall,
    }


# =============================================================================
# W47 TRAINING INPUT DISCOVERY / VALIDATION
# =============================================================================


def _w47_stable_uid_hash(uid: str) -> str:
    return hashlib.md5(str(uid).encode("utf-8")).hexdigest()


def _w47_seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _w47_fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    x = assignments[[UID_COLUMN, "OuterFold"]].copy()
    x[UID_COLUMN] = x[UID_COLUMN].astype(str)
    x = x.sort_values(UID_COLUMN).reset_index(drop=True)
    payload = "".join(
        f"{uid},{int(fold)}\n" for uid, fold in zip(x[UID_COLUMN], x["OuterFold"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _w47_discover_fold_csv(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        Path(
            "/kaggle/input/datasets/isayem/rsna-w4-0-curia2/results/00_outer_fold_assignments.csv"
        ),
        _w47_project_root() / "output/results/00_outer_fold_assignments.csv",
    ]
    root = Path("/kaggle/input/datasets")
    if root.is_dir():
        candidates += list(root.glob("*/*/results/00_outer_fold_assignments.csv"))
        candidates += list(root.glob("*/*/*/results/00_outer_fold_assignments.csv"))
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("Locked fold CSV not found")


def _w47_discover_w40_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "results/16_final_probabilities_wide.csv").is_file():
            return root
        raise FileNotFoundError(f"Invalid W40 root: {root}")

    candidates = [
        Path(
            "/kaggle/input/datasets/isayem/rsna-w40-fs2-production-teacher-v1/rsna_w40_fs2_production_teacher_v1"
        ),
        _w47_project_root() / "output/results/rsna_w40_fs2_production_teacher_v1",
    ]
    kaggle = Path("/kaggle/input/datasets")
    if kaggle.is_dir():
        for p in kaggle.glob("*/*"):
            if (p / "results/16_final_probabilities_wide.csv").is_file():
                candidates.append(p)
        for p in kaggle.glob("*/*/*"):
            if (p / "results/16_final_probabilities_wide.csv").is_file():
                candidates.append(p)

    for root in candidates:
        if (
            (root / "results/16_final_probabilities_wide.csv").is_file()
            and (root / "results/17_teacher_weights_wide.csv").is_file()
            and (root / "results/18_teacher_mask_wide.csv").is_file()
        ):
            return root.resolve()
    raise FileNotFoundError("W40 production teacher root not found")


def _w47_train_cache_root_usable(root: Path) -> bool:
    studies = root / "studies"
    return studies.is_dir() and any(studies.glob("*.npz"))


def _w47_discover_train_cache_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if _w47_train_cache_root_usable(root):
            return root
        raise FileNotFoundError(f"Invalid W44 MI2 train cache root: {root}")

    candidates = [
        Path(
            "/kaggle/input/datasets/isayem/rsna-w44-multimodel-medical-embedding-fusion-v3/"
            "rsna_w44_multimodel_medical_embedding_fusion_v3/embedding_cache/mi2"
        ),
        _w47_project_root()
        / "output/results/rsna_w44_multimodel_medical_embedding_fusion_v3/embedding_cache/mi2",
    ]
    kaggle = Path("/kaggle/input/datasets")
    if kaggle.is_dir():
        candidates += list(kaggle.glob("*/*/embedding_cache/mi2"))
        candidates += list(kaggle.glob("*/*/*/embedding_cache/mi2"))

    for root in candidates:
        if _w47_train_cache_root_usable(root):
            return root.resolve()
    raise FileNotFoundError("W44 frozen MI2 train embedding cache not found")


def _w47_train_cache_path(root: Path, uid: str) -> Path:
    return root / "studies" / f"{_w47_stable_uid_hash(uid)}.npz"


def _w47_train_cache_usable(path: Path, uid: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as x:
            if str(x["cache_version"].item()) != W47_TRAIN_CACHE_VERSION:
                return False
            if str(x["study_uid"].item()) != str(uid):
                return False
            if str(x["encoder"].item()) != "mi2":
                return False
            if str(x["input_mode"].item()) != W47_TRAIN_INPUT_MODE:
                return False
            if str(x["encoder_sha256"].item()) != W47_MI2_WEIGHT_SHA256:
                return False
            emb = x["embeddings"]
            sem = x["semantic"]
            if emb.ndim != 2 or emb.shape[1] != W47_MI2_DIM or emb.dtype != np.float16:
                return False
            if sem.shape != (len(emb), NUM_LABELS) or sem.dtype != np.float16:
                return False
            if len(emb) <= 0:
                return False
            for key in ("plane", "fluid", "fat_suppression", "slice_position"):
                if len(x[key]) != len(emb):
                    return False
            if not np.isfinite(emb).all() or not np.isfinite(sem).all():
                return False
        return True
    except Exception:
        return False


def _w47_load_train_tables(
    data_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_root / "train.csv")
    train[UID_COLUMN] = train[UID_COLUMN].astype(str)
    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN} train rows, got {len(train)}")
    missing = [c for c in LABEL_COLUMNS if c not in train.columns]
    if missing:
        raise RuntimeError(f"train.csv missing labels: {missing}")
    gold_mask = train[LABEL_COLUMNS].notna().all(axis=1)
    gold = train.loc[gold_mask].copy().reset_index(drop=True)
    unlabeled = train.loc[~gold_mask].copy().reset_index(drop=True)
    if len(gold) != EXPECTED_GOLD or len(unlabeled) != EXPECTED_UNLABELED:
        raise RuntimeError(f"Gold/unlabeled mismatch: {len(gold)}/{len(unlabeled)}")
    return train, gold, unlabeled


def _w47_validate_folds(fold_csv: Path, gold: pd.DataFrame) -> Dict[str, Any]:
    folds = pd.read_csv(fold_csv)
    folds[UID_COLUMN] = folds[UID_COLUMN].astype(str)
    if "OuterFold" not in folds.columns:
        raise RuntimeError("Fold CSV missing OuterFold")
    folds = folds[[UID_COLUMN, "OuterFold"]].copy()
    folds["OuterFold"] = folds["OuterFold"].astype(int)
    sha = _w47_fold_assignment_sha256(folds)
    counts = folds["OuterFold"].value_counts().sort_index().to_dict()
    expected_counts = {1: 11, 2: 12, 3: 11, 4: 11, 5: 13}
    return {
        "path": str(fold_csv),
        "sha256": sha,
        "expected_sha256": EXPECTED_FOLD_SHA256,
        "sha_match": sha == EXPECTED_FOLD_SHA256,
        "uid_set_match": set(folds[UID_COLUMN]) == set(gold[UID_COLUMN]),
        "fold_counts": {str(k): int(v) for k, v in counts.items()},
        "counts_match": counts == expected_counts,
        "complete": (
            sha == EXPECTED_FOLD_SHA256
            and set(folds[UID_COLUMN]) == set(gold[UID_COLUMN])
            and counts == expected_counts
        ),
    }


def _w47_load_w40_teacher(
    w40_root: Path,
    unlabeled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    result_root = w40_root / "results"
    prob_path = result_root / "16_final_probabilities_wide.csv"
    weight_path = result_root / "17_teacher_weights_wide.csv"
    mask_path = result_root / "18_teacher_mask_wide.csv"

    probs = pd.read_csv(prob_path)
    weights = pd.read_csv(weight_path)
    masks = pd.read_csv(mask_path)

    for frame in (probs, weights, masks):
        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
        if frame[UID_COLUMN].duplicated().any():
            raise RuntimeError("W40 wide table contains duplicate UIDs")
        if set(frame[UID_COLUMN]) != set(unlabeled[UID_COLUMN]):
            raise RuntimeError("W40 UID set mismatch")

    order = unlabeled[UID_COLUMN].astype(str).tolist()
    probs = probs.set_index(UID_COLUMN).reindex(order).reset_index()
    weights = weights.set_index(UID_COLUMN).reindex(order).reset_index()
    masks = masks.set_index(UID_COLUMN).reindex(order).reset_index()

    for label in LABEL_COLUMNS:
        probs[label] = pd.to_numeric(probs[label], errors="raise").astype(np.float32)
        weights[label] = pd.to_numeric(weights[label], errors="raise").astype(
            np.float32
        )
        masks[label] = coerce_bool_series(masks[label])

    selected = int(masks[LABEL_COLUMNS].to_numpy(bool).sum())
    prob_sha = _w47_sha256_file(prob_path)
    weight_sha = _w47_sha256_file(weight_path)
    mask_sha = _w47_sha256_file(mask_path)
    info = {
        "root": str(w40_root),
        "rows": int(len(probs)),
        "selected_cells": selected,
        "probability_sha256": prob_sha,
        "weight_sha256": weight_sha,
        "mask_sha256": mask_sha,
        "hashes_match": (
            prob_sha == EXPECTED_W40_PROB_SHA256
            and weight_sha == EXPECTED_W40_WEIGHT_SHA256
            and mask_sha == EXPECTED_W40_MASK_SHA256
        ),
        "complete": (
            len(probs) == EXPECTED_UNLABELED
            and selected == EXPECTED_SELECTED_CELLS
            and prob_sha == EXPECTED_W40_PROB_SHA256
            and weight_sha == EXPECTED_W40_WEIGHT_SHA256
            and mask_sha == EXPECTED_W40_MASK_SHA256
        ),
    }
    if not info["complete"]:
        raise RuntimeError(f"W40 teacher identity failed: {json.dumps(info, indent=2)}")
    return probs, weights, masks, info


def _w47_train_cache_summary(root: Path, train: pd.DataFrame) -> Dict[str, Any]:
    usable = 0
    tokens = 0
    missing, invalid = [], []
    for uid in train[UID_COLUMN].astype(str):
        path = _w47_train_cache_path(root, uid)
        if _w47_train_cache_usable(path, uid):
            usable += 1
            with np.load(path, allow_pickle=False) as x:
                tokens += int(len(x["embeddings"]))
        elif not path.is_file():
            if len(missing) < 10:
                missing.append(uid)
        elif len(invalid) < 10:
            invalid.append(uid)
    return {
        "root": str(root),
        "cache_version": W47_TRAIN_CACHE_VERSION,
        "encoder_sha256": W47_MI2_WEIGHT_SHA256,
        "usable_studies": usable,
        "expected_studies": int(len(train)),
        "total_tokens": tokens,
        "expected_tokens": EXPECTED_TRAIN_MI2_TOKENS,
        "token_count_matches": tokens == EXPECTED_TRAIN_MI2_TOKENS,
        "complete": usable == len(train) and tokens == EXPECTED_TRAIN_MI2_TOKENS,
        "missing_examples": missing,
        "invalid_examples": invalid,
    }


# =============================================================================
# W47 FROZEN FEATURE STORE / TRAINING DATASET
# =============================================================================


class W47TrainStore:
    def __init__(self, cache_root: Path, ordered_uids: Sequence[str]):
        self.uids = [str(u) for u in ordered_uids]
        self.records: Dict[str, Dict[str, np.ndarray]] = {}
        started = time.time()
        _w47_log("Loading W44 frozen MI2 train cache into CPU RAM...")
        for i, uid in enumerate(self.uids, start=1):
            path = _w47_train_cache_path(cache_root, uid)
            if not _w47_train_cache_usable(path, uid):
                raise RuntimeError(f"Missing/incompatible W44 MI2 cache: {uid}")
            with np.load(path, allow_pickle=False) as x:
                plane = x["plane"].astype(np.int64)
                selected = _w47_select_balanced_indices(
                    plane, len(plane), W47_MI2_MAX_TOKENS
                )
                self.records[uid] = {
                    "embeddings": x["embeddings"][selected].copy(),
                    "semantic": x["semantic"][selected].copy(),
                    "plane": x["plane"][selected].astype(np.int8),
                    "fluid": x["fluid"][selected].astype(np.int8),
                    "fat_suppression": x["fat_suppression"][selected].astype(np.int8),
                    "slice_position": x["slice_position"][selected].astype(np.float16),
                }
            if i % 500 == 0 or i == len(self.uids):
                _w47_log(f"  loaded {i}/{len(self.uids)}")
        _w47_log(f"W47 MI2 TrainStore ready in {time.time()-started:.1f}s")

    def get(self, uid: str) -> Dict[str, np.ndarray]:
        return self.records[str(uid)]


class W47TrainDataset(Dataset):
    def __init__(
        self,
        store: W47TrainStore,
        uids: Sequence[str],
        targets: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        weights: Mapping[str, np.ndarray],
    ):
        self.store = store
        self.uids = [str(u) for u in uids]
        self.targets = targets
        self.masks = masks
        self.weights = weights

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        uid = self.uids[index]
        return {
            UID_COLUMN: uid,
            "features": self.store.get(uid),
            "targets": self.targets[uid],
            "target_mask": self.masks[uid],
            "target_weight": self.weights[uid],
        }


def _w47_pad_train_records(
    records: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, torch.Tensor]:
    b = len(records)
    max_t = max(len(r["embeddings"]) for r in records)
    embeddings = torch.zeros(b, max_t, W47_MI2_DIM, dtype=torch.float32)
    semantic = torch.zeros(b, max_t, NUM_LABELS, dtype=torch.float32)
    mask = torch.zeros(b, max_t, dtype=torch.bool)
    plane = torch.zeros(b, max_t, dtype=torch.long)
    fluid = torch.zeros(b, max_t, dtype=torch.long)
    fs = torch.zeros(b, max_t, dtype=torch.long)
    position = torch.zeros(b, max_t, dtype=torch.float32)

    for i, rec in enumerate(records):
        n = len(rec["embeddings"])
        embeddings[i, :n] = torch.from_numpy(rec["embeddings"].astype(np.float32))
        semantic[i, :n] = torch.from_numpy(rec["semantic"].astype(np.float32))
        mask[i, :n] = True
        plane[i, :n] = torch.from_numpy(rec["plane"].astype(np.int64))
        fluid[i, :n] = torch.from_numpy(rec["fluid"].astype(np.int64))
        fs[i, :n] = torch.from_numpy(rec["fat_suppression"].astype(np.int64))
        position[i, :n] = torch.from_numpy(rec["slice_position"].astype(np.float32))

    return {
        "embeddings": embeddings,
        "semantic": semantic,
        "mask": mask,
        "plane": plane,
        "fluid": fluid,
        "fat_suppression": fs,
        "slice_position": position,
    }


def _w47_collate_train(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        UID_COLUMN: [str(item[UID_COLUMN]) for item in batch],
        "features": _w47_pad_train_records([item["features"] for item in batch]),
        "targets": torch.from_numpy(
            np.stack([item["targets"] for item in batch]).astype(np.float32)
        ),
        "target_mask": torch.from_numpy(
            np.stack([item["target_mask"] for item in batch]).astype(bool)
        ),
        "target_weight": torch.from_numpy(
            np.stack([item["target_weight"] for item in batch]).astype(np.float32)
        ),
    }


def _w47_make_train_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_w47_collate_train,
        drop_last=False,
    )


def _w47_move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            out[key] = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out


def _w47_build_target_maps(
    gold: pd.DataFrame,
    unlabeled: pd.DataFrame,
    probs: pd.DataFrame,
    weights: pd.DataFrame,
    masks: pd.DataFrame,
):
    gold_t, gold_m, gold_w = {}, {}, {}
    for _, row in gold.iterrows():
        uid = str(row[UID_COLUMN])
        gold_t[uid] = row[LABEL_COLUMNS].to_numpy(np.float32)
        gold_m[uid] = np.ones(NUM_LABELS, dtype=bool)
        gold_w[uid] = np.ones(NUM_LABELS, dtype=np.float32)

    pidx = probs.set_index(UID_COLUMN)
    widx = weights.set_index(UID_COLUMN)
    midx = masks.set_index(UID_COLUMN)
    pseudo_t, pseudo_m, pseudo_w = {}, {}, {}
    for uid in unlabeled[UID_COLUMN].astype(str):
        pseudo_t[uid] = pidx.loc[uid, LABEL_COLUMNS].to_numpy(np.float32)
        pseudo_m[uid] = midx.loc[uid, LABEL_COLUMNS].to_numpy(bool)
        pseudo_w[uid] = widx.loc[uid, LABEL_COLUMNS].to_numpy(np.float32)
    return gold_t, gold_m, gold_w, pseudo_t, pseudo_m, pseudo_w


def _w47_pseudo_macro_loss(logits, targets, mask, weights) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    effective = mask.float() * weights.float()
    denom = effective.sum(dim=0)
    active = denom > 0
    if not active.any():
        raise RuntimeError("Pseudo batch has no active cells")
    per_label = (cell * effective).sum(dim=0) / denom.clamp_min(1e-6)
    loss = per_label[active].mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Pseudo loss is non-finite")
    return loss


def _w47_gold_pos_weight(gold: pd.DataFrame) -> torch.Tensor:
    y = gold[LABEL_COLUMNS].to_numpy(np.float64)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 5.0)
    return torch.tensor(weight.astype(np.float32))


def _w47_gold_macro_loss(logits, targets, pos_weight) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
        pos_weight=pos_weight.float(),
    )
    loss = cell.mean(dim=0).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Gold loss is non-finite")
    return loss


def _w47_train_pseudo(
    model: W47MI2Head,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PSEUDO_LR, weight_decay=WEIGHT_DECAY
    )
    history = []
    for epoch in range(1, PSEUDO_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        started = time.time()
        for raw in loader:
            batch = _w47_move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"])
            loss = _w47_pseudo_macro_loss(
                logits,
                batch["targets"],
                batch["target_mask"],
                batch["target_weight"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            bs = len(raw[UID_COLUMN])
            total += float(loss.detach().cpu()) * bs
            n += bs
        value = total / max(n, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": value,
                "seconds": time.time() - started,
            }
        )
        _w47_log(f"pseudo epoch {epoch:02d}/{PSEUDO_EPOCHS} loss={value:.6f}")
    return history


def _w47_adapt_gold(
    model: W47MI2Head,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> List[Dict[str, Any]]:
    model.to(device)
    pos_weight = pos_weight.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GOLD_LR, weight_decay=WEIGHT_DECAY
    )
    history = []
    for epoch in range(1, GOLD_ADAPT_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for raw in loader:
            batch = _w47_move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"])
            loss = _w47_gold_macro_loss(logits, batch["targets"], pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            bs = len(raw[UID_COLUMN])
            total += float(loss.detach().cpu()) * bs
            n += bs
        value = total / max(n, 1)
        history.append({"epoch": epoch, "loss": value})
        if epoch in {1, GOLD_ADAPT_EPOCHS}:
            _w47_log(f"gold epoch {epoch:02d}/{GOLD_ADAPT_EPOCHS} loss={value:.6f}")
    return history


def _w47_cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _w47_atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def _w47_checkpoint_dir(output_root: Path) -> Path:
    return output_root / "checkpoints"


def _w47_pseudo_checkpoint(output_root: Path) -> Path:
    return _w47_checkpoint_dir(output_root) / "pseudo_seed_44100.pt"


def _w47_full_checkpoint(output_root: Path, seed: int) -> Path:
    return _w47_checkpoint_dir(output_root) / f"full_seed_{seed}.pt"


def _w47_checkpoint_valid(path: Path, expected_seed: Optional[int] = None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("script_version") != W47_SCRIPT_VERSION:
            return False
        if payload.get("plane_prior_order") != W47_PLANE_PRIOR_ORDER:
            return False
        if expected_seed is not None and int(payload.get("seed", -1)) != int(
            expected_seed
        ):
            return False
        state = payload.get("model_state")
        if not isinstance(state, dict):
            return False
        probe = W47MI2Head()
        probe.load_state_dict(state, strict=True)
        return True
    except Exception:
        return False


def _w47_resolve_train_device(accelerator: str) -> torch.device:
    value = str(accelerator).strip().lower()
    if value == "kaggle_tpu":
        raise RuntimeError("W47 does not implement TPU execution")
    if value == "apple_mps":
        raise RuntimeError("W47 does not implement Apple MPS execution")
    if value == "cpu":
        return torch.device("cpu")
    if value in {"localgpu", "kaggle_t4"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{accelerator} requested but CUDA unavailable")
        return torch.device("cuda:0")
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raise ValueError(
        "accelerator must be auto | localGPU | kaggle_t4 | apple_mps | cpu | kaggle_tpu"
    )


# =============================================================================
# W47 FULL PRODUCTION TRAINING
# =============================================================================


def _w47_train_full(args) -> Dict[str, Any]:
    output_root = _w47_output_root(args.output_root)
    result_root = output_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    _w47_checkpoint_dir(output_root).mkdir(parents=True, exist_ok=True)

    data_root = _w47_data_root(args.data_root)
    train, gold, unlabeled = _w47_load_train_tables(data_root)
    w40_root = _w47_discover_w40_root(args.w40_root)
    train_cache_root = _w47_discover_train_cache_root(args.train_cache_root)
    teacher_p, teacher_w, teacher_m, teacher_info = _w47_load_w40_teacher(
        w40_root, unlabeled
    )
    cache_summary = _w47_train_cache_summary(train_cache_root, train)
    if not cache_summary["complete"]:
        raise RuntimeError(
            "W44 MI2 train cache is incomplete; do not rebuild it for W47. "
            + json.dumps(cache_summary, indent=2)
        )

    device = _w47_resolve_train_device(args.accelerator)
    _w47_log("=" * 100)
    _w47_log(f"{W47_EXPERIMENT} | FULL PRODUCTION TRAINING")
    _w47_log("=" * 100)
    _w47_log(f"Training device       : {device}")
    _w47_log(f"Train MI2 cache       : {train_cache_root}")
    _w47_log(f"MI2 train tokens      : {cache_summary['total_tokens']}")
    _w47_log(f"Pseudo studies        : {len(unlabeled)}")
    _w47_log(f"Selected pseudo cells : {teacher_info['selected_cells']}")
    _w47_log(f"Gold studies          : {len(gold)}")
    _w47_log(f"Production seeds      : {PRODUCTION_SEEDS}")
    _w47_log("Large MI2 encoder      : FROZEN / not loaded")
    _w47_log(f"Plane IDs             : {W47_W43_PLANE_TO_INDEX}")
    _w47_log(f"Prior column order    : {W47_PLANE_PRIOR_ORDER}")

    store = W47TrainStore(train_cache_root, train[UID_COLUMN].astype(str).tolist())

    gold_t, gold_m, gold_w, pseudo_t, pseudo_m, pseudo_w = _w47_build_target_maps(
        gold, unlabeled, teacher_p, teacher_w, teacher_m
    )

    pseudo_ds = W47TrainDataset(
        store,
        unlabeled[UID_COLUMN].astype(str).tolist(),
        pseudo_t,
        pseudo_m,
        pseudo_w,
    )

    # Pseudo initialization is trained once, exactly as the W45 production recipe.
    _w47_seed_everything(PSEUDO_SEED)
    pseudo_model = W47MI2Head()
    pseudo_loader = _w47_make_train_loader(pseudo_ds, PSEUDO_BATCH, True)

    _w47_log("-" * 100)
    _w47_log("STAGE P | W40 PSEUDO PRETRAIN | corrected plane prior")
    _w47_log("-" * 100)
    pseudo_history = _w47_train_pseudo(pseudo_model, pseudo_loader, device)
    pseudo_state = _w47_cpu_state_dict(pseudo_model)

    pseudo_payload = {
        "script_version": W47_SCRIPT_VERSION,
        "experiment": W47_EXPERIMENT,
        "stage": "pseudo",
        "seed": PSEUDO_SEED,
        "plane_ids": W47_W43_PLANE_TO_INDEX,
        "plane_prior_order": W47_PLANE_PRIOR_ORDER,
        "plane_prior": W47_PLANE_PRIOR.tolist(),
        "model_state": pseudo_state,
        "history": pseudo_history,
        "teacher": teacher_info,
        "train_cache": cache_summary,
    }
    _w47_atomic_torch_save(pseudo_payload, _w47_pseudo_checkpoint(output_root))

    del pseudo_model, pseudo_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gold_targets = {
        str(row[UID_COLUMN]): row[LABEL_COLUMNS].to_numpy(np.float32)
        for _, row in gold.iterrows()
    }
    gold_masks = {uid: np.ones(NUM_LABELS, dtype=bool) for uid in gold_targets}
    gold_weights = {uid: np.ones(NUM_LABELS, dtype=np.float32) for uid in gold_targets}
    gold_ds = W47TrainDataset(
        store,
        gold[UID_COLUMN].astype(str).tolist(),
        gold_targets,
        gold_masks,
        gold_weights,
    )
    pos_weight = _w47_gold_pos_weight(gold)

    seed_summaries = []
    for seed in PRODUCTION_SEEDS:
        _w47_log("-" * 100)
        _w47_log(f"FULL ALL-58 | seed={seed} | corrected plane prior")
        _w47_log("-" * 100)
        _w47_seed_everything(seed)
        model = W47MI2Head()
        model.load_state_dict(pseudo_state, strict=True)
        gold_loader = _w47_make_train_loader(gold_ds, GOLD_BATCH, True)
        history = _w47_adapt_gold(model, gold_loader, device, pos_weight)

        payload = {
            "script_version": W47_SCRIPT_VERSION,
            "experiment": W47_EXPERIMENT,
            "stage": "full_all58",
            "seed": int(seed),
            "pseudo_seed": PSEUDO_SEED,
            "plane_ids": W47_W43_PLANE_TO_INDEX,
            "plane_prior_order": W47_PLANE_PRIOR_ORDER,
            "plane_prior": W47_PLANE_PRIOR.tolist(),
            "historical_w45_prior": W47_W45_HISTORICAL_PRIOR.tolist(),
            "model_state": _w47_cpu_state_dict(model),
            "history": history,
            "teacher": teacher_info,
            "train_cache": cache_summary,
            "encoder_frozen": True,
        }
        ckpt = _w47_full_checkpoint(output_root, seed)
        _w47_atomic_torch_save(payload, ckpt)
        seed_summaries.append(
            {
                "seed": seed,
                "checkpoint": str(ckpt),
                "sha256": _w47_sha256_file(ckpt),
                "final_gold_loss": float(history[-1]["loss"]),
            }
        )

        del model, gold_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "script_version": W47_SCRIPT_VERSION,
        "experiment": W47_EXPERIMENT,
        "one_variable_change": "correct W44/W45 plane-prior column mapping to W43 Axial=0,Coronal=1,Sagittal=2 IDs",
        "plane_ids": W47_W43_PLANE_TO_INDEX,
        "plane_prior_order": W47_PLANE_PRIOR_ORDER,
        "corrected_plane_prior": W47_PLANE_PRIOR.tolist(),
        "historical_w45_prior": W47_W45_HISTORICAL_PRIOR.tolist(),
        "pseudo_seed": PSEUDO_SEED,
        "production_seeds": PRODUCTION_SEEDS,
        "pseudo_epochs": PSEUDO_EPOCHS,
        "gold_epochs": GOLD_ADAPT_EPOCHS,
        "teacher": teacher_info,
        "train_cache": cache_summary,
        "checkpoints": seed_summaries,
        "encoder_frozen": True,
    }
    (result_root / "01_full_training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _w47_log("=" * 100)
    _w47_log("W47 FULL TRAINING COMPLETE")
    _w47_log("=" * 100)
    for row in seed_summaries:
        _w47_log(f"seed={row['seed']} final_gold_loss={row['final_gold_loss']:.6f}")
    return summary


# =============================================================================
# W47 HIDDEN TEST EXTRACTION / PREDICTION
# =============================================================================


def _w47_load_full_head(
    checkpoint: Path,
    seed: int,
    device: torch.device,
) -> W47MI2Head:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("script_version") != W47_SCRIPT_VERSION:
        raise RuntimeError(f"W47 checkpoint version mismatch: {checkpoint}")
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"W47 checkpoint seed mismatch: {checkpoint}")
    if payload.get("plane_prior_order") != W47_PLANE_PRIOR_ORDER:
        raise RuntimeError(f"W47 checkpoint plane prior order mismatch: {checkpoint}")
    model = W47MI2Head()
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def _w47_predict_test_seed(
    model: W47MI2Head,
    store: W47MI2TestStore,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(store.uids), INFERENCE_BATCH):
        indices = list(range(start, min(start + INFERENCE_BATCH, len(store.uids))))
        batch = store.make_batch(indices, device)
        logits = model(batch)
        prob = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
        chunks.append(prob)
        del batch, logits
    return np.concatenate(chunks, axis=0)


def _w47_extract_test_mode(args) -> Dict[str, Any]:
    data_root = _w47_data_root(args.data_root)
    output_root = _w47_output_root(args.output_root)
    cache_root = output_root / "test_embedding_cache" / "mi2"
    cache_root.mkdir(parents=True, exist_ok=True)

    mi2_root = _w47_discover_mi2_root(args.mi2_root)
    mi2_info = _w47_validate_mi2_assets(mi2_root)
    if not mi2_info["complete"]:
        raise RuntimeError("MedImageInsight asset identity validation failed")

    # Refuse mutation of attached /kaggle/input cache roots.
    if str(cache_root).startswith("/kaggle/input/"):
        raise RuntimeError("W47 test cache must be writable under /kaggle/working")

    return _w47_extract_mi2_test(
        data_root,
        mi2_root,
        cache_root,
        args.mi2_batch,
    )


def _w47_submission_frame(sample_df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    if values.shape != (len(sample_df), NUM_LABELS):
        raise RuntimeError(f"Prediction shape mismatch: {values.shape}")
    frame = sample_df.copy()
    for j, label in enumerate(LABEL_COLUMNS):
        frame[label] = values[:, j]
    return frame[sample_df.columns]


def _w47_validate_submission_frame(
    frame: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> None:
    if list(frame.columns) != list(sample_df.columns):
        raise RuntimeError("submission column/order mismatch")
    if (
        frame[UID_COLUMN].astype(str).tolist()
        != sample_df[UID_COLUMN].astype(str).tolist()
    ):
        raise RuntimeError("submission UID order mismatch")
    if frame[UID_COLUMN].duplicated().any():
        raise RuntimeError("submission contains duplicate UIDs")
    values = frame[LABEL_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("submission contains non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError("submission probabilities outside [0,1]")


def _w47_submit(args) -> Dict[str, Any]:
    output_root = _w47_output_root(args.output_root)
    result_root = output_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    data_root = _w47_data_root(args.data_root)
    cache_root = output_root / "test_embedding_cache" / "mi2"

    _, _, sample_df = _w47_prepare_test_series(data_root)
    checkpoint_missing = [
        seed
        for seed in PRODUCTION_SEEDS
        if not _w47_checkpoint_valid(_w47_full_checkpoint(output_root, seed), seed)
    ]
    if checkpoint_missing:
        raise RuntimeError(
            f"Missing/incompatible W47 production checkpoints: {checkpoint_missing}. "
            "Run train_full first."
        )

    cache_summary = _w47_mi2_cache_summary(
        cache_root,
        sample_df[UID_COLUMN].astype(str).tolist(),
    )
    if not cache_summary["complete"]:
        raise RuntimeError(
            "W47 hidden-test MI2 cache incomplete. Run extract_test first."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("W47 submission prediction requires CUDA")
    device = torch.device("cuda:0")
    store = W47MI2TestStore(
        cache_root,
        sample_df[UID_COLUMN].astype(str).tolist(),
    )

    _w47_log("=" * 100)
    _w47_log(f"{W47_EXPERIMENT} | 5-SEED TEST PREDICTION")
    _w47_log("=" * 100)
    _w47_log(f"Prediction device     : {device}")
    _w47_log(f"Test studies          : {len(sample_df)}")
    _w47_log(f"Production seeds      : {PRODUCTION_SEEDS}")

    seed_predictions: Dict[int, np.ndarray] = {}
    for seed in PRODUCTION_SEEDS:
        _w47_log(f"Predicting seed={seed}")
        model = _w47_load_full_head(
            _w47_full_checkpoint(output_root, seed),
            seed,
            device,
        )
        seed_predictions[seed] = _w47_predict_test_seed(model, store, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    stack = np.stack([seed_predictions[s] for s in PRODUCTION_SEEDS], axis=0)
    mean_prob = np.clip(
        stack.mean(axis=0),
        1e-6,
        1.0 - 1e-6,
    ).astype(np.float32)
    seed_std = stack.std(axis=0)

    submission = _w47_submission_frame(sample_df, mean_prob)
    _w47_validate_submission_frame(submission, sample_df)

    saved_copy = result_root / "submission_w47_corrected_plane_mi2_5seed.csv"
    submission.to_csv(saved_copy, index=False)
    canonical = (
        Path("/kaggle/working/submission.csv")
        if _w47_is_kaggle()
        else output_root / "submission.csv"
    )
    submission.to_csv(canonical, index=False)

    np.savez_compressed(
        result_root / "20_test_seed_probabilities.npz",
        **{f"seed_{seed}": seed_predictions[seed] for seed in PRODUCTION_SEEDS},
        mean_probability=mean_prob,
    )

    disagreement_rows = []
    for i, uid in enumerate(sample_df[UID_COLUMN].astype(str)):
        for j, label in enumerate(LABEL_COLUMNS):
            disagreement_rows.append(
                {
                    UID_COLUMN: uid,
                    "Label": label,
                    "MeanProbability": float(mean_prob[i, j]),
                    "SeedStd": float(seed_std[i, j]),
                }
            )
    pd.DataFrame(disagreement_rows).to_csv(
        result_root / "21_test_seed_disagreement.csv",
        index=False,
    )

    summary = {
        "script_version": W47_SCRIPT_VERSION,
        "experiment": W47_EXPERIMENT,
        "test_studies": int(len(sample_df)),
        "production_seeds": PRODUCTION_SEEDS,
        "probability_ensemble": "mean",
        "calibration": "none",
        "plane_ids": W47_W43_PLANE_TO_INDEX,
        "plane_prior_order": W47_PLANE_PRIOR_ORDER,
        "corrected_plane_prior": W47_PLANE_PRIOR.tolist(),
        "probability_min": float(mean_prob.min()),
        "probability_max": float(mean_prob.max()),
        "mean_seed_std": float(seed_std.mean()),
        "max_seed_std": float(seed_std.max()),
        "saved_copy": str(saved_copy),
        "canonical_submission": str(canonical),
        "canonical_sha256": _w47_sha256_file(canonical),
    }
    (result_root / "22_submission_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    _w47_log("=" * 100)
    _w47_log("W47 MI2 SUBMISSION READY")
    _w47_log("=" * 100)
    _w47_log(f"Kaggle submission : {canonical}")
    _w47_log(f"Saved copy        : {saved_copy}")
    _w47_log(f"Probability range : [{mean_prob.min():.6f}, {mean_prob.max():.6f}]")
    _w47_log(f"Mean seed std     : {seed_std.mean():.6f}")
    _w47_log(f"Max seed std      : {seed_std.max():.6f}")
    return summary


# =============================================================================
# W47 STATUS / VALIDATION
# =============================================================================


def _w47_production_checkpoint_summary(output_root: Path) -> Dict[str, Any]:
    rows = []
    for seed in PRODUCTION_SEEDS:
        path = _w47_full_checkpoint(output_root, seed)
        valid = _w47_checkpoint_valid(path, seed)
        rows.append(
            {
                "seed": seed,
                "path": str(path),
                "exists": path.is_file(),
                "valid": valid,
                "sha256": _w47_sha256_file(path) if valid else None,
            }
        )
    return {
        "expected_seeds": PRODUCTION_SEEDS,
        "valid_checkpoints": int(sum(bool(x["valid"]) for x in rows)),
        "expected_checkpoints": len(PRODUCTION_SEEDS),
        "complete": all(bool(x["valid"]) for x in rows),
        "checkpoints": rows,
    }


def _w47_status(args) -> Dict[str, Any]:
    output_root = _w47_output_root(args.output_root)
    result_root = output_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    data_root = _w47_data_root(args.data_root)

    dependencies = {
        name: _w47_dependency_available(name)
        for name in [
            "torch",
            "torchvision",
            "transformers",
            "timm",
            "yaml",
            "einops",
            "ftfy",
            "fvcore",
            "mup",
            "sentencepiece",
            "safetensors",
            "pydicom",
            "PIL",
        ]
    }
    dependencies.update(
        {
            "libjpeg": _w47_dependency_available("libjpeg"),
            "openjpeg": _w47_dependency_available("openjpeg"),
            "cv2": _w47_dependency_available("cv2"),
        }
    )

    train_info = {}
    train_error = None
    train = gold = unlabeled = None
    try:
        train, gold, unlabeled = _w47_load_train_tables(data_root)
        train_info = {
            "train": len(train),
            "gold": len(gold),
            "unlabeled": len(unlabeled),
        }
    except Exception as exc:
        train_error = repr(exc)

    folds_info, folds_error = {}, None
    if gold is not None:
        try:
            fold_csv = _w47_discover_fold_csv(args.fold_csv)
            folds_info = _w47_validate_folds(fold_csv, gold)
            if not folds_info["complete"]:
                raise RuntimeError("Locked fold validation failed")
        except Exception as exc:
            folds_error = repr(exc)

    teacher_info, teacher_error = {}, None
    if unlabeled is not None:
        try:
            w40_root = _w47_discover_w40_root(args.w40_root)
            _, _, _, teacher_info = _w47_load_w40_teacher(w40_root, unlabeled)
        except Exception as exc:
            teacher_error = repr(exc)

    train_cache_info, train_cache_error = {}, None
    if train is not None:
        try:
            train_cache_root = _w47_discover_train_cache_root(args.train_cache_root)
            train_cache_info = _w47_train_cache_summary(train_cache_root, train)
            if not train_cache_info["complete"]:
                raise RuntimeError("W44 MI2 train cache validation failed")
        except Exception as exc:
            train_cache_error = repr(exc)

    mi2_info, mi2_error = {}, None
    mi2_root = None
    try:
        mi2_root = _w47_discover_mi2_root(args.mi2_root)
        mi2_info = _w47_validate_mi2_assets(mi2_root)
        if not mi2_info["complete"]:
            raise RuntimeError("MI2 asset identity validation failed")
    except Exception as exc:
        mi2_error = repr(exc)

    dicom_info, dicom_error = {}, None
    try:
        dicom_info = _w47_dicom_preflight(data_root)
        if not dicom_info["overall_pass"]:
            raise RuntimeError("DICOM preflight failed")
    except Exception as exc:
        dicom_error = repr(exc)

    checkpoints = _w47_production_checkpoint_summary(output_root)

    test_cache_info = {}
    try:
        _, _, sample_df = _w47_prepare_test_series(data_root)
        test_cache_info = _w47_mi2_cache_summary(
            output_root / "test_embedding_cache" / "mi2",
            sample_df[UID_COLUMN].astype(str).tolist(),
        )
    except Exception as exc:
        test_cache_info = {"complete": False, "error": repr(exc)}

    prior_audit = {
        "actual_cache_plane_ids": W47_W43_PLANE_TO_INDEX,
        "historical_authored_order": ["Sagittal", "Coronal", "Axial"],
        "corrected_runtime_order": W47_PLANE_PRIOR_ORDER,
        "historical_w45_matrix": W47_W45_HISTORICAL_PRIOR.tolist(),
        "corrected_w47_matrix": W47_PLANE_PRIOR.tolist(),
        "is_exact_column_reversal_0_2": bool(
            torch.equal(
                W47_PLANE_PRIOR,
                W47_W45_HISTORICAL_PRIOR[:, [2, 1, 0]],
            )
        ),
    }

    ready_train = bool(
        train_error is None
        and folds_error is None
        and teacher_error is None
        and train_cache_error is None
    )
    ready_extract = bool(
        mi2_error is None and dicom_error is None and torch.cuda.is_available()
    )
    ready_submit = bool(
        checkpoints["complete"] and test_cache_info.get("complete", False)
    )

    payload = {
        "script_version": W47_SCRIPT_VERSION,
        "experiment": W47_EXPERIMENT,
        "one_variable_change": "correct plane-prior mapping only",
        "runtime": {
            "requested_accelerator": args.accelerator,
            "cuda_available": torch.cuda.is_available(),
            "visible_cuda_devices": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
            "cuda_devices": (
                [
                    {"index": i, "name": torch.cuda.get_device_name(i)}
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else []
            ),
            "data_parallel": False,
            "head_training": "single_gpu_fp32",
            "test_extraction_policy": "one independent MI2 worker per visible GPU",
        },
        "dependencies": dependencies,
        "paths": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "mi2_root": str(mi2_root) if mi2_root else None,
        },
        "train": train_info,
        "train_error": train_error,
        "folds": folds_info,
        "folds_error": folds_error,
        "teacher": teacher_info,
        "teacher_error": teacher_error,
        "train_mi2_cache": train_cache_info,
        "train_cache_error": train_cache_error,
        "mi2": mi2_info,
        "mi2_error": mi2_error,
        "dicom_preflight": dicom_info,
        "dicom_error": dicom_error,
        "production_checkpoints": checkpoints,
        "test_cache": test_cache_info,
        "plane_prior_audit": prior_audit,
        "production_recipe": {
            "pseudo_seed": PSEUDO_SEED,
            "pseudo_epochs": PSEUDO_EPOCHS,
            "all58_gold_epochs": GOLD_ADAPT_EPOCHS,
            "full_seeds": PRODUCTION_SEEDS,
            "encoder_frozen": True,
            "probability_ensemble": "mean",
            "calibration": "none",
            "plane_mapping": "corrected_to_W43_Axial0_Coronal1_Sagittal2",
        },
        "ready_for_train_full": ready_train,
        "ready_for_extract_test": ready_extract,
        "ready_for_submit": ready_submit,
        "project_python_imports": False,
        "reports_used_at_test": False,
    }
    (result_root / "00_status.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _w47_log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def _w47_validate(args, raise_on_failure: bool = True) -> Dict[str, Any]:
    output_root = _w47_output_root(args.output_root)
    result_root = output_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    data_root = _w47_data_root(args.data_root)

    checks: Dict[str, bool] = {}
    errors: Dict[str, str] = {}

    try:
        train, gold, unlabeled = _w47_load_train_tables(data_root)
        checks["train_4407"] = len(train) == EXPECTED_TRAIN
        checks["gold_58"] = len(gold) == EXPECTED_GOLD
        checks["unlabeled_4349"] = len(unlabeled) == EXPECTED_UNLABELED
    except Exception as exc:
        errors["train"] = repr(exc)
        checks["train_4407"] = checks["gold_58"] = checks["unlabeled_4349"] = False
        train = gold = unlabeled = None

    try:
        fold_csv = _w47_discover_fold_csv(args.fold_csv)
        fold_info = _w47_validate_folds(fold_csv, gold)
        checks["fold_sha_locked"] = bool(fold_info["complete"])
    except Exception as exc:
        errors["folds"] = repr(exc)
        checks["fold_sha_locked"] = False

    try:
        w40_root = _w47_discover_w40_root(args.w40_root)
        _, _, _, teacher = _w47_load_w40_teacher(w40_root, unlabeled)
        checks["teacher_32027"] = (
            teacher["selected_cells"] == EXPECTED_SELECTED_CELLS
            and teacher["hashes_match"]
        )
    except Exception as exc:
        errors["teacher"] = repr(exc)
        checks["teacher_32027"] = False

    try:
        cache_root = _w47_discover_train_cache_root(args.train_cache_root)
        cache = _w47_train_cache_summary(cache_root, train)
        checks["train_mi2_cache_complete"] = bool(cache["complete"])
        checks["train_mi2_token_count"] = bool(cache["token_count_matches"])
    except Exception as exc:
        errors["train_cache"] = repr(exc)
        checks["train_mi2_cache_complete"] = False
        checks["train_mi2_token_count"] = False

    checkpoints = _w47_production_checkpoint_summary(output_root)
    checks["production_5seed_complete"] = bool(checkpoints["complete"])

    checks["corrected_plane_prior_order"] = (
        W47_PLANE_PRIOR_ORDER == ["Axial", "Coronal", "Sagittal"]
        and W47_W43_PLANE_TO_INDEX == {"Axial": 0, "Coronal": 1, "Sagittal": 2}
        and torch.equal(
            W47_PLANE_PRIOR,
            W47_W45_HISTORICAL_PRIOR[:, [2, 1, 0]],
        )
    )

    try:
        _, _, sample_df = _w47_prepare_test_series(data_root)
        test_cache = _w47_mi2_cache_summary(
            output_root / "test_embedding_cache" / "mi2",
            sample_df[UID_COLUMN].astype(str).tolist(),
        )
        checks["test_mi2_cache_complete"] = bool(test_cache["complete"])
    except Exception as exc:
        errors["test_cache"] = repr(exc)
        checks["test_mi2_cache_complete"] = False
        sample_df = None

    canonical = (
        Path("/kaggle/working/submission.csv")
        if _w47_is_kaggle()
        else output_root / "submission.csv"
    )
    try:
        submission = pd.read_csv(canonical)
        if sample_df is None:
            _, _, sample_df = _w47_prepare_test_series(data_root)
        _w47_validate_submission_frame(submission, sample_df)
        checks["submission_valid"] = True
    except Exception as exc:
        errors["submission"] = repr(exc)
        checks["submission_valid"] = False

    checks["standalone_project_guard"] = True
    checks["reports_not_used"] = True

    overall = bool(checks and all(checks.values()))
    payload = {
        "script_version": W47_SCRIPT_VERSION,
        "experiment": W47_EXPERIMENT,
        "overall_pass": overall,
        "checks": checks,
        "errors": errors,
        "canonical_submission": str(canonical),
        "plane_mapping": {
            "ids": W47_W43_PLANE_TO_INDEX,
            "prior_order": W47_PLANE_PRIOR_ORDER,
        },
        "production_checkpoints": checkpoints,
    }
    (result_root / "99_validation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _w47_log(json.dumps(payload, indent=2, allow_nan=True))
    if raise_on_failure and not overall:
        raise RuntimeError("W47 validation failed")
    return payload


# =============================================================================
# CLI
# =============================================================================


def _w47_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="W47 standalone MedImageInsight corrected-plane-prior production submission"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "status",
            "train_full",
            "extract_test",
            "submit",
            "validate",
            "run_all",
        ],
        default=None,
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=[
            "status",
            "train_full",
            "extract_test",
            "submit",
            "validate",
            "run_all",
        ],
        default=None,
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="auto | localGPU | kaggle_t4 | apple_mps | cpu | kaggle_tpu",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--mi2-root", default=None)
    parser.add_argument("--w40-root", default=None)
    parser.add_argument("--train-cache-root", default=None)
    parser.add_argument("--fold-csv", default=None)
    parser.add_argument("--mi2-batch", type=int, default=W47_MI2_BATCH)
    return parser


def w47_main() -> None:
    parser = _w47_parser()
    args = parser.parse_args()
    mode = args.mode_flag or args.mode or "status"
    if args.mi2_batch < 1:
        raise ValueError("--mi2-batch must be >= 1")

    output_root = _w47_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if mode == "status":
        _w47_status(args)
    elif mode == "train_full":
        _w47_train_full(args)
    elif mode == "extract_test":
        _w47_extract_test_mode(args)
    elif mode == "submit":
        _w47_submit(args)
    elif mode == "validate":
        _w47_validate(args, raise_on_failure=True)
    elif mode == "run_all":
        status = _w47_status(args)
        if not status["ready_for_train_full"]:
            raise RuntimeError("W47 status is not ready for train_full")
        _w47_train_full(args)
        _w47_extract_test_mode(args)
        _w47_submit(args)
        _w47_validate(args, raise_on_failure=True)
    else:
        raise AssertionError(mode)


if __name__ == "__main__":
    w47_main()
