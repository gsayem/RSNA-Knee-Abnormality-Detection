#!/usr/bin/env python3
# ============================================================
# RSNA Knee Abnormality Detection - W4.0
# CURIA-2 FROZEN RADIOLOGY FOUNDATION MODEL
# + HIERARCHICAL DIAGNOSIS-SPECIFIC ATTENTION
# + GOLD-ONLY CONTROL
# + FOLD-SAFE W2.3 WEAK SUPERVISION
# ============================================================
#
# This is the production W4.0 training script that follows the
# successful Curia-2 technical preflight.
#
# Expensive stage (run once):
#   raw DICOM -> canonical Curia orientation -> official Curia
#   processor -> frozen Curia-2 CLS embedding -> resumable cache.
#
# Cheap stage (run repeatedly):
#   cached slice embeddings -> hierarchical trainable head.
#
# Curia-2 is kept FROZEN in W4.0.  W4.1 is reserved for partial
# backbone fine-tuning if W4.0 establishes a strong representation.
#
# ------------------------------------------------------------
# W4.0 representation
# ------------------------------------------------------------
# Study
#   -> ALL series from train_series.csv
#   -> physical/geometric slice ordering
#   -> up to 24 UNIQUE uniformly sampled slices / series
#   -> orientation canonicalization:
#         axial    -> PL
#         coronal  -> IL
#         sagittal -> IP
#   -> official Curia-2 slow image processor (1 x 512 x 512)
#   -> frozen Curia-2
#   -> CLS token: 768-D per slice (FP16 cache)
#   -> normalized slice position
#   -> 2-layer slice Transformer
#   -> 12 learned diagnosis-specific slice queries
#   -> label-specific series representations
#   -> series metadata:
#         plane, fluid-sensitive, fat-suppression,
#         original slice count, physical stack span
#   -> 12 learned diagnosis-specific series queries
#   -> 12 logits
#
# ------------------------------------------------------------
# Supervision variants
# ------------------------------------------------------------
# A) gold_only
#      exact outer-train gold studies only.
#
# B) fold_safe_weak
#      same gold data PLUS only that outer fold's W2.3 soft labels:
#        SoftLabelAvailable = True
#        CandidateSelectionScore >= 0.50
#        finite soft probability
#
#      Gold and pseudo losses are normalized PER LABEL first so
#      labels with more pseudo cells cannot dominate macro-AUROC.
#
#      total = (8 * gold_loss + 1 * pseudo_loss) / 9
#
# No:
#   - all-58 Stage-B mapper
#   - validation-gold leakage
#   - hard pseudo conversion
#   - candidate-score loss weighting
#   - pseudo pos_weight
#
# ------------------------------------------------------------
# Notebook usage (no bash required)
# ------------------------------------------------------------
# Execute this script in a cell, then:
#
#   run_w40_curia("cache")
#   run_w40_curia("train")
#
# or:
#
#   run_w40_curia("all")
#
# Individual training variants:
#
#   run_w40_curia("train_gold")
#   run_w40_curia("train_weak")
#
# Status only:
#
#   run_w40_curia("status")
#
# ------------------------------------------------------------
# Recommended environment overrides BEFORE executing this cell
# ------------------------------------------------------------
#
# os.environ["W40_CURIA_ROOT"] = \
#     "/kaggle/input/datasets/<owner>/<curia-dataset>/<curia-folder>"
#
# os.environ["W40_W23_ROOT"] = \
#     "/kaggle/input/datasets/<owner>/<w23-dataset>/<w23-folder>"
#
# ============================================================

from __future__ import annotations

import os
import gc
import json
import math
import time
import random
import hashlib
import argparse
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "W40_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"

WORK_ROOT = Path(
    os.environ.get(
        "W40_WORK_ROOT",
        "/kaggle/working/rsna_w4_0_curia2",
    )
)

CACHE_ROOT = WORK_ROOT / "feature_cache"
CACHE_STUDY_ROOT = CACHE_ROOT / "studies"
RESULT_ROOT = WORK_ROOT / "results"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

for _path in (WORK_ROOT, CACHE_ROOT, CACHE_STUDY_ROOT, RESULT_ROOT, CHECKPOINT_ROOT):
    _path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Curia-2
# ------------------------------------------------------------

CURIA_REPO_ID = "raidium/curia-2"
CURIA_REVISION = os.environ.get(
    "W40_CURIA_REVISION",
    "645f566dd9e002505691178917cee265b491c7f7",
).strip()

EXPECTED_CURIA_MODEL_SHA256 = (
    "403a02e27531d2858ecd1e9b1ec2d5ea" "7bfa909f10ff0d9e8416090a6a8c96ef"
)

ALLOW_CURIA_SHA_MISMATCH = os.environ.get(
    "W40_ALLOW_CURIA_SHA_MISMATCH",
    "0",
).strip().lower() in {"1", "true", "yes"}

EXPLICIT_CURIA_ROOT = os.environ.get(
    "W40_CURIA_ROOT",
    "/kaggle/input/datasets/isayem/curia-2-model/curia-2-model",
).strip()

CURIA_HIDDEN_DIM = 768
CURIA_IMAGE_SIZE = 512
CURIA_NUM_CHANNELS = 1
CURIA_PATCH_SIZE = 16

CURIA_BATCH_OVERRIDE = int(
    os.environ.get(
        "W40_CURIA_BATCH",
        "0",
    )
)

CURIA_BATCH_CANDIDATES = [
    int(x)
    for x in os.environ.get(
        "W40_CURIA_BATCH_CANDIDATES",
        "16,32,64,96,128",
    ).split(",")
    if x.strip()
]

MAX_SLICES_PER_SERIES = int(
    os.environ.get(
        "W40_MAX_SLICES_PER_SERIES",
        "24",
    )
)

ORIENTATION_MIN_ALIGNMENT = float(
    os.environ.get(
        "W40_ORIENTATION_MIN_ALIGNMENT",
        "0.70",
    )
)

GEOMETRY_PLANE_CONFIDENCE = float(
    os.environ.get(
        "W40_GEOMETRY_PLANE_CONFIDENCE",
        "0.80",
    )
)

CACHE_VERSION = (
    "w4_0_curia2_cls_allseries_24slice_" "canonical_orientation_slow_processor_v1"
)

RESET_FEATURE_CACHE = os.environ.get(
    "W40_RESET_CACHE",
    "0",
).strip().lower() in {"1", "true", "yes"}


# ------------------------------------------------------------
# W2.3
# ------------------------------------------------------------

EXPLICIT_W23_ROOT = os.environ.get(
    "W40_W23_ROOT",
    "/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3",
).strip()

SELECTION_THRESHOLD = float(
    os.environ.get(
        "W40_SELECTION_THRESHOLD",
        "0.50",
    )
)

GOLD_AUTHORITY = float(
    os.environ.get(
        "W40_GOLD_AUTHORITY",
        "8.0",
    )
)

PSEUDO_AUTHORITY = float(
    os.environ.get(
        "W40_PSEUDO_AUTHORITY",
        "1.0",
    )
)


# ------------------------------------------------------------
# Head
# ------------------------------------------------------------

HEAD_HIDDEN_DIM = int(
    os.environ.get(
        "W40_HEAD_HIDDEN_DIM",
        "384",
    )
)

HEAD_NUM_HEADS = int(
    os.environ.get(
        "W40_HEAD_NUM_HEADS",
        "8",
    )
)

HEAD_TRANSFORMER_LAYERS = int(
    os.environ.get(
        "W40_HEAD_TRANSFORMER_LAYERS",
        "2",
    )
)

HEAD_DROPOUT = float(
    os.environ.get(
        "W40_HEAD_DROPOUT",
        "0.15",
    )
)

SLICE_DROPOUT = float(
    os.environ.get(
        "W40_SLICE_DROPOUT",
        "0.05",
    )
)

SERIES_DROPOUT = float(
    os.environ.get(
        "W40_SERIES_DROPOUT",
        "0.05",
    )
)

HEAD_EPOCHS = int(
    os.environ.get(
        "W40_HEAD_EPOCHS",
        "24",
    )
)

STEPS_PER_EPOCH = int(
    os.environ.get(
        "W40_STEPS_PER_EPOCH",
        "32",
    )
)

GOLD_BATCH_SIZE = int(
    os.environ.get(
        "W40_GOLD_BATCH_SIZE",
        "16",
    )
)

PSEUDO_BATCH_SIZE = int(
    os.environ.get(
        "W40_PSEUDO_BATCH_SIZE",
        "64",
    )
)

VALIDATION_BATCH_SIZE = int(
    os.environ.get(
        "W40_VALIDATION_BATCH_SIZE",
        "8",
    )
)

HEAD_MAX_LR = float(
    os.environ.get(
        "W40_HEAD_MAX_LR",
        "0.001",
    )
)

HEAD_WEIGHT_DECAY = float(
    os.environ.get(
        "W40_HEAD_WEIGHT_DECAY",
        "0.001",
    )
)

GRAD_CLIP_NORM = float(
    os.environ.get(
        "W40_GRAD_CLIP_NORM",
        "5.0",
    )
)

BOOTSTRAP_ITERATIONS = int(
    os.environ.get(
        "W40_BOOTSTRAP_ITERATIONS",
        "2000",
    )
)


# ------------------------------------------------------------
# Exact W3/V4 fold identity
# ------------------------------------------------------------

NUM_FOLDS = 5
RANDOM_SEED = 42

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

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

NUM_LABELS = len(LABEL_COLUMNS)

W3_0_REFERENCE_AUROC = 0.5459179096093576


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
TRAIN_DEVICE = torch.device("cuda:0" if GPU_COUNT > 0 else "cpu")

CACHE_GPU_IDS = list(range(min(GPU_COUNT, 2))) if GPU_COUNT > 0 else []

# Avoid pathological CPU thread oversubscription when two independent
# GPU workers run Curia simultaneously.
try:
    torch.set_num_threads(
        max(
            1,
            min(
                4,
                os.cpu_count() or 4,
            ),
        )
    )
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================================================
# 2. GENERAL HELPERS
# ============================================================


def log(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()


def elapsed_string(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return (
        f"{seconds // 3600:02d}:" f"{(seconds % 3600) // 60:02d}:" f"{seconds % 60:02d}"
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_uid_hash(uid: str) -> str:
    return hashlib.md5(uid.encode("utf-8")).hexdigest()


def study_cache_path(uid: str) -> Path:
    return CACHE_STUDY_ROOT / f"{stable_uid_hash(uid)}.pt"


def atomic_torch_save(payload: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


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


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    return nullcontext()


def make_grad_scaler():
    if TRAIN_DEVICE.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


# ============================================================
# 3. SHALLOW KAGGLE DATASET DISCOVERY
# ============================================================


def shallow_kaggle_dirs(max_depth: int = 4) -> List[Path]:
    """
    Discover attached datasets without recursively traversing the huge
    competition DICOM tree.
    """

    base = Path("/kaggle/input")
    if not base.exists():
        return []

    results: List[Path] = []
    queue: List[Tuple[Path, int]] = [(base, 0)]

    while queue:
        parent, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        try:
            children = [x for x in parent.iterdir() if x.is_dir()]
        except Exception:
            continue

        for child in children:
            # Never descend into the competition dataset hierarchy.
            if child == Path("/kaggle/input/competitions"):
                results.append(child)
                continue

            if str(child).startswith("/kaggle/input/competitions/"):
                continue

            results.append(child)
            queue.append((child, depth + 1))

    return results


# ============================================================
# 4. LOAD DATA + EXACT OUTER FOLDS
# ============================================================


def greedy_multilabel_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples, n_labels = y.shape

    assignments = -np.ones(n_samples, dtype=int)
    label_frequency = y.sum(axis=0) + 1e-8
    sample_rarity = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        positive = np.where(y[i] > 0)[0]
        sample_rarity[i] = (
            float(np.sum(1.0 / label_frequency[positive])) if len(positive) else 0.0
        )

    tie_noise = rng.random(n_samples) * 1e-6
    order = np.argsort(-sample_rarity - tie_noise)

    fold_label_counts = np.zeros((n_splits, n_labels), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=int)
    desired = label_frequency / n_splits

    for sample_idx in order:
        sample = y[sample_idx]
        positive = np.where(sample > 0)[0]
        scores = []

        for fold in range(n_splits):
            if len(positive):
                ratios = fold_label_counts[fold, positive] / (desired[positive] + 1e-8)
                label_score = float(np.mean(ratios))
            else:
                label_score = 0.0

            size_score = fold_sizes[fold] / max(
                1,
                math.ceil(n_samples / n_splits),
            )

            scores.append(label_score + 0.05 * size_score)

        best_fold = int(np.argmin(scores))
        assignments[sample_idx] = best_fold
        fold_sizes[best_fold] += 1
        fold_label_counts[best_fold] += sample

    return assignments


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    ordered = assignments.sort_values(UID_COLUMN).reset_index(drop=True)
    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(
            ordered[UID_COLUMN].astype(str),
            ordered["OuterFold"].astype(int),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_training_tables() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]
):
    for path in (TRAIN_CSV, TRAIN_SERIES_CSV, TRAIN_SERIES_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = pd.read_csv(TRAIN_CSV)
    series_df = pd.read_csv(TRAIN_SERIES_CSV)

    train_df[UID_COLUMN] = train_df[UID_COLUMN].astype(str)
    series_df[UID_COLUMN] = series_df[UID_COLUMN].astype(str)
    series_df[SERIES_UID_COLUMN] = series_df[SERIES_UID_COLUMN].astype(str)

    required_series_columns = {
        UID_COLUMN,
        SERIES_UID_COLUMN,
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }

    missing = required_series_columns - set(series_df.columns)
    if missing:
        raise RuntimeError(
            f"train_series.csv missing required columns: {sorted(missing)}"
        )

    series_df["_fluid"] = coerce_bool_series(series_df["Fluid_Sensitive"]).astype(int)
    series_df["_fs"] = coerce_bool_series(series_df["Fat_Suppression"]).astype(int)

    valid_planes = {"Axial", "Coronal", "Sagittal"}
    bad_planes = (
        set(series_df["Anatomical_Plane"].dropna().astype(str).unique()) - valid_planes
    )
    if bad_planes:
        raise RuntimeError(f"Unexpected anatomical planes: {sorted(bad_planes)}")

    gold_df = (
        train_df[train_df[LABEL_COLUMNS].notna().all(axis=1)]
        .copy()
        .sort_values(UID_COLUMN)
        .reset_index(drop=True)
    )

    if len(gold_df) != 58:
        raise RuntimeError(f"Expected 58 fully labeled studies, found {len(gold_df)}")

    gold_y = gold_df[LABEL_COLUMNS].values.astype(np.int64)
    fold_zero = greedy_multilabel_folds(gold_y, NUM_FOLDS, RANDOM_SEED)

    fold_df = gold_df[[UID_COLUMN, *LABEL_COLUMNS]].copy()
    fold_df["OuterFold"] = fold_zero + 1
    fold_df = fold_df[[UID_COLUMN, "OuterFold", *LABEL_COLUMNS]]

    fold_sha = fold_assignment_sha256(fold_df[[UID_COLUMN, "OuterFold"]])
    if fold_sha != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "W4.0 outer-fold checksum mismatch.\n"
            f"Found   : {fold_sha}\n"
            f"Expected: {EXPECTED_FOLD_SHA256}"
        )

    fold_df.to_csv(
        RESULT_ROOT / "00_outer_fold_assignments.csv",
        index=False,
    )

    return train_df, series_df, gold_df, fold_zero


# ============================================================
# 5. CURIA DISCOVERY + IDENTITY
# ============================================================


def looks_like_curia2_root(path: Path) -> bool:
    required = [
        path / "config.json",
        path / "model.safetensors",
        path / "preprocessor_config.json",
        path / "curia_image_processor.py",
    ]

    if not all(x.exists() for x in required):
        return False

    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        config.get("model_type") == "dinov2"
        and int(config.get("hidden_size", -1)) == CURIA_HIDDEN_DIM
        and int(config.get("num_channels", -1)) == CURIA_NUM_CHANNELS
        and int(config.get("image_size", -1)) == CURIA_IMAGE_SIZE
        and int(config.get("patch_size", -1)) == CURIA_PATCH_SIZE
    )


def discover_curia2_root() -> Path:
    candidates: List[Path] = []

    if EXPLICIT_CURIA_ROOT:
        explicit = Path(EXPLICIT_CURIA_ROOT)
        candidates.extend([explicit, explicit / "curia-2", explicit / "curia-2-model"])

    candidates.extend(
        [
            Path("/kaggle/working/curia-2"),
            Path("/kaggle/working/rsna_w4_0_curia2/pretrained/curia-2"),
        ]
    )

    for root in shallow_kaggle_dirs(max_depth=4):
        candidates.append(root)

    for path in dict.fromkeys(candidates):
        if looks_like_curia2_root(path):
            return path

    raise FileNotFoundError(
        "Curia-2 could not be found. Set W40_CURIA_ROOT to the folder "
        "containing config.json, model.safetensors, preprocessor_config.json, "
        "and curia_image_processor.py."
    )


def validate_curia_identity(curia_root: Path) -> Tuple[str, Dict[str, Any]]:
    weights_path = curia_root / "model.safetensors"
    model_sha = sha256_file(weights_path)

    if model_sha != EXPECTED_CURIA_MODEL_SHA256 and not ALLOW_CURIA_SHA_MISMATCH:
        raise RuntimeError(
            "Curia-2 weight SHA256 mismatch.\n"
            f"Found   : {model_sha}\n"
            f"Expected: {EXPECTED_CURIA_MODEL_SHA256}\n"
            "Set W40_ALLOW_CURIA_SHA_MISMATCH=1 only if this is intentional."
        )

    config = json.loads((curia_root / "config.json").read_text(encoding="utf-8"))

    checks = {
        "model_sha256": model_sha,
        "hidden_size": int(config.get("hidden_size", -1)),
        "num_channels": int(config.get("num_channels", -1)),
        "image_size": int(config.get("image_size", -1)),
        "patch_size": int(config.get("patch_size", -1)),
        "num_hidden_layers": int(config.get("num_hidden_layers", -1)),
        "num_attention_heads": int(config.get("num_attention_heads", -1)),
    }

    expected = {
        "hidden_size": CURIA_HIDDEN_DIM,
        "num_channels": CURIA_NUM_CHANNELS,
        "image_size": CURIA_IMAGE_SIZE,
        "patch_size": CURIA_PATCH_SIZE,
    }

    for key, value in expected.items():
        if checks[key] != value:
            raise RuntimeError(
                f"Curia-2 config mismatch for {key}: {checks[key]} vs {value}"
            )

    return model_sha, checks


def load_curia_processor_and_model(curia_root: Path, device: torch.device):
    try:
        from transformers import AutoModel, AutoImageProcessor
    except Exception as exc:
        raise RuntimeError("transformers is required for Curia-2.") from exc

    # Explicit slow processor is intentional for reproducibility.
    processor = AutoImageProcessor.from_pretrained(
        str(curia_root),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        str(curia_root),
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return processor, model


# ============================================================
# 6. DICOM GEOMETRY / ORIENTATION
# ============================================================

PLANE_TO_INDEX = {
    "Axial": 0,
    "Coronal": 1,
    "Sagittal": 2,
}

INDEX_TO_PLANE = {value: key for key, value in PLANE_TO_INDEX.items()}

PLANE_TARGET_AXES = {
    "Axial": ("P", "L"),
    "Coronal": ("I", "L"),
    "Sagittal": ("I", "P"),
}

LPS_VECTOR = {
    "L": np.asarray([+1.0, 0.0, 0.0], dtype=np.float64),
    "R": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    "P": np.asarray([0.0, +1.0, 0.0], dtype=np.float64),
    "A": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "S": np.asarray([0.0, 0.0, +1.0], dtype=np.float64),
    "I": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
}


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero orientation vector.")
    return vector / norm


def array_axis_vectors_from_iop(
    image_orientation_patient,
) -> Tuple[np.ndarray, np.ndarray]:
    iop = np.asarray(image_orientation_patient, dtype=np.float64)
    if iop.shape != (6,):
        raise ValueError(f"Expected six IOP values, got shape {iop.shape}.")

    # pixel_array axis 0 follows DICOM column direction (IOP[3:6]);
    # pixel_array axis 1 follows DICOM row direction (IOP[0:3]).
    axis0 = unit_vector(iop[3:6])
    axis1 = unit_vector(iop[:3])
    return axis0, axis1


def geometry_plane_from_iop(image_orientation_patient) -> Tuple[str, float]:
    axis0, axis1 = array_axis_vectors_from_iop(image_orientation_patient)
    normal = unit_vector(np.cross(axis1, axis0))

    scores = {
        "Axial": abs(float(normal[2])),
        "Coronal": abs(float(normal[1])),
        "Sagittal": abs(float(normal[0])),
    }

    plane = max(scores, key=scores.get)
    return plane, float(scores[plane])


def orientation_transform_spec(
    image_orientation_patient,
    target_plane: str,
) -> Dict[str, Any]:
    if target_plane not in PLANE_TARGET_AXES:
        raise ValueError(f"Unsupported target plane: {target_plane!r}")

    target_letters = PLANE_TARGET_AXES[target_plane]
    target0 = LPS_VECTOR[target_letters[0]]
    target1 = LPS_VECTOR[target_letters[1]]

    current0, current1 = array_axis_vectors_from_iop(image_orientation_patient)

    identity_score = abs(float(np.dot(current0, target0))) + abs(
        float(np.dot(current1, target1))
    )

    transpose_score = abs(float(np.dot(current1, target0))) + abs(
        float(np.dot(current0, target1))
    )

    transpose = transpose_score > identity_score

    if transpose:
        new0 = current1.copy()
        new1 = current0.copy()
    else:
        new0 = current0.copy()
        new1 = current1.copy()

    flip0 = float(np.dot(new0, target0)) < 0
    if flip0:
        new0 *= -1.0

    flip1 = float(np.dot(new1, target1)) < 0
    if flip1:
        new1 *= -1.0

    alignment0 = float(np.dot(unit_vector(new0), target0))
    alignment1 = float(np.dot(unit_vector(new1), target1))

    return {
        "target_plane": target_plane,
        "target_orientation": "".join(target_letters),
        "transpose": bool(transpose),
        "flip0": bool(flip0),
        "flip1": bool(flip1),
        "alignment0": alignment0,
        "alignment1": alignment1,
        "min_alignment": min(alignment0, alignment1),
    }


def apply_orientation_transform(image: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    result = np.asarray(image)

    if spec["transpose"]:
        result = result.T

    if spec["flip0"]:
        result = np.flip(result, axis=0)

    if spec["flip1"]:
        result = np.flip(result, axis=1)

    return np.ascontiguousarray(result)


def scalar_slice_position(ds) -> Optional[float]:
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


def read_series_headers(study_uid: str, series_uid: str) -> List[Dict[str, Any]]:
    series_dir = TRAIN_SERIES_ROOT / study_uid / series_uid
    paths = sorted(series_dir.glob("*.dcm"))

    records: List[Dict[str, Any]] = []

    for path in paths:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )

            iop = getattr(ds, "ImageOrientationPatient", None)

            records.append(
                {
                    "path": str(path),
                    "position": scalar_slice_position(ds),
                    "instance": getattr(ds, "InstanceNumber", 0),
                    "iop": list(iop) if iop is not None else None,
                }
            )
        except Exception:
            records.append(
                {
                    "path": str(path),
                    "position": None,
                    "instance": 0,
                    "iop": None,
                }
            )

    if records and all(item["position"] is not None for item in records):
        records.sort(key=lambda item: item["position"])
    else:

        def instance_key(item):
            try:
                return float(item["instance"])
            except Exception:
                return 0.0

        records.sort(key=instance_key)

    return records


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


def selected_unique_indices(n_slices: int, max_slices: int) -> List[int]:
    if n_slices <= 0:
        return []

    wanted = min(n_slices, max_slices)

    if wanted == n_slices:
        return list(range(n_slices))

    raw = np.linspace(0, n_slices - 1, wanted)
    indices = np.rint(raw).astype(int).tolist()
    indices = list(dict.fromkeys(indices))

    # Defensive fill: round(linspace) should be unique when wanted <= n,
    # but make the contract exact if an unusual numerical edge occurs.
    if len(indices) < wanted:
        used = set(indices)
        for index in range(n_slices):
            if index not in used:
                indices.append(index)
                used.add(index)
                if len(indices) == wanted:
                    break

    return sorted(indices[:wanted])


def normalized_slice_position(records: List[Dict[str, Any]], index: int) -> float:
    positions = [item["position"] for item in records]

    if positions and all(value is not None for value in positions):
        values = np.asarray(positions, dtype=np.float64)
        low = float(np.min(values))
        high = float(np.max(values))

        if np.isfinite(low) and np.isfinite(high) and high > low:
            return float(2.0 * ((values[index] - low) / (high - low)) - 1.0)

    if len(records) <= 1:
        return 0.0

    return float(2.0 * (index / (len(records) - 1)) - 1.0)


def physical_stack_span_mm(records: List[Dict[str, Any]]) -> float:
    positions = [item["position"] for item in records]

    if positions and all(value is not None for value in positions):
        values = np.asarray(positions, dtype=np.float64)
        if np.isfinite(values).all():
            return float(abs(values[-1] - values[0]))

    return float("nan")


def decode_nearest_unused_slice(
    records: List[Dict[str, Any]],
    target_index: int,
    used_indices: set,
) -> Tuple[np.ndarray, Any, int, str]:
    n = len(records)
    candidates = [target_index]

    for distance in range(1, n):
        left = target_index - distance
        right = target_index + distance

        if left >= 0:
            candidates.append(left)

        if right < n:
            candidates.append(right)

    last_error = None

    for index in candidates:
        if index in used_indices:
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

            return image, ds, index, path

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"No unused decodable slice remained near selected index {target_index}."
    ) from last_error


# ============================================================
# 7. CACHE CONFIG + CURIA BATCH AUTOTUNE
# ============================================================

CACHE_META_PATH = CACHE_ROOT / "cache_meta.json"
CACHE_BENCHMARK_PATH = RESULT_ROOT / "curia_batch_benchmark.csv"


def cache_identity(model_sha: str) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "curia_repo_id": CURIA_REPO_ID,
        "curia_revision": CURIA_REVISION,
        "curia_model_sha256": model_sha,
        "embedding": "last_hidden_state[:,0] CLS token",
        "embedding_dim": CURIA_HIDDEN_DIM,
        "processor": "official CuriaImageProcessor, use_fast=False",
        "image_size": CURIA_IMAGE_SIZE,
        "num_channels": CURIA_NUM_CHANNELS,
        "max_slices_per_series": MAX_SLICES_PER_SERIES,
        "orientation_targets": {
            key: list(value) for key, value in PLANE_TARGET_AXES.items()
        },
        "orientation_min_alignment": ORIENTATION_MIN_ALIGNMENT,
        "geometry_plane_confidence": GEOMETRY_PLANE_CONFIDENCE,
        "all_train_series": True,
        "feature_dtype": "float16",
    }


def prepare_cache_identity(model_sha: str) -> None:
    expected = cache_identity(model_sha)

    if RESET_FEATURE_CACHE:
        log("W40_RESET_CACHE=1 -> deleting existing per-study feature cache.")
        for path in CACHE_STUDY_ROOT.glob("*.pt"):
            path.unlink()
        if CACHE_META_PATH.exists():
            CACHE_META_PATH.unlink()

    if CACHE_META_PATH.exists():
        actual = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                "Existing W4.0 cache identity differs from the current run. "
                "Set W40_RESET_CACHE=1 only if you intentionally want to rebuild it.\n"
                f"Existing: {json.dumps(actual, indent=2)}\n"
                f"Current : {json.dumps(expected, indent=2)}"
            )
    else:
        CACHE_META_PATH.write_text(
            json.dumps(expected, indent=2),
            encoding="utf-8",
        )


def benchmark_curia_batch_size(curia_root: Path) -> int:
    if CURIA_BATCH_OVERRIDE > 0:
        log(f"Curia batch override: {CURIA_BATCH_OVERRIDE}")
        return CURIA_BATCH_OVERRIDE

    if GPU_COUNT <= 0:
        return 4

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    processor, model = load_curia_processor_and_model(curia_root, device)
    del processor

    rows: List[Dict[str, Any]] = []

    log("\nAutotuning Curia inference batch on GPU 0...")

    for batch_size in CURIA_BATCH_CANDIDATES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            dummy = torch.zeros(
                batch_size,
                1,
                CURIA_IMAGE_SIZE,
                CURIA_IMAGE_SIZE,
                device=device,
                dtype=torch.float32,
            )

            # Warm-up.
            with torch.inference_mode(), autocast_context(device):
                output = model(pixel_values=dummy)
                _ = output.last_hidden_state[:, 0]

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            iterations = 2

            for _ in range(iterations):
                with torch.inference_mode(), autocast_context(device):
                    output = model(pixel_values=dummy)
                    _ = output.last_hidden_state[:, 0]

            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started

            rate = batch_size * iterations / max(seconds, 1e-9)
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

            rows.append(
                {
                    "BatchSize": batch_size,
                    "SlicesPerSecond": rate,
                    "PeakAllocatedGB": peak_allocated,
                    "PeakReservedGB": peak_reserved,
                    "Status": "PASS",
                }
            )

            log(
                f"  batch={batch_size:3d} "
                f"rate={rate:7.2f} slices/s "
                f"peak={peak_allocated:5.2f} GB"
            )

            del dummy, output

        except torch.cuda.OutOfMemoryError:
            rows.append(
                {
                    "BatchSize": batch_size,
                    "SlicesPerSecond": float("nan"),
                    "PeakAllocatedGB": float("nan"),
                    "PeakReservedGB": float("nan"),
                    "Status": "OOM",
                }
            )
            log(f"  batch={batch_size:3d} OOM")
            torch.cuda.empty_cache()
            break

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(CACHE_BENCHMARK_PATH, index=False)

    passed = benchmark_df[benchmark_df["Status"] == "PASS"].copy()
    if len(passed) == 0:
        raise RuntimeError("No Curia batch size passed on GPU 0.")

    best_rate = float(passed["SlicesPerSecond"].max())

    # Use the smallest batch within 98% of the best measured throughput.
    # This leaves additional memory margin without sacrificing material speed.
    near_best = passed[passed["SlicesPerSecond"] >= 0.98 * best_rate].sort_values(
        "BatchSize"
    )

    chosen = int(near_best.iloc[0]["BatchSize"])

    log(f"Chosen Curia FP16 batch size: {chosen}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return chosen


# ============================================================
# 8. ENCODE ONE STUDY WITH CURIA-2
# ============================================================


def encode_study(
    study_uid: str,
    study_series: pd.DataFrame,
    processor,
    model: nn.Module,
    device: torch.device,
    encoder_batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns a compact, variable-series cache payload.

    payload["features"]       : [S, K, 768] float16, K=max selected slices
    payload["slice_mask"]     : [S, K] bool
    payload["slice_position"] : [S, K] float16 in [-1,1]
    payload["series_meta"]    : [S,3] int8 -> plane, fluid, FS
    payload["series_cont"]    : [S,2] float16 -> log slice count, span
    """

    # Deterministic series order independent of CSV row ordering.
    study_series = study_series.copy()
    study_series["_plane_idx"] = study_series["Anatomical_Plane"].map(PLANE_TO_INDEX)
    study_series = study_series.sort_values(
        ["_plane_idx", "_fluid", "_fs", SERIES_UID_COLUMN],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    series_raw: List[Dict[str, Any]] = []
    series_audit: List[Dict[str, Any]] = []

    for _, row in study_series.iterrows():
        series_uid = str(row[SERIES_UID_COLUMN])
        metadata_plane = str(row["Anatomical_Plane"])
        fluid = int(row["_fluid"])
        fat_suppression = int(row["_fs"])

        records = read_series_headers(study_uid, series_uid)

        if len(records) == 0:
            series_audit.append(
                {
                    UID_COLUMN: study_uid,
                    SERIES_UID_COLUMN: series_uid,
                    "Status": "NO_DICOMS",
                    "MetadataPlane": metadata_plane,
                    "OriginalSliceCount": 0,
                }
            )
            continue

        valid_iop = next(
            (item["iop"] for item in records if item["iop"] is not None),
            None,
        )

        geometry_plane = metadata_plane
        geometry_confidence = float("nan")
        plane_match = True
        orientation_spec = None

        if valid_iop is not None:
            geometry_plane, geometry_confidence = geometry_plane_from_iop(valid_iop)
            plane_match = geometry_plane == metadata_plane

            if not plane_match and geometry_confidence >= GEOMETRY_PLANE_CONFIDENCE:
                orientation_plane = geometry_plane
            else:
                orientation_plane = metadata_plane

            orientation_spec = orientation_transform_spec(valid_iop, orientation_plane)
        else:
            orientation_plane = metadata_plane

        selected_targets = selected_unique_indices(
            len(records),
            MAX_SLICES_PER_SERIES,
        )

        used_indices: set = set()
        decoded_slices: List[Dict[str, Any]] = []
        decode_failures = 0

        for target_index in selected_targets:
            try:
                image, ds, actual_index, path = decode_nearest_unused_slice(
                    records,
                    target_index,
                    used_indices,
                )

                used_indices.add(actual_index)

                # If the header had no IOP but this full DICOM has one,
                # construct the transform from the decoded slice.
                if orientation_spec is None:
                    iop = getattr(ds, "ImageOrientationPatient", None)
                    if iop is not None:
                        geometry_plane, geometry_confidence = geometry_plane_from_iop(
                            iop
                        )
                        plane_match = geometry_plane == metadata_plane
                        orientation_plane = (
                            geometry_plane
                            if (
                                not plane_match
                                and geometry_confidence >= GEOMETRY_PLANE_CONFIDENCE
                            )
                            else metadata_plane
                        )
                        orientation_spec = orientation_transform_spec(
                            iop, orientation_plane
                        )

                if orientation_spec is not None:
                    image = apply_orientation_transform(image, orientation_spec)

                decoded_slices.append(
                    {
                        "target_index": int(target_index),
                        "actual_index": int(actual_index),
                        "image": image,
                        "position": normalized_slice_position(records, actual_index),
                        "path": path,
                    }
                )

            except Exception:
                decode_failures += 1

        # Nearest-substitution can change indices; restore physical order.
        decoded_slices.sort(key=lambda item: item["actual_index"])

        if len(decoded_slices) == 0:
            series_audit.append(
                {
                    UID_COLUMN: study_uid,
                    SERIES_UID_COLUMN: series_uid,
                    "Status": "NO_DECODABLE_SELECTED_SLICES",
                    "MetadataPlane": metadata_plane,
                    "GeometryPlane": geometry_plane,
                    "GeometryConfidence": geometry_confidence,
                    "PlaneMatch": plane_match,
                    "OriginalSliceCount": len(records),
                    "RequestedSelectedSlices": len(selected_targets),
                    "DecodeFailures": decode_failures,
                }
            )
            continue

        if orientation_spec is not None:
            min_alignment = float(orientation_spec["min_alignment"])
            transform_text = (
                "+".join(
                    [
                        name
                        for name, enabled in (
                            ("transpose", orientation_spec["transpose"]),
                            ("flip0", orientation_spec["flip0"]),
                            ("flip1", orientation_spec["flip1"]),
                        )
                        if enabled
                    ]
                )
                or "none"
            )
            target_orientation = orientation_spec["target_orientation"]
        else:
            min_alignment = float("nan")
            transform_text = "missing_iop"
            target_orientation = "unknown"

        span_mm = physical_stack_span_mm(records)

        series_raw.append(
            {
                "series_uid": series_uid,
                "metadata_plane": metadata_plane,
                "plane_used": orientation_plane,
                "fluid": fluid,
                "fat_suppression": fat_suppression,
                "original_slice_count": len(records),
                "span_mm": span_mm,
                "decoded_slices": decoded_slices,
            }
        )

        series_audit.append(
            {
                UID_COLUMN: study_uid,
                SERIES_UID_COLUMN: series_uid,
                "Status": "OK",
                "MetadataPlane": metadata_plane,
                "GeometryPlane": geometry_plane,
                "GeometryConfidence": geometry_confidence,
                "PlaneMatch": plane_match,
                "PlaneUsed": orientation_plane,
                "TargetOrientation": target_orientation,
                "OrientationTransform": transform_text,
                "MinOrientationAlignment": min_alignment,
                "OriginalSliceCount": len(records),
                "RequestedSelectedSlices": len(selected_targets),
                "EncodedSliceCount": len(decoded_slices),
                "DecodeFailures": decode_failures,
                "PhysicalSpanMM": span_mm,
            }
        )

    if len(series_raw) == 0:
        raise RuntimeError(f"Study {study_uid} has no encodable MRI series.")

    # Flatten all selected slices in the study so the Curia batch can
    # span multiple series; this materially improves GPU utilization.
    flat_images: List[np.ndarray] = []
    flat_mapping: List[Tuple[int, int]] = []

    max_k = max(len(item["decoded_slices"]) for item in series_raw)
    max_k = min(max_k, MAX_SLICES_PER_SERIES)

    n_series = len(series_raw)

    features = torch.zeros(
        n_series,
        max_k,
        CURIA_HIDDEN_DIM,
        dtype=torch.float16,
    )

    slice_mask = torch.zeros(
        n_series,
        max_k,
        dtype=torch.bool,
    )

    slice_position = torch.zeros(
        n_series,
        max_k,
        dtype=torch.float16,
    )

    series_meta = torch.zeros(
        n_series,
        3,
        dtype=torch.int8,
    )

    series_cont = torch.zeros(
        n_series,
        2,
        dtype=torch.float16,
    )

    series_uids: List[str] = []

    for series_index, item in enumerate(series_raw):
        series_uids.append(item["series_uid"])

        plane_index = PLANE_TO_INDEX.get(
            item["plane_used"], PLANE_TO_INDEX[item["metadata_plane"]]
        )
        series_meta[series_index, 0] = int(plane_index)
        series_meta[series_index, 1] = int(item["fluid"])
        series_meta[series_index, 2] = int(item["fat_suppression"])

        log_slice_count = math.log1p(item["original_slice_count"]) / math.log1p(512.0)
        span_mm = item["span_mm"]
        span_scaled = (
            float(np.clip(span_mm / 300.0, 0.0, 2.0)) if np.isfinite(span_mm) else 0.0
        )

        series_cont[series_index, 0] = float(log_slice_count)
        series_cont[series_index, 1] = float(span_scaled)

        for slice_index, slice_item in enumerate(item["decoded_slices"][:max_k]):
            flat_mapping.append((series_index, slice_index))
            flat_images.append(slice_item["image"])
            slice_mask[series_index, slice_index] = True
            slice_position[series_index, slice_index] = float(slice_item["position"])

    # Process in chunks to avoid a large CPU 512x512 float tensor for a
    # whole high-series study while keeping the GPU well utilized.
    for start in range(0, len(flat_images), encoder_batch_size):
        end = min(len(flat_images), start + encoder_batch_size)
        chunk_images = flat_images[start:end]

        processed = processor(
            images=chunk_images,
            return_tensors="pt",
        )

        pixel_values = processed["pixel_values"]

        if (
            pixel_values.ndim != 4
            or pixel_values.shape[1] != 1
            or pixel_values.shape[2] != CURIA_IMAGE_SIZE
            or pixel_values.shape[3] != CURIA_IMAGE_SIZE
        ):
            raise RuntimeError(
                "Unexpected Curia processor output shape: "
                f"{tuple(pixel_values.shape)}"
            )

        pixel_values = pixel_values.to(
            device,
            dtype=torch.float32,
            non_blocking=False,
        )

        with torch.inference_mode(), autocast_context(device):
            output = model(pixel_values=pixel_values)
            cls_features = output.last_hidden_state[:, 0]

        cls_features = cls_features.float().cpu().to(torch.float16)

        for local_index, feature in enumerate(cls_features):
            series_index, slice_index = flat_mapping[start + local_index]
            features[series_index, slice_index] = feature

        del processed, pixel_values, output, cls_features

    payload = {
        "cache_version": CACHE_VERSION,
        "study_uid": study_uid,
        "features": features.contiguous(),
        "slice_mask": slice_mask.contiguous(),
        "slice_position": slice_position.contiguous(),
        "series_meta": series_meta.contiguous(),
        "series_cont": series_cont.contiguous(),
        "series_uids": series_uids,
        "series_audit": series_audit,
    }

    return payload, series_audit


# ============================================================
# 9. DUAL-T4 RESUMABLE FEATURE CACHE
# ============================================================


def cache_file_is_usable(path: Path, uid: str) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        return (
            payload.get("cache_version") == CACHE_VERSION
            and str(payload.get("study_uid")) == str(uid)
            and isinstance(payload.get("features"), torch.Tensor)
            and payload["features"].ndim == 3
            and payload["features"].shape[-1] == CURIA_HIDDEN_DIM
            and payload["features"].dtype == torch.float16
            and payload["slice_mask"].shape == payload["features"].shape[:2]
            and payload["slice_position"].shape == payload["features"].shape[:2]
            and payload["series_meta"].shape[0] == payload["features"].shape[0]
        )

    except Exception:
        return False


def build_cache_manifest(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    study_rows = []
    series_rows: List[Dict[str, Any]] = []

    for uid in sorted(train_df[UID_COLUMN].astype(str).tolist()):
        path = study_cache_path(uid)

        if not cache_file_is_usable(path, uid):
            study_rows.append(
                {
                    UID_COLUMN: uid,
                    "Cached": False,
                    "CachePath": str(path),
                    "EncodedSeries": 0,
                    "EncodedSlices": 0,
                }
            )
            continue

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        study_rows.append(
            {
                UID_COLUMN: uid,
                "Cached": True,
                "CachePath": str(path),
                "EncodedSeries": int(payload["features"].shape[0]),
                "EncodedSlices": int(payload["slice_mask"].sum().item()),
            }
        )

        series_rows.extend(payload.get("series_audit", []))

    manifest_df = pd.DataFrame(study_rows)
    audit_df = pd.DataFrame(series_rows)

    manifest_df.to_csv(
        RESULT_ROOT / "feature_cache_manifest.csv",
        index=False,
    )

    audit_df.to_csv(
        RESULT_ROOT / "series_cache_audit.csv",
        index=False,
    )

    return manifest_df, audit_df


def _cache_worker(
    worker_id: int,
    device_id: int,
    study_uids: List[str],
    grouped_series: Dict[str, pd.DataFrame],
    curia_root: Path,
    encoder_batch_size: int,
    shared_progress: Dict[str, Any],
    progress_lock: threading.Lock,
    total_requested: int,
    started: float,
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}") if GPU_COUNT > 0 else torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device_id)

    processor, model = load_curia_processor_and_model(curia_root, device)

    local_failures: List[Dict[str, Any]] = []
    encoded = 0
    skipped = 0

    log(
        f"Worker {worker_id}: device={device}, studies={len(study_uids)}, "
        f"batch={encoder_batch_size}"
    )

    for local_index, uid in enumerate(study_uids, start=1):
        path = study_cache_path(uid)

        if cache_file_is_usable(path, uid):
            skipped += 1

            with progress_lock:
                shared_progress["done"] += 1
                done_now = int(shared_progress["done"])

            if done_now <= 10 or done_now % 100 == 0 or done_now == total_requested:
                elapsed = time.time() - started
                rate = done_now / max(elapsed, 1e-6)
                eta = (total_requested - done_now) / max(rate, 1e-6)
                log(
                    f"Cache {done_now:4d}/{total_requested} "
                    f"({100.0 * done_now / total_requested:5.1f}%) "
                    f"elapsed={elapsed_string(elapsed)} ETA={elapsed_string(eta)} "
                    f"worker={worker_id} last=already_cached"
                )

            continue

        try:
            study_series = grouped_series.get(uid)

            if study_series is None or len(study_series) == 0:
                raise RuntimeError("No train_series.csv rows for study.")

            payload, _ = encode_study(
                uid,
                study_series,
                processor,
                model,
                device,
                encoder_batch_size,
            )

            atomic_torch_save(payload, path)
            encoded += 1

        except Exception as exc:
            local_failures.append(
                {
                    UID_COLUMN: uid,
                    "Worker": worker_id,
                    "Device": str(device),
                    "Error": repr(exc),
                }
            )

        with progress_lock:
            shared_progress["done"] += 1
            done_now = int(shared_progress["done"])

        if done_now <= 10 or done_now % 100 == 0 or done_now == total_requested:
            elapsed = time.time() - started
            rate = done_now / max(elapsed, 1e-6)
            eta = (total_requested - done_now) / max(rate, 1e-6)
            log(
                f"Cache {done_now:4d}/{total_requested} "
                f"({100.0 * done_now / total_requested:5.1f}%) "
                f"elapsed={elapsed_string(elapsed)} ETA={elapsed_string(eta)} "
                f"worker={worker_id} encoded={encoded} failures={len(local_failures)}"
            )

        if local_index % 50 == 0:
            gc.collect()

    del model, processor
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "worker_id": worker_id,
        "device": str(device),
        "encoded": encoded,
        "skipped": skipped,
        "failures": local_failures,
    }


def build_feature_cache(
    train_df: pd.DataFrame,
    series_df: pd.DataFrame,
) -> Dict[str, Any]:
    if GPU_COUNT <= 0:
        raise RuntimeError(
            "W4.0 Curia feature caching requires CUDA for a practical runtime."
        )

    curia_root = discover_curia2_root()
    model_sha, curia_config = validate_curia_identity(curia_root)
    prepare_cache_identity(model_sha)

    encoder_batch_size = benchmark_curia_batch_size(curia_root)

    all_uids = sorted(train_df[UID_COLUMN].astype(str).tolist())
    grouped_series = {
        str(uid): group.copy() for uid, group in series_df.groupby(UID_COLUMN)
    }

    already_cached = [
        uid for uid in all_uids if cache_file_is_usable(study_cache_path(uid), uid)
    ]

    need = [uid for uid in all_uids if uid not in set(already_cached)]

    log("\n" + "=" * 88)
    log("W4.0 CURIA-2 FEATURE CACHE")
    log("=" * 88)
    log(f"Curia root            : {curia_root}")
    log(f"Curia SHA256          : {model_sha}")
    log(f"CUDA GPU count        : {GPU_COUNT}")
    log(f"Cache GPU IDs         : {CACHE_GPU_IDS}")
    log(f"Curia FP16 batch      : {encoder_batch_size}")
    log(f"Train studies         : {len(all_uids)}")
    log(f"Already cached        : {len(already_cached)}")
    log(f"Need encoding         : {len(need)}")
    log(f"Max slices / series   : {MAX_SLICES_PER_SERIES}")
    log("Cache representation  : frozen Curia-2 CLS token, FP16")

    if len(need) == 0:
        manifest_df, audit_df = build_cache_manifest(train_df)
        return summarize_cache(
            train_df,
            manifest_df,
            audit_df,
            model_sha,
            curia_config,
            encoder_batch_size,
            runtime_seconds=0.0,
        )

    worker_gpu_ids = CACHE_GPU_IDS if CACHE_GPU_IDS else [0]
    n_workers = len(worker_gpu_ids)

    shards = [need[i::n_workers] for i in range(n_workers)]
    shared_progress = {"done": 0}
    progress_lock = threading.Lock()
    started = time.time()

    worker_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        for worker_id, device_id in enumerate(worker_gpu_ids):
            futures.append(
                executor.submit(
                    _cache_worker,
                    worker_id,
                    device_id,
                    shards[worker_id],
                    grouped_series,
                    curia_root,
                    encoder_batch_size,
                    shared_progress,
                    progress_lock,
                    len(need),
                    started,
                )
            )

        for future in as_completed(futures):
            worker_results.append(future.result())

    failures = []
    for result in worker_results:
        failures.extend(result["failures"])

    failure_df = pd.DataFrame(failures)
    failure_df.to_csv(
        RESULT_ROOT / "feature_cache_failures.csv",
        index=False,
    )

    manifest_df, audit_df = build_cache_manifest(train_df)

    summary = summarize_cache(
        train_df,
        manifest_df,
        audit_df,
        model_sha,
        curia_config,
        encoder_batch_size,
        runtime_seconds=time.time() - started,
    )

    if not bool(summary["cache_complete"]):
        raise RuntimeError(
            "Curia feature cache is incomplete. Inspect "
            f"{RESULT_ROOT / 'feature_cache_failures.csv'} and rerun cache."
        )

    return summary


def summarize_cache(
    train_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    model_sha: str,
    curia_config: Dict[str, Any],
    encoder_batch_size: int,
    runtime_seconds: float,
) -> Dict[str, Any]:
    cached_n = int(manifest_df["Cached"].sum()) if len(manifest_df) else 0
    encoded_series = int(manifest_df["EncodedSeries"].sum()) if len(manifest_df) else 0
    encoded_slices = int(manifest_df["EncodedSlices"].sum()) if len(manifest_df) else 0

    ok_audit = (
        audit_df[audit_df["Status"] == "OK"].copy()
        if len(audit_df) and "Status" in audit_df.columns
        else pd.DataFrame()
    )

    if len(ok_audit):
        orientation_values = pd.to_numeric(
            ok_audit["MinOrientationAlignment"],
            errors="coerce",
        )

        min_orientation = (
            float(orientation_values.min())
            if orientation_values.notna().any()
            else float("nan")
        )
        below_threshold = int((orientation_values < ORIENTATION_MIN_ALIGNMENT).sum())
        plane_matches = int(ok_audit["PlaneMatch"].fillna(False).astype(bool).sum())
    else:
        min_orientation = float("nan")
        below_threshold = 0
        plane_matches = 0

    summary = {
        "cache_version": CACHE_VERSION,
        "cache_complete": cached_n == len(train_df),
        "train_studies": int(len(train_df)),
        "cached_studies": cached_n,
        "encoded_series": encoded_series,
        "encoded_slices": encoded_slices,
        "mean_encoded_series_per_study": (encoded_series / max(cached_n, 1)),
        "mean_encoded_slices_per_study": (encoded_slices / max(cached_n, 1)),
        "curia_model_sha256": model_sha,
        "curia_config": curia_config,
        "encoder_batch_size": int(encoder_batch_size),
        "orientation_min_alignment_seen": min_orientation,
        "series_below_orientation_threshold": below_threshold,
        "series_plane_metadata_geometry_matches": plane_matches,
        "series_audit_rows": int(len(ok_audit)),
        "runtime_seconds_this_call": float(runtime_seconds),
    }

    (RESULT_ROOT / "feature_cache_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log("\nFeature-cache summary:")
    log(json.dumps(summary, indent=2, allow_nan=True))

    return summary


# ============================================================
# 10. W2.3 DISCOVERY + FOLD-SAFE PSEUDO TARGETS
# ============================================================


def looks_like_w23_root(path: Path) -> bool:
    return (
        (path / "results" / "00_outer_fold_assignments.csv").exists()
        and (path / "folds" / "fold_1" / "soft_probabilities_wide.csv").exists()
        and (path / "folds" / "fold_1" / "soft_label_availability_wide.csv").exists()
        and (path / "folds" / "fold_1" / "candidate_selection_scores_wide.csv").exists()
    )


def discover_w23_root() -> Path:
    candidates: List[Path] = []

    if EXPLICIT_W23_ROOT:
        explicit = Path(EXPLICIT_W23_ROOT)
        candidates.extend([explicit, explicit / "rsna_w2_3"])

    candidates.extend(
        [
            Path("/kaggle/working/rsna_w2_3"),
            Path("/kaggle/working/w2_3"),
        ]
    )

    for root in shallow_kaggle_dirs(max_depth=4):
        candidates.extend([root, root / "rsna_w2_3"])

    for path in dict.fromkeys(candidates):
        if looks_like_w23_root(path):
            return path

    raise FileNotFoundError(
        "W2.3 fold-safe outputs not found. Set W40_W23_ROOT to the directory "
        "containing results/00_outer_fold_assignments.csv and folds/fold_*/."
    )


def verify_w23_outer_folds(
    w23_root: Path,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
) -> None:
    path = w23_root / "results" / "00_outer_fold_assignments.csv"
    actual = pd.read_csv(path)
    actual[UID_COLUMN] = actual[UID_COLUMN].astype(str)
    actual = (
        actual[[UID_COLUMN, "OuterFold"]].sort_values(UID_COLUMN).reset_index(drop=True)
    )

    expected = (
        pd.DataFrame(
            {
                UID_COLUMN: gold_df[UID_COLUMN].astype(str),
                "OuterFold": fold_zero + 1,
            }
        )
        .sort_values(UID_COLUMN)
        .reset_index(drop=True)
    )

    if not actual.equals(expected):
        raise RuntimeError("W2.3 outer folds do not match the exact W3/V4 folds.")

    sha = fold_assignment_sha256(actual)
    if sha != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"W2.3 fold checksum mismatch: {sha}")


def read_w23_wide(w23_root: Path, fold: int, filename: str) -> pd.DataFrame:
    path = w23_root / "folds" / f"fold_{fold}" / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df[UID_COLUMN] = df[UID_COLUMN].astype(str)

    if df[UID_COLUMN].duplicated().any():
        raise RuntimeError(f"Duplicate UIDs in {path}")

    missing = [label for label in LABEL_COLUMNS if label not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing labels: {missing}")

    return df


def load_fold_pseudo_targets(
    w23_root: Path,
    fold: int,
    gold_uid_set: set,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    probability_df = read_w23_wide(
        w23_root,
        fold,
        "soft_probabilities_wide.csv",
    )

    availability_df = read_w23_wide(
        w23_root,
        fold,
        "soft_label_availability_wide.csv",
    )

    selection_df = read_w23_wide(
        w23_root,
        fold,
        "candidate_selection_scores_wide.csv",
    )

    if not (
        probability_df[UID_COLUMN].tolist()
        == availability_df[UID_COLUMN].tolist()
        == selection_df[UID_COLUMN].tolist()
    ):
        raise RuntimeError(f"W2.3 fold {fold} wide-file UID mismatch.")

    pseudo_uids = probability_df[UID_COLUMN].astype(str)
    overlap = set(pseudo_uids) & gold_uid_set
    if overlap:
        raise RuntimeError(
            "W2.3 pseudo pool unexpectedly contains gold studies. "
            f"Examples: {sorted(overlap)[:3]}"
        )

    probability = probability_df[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    selection = (
        selection_df[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    availability = pd.DataFrame(index=availability_df.index)
    for label in LABEL_COLUMNS:
        availability[label] = coerce_bool_series(availability_df[label])

    mask = (
        availability[LABEL_COLUMNS]
        & (selection[LABEL_COLUMNS] >= SELECTION_THRESHOLD)
        & probability[LABEL_COLUMNS].notna()
    )

    high_mask_path = (
        w23_root / "folds" / f"fold_{fold}" / "high_selection_candidate_mask_wide.csv"
    )

    if high_mask_path.exists() and abs(SELECTION_THRESHOLD - 0.50) < 1e-12:
        high_df = pd.read_csv(high_mask_path)
        high_df[UID_COLUMN] = high_df[UID_COLUMN].astype(str)

        if high_df[UID_COLUMN].tolist() != pseudo_uids.tolist():
            raise RuntimeError(f"W2.3 fold {fold} high-mask UID mismatch.")

        high_bool = pd.DataFrame(index=high_df.index)
        for label in LABEL_COLUMNS:
            high_bool[label] = coerce_bool_series(high_df[label])

        if not np.array_equal(
            high_bool[LABEL_COLUMNS].values.astype(bool),
            mask[LABEL_COLUMNS].values.astype(bool),
        ):
            raise RuntimeError(
                f"Reconstructed W2.3 fold {fold} high-selection mask differs "
                "from high_selection_candidate_mask_wide.csv."
            )

    output = pd.DataFrame({UID_COLUMN: pseudo_uids})

    for label in LABEL_COLUMNS:
        output[f"{label}__target"] = probability[label].astype(np.float32)
        output[f"{label}__mask"] = mask[label].astype(bool)

    selected_study_mask = mask[LABEL_COLUMNS].any(axis=1)
    selected = output[selected_study_mask].reset_index(drop=True)

    audit_rows = []
    for label in LABEL_COLUMNS:
        label_mask = mask[label].values.astype(bool)
        audit_rows.append(
            {
                "OuterFold": fold,
                "Label": label,
                "SelectedCellN": int(label_mask.sum()),
                "SelectedStudyFraction": float(label_mask.mean()),
                "MeanSoftTarget": (
                    float(probability.loc[label_mask, label].mean())
                    if label_mask.any()
                    else float("nan")
                ),
            }
        )

    return selected, pd.DataFrame(audit_rows)


# ============================================================
# 11. LOAD COMPACT CURIA CACHE INTO CPU RAM
# ============================================================


class CuriaFeatureStore:
    def __init__(self, train_df: pd.DataFrame):
        self.uids = sorted(train_df[UID_COLUMN].astype(str).tolist())
        self.uid_to_index = {uid: i for i, uid in enumerate(self.uids)}
        self.records: List[Dict[str, Any]] = []

        started = time.time()
        total_feature_bytes = 0

        log("\nLoading compact Curia feature cache into CPU RAM...")

        for n, uid in enumerate(self.uids, start=1):
            path = study_cache_path(uid)

            if not cache_file_is_usable(path, uid):
                raise RuntimeError(
                    f"Missing/incompatible Curia cache for study {uid}. "
                    "Run run_w40_curia('cache') first."
                )

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            record = {
                "uid": uid,
                "features": payload["features"].contiguous(),
                "slice_mask": payload["slice_mask"].contiguous(),
                "slice_position": payload["slice_position"].contiguous(),
                "series_meta": payload["series_meta"].contiguous(),
                "series_cont": payload["series_cont"].contiguous(),
            }

            total_feature_bytes += (
                record["features"].numel() * record["features"].element_size()
            )
            self.records.append(record)

            if n <= 5 or n % 500 == 0 or n == len(self.uids):
                log(
                    f"  loaded {n:4d}/{len(self.uids)} "
                    f"elapsed={elapsed_string(time.time() - started)}"
                )

        log(f"Feature payload size : {total_feature_bytes / (1024 ** 3):.3f} GB FP16")

    def indices_for_uids(self, uids: Iterable[str]) -> np.ndarray:
        return np.asarray(
            [self.uid_to_index[str(uid)] for uid in uids],
            dtype=np.int64,
        )

    def make_batch(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        records = [self.records[int(index)] for index in indices]

        batch_size = len(records)
        max_series = max(record["features"].shape[0] for record in records)
        max_slices = max(record["features"].shape[1] for record in records)

        features = torch.zeros(
            batch_size,
            max_series,
            max_slices,
            CURIA_HIDDEN_DIM,
            dtype=torch.float16,
        )

        slice_mask = torch.zeros(
            batch_size,
            max_series,
            max_slices,
            dtype=torch.bool,
        )

        slice_position = torch.zeros(
            batch_size,
            max_series,
            max_slices,
            dtype=torch.float32,
        )

        series_meta = torch.zeros(
            batch_size,
            max_series,
            3,
            dtype=torch.long,
        )

        series_cont = torch.zeros(
            batch_size,
            max_series,
            2,
            dtype=torch.float32,
        )

        series_mask = torch.zeros(
            batch_size,
            max_series,
            dtype=torch.bool,
        )

        for batch_index, record in enumerate(records):
            n_series, n_slices, _ = record["features"].shape

            features[batch_index, :n_series, :n_slices] = record["features"]
            slice_mask[batch_index, :n_series, :n_slices] = record["slice_mask"]
            slice_position[batch_index, :n_series, :n_slices] = record[
                "slice_position"
            ].float()
            series_meta[batch_index, :n_series] = record["series_meta"].long()
            series_cont[batch_index, :n_series] = record["series_cont"].float()
            series_mask[batch_index, :n_series] = True

        return {
            "features": features.to(device),
            "slice_mask": slice_mask.to(device),
            "slice_position": slice_position.to(device),
            "series_meta": series_meta.to(device),
            "series_cont": series_cont.to(device),
            "series_mask": series_mask.to(device),
        }


# ============================================================
# 12. HIERARCHICAL DIAGNOSIS-SPECIFIC HEAD
# ============================================================


def _drop_mask_tokens(mask: torch.Tensor, drop_probability: float) -> torch.Tensor:
    """
    Drop valid tokens but guarantee at least one valid token for every
    originally non-empty sequence.
    """

    if drop_probability <= 0.0:
        return mask

    output = mask & (torch.rand_like(mask.float()) >= drop_probability)

    original_nonempty = mask.any(dim=-1)
    output_nonempty = output.any(dim=-1)
    need_restore = original_nonempty & (~output_nonempty)

    if need_restore.any():
        first_valid = mask.float().argmax(dim=-1)
        coordinates = torch.nonzero(need_restore, as_tuple=False)

        for coordinate in coordinates:
            prefix = tuple(int(x) for x in coordinate.tolist())
            token_index = int(first_valid[prefix].item())
            output[prefix + (token_index,)] = True

    return output


class CuriaHierarchicalDiagnosisHead(nn.Module):
    def __init__(self):
        super().__init__()

        if HEAD_HIDDEN_DIM % HEAD_NUM_HEADS != 0:
            raise ValueError("HEAD_HIDDEN_DIM must be divisible by HEAD_NUM_HEADS.")

        self.input_norm = nn.LayerNorm(CURIA_HIDDEN_DIM)
        self.input_projection = nn.Linear(CURIA_HIDDEN_DIM, HEAD_HIDDEN_DIM)

        # pos + sin/cos at 1,2,4,8*pi => 9 scalar channels.
        self.position_projection = nn.Sequential(
            nn.Linear(9, HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN_DIM, HEAD_HIDDEN_DIM),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HEAD_HIDDEN_DIM,
            nhead=HEAD_NUM_HEADS,
            dim_feedforward=4 * HEAD_HIDDEN_DIM,
            dropout=HEAD_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.slice_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=HEAD_TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(HEAD_HIDDEN_DIM),
        )

        self.slice_queries = nn.Parameter(
            torch.randn(NUM_LABELS, HEAD_HIDDEN_DIM) * 0.02
        )

        self.plane_embedding = nn.Embedding(3, 32)
        self.fluid_embedding = nn.Embedding(2, 16)
        self.fs_embedding = nn.Embedding(2, 16)

        self.metadata_projection = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 2, HEAD_HIDDEN_DIM),
            nn.LayerNorm(HEAD_HIDDEN_DIM),
            nn.GELU(),
        )

        self.series_norm = nn.LayerNorm(HEAD_HIDDEN_DIM)
        self.series_queries = nn.Parameter(
            torch.randn(NUM_LABELS, HEAD_HIDDEN_DIM) * 0.02
        )

        self.final_norm = nn.LayerNorm(HEAD_HIDDEN_DIM)
        self.dropout = nn.Dropout(HEAD_DROPOUT)

        self.classifier_weight = nn.Parameter(
            torch.randn(NUM_LABELS, HEAD_HIDDEN_DIM) * 0.02
        )
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    @staticmethod
    def _fourier_position(position: torch.Tensor) -> torch.Tensor:
        position = position.clamp(-1.0, 1.0)
        frequencies = torch.tensor(
            [1.0, 2.0, 4.0, 8.0],
            device=position.device,
            dtype=position.dtype,
        )

        angles = math.pi * position.unsqueeze(-1) * frequencies

        return torch.cat(
            [
                position.unsqueeze(-1),
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )

    def forward(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        slice_position: torch.Tensor,
        series_meta: torch.Tensor,
        series_cont: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Feature-level stochastic regularization is applied only to the
        # frozen representation masks; Curia embeddings themselves remain
        # unchanged.
        if self.training and SERIES_DROPOUT > 0:
            series_mask_effective = _drop_mask_tokens(
                series_mask,
                SERIES_DROPOUT,
            )
        else:
            series_mask_effective = series_mask

        slice_mask_effective = slice_mask & series_mask_effective.unsqueeze(-1)

        if self.training and SLICE_DROPOUT > 0:
            # Flatten B,S so the helper guarantees one slice per retained series.
            b, s, k = slice_mask_effective.shape
            flat = slice_mask_effective.reshape(b * s, k)
            flat = _drop_mask_tokens(flat, SLICE_DROPOUT)
            slice_mask_effective = flat.reshape(b, s, k)

        batch_size, max_series, max_slices, _ = features.shape

        # Only process retained real series.  This avoids all-padding
        # Transformer sequences and saves compute on padded series.
        real_series_flat = series_mask_effective.reshape(-1)

        flat_features = features.reshape(
            batch_size * max_series,
            max_slices,
            CURIA_HIDDEN_DIM,
        )[real_series_flat]

        flat_slice_mask = slice_mask_effective.reshape(
            batch_size * max_series,
            max_slices,
        )[real_series_flat]

        flat_position = slice_position.reshape(
            batch_size * max_series,
            max_slices,
        )[real_series_flat]

        hidden = self.input_projection(self.input_norm(flat_features.float()))
        hidden = hidden + self.position_projection(
            self._fourier_position(flat_position)
        )

        hidden = self.slice_transformer(
            hidden,
            src_key_padding_mask=(~flat_slice_mask),
        )

        # Label-specific slice attention: [R,L,K].
        slice_scores = torch.einsum(
            "rkh,lh->rlk",
            hidden,
            self.slice_queries,
        ) / math.sqrt(HEAD_HIDDEN_DIM)

        slice_scores = slice_scores.masked_fill(
            ~flat_slice_mask.unsqueeze(1),
            -1e4,
        )

        slice_attention = torch.softmax(slice_scores, dim=-1)

        pooled_series = torch.einsum(
            "rlk,rkh->rlh",
            slice_attention,
            hidden,
        )

        # Metadata for the same retained real-series rows.
        flat_meta = series_meta.reshape(batch_size * max_series, 3)[real_series_flat]
        flat_cont = series_cont.reshape(batch_size * max_series, 2)[real_series_flat]

        metadata_hidden = self.metadata_projection(
            torch.cat(
                [
                    self.plane_embedding(flat_meta[:, 0]),
                    self.fluid_embedding(flat_meta[:, 1]),
                    self.fs_embedding(flat_meta[:, 2]),
                    flat_cont,
                ],
                dim=-1,
            )
        )

        pooled_series = self.series_norm(pooled_series + metadata_hidden.unsqueeze(1))

        # Scatter retained series back into [B,S,L,H].
        series_representation = torch.zeros(
            batch_size * max_series,
            NUM_LABELS,
            HEAD_HIDDEN_DIM,
            device=features.device,
            dtype=pooled_series.dtype,
        )

        series_representation[real_series_flat] = pooled_series
        series_representation = series_representation.reshape(
            batch_size,
            max_series,
            NUM_LABELS,
            HEAD_HIDDEN_DIM,
        )

        # Label-specific series attention: [B,S,L].
        series_scores = torch.einsum(
            "bslh,lh->bsl",
            series_representation,
            self.series_queries,
        ) / math.sqrt(HEAD_HIDDEN_DIM)

        series_scores = series_scores.masked_fill(
            ~series_mask_effective.unsqueeze(-1),
            -1e4,
        )

        series_attention = torch.softmax(series_scores, dim=1)

        study_representation = torch.einsum(
            "bsl,bslh->blh",
            series_attention,
            series_representation,
        )

        study_representation = self.final_norm(study_representation)
        study_representation = self.dropout(study_representation)

        logits = (study_representation * self.classifier_weight.unsqueeze(0)).sum(
            dim=-1
        ) + self.classifier_bias.unsqueeze(0)

        return {
            "logits": logits,
            "slice_attention": slice_attention,
            "series_attention": series_attention,
        }


# ============================================================
# 13. MACRO-AWARE LOSSES
# ============================================================


def compute_gold_pos_weight(train_labels: np.ndarray) -> torch.Tensor:
    positives = train_labels.sum(axis=0).astype(np.float64)
    negatives = len(train_labels) - positives

    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 5.0).astype(np.float32)

    return torch.tensor(weights, device=TRAIN_DEVICE, dtype=torch.float32)


def macro_gold_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    per_cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
        pos_weight=pos_weight,
    )

    # Every gold study has every label. Equal label contribution first.
    per_label = per_cell.mean(dim=0)
    return per_label.mean()


def macro_masked_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
    )

    mask_float = mask.float()
    counts = mask_float.sum(dim=0)
    active_labels = counts > 0

    if not active_labels.any():
        raise RuntimeError("Pseudo batch contains no active target label cells.")

    per_label = (per_cell * mask_float).sum(dim=0) / counts.clamp_min(1.0)

    return per_label[active_labels].mean(), counts


# ============================================================
# 14. METRICS
# ============================================================


def metric_tables(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    aucs = []
    aps = []

    for label_index, label in enumerate(LABEL_COLUMNS):
        auc = safe_auc(y_true[:, label_index], y_prob[:, label_index])
        ap = safe_ap(y_true[:, label_index], y_prob[:, label_index])

        rows.append(
            {
                "Label": label,
                "PositiveCount": int(y_true[:, label_index].sum()),
                "AUROC": auc,
                "AveragePrecision": ap,
            }
        )

        if np.isfinite(auc):
            aucs.append(auc)

        if np.isfinite(ap):
            aps.append(ap)

    y_pred = (y_prob >= 0.5).astype(np.int64)

    summary = {
        "macro_AUROC": float(np.mean(aucs)),
        "macro_AP": float(np.mean(aps)),
        "macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return pd.DataFrame(rows), summary


# ============================================================
# 15. TRAINING SUPERVISION OBJECTS
# ============================================================


def build_gold_arrays(
    store: CuriaFeatureStore,
    gold_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    gold_indices = store.indices_for_uids(gold_df[UID_COLUMN].astype(str).tolist())
    gold_targets = gold_df[LABEL_COLUMNS].values.astype(np.float32)
    return gold_indices, gold_targets


def pseudo_dataframe_to_arrays(
    store: CuriaFeatureStore,
    pseudo_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = store.indices_for_uids(pseudo_df[UID_COLUMN].astype(str).tolist())

    targets = np.stack(
        [
            pd.to_numeric(
                pseudo_df[f"{label}__target"],
                errors="coerce",
            )
            .fillna(0.5)
            .values.astype(np.float32)
            for label in LABEL_COLUMNS
        ],
        axis=1,
    )

    masks = np.stack(
        [pseudo_df[f"{label}__mask"].values.astype(bool) for label in LABEL_COLUMNS],
        axis=1,
    )

    return indices, targets, masks


def head_config_payload() -> Dict[str, Any]:
    return {
        "experiment": "W4.0",
        "backbone": CURIA_REPO_ID,
        "backbone_frozen": True,
        "cache_version": CACHE_VERSION,
        "embedding": "CLS last_hidden_state[:,0]",
        "max_slices_per_series": MAX_SLICES_PER_SERIES,
        "all_series": True,
        "hidden_dim": HEAD_HIDDEN_DIM,
        "num_heads": HEAD_NUM_HEADS,
        "transformer_layers": HEAD_TRANSFORMER_LAYERS,
        "head_dropout": HEAD_DROPOUT,
        "slice_dropout": SLICE_DROPOUT,
        "series_dropout": SERIES_DROPOUT,
        "epochs": HEAD_EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "gold_batch_size": GOLD_BATCH_SIZE,
        "pseudo_batch_size": PSEUDO_BATCH_SIZE,
        "head_max_lr": HEAD_MAX_LR,
        "head_weight_decay": HEAD_WEIGHT_DECAY,
        "selection_threshold": SELECTION_THRESHOLD,
        "gold_authority": GOLD_AUTHORITY,
        "pseudo_authority": PSEUDO_AUTHORITY,
        "fold_sha256": EXPECTED_FOLD_SHA256,
    }


def training_config_hash() -> str:
    payload = json.dumps(
        head_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================
# 16. BATCH PREDICTION
# ============================================================


@torch.no_grad()
def predict_indices(
    model: nn.Module,
    store: CuriaFeatureStore,
    indices: np.ndarray,
    batch_size: int = VALIDATION_BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    probabilities: List[np.ndarray] = []

    for start in range(0, len(indices), batch_size):
        part = indices[start : start + batch_size]
        batch = store.make_batch(part, TRAIN_DEVICE)

        with autocast_context(TRAIN_DEVICE):
            output = model(**batch)

        prob = torch.sigmoid(output["logits"].float()).cpu().numpy().astype(np.float32)
        probabilities.append(prob)

        del batch, output, prob

    if not probabilities:
        return np.empty((0, NUM_LABELS), dtype=np.float32)

    return np.concatenate(probabilities, axis=0)


# ============================================================
# 17. TRAIN ONE FOLD
# ============================================================


def _sample_rows(
    rng: np.random.Generator,
    values: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(values) == 0:
        raise RuntimeError("Cannot sample from an empty training pool.")

    positions = rng.integers(
        low=0,
        high=len(values),
        size=batch_size,
    )

    return values[positions]


def train_one_fold(
    variant: str,
    fold: int,
    store: CuriaFeatureStore,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
    w23_root: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    config_hash = training_config_hash()

    variant_result_root = RESULT_ROOT / variant
    variant_checkpoint_root = CHECKPOINT_ROOT / variant
    variant_result_root.mkdir(parents=True, exist_ok=True)
    variant_checkpoint_root.mkdir(parents=True, exist_ok=True)

    history_path = variant_result_root / f"fold_{fold}_history.csv"
    prediction_path = variant_result_root / f"fold_{fold}_epoch_predictions.csv"
    checkpoint_path = variant_checkpoint_root / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"
    pseudo_audit_path = variant_result_root / f"fold_{fold}_pseudo_audit.csv"

    # Fold-level resume.  We only skip a fold if all three artifacts are
    # complete and the checkpoint carries the exact current config hash.
    if history_path.exists() and prediction_path.exists() and checkpoint_path.exists():
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            history_df = pd.read_csv(history_path)
            prediction_df = pd.read_csv(prediction_path)

            if (
                checkpoint.get("config_hash") == config_hash
                and int(checkpoint.get("fold", -1)) == fold
                and int(checkpoint.get("epoch", -1)) == HEAD_EPOCHS
                and len(history_df) == HEAD_EPOCHS
                and set(prediction_df["Epoch"].astype(int).unique())
                == set(range(1, HEAD_EPOCHS + 1))
            ):
                pseudo_audit = (
                    pd.read_csv(pseudo_audit_path)
                    if pseudo_audit_path.exists()
                    else None
                )

                log(
                    f"{variant} fold {fold}: complete checkpoint found -> skipping retrain."
                )
                return history_df, prediction_df, pseudo_audit

        except Exception:
            pass

    fold_index_zero = fold - 1
    gold_train_rows = np.where(fold_zero != fold_index_zero)[0]
    gold_val_rows = np.where(fold_zero == fold_index_zero)[0]

    gold_train_uids = gold_df.iloc[gold_train_rows][UID_COLUMN].astype(str).tolist()
    gold_val_uids = gold_df.iloc[gold_val_rows][UID_COLUMN].astype(str).tolist()

    gold_train_indices = store.indices_for_uids(gold_train_uids)
    val_indices = store.indices_for_uids(gold_val_uids)

    gold_train_targets = gold_df.iloc[gold_train_rows][LABEL_COLUMNS].values.astype(
        np.float32
    )
    val_targets = gold_df.iloc[gold_val_rows][LABEL_COLUMNS].values.astype(np.int64)

    # Map feature-store index -> gold target row for fast sampled lookup.
    gold_target_by_store_index = {
        int(store_index): gold_train_targets[row_index]
        for row_index, store_index in enumerate(gold_train_indices)
    }

    gold_pos_weight = compute_gold_pos_weight(gold_train_targets)

    pseudo_indices = None
    pseudo_targets = None
    pseudo_masks = None
    pseudo_audit = None

    if variant == "fold_safe_weak":
        if w23_root is None:
            raise RuntimeError("fold_safe_weak requires W2.3 outputs.")

        gold_uid_set = set(gold_df[UID_COLUMN].astype(str))
        pseudo_df, pseudo_audit = load_fold_pseudo_targets(
            w23_root,
            fold,
            gold_uid_set,
        )

        pseudo_indices, pseudo_targets, pseudo_masks = pseudo_dataframe_to_arrays(
            store,
            pseudo_df,
        )

        if len(pseudo_indices) == 0:
            raise RuntimeError(f"W2.3 fold {fold} produced no selected pseudo studies.")

        pseudo_audit.to_csv(pseudo_audit_path, index=False)

    elif variant != "gold_only":
        raise ValueError(f"Unknown variant: {variant}")

    seed = 40_000 + fold
    seed_everything(seed)

    model = CuriaHierarchicalDiagnosisHead().to(TRAIN_DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HEAD_MAX_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )

    total_steps = HEAD_EPOCHS * STEPS_PER_EPOCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HEAD_MAX_LR,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
    )

    scaler = make_grad_scaler()

    # Gold sampling is identical across corresponding gold/weak folds.
    gold_rng = np.random.default_rng(seed + 1_000)
    pseudo_rng = np.random.default_rng(seed + 2_000)

    history_rows: List[Dict[str, Any]] = []
    epoch_prediction_rows: List[Dict[str, Any]] = []

    log("\n" + "-" * 88)
    log(f"{variant} | fold {fold}/{NUM_FOLDS}")
    log("-" * 88)
    log(f"Gold train studies     : {len(gold_train_indices)}")
    log(f"Gold validation studies: {len(val_indices)}")

    if variant == "fold_safe_weak":
        log(f"Pseudo selected studies: {len(pseudo_indices)}")
        log(f"Pseudo selected cells  : {int(pseudo_masks.sum())}")
        log(
            f"Loss authority         : gold={GOLD_AUTHORITY:g}, pseudo={PSEUDO_AUTHORITY:g}"
        )

    started = time.time()

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()

        epoch_gold_loss = 0.0
        epoch_pseudo_loss = 0.0
        epoch_total_loss = 0.0
        epoch_pseudo_label_counts = np.zeros(NUM_LABELS, dtype=np.int64)

        for step in range(STEPS_PER_EPOCH):
            sampled_gold_indices = _sample_rows(
                gold_rng,
                gold_train_indices,
                GOLD_BATCH_SIZE,
            )

            gold_batch = store.make_batch(sampled_gold_indices, TRAIN_DEVICE)
            gold_target_np = np.stack(
                [
                    gold_target_by_store_index[int(index)]
                    for index in sampled_gold_indices
                ],
                axis=0,
            )

            gold_target_tensor = torch.tensor(
                gold_target_np,
                device=TRAIN_DEVICE,
                dtype=torch.float32,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(TRAIN_DEVICE):
                gold_output = model(**gold_batch)
                gold_loss = macro_gold_bce(
                    gold_output["logits"],
                    gold_target_tensor,
                    gold_pos_weight,
                )

                if variant == "fold_safe_weak":
                    sampled_pseudo_positions = pseudo_rng.integers(
                        low=0,
                        high=len(pseudo_indices),
                        size=PSEUDO_BATCH_SIZE,
                    )

                    sampled_pseudo_indices = pseudo_indices[sampled_pseudo_positions]
                    pseudo_batch = store.make_batch(
                        sampled_pseudo_indices, TRAIN_DEVICE
                    )

                    pseudo_target_tensor = torch.tensor(
                        pseudo_targets[sampled_pseudo_positions],
                        device=TRAIN_DEVICE,
                        dtype=torch.float32,
                    )

                    pseudo_mask_tensor = torch.tensor(
                        pseudo_masks[sampled_pseudo_positions],
                        device=TRAIN_DEVICE,
                        dtype=torch.bool,
                    )

                    pseudo_output = model(**pseudo_batch)
                    pseudo_loss, pseudo_counts = macro_masked_soft_bce(
                        pseudo_output["logits"],
                        pseudo_target_tensor,
                        pseudo_mask_tensor,
                    )

                    total_loss = (
                        GOLD_AUTHORITY * gold_loss + PSEUDO_AUTHORITY * pseudo_loss
                    ) / (GOLD_AUTHORITY + PSEUDO_AUTHORITY)

                else:
                    pseudo_loss = torch.zeros((), device=TRAIN_DEVICE)
                    pseudo_counts = torch.zeros(NUM_LABELS, device=TRAIN_DEVICE)
                    total_loss = gold_loss

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            scheduler.step()

            epoch_gold_loss += float(gold_loss.detach().cpu().item())
            epoch_pseudo_loss += float(pseudo_loss.detach().cpu().item())
            epoch_total_loss += float(total_loss.detach().cpu().item())
            epoch_pseudo_label_counts += (
                pseudo_counts.detach().cpu().numpy().astype(np.int64)
            )

            del gold_batch, gold_output, gold_target_tensor

            if variant == "fold_safe_weak":
                del (
                    pseudo_batch,
                    pseudo_output,
                    pseudo_target_tensor,
                    pseudo_mask_tensor,
                )

        val_probability = predict_indices(
            model,
            store,
            val_indices,
            VALIDATION_BATCH_SIZE,
        )

        _, val_summary = metric_tables(val_targets, val_probability)

        history_row = {
            "Variant": variant,
            "Fold": fold,
            "Epoch": epoch,
            "GoldLoss": epoch_gold_loss / STEPS_PER_EPOCH,
            "PseudoLoss": epoch_pseudo_loss / STEPS_PER_EPOCH,
            "TotalLoss": epoch_total_loss / STEPS_PER_EPOCH,
            "LearningRate": float(optimizer.param_groups[0]["lr"]),
            "ValMacroAUROC": val_summary["macro_AUROC"],
            "ValMacroAP": val_summary["macro_AP"],
            "ValMacroF1": val_summary["macro_F1"],
        }

        for label_index, label in enumerate(LABEL_COLUMNS):
            history_row[f"PseudoSeenCells__{label}"] = int(
                epoch_pseudo_label_counts[label_index]
            )

        history_rows.append(history_row)

        for val_row, uid in enumerate(gold_val_uids):
            record = {
                "Variant": variant,
                "Fold": fold,
                "Epoch": epoch,
                UID_COLUMN: uid,
            }

            for label_index, label in enumerate(LABEL_COLUMNS):
                record[f"{label}__true"] = int(val_targets[val_row, label_index])
                record[f"{label}__prob"] = float(val_probability[val_row, label_index])

            epoch_prediction_rows.append(record)

        # Persist fold progress at every epoch.  A crash still retrains the
        # current fold from scratch, but completed epoch diagnostics survive.
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        pd.DataFrame(epoch_prediction_rows).to_csv(prediction_path, index=False)

        if epoch == 1 or epoch % 4 == 0 or epoch == HEAD_EPOCHS:
            log(
                f"{variant:14s} fold={fold} "
                f"epoch={epoch:02d}/{HEAD_EPOCHS} "
                f"gold={history_row['GoldLoss']:.4f} "
                f"pseudo={history_row['PseudoLoss']:.4f} "
                f"total={history_row['TotalLoss']:.4f} "
                f"val_auc={history_row['ValMacroAUROC']:.4f} "
                f"val_ap={history_row['ValMacroAP']:.4f} "
                f"elapsed={elapsed_string(time.time() - started)}"
            )

    checkpoint = {
        "experiment": "W4.0",
        "variant": variant,
        "fold": fold,
        "epoch": HEAD_EPOCHS,
        "config_hash": config_hash,
        "config": head_config_payload(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }

    atomic_torch_save(checkpoint, checkpoint_path)

    history_df = pd.DataFrame(history_rows)
    prediction_df = pd.DataFrame(epoch_prediction_rows)

    del model, optimizer, scheduler, scaler
    gc.collect()

    if TRAIN_DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return history_df, prediction_df, pseudo_audit


# ============================================================
# 18. AGGREGATE 5-FOLD OOF
# ============================================================


def aggregate_variant_oof(
    variant: str,
    fold_prediction_frames: List[pd.DataFrame],
    fold_history_frames: List[pd.DataFrame],
) -> Dict[str, Any]:
    variant_result_root = RESULT_ROOT / variant
    variant_result_root.mkdir(parents=True, exist_ok=True)

    prediction_df = pd.concat(
        fold_prediction_frames,
        ignore_index=True,
    )

    history_df = pd.concat(
        fold_history_frames,
        ignore_index=True,
    )

    history_df.to_csv(
        variant_result_root / "all_fold_history.csv",
        index=False,
    )

    prediction_df.to_csv(
        variant_result_root / "epoch_oof_predictions.csv",
        index=False,
    )

    epoch_rows = []

    for epoch in range(1, HEAD_EPOCHS + 1):
        part = (
            prediction_df[prediction_df["Epoch"] == epoch]
            .sort_values(UID_COLUMN)
            .reset_index(drop=True)
        )

        if len(part) != 58:
            raise RuntimeError(
                f"{variant} epoch {epoch} expected 58 OOF rows, found {len(part)}"
            )

        y_true = np.stack(
            [part[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
            axis=1,
        )

        y_prob = np.stack(
            [
                part[f"{label}__prob"].values.astype(np.float32)
                for label in LABEL_COLUMNS
            ],
            axis=1,
        )

        _, summary = metric_tables(y_true, y_prob)

        epoch_rows.append(
            {
                "Epoch": epoch,
                "OOF_Macro_AUROC": summary["macro_AUROC"],
                "OOF_Macro_AP": summary["macro_AP"],
                "OOF_Macro_F1": summary["macro_F1"],
            }
        )

    epoch_summary_df = pd.DataFrame(epoch_rows)
    epoch_summary_df.to_csv(
        variant_result_root / "epoch_oof_summary.csv",
        index=False,
    )

    final = (
        prediction_df[prediction_df["Epoch"] == HEAD_EPOCHS]
        .sort_values(UID_COLUMN)
        .reset_index(drop=True)
    )

    y_true = np.stack(
        [final[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
        axis=1,
    )

    y_prob = np.stack(
        [final[f"{label}__prob"].values.astype(np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    per_label_df, final_summary = metric_tables(y_true, y_prob)

    per_label_df.to_csv(
        variant_result_root / "oof_per_label_metrics.csv",
        index=False,
    )

    final.to_csv(
        variant_result_root / "oof_predictions.csv",
        index=False,
    )

    fold_metric_rows = []

    for fold in range(1, NUM_FOLDS + 1):
        part = final[final["Fold"] == fold].copy()

        fold_y_true = np.stack(
            [part[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
            axis=1,
        )

        fold_y_prob = np.stack(
            [
                part[f"{label}__prob"].values.astype(np.float32)
                for label in LABEL_COLUMNS
            ],
            axis=1,
        )

        _, fold_summary = metric_tables(fold_y_true, fold_y_prob)
        fold_metric_rows.append(
            {
                "Fold": fold,
                **fold_summary,
            }
        )

    pd.DataFrame(fold_metric_rows).to_csv(
        variant_result_root / "fold_metrics.csv",
        index=False,
    )

    final_summary = {
        **final_summary,
        "final_epoch": HEAD_EPOCHS,
        "n_oof_studies": 58,
        "delta_AUROC_vs_W3_0": final_summary["macro_AUROC"] - W3_0_REFERENCE_AUROC,
    }

    (variant_result_root / "summary.json").write_text(
        json.dumps(final_summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    return {
        "summary": final_summary,
        "per_label": per_label_df,
        "oof": final,
        "epoch_summary": epoch_summary_df,
    }


# ============================================================
# 19. BOOTSTRAP WEAK - GOLD DELTA
# ============================================================


def bootstrap_macro_auc_delta(
    gold_oof: pd.DataFrame,
    weak_oof: pd.DataFrame,
    iterations: int,
    seed: int = 20260823,
) -> Dict[str, float]:
    gold = gold_oof.sort_values(UID_COLUMN).reset_index(drop=True)
    weak = weak_oof.sort_values(UID_COLUMN).reset_index(drop=True)

    if gold[UID_COLUMN].tolist() != weak[UID_COLUMN].tolist():
        raise RuntimeError("Gold and weak OOF UID alignment mismatch for bootstrap.")

    y_true = np.stack(
        [gold[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
        axis=1,
    )

    gold_prob = np.stack(
        [gold[f"{label}__prob"].values.astype(np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    weak_prob = np.stack(
        [weak[f"{label}__prob"].values.astype(np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_true)

    def macro_auc_for_sample(prob: np.ndarray, sample_index: np.ndarray) -> float:
        aucs = []
        sampled_y = y_true[sample_index]
        sampled_prob = prob[sample_index]

        for label_index in range(NUM_LABELS):
            auc = safe_auc(sampled_y[:, label_index], sampled_prob[:, label_index])
            if np.isfinite(auc):
                aucs.append(auc)

        return float(np.mean(aucs)) if aucs else float("nan")

    for _ in range(iterations):
        sample_index = rng.integers(0, n, size=n)
        gold_auc = macro_auc_for_sample(gold_prob, sample_index)
        weak_auc = macro_auc_for_sample(weak_prob, sample_index)

        if np.isfinite(gold_auc) and np.isfinite(weak_auc):
            deltas.append(weak_auc - gold_auc)

    values = np.asarray(deltas, dtype=np.float64)

    if len(values) == 0:
        return {
            "iterations_requested": iterations,
            "iterations_valid": 0,
            "mean_delta": float("nan"),
            "ci_2_5": float("nan"),
            "ci_97_5": float("nan"),
            "p_delta_gt_0": float("nan"),
        }

    return {
        "iterations_requested": int(iterations),
        "iterations_valid": int(len(values)),
        "mean_delta": float(values.mean()),
        "ci_2_5": float(np.percentile(values, 2.5)),
        "ci_97_5": float(np.percentile(values, 97.5)),
        "p_delta_gt_0": float((values > 0).mean()),
    }


# ============================================================
# 20. TRAIN VARIANT / BOTH VARIANTS
# ============================================================


def train_variant(
    variant: str,
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
    store: CuriaFeatureStore,
    w23_root: Optional[Path],
) -> Dict[str, Any]:
    log("\n" + "=" * 88)
    log(f"W4.0 TRAINING VARIANT: {variant}")
    log("=" * 88)

    fold_histories: List[pd.DataFrame] = []
    fold_predictions: List[pd.DataFrame] = []
    pseudo_audits: List[pd.DataFrame] = []

    for fold in range(1, NUM_FOLDS + 1):
        history_df, prediction_df, pseudo_audit = train_one_fold(
            variant,
            fold,
            store,
            gold_df,
            fold_zero,
            w23_root,
        )

        fold_histories.append(history_df)
        fold_predictions.append(prediction_df)

        if pseudo_audit is not None:
            pseudo_audits.append(pseudo_audit)

    if pseudo_audits:
        pd.concat(pseudo_audits, ignore_index=True).to_csv(
            RESULT_ROOT / variant / "pseudo_audit_all_folds.csv",
            index=False,
        )

    result = aggregate_variant_oof(
        variant,
        fold_predictions,
        fold_histories,
    )

    log(f"\n{variant} final pooled OOF:")
    log(json.dumps(result["summary"], indent=2, allow_nan=True))

    return result


def train_all_variants(
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
) -> Dict[str, Any]:
    w23_root = discover_w23_root()
    verify_w23_outer_folds(w23_root, gold_df, fold_zero)

    log(f"W2.3 root             : {w23_root}")

    store = CuriaFeatureStore(train_df)

    gold_result = train_variant(
        "gold_only",
        train_df,
        gold_df,
        fold_zero,
        store,
        w23_root=None,
    )

    weak_result = train_variant(
        "fold_safe_weak",
        train_df,
        gold_df,
        fold_zero,
        store,
        w23_root=w23_root,
    )

    bootstrap = bootstrap_macro_auc_delta(
        gold_result["oof"],
        weak_result["oof"],
        BOOTSTRAP_ITERATIONS,
    )

    gold_per = gold_result["per_label"].copy().set_index("Label")
    weak_per = weak_result["per_label"].copy().set_index("Label")

    per_label_comparison = pd.DataFrame(
        {
            "Gold_AUROC": gold_per["AUROC"],
            "Weak_AUROC": weak_per["AUROC"],
            "Delta_AUROC": weak_per["AUROC"] - gold_per["AUROC"],
            "Gold_AP": gold_per["AveragePrecision"],
            "Weak_AP": weak_per["AveragePrecision"],
            "Delta_AP": weak_per["AveragePrecision"] - gold_per["AveragePrecision"],
        }
    ).reset_index()

    per_label_comparison.to_csv(
        RESULT_ROOT / "W4_0_PER_LABEL_COMPARISON.csv",
        index=False,
    )

    comparison = {
        "gold_only": gold_result["summary"],
        "fold_safe_weak": weak_result["summary"],
        "weak_minus_gold": {
            "macro_AUROC": (
                weak_result["summary"]["macro_AUROC"]
                - gold_result["summary"]["macro_AUROC"]
            ),
            "macro_AP": (
                weak_result["summary"]["macro_AP"] - gold_result["summary"]["macro_AP"]
            ),
            "macro_F1": (
                weak_result["summary"]["macro_F1"] - gold_result["summary"]["macro_F1"]
            ),
        },
        "paired_study_bootstrap_AUROC_delta": bootstrap,
        "w3_0_reference_AUROC": W3_0_REFERENCE_AUROC,
        "config_hash": training_config_hash(),
        "config": head_config_payload(),
    }

    (RESULT_ROOT / "W4_0_COMPARISON.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log("\n" + "=" * 88)
    log("W4.0 COMPLETE")
    log("=" * 88)
    log(f"Gold-only Curia AUROC : " f"{gold_result['summary']['macro_AUROC']:.6f}")
    log(f"Weak Curia AUROC      : " f"{weak_result['summary']['macro_AUROC']:.6f}")
    log(
        f"Weak - Gold AUROC     : "
        f"{comparison['weak_minus_gold']['macro_AUROC']:+.6f}"
    )
    log(
        f"Bootstrap 95% CI      : "
        f"[{bootstrap['ci_2_5']:+.6f}, {bootstrap['ci_97_5']:+.6f}]"
    )
    log(f"P(delta>0)            : " f"{bootstrap['p_delta_gt_0']:.3f}")
    log(f"W3.0 OOF reference    : " f"{W3_0_REFERENCE_AUROC:.6f}")
    log(f"Results               : {RESULT_ROOT}")
    log(f"Checkpoints           : {CHECKPOINT_ROOT}")

    return comparison


def train_single_variant(
    variant: str,
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
) -> Dict[str, Any]:
    store = CuriaFeatureStore(train_df)

    if variant == "fold_safe_weak":
        w23_root = discover_w23_root()
        verify_w23_outer_folds(w23_root, gold_df, fold_zero)
    else:
        w23_root = None

    result = train_variant(
        variant,
        train_df,
        gold_df,
        fold_zero,
        store,
        w23_root,
    )

    return result["summary"]


# ============================================================
# 21. STATUS
# ============================================================


def show_status(
    train_df: pd.DataFrame,
    series_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> Dict[str, Any]:
    curia_root = None
    curia_error = None

    try:
        curia_root = discover_curia2_root()
    except Exception as exc:
        curia_error = repr(exc)

    cached = 0
    for uid in train_df[UID_COLUMN].astype(str):
        if cache_file_is_usable(study_cache_path(uid), uid):
            cached += 1

    status = {
        "device": str(TRAIN_DEVICE),
        "gpu_count": GPU_COUNT,
        "gpu_names": (
            [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
            if GPU_COUNT
            else []
        ),
        "train_studies": int(len(train_df)),
        "train_series": int(len(series_df)),
        "gold_studies": int(len(gold_df)),
        "cached_studies": int(cached),
        "cache_complete": cached == len(train_df),
        "curia_root": str(curia_root) if curia_root is not None else None,
        "curia_error": curia_error,
        "work_root": str(WORK_ROOT),
        "config_hash": training_config_hash(),
    }

    log(json.dumps(status, indent=2))
    return status


# ============================================================
# 22. ORCHESTRATION
# ============================================================


def run_w40_curia(mode: str = "all"):
    mode = str(mode).strip().lower()

    valid_modes = {
        "status",
        "cache",
        "train_gold",
        "train_weak",
        "train",
        "all",
    }

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    train_df, series_df, gold_df, fold_zero = load_training_tables()

    if mode == "status":
        return show_status(train_df, series_df, gold_df)

    if mode in {"cache", "all"}:
        build_feature_cache(train_df, series_df)

    if mode == "cache":
        return show_status(train_df, series_df, gold_df)

    if mode == "train_gold":
        return train_single_variant(
            "gold_only",
            train_df,
            gold_df,
            fold_zero,
        )

    if mode == "train_weak":
        return train_single_variant(
            "fold_safe_weak",
            train_df,
            gold_df,
            fold_zero,
        )

    if mode in {"train", "all"}:
        return train_all_variants(
            train_df,
            gold_df,
            fold_zero,
        )

    return None


# ============================================================
# 23. CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("RSNA W4.0 Curia-2 frozen feature cache + fold-safe training")
    )

    parser.add_argument(
        "--mode",
        choices=[
            "status",
            "cache",
            "train_gold",
            "train_weak",
            "train",
            "all",
        ],
        default="all",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_w40_curia(args.mode)


if __name__ == "__main__":
    main()
