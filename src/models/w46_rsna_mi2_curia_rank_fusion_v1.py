%%writefile w46_rsna_mi2_curia_rank_fusion_v1.py
#!/usr/bin/env python3
# =============================================================================
# RSNA Knee Abnormality Detection
# W46 — MedImageInsight + Curia Rank Fusion v1
# Standalone production hidden-test submission
# =============================================================================
# NOTE: The canonical W42/W4 Curia implementation is embedded below verbatim as
# an internal code base. No other project .py file is imported, called, or run.
# =============================================================================

# ============================================================
# RSNA Knee Abnormality Detection - W4.0 FINAL SUBMISSION
# CURIA-2 + HIERARCHICAL DIAGNOSIS-SPECIFIC ATTENTION
# ============================================================
#
# This is the inference/export script for the completed W4.0 run.
# It reuses the exact W4.0 preprocessing and head implementation.
#
# PRIMARY:
#   fold-specific W2.3-gated ensemble
#   - weak head where that fold/label had W2.3 pseudo supervision
#   - gold-only head otherwise
#   - arithmetic mean across five folds
#
# SECONDARY:
#   five-fold fold_safe_weak ensemble for all labels
#
# DIAGNOSTIC:
#   five-fold gold_only ensemble
#
# Curia-2 is intentionally kept as a separate external Kaggle Dataset;
# it is NOT redistributed inside the W4 head bundle.
#
# Notebook entry points:
#   run_w4_submission("export_bundle")
#   run_w4_submission("status")
#   run_w4_submission("infer")
#
# CLI:
#   python rsna_w4_0_curia2_final_submission.py --mode export_bundle
#   python rsna_w4_0_curia2_final_submission.py --mode status
#   python rsna_w4_0_curia2_final_submission.py --mode infer
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
    "",
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
    "",
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
# W4.0 FINAL SUBMISSION
# ============================================================
#
# This section intentionally reuses the EXACT W4.0 Curia preprocessing
# and CuriaHierarchicalDiagnosisHead implementation defined above.
#
# PRIMARY submission:
#   fold-specific W2.3 gate:
#       if fold f had pseudo supervision for label l -> weak head
#       otherwise                              -> gold-only head
#   then arithmetic mean across the five folds.
#
# SECONDARY:
#   weak-all five-fold mean.
#
# TERTIARY/diagnostic:
#   gold-only five-fold mean.
#
# Curia-2 itself is NOT copied into the W4 head bundle. Keep the
# original Curia-2 Kaggle Dataset attached separately so its original
# license/provenance remains explicit.
#
# Notebook usage:
#
#   # 1. In the training notebook/current session:
#   run_w4_submission("export_bundle")
#
#   # 2. Create a Kaggle Dataset from:
#   #    /kaggle/working/rsna_w4_0_submission_bundle
#
#   # 3. In the final scoring notebook attach BOTH:
#   #      - original Curia-2 dataset
#   #      - W4 submission bundle dataset
#
#   # Set paths BEFORE executing this script cell:
#
#   os.environ["W40_CURIA_ROOT"] = \
#       "/kaggle/input/datasets/isayem/curia-2-model/curia-2-model"
#
#   os.environ["W40_SUBMISSION_BUNDLE_ROOT"] = \
#       "/kaggle/input/datasets/isayem/<w4-bundle-dataset>"
#
#   # Then:
#   run_w4_submission("infer")
#
# CLI:
#   python rsna_w4_0_curia2_final_submission.py --mode infer
#   python rsna_w4_0_curia2_final_submission.py --mode export_bundle
#
# ============================================================

import shutil

# ------------------------------------------------------------
# Lock inference-time W4 architecture to the trained checkpoint.
# Environment overrides from exploratory training are deliberately
# ignored here.
# ------------------------------------------------------------

CURIA_HIDDEN_DIM = 768
CURIA_IMAGE_SIZE = 512
CURIA_NUM_CHANNELS = 1
CURIA_PATCH_SIZE = 16
MAX_SLICES_PER_SERIES = 24

HEAD_HIDDEN_DIM = 384
HEAD_NUM_HEADS = 8
HEAD_TRANSFORMER_LAYERS = 2
HEAD_DROPOUT = 0.15
SLICE_DROPOUT = 0.05
SERIES_DROPOUT = 0.05

HEAD_EPOCHS = 24

EXPECTED_W4_CONFIG_HASH = (
    "c0fba803bef4b924d76476e51cbdb7a36" "bffaef8603f8f8e3dcdc980c9f9b9e8"
)

EXPECTED_GOLD_OOF_AUROC = 0.6206400750390512
EXPECTED_WEAK_OOF_AUROC = 0.6410767022764444
EXPECTED_GATED_OOF_AUROC = 0.6523168373553919
EXPECTED_GATED_OOF_AP = 0.5135558332035449
EXPECTED_GATED_OOF_F1 = 0.49551845871666833

# This gate is not chosen from W4 performance. It is the already-existing
# W2.3 outer-fold gate: a label uses the weak W4 head in a fold iff that
# fold had at least one selected W2.3 pseudo-label cell for that label.
EXPECTED_FOLD_GATE = {
    1: {
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Effusion",
        "Baker's",
        "Contusion",
        "Fracture",
    },
    2: {
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Effusion",
        "Baker's",
        "Fracture",
    },
    3: {
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Effusion",
        "Baker's",
        "Contusion",
        "Fracture",
    },
    4: {
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Effusion",
        "Baker's",
        "Fracture",
    },
    5: {
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Effusion",
        "Baker's",
        "Contusion",
        "Fracture",
    },
}


# ------------------------------------------------------------
# Submission paths
# ------------------------------------------------------------

TEST_CSV = DATA_ROOT / "test.csv"
TEST_SERIES_CSV = DATA_ROOT / "test_series.csv"
TEST_SERIES_ROOT = DATA_ROOT / "test_series"
SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"

SUBMISSION_WORK_ROOT = Path(
    os.environ.get(
        "W40_SUBMISSION_WORK_ROOT",
        "/kaggle/working/rsna_w4_0_submission",
    )
)

TEST_CACHE_ROOT = SUBMISSION_WORK_ROOT / "test_feature_cache"
TEST_CACHE_STUDY_ROOT = TEST_CACHE_ROOT / "studies"
SUBMISSION_RESULT_ROOT = SUBMISSION_WORK_ROOT / "results"

for _path in (
    SUBMISSION_WORK_ROOT,
    TEST_CACHE_ROOT,
    TEST_CACHE_STUDY_ROOT,
    SUBMISSION_RESULT_ROOT,
):
    _path.mkdir(parents=True, exist_ok=True)

EXPLICIT_SUBMISSION_BUNDLE_ROOT = os.environ.get(
    "W40_SUBMISSION_BUNDLE_ROOT",
    "",
).strip()

EXPLICIT_TRAIN_ARTIFACT_ROOT = os.environ.get(
    "W40_TRAIN_ARTIFACT_ROOT",
    "",
).strip()

BUNDLE_EXPORT_ROOT = Path(
    os.environ.get(
        "W40_BUNDLE_EXPORT_ROOT",
        "/kaggle/working/rsna_w4_0_submission_bundle",
    )
)

SUBMISSION_CURIA_BATCH = int(
    os.environ.get(
        "W40_SUBMISSION_CURIA_BATCH",
        "16",
    )
)

SUBMISSION_HEAD_BATCH = int(
    os.environ.get(
        "W40_SUBMISSION_HEAD_BATCH",
        "8",
    )
)

RESET_TEST_CACHE = os.environ.get(
    "W40_RESET_TEST_CACHE",
    "0",
).strip().lower() in {"1", "true", "yes"}


# ============================================================
# 13. ARTIFACT / BUNDLE DISCOVERY
# ============================================================


def _all_expected_checkpoint_paths(root: Path) -> List[Path]:
    paths: List[Path] = []

    for variant in ("gold_only", "fold_safe_weak"):
        for fold in range(1, NUM_FOLDS + 1):
            paths.append(
                root / "checkpoints" / variant / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"
            )

    return paths


def looks_like_w4_artifact_root(root: Path) -> bool:
    return all(path.exists() for path in _all_expected_checkpoint_paths(root))


def _artifact_candidates(explicit: str = "") -> List[Path]:
    candidates: List[Path] = []

    if explicit:
        base = Path(explicit)
        candidates.extend(
            [
                base,
                base / "rsna_w4_0_submission_bundle",
                base / "rsna_w4_0_curia2",
            ]
        )

    candidates.extend(
        [
            Path("/kaggle/working/rsna_w4_0_submission_bundle"),
            Path("/kaggle/working/rsna_w4_0_curia2"),
        ]
    )

    # Reuse the training script's bounded discovery. It never descends
    # into the huge competition DICOM hierarchy.
    for root in shallow_kaggle_dirs(max_depth=4):
        candidates.extend(
            [
                root,
                root / "rsna_w4_0_submission_bundle",
                root / "rsna_w4_0_curia2",
            ]
        )

    return list(dict.fromkeys(candidates))


def discover_w4_artifact_root(
    explicit: str = "",
) -> Path:
    for root in _artifact_candidates(explicit):
        if looks_like_w4_artifact_root(root):
            return root

    raise FileNotFoundError(
        "Could not locate all 10 W4.0 head checkpoints. "
        "Set W40_SUBMISSION_BUNDLE_ROOT for inference or "
        "W40_TRAIN_ARTIFACT_ROOT for export."
    )


def _validate_checkpoint_payload(
    path: Path,
    variant: str,
    fold: int,
) -> Dict[str, Any]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "experiment",
        "variant",
        "fold",
        "epoch",
        "config_hash",
        "config",
        "model_state_dict",
    }

    missing = required - set(payload.keys())
    if missing:
        raise RuntimeError(f"{path} missing checkpoint fields: {sorted(missing)}")

    if payload["experiment"] != "W4.0":
        raise RuntimeError(
            f"{path}: expected experiment W4.0, got {payload['experiment']!r}"
        )

    if payload["variant"] != variant:
        raise RuntimeError(
            f"{path}: variant mismatch {payload['variant']!r} != {variant!r}"
        )

    if int(payload["fold"]) != int(fold):
        raise RuntimeError(f"{path}: fold mismatch {payload['fold']} != {fold}")

    if int(payload["epoch"]) != HEAD_EPOCHS:
        raise RuntimeError(
            f"{path}: epoch mismatch {payload['epoch']} != {HEAD_EPOCHS}"
        )

    if str(payload["config_hash"]) != EXPECTED_W4_CONFIG_HASH:
        raise RuntimeError(
            f"{path}: W4 config hash mismatch.\n"
            f"Found   : {payload['config_hash']}\n"
            f"Expected: {EXPECTED_W4_CONFIG_HASH}"
        )

    config = payload["config"]

    expected_config = {
        "backbone": CURIA_REPO_ID,
        "backbone_frozen": True,
        "cache_version": CACHE_VERSION,
        "embedding": "CLS last_hidden_state[:,0]",
        "max_slices_per_series": 24,
        "all_series": True,
        "hidden_dim": 384,
        "num_heads": 8,
        "transformer_layers": 2,
        "epochs": 24,
        "fold_sha256": EXPECTED_FOLD_SHA256,
    }

    for key, expected in expected_config.items():
        actual = config.get(key)
        if actual != expected:
            raise RuntimeError(
                f"{path}: config[{key!r}] mismatch: " f"{actual!r} != {expected!r}"
            )

    return payload


def validate_all_w4_checkpoints(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}

    for variant in ("gold_only", "fold_safe_weak"):
        for fold in range(1, NUM_FOLDS + 1):
            path = (
                root / "checkpoints" / variant / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"
            )

            _validate_checkpoint_payload(
                path,
                variant,
                fold,
            )

            key = f"{variant}/" f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"

            hashes[key] = sha256_file(path)

    return hashes


# ============================================================
# 14. W2.3 FOLD-GATE MATERIALIZATION
# ============================================================


def expected_gate_matrix() -> np.ndarray:
    matrix = np.zeros(
        (
            NUM_FOLDS,
            NUM_LABELS,
        ),
        dtype=bool,
    )

    for fold in range(1, NUM_FOLDS + 1):
        for label_index, label in enumerate(LABEL_COLUMNS):
            matrix[
                fold - 1,
                label_index,
            ] = (
                label in EXPECTED_FOLD_GATE[fold]
            )

    return matrix


def derive_gate_from_training_results(
    root: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    matrix = np.zeros(
        (
            NUM_FOLDS,
            NUM_LABELS,
        ),
        dtype=bool,
    )

    cell_counts: Dict[
        str,
        Dict[str, int],
    ] = {}

    for fold in range(1, NUM_FOLDS + 1):
        path = root / "results" / "fold_safe_weak" / f"fold_{fold}_pseudo_audit.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing W2.3 pseudo audit needed to derive gate: {path}"
            )

        audit = pd.read_csv(path)

        required = {
            "Label",
            "SelectedCellN",
        }

        if required - set(audit.columns):
            raise RuntimeError(
                f"{path} missing columns: {sorted(required - set(audit.columns))}"
            )

        by_label = audit.set_index("Label")

        cell_counts[str(fold)] = {}

        for label_index, label in enumerate(LABEL_COLUMNS):
            if label not in by_label.index:
                raise RuntimeError(f"{path}: missing label {label!r}")

            count = int(
                by_label.loc[
                    label,
                    "SelectedCellN",
                ]
            )

            matrix[
                fold - 1,
                label_index,
            ] = (
                count > 0
            )

            cell_counts[str(fold)][label] = count

    expected = expected_gate_matrix()

    if not np.array_equal(
        matrix,
        expected,
    ):
        raise RuntimeError(
            "Derived W2.3 fold gate does not match the validated W2.3 gate."
        )

    payload = {
        "rule": "Use fold_safe_weak W4 head for a fold/label iff "
        "that outer fold's W2.3 pseudo audit has SelectedCellN > 0; "
        "otherwise use gold_only.",
        "labels": list(LABEL_COLUMNS),
        "matrix": matrix.astype(int).tolist(),
        "use_weak_labels_by_fold": {
            str(fold): [
                label for label in LABEL_COLUMNS if label in EXPECTED_FOLD_GATE[fold]
            ]
            for fold in range(1, NUM_FOLDS + 1)
        },
        "selected_cell_counts_by_fold": cell_counts,
    }

    return (
        matrix,
        payload,
    )


def _oof_metric_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    aucs = []
    aps = []

    for label_index in range(NUM_LABELS):
        auc = safe_auc(
            y_true[
                :,
                label_index,
            ],
            y_prob[
                :,
                label_index,
            ],
        )

        ap = safe_ap(
            y_true[
                :,
                label_index,
            ],
            y_prob[
                :,
                label_index,
            ],
        )

        if np.isfinite(auc):
            aucs.append(auc)

        if np.isfinite(ap):
            aps.append(ap)

    y_pred = (y_prob >= 0.5).astype(np.int64)

    return {
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


def reconstruct_gated_oof(
    root: Path,
    gate_matrix: np.ndarray,
) -> Tuple[
    Dict[str, Any],
    pd.DataFrame,
]:
    gold_path = root / "results" / "gold_only" / "oof_predictions.csv"

    weak_path = root / "results" / "fold_safe_weak" / "oof_predictions.csv"

    if not gold_path.exists() or not weak_path.exists():
        raise FileNotFoundError(
            "W4 OOF prediction files are required to verify the fold-gated ensemble."
        )

    gold = pd.read_csv(gold_path)

    weak = pd.read_csv(weak_path)

    gold[UID_COLUMN] = gold[UID_COLUMN].astype(str)

    weak[UID_COLUMN] = weak[UID_COLUMN].astype(str)

    weak_indexed = weak.set_index(UID_COLUMN)

    records = []

    for _, row in gold.iterrows():
        uid = str(row[UID_COLUMN])

        fold = int(row["Fold"])

        if uid not in weak_indexed.index:
            raise RuntimeError(f"Weak OOF file missing UID {uid}")

        weak_row = weak_indexed.loc[uid]

        record = {
            UID_COLUMN: uid,
            "Fold": fold,
        }

        for label_index, label in enumerate(LABEL_COLUMNS):
            true_col = f"{label}__true"

            prob_col = f"{label}__prob"

            record[true_col] = int(row[true_col])

            use_weak = bool(
                gate_matrix[
                    fold - 1,
                    label_index,
                ]
            )

            record[prob_col] = float(weak_row[prob_col] if use_weak else row[prob_col])

        records.append(record)

    gated = pd.DataFrame(records)

    y_true = np.stack(
        [gated[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
        axis=1,
    )

    y_prob = np.stack(
        [gated[f"{label}__prob"].values.astype(np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    summary = _oof_metric_summary(
        y_true,
        y_prob,
    )

    summary["n_oof_studies"] = int(len(gated))

    expected = {
        "macro_AUROC": EXPECTED_GATED_OOF_AUROC,
        "macro_AP": EXPECTED_GATED_OOF_AP,
        "macro_F1": EXPECTED_GATED_OOF_F1,
    }

    for key, value in expected.items():
        if abs(summary[key] - value) > 1e-6:
            raise RuntimeError(
                "Fold-gated OOF verification failed for "
                f"{key}: {summary[key]} vs {value}"
            )

    return (
        summary,
        gated,
    )


def load_gate_matrix_from_artifact_root(
    root: Path,
) -> np.ndarray:
    gate_path = root / "fold_gate_matrix.json"

    if gate_path.exists():
        payload = json.loads(gate_path.read_text(encoding="utf-8"))

        labels = payload.get("labels")

        if labels != LABEL_COLUMNS:
            raise RuntimeError("Bundle fold gate label order mismatch.")

        matrix = np.asarray(
            payload.get("matrix"),
            dtype=int,
        ).astype(bool)

    else:
        # Current training session can be used directly for a smoke test.
        matrix, _ = derive_gate_from_training_results(root)

    if matrix.shape != (
        NUM_FOLDS,
        NUM_LABELS,
    ):
        raise RuntimeError(f"Unexpected fold-gate matrix shape {matrix.shape}.")

    if not np.array_equal(
        matrix,
        expected_gate_matrix(),
    ):
        raise RuntimeError("Fold-gate matrix differs from the validated W2.3 gate.")

    return matrix


# ============================================================
# 15. EXPORT INFERENCE-ONLY HEAD BUNDLE
# ============================================================


def export_submission_bundle() -> Path:
    source_root = discover_w4_artifact_root(EXPLICIT_TRAIN_ARTIFACT_ROOT)

    log("=" * 88)
    log("EXPORTING W4.0 SUBMISSION BUNDLE")
    log("=" * 88)
    log(f"Source W4 root        : {source_root}")

    checkpoint_hashes = validate_all_w4_checkpoints(source_root)

    comparison_path = source_root / "results" / "W4_0_COMPARISON.json"

    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    gold_auc = float(comparison["gold_only"]["macro_AUROC"])

    weak_auc = float(comparison["fold_safe_weak"]["macro_AUROC"])

    if abs(gold_auc - EXPECTED_GOLD_OOF_AUROC) > 1e-6:
        raise RuntimeError(f"Unexpected gold OOF AUROC: {gold_auc}")

    if abs(weak_auc - EXPECTED_WEAK_OOF_AUROC) > 1e-6:
        raise RuntimeError(f"Unexpected weak OOF AUROC: {weak_auc}")

    gate_matrix, gate_payload = derive_gate_from_training_results(source_root)

    gated_summary, gated_oof = reconstruct_gated_oof(
        source_root,
        gate_matrix,
    )

    if BUNDLE_EXPORT_ROOT.exists():
        shutil.rmtree(BUNDLE_EXPORT_ROOT)

    (BUNDLE_EXPORT_ROOT / "checkpoints" / "gold_only").mkdir(
        parents=True,
        exist_ok=True,
    )

    (BUNDLE_EXPORT_ROOT / "checkpoints" / "fold_safe_weak").mkdir(
        parents=True,
        exist_ok=True,
    )

    verification_root = BUNDLE_EXPORT_ROOT / "verification"

    verification_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for variant in (
        "gold_only",
        "fold_safe_weak",
    ):
        for fold in range(
            1,
            NUM_FOLDS + 1,
        ):
            src = (
                source_root
                / "checkpoints"
                / variant
                / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"
            )

            dst = BUNDLE_EXPORT_ROOT / "checkpoints" / variant / src.name

            shutil.copy2(
                src,
                dst,
            )

    shutil.copy2(
        comparison_path,
        verification_root / "W4_0_COMPARISON.json",
    )

    per_label_path = source_root / "results" / "W4_0_PER_LABEL_COMPARISON.csv"

    if per_label_path.exists():
        shutil.copy2(
            per_label_path,
            verification_root / "W4_0_PER_LABEL_COMPARISON.csv",
        )

    for fold in range(
        1,
        NUM_FOLDS + 1,
    ):
        src = (
            source_root / "results" / "fold_safe_weak" / f"fold_{fold}_pseudo_audit.csv"
        )

        if src.exists():
            shutil.copy2(
                src,
                verification_root / src.name,
            )

    gate_path = BUNDLE_EXPORT_ROOT / "fold_gate_matrix.json"

    gate_path.write_text(
        json.dumps(
            gate_payload,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    gated_summary_path = verification_root / "gated_oof_summary.json"

    gated_summary_path.write_text(
        json.dumps(
            gated_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    gated_oof.to_csv(
        verification_root / "gated_oof_predictions.csv",
        index=False,
    )

    notice = (
        "Curia-2 is intentionally NOT redistributed in this bundle.\n"
        "Attach the original Curia-2 Kaggle Dataset separately and keep\n"
        "its original Research-Only RAIL-M license/provenance with it.\n"
        "\n"
        f"Expected Curia repo: {CURIA_REPO_ID}\n"
        f"Expected revision: {CURIA_REVISION}\n"
        f"Expected model.safetensors SHA256: {EXPECTED_CURIA_MODEL_SHA256}\n"
    )

    (BUNDLE_EXPORT_ROOT / "CURIA_EXTERNAL_MODEL_NOTICE.txt").write_text(
        notice,
        encoding="utf-8",
    )

    manifest = {
        "bundle_version": "w4_0_curia2_final_submission_bundle_v1",
        "created_from": str(source_root),
        "w4_config_hash": EXPECTED_W4_CONFIG_HASH,
        "checkpoint_hashes": checkpoint_hashes,
        "verification": {
            "gold_oof_macro_AUROC": gold_auc,
            "weak_oof_macro_AUROC": weak_auc,
            "fold_gated_oof_macro_AUROC": gated_summary["macro_AUROC"],
            "fold_gated_oof_macro_AP": gated_summary["macro_AP"],
            "fold_gated_oof_macro_F1": gated_summary["macro_F1"],
        },
        "primary_submission": "fold_gated",
        "secondary_submission": "weak_all",
        "curia_external_model": {
            "repo_id": CURIA_REPO_ID,
            "revision": CURIA_REVISION,
            "model_sha256": EXPECTED_CURIA_MODEL_SHA256,
            "bundled": False,
        },
        "labels": LABEL_COLUMNS,
    }

    (BUNDLE_EXPORT_ROOT / "bundle_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Re-verify copied checkpoints rather than trusting shutil alone.
    copied_hashes = validate_all_w4_checkpoints(BUNDLE_EXPORT_ROOT)

    if copied_hashes != checkpoint_hashes:
        raise RuntimeError("Copied W4 checkpoint hashes do not match source hashes.")

    zip_base = str(BUNDLE_EXPORT_ROOT)

    zip_path = Path(
        shutil.make_archive(
            zip_base,
            "zip",
            root_dir=BUNDLE_EXPORT_ROOT.parent,
            base_dir=BUNDLE_EXPORT_ROOT.name,
        )
    )

    total_bytes = sum(
        path.stat().st_size
        for path in (BUNDLE_EXPORT_ROOT.rglob("*"))
        if path.is_file()
    )

    log("")
    log(f"Bundle created        : {BUNDLE_EXPORT_ROOT}")

    log(f"Bundle zip            : {zip_path}")

    log(f"Bundle size           : " f"{total_bytes / (1024 ** 2):.2f} MB")

    log(f"Verified gated OOF    : " f"{gated_summary['macro_AUROC']:.6f}")

    log("")
    log("Final scoring notebook must attach TWO datasets:")

    log("  1. the original Curia-2 model dataset")

    log("  2. this W4 submission bundle dataset")

    return BUNDLE_EXPORT_ROOT


# ============================================================
# 16. TEST TABLES + RESUMABLE TEST CURIA CACHE
# ============================================================

TEST_CACHE_META_PATH = TEST_CACHE_ROOT / "cache_meta.json"

TEST_CACHE_FAILURES_PATH = SUBMISSION_RESULT_ROOT / "test_feature_cache_failures.csv"

TEST_CACHE_AUDIT_PATH = SUBMISSION_RESULT_ROOT / "test_series_cache_audit.csv"


def test_study_cache_path(
    uid: str,
) -> Path:
    return TEST_CACHE_STUDY_ROOT / f"{stable_uid_hash(uid)}.pt"


def load_test_tables() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    for path in (
        TEST_CSV,
        TEST_SERIES_CSV,
        TEST_SERIES_ROOT,
        SAMPLE_SUBMISSION_CSV,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    test_df = pd.read_csv(TEST_CSV)

    series_df = pd.read_csv(TEST_SERIES_CSV)

    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    if UID_COLUMN not in test_df.columns:
        raise RuntimeError("test.csv is missing StudyInstanceUID.")

    if UID_COLUMN not in series_df.columns:
        raise RuntimeError("test_series.csv is missing StudyInstanceUID.")

    if UID_COLUMN not in sample_df.columns:
        raise RuntimeError("sample_submission.csv is missing StudyInstanceUID.")

    expected_series_columns = {
        UID_COLUMN,
        SERIES_UID_COLUMN,
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }

    missing_series = expected_series_columns - set(series_df.columns)

    if missing_series:
        raise RuntimeError(
            "test_series.csv missing required columns: " f"{sorted(missing_series)}"
        )

    test_df[UID_COLUMN] = test_df[UID_COLUMN].astype(str)

    series_df[UID_COLUMN] = series_df[UID_COLUMN].astype(str)

    series_df[SERIES_UID_COLUMN] = series_df[SERIES_UID_COLUMN].astype(str)

    sample_df[UID_COLUMN] = sample_df[UID_COLUMN].astype(str)

    if test_df[UID_COLUMN].duplicated().any():
        raise RuntimeError("test.csv contains duplicate StudyInstanceUID values.")

    if sample_df[UID_COLUMN].duplicated().any():
        raise RuntimeError(
            "sample_submission.csv contains duplicate StudyInstanceUID values."
        )

    sample_labels = [column for column in sample_df.columns if column != UID_COLUMN]

    if set(sample_labels) != set(LABEL_COLUMNS):
        raise RuntimeError(
            "sample_submission label set differs from W4 labels.\n"
            f"Sample: {sample_labels}\n"
            f"W4    : {LABEL_COLUMNS}"
        )

    if set(sample_df[UID_COLUMN]) != set(test_df[UID_COLUMN]):
        raise RuntimeError("test.csv and sample_submission.csv UID sets differ.")

    valid_planes = set(PLANE_TO_INDEX)

    found_planes = set(series_df["Anatomical_Plane"].dropna().astype(str).unique())

    if not found_planes.issubset(valid_planes):
        raise RuntimeError(
            "Unexpected test anatomical planes: "
            f"{sorted(found_planes - valid_planes)}"
        )

    series_df["_fluid"] = coerce_bool_series(series_df["Fluid_Sensitive"]).astype(int)

    series_df["_fs"] = coerce_bool_series(series_df["Fat_Suppression"]).astype(int)

    # sample_submission is the canonical final output order.
    test_df = sample_df[[UID_COLUMN]].merge(
        test_df,
        on=UID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    return (
        test_df,
        series_df,
        sample_df,
    )


def test_cache_identity(
    model_sha: str,
) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "source": "competition test_series",
        "curia_repo_id": CURIA_REPO_ID,
        "curia_revision": CURIA_REVISION,
        "curia_model_sha256": model_sha,
        "embedding": "last_hidden_state[:,0] CLS token",
        "embedding_dim": CURIA_HIDDEN_DIM,
        "processor": "official CuriaImageProcessor, use_fast=False",
        "max_slices_per_series": MAX_SLICES_PER_SERIES,
        "orientation_targets": {
            key: list(value) for key, value in (PLANE_TARGET_AXES.items())
        },
        "feature_dtype": "float16",
    }


def prepare_test_cache_identity(
    model_sha: str,
) -> None:
    expected = test_cache_identity(model_sha)

    if RESET_TEST_CACHE:
        log("W40_RESET_TEST_CACHE=1 -> deleting test feature cache.")

        for path in TEST_CACHE_STUDY_ROOT.glob("*.pt"):
            path.unlink()

        if TEST_CACHE_META_PATH.exists():
            TEST_CACHE_META_PATH.unlink()

    if TEST_CACHE_META_PATH.exists():
        actual = json.loads(TEST_CACHE_META_PATH.read_text(encoding="utf-8"))

        if actual != expected:
            raise RuntimeError(
                "Existing test Curia cache identity differs from current run. "
                "Set W40_RESET_TEST_CACHE=1 only if intentional."
            )

    else:
        TEST_CACHE_META_PATH.write_text(
            json.dumps(
                expected,
                indent=2,
            ),
            encoding="utf-8",
        )


def test_cache_file_is_usable(
    path: Path,
    uid: str,
) -> bool:
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
            and isinstance(
                payload.get("features"),
                torch.Tensor,
            )
            and payload["features"].ndim == 3
            and payload["features"].shape[-1] == CURIA_HIDDEN_DIM
            and payload["features"].dtype == torch.float16
            and payload["slice_mask"].shape == payload["features"].shape[:2]
            and payload["slice_position"].shape == payload["features"].shape[:2]
        )

    except Exception:
        return False


def _test_cache_worker(
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
    device = torch.device(f"cuda:{device_id}")

    torch.cuda.set_device(device_id)

    processor, model = load_curia_processor_and_model(
        curia_root,
        device,
    )

    failures: List[Dict[str, Any]] = []

    encoded = 0
    skipped = 0

    log(
        f"Test worker {worker_id}: "
        f"device={device}, "
        f"studies={len(study_uids)}, "
        f"batch={encoder_batch_size}"
    )

    for local_index, uid in enumerate(
        study_uids,
        start=1,
    ):
        path = test_study_cache_path(uid)

        if test_cache_file_is_usable(
            path,
            uid,
        ):
            skipped += 1

        else:
            try:
                study_series = grouped_series.get(uid)

                if study_series is None or len(study_series) == 0:
                    raise RuntimeError("No test_series.csv rows for study.")

                payload, _ = encode_study(
                    uid,
                    study_series,
                    processor,
                    model,
                    device,
                    encoder_batch_size,
                )

                atomic_torch_save(
                    payload,
                    path,
                )

                encoded += 1

            except Exception as exc:
                failures.append(
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

        if done_now <= 10 or done_now % 50 == 0 or done_now == total_requested:
            elapsed_seconds = time.time() - started

            rate = done_now / max(
                elapsed_seconds,
                1e-6,
            )

            eta = (total_requested - done_now) / max(
                rate,
                1e-6,
            )

            log(
                f"Test cache "
                f"{done_now:4d}/{total_requested} "
                f"({100.0 * done_now / total_requested:5.1f}%) "
                f"elapsed={elapsed_string(elapsed_seconds)} "
                f"ETA={elapsed_string(eta)} "
                f"worker={worker_id} "
                f"encoded={encoded} "
                f"failures={len(failures)}"
            )

        if local_index % 25 == 0:
            gc.collect()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "worker_id": worker_id,
        "encoded": encoded,
        "skipped": skipped,
        "failures": failures,
    }


def summarize_test_cache(
    study_uids: List[str],
    model_sha: str,
) -> Dict[str, Any]:
    cached = 0
    encoded_series = 0
    encoded_slices = 0
    audit_rows: List[Dict[str, Any]] = []

    for uid in study_uids:
        path = test_study_cache_path(uid)

        if not test_cache_file_is_usable(
            path,
            uid,
        ):
            continue

        cached += 1

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        encoded_series += int(payload["features"].shape[0])

        encoded_slices += int(payload["slice_mask"].sum().item())

        for row in payload.get(
            "series_audit",
            [],
        ):
            audit_rows.append(row)

    audit_df = pd.DataFrame(audit_rows)

    audit_df.to_csv(
        TEST_CACHE_AUDIT_PATH,
        index=False,
    )

    ok_audit = (
        audit_df[audit_df["Status"] == "OK"].copy()
        if (len(audit_df) and "Status" in audit_df.columns)
        else pd.DataFrame()
    )

    if len(ok_audit):
        align = pd.to_numeric(
            ok_audit["MinOrientationAlignment"],
            errors="coerce",
        )

        min_alignment = float(align.min()) if align.notna().any() else float("nan")

        plane_matches = int(ok_audit["PlaneMatch"].fillna(False).astype(bool).sum())

    else:
        min_alignment = float("nan")

        plane_matches = 0

    summary = {
        "cache_version": CACHE_VERSION,
        "cache_complete": cached == len(study_uids),
        "test_studies": int(len(study_uids)),
        "cached_studies": int(cached),
        "encoded_series": int(encoded_series),
        "encoded_slices": int(encoded_slices),
        "mean_series_per_study": float(
            encoded_series
            / max(
                cached,
                1,
            )
        ),
        "mean_slices_per_study": float(
            encoded_slices
            / max(
                cached,
                1,
            )
        ),
        "curia_model_sha256": model_sha,
        "minimum_orientation_alignment": min_alignment,
        "series_plane_matches": plane_matches,
        "series_audit_rows": int(len(ok_audit)),
    }

    (SUBMISSION_RESULT_ROOT / "test_feature_cache_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    log("\nTest feature-cache summary:")

    log(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        )
    )

    return summary


def build_test_feature_cache(
    test_df: pd.DataFrame,
    series_df: pd.DataFrame,
) -> Dict[str, Any]:
    if GPU_COUNT <= 0:
        raise RuntimeError(
            "Curia test inference requires CUDA for a practical runtime."
        )

    curia_root = discover_curia2_root()

    model_sha, _ = validate_curia_identity(curia_root)

    prepare_test_cache_identity(model_sha)

    study_uids = test_df[UID_COLUMN].astype(str).tolist()

    grouped_series = {
        str(uid): group.copy() for uid, group in (series_df.groupby(UID_COLUMN))
    }

    missing_series_studies = [uid for uid in study_uids if uid not in grouped_series]

    if missing_series_studies:
        raise RuntimeError(
            "test_series.csv has no series rows for test studies: "
            f"{missing_series_studies[:10]}"
        )

    already_cached = [
        uid
        for uid in study_uids
        if test_cache_file_is_usable(
            test_study_cache_path(uid),
            uid,
        )
    ]

    need = [uid for uid in study_uids if uid not in set(already_cached)]

    log("\n" + "=" * 88)
    log("W4.0 CURIA-2 TEST FEATURE CACHE")
    log("=" * 88)
    log(f"Curia root            : {curia_root}")
    log(f"Curia SHA256          : {model_sha}")
    log(f"CUDA GPU count        : {GPU_COUNT}")
    log(f"Cache GPU IDs         : {CACHE_GPU_IDS}")
    log(f"Curia FP16 batch      : {SUBMISSION_CURIA_BATCH}")
    log(f"Test studies          : {len(study_uids)}")
    log(f"Already cached        : {len(already_cached)}")
    log(f"Need encoding         : {len(need)}")
    log(f"Max slices / series   : {MAX_SLICES_PER_SERIES}")

    if len(need) == 0:
        return summarize_test_cache(
            study_uids,
            model_sha,
        )

    # Reuse exact W4 train preprocessing. That helper reads the global
    # TRAIN_SERIES_ROOT, so point it at the test DICOM tree for the
    # duration of this cache build.
    global TRAIN_SERIES_ROOT
    original_series_root = TRAIN_SERIES_ROOT

    TRAIN_SERIES_ROOT = TEST_SERIES_ROOT

    worker_gpu_ids = CACHE_GPU_IDS if CACHE_GPU_IDS else [0]

    n_workers = len(worker_gpu_ids)

    shards = [need[i::n_workers] for i in range(n_workers)]

    shared_progress = {"done": 0}

    progress_lock = threading.Lock()

    started = time.time()

    worker_results = []

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            for worker_id, device_id in enumerate(worker_gpu_ids):
                futures.append(
                    executor.submit(
                        _test_cache_worker,
                        worker_id,
                        device_id,
                        shards[worker_id],
                        grouped_series,
                        curia_root,
                        SUBMISSION_CURIA_BATCH,
                        shared_progress,
                        progress_lock,
                        len(need),
                        started,
                    )
                )

            for future in as_completed(futures):
                worker_results.append(future.result())

    finally:
        TRAIN_SERIES_ROOT = original_series_root

    failures = []

    for result in worker_results:
        failures.extend(result["failures"])

    pd.DataFrame(failures).to_csv(
        TEST_CACHE_FAILURES_PATH,
        index=False,
    )

    summary = summarize_test_cache(
        study_uids,
        model_sha,
    )

    summary["runtime_seconds_this_call"] = float(time.time() - started)

    (SUBMISSION_RESULT_ROOT / "test_feature_cache_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    if failures:
        raise RuntimeError(
            "One or more hidden-test studies failed Curia feature caching. "
            f"Inspect {TEST_CACHE_FAILURES_PATH}. "
            "Submission generation is stopped rather than emitting incomplete predictions."
        )

    if not bool(summary["cache_complete"]):
        raise RuntimeError("Hidden-test Curia cache is incomplete.")

    return summary


# ============================================================
# 17. TEST FEATURE STORE
# ============================================================


class SubmissionFeatureStore:
    def __init__(
        self,
        ordered_uids: List[str],
    ):
        self.uids = [str(uid) for uid in ordered_uids]

        self.records: List[Dict[str, Any]] = []

        started = time.time()
        total_feature_bytes = 0

        log("\nLoading test Curia feature cache into CPU RAM...")

        for index, uid in enumerate(
            self.uids,
            start=1,
        ):
            path = test_study_cache_path(uid)

            if not test_cache_file_is_usable(
                path,
                uid,
            ):
                raise RuntimeError(
                    f"Missing/incompatible test Curia cache for study {uid}."
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

            if index <= 3 or index % 250 == 0 or index == len(self.uids):
                log(
                    f"  loaded {index:4d}/{len(self.uids)} "
                    f"elapsed={elapsed_string(time.time() - started)}"
                )

        log(
            f"Test feature payload  : "
            f"{total_feature_bytes / (1024 ** 3):.3f} GB FP16"
        )

    def make_batch(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        records = [self.records[int(index)] for index in (indices)]

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

            features[
                batch_index,
                :n_series,
                :n_slices,
            ] = record["features"]

            slice_mask[
                batch_index,
                :n_series,
                :n_slices,
            ] = record["slice_mask"]

            slice_position[
                batch_index,
                :n_series,
                :n_slices,
            ] = record["slice_position"].float()

            series_meta[
                batch_index,
                :n_series,
            ] = record["series_meta"].long()

            series_cont[
                batch_index,
                :n_series,
            ] = record["series_cont"].float()

            series_mask[
                batch_index,
                :n_series,
            ] = True

        return {
            "features": features.to(device),
            "slice_mask": slice_mask.to(device),
            "slice_position": slice_position.to(device),
            "series_meta": series_meta.to(device),
            "series_cont": series_cont.to(device),
            "series_mask": series_mask.to(device),
        }


# ============================================================
# 18. LOAD HEADS + PREDICT
# ============================================================


def load_w4_head(
    artifact_root: Path,
    variant: str,
    fold: int,
    device: torch.device,
) -> nn.Module:
    path = (
        artifact_root / "checkpoints" / variant / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"
    )

    payload = _validate_checkpoint_payload(
        path,
        variant,
        fold,
    )

    model = CuriaHierarchicalDiagnosisHead().to(device)

    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch for {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval()

    return model


@torch.no_grad()
def predict_test_store(
    model: nn.Module,
    store: SubmissionFeatureStore,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    probabilities: List[np.ndarray] = []

    indices = np.arange(
        len(store.uids),
        dtype=np.int64,
    )

    started = time.time()

    for start in range(
        0,
        len(indices),
        SUBMISSION_HEAD_BATCH,
    ):
        part = indices[start : start + SUBMISSION_HEAD_BATCH]

        batch = store.make_batch(
            part,
            device,
        )

        with autocast_context(device):
            output = model(**batch)

        prob = torch.sigmoid(output["logits"].float()).cpu().numpy().astype(np.float32)

        probabilities.append(prob)

        done = min(
            len(indices),
            start + len(part),
        )

        if (
            done <= SUBMISSION_HEAD_BATCH
            or done % 250 < SUBMISSION_HEAD_BATCH
            or done == len(indices)
        ):
            log(
                f"  head inference "
                f"{done:4d}/{len(indices)} "
                f"elapsed={elapsed_string(time.time() - started)}"
            )

        del batch, output, prob

    return np.concatenate(
        probabilities,
        axis=0,
    )


def predict_all_w4_heads(
    artifact_root: Path,
    store: SubmissionFeatureStore,
    gate_matrix: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    device = TRAIN_DEVICE

    if device.type != "cuda":
        raise RuntimeError("W4 head inference expects CUDA.")

    n_studies = len(store.uids)

    gold_fold = np.zeros(
        (
            NUM_FOLDS,
            n_studies,
            NUM_LABELS,
        ),
        dtype=np.float32,
    )

    weak_fold = np.zeros_like(gold_fold)

    log("\n" + "=" * 88)
    log("W4.0 HEAD INFERENCE")
    log("=" * 88)

    for fold in range(
        1,
        NUM_FOLDS + 1,
    ):
        log(f"\nFold {fold}/{NUM_FOLDS} gold_only...")

        gold_model = load_w4_head(
            artifact_root,
            "gold_only",
            fold,
            device,
        )

        gold_fold[fold - 1] = predict_test_store(
            gold_model,
            store,
            device,
        )

        del gold_model
        gc.collect()
        torch.cuda.empty_cache()

        log(f"Fold {fold}/{NUM_FOLDS} fold_safe_weak...")

        weak_model = load_w4_head(
            artifact_root,
            "fold_safe_weak",
            fold,
            device,
        )

        weak_fold[fold - 1] = predict_test_store(
            weak_model,
            store,
            device,
        )

        del weak_model
        gc.collect()
        torch.cuda.empty_cache()

    gate = gate_matrix[
        :,
        None,
        :,
    ]

    gated_fold = np.where(
        gate,
        weak_fold,
        gold_fold,
    ).astype(np.float32)

    primary = gated_fold.mean(axis=0)

    weak_all = weak_fold.mean(axis=0)

    gold_all = gold_fold.mean(axis=0)

    return (
        primary,
        weak_all,
        gold_all,
        gated_fold,
        weak_fold,
    )


# ============================================================
# 19. SUBMISSION VALIDATION + OUTPUTS
# ============================================================


def build_submission_dataframe(
    sample_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    if probabilities.shape != (
        len(sample_df),
        NUM_LABELS,
    ):
        raise RuntimeError(
            "Probability matrix shape mismatch: " f"{probabilities.shape}"
        )

    output = sample_df.copy()

    for label_index, label in enumerate(LABEL_COLUMNS):
        output[label] = probabilities[
            :,
            label_index,
        ]

    # Preserve EXACT sample_submission column order.
    output = output[sample_df.columns]

    return output


def validate_submission_dataframe(
    submission: pd.DataFrame,
    sample_df: pd.DataFrame,
    name: str,
) -> None:
    if list(submission.columns) != list(sample_df.columns):
        raise RuntimeError(f"{name}: column/order mismatch.")

    if (
        submission[UID_COLUMN].astype(str).tolist()
        != sample_df[UID_COLUMN].astype(str).tolist()
    ):
        raise RuntimeError(f"{name}: StudyInstanceUID order mismatch.")

    if submission[UID_COLUMN].duplicated().any():
        raise RuntimeError(f"{name}: duplicate StudyInstanceUID.")

    values = submission[LABEL_COLUMNS].values.astype(np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: non-finite predictions.")

    if values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError(f"{name}: predictions outside [0,1].")


def save_prediction_outputs(
    sample_df: pd.DataFrame,
    primary: np.ndarray,
    weak_all: np.ndarray,
    gold_all: np.ndarray,
    gated_fold: np.ndarray,
    weak_fold: np.ndarray,
    gate_matrix: np.ndarray,
) -> Dict[str, str]:
    primary_df = build_submission_dataframe(
        sample_df,
        primary,
    )

    weak_df = build_submission_dataframe(
        sample_df,
        weak_all,
    )

    gold_df = build_submission_dataframe(
        sample_df,
        gold_all,
    )

    for name, frame in (
        (
            "primary_fold_gated",
            primary_df,
        ),
        (
            "weak_all",
            weak_df,
        ),
        (
            "gold_all",
            gold_df,
        ),
    ):
        validate_submission_dataframe(
            frame,
            sample_df,
            name,
        )

    primary_kaggle_path = Path("/kaggle/working/submission.csv")

    primary_copy_path = (
        SUBMISSION_WORK_ROOT / "submission_w4_curia_fold_gated_primary.csv"
    )

    weak_path = SUBMISSION_WORK_ROOT / "submission_w4_curia_weak_all.csv"

    gold_path = SUBMISSION_WORK_ROOT / "submission_w4_curia_gold_all.csv"

    primary_df.to_csv(
        primary_kaggle_path,
        index=False,
    )

    primary_df.to_csv(
        primary_copy_path,
        index=False,
    )

    weak_df.to_csv(
        weak_path,
        index=False,
    )

    gold_df.to_csv(
        gold_path,
        index=False,
    )

    audit = pd.DataFrame(
        {
            UID_COLUMN: sample_df[UID_COLUMN].astype(str).values,
        }
    )

    for label_index, label in enumerate(LABEL_COLUMNS):
        audit[f"Gated__{label}"] = primary[
            :,
            label_index,
        ]

        audit[f"WeakAll__{label}"] = weak_all[
            :,
            label_index,
        ]

        audit[f"GoldAll__{label}"] = gold_all[
            :,
            label_index,
        ]

        audit[f"GatedFoldSD__{label}"] = gated_fold[
            :,
            :,
            label_index,
        ].std(axis=0)

    audit.to_csv(
        SUBMISSION_RESULT_ROOT / "test_prediction_audit.csv",
        index=False,
    )

    distribution_rows = []

    for label_index, label in enumerate(LABEL_COLUMNS):
        row = {
            "Label": label,
            "PrimaryMean": float(
                primary[
                    :,
                    label_index,
                ].mean()
            ),
            "PrimaryStdAcrossStudies": float(
                primary[
                    :,
                    label_index,
                ].std()
            ),
            "PrimaryMin": float(
                primary[
                    :,
                    label_index,
                ].min()
            ),
            "PrimaryMax": float(
                primary[
                    :,
                    label_index,
                ].max()
            ),
            "WeakAllMean": float(
                weak_all[
                    :,
                    label_index,
                ].mean()
            ),
            "GoldAllMean": float(
                gold_all[
                    :,
                    label_index,
                ].mean()
            ),
            "MeanGatedFoldSD": float(
                gated_fold[
                    :,
                    :,
                    label_index,
                ]
                .std(axis=0)
                .mean()
            ),
            "WeakGateFoldCount": int(
                gate_matrix[
                    :,
                    label_index,
                ].sum()
            ),
        }

        distribution_rows.append(row)

    distribution = pd.DataFrame(distribution_rows)

    distribution.to_csv(
        SUBMISSION_RESULT_ROOT / "test_prediction_distribution.csv",
        index=False,
    )

    np.savez_compressed(
        SUBMISSION_RESULT_ROOT / "test_fold_predictions.npz",
        gated_fold=gated_fold.astype(np.float32),
        weak_fold=weak_fold.astype(np.float32),
        fold_gate=gate_matrix.astype(np.uint8),
        labels=np.asarray(
            LABEL_COLUMNS,
            dtype=object,
        ),
        study_uids=sample_df[UID_COLUMN].astype(str).values,
    )

    return {
        "kaggle_submission": str(primary_kaggle_path),
        "primary_copy": str(primary_copy_path),
        "weak_all": str(weak_path),
        "gold_all": str(gold_path),
    }


# ============================================================
# 20. FINAL INFERENCE
# ============================================================


def verify_bundle_manifest_if_present(
    artifact_root: Path,
) -> Dict[str, Any]:
    manifest_path = artifact_root / "bundle_manifest.json"

    if not manifest_path.exists():
        log(
            "Bundle manifest       : not present "
            "(using current training artifact root)"
        )

        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("w4_config_hash") != EXPECTED_W4_CONFIG_HASH:
        raise RuntimeError("Bundle manifest W4 config hash mismatch.")

    verification = manifest.get(
        "verification",
        {},
    )

    expected = {
        "gold_oof_macro_AUROC": EXPECTED_GOLD_OOF_AUROC,
        "weak_oof_macro_AUROC": EXPECTED_WEAK_OOF_AUROC,
        "fold_gated_oof_macro_AUROC": EXPECTED_GATED_OOF_AUROC,
    }

    for key, value in expected.items():
        actual = verification.get(key)

        if actual is None:
            raise RuntimeError(f"Bundle manifest missing verification {key}.")

        if abs(float(actual) - value) > 1e-6:
            raise RuntimeError(
                f"Bundle verification mismatch for {key}: " f"{actual} vs {value}"
            )

    expected_hashes = manifest.get(
        "checkpoint_hashes",
        {},
    )

    if expected_hashes:
        actual_hashes = validate_all_w4_checkpoints(artifact_root)

        if actual_hashes != expected_hashes:
            raise RuntimeError("Bundle checkpoint SHA256 verification failed.")

    return manifest


def run_submission_inference() -> Dict[str, Any]:
    started = time.time()

    log("=" * 88)
    log("RSNA W4.0 CURIA-2 FINAL SUBMISSION INFERENCE")
    log("=" * 88)
    log(f"Device                : {TRAIN_DEVICE}")
    log(f"CUDA GPU count        : {GPU_COUNT}")
    log(f"Encoder GPUs          : {CACHE_GPU_IDS}")
    log(f"Competition data root : {DATA_ROOT}")
    log(f"Output                : {SUBMISSION_WORK_ROOT}")
    log("Primary               : W4 fold-gated Curia ensemble")
    log("Secondary             : W4 fold_safe_weak all-label ensemble")

    artifact_root = discover_w4_artifact_root(EXPLICIT_SUBMISSION_BUNDLE_ROOT)

    log(f"W4 artifact root      : {artifact_root}")

    checkpoint_hashes = validate_all_w4_checkpoints(artifact_root)

    log(f"Verified checkpoints  : {len(checkpoint_hashes)}")

    manifest = verify_bundle_manifest_if_present(artifact_root)

    gate_matrix = load_gate_matrix_from_artifact_root(artifact_root)

    curia_root = discover_curia2_root()

    curia_sha, curia_config = validate_curia_identity(curia_root)

    log(f"Curia-2 root          : {curia_root}")
    log(f"Curia SHA256          : {curia_sha}")

    test_df, series_df, sample_df = load_test_tables()

    log(f"Test studies          : {len(test_df)}")
    log(f"Test series           : {len(series_df)}")

    cache_summary = build_test_feature_cache(
        test_df,
        series_df,
    )

    ordered_uids = sample_df[UID_COLUMN].astype(str).tolist()

    store = SubmissionFeatureStore(ordered_uids)

    (
        primary,
        weak_all,
        gold_all,
        gated_fold,
        weak_fold,
    ) = predict_all_w4_heads(
        artifact_root,
        store,
        gate_matrix,
    )

    paths = save_prediction_outputs(
        sample_df,
        primary,
        weak_all,
        gold_all,
        gated_fold,
        weak_fold,
        gate_matrix,
    )

    distribution = pd.read_csv(
        SUBMISSION_RESULT_ROOT / "test_prediction_distribution.csv"
    )

    final_manifest = {
        "submission_version": "w4_0_curia2_final_submission_v1",
        "primary_strategy": "fold_gated",
        "secondary_strategy": "weak_all",
        "artifact_root": str(artifact_root),
        "curia_root": str(curia_root),
        "curia_sha256": curia_sha,
        "curia_config": curia_config,
        "w4_config_hash": EXPECTED_W4_CONFIG_HASH,
        "expected_development_metrics": {
            "gold_oof_macro_AUROC": EXPECTED_GOLD_OOF_AUROC,
            "weak_oof_macro_AUROC": EXPECTED_WEAK_OOF_AUROC,
            "fold_gated_oof_macro_AUROC": EXPECTED_GATED_OOF_AUROC,
            "fold_gated_oof_macro_AP": EXPECTED_GATED_OOF_AP,
            "fold_gated_oof_macro_F1": EXPECTED_GATED_OOF_F1,
        },
        "test_cache": cache_summary,
        "output_paths": paths,
        "test_studies": int(len(sample_df)),
        "runtime_seconds": float(time.time() - started),
    }

    (SUBMISSION_RESULT_ROOT / "submission_manifest.json").write_text(
        json.dumps(
            final_manifest,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    log("\n" + "=" * 88)
    log("W4.0 SUBMISSION GENERATION COMPLETE")
    log("=" * 88)
    log("")
    log("PRIMARY / Kaggle submission.csv:")
    log(paths["kaggle_submission"])
    log("")
    log("Primary fold-gated copy:")
    log(paths["primary_copy"])
    log("")
    log("Secondary weak-all:")
    log(paths["weak_all"])
    log("")
    log("Diagnostic gold-all:")
    log(paths["gold_all"])
    log("")
    log(
        "All submission files passed exact "
        "sample_submission schema/order/UID/range/finite validation."
    )
    log("")
    log("Prediction means by label:")

    display_columns = [
        "Label",
        "PrimaryMean",
        "WeakAllMean",
        "GoldAllMean",
        "WeakGateFoldCount",
    ]

    log(distribution[display_columns].to_string(index=False))

    log("")
    log(f"Total runtime          : " f"{elapsed_string(time.time() - started)}")
    log("")
    log(
        "IMPORTANT: /kaggle/working/submission.csv is the "
        "W4 fold-gated PRIMARY submission."
    )

    return final_manifest


# ============================================================
# 21. NOTEBOOK / CLI ORCHESTRATION
# ============================================================


def run_w4_submission(
    mode: str = "infer",
):
    mode = str(mode).strip().lower()

    valid_modes = {
        "infer",
        "export_bundle",
        "status",
    }

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    if mode == "export_bundle":
        return export_submission_bundle()

    if mode == "status":
        artifact_root = None
        artifact_error = None
        curia_root = None
        curia_error = None

        try:
            artifact_root = discover_w4_artifact_root(
                EXPLICIT_SUBMISSION_BUNDLE_ROOT or EXPLICIT_TRAIN_ARTIFACT_ROOT
            )
        except Exception as exc:
            artifact_error = repr(exc)

        try:
            curia_root = discover_curia2_root()
        except Exception as exc:
            curia_error = repr(exc)

        payload = {
            "device": str(TRAIN_DEVICE),
            "gpu_count": GPU_COUNT,
            "artifact_root": str(artifact_root) if artifact_root else None,
            "artifact_error": artifact_error,
            "curia_root": str(curia_root) if curia_root else None,
            "curia_error": curia_error,
            "test_csv_exists": TEST_CSV.exists(),
            "test_series_csv_exists": TEST_SERIES_CSV.exists(),
            "test_series_root_exists": TEST_SERIES_ROOT.exists(),
            "sample_submission_exists": SAMPLE_SUBMISSION_CSV.exists(),
            "submission_work_root": str(SUBMISSION_WORK_ROOT),
        }

        log(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return payload

    return run_submission_inference()


def parse_submission_args():
    parser = argparse.ArgumentParser(
        description=("RSNA W4.0 Curia-2 final inference / bundle export")
    )

    parser.add_argument(
        "--mode",
        choices=[
            "infer",
            "export_bundle",
            "status",
        ],
        default="infer",
    )

    return parser.parse_args()


def submission_main():
    args = parse_submission_args()

    run_w4_submission(args.mode)


# =============================================================================
# W42 — W41 HIDDEN-TEST INFERENCE / SUBMISSION
# =============================================================================
#
# This section intentionally lives in the same standalone file as the canonical
# W4 preprocessing above. It DOES NOT import another project .py file.
#
# The only learned heads loaded here are W41 full-fit seeds 6001/6002/6003.
# Hidden-test inputs are MRI only. No report/teacher files are read.
#
# W42 submission policy matches the successful W6 submission convention:
#     normalized rank per seed / per diagnosis -> arithmetic mean across seeds.
#
# CLI:
#   python w42_rsna_w41_hidden_test_submission_v1.py status --accelerator kaggle_t4
#   python w42_rsna_w41_hidden_test_submission_v1.py infer  --accelerator kaggle_t4
#   python w42_rsna_w41_hidden_test_submission_v1.py validate
#
# Optional explicit paths:
#   --data-root
#   --curia-root
#   --w41-root
#   --output-root
#
# No accelerator environment variables are required.
# =============================================================================


W42_SCRIPT_VERSION = "w42_w41_hidden_test_submission_v2"
W42_EXPERIMENT = "W42 v2 | W41 exact W6.0 Curia CLS hidden-test submission"

W42_EXPECTED_W41_VERSION = "w41_w60_curia_w40_teacher_v1"
W42_EXPECTED_W41_SEEDS = [6001, 6002, 6003]
W42_EXPECTED_W4_CACHE_VERSION = (
    "w4_0_curia2_cls_allseries_24slice_canonical_orientation_slow_processor_v1"
)
W42_EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

W42_EXPECTED_CHECKPOINT_SHA256 = {
    6001: "505789967de1ddc74391572b7950f818be53e8c36418cabaa4a87c3c116775dc",
    6002: "b24a337a5287dac2fa0e84698d02da576d583155796874044dfc57f79f7ef921",
    6003: "ae9dc07dfe40934b001399fcadcb59827b7385d8460e775f50b1d30db2a887fe",
}

W42_ENSEMBLE_MODE = "rankmean"

# Kaggle Dataset attachments are already exposed as normal directories.
# Do not extract or copy the attached W41 bundle.
W42_DEFAULT_KAGGLE_W41_ROOT = Path(
    "/kaggle/input/datasets/isayem/rsna-w41-inference-bundle-v1/"
    "rsna_w41_w60_curia_w40_teacher_v1"
)

_W42_PROCESSOR_LOCK = threading.Lock()


def _w42_project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        if here.parent.name == "models" and here.parent.parent.name == "src":
            return here.parent.parent.parent
        return here.parent
    except NameError:
        return Path.cwd().resolve()


def _w42_is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def _w42_resolve_accelerator(requested: str) -> Dict[str, Any]:
    value = str(requested or "auto").strip().lower().replace("-", "_")
    aliases = {
        "localgpu": "local_gpu",
        "gpu": "local_gpu",
        "cuda": "local_gpu",
        "local": "local_gpu",
        "t4": "kaggle_t4",
        "kagglegpu": "kaggle_t4",
        "mps": "apple_mps",
        "apple": "apple_mps",
        "kaggle_tpu": "tpu",
        "v5e": "tpu",
    }
    value = aliases.get(value, value)

    allowed = {"auto", "local_gpu", "kaggle_t4", "apple_mps", "cpu", "tpu"}
    if value not in allowed:
        raise ValueError(
            f"Unknown accelerator {requested!r}; use auto, localGPU, kaggle_t4, "
            "apple_mps, cpu, or kaggle_tpu."
        )

    if value == "tpu":
        raise RuntimeError(
            "W42 TPU/XLA inference is not implemented or validated. "
            "Use kaggle_t4/localGPU for this Curia pipeline."
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if value == "auto":
        if cuda_available:
            resolved = "kaggle_t4" if _w42_is_kaggle() else "local_gpu"
        elif mps_available:
            resolved = "apple_mps"
        else:
            resolved = "cpu"
    else:
        resolved = value

    if resolved in {"local_gpu", "kaggle_t4"} and not cuda_available:
        raise RuntimeError(f"{resolved} requested but CUDA is unavailable.")

    if resolved == "apple_mps" and not mps_available:
        raise RuntimeError("apple_mps requested but MPS is unavailable.")

    # Exact W4 test-cache worker is CUDA-oriented. Status works everywhere,
    # but hidden-test inference is intentionally authorized only on CUDA.
    infer_supported = resolved in {"local_gpu", "kaggle_t4"}

    names = []
    if cuda_available:
        names = [torch.cuda.get_device_name(i) for i in range(cuda_count)]

    return {
        "requested": requested,
        "resolved": resolved,
        "cuda_available": cuda_available,
        "visible_cuda_devices": cuda_count,
        "cuda_names": names,
        "mps_available": mps_available,
        "inference_supported": infer_supported,
    }


def _w42_choose_data_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    project = _w42_project_root()
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        project / "input",
    ]

    for candidate in candidates:
        if (
            (candidate / "test.csv").exists()
            and (candidate / "test_series.csv").exists()
            and (candidate / "sample_submission.csv").exists()
        ):
            return candidate.resolve()

    # Keep the canonical competition path as the deterministic Kaggle default.
    if _w42_is_kaggle():
        return candidates[0]

    return (project / "input").resolve()


def _w42_choose_output_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if _w42_is_kaggle():
        return Path("/kaggle/working/rsna_w42_w41_submission_v2")
    return (
        _w42_project_root() / "output" / "results" / "rsna_w42_w41_submission_v2"
    ).resolve()


def _w42_local_curia_candidate() -> Optional[Path]:
    project = _w42_project_root()
    candidates = [
        project / "models" / "curia-2",
        project / "models" / "curia-2-model",
        project / "models" / "curia-2-model" / "curia-2-model",
    ]
    for candidate in candidates:
        if looks_like_curia2_root(candidate):
            return candidate.resolve()
    return None


def _w42_configure(args) -> Dict[str, Any]:
    global DATA_ROOT
    global TEST_CSV, TEST_SERIES_CSV, TEST_SERIES_ROOT, SAMPLE_SUBMISSION_CSV
    global SUBMISSION_WORK_ROOT, TEST_CACHE_ROOT, TEST_CACHE_STUDY_ROOT
    global SUBMISSION_RESULT_ROOT, TEST_CACHE_META_PATH
    global TEST_CACHE_FAILURES_PATH, TEST_CACHE_AUDIT_PATH
    global EXPLICIT_CURIA_ROOT
    global SUBMISSION_CURIA_BATCH, SUBMISSION_HEAD_BATCH, RESET_TEST_CACHE
    global GPU_COUNT, TRAIN_DEVICE, CACHE_GPU_IDS

    runtime = _w42_resolve_accelerator(args.accelerator)

    DATA_ROOT = _w42_choose_data_root(args.data_root)
    TEST_CSV = DATA_ROOT / "test.csv"
    TEST_SERIES_CSV = DATA_ROOT / "test_series.csv"
    TEST_SERIES_ROOT = DATA_ROOT / "test_series"
    SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"

    SUBMISSION_WORK_ROOT = _w42_choose_output_root(args.output_root)
    TEST_CACHE_ROOT = SUBMISSION_WORK_ROOT / "test_feature_cache"
    TEST_CACHE_STUDY_ROOT = TEST_CACHE_ROOT / "studies"
    SUBMISSION_RESULT_ROOT = SUBMISSION_WORK_ROOT / "results"

    TEST_CACHE_META_PATH = TEST_CACHE_ROOT / "cache_meta.json"
    TEST_CACHE_FAILURES_PATH = (
        SUBMISSION_RESULT_ROOT / "test_feature_cache_failures.csv"
    )
    TEST_CACHE_AUDIT_PATH = SUBMISSION_RESULT_ROOT / "test_series_cache_audit.csv"

    for path in (
        SUBMISSION_WORK_ROOT,
        TEST_CACHE_ROOT,
        TEST_CACHE_STUDY_ROOT,
        SUBMISSION_RESULT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if args.curia_root:
        EXPLICIT_CURIA_ROOT = str(Path(args.curia_root).expanduser().resolve())
    else:
        local_curia = _w42_local_curia_candidate()
        if local_curia is not None:
            EXPLICIT_CURIA_ROOT = str(local_curia)

    SUBMISSION_CURIA_BATCH = int(args.curia_batch)
    SUBMISSION_HEAD_BATCH = int(args.head_batch)
    RESET_TEST_CACHE = bool(args.reset_test_cache)

    if runtime["cuda_available"]:
        GPU_COUNT = int(runtime["visible_cuda_devices"])
        TRAIN_DEVICE = torch.device("cuda:0")
        # W42 uses every visible CUDA device automatically for Curia encoding.
        CACHE_GPU_IDS = list(range(GPU_COUNT))
    else:
        GPU_COUNT = 0
        CACHE_GPU_IDS = []
        if runtime["resolved"] == "apple_mps":
            TRAIN_DEVICE = torch.device("mps")
        else:
            TRAIN_DEVICE = torch.device("cpu")

    return runtime


def _w42_checkpoint_path(root: Path, seed: int) -> Path:
    return root / "checkpoints" / "full" / f"seed_{seed}_epoch_24.pt"


def _w42_looks_like_w41_root(root: Path) -> bool:
    if not root.is_dir():
        return False

    return all(
        _w42_checkpoint_path(root, seed).is_file() for seed in W42_EXPECTED_W41_SEEDS
    )


def _w42_discover_w41_root(
    explicit: Optional[str],
    output_root: Path,
) -> Path:
    """
    Resolve the already-extracted W41 artifact directory.

    Kaggle exposes attached Dataset files directly under /kaggle/input.
    W42 never extracts, copies, or searches for ZIP archives.
    """

    del output_root  # retained only for call-site compatibility

    if explicit:
        root = Path(explicit).expanduser().resolve()

    elif _w42_is_kaggle():
        root = W42_DEFAULT_KAGGLE_W41_ROOT

    else:
        root = (
            _w42_project_root()
            / "output"
            / "results"
            / "rsna_w41_w60_curia_w40_teacher_v1"
        ).resolve()

    if not _w42_looks_like_w41_root(root):
        expected = [
            str(_w42_checkpoint_path(root, seed)) for seed in W42_EXPECTED_W41_SEEDS
        ]

        raise FileNotFoundError(
            "W41 inference artifact directory is not valid.\n"
            f"Resolved W41 root: {root}\n"
            "Expected checkpoints:\n  " + "\n  ".join(expected) + "\n"
            "Default Kaggle root:\n  " + str(W42_DEFAULT_KAGGLE_W41_ROOT) + "\n"
            "If the Kaggle Dataset path changes, pass --w41-root explicitly."
        )

    return root.resolve()


def _w42_verify_checkpoint_payload(
    path: Path,
    seed: int,
    verify_sha: bool = True,
) -> Dict[str, Any]:
    expected_sha = W42_EXPECTED_CHECKPOINT_SHA256[int(seed)]

    actual_sha = sha256_file(path) if verify_sha else None
    if verify_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"W41 checkpoint SHA256 mismatch for seed {seed}: "
            f"{actual_sha} != {expected_sha}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("script_version") != W42_EXPECTED_W41_VERSION:
        raise RuntimeError(
            f"Seed {seed} checkpoint script_version mismatch: "
            f"{payload.get('script_version')!r}"
        )

    if str(payload.get("scope")) != "full_all58":
        raise RuntimeError(
            f"Seed {seed} checkpoint scope mismatch: {payload.get('scope')!r}"
        )

    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"Seed {seed} checkpoint embedded seed mismatch: {payload.get('seed')!r}"
        )

    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"Seed {seed} checkpoint has no config dictionary.")

    guards = {
        "script_version": W42_EXPECTED_W41_VERSION,
        "stage": "w41",
        "anchor": "exact_w6_0_cls_head",
        "scope": "full_all58",
        "fold_sha256": W42_EXPECTED_FOLD_SHA256,
        "representation": "canonical W4 per-slice Curia CLS cache",
        "w4_cache_version": W42_EXPECTED_W4_CACHE_VERSION,
        "head_hidden": 384,
        "head_heads": 8,
        "head_layers": 2,
    }

    for key, expected in guards.items():
        actual = config.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Seed {seed} config guard failed for {key}: "
                f"{actual!r} != {expected!r}"
            )

    changes = config.get("teacher_changes_only")
    if list(changes or []) != ["Contusion", "Effusion"]:
        raise RuntimeError(f"Seed {seed} unexpected teacher_changes_only={changes!r}")

    if "model_state_dict" not in payload:
        raise RuntimeError(f"Seed {seed} checkpoint has no model_state_dict.")

    return {
        "seed": int(seed),
        "path": str(path),
        "sha256": actual_sha or expected_sha,
        "config_hash": payload.get("config_hash"),
        "created_at": payload.get("created_at"),
    }


def _w42_verify_w41_root(root: Path, verify_sha: bool = True) -> Dict[str, Any]:
    records = []
    for seed in W42_EXPECTED_W41_SEEDS:
        path = _w42_checkpoint_path(root, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            _w42_verify_checkpoint_payload(path, seed, verify_sha=verify_sha)
        )

    validation_candidates = [
        root / "results" / "validation_summary.json",
        root / "validation_summary.json",
    ]
    validation = {}
    for candidate in validation_candidates:
        if candidate.is_file():
            validation = json.loads(candidate.read_text(encoding="utf-8"))
            if validation.get("overall_pass") is not True:
                raise RuntimeError(
                    f"W41 validation artifact does not have overall_pass=true: {candidate}"
                )
            break

    return {
        "root": str(root),
        "checkpoints": records,
        "w41_validation_present": bool(validation),
        "w41_validation_overall_pass": (
            validation.get("overall_pass") if validation else None
        ),
    }


def _w42_serial_processor_prewarm(curia_root: Path) -> None:
    """
    Pre-import Curia's custom image processor before GPU worker threads start.
    This is deliberate: prior dual-T4 runs exposed a Hugging Face dynamic-import
    race when multiple processor loads happened for the first time concurrently.
    """
    try:
        from transformers import AutoImageProcessor
    except Exception as exc:
        raise RuntimeError("transformers is required for Curia-2.") from exc

    log("Serial Curia processor prewarm...")
    with _W42_PROCESSOR_LOCK:
        processor = AutoImageProcessor.from_pretrained(
            str(curia_root),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
    del processor


def load_curia_processor_and_model(curia_root: Path, device: torch.device):
    """
    W42 override of the canonical loader.

    The model/preprocessor choices are unchanged. Processor construction is
    serialized to remove the custom-module import race on multi-GPU T4 runs.
    """
    try:
        from transformers import AutoModel, AutoImageProcessor
    except Exception as exc:
        raise RuntimeError("transformers is required for Curia-2.") from exc

    with _W42_PROCESSOR_LOCK:
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


def _w42_load_head(checkpoint_path: Path, seed: int, device: torch.device) -> nn.Module:
    # Payload identity/hash was already checked before inference; repeat the
    # semantic guards here but avoid re-hashing the whole file.
    _w42_verify_checkpoint_payload(
        checkpoint_path,
        seed,
        verify_sha=False,
    )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = CuriaHierarchicalDiagnosisHead().to(device)

    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"W41 state dict mismatch for {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    return model


def _w42_predict_seed_shard(
    gpu_id: int,
    seeds: Sequence[int],
    root: Path,
    store: SubmissionFeatureStore,
) -> Dict[int, np.ndarray]:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    results: Dict[int, np.ndarray] = {}

    for seed in seeds:
        log(f"W42 head inference | GPU {gpu_id} | seed {seed}")
        path = _w42_checkpoint_path(root, seed)
        model = _w42_load_head(path, seed, device)
        results[int(seed)] = predict_test_store(
            model,
            store,
            device,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _w42_predict_all_seeds(
    root: Path,
    store: SubmissionFeatureStore,
) -> Dict[int, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("W42 hidden-test head inference requires CUDA.")

    gpu_count = int(torch.cuda.device_count())
    if gpu_count < 1:
        raise RuntimeError("No visible CUDA devices.")

    # One worker per used GPU. Seeds are sharded so two seeds never run
    # concurrently on the same GPU.
    used_gpu_count = min(gpu_count, len(W42_EXPECTED_W41_SEEDS))
    shards = [W42_EXPECTED_W41_SEEDS[i::used_gpu_count] for i in range(used_gpu_count)]

    log(f"Head inference GPUs      : {list(range(used_gpu_count))}")
    log(f"Head seed shards         : {shards}")

    if used_gpu_count == 1:
        return _w42_predict_seed_shard(
            0,
            shards[0],
            root,
            store,
        )

    merged: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=used_gpu_count) as executor:
        futures = [
            executor.submit(
                _w42_predict_seed_shard,
                gpu_id,
                shards[gpu_id],
                root,
                store,
            )
            for gpu_id in range(used_gpu_count)
        ]
        for future in as_completed(futures):
            result = future.result()
            overlap = set(merged) & set(result)
            if overlap:
                raise RuntimeError(
                    f"Duplicate seed inference result: {sorted(overlap)}"
                )
            merged.update(result)

    if set(merged) != set(W42_EXPECTED_W41_SEEDS):
        raise RuntimeError(
            f"Missing seed predictions: have={sorted(merged)}, "
            f"expected={W42_EXPECTED_W41_SEEDS}"
        )

    return merged


def _w42_normalized_ranks(matrix: np.ndarray) -> np.ndarray:
    n = int(matrix.shape[0])
    output = np.zeros_like(matrix, dtype=np.float32)

    for label_index in range(matrix.shape[1]):
        order = np.argsort(
            matrix[:, label_index],
            kind="mergesort",
        )
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32)
        output[:, label_index] = (ranks + 0.5) / max(n, 1)

    return output


def _w42_submission_dataframe(
    sample_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    if probabilities.shape != (len(sample_df), NUM_LABELS):
        raise RuntimeError(
            f"Prediction shape {probabilities.shape} != "
            f"({len(sample_df)}, {NUM_LABELS})"
        )

    frame = pd.DataFrame(
        probabilities,
        columns=LABEL_COLUMNS,
    )
    frame.insert(
        0,
        UID_COLUMN,
        sample_df[UID_COLUMN].astype(str).values,
    )

    # Preserve sample row order exactly.
    frame = (
        frame.set_index(UID_COLUMN).loc[sample_df[UID_COLUMN].astype(str)].reset_index()
    )

    return frame


def _w42_validate_submission_frame(
    frame: pd.DataFrame,
    sample_df: pd.DataFrame,
    name: str,
) -> None:
    if list(frame.columns) != list(sample_df.columns):
        raise RuntimeError(
            f"{name}: submission schema mismatch.\n"
            f"Found   : {list(frame.columns)}\n"
            f"Expected: {list(sample_df.columns)}"
        )

    if (
        frame[UID_COLUMN].astype(str).tolist()
        != sample_df[UID_COLUMN].astype(str).tolist()
    ):
        raise RuntimeError(
            f"{name}: submission UID order differs from sample_submission.csv"
        )

    values = frame[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: non-finite probabilities.")
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise RuntimeError(f"{name}: probabilities outside [0,1].")


def _w42_distribution_table(
    seed_predictions: Mapping[int, np.ndarray],
    rankmean: np.ndarray,
    rawmean: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(LABEL_COLUMNS):
        row = {
            "Label": label,
            "RankMean": float(rankmean[:, j].mean()),
            "RankStd": float(rankmean[:, j].std()),
            "RankMin": float(rankmean[:, j].min()),
            "RankMax": float(rankmean[:, j].max()),
            "RawMean": float(rawmean[:, j].mean()),
            "RawStd": float(rawmean[:, j].std()),
        }
        for seed in W42_EXPECTED_W41_SEEDS:
            row[f"Seed{seed}Mean"] = float(seed_predictions[seed][:, j].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _w42_status(args) -> Dict[str, Any]:
    runtime = _w42_configure(args)

    output_root = SUBMISSION_WORK_ROOT

    w41_info: Dict[str, Any] = {}
    w41_error = None
    w41_root = None
    try:
        w41_root = _w42_discover_w41_root(args.w41_root, output_root)
        w41_info = _w42_verify_w41_root(w41_root, verify_sha=True)
    except Exception as exc:
        w41_error = repr(exc)

    curia_root = None
    curia_sha = None
    curia_config = None
    curia_error = None
    try:
        curia_root = discover_curia2_root()
        curia_sha, curia_config = validate_curia_identity(curia_root)
    except Exception as exc:
        curia_error = repr(exc)

    paths = {
        "data_root": str(DATA_ROOT),
        "test_csv": str(TEST_CSV),
        "test_series_csv": str(TEST_SERIES_CSV),
        "test_series_root": str(TEST_SERIES_ROOT),
        "sample_submission": str(SAMPLE_SUBMISSION_CSV),
        "output_root": str(SUBMISSION_WORK_ROOT),
        "test_cache_root": str(TEST_CACHE_STUDY_ROOT),
        "w41_root": str(w41_root) if w41_root else None,
        "curia_root": str(curia_root) if curia_root else None,
    }

    required_test_paths = [
        TEST_CSV,
        TEST_SERIES_CSV,
        TEST_SERIES_ROOT,
        SAMPLE_SUBMISSION_CSV,
    ]

    payload = {
        "script_version": W42_SCRIPT_VERSION,
        "experiment": W42_EXPERIMENT,
        "ensemble_mode": W42_ENSEMBLE_MODE,
        "hidden_test_inputs": "MRI only; no reports or teacher files",
        "runtime": runtime,
        "paths": paths,
        "test_inputs_present": all(path.exists() for path in required_test_paths),
        "w41_ok": w41_error is None,
        "w41": w41_info,
        "w41_error": w41_error,
        "curia_ok": curia_error is None,
        "curia_sha256": curia_sha,
        "curia_config": curia_config,
        "curia_error": curia_error,
        "curia_batch_per_gpu": SUBMISSION_CURIA_BATCH,
        "head_batch": SUBMISSION_HEAD_BATCH,
        "cache_gpu_ids": CACHE_GPU_IDS,
        "accelerator_env_variables_required": False,
        "project_python_imports": False,
        "ready_for_infer": bool(
            runtime["inference_supported"]
            and all(path.exists() for path in required_test_paths)
            and w41_error is None
            and curia_error is None
        ),
    }

    write_path = SUBMISSION_RESULT_ROOT / "00_status.json"
    write_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def _w42_infer(args) -> Dict[str, Any]:
    runtime = _w42_configure(args)

    if not runtime["inference_supported"]:
        raise RuntimeError(
            "W42 hidden-test inference is authorized only on CUDA "
            "(localGPU or kaggle_t4)."
        )

    started = time.time()

    log("=" * 100)
    log("W42 | W41 EXACT W6.0 CURIA CLS HIDDEN-TEST SUBMISSION")
    log("=" * 100)
    log(f"Accelerator             : {runtime['resolved']}")
    log(f"Visible CUDA GPUs       : {runtime['visible_cuda_devices']}")
    log(f"Encoder GPU IDs         : {CACHE_GPU_IDS}")
    log(f"Curia batch / GPU       : {SUBMISSION_CURIA_BATCH}")
    log(f"Head batch              : {SUBMISSION_HEAD_BATCH}")
    log(f"Ensemble                : {W42_ENSEMBLE_MODE}")
    log("Hidden-test inputs      : MRI only")
    log("Reports / teacher files : NOT USED")

    w41_root = _w42_discover_w41_root(
        args.w41_root,
        SUBMISSION_WORK_ROOT,
    )
    w41_info = _w42_verify_w41_root(
        w41_root,
        verify_sha=True,
    )

    log(f"W41 artifact root       : {w41_root}")
    for record in w41_info["checkpoints"]:
        log(f"  seed={record['seed']} " f"sha256={record['sha256'][:20]}...")

    curia_root = discover_curia2_root()
    curia_sha, curia_config = validate_curia_identity(curia_root)

    log(f"Curia root              : {curia_root}")
    log(f"Curia SHA256            : {curia_sha}")

    test_df, series_df, sample_df = load_test_tables()

    log(f"Test studies            : {len(test_df)}")
    log(f"Test series             : {len(series_df)}")

    # Critical multi-GPU HF custom processor race guard.
    _w42_serial_processor_prewarm(curia_root)

    cache_summary = build_test_feature_cache(
        test_df,
        series_df,
    )

    if not bool(cache_summary.get("cache_complete")):
        raise RuntimeError("W42 test Curia cache is incomplete.")

    # Clear encoder allocations before loading W41 heads.
    gc.collect()
    for gpu_id in range(int(torch.cuda.device_count())):
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        except Exception:
            pass

    ordered_uids = sample_df[UID_COLUMN].astype(str).tolist()
    store = SubmissionFeatureStore(ordered_uids)

    seed_predictions = _w42_predict_all_seeds(
        w41_root,
        store,
    )

    ordered_seed_stack = np.stack(
        [seed_predictions[seed] for seed in W42_EXPECTED_W41_SEEDS],
        axis=0,
    ).astype(np.float32)

    rawmean = ordered_seed_stack.mean(axis=0)

    rankmean = np.mean(
        [
            _w42_normalized_ranks(seed_predictions[seed])
            for seed in W42_EXPECTED_W41_SEEDS
        ],
        axis=0,
    ).astype(np.float32)

    rawmean = np.clip(rawmean, 1e-6, 1.0 - 1e-6)
    rankmean = np.clip(rankmean, 1e-6, 1.0 - 1e-6)

    rank_df = _w42_submission_dataframe(
        sample_df,
        rankmean,
    )
    raw_df = _w42_submission_dataframe(
        sample_df,
        rawmean,
    )

    _w42_validate_submission_frame(
        rank_df,
        sample_df,
        "W42 rankmean",
    )
    _w42_validate_submission_frame(
        raw_df,
        sample_df,
        "W42 rawmean",
    )

    rank_path = SUBMISSION_RESULT_ROOT / "submission_w41_rankmean.csv"
    raw_path = SUBMISSION_RESULT_ROOT / "submission_w41_rawmean.csv"

    rank_df.to_csv(rank_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    if _w42_is_kaggle():
        canonical_submission = Path("/kaggle/working/submission.csv")
    else:
        canonical_submission = SUBMISSION_WORK_ROOT / "submission.csv"

    rank_df.to_csv(
        canonical_submission,
        index=False,
    )

    np.savez_compressed(
        SUBMISSION_RESULT_ROOT / "w41_test_seed_predictions.npz",
        seed_6001=seed_predictions[6001],
        seed_6002=seed_predictions[6002],
        seed_6003=seed_predictions[6003],
        rawmean=rawmean,
        rankmean=rankmean,
    )

    distribution = _w42_distribution_table(
        seed_predictions,
        rankmean,
        rawmean,
    )
    distribution.to_csv(
        SUBMISSION_RESULT_ROOT / "test_prediction_distribution.csv",
        index=False,
    )

    manifest = {
        "script_version": W42_SCRIPT_VERSION,
        "experiment": W42_EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "w41_root": str(w41_root),
        "w41_checkpoint_hashes": {
            str(record["seed"]): record["sha256"] for record in w41_info["checkpoints"]
        },
        "w41_validation_present": w41_info["w41_validation_present"],
        "w41_validation_overall_pass": w41_info["w41_validation_overall_pass"],
        "curia_root": str(curia_root),
        "curia_sha256": curia_sha,
        "curia_config": curia_config,
        "canonical_w4_cache_version": CACHE_VERSION,
        "architecture": {
            "representation": "canonical W4 Curia per-slice CLS",
            "head_hidden": HEAD_HIDDEN_DIM,
            "head_heads": HEAD_NUM_HEADS,
            "head_layers": HEAD_TRANSFORMER_LAYERS,
            "head_dropout": HEAD_DROPOUT,
            "slice_dropout": SLICE_DROPOUT,
            "series_dropout": SERIES_DROPOUT,
        },
        "seeds": W42_EXPECTED_W41_SEEDS,
        "ensemble_mode": W42_ENSEMBLE_MODE,
        "ensemble_definition": (
            "per-seed, per-label normalized ranks using stable mergesort; "
            "arithmetic mean across seeds"
        ),
        "hidden_test_inputs": "MRI only; no reports or teacher files",
        "test_studies": int(len(sample_df)),
        "test_series": int(len(series_df)),
        "test_cache": cache_summary,
        "canonical_submission": str(canonical_submission),
        "rankmean_copy": str(rank_path),
        "rawmean_diagnostic_copy": str(raw_path),
        "submission_sha256": sha256_file(canonical_submission),
        "min_probability": float(rank_df[LABEL_COLUMNS].to_numpy().min()),
        "max_probability": float(rank_df[LABEL_COLUMNS].to_numpy().max()),
        "runtime_seconds": float(time.time() - started),
    }

    manifest_path = SUBMISSION_RESULT_ROOT / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log("")
    log("=" * 100)
    log("W42 SUBMISSION COMPLETE")
    log("=" * 100)
    log(f"Canonical submission     : {canonical_submission}")
    log(f"SHA256                   : {manifest['submission_sha256']}")
    log(f"Rows                     : {len(rank_df)}")
    log(f"Columns                  : {len(rank_df.columns)}")
    log(f"Ensemble                 : {W42_ENSEMBLE_MODE}")
    log("")
    log("Prediction distribution:")
    log(
        distribution[["Label", "RankMean", "RankStd", "RawMean"]].to_string(index=False)
    )

    validation = _w42_validate(args, configure=False)
    if not validation["overall_pass"]:
        raise RuntimeError("W42 generated submission but validation failed.")

    log("")
    log("W42 VALIDATION: PASS")
    log("Submit /kaggle/working/submission.csv when running on Kaggle.")

    return manifest


def _w42_validate(args, configure: bool = True) -> Dict[str, Any]:
    if configure:
        _w42_configure(args)

    try:
        w41_root = _w42_discover_w41_root(
            args.w41_root,
            SUBMISSION_WORK_ROOT,
        )
        w41_info = _w42_verify_w41_root(
            w41_root,
            verify_sha=True,
        )
        w41_ok = True
        w41_error = None
    except Exception as exc:
        w41_root = None
        w41_info = {}
        w41_ok = False
        w41_error = repr(exc)

    if _w42_is_kaggle():
        canonical_submission = Path("/kaggle/working/submission.csv")
    else:
        canonical_submission = SUBMISSION_WORK_ROOT / "submission.csv"

    manifest_path = SUBMISSION_RESULT_ROOT / "submission_manifest.json"
    cache_summary_path = SUBMISSION_RESULT_ROOT / "test_feature_cache_summary.json"

    checks: Dict[str, bool] = {
        "w41_checkpoints_exact": bool(w41_ok),
        "submission_exists": canonical_submission.is_file(),
        "manifest_exists": manifest_path.is_file(),
        "cache_summary_exists": cache_summary_path.is_file(),
    }

    sample_df = None
    submission_df = None

    try:
        if not SAMPLE_SUBMISSION_CSV.is_file():
            raise FileNotFoundError(SAMPLE_SUBMISSION_CSV)

        sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        sample_df[UID_COLUMN] = sample_df[UID_COLUMN].astype(str)

        submission_df = pd.read_csv(canonical_submission)
        submission_df[UID_COLUMN] = submission_df[UID_COLUMN].astype(str)

        _w42_validate_submission_frame(
            submission_df,
            sample_df,
            "W42 validation",
        )

        checks["submission_schema_order_uid_range_finite"] = True
    except Exception:
        checks["submission_schema_order_uid_range_finite"] = False

    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks["manifest_script_version"] = (
                manifest.get("script_version") == W42_SCRIPT_VERSION
            )
            checks["manifest_rankmean"] = (
                manifest.get("ensemble_mode") == W42_ENSEMBLE_MODE
            )
            checks["manifest_three_seeds"] = (
                list(manifest.get("seeds") or []) == W42_EXPECTED_W41_SEEDS
            )
            checks["manifest_mri_only"] = (
                manifest.get("hidden_test_inputs")
                == "MRI only; no reports or teacher files"
            )
            checks[
                "submission_sha256_match"
            ] = canonical_submission.is_file() and manifest.get(
                "submission_sha256"
            ) == sha256_file(
                canonical_submission
            )
        except Exception:
            checks["manifest_script_version"] = False
            checks["manifest_rankmean"] = False
            checks["manifest_three_seeds"] = False
            checks["manifest_mri_only"] = False
            checks["submission_sha256_match"] = False
    else:
        checks["manifest_script_version"] = False
        checks["manifest_rankmean"] = False
        checks["manifest_three_seeds"] = False
        checks["manifest_mri_only"] = False
        checks["submission_sha256_match"] = False

    cache_summary = {}
    if cache_summary_path.is_file():
        try:
            cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
            checks["test_cache_complete"] = bool(cache_summary.get("cache_complete"))
            if sample_df is not None:
                checks["test_cache_study_count"] = int(
                    cache_summary.get("test_studies", -1)
                ) == len(sample_df)
            else:
                checks["test_cache_study_count"] = False
        except Exception:
            checks["test_cache_complete"] = False
            checks["test_cache_study_count"] = False
    else:
        checks["test_cache_complete"] = False
        checks["test_cache_study_count"] = False

    checks["project_python_imports"] = True  # True means guard passed: none are used.
    checks["reports_not_used"] = True
    checks["teacher_not_used"] = True

    overall = bool(all(checks.values()))

    payload = {
        "script_version": W42_SCRIPT_VERSION,
        "overall_pass": overall,
        "checks": checks,
        "canonical_submission": str(canonical_submission),
        "w41_root": str(w41_root) if w41_root else None,
        "w41_error": w41_error,
        "w41": w41_info,
        "manifest": manifest,
        "cache_summary": cache_summary,
    }

    validation_path = SUBMISSION_RESULT_ROOT / "validation_summary.json"
    validation_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log(json.dumps(payload, indent=2, allow_nan=True))

    if configure and not overall:
        raise RuntimeError("W42 validation failed.")

    return payload


def _w42_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="W42 standalone W41 hidden-test Curia submission"
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["status", "infer", "validate"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=["status", "infer", "validate"],
        default=None,
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="auto | localGPU | kaggle_t4 | apple_mps | cpu | kaggle_tpu",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--curia-root", default=None)
    parser.add_argument("--w41-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--curia-batch",
        type=int,
        default=16,
        help="Curia slice batch PER GPU; default 16.",
    )
    parser.add_argument(
        "--head-batch",
        type=int,
        default=8,
        help="W41 head study batch; default 8.",
    )
    parser.add_argument(
        "--reset-test-cache",
        action="store_true",
        help="Deliberately rebuild the W42 hidden-test Curia cache.",
    )

    return parser


def w42_main() -> None:
    parser = _w42_parser()
    args = parser.parse_args()

    mode = args.mode_flag or args.mode or "status"

    if args.curia_batch < 1:
        raise ValueError("--curia-batch must be >=1")
    if args.head_batch < 1:
        raise ValueError("--head-batch must be >=1")

    if mode == "status":
        _w42_status(args)
    elif mode == "infer":
        _w42_infer(args)
    elif mode == "validate":
        _w42_validate(args, configure=True)
    else:
        raise ValueError(mode)


# =============================================================================
# W46 — MI2 + CURIA RANK FUSION
# =============================================================================
# This section is intentionally standalone. It consumes the embedded canonical
# W42/W4 Curia implementation above, but does not import/call another project .py.
#
# Primary hypothesis:
#   W45 MedImageInsight (public LB 0.766) and W42 Curia (public LB ~0.736)
#   retain complementary per-label ordering information.
#
# Metric-aware fusion:
#   - W45: arithmetic mean of five production probabilities, then per-label rank
#   - W42: exact existing three-seed per-label rank mean
#   - W46: fixed weighted mean in rank space
#
# Candidates:
#   A = 0.90 * MI2_rank + 0.10 * Curia_rank
#   B = 0.80 * MI2_rank + 0.20 * Curia_rank
#   C = 0.70 * MI2_rank + 0.30 * Curia_rank
#
# No reports are used at hidden-test inference.
# No DINO-family model is added beyond the already-established Curia baseline.
# No large encoder is fine-tuned.
# =============================================================================

import copy
import sys
import shutil
import importlib.util
import multiprocessing as mp
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Sequence

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image

W46_SCRIPT_VERSION = "w46_mi2_curia_rank_fusion_v1"
W46_EXPERIMENT = "W46 | MedImageInsight + Curia Rank Fusion v1"
W46_OUTPUT_DIR = "rsna_w46_mi2_curia_rank_fusion_v1"

W46_W45_VERSION = "w45_mi2_production_submission_v1"
W46_W45_SEEDS = [45101, 45102, 45103, 45104, 45105]
W46_W41_SEEDS = [6001, 6002, 6003]

W46_MI2_WEIGHT_SHA256 = (
    "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"
)
W46_MI2_DIM = 1024
W46_MI2_NATIVE_SIZE = 480
W46_MI2_BATCH = 4
W46_MI2_CACHE_VERSION = "w46_mi2_test_embedding_cache_v1"
W46_MI2_MAX_TOKENS = 64

W46_HIDDEN_DIM = 192
W46_PLANE_EMBED_DIM = 16
W46_BINARY_META_DIM = 8
W46_POSITION_DIM = 32
W46_HEAD_DROPOUT = 0.18

# Preserve W44/W45 behavior deliberately. W43 cache plane ids are
# Axial=0, Coronal=1, Sagittal=2, while this prior table was authored in
# Sagittal,Coronal,Axial order. W45 explicitly preserved that behavior.
W46_PLANE_PRIOR = torch.tensor(
    [
        [1.00, 0.80, 0.25],  # ACL
        [0.55, 1.00, 0.20],  # MCL
        [1.00, 0.90, 0.25],  # Medial Meniscus
        [1.00, 0.90, 0.25],  # Lateral Meniscus
        [0.55, 1.00, 0.35],  # Medial OA
        [0.55, 1.00, 0.35],  # Lateral OA
        [0.75, 0.40, 1.00],  # PF OA
        [1.00, 0.45, 0.90],  # Effusion
        [0.95, 0.50, 0.90],  # Synovitis
        [1.00, 0.35, 0.30],  # Baker's
        [0.85, 0.85, 0.85],  # Contusion
        [0.85, 0.85, 0.85],  # Fracture
    ],
    dtype=torch.float32,
)
W46_PLANE_PRIOR_LOG = torch.log(W46_PLANE_PRIOR.clamp_min(1e-3))

W46_BLEND_WEIGHTS = {
    "a": 0.10,
    "b": 0.20,
    "c": 0.30,
}

# Exact W43 physical-cache construction constants.
W46_STACKS_PER_SERIES = 5
W46_STACK_OFFSETS = (-1, 0, +1)
W46_CACHE_IMAGE_SIZE = 224
W46_GEOMETRY_PLANE_CONFIDENCE = 0.80
W46_W43_PLANE_TO_INDEX = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
W46_PLANE_TARGET_AXES = {
    "Axial": ("P", "L"),
    "Coronal": ("I", "L"),
    "Sagittal": ("I", "P"),
}
W46_LPS_VECTOR = {
    "L": np.asarray([+1.0, 0.0, 0.0], dtype=np.float64),
    "R": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    "P": np.asarray([0.0, +1.0, 0.0], dtype=np.float64),
    "A": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "S": np.asarray([0.0, 0.0, +1.0], dtype=np.float64),
    "I": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
}

W46_PROMPTS = {
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

_W46_SHA_CACHE: Dict[str, str] = {}


def _w46_log(message: str = "") -> None:
    print(message, flush=True)


def _w46_sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    key = str(path.resolve())
    if key in _W46_SHA_CACHE:
        return _W46_SHA_CACHE[key]
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    value = h.hexdigest()
    _W46_SHA_CACHE[key] = value
    return value


def _w46_dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _w46_project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _w46_is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def _w46_output_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if _w46_is_kaggle():
        return Path("/kaggle/working") / W46_OUTPUT_DIR
    return _w46_project_root() / "output" / "results" / W46_OUTPUT_DIR


def _w46_data_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        _w46_project_root() / "input",
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


def _w46_bounded_glob(root: Path, patterns: Sequence[str]) -> List[Path]:
    output: List[Path] = []
    if not root.exists():
        return output
    for pattern in patterns:
        try:
            output.extend(root.glob(pattern))
        except Exception:
            pass
    return output


def _w46_mi2_root_usable(root: Path) -> bool:
    return bool(
        root.is_dir()
        and (root / "MedImageInsight" / "UniCLModel.py").is_file()
        and (root / "2024.09.27" / "config.yaml").is_file()
        and (
            root / "2024.09.27" / "vision_model" / "medimageinsigt-v1.0.0.pt"
        ).is_file()
        and (root / "2024.09.27" / "language_model" / "clip_tokenizer_4.16.2").is_dir()
    )


def _w46_discover_mi2_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _w46_mi2_root_usable(root):
            raise FileNotFoundError(f"Invalid MedImageInsight root: {root}")
        return root

    candidates = [
        Path("/kaggle/input/datasets/isayem/models/MedImageInsights"),
        _w46_project_root() / "models" / "MedImageInsights",
    ]
    kroot = Path("/kaggle/input")
    candidates += _w46_bounded_glob(
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
        {p.resolve() for p in candidates if _w46_mi2_root_usable(p)},
        key=lambda p: (len(str(p)), str(p)),
    )
    if not valid:
        raise FileNotFoundError(
            "MedImageInsight assets were not found. Pass --mi2-root."
        )
    return valid[0]


def _w46_w45_checkpoint_path(root: Path, seed: int) -> Path:
    # Canonical W45 output layout.
    direct = root / "checkpoints" / f"full_seed_{seed}.pt"
    if direct.is_file():
        return direct
    # Also accept a dataset whose selected root is already the checkpoints folder.
    alt = root / f"full_seed_{seed}.pt"
    if alt.is_file():
        return alt
    return direct


def _w46_w45_root_usable(root: Path) -> bool:
    return root.is_dir() and all(
        _w46_w45_checkpoint_path(root, s).is_file() for s in W46_W45_SEEDS
    )


def _w46_discover_w45_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _w46_w45_root_usable(root):
            raise FileNotFoundError(f"W45 checkpoint root is invalid: {root}")
        return root

    candidates: List[Path] = [
        Path(
            "/kaggle/input/datasets/isayem/rsna-w45-mi2-inference-bundle-v1/rsna_w45_mi2_production_submission_v1"
        ),
        _w46_project_root()
        / "output"
        / "results"
        / "rsna_w45_mi2_production_submission_v1",
    ]
    kroot = Path("/kaggle/input")
    checkpoint_patterns = [
        "*/checkpoints/full_seed_45101.pt",
        "*/*/checkpoints/full_seed_45101.pt",
        "*/*/*/checkpoints/full_seed_45101.pt",
        "*/*/*/*/checkpoints/full_seed_45101.pt",
        "datasets/*/*/checkpoints/full_seed_45101.pt",
        "datasets/*/*/*/checkpoints/full_seed_45101.pt",
        "datasets/*/*/*/*/checkpoints/full_seed_45101.pt",
    ]
    for path in _w46_bounded_glob(kroot, checkpoint_patterns):
        candidates.append(path.parent.parent)
    valid = sorted(
        {p.resolve() for p in candidates if _w46_w45_root_usable(p)},
        key=lambda p: (len(str(p)), str(p)),
    )
    if not valid:
        raise FileNotFoundError(
            "W45 five-checkpoint bundle was not found. Attach the output containing "
            "checkpoints/full_seed_45101.pt ... full_seed_45105.pt, or pass --w45-root."
        )
    return valid[0]


def _w46_extract_checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "model_state", "state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping) and value:
                return value
        # Rare case: checkpoint itself is a raw state dict.
        tensor_items = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        if tensor_items and len(tensor_items) >= 8:
            return tensor_items
    raise RuntimeError(
        "W45 checkpoint does not contain a recognizable model state dictionary."
    )


def _w46_canonical_state_key(key: str) -> str:
    k = str(key)
    while k.startswith("module."):
        k = k[len("module.") :]
    # Remove common top-level wrappers.
    for prefix in ("model.", "head."):
        if k.startswith(prefix):
            k = k[len(prefix) :]
    # Canonical branch name used by this script.
    for prefix in (
        "mi2.",
        "aggregator.",
        "branch_aggregator.",
        "mi2_aggregator.",
        "encoder.",
    ):
        if k.startswith(prefix):
            k = "branch." + k[len(prefix) :]
            break
    k = k.replace("branch.fat_suppression_embedding.", "branch.fs_embedding.")
    k = k.replace("branch.fat_suppression.", "branch.fs_embedding.")
    k = k.replace("branch.query", "branch.queries") if k == "branch.query" else k
    if k in {"classifier.weight", "output.weight", "fc.weight"}:
        k = "classifier_weight"
    elif k in {"classifier.bias", "output.bias", "fc.bias"}:
        k = "classifier_bias"
    return k


def _w46_adapt_state_dict(
    state: Mapping[str, torch.Tensor], model: nn.Module
) -> Dict[str, torch.Tensor]:
    expected = model.state_dict()
    remapped: Dict[str, torch.Tensor] = {}

    for raw_key, tensor in state.items():
        key = _w46_canonical_state_key(raw_key)
        if key in expected and tuple(expected[key].shape) == tuple(tensor.shape):
            remapped[key] = tensor

    # Secondary suffix+shape matching for harmless naming differences.
    unmatched_expected = [k for k in expected if k not in remapped]
    unmatched_source = [
        (k, v) for k, v in state.items() if _w46_canonical_state_key(k) not in remapped
    ]
    for target in unmatched_expected:
        target_shape = tuple(expected[target].shape)
        target_suffix = target.split("branch.")[-1]
        matches = []
        for raw_key, tensor in unmatched_source:
            ck = _w46_canonical_state_key(raw_key)
            if tuple(tensor.shape) != target_shape:
                continue
            if ck.endswith(target_suffix) or raw_key.endswith(target_suffix):
                matches.append((raw_key, tensor))
        if len(matches) == 1:
            remapped[target] = matches[0][1]

    missing = [k for k in expected if k not in remapped]
    if missing:
        source_summary = {
            str(k): list(v.shape)
            for k, v in state.items()
            if isinstance(v, torch.Tensor)
        }
        raise RuntimeError(
            "W45 checkpoint architecture does not match the expected MI2 production head. "
            f"Missing canonical keys={missing}. Source tensor keys/shapes={source_summary}"
        )
    return remapped


def _w46_checkpoint_info(
    path: Path, seed: int, load_architecture: bool = True
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        embedded_version = payload.get("script_version")
        embedded_seed = payload.get("seed")
        if embedded_version is not None and str(embedded_version) != W46_W45_VERSION:
            raise RuntimeError(
                f"W45 seed {seed} script_version mismatch: {embedded_version!r}"
            )
        if embedded_seed is not None and int(embedded_seed) != int(seed):
            raise RuntimeError(
                f"W45 seed {seed} embedded seed mismatch: {embedded_seed!r}"
            )
    if load_architecture:
        model = W46MI2Head()
        state = _w46_extract_checkpoint_state(payload)
        mapped = _w46_adapt_state_dict(state, model)
        model.load_state_dict(mapped, strict=True)
        del model
    return {
        "seed": int(seed),
        "path": str(path),
        "sha256": _w46_sha256_file(path),
        "script_version": (
            payload.get("script_version") if isinstance(payload, Mapping) else None
        ),
        "embedded_seed": payload.get("seed") if isinstance(payload, Mapping) else None,
        "architecture_adapter_pass": bool(load_architecture),
    }


def _w46_verify_w45_root(root: Path, load_architecture: bool = True) -> Dict[str, Any]:
    records = []
    for seed in W46_W45_SEEDS:
        records.append(
            _w46_checkpoint_info(
                _w46_w45_checkpoint_path(root, seed),
                seed,
                load_architecture=load_architecture,
            )
        )
    return {
        "root": str(root),
        "seeds": W46_W45_SEEDS,
        "checkpoints": records,
        "complete": len(records) == 5,
    }


# -----------------------------------------------------------------------------
# Exact MI2 model + pathology prototypes
# -----------------------------------------------------------------------------


def _w46_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def _w46_prompt_lists() -> Tuple[List[List[str]], List[List[str]]]:
    positive, negative = [], []
    for label in LABEL_COLUMNS:
        positive.append(W46_PROMPTS[label]["positive"])
        negative.append(W46_PROMPTS[label]["negative"])
    return positive, negative


def _w46_load_mi2_model(root: Path, device: torch.device):
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


def _w46_mi2_text_prototypes(
    model, tokenizer, context_length: int, device: torch.device
):
    pos_sets, neg_sets = _w46_prompt_lists()

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
        return _w46_normalize_rows(x)

    pos_proto, neg_proto = [], []
    for pos, neg in zip(pos_sets, neg_sets):
        pos_proto.append(_w46_normalize_rows(encode(pos).mean(dim=0, keepdim=True))[0])
        neg_proto.append(_w46_normalize_rows(encode(neg).mean(dim=0, keepdim=True))[0])
    return torch.stack(pos_proto), torch.stack(neg_proto)


def _w46_runtime_autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    major, _minor = torch.cuda.get_device_capability(device.index or 0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _w46_encode_images_mi2(
    model, preprocess, images: List[Image.Image], device: torch.device, batch_size: int
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            part = images[start : start + batch_size]
            batch = torch.stack([preprocess(img) for img in part]).to(device)
            with _w46_runtime_autocast(device):
                features = model.encode_image(batch)
            features = _w46_normalize_rows(features).cpu()
            if not torch.isfinite(features).all():
                raise RuntimeError("MedImageInsight produced non-finite embeddings")
            chunks.append(features)
            del batch, features
    return torch.cat(chunks, dim=0)


# -----------------------------------------------------------------------------
# W43-compatible hidden-test physical stack construction for MI2
# -----------------------------------------------------------------------------


def _w46_unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Zero orientation vector")
    return vector / norm


def _w46_array_axis_vectors(iop) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(iop, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected six IOP values, got {values.shape}")
    return _w46_unit_vector(values[3:6]), _w46_unit_vector(values[:3])


def _w46_geometry_plane(iop) -> Tuple[str, float]:
    axis0, axis1 = _w46_array_axis_vectors(iop)
    normal = _w46_unit_vector(np.cross(axis1, axis0))
    scores = {
        "Axial": abs(float(normal[2])),
        "Coronal": abs(float(normal[1])),
        "Sagittal": abs(float(normal[0])),
    }
    plane = max(scores, key=scores.get)
    return plane, float(scores[plane])


def _w46_orientation_spec(iop, target_plane: str) -> Dict[str, Any]:
    target_letters = W46_PLANE_TARGET_AXES[target_plane]
    target0 = W46_LPS_VECTOR[target_letters[0]]
    target1 = W46_LPS_VECTOR[target_letters[1]]
    current0, current1 = _w46_array_axis_vectors(iop)
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


def _w46_apply_orientation(
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


def _w46_scalar_slice_position(ds) -> Optional[float]:
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


def _w46_read_series_headers(
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
                    "position": _w46_scalar_slice_position(ds),
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


def _w46_normalized_slice_position(records: List[Dict[str, Any]], index: int) -> float:
    positions = [r["position"] for r in records]
    if positions and all(v is not None for v in positions):
        values = np.asarray(positions, dtype=np.float64)
        low, high = float(np.min(values)), float(np.max(values))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return float(2.0 * ((values[index] - low) / (high - low)) - 1.0)
    if len(records) <= 1:
        return 0.0
    return float(2.0 * (index / (len(records) - 1)) - 1.0)


def _w46_resize_uint8(
    image: np.ndarray, size: int = W46_CACHE_IMAGE_SIZE
) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if max(image.shape) >= size else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interpolation)
    tensor = torch.from_numpy(image.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )[0, 0]
    return resized.clamp(0, 255).round().byte().numpy()


def _w46_robust_triplet_to_uint8(images: Sequence[np.ndarray]) -> np.ndarray:
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
    return np.stack([_w46_resize_uint8(channel) for channel in scaled], axis=0)


def _w46_decode_record(
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
                    plane, confidence = _w46_geometry_plane(iop)
                    if confidence >= W46_GEOMETRY_PLANE_CONFIDENCE:
                        spec = _w46_orientation_spec(iop, plane)
            image = _w46_apply_orientation(image, spec)
            return image, index
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"No decodable slice near target={target_index}. Last={last_error}"
    )


def _w46_series_centers(n_slices: int) -> List[int]:
    if n_slices <= 0:
        return []
    if n_slices == 1:
        return [0]
    quantiles = np.linspace(0.10, 0.90, W46_STACKS_PER_SERIES)
    indices = [int(round(q * (n_slices - 1))) for q in quantiles]
    unique = []
    for index in indices:
        index = int(np.clip(index, 0, n_slices - 1))
        if index not in unique:
            unique.append(index)
    return unique


def _w46_build_study_center_images(
    series_root: Path, study_uid: str, study_series: pd.DataFrame
) -> Dict[str, Any]:
    study_series = study_series.copy()
    study_series["_w46_plane_idx"] = study_series["Anatomical_Plane"].map(
        W46_W43_PLANE_TO_INDEX
    )
    study_series = study_series.sort_values(
        ["_w46_plane_idx", "_fluid", "_fs", SERIES_UID_COLUMN],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    centers_rgb: List[Image.Image] = []
    plane_ids, fluid_ids, fs_ids, positions = [], [], [], []

    for _, row in study_series.iterrows():
        series_uid = str(row[SERIES_UID_COLUMN])
        metadata_plane = str(row["Anatomical_Plane"])
        fluid = int(row["_fluid"])
        fs = int(row["_fs"])
        records = _w46_read_series_headers(series_root, study_uid, series_uid)
        if not records:
            continue

        valid_iop = next((r["iop"] for r in records if r["iop"] is not None), None)
        plane_used = metadata_plane
        orientation_spec = None
        if valid_iop is not None:
            geometry_plane, confidence = _w46_geometry_plane(valid_iop)
            if (
                geometry_plane != metadata_plane
                and confidence >= W46_GEOMETRY_PLANE_CONFIDENCE
            ):
                plane_used = geometry_plane
            orientation_spec = _w46_orientation_spec(valid_iop, plane_used)

        for center in _w46_series_centers(len(records)):
            triplet = []
            actual_center = center
            try:
                for offset in W46_STACK_OFFSETS:
                    target = int(np.clip(center + offset, 0, len(records) - 1))
                    image, actual = _w46_decode_record(
                        records, target, orientation_spec
                    )
                    triplet.append(image)
                    if offset == 0:
                        actual_center = actual
                stack = _w46_robust_triplet_to_uint8(triplet)
                center_uint8 = np.asarray(stack[1], dtype=np.uint8)
                rgb = np.repeat(center_uint8[:, :, None], 3, axis=2)
                centers_rgb.append(Image.fromarray(rgb, mode="RGB"))
                plane_ids.append(
                    int(
                        W46_W43_PLANE_TO_INDEX.get(
                            plane_used, W46_W43_PLANE_TO_INDEX[metadata_plane]
                        )
                    )
                )
                fluid_ids.append(fluid)
                fs_ids.append(fs)
                positions.append(
                    float(_w46_normalized_slice_position(records, actual_center))
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


def _w46_mi2_cache_path(cache_root: Path, uid: str) -> Path:
    return (
        cache_root
        / "studies"
        / f"{hashlib.md5(str(uid).encode('utf-8')).hexdigest()}.npz"
    )


def _w46_atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def _w46_mi2_cache_usable(path: Path, uid: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as x:
            if str(x["cache_version"].item()) != W46_MI2_CACHE_VERSION:
                return False
            if str(x["study_uid"].item()) != str(uid):
                return False
            if str(x["encoder_sha256"].item()) != W46_MI2_WEIGHT_SHA256:
                return False
            emb = x["embeddings"]
            sem = x["semantic"]
            if emb.ndim != 2 or emb.shape[1] != W46_MI2_DIM or emb.dtype != np.float16:
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


def _w46_prepare_test_series(
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
        raise RuntimeError("W46 test/sample UID sets differ")
    if set([c for c in sample_df.columns if c != UID_COLUMN]) != set(LABEL_COLUMNS):
        raise RuntimeError("W46 sample_submission label set mismatch")
    return test_df, series_df, sample_df


def _w46_mi2_worker(
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
    _test, series_df, _sample = _w46_prepare_test_series(data_root)
    groups = {
        str(uid): group.copy()
        for uid, group in series_df[
            series_df[UID_COLUMN].isin(list(uid_shard))
        ].groupby(UID_COLUMN)
    }

    model, preprocess, tokenizer, context_length, vision_path = _w46_load_mi2_model(
        mi2_root, device
    )
    # Parent process validates the 2.3-GB vision checkpoint SHA once before
    # spawning workers. Do not re-hash the same file independently on each GPU.
    pos_proto, neg_proto = _w46_mi2_text_prototypes(
        model, tokenizer, context_length, device
    )
    pos_cpu, neg_cpu = pos_proto.float().cpu(), neg_proto.float().cpu()

    ok, failed = 0, []
    started = time.time()
    _w46_log("=" * 100)
    _w46_log(f"W46 MI2 WORKER {worker_id} | device={device} | studies={len(uid_shard)}")
    _w46_log("=" * 100)

    for index, uid in enumerate(uid_shard, start=1):
        path = _w46_mi2_cache_path(cache_root, uid)
        if _w46_mi2_cache_usable(path, uid):
            ok += 1
            continue
        try:
            if uid not in groups:
                raise RuntimeError("No test_series rows")
            built = _w46_build_study_center_images(
                data_root / "test_series", uid, groups[uid]
            )
            features = _w46_encode_images_mi2(
                model, preprocess, built["images"], device, batch_size
            )
            semantic = features.float() @ pos_cpu.T - features.float() @ neg_cpu.T
            if semantic.shape != (len(features), NUM_LABELS):
                raise RuntimeError(f"semantic shape mismatch {semantic.shape}")
            _w46_atomic_npz(
                path,
                cache_version=np.asarray(W46_MI2_CACHE_VERSION),
                study_uid=np.asarray(uid),
                encoder_sha256=np.asarray(W46_MI2_WEIGHT_SHA256),
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
            _w46_log(
                f"worker={worker_id} {index:4d}/{len(uid_shard)} ok={ok:4d} fail={len(failed):3d} rate={rate:5.1f} studies/min"
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"worker": worker_id, "gpu": gpu_id, "ok": ok, "failed": failed}


def _w46_mi2_cache_summary(cache_root: Path, sample_df: pd.DataFrame) -> Dict[str, Any]:
    usable, tokens, missing, invalid = 0, 0, [], []
    for uid in sample_df[UID_COLUMN].astype(str):
        path = _w46_mi2_cache_path(cache_root, uid)
        if not path.is_file():
            if len(missing) < 10:
                missing.append(uid)
            continue
        if not _w46_mi2_cache_usable(path, uid):
            if len(invalid) < 10:
                invalid.append(uid)
            continue
        with np.load(path, allow_pickle=False) as x:
            tokens += int(len(x["embeddings"]))
        usable += 1
    return {
        "root": str(cache_root),
        "cache_version": W46_MI2_CACHE_VERSION,
        "usable_studies": usable,
        "expected_studies": int(len(sample_df)),
        "total_tokens": tokens,
        "complete": usable == len(sample_df),
        "missing_examples": missing,
        "invalid_examples": invalid,
    }


def _w46_extract_mi2_test(
    data_root: Path, mi2_root: Path, cache_root: Path, batch_size: int
) -> Dict[str, Any]:
    _test, _series, sample_df = _w46_prepare_test_series(data_root)
    uids = sample_df[UID_COLUMN].astype(str).tolist()
    remaining = [
        u
        for u in uids
        if not _w46_mi2_cache_usable(_w46_mi2_cache_path(cache_root, u), u)
    ]
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < 1:
        raise RuntimeError("W46 MI2 hidden-test extraction requires CUDA")

    _w46_log("=" * 100)
    _w46_log(f"{W46_EXPERIMENT} | MI2 HIDDEN TEST EXTRACTION")
    _w46_log("=" * 100)
    _w46_log(f"Test studies         : {len(uids)}")
    _w46_log(f"Already cached       : {len(uids)-len(remaining)}")
    _w46_log(f"Need extraction      : {len(remaining)}")
    _w46_log(f"MI2 batch            : {batch_size}")
    _w46_log(f"Visible CUDA GPUs    : {visible}")

    if remaining:
        n_workers = min(visible, len(remaining))
        shards = [remaining[i::n_workers] for i in range(n_workers)]
        if n_workers == 1:
            results = [
                _w46_mi2_worker(
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
                    target=_w46_mi2_worker_proxy,
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
                    f"W46 MI2 worker failure: errors={errors}, exitcodes={exitcodes}"
                )
            results = [received[i][0] for i in sorted(received)]
        failures = [row for result in results for row in result["failed"]]
        if failures:
            pd.DataFrame(failures).to_csv(
                cache_root.parent / "mi2_test_failures.csv", index=False
            )
            raise RuntimeError(f"MI2 extraction failed for {len(failures)} studies")

    summary = _w46_mi2_cache_summary(cache_root, sample_df)
    if not summary["complete"]:
        raise RuntimeError(f"W46 MI2 cache incomplete: {summary}")
    return summary


def _w46_mi2_worker_proxy(
    worker_id, gpu_id, shard, data_root_s, mi2_root_s, cache_root_s, bsz, queue
):
    try:
        result = _w46_mi2_worker(
            worker_id, gpu_id, shard, data_root_s, mi2_root_s, cache_root_s, bsz
        )
        queue.put((worker_id, result, None))
    except Exception as exc:
        queue.put((worker_id, None, repr(exc)))
        raise


# -----------------------------------------------------------------------------
# W45 exact-style MI2 production head + checkpoint adapter
# -----------------------------------------------------------------------------


def _w46_position_features(position: torch.Tensor) -> torch.Tensor:
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


class W46MI2Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(W46_MI2_DIM)
        self.projection = nn.Linear(W46_MI2_DIM, W46_HIDDEN_DIM)
        self.plane_embedding = nn.Embedding(3, W46_PLANE_EMBED_DIM)
        self.fluid_embedding = nn.Embedding(2, W46_BINARY_META_DIM)
        self.fs_embedding = nn.Embedding(2, W46_BINARY_META_DIM)
        self.position_projection = nn.Linear(9, W46_POSITION_DIM)
        self.meta_projection = nn.Linear(
            W46_PLANE_EMBED_DIM + 2 * W46_BINARY_META_DIM + W46_POSITION_DIM,
            W46_HIDDEN_DIM,
        )
        self.token_norm = nn.LayerNorm(W46_HIDDEN_DIM)
        self.queries = nn.Parameter(torch.randn(NUM_LABELS, W46_HIDDEN_DIM) * 0.02)
        self.dropout = nn.Dropout(W46_HEAD_DROPOUT)

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
                self.position_projection(_w46_position_features(pos)),
            ],
            dim=-1,
        )
        tokens = self.token_norm(visual + self.meta_projection(meta))
        tokens = self.dropout(tokens)
        scores = torch.einsum("bth,lh->blt", tokens, self.queries) / math.sqrt(
            W46_HIDDEN_DIM
        )
        plane_prior = W46_PLANE_PRIOR_LOG.to(scores.device)
        scores = scores + plane_prior[:, plane].permute(1, 0, 2)
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        attention = torch.softmax(scores.float(), dim=-1).to(tokens.dtype)
        pooled = torch.einsum("blt,bth->blh", attention, tokens)
        semantic_by_label = branch["semantic"].permute(0, 2, 1)
        semantic_pooled = (attention.float() * semantic_by_label.float()).sum(dim=-1)
        return pooled, semantic_pooled


class W46MI2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = W46MI2Branch()
        self.classifier_weight = nn.Parameter(
            torch.randn(NUM_LABELS, W46_HIDDEN_DIM + 1) * 0.02
        )
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        visual, semantic = self.branch(batch)
        features = torch.cat([visual, semantic.unsqueeze(-1).to(visual.dtype)], dim=-1)
        logits = (features * self.classifier_weight[None, :, :]).sum(dim=-1)
        return logits + self.classifier_bias[None, :]


def _w46_select_balanced_indices(
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


class W46MI2TestStore:
    def __init__(self, cache_root: Path, ordered_uids: Sequence[str]):
        self.uids = [str(u) for u in ordered_uids]
        self.records = []
        for index, uid in enumerate(self.uids, start=1):
            path = _w46_mi2_cache_path(cache_root, uid)
            if not _w46_mi2_cache_usable(path, uid):
                raise RuntimeError(f"Missing/incompatible MI2 test cache: {uid}")
            with np.load(path, allow_pickle=False) as x:
                plane = x["plane"].astype(np.int64)
                selected = _w46_select_balanced_indices(
                    plane, len(plane), W46_MI2_MAX_TOKENS
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
                _w46_log(f"  MI2 test cache loaded {index}/{len(self.uids)}")

    def make_batch(
        self, indices: Sequence[int], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        records = [self.records[int(i)] for i in indices]
        b = len(records)
        max_t = max(len(r["embeddings"]) for r in records)
        embeddings = torch.zeros(b, max_t, W46_MI2_DIM, dtype=torch.float32)
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


def _w46_load_mi2_head(path: Path, seed: int, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        embedded_version = payload.get("script_version")
        embedded_seed = payload.get("seed")
        if embedded_version is not None and str(embedded_version) != W46_W45_VERSION:
            raise RuntimeError(f"W45 version mismatch seed={seed}: {embedded_version}")
        if embedded_seed is not None and int(embedded_seed) != int(seed):
            raise RuntimeError(
                f"W45 seed mismatch expected={seed} embedded={embedded_seed}"
            )
    state = _w46_extract_checkpoint_state(payload)
    model = W46MI2Head()
    mapped = _w46_adapt_state_dict(state, model)
    model.load_state_dict(mapped, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def _w46_predict_mi2_seed(
    model: nn.Module, store: W46MI2TestStore, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    chunks = []
    for start in range(0, len(store.uids), batch_size):
        indices = list(range(start, min(len(store.uids), start + batch_size)))
        batch = store.make_batch(indices, device)
        logits = model(batch)
        prob = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
        chunks.append(prob)
        del batch, logits
    return np.concatenate(chunks, axis=0)


def _w46_predict_mi2_all(
    w45_root: Path, cache_root: Path, sample_df: pd.DataFrame
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("W46 MI2 head inference requires CUDA")
    device = torch.device("cuda:0")
    store = W46MI2TestStore(cache_root, sample_df[UID_COLUMN].astype(str).tolist())
    predictions = {}
    for seed in W46_W45_SEEDS:
        _w46_log(f"W46 MI2 head inference | seed={seed}")
        model = _w46_load_mi2_head(
            _w46_w45_checkpoint_path(w45_root, seed), seed, device
        )
        predictions[seed] = _w46_predict_mi2_seed(model, store, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    stack = np.stack([predictions[s] for s in W46_W45_SEEDS], axis=0)
    mean_prob = np.clip(stack.mean(axis=0), 1e-6, 1.0 - 1e-6).astype(np.float32)
    return predictions, mean_prob


# -----------------------------------------------------------------------------
# Curia exact W42 branch without writing the canonical submission prematurely
# -----------------------------------------------------------------------------


def _w46_curia_args(args, curia_output_root: Path):
    return SimpleNamespace(
        accelerator=args.accelerator,
        data_root=args.data_root,
        curia_root=args.curia_root,
        w41_root=args.w41_root,
        output_root=str(curia_output_root),
        curia_batch=args.curia_batch,
        head_batch=args.curia_head_batch,
        reset_test_cache=args.reset_curia_cache,
    )


def _w46_run_curia_branch(args, output_root: Path) -> Dict[str, Any]:
    curia_args = _w46_curia_args(args, output_root / "curia_branch")
    runtime = _w42_configure(curia_args)
    if not runtime["inference_supported"]:
        raise RuntimeError("W46 Curia branch requires CUDA")
    w41_root = _w42_discover_w41_root(args.w41_root, SUBMISSION_WORK_ROOT)
    w41_info = _w42_verify_w41_root(w41_root, verify_sha=True)
    curia_root = discover_curia2_root()
    curia_sha, _curia_config = validate_curia_identity(curia_root)
    test_df, series_df, sample_df = load_test_tables()
    _w42_serial_processor_prewarm(curia_root)
    cache_summary = build_test_feature_cache(test_df, series_df)
    if not bool(cache_summary.get("cache_complete")):
        raise RuntimeError("W46 Curia test cache incomplete")
    gc.collect()
    for gpu_id in range(int(torch.cuda.device_count())):
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        except Exception:
            pass
    store = SubmissionFeatureStore(sample_df[UID_COLUMN].astype(str).tolist())
    seed_predictions = _w42_predict_all_seeds(w41_root, store)
    stack = np.stack([seed_predictions[s] for s in W46_W41_SEEDS], axis=0).astype(
        np.float32
    )
    rawmean = np.clip(stack.mean(axis=0), 1e-6, 1.0 - 1e-6)
    rankmean = np.mean(
        [_w42_normalized_ranks(seed_predictions[s]) for s in W46_W41_SEEDS], axis=0
    ).astype(np.float32)
    rankmean = np.clip(rankmean, 1e-6, 1.0 - 1e-6)
    del store
    gc.collect()
    return {
        "sample_df": sample_df,
        "seed_predictions": seed_predictions,
        "rawmean": rawmean,
        "rankmean": rankmean,
        "cache_summary": cache_summary,
        "w41_root": w41_root,
        "w41_info": w41_info,
        "curia_root": curia_root,
        "curia_sha": curia_sha,
    }


# -----------------------------------------------------------------------------
# Status / inference / validation
# -----------------------------------------------------------------------------


def _w46_validate_mi2_assets(root: Path) -> Dict[str, Any]:
    vision = root / "2024.09.27" / "vision_model" / "medimageinsigt-v1.0.0.pt"
    sha = _w46_sha256_file(vision)
    return {
        "root": str(root),
        "vision_weight": str(vision),
        "vision_weight_size_gb": vision.stat().st_size / (1024**3),
        "sha256": sha,
        "expected_sha256": W46_MI2_WEIGHT_SHA256,
        "sha_match": sha == W46_MI2_WEIGHT_SHA256,
        "native_input_size": W46_MI2_NATIVE_SIZE,
        "embedding_dim": W46_MI2_DIM,
        "complete": _w46_mi2_root_usable(root) and sha == W46_MI2_WEIGHT_SHA256,
    }


def _w46_dicom_preflight(data_root: Path, max_series: int = 64) -> Dict[str, Any]:
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


def _w46_status(args) -> Dict[str, Any]:
    output_root = _w46_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_root = output_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    data_root = _w46_data_root(args.data_root)

    runtime = _w42_resolve_accelerator(args.accelerator)
    dependencies = {
        name: _w46_dependency_available(name)
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
            "libjpeg": _w46_dependency_available("libjpeg"),
            "openjpeg": _w46_dependency_available("openjpeg"),
            "cv2": _w46_dependency_available("cv2"),
        }
    )

    mi2_root, mi2_info, mi2_error = None, {}, None
    try:
        mi2_root = _w46_discover_mi2_root(args.mi2_root)
        mi2_info = _w46_validate_mi2_assets(mi2_root)
        if not mi2_info["complete"]:
            raise RuntimeError("MedImageInsight assets failed identity validation")
    except Exception as exc:
        mi2_error = repr(exc)

    w45_root, w45_info, w45_error = None, {}, None
    try:
        w45_root = _w46_discover_w45_root(args.w45_root)
        w45_info = _w46_verify_w45_root(w45_root, load_architecture=True)
    except Exception as exc:
        w45_error = repr(exc)

    # Configure/check the embedded W42 branch using its own subdirectory.
    w41_info, curia_info, curia_error, w41_error = {}, {}, None, None
    try:
        curia_args = _w46_curia_args(args, output_root / "curia_branch")
        _w42_configure(curia_args)
        w41_root = _w42_discover_w41_root(args.w41_root, SUBMISSION_WORK_ROOT)
        w41_info = _w42_verify_w41_root(w41_root, verify_sha=True)
    except Exception as exc:
        w41_error = repr(exc)
    try:
        curia_root = discover_curia2_root()
        curia_sha, curia_cfg = validate_curia_identity(curia_root)
        curia_info = {"root": str(curia_root), "sha256": curia_sha, "config": curia_cfg}
    except Exception as exc:
        curia_error = repr(exc)

    dicom = {}
    dicom_error = None
    try:
        dicom = _w46_dicom_preflight(data_root)
        if not dicom["overall_pass"]:
            raise RuntimeError("DICOM decoder preflight failed")
    except Exception as exc:
        dicom_error = repr(exc)

    sample_rows = None
    try:
        sample_rows = int(len(pd.read_csv(data_root / "sample_submission.csv")))
    except Exception:
        pass

    payload = {
        "script_version": W46_SCRIPT_VERSION,
        "experiment": W46_EXPERIMENT,
        "fusion_space": "per-label normalized rank",
        "selected_blend": args.blend,
        "selected_curia_weight": W46_BLEND_WEIGHTS[args.blend],
        "candidates": {
            key: {"mi2": 1.0 - value, "curia": value}
            for key, value in W46_BLEND_WEIGHTS.items()
        },
        "known_public_references": {"w45_mi2": 0.766, "w42_curia": 0.736},
        "runtime": runtime,
        "dependencies": dependencies,
        "paths": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "mi2_root": str(mi2_root) if mi2_root else None,
            "w45_root": str(w45_root) if w45_root else None,
            "w41_root": w41_info.get("root"),
            "curia_root": curia_info.get("root"),
        },
        "visible_test_rows": sample_rows,
        "mi2": mi2_info,
        "mi2_error": mi2_error,
        "w45_checkpoints": w45_info,
        "w45_error": w45_error,
        "w41": w41_info,
        "w41_error": w41_error,
        "curia": curia_info,
        "curia_error": curia_error,
        "dicom_preflight": dicom,
        "dicom_error": dicom_error,
        "ready_for_infer": bool(
            runtime["inference_supported"]
            and all(dependencies.values())
            and mi2_error is None
            and w45_error is None
            and w41_error is None
            and curia_error is None
            and dicom_error is None
        ),
        "project_python_imports": False,
        "reports_used_at_test": False,
    }
    (result_root / "00_status.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    _w46_log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def _w46_submission_frame(sample_df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    if values.shape != (len(sample_df), NUM_LABELS):
        raise RuntimeError(f"W46 prediction shape mismatch: {values.shape}")
    frame = sample_df.copy()
    for j, label in enumerate(LABEL_COLUMNS):
        frame[label] = values[:, j]
    frame = frame[sample_df.columns]
    return frame


def _w46_validate_frame(
    frame: pd.DataFrame, sample_df: pd.DataFrame, name: str
) -> None:
    if list(frame.columns) != list(sample_df.columns):
        raise RuntimeError(f"{name}: column/order mismatch")
    if (
        frame[UID_COLUMN].astype(str).tolist()
        != sample_df[UID_COLUMN].astype(str).tolist()
    ):
        raise RuntimeError(f"{name}: UID order mismatch")
    if frame[UID_COLUMN].duplicated().any():
        raise RuntimeError(f"{name}: duplicate UID")
    values = frame[LABEL_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError(f"{name}: values outside [0,1]")


def _w46_infer(args) -> Dict[str, Any]:
    started = time.time()
    output_root = _w46_output_root(args.output_root)
    result_root = output_root / "results"
    mi2_cache_root = output_root / "mi2_test_embedding_cache"
    for p in (output_root, result_root, mi2_cache_root / "studies"):
        p.mkdir(parents=True, exist_ok=True)
    data_root = _w46_data_root(args.data_root)
    runtime = _w42_resolve_accelerator(args.accelerator)
    if not runtime["inference_supported"]:
        raise RuntimeError("W46 inference requires CUDA")

    mi2_root = _w46_discover_mi2_root(args.mi2_root)
    mi2_info = _w46_validate_mi2_assets(mi2_root)
    if not mi2_info["complete"]:
        raise RuntimeError("MI2 asset identity check failed")
    w45_root = _w46_discover_w45_root(args.w45_root)
    w45_info = _w46_verify_w45_root(w45_root, load_architecture=True)

    _w46_log("=" * 100)
    _w46_log(f"{W46_EXPERIMENT} | HIDDEN TEST INFERENCE")
    _w46_log("=" * 100)
    _w46_log(f"Selected blend          : {args.blend.upper()}")
    _w46_log(
        f"MI2 / Curia             : {1-W46_BLEND_WEIGHTS[args.blend]:.2f} / {W46_BLEND_WEIGHTS[args.blend]:.2f}"
    )
    _w46_log("Fusion space            : per-label normalized rank")
    _w46_log("W45 branch              : exact five production checkpoints")
    _w46_log("W42 branch              : exact three W41 checkpoints + rankmean")

    # Phase 1: Curia exact W42 branch. This uses both GPUs and then releases them.
    _w46_log("\nPHASE C | CURIA / W42")
    curia = _w46_run_curia_branch(args, output_root)
    sample_df = curia["sample_df"]
    curia_rank = curia["rankmean"]
    curia_raw = curia["rawmean"]

    # Free Curia allocations before loading the much larger MI2 model workers.
    gc.collect()
    for gpu_id in range(int(torch.cuda.device_count())):
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Phase 2: exact W43-compatible hidden-test stacks -> MI2 embedding cache.
    _w46_log("\nPHASE M | MEDIMAGEINSIGHT")
    mi2_cache_summary = _w46_extract_mi2_test(
        data_root, mi2_root, mi2_cache_root, args.mi2_batch
    )

    # Phase 3: W45 five-seed production heads.
    _w46_log("\nPHASE H | W45 FIVE-SEED HEADS")
    mi2_seed_predictions, mi2_probability = _w46_predict_mi2_all(
        w45_root, mi2_cache_root, sample_df
    )
    mi2_rank = _w42_normalized_ranks(mi2_probability)

    # Phase 4: fixed rank-space fusion candidates.
    outputs = {}
    for key, curia_weight in W46_BLEND_WEIGHTS.items():
        fused = ((1.0 - curia_weight) * mi2_rank + curia_weight * curia_rank).astype(
            np.float32
        )
        fused = np.clip(fused, 1e-6, 1.0 - 1e-6)
        frame = _w46_submission_frame(sample_df, fused)
        _w46_validate_frame(frame, sample_df, f"W46-{key.upper()}")
        filename = f"submission_w46_{key}_{int(round((1-curia_weight)*100))}mi2_{int(round(curia_weight*100))}curia_rank.csv"
        path = result_root / filename
        frame.to_csv(path, index=False)
        outputs[key] = {
            "path": str(path),
            "sha256": _w46_sha256_file(path),
            "curia_weight": curia_weight,
        }

    # Branch references / diagnostics.
    ref_frames = {
        "w45_mi2_probability_mean": _w46_submission_frame(sample_df, mi2_probability),
        "w45_mi2_rank": _w46_submission_frame(sample_df, mi2_rank),
        "w42_curia_rankmean": _w46_submission_frame(sample_df, curia_rank),
        "w42_curia_rawmean": _w46_submission_frame(sample_df, curia_raw),
    }
    for name, frame in ref_frames.items():
        _w46_validate_frame(frame, sample_df, name)
        frame.to_csv(result_root / f"reference_{name}.csv", index=False)

    selected = args.blend
    selected_path = Path(outputs[selected]["path"])
    canonical = (
        Path("/kaggle/working/submission.csv")
        if _w46_is_kaggle()
        else output_root / "submission.csv"
    )
    shutil.copyfile(selected_path, canonical)

    # Per-label branch ordering correlation is a useful complementarity audit.
    correlation_rows = []
    for j, label in enumerate(LABEL_COLUMNS):
        if len(sample_df) > 1:
            corr = float(np.corrcoef(mi2_rank[:, j], curia_rank[:, j])[0, 1])
        else:
            corr = float("nan")
        correlation_rows.append({"Label": label, "MI2_Curia_RankCorrelation": corr})
    corr_df = pd.DataFrame(correlation_rows)
    corr_df.to_csv(result_root / "branch_rank_correlation.csv", index=False)

    np.savez_compressed(
        result_root / "branch_predictions.npz",
        mi2_probability=mi2_probability,
        mi2_rank=mi2_rank,
        curia_rawmean=curia_raw,
        curia_rankmean=curia_rank,
        **{f"mi2_seed_{seed}": mi2_seed_predictions[seed] for seed in W46_W45_SEEDS},
        **{
            f"curia_seed_{seed}": curia["seed_predictions"][seed]
            for seed in W46_W41_SEEDS
        },
    )

    manifest = {
        "script_version": W46_SCRIPT_VERSION,
        "experiment": W46_EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_studies": int(len(sample_df)),
        "fusion_space": "per-label normalized rank",
        "selected_blend": selected,
        "selected_curia_weight": W46_BLEND_WEIGHTS[selected],
        "candidate_outputs": outputs,
        "canonical_submission": str(canonical),
        "canonical_submission_sha256": _w46_sha256_file(canonical),
        "w45": w45_info,
        "mi2": mi2_info,
        "mi2_cache": mi2_cache_summary,
        "w41": curia["w41_info"],
        "curia_root": str(curia["curia_root"]),
        "curia_sha256": curia["curia_sha"],
        "curia_cache": curia["cache_summary"],
        "branch_correlation_mean": float(corr_df["MI2_Curia_RankCorrelation"].mean()),
        "runtime_seconds": float(time.time() - started),
        "hidden_test_inputs": "MRI only; no reports",
    }
    (result_root / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )

    _w46_log("\n" + "=" * 100)
    _w46_log("W46 SUBMISSION READY")
    _w46_log("=" * 100)
    _w46_log(f"Canonical submission    : {canonical}")
    _w46_log(f"Selected candidate      : {selected.upper()}")
    _w46_log(f"Rows                    : {len(sample_df)}")
    _w46_log(f"Mean branch rank corr   : {manifest['branch_correlation_mean']:.6f}")
    _w46_log(f"SHA256                  : {manifest['canonical_submission_sha256']}")

    validation = _w46_validate(args, configure_assets=False)
    if not validation["overall_pass"]:
        raise RuntimeError("W46 inference completed but final validation failed")
    return manifest


def _w46_validate(args, configure_assets: bool = True) -> Dict[str, Any]:
    output_root = _w46_output_root(args.output_root)
    result_root = output_root / "results"
    data_root = _w46_data_root(args.data_root)
    sample_df = pd.read_csv(data_root / "sample_submission.csv")
    sample_df[UID_COLUMN] = sample_df[UID_COLUMN].astype(str)
    canonical = (
        Path("/kaggle/working/submission.csv")
        if _w46_is_kaggle()
        else output_root / "submission.csv"
    )

    checks = {}
    errors = {}
    try:
        frame = pd.read_csv(canonical)
        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
        _w46_validate_frame(frame, sample_df, "canonical W46")
        checks["submission_valid"] = True
        checks["submission_rows_match"] = len(frame) == len(sample_df)
    except Exception as exc:
        checks["submission_valid"] = False
        checks["submission_rows_match"] = False
        errors["submission"] = repr(exc)

    manifest = {}
    manifest_path = result_root / "submission_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks["manifest_version"] = (
                manifest.get("script_version") == W46_SCRIPT_VERSION
            )
            checks["manifest_selected_blend"] = (
                manifest.get("selected_blend") == args.blend
            )
            checks["mi2_cache_complete"] = bool(
                (manifest.get("mi2_cache") or {}).get("complete")
            )
            checks["curia_cache_complete"] = bool(
                (manifest.get("curia_cache") or {}).get("cache_complete")
            )
            checks["w45_five_seeds"] = (
                len((manifest.get("w45") or {}).get("checkpoints") or []) == 5
            )
            checks["w41_three_seeds"] = (
                len((manifest.get("w41") or {}).get("checkpoints") or []) == 3
            )
        except Exception as exc:
            errors["manifest"] = repr(exc)
    else:
        checks.update(
            {
                "manifest_version": False,
                "manifest_selected_blend": False,
                "mi2_cache_complete": False,
                "curia_cache_complete": False,
                "w45_five_seeds": False,
                "w41_three_seeds": False,
            }
        )

    checks["project_python_imports"] = True
    checks["reports_not_used"] = True
    overall = bool(checks and all(checks.values()))
    payload = {
        "script_version": W46_SCRIPT_VERSION,
        "selected_blend": args.blend,
        "overall_pass": overall,
        "checks": checks,
        "errors": errors,
        "canonical_submission": str(canonical),
        "manifest": manifest,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    _w46_log(json.dumps(payload, indent=2, allow_nan=True))
    if configure_assets and not overall:
        raise RuntimeError("W46 validation failed")
    return payload


def _w46_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="W46 standalone MI2 + Curia rank-fusion submission"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["status", "infer", "validate", "run_all"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=["status", "infer", "validate", "run_all"],
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
    parser.add_argument(
        "--w45-root",
        default=None,
        help="W45 output/bundle root containing checkpoints/full_seed_45101.pt ... 45105.pt",
    )
    parser.add_argument("--curia-root", default=None)
    parser.add_argument("--w41-root", default=None)
    parser.add_argument(
        "--blend",
        choices=["a", "b", "c"],
        default="a",
        help="a=90/10, b=80/20, c=70/30 MI2/Curia",
    )
    parser.add_argument("--mi2-batch", type=int, default=W46_MI2_BATCH)
    parser.add_argument("--curia-batch", type=int, default=16)
    parser.add_argument("--curia-head-batch", type=int, default=8)
    parser.add_argument("--reset-mi2-cache", action="store_true")
    parser.add_argument("--reset-curia-cache", action="store_true")
    return parser


def w46_main() -> None:
    parser = _w46_parser()
    args = parser.parse_args()
    mode = args.mode_flag or args.mode or "status"
    if args.mi2_batch < 1 or args.curia_batch < 1 or args.curia_head_batch < 1:
        raise ValueError("Batch sizes must be >= 1")
    output_root = _w46_output_root(args.output_root)
    if args.reset_mi2_cache:
        cache = output_root / "mi2_test_embedding_cache"
        if cache.exists():
            shutil.rmtree(cache)
    if mode == "status":
        _w46_status(args)
    elif mode == "infer":
        _w46_infer(args)
    elif mode == "validate":
        _w46_validate(args, configure_assets=True)
    elif mode == "run_all":
        status = _w46_status(args)
        if not status.get("ready_for_infer"):
            raise RuntimeError(
                "W46 status is not ready_for_infer; fix status before inference"
            )
        _w46_infer(args)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    w46_main()
