#!/usr/bin/env python3
# ============================================================
# RSNA Knee Abnormality Detection — W5.0
# ORTHODIFFUSION 3-D SPATIAL HIERARCHICAL MODEL
# ============================================================
#
# PURPOSE
# -------
# W5.0 is the first standalone OrthoDiffusion competition experiment.
# It is deliberately designed to answer one question before any
# Curia+Ortho mixture-of-experts work:
#
#     Does a knee-specific 3-D OrthoDiffusion representation provide
#     strong and complementary signal relative to W4 Curia-2?
#
# Representation:
#   * official sagittal/coronal/axial OrthoDiffusion backbones
#   * official 16 x 256 x 256 volume convention
#   * official t=100 / mid_2 feature point validated by preflight
#   * up to three 16-slice windows per qualifying series
#   * full [256,16,16] spatial feature map retained — NOT global pooled
#   * deterministic per-window diffusion noise for a reproducible cache
#
# Study head:
#   spatial attention -> window attention -> series attention -> 12 logits
#   with diagnosis-specific queries at every aggregation level.
#
# Supervision:
#   1. gold_only       : same 58 expert-labelled studies / same 5 folds
#   2. fold_safe_weak  : same current W2.3 continuous fold-safe targets
#
# W2.3 is intentionally NOT redesigned in W5.0.  This isolates the
# representation/architecture change.  A supervision update can follow
# once W5.0 tells us whether Ortho is worth carrying forward.
#
# Modes:
#   run_w50_ortho("status")
#   run_w50_ortho("cache")
#   run_w50_ortho("train_gold")
#   run_w50_ortho("train_weak")
#   run_w50_ortho("train")
#   run_w50_ortho("compare_w4")
#   run_w50_ortho("all")
#
# Recommended first run:
#   run_w50_ortho("status")
#   run_w50_ortho("cache")
#   # inspect cache summary, then:
#   run_w50_ortho("train")
#
# Required external resources:
#   W50_ORTHO_CODE_ROOT    -> official lt-0123/OrthoDiffusion source
#   W50_ORTHO_WEIGHTS_ROOT -> sagittal_model.pt / coronal_model.pt /
#                              axial_model.pt
#
# Optional but required for weak training:
#   W50_W23_ROOT           -> rsna_w2_3 root
#
# Optional for automatic Curia comparison:
#   W50_W4_ROOT            -> full rsna_w4_0_curia2 training output root
#
# ============================================================

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "W50_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"

WORK_ROOT = Path(
    os.environ.get(
        "W50_WORK_ROOT",
        "/kaggle/working/rsna_w5_0_orthodiffusion",
    )
)

CACHE_ROOT = WORK_ROOT / "feature_cache"
CACHE_STUDY_ROOT = CACHE_ROOT / "studies"
RESULT_ROOT = WORK_ROOT / "results"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

for _path in (
    WORK_ROOT,
    CACHE_ROOT,
    CACHE_STUDY_ROOT,
    RESULT_ROOT,
    CHECKPOINT_ROOT,
):
    _path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# OrthoDiffusion identity
# ------------------------------------------------------------
# EXPLICIT_CODE_ROOT = os.environ.get("W50_ORTHO_CODE_ROOT", "/kaggle/input/datasets/isayem/orthodiffusion-official/OrthoDiffusion-Official/OrthoDiffusion").strip()
# EXPLICIT_WEIGHTS_ROOT = os.environ.get("W50_ORTHO_WEIGHTS_ROOT", "/kaggle/input/datasets/isayem/orthodiffusion-official/OrthoDiffusion-Official/weights").strip()
EXPLICIT_ORTHO_CODE_ROOT = os.environ.get(
    "W50_ORTHO_CODE_ROOT",
    "/kaggle/input/datasets/isayem/orthodiffusion-official/OrthoDiffusion-Official/OrthoDiffusion",
).strip()

EXPLICIT_ORTHO_WEIGHTS_ROOT = os.environ.get(
    "W50_ORTHO_WEIGHTS_ROOT",
    "/kaggle/input/datasets/isayem/orthodiffusion-official/OrthoDiffusion-Official/weights",
).strip()

EXPECTED_ORTHO_GIT_HEAD = "99d15186bb8face1728263023d8414d9405c8f7e"

WEIGHT_FILENAMES = {
    "Sagittal": "sagittal_model.pt",
    "Coronal": "coronal_model.pt",
    "Axial": "axial_model.pt",
}

EXPECTED_WEIGHT_SHA256 = {
    "Sagittal": "3afc323c58b6446c1ea6dc4f09133d744e5a1609dba7662ac9a15ca688e34951",
    "Coronal": "dcce97a061af1a6df1f902cc0d5a7f604dc9bd0f52f3b3714be39b288dc541e6",
    "Axial": "f7daa0406c6312dacebf1e56ecfdf14b1f3c98c59b24f2fa75d1899490d31a3e",
}

ALLOW_ORTHO_SHA_MISMATCH = os.environ.get(
    "W50_ALLOW_ORTHO_SHA_MISMATCH",
    "0",
).strip().lower() in {"1", "true", "yes"}

ALLOW_ORTHO_CODE_MISMATCH = os.environ.get(
    "W50_ALLOW_ORTHO_CODE_MISMATCH",
    "0",
).strip().lower() in {"1", "true", "yes"}

ORTHO_INPUT_SIZE = 256
ORTHO_DEPTH_SIZE = 16
ORTHO_IN_CHANNELS = 1
ORTHO_OUT_CHANNELS = 1
ORTHO_BASE_CHANNELS = 64
ORTHO_NUM_RES_BLOCKS = 1
ORTHO_TIMESTEPS = 1000

# W5.0 defaults are the exact feature point successfully tested in the
# technical preflight.  Environment overrides exist for a future controlled
# experiment, but this W5.0 head is intentionally locked to the validated
# [256,1,16,16] mid_2 feature shape.
FEATURE_TIMESTEP_BY_PLANE = {
    "Sagittal": int(os.environ.get("W50_SAGITTAL_TIMESTEP", "100")),
    "Coronal": int(os.environ.get("W50_CORONAL_TIMESTEP", "100")),
    "Axial": int(os.environ.get("W50_AXIAL_TIMESTEP", "100")),
}

FEATURE_BLOCK_BY_PLANE = {
    "Sagittal": os.environ.get("W50_SAGITTAL_BLOCK", "mid_2").strip(),
    "Coronal": os.environ.get("W50_CORONAL_BLOCK", "mid_2").strip(),
    "Axial": os.environ.get("W50_AXIAL_BLOCK", "mid_2").strip(),
}

ORTHO_FEATURE_CHANNELS = 256
ORTHO_FEATURE_HEIGHT = 16
ORTHO_FEATURE_WIDTH = 16

# Empirically measured in the user's T4 x2 preflight.  Larger batches did
# not materially improve throughput and consume more VRAM.
ORTHO_BATCH_BY_PLANE = {
    "Sagittal": int(os.environ.get("W50_SAGITTAL_BATCH", "1")),
    "Coronal": int(os.environ.get("W50_CORONAL_BATCH", "2")),
    "Axial": int(os.environ.get("W50_AXIAL_BATCH", "2")),
}

PLANES = ("Sagittal", "Coronal", "Axial")
PLANE_TO_INDEX = {
    "Sagittal": 0,
    "Coronal": 1,
    "Axial": 2,
}

MIN_SLICES_PER_SERIES = ORTHO_DEPTH_SIZE
MAX_WINDOWS_PER_SERIES = int(os.environ.get("W50_MAX_WINDOWS_PER_SERIES", "3"))

WINDOW_CENTER_FRACTIONS = tuple(
    float(x)
    for x in os.environ.get(
        "W50_WINDOW_CENTER_FRACTIONS",
        "0.25,0.50,0.75",
    ).split(",")
    if x.strip()
)

if MAX_WINDOWS_PER_SERIES < 1:
    raise ValueError("W50_MAX_WINDOWS_PER_SERIES must be >= 1")

if not WINDOW_CENTER_FRACTIONS:
    raise ValueError("At least one window-center fraction is required")

if any(x < 0.0 or x > 1.0 for x in WINDOW_CENTER_FRACTIONS):
    raise ValueError("Window-center fractions must lie in [0,1]")

WINDOW_CENTER_FRACTIONS = WINDOW_CENTER_FRACTIONS[:MAX_WINDOWS_PER_SERIES]

GEOMETRY_PLANE_CONFIDENCE = float(
    os.environ.get("W50_GEOMETRY_PLANE_CONFIDENCE", "0.80")
)

CACHE_VERSION = (
    "w5_0_orthodiffusion_t100_mid2_spatial_" "three_window_deterministic_noise_fp16_v1"
)

RESET_FEATURE_CACHE = os.environ.get(
    "W50_RESET_CACHE",
    "0",
).strip().lower() in {"1", "true", "yes"}


# ------------------------------------------------------------
# W2.3 — unchanged W5.0 supervision reference
# ------------------------------------------------------------

EXPLICIT_W23_ROOT = os.environ.get(
    "W50_W23_ROOT",
    "/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3",
).strip()

SELECTION_THRESHOLD = float(os.environ.get("W50_SELECTION_THRESHOLD", "0.50"))

GOLD_AUTHORITY = float(os.environ.get("W50_GOLD_AUTHORITY", "8.0"))

PSEUDO_AUTHORITY = float(os.environ.get("W50_PSEUDO_AUTHORITY", "1.0"))


# ------------------------------------------------------------
# Optional W4 Curia OOF reference
# ------------------------------------------------------------

EXPLICIT_W4_ROOT = os.environ.get(
    "W50_W4_ROOT",
    "/kaggle/input/datasets/isayem/rsna-w4-0-curia2",
).strip()

W4_WEAK_REFERENCE_AUROC = 0.6410767022764444
W4_GOLD_REFERENCE_AUROC = 0.6206400750390512
W4_PUBLIC_WEAK_LB = 0.721
W4_PUBLIC_GATED_LB = 0.717


# ------------------------------------------------------------
# Spatial hierarchical head
# ------------------------------------------------------------

HEAD_HIDDEN_DIM = int(os.environ.get("W50_HEAD_HIDDEN_DIM", "192"))

HEAD_DROPOUT = float(os.environ.get("W50_HEAD_DROPOUT", "0.15"))

WINDOW_DROPOUT = float(os.environ.get("W50_WINDOW_DROPOUT", "0.05"))

SERIES_DROPOUT = float(os.environ.get("W50_SERIES_DROPOUT", "0.05"))

HEAD_EPOCHS = int(os.environ.get("W50_HEAD_EPOCHS", "24"))

STEPS_PER_EPOCH = int(os.environ.get("W50_STEPS_PER_EPOCH", "32"))

GOLD_BATCH_SIZE = int(os.environ.get("W50_GOLD_BATCH_SIZE", "12"))

PSEUDO_BATCH_SIZE = int(os.environ.get("W50_PSEUDO_BATCH_SIZE", "24"))

VALIDATION_BATCH_SIZE = int(os.environ.get("W50_VALIDATION_BATCH_SIZE", "6"))

HEAD_MAX_LR = float(os.environ.get("W50_HEAD_MAX_LR", "0.001"))

HEAD_WEIGHT_DECAY = float(os.environ.get("W50_HEAD_WEIGHT_DECAY", "0.001"))

GRAD_CLIP_NORM = float(os.environ.get("W50_GRAD_CLIP_NORM", "5.0"))

BOOTSTRAP_ITERATIONS = int(os.environ.get("W50_BOOTSTRAP_ITERATIONS", "2000"))


# ------------------------------------------------------------
# Exact W3/W4 fold identity
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


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
TRAIN_DEVICE = torch.device("cuda:0" if GPU_COUNT > 0 else "cpu")
CACHE_GPU_IDS = list(range(min(GPU_COUNT, 2))) if GPU_COUNT else []

try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================================================
# 2. GENERAL HELPERS
# ============================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def elapsed_string(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_uid_hash(uid: str) -> str:
    return hashlib.md5(str(uid).encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # torch generators accept a wider range, but 31-bit keeps portability.
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def study_cache_path(uid: str) -> Path:
    return CACHE_STUDY_ROOT / f"{stable_uid_hash(uid)}.pt"


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return numeric != 0

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler():
    if TRAIN_DEVICE.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks without scipy; deterministic for tied values."""
    return (
        pd.Series(np.asarray(values)).rank(method="average").to_numpy(dtype=np.float64)
    )


def finite_corr(first: np.ndarray, second: np.ndarray, spearman: bool = False) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if spearman:
        a = rankdata_average(a)
        b = rankdata_average(b)
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# 3. SHALLOW KAGGLE RESOURCE DISCOVERY
# ============================================================


def shallow_kaggle_dirs(max_depth: int = 4) -> List[Path]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []

    results = [root]
    queue = [(root, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        try:
            children = [item for item in current.iterdir() if item.is_dir()]
        except Exception:
            continue

        for child in children:
            if child.name in {"train_series", "test_series"}:
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

            size_score = fold_sizes[fold] / max(1, math.ceil(n_samples / n_splits))
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

    bad_planes = set(series_df["Anatomical_Plane"].dropna().astype(str).unique()) - set(
        PLANES
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
            "W5.0 outer-fold checksum mismatch.\n"
            f"Found   : {fold_sha}\n"
            f"Expected: {EXPECTED_FOLD_SHA256}"
        )

    fold_df.to_csv(RESULT_ROOT / "00_outer_fold_assignments.csv", index=False)
    return train_df, series_df, gold_df, fold_zero


# ============================================================
# 5. ORTHODIFFUSION DISCOVERY + IDENTITY
# ============================================================


def looks_like_ortho_code_root(path: Path) -> bool:
    return all(
        p.exists()
        for p in (
            path / "diffusion_model" / "trainer.py",
            path / "diffusion_model" / "unet.py",
            path / "dataset.py",
            path / "finetune_classifier.py",
        )
    )


def looks_like_ortho_weights_root(path: Path) -> bool:
    return all((path / filename).exists() for filename in WEIGHT_FILENAMES.values())


def discover_ortho_code_root() -> Path:
    candidates: List[Path] = []

    if EXPLICIT_ORTHO_CODE_ROOT:
        explicit = Path(EXPLICIT_ORTHO_CODE_ROOT)
        candidates.extend([explicit, explicit / "OrthoDiffusion"])

    candidates.extend(
        [
            Path("/kaggle/working/rsna_w5_0_ortho_preflight/official/OrthoDiffusion"),
            Path("/kaggle/working/OrthoDiffusion"),
        ]
    )

    for root in shallow_kaggle_dirs(max_depth=4):
        candidates.extend([root, root / "OrthoDiffusion"])

    for path in dict.fromkeys(candidates):
        if looks_like_ortho_code_root(path):
            return path

    raise FileNotFoundError(
        "Official OrthoDiffusion code root not found. Set W50_ORTHO_CODE_ROOT."
    )


def discover_ortho_weights_root() -> Path:
    candidates: List[Path] = []

    if EXPLICIT_ORTHO_WEIGHTS_ROOT:
        explicit = Path(EXPLICIT_ORTHO_WEIGHTS_ROOT)
        candidates.extend([explicit, explicit / "weights", explicit / "orthodiffusion"])

    candidates.extend(
        [
            Path("/kaggle/working/rsna_w5_0_ortho_preflight/official/weights"),
            Path("/kaggle/working/orthodiffusion"),
        ]
    )

    for root in shallow_kaggle_dirs(max_depth=4):
        candidates.extend([root, root / "weights", root / "orthodiffusion"])

    for path in dict.fromkeys(candidates):
        if looks_like_ortho_weights_root(path):
            return path

    raise FileNotFoundError(
        "Official OrthoDiffusion weights root not found. Set W50_ORTHO_WEIGHTS_ROOT."
    )


def git_head(path: Path) -> Optional[str]:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return None


def validate_ortho_identity(
    code_root: Path,
    weights_root: Path,
) -> Dict[str, Any]:
    head = git_head(code_root)

    if (
        head is not None
        and head != EXPECTED_ORTHO_GIT_HEAD
        and not ALLOW_ORTHO_CODE_MISMATCH
    ):
        raise RuntimeError(
            "OrthoDiffusion git HEAD differs from the exact preflight commit.\n"
            f"Found   : {head}\n"
            f"Expected: {EXPECTED_ORTHO_GIT_HEAD}\n"
            "Use the preflight source tree, or set "
            "W50_ALLOW_ORTHO_CODE_MISMATCH=1 only for an intentional new-code experiment."
        )

    weights = {}
    for plane, filename in WEIGHT_FILENAMES.items():
        path = weights_root / filename
        if not path.exists():
            raise FileNotFoundError(path)

        sha = sha256_file(path)
        expected = EXPECTED_WEIGHT_SHA256[plane]
        if sha != expected and not ALLOW_ORTHO_SHA_MISMATCH:
            raise RuntimeError(
                f"{plane} OrthoDiffusion weight SHA256 mismatch.\n"
                f"Found   : {sha}\n"
                f"Expected: {expected}"
            )

        weights[plane] = {
            "path": str(path),
            "sha256": sha,
            "size_mb": path.stat().st_size / (1024**2),
        }

    return {
        "code_root": str(code_root),
        "git_head": head,
        "expected_git_head": EXPECTED_ORTHO_GIT_HEAD,
        "weights_root": str(weights_root),
        "weights": weights,
    }


def import_official_ortho_code(code_root: Path):
    root = str(code_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    trainer = importlib.import_module("diffusion_model.trainer")
    unet = importlib.import_module("diffusion_model.unet")

    if not hasattr(trainer, "GaussianDiffusion"):
        raise RuntimeError("Official trainer.py does not expose GaussianDiffusion")
    if not hasattr(unet, "create_model"):
        raise RuntimeError("Official unet.py does not expose create_model")

    return trainer.GaussianDiffusion, unet.create_model


def load_one_ortho_model(
    code_root: Path,
    weight_path: Path,
    device: torch.device,
):
    GaussianDiffusion, create_model = import_official_ortho_code(code_root)

    denoise = create_model(
        ORTHO_INPUT_SIZE,
        ORTHO_BASE_CHANNELS,
        ORTHO_NUM_RES_BLOCKS,
        in_channels=ORTHO_IN_CHANNELS,
        out_channels=ORTHO_OUT_CHANNELS,
    ).to(device)

    diffusion = GaussianDiffusion(
        denoise,
        image_size=ORTHO_INPUT_SIZE,
        depth_size=ORTHO_DEPTH_SIZE,
        timesteps=ORTHO_TIMESTEPS,
        loss_type="l2",
    ).to(device)

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "ema" not in checkpoint:
        raise RuntimeError(f"{weight_path} does not contain checkpoint['ema']")

    state = {}
    for key, value in checkpoint["ema"].items():
        if key.startswith("denoise_fn.module."):
            key = key.replace("denoise_fn.module.", "denoise_fn.", 1)
        state[key] = value

    incompatible = diffusion.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{weight_path.name}: state mismatch. "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    diffusion.eval()
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)

    del checkpoint, state
    gc.collect()
    return diffusion


def load_all_ortho_models(
    code_root: Path,
    weights_root: Path,
    device: torch.device,
) -> Dict[str, nn.Module]:
    models = {}
    for plane in PLANES:
        models[plane] = load_one_ortho_model(
            code_root,
            weights_root / WEIGHT_FILENAMES[plane],
            device,
        )
    return models


# ============================================================
# 6. DICOM GEOMETRY / ROBUST DECODING
# ============================================================


def unit_vector(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("Zero-length vector")
    return value / norm


def geometry_plane_from_iop(image_orientation_patient) -> Tuple[str, float]:
    iop = np.asarray(image_orientation_patient, dtype=np.float64)
    if iop.shape != (6,):
        raise ValueError(f"Expected six IOP values, got {iop.shape}")

    row = unit_vector(iop[:3])
    column = unit_vector(iop[3:])
    normal = unit_vector(np.cross(row, column))

    scores = {
        "Sagittal": abs(float(normal[0])),
        "Coronal": abs(float(normal[1])),
        "Axial": abs(float(normal[2])),
    }
    plane = max(scores, key=scores.get)
    return plane, float(scores[plane])


def scalar_slice_position(ds) -> Optional[float]:
    orientation = getattr(ds, "ImageOrientationPatient", None)
    position = getattr(ds, "ImagePositionPatient", None)
    if orientation is None or position is None:
        return None
    try:
        row = np.asarray(orientation[:3], dtype=np.float64)
        column = np.asarray(orientation[3:], dtype=np.float64)
        normal = unit_vector(np.cross(row, column))
        return float(np.dot(np.asarray(position, dtype=np.float64), normal))
    except Exception:
        return None


def read_series_headers(study_uid: str, series_uid: str) -> List[Dict[str, Any]]:
    series_dir = TRAIN_SERIES_ROOT / study_uid / series_uid
    paths = sorted(series_dir.glob("*.dcm"))
    records: List[Dict[str, Any]] = []

    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            iop = getattr(ds, "ImageOrientationPatient", None)
            ps = getattr(ds, "PixelSpacing", None)
            records.append(
                {
                    "path": str(path),
                    "position": scalar_slice_position(ds),
                    "instance": getattr(ds, "InstanceNumber", 0),
                    "iop": list(iop) if iop is not None else None,
                    "pixel_spacing": (
                        [float(ps[0]), float(ps[1])]
                        if ps is not None and len(ps) >= 2
                        else None
                    ),
                }
            )
        except Exception:
            records.append(
                {
                    "path": str(path),
                    "position": None,
                    "instance": 0,
                    "iop": None,
                    "pixel_spacing": None,
                }
            )

    if records and all(item["position"] is not None for item in records):
        records.sort(key=lambda item: float(item["position"]))
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
    Safe native-DICOM repair retained from the proven W4 pipeline.
    Pixel bytes are never altered and compressed transfer syntaxes are
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


def decode_rescaled_slice(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    image = decode_dicom_pixel_array(ds, path)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise RuntimeError(f"Expected 2-D DICOM slice, got {image.shape}: {path}")

    image = image.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return image * slope + intercept


def physical_stack_span_mm(records: List[Dict[str, Any]]) -> float:
    positions = [item["position"] for item in records]
    if positions and all(value is not None for value in positions):
        values = np.asarray(positions, dtype=np.float64)
        if np.isfinite(values).all():
            return float(abs(values[-1] - values[0]))
    return float("nan")


def median_pixel_spacing(records: List[Dict[str, Any]]) -> Tuple[float, float]:
    values = [
        item["pixel_spacing"] for item in records if item["pixel_spacing"] is not None
    ]
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array[:, 0])), float(np.median(array[:, 1]))


def window_start_indices(n_slices: int) -> List[int]:
    """
    Up to three 16-slice windows centred near 25%, 50%, 75% of the
    physically ordered series. Boundary clipping + de-duplication are
    deterministic.
    """
    if n_slices < ORTHO_DEPTH_SIZE:
        return []

    if n_slices == ORTHO_DEPTH_SIZE:
        return [0]

    max_start = n_slices - ORTHO_DEPTH_SIZE
    starts: List[int] = []

    for fraction in WINDOW_CENTER_FRACTIONS:
        desired_center = fraction * (n_slices - 1)
        start = int(round(desired_center - (ORTHO_DEPTH_SIZE - 1) / 2.0))
        start = min(max(start, 0), max_start)
        if start not in starts:
            starts.append(start)

    # Ensure the central window is represented even if user supplied
    # unusual fractions that collapse after clipping.
    center_start = int(round(0.5 * max_start))
    if center_start not in starts and len(starts) < MAX_WINDOWS_PER_SERIES:
        starts.append(center_start)

    starts = sorted(dict.fromkeys(starts))[:MAX_WINDOWS_PER_SERIES]
    return starts


def normalized_window_center(start: int, n_slices: int) -> float:
    center = start + (ORTHO_DEPTH_SIZE - 1) / 2.0
    if n_slices <= 1:
        return 0.0
    return float(2.0 * (center / (n_slices - 1)) - 1.0)


# ============================================================
# 7. CACHE IDENTITY + OFFICIAL-STYLE 3-D VOLUMES
# ============================================================


def cache_identity(identity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "orthodiffusion_git_head": identity.get("git_head"),
        "expected_git_head": EXPECTED_ORTHO_GIT_HEAD,
        "weight_sha256": {
            plane: identity["weights"][plane]["sha256"] for plane in PLANES
        },
        "input_shape": [1, ORTHO_DEPTH_SIZE, ORTHO_INPUT_SIZE, ORTHO_INPUT_SIZE],
        "feature_timestep_by_plane": FEATURE_TIMESTEP_BY_PLANE,
        "feature_block_by_plane": FEATURE_BLOCK_BY_PLANE,
        "expected_feature_shape": [
            ORTHO_FEATURE_CHANNELS,
            ORTHO_FEATURE_HEIGHT,
            ORTHO_FEATURE_WIDTH,
        ],
        "window_center_fractions": list(WINDOW_CENTER_FRACTIONS),
        "max_windows_per_series": MAX_WINDOWS_PER_SERIES,
        "min_slices_per_series": MIN_SLICES_PER_SERIES,
        "volume_normalization": "per-window minmax [0,1] then [-1,1]",
        "resize": "cv2.INTER_LINEAR to 256x256",
        "slice_order": "ImagePositionPatient projection; InstanceNumber fallback",
        "diffusion_noise": "deterministic stable per StudyUID/SeriesUID/window",
        "feature_dtype": "float16",
        "short_series_policy": "skip series with <16 slices",
    }


def prepare_cache_identity(identity: Dict[str, Any]) -> None:
    metadata_path = CACHE_ROOT / "cache_meta.json"
    expected = cache_identity(identity)

    if RESET_FEATURE_CACHE:
        log("W50_RESET_CACHE=1 -> deleting existing W5 feature cache.")
        for path in CACHE_STUDY_ROOT.glob("*.pt"):
            path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    if metadata_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                "Existing W5 feature-cache identity differs from the current "
                "configuration. Set W50_RESET_CACHE=1 only if an intentional "
                "full rebuild is desired."
            )
    else:
        metadata_path.write_text(
            json.dumps(expected, indent=2, allow_nan=True),
            encoding="utf-8",
        )


def effective_series_plane(
    metadata_plane: str,
    records: List[Dict[str, Any]],
) -> Tuple[str, Optional[str], float, bool]:
    valid_iop = [item["iop"] for item in records if item.get("iop") is not None]
    geometry_plane = None
    confidence = float("nan")

    if valid_iop:
        geometry_plane, confidence = geometry_plane_from_iop(
            valid_iop[len(valid_iop) // 2]
        )

    mismatch = geometry_plane is not None and geometry_plane != metadata_plane

    if mismatch and np.isfinite(confidence) and confidence >= GEOMETRY_PLANE_CONFIDENCE:
        effective = geometry_plane
    else:
        effective = metadata_plane

    if effective not in PLANES:
        raise RuntimeError(f"Unsupported effective anatomical plane: {effective!r}")

    return effective, geometry_plane, confidence, mismatch


def prepare_series_windows(
    study_uid: str,
    series_uid: str,
    metadata_plane: str,
) -> Tuple[List[torch.Tensor], List[int], List[float], Dict[str, Any]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV is required for OrthoDiffusion preprocessing"
        ) from exc

    records = read_series_headers(study_uid, series_uid)
    n_slices = len(records)

    effective_plane, geometry_plane, plane_confidence, plane_mismatch = (
        effective_series_plane(metadata_plane, records)
    )

    base_audit = {
        UID_COLUMN: study_uid,
        SERIES_UID_COLUMN: series_uid,
        "MetadataPlane": metadata_plane,
        "GeometryPlane": geometry_plane,
        "EffectivePlane": effective_plane,
        "PlaneMatch": (
            bool(geometry_plane == metadata_plane) if geometry_plane else False
        ),
        "GeometryPlaneConfidence": plane_confidence,
        "MetadataGeometryMismatch": bool(plane_mismatch),
        "SourceSliceCount": int(n_slices),
    }

    starts = window_start_indices(n_slices)
    if not starts:
        audit = dict(base_audit)
        audit.update(
            {
                "Status": "SKIP_SHORT",
                "WindowCount": 0,
                "WindowStarts": "",
                "WindowCenters": "",
                "StackSpanMM": physical_stack_span_mm(records),
                "PixelSpacingMeanMM": float("nan"),
            }
        )
        return [], [], [], audit

    needed_indices = sorted(
        {index for start in starts for index in range(start, start + ORTHO_DEPTH_SIZE)}
    )

    resized_by_index: Dict[int, np.ndarray] = {}

    for index in needed_indices:
        image = decode_rescaled_slice(records[index]["path"])
        if image.shape != (ORTHO_INPUT_SIZE, ORTHO_INPUT_SIZE):
            image = cv2.resize(
                image,
                (ORTHO_INPUT_SIZE, ORTHO_INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        resized_by_index[index] = image.astype(np.float32, copy=False)

    windows: List[torch.Tensor] = []
    centers: List[float] = []

    for start in starts:
        images = [
            resized_by_index[index] for index in range(start, start + ORTHO_DEPTH_SIZE)
        ]

        # Official classification preprocessing operates on [H,W,D].
        volume_hwd = np.stack(images, axis=2).astype(np.float32, copy=False)
        minimum = float(np.min(volume_hwd))
        maximum = float(np.max(volume_hwd))

        normalized = (volume_hwd - minimum) / (maximum - minimum + 1e-8)
        normalized = (normalized * 2.0 - 1.0).astype(np.float32, copy=False)

        tensor = (
            torch.from_numpy(normalized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .float()
        )

        if tuple(tensor.shape) != (
            1,
            ORTHO_DEPTH_SIZE,
            ORTHO_INPUT_SIZE,
            ORTHO_INPUT_SIZE,
        ):
            raise RuntimeError(f"Unexpected Ortho volume shape {tuple(tensor.shape)}")

        windows.append(tensor)
        centers.append(normalized_window_center(start, n_slices))

    ps0, ps1 = median_pixel_spacing(records)
    pixel_mean = (
        float(np.mean([ps0, ps1]))
        if np.isfinite(ps0) and np.isfinite(ps1)
        else float("nan")
    )

    audit = dict(base_audit)
    audit.update(
        {
            "Status": "READY",
            "WindowCount": int(len(windows)),
            "WindowStarts": "|".join(str(x) for x in starts),
            "WindowCenters": "|".join(f"{x:.6f}" for x in centers),
            "StackSpanMM": physical_stack_span_mm(records),
            "PixelSpacingMeanMM": pixel_mean,
        }
    )

    return windows, starts, centers, audit


# ============================================================
# 8. DETERMINISTIC ORTHODIFFUSION FEATURE EXTRACTION
# ============================================================


@torch.inference_mode()
def deterministic_ortho_feature(
    diffusion,
    volume_batch: torch.Tensor,
    seeds: Sequence[int],
    plane: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Equivalent to official GaussianDiffusion.get_feature, except the q_sample
    noise is explicitly supplied per window.  Official get_feature internally
    calls q_sample without a noise argument, making the representation depend
    on global RNG state and batch composition.  W5 fixes that reproducibility
    issue while preserving the same diffusion equation and denoiser feature.
    """
    if len(seeds) != len(volume_batch):
        raise ValueError("One deterministic noise seed is required per volume")

    image = volume_batch.to(device, non_blocking=True).float()
    timestep = FEATURE_TIMESTEP_BY_PLANE[plane]
    block = FEATURE_BLOCK_BY_PLANE[plane]

    t_tensor = torch.full(
        (image.shape[0],),
        timestep,
        dtype=torch.long,
        device=device,
    )

    noise_parts = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noise_parts.append(
            torch.randn(
                (1, *image.shape[1:]),
                generator=generator,
                device=device,
                dtype=image.dtype,
            )
        )
    noise = torch.cat(noise_parts, dim=0)

    x_noisy = diffusion.q_sample(
        x_start=image,
        t=t_tensor,
        noise=noise,
    )

    if getattr(diffusion, "with_condition", False):
        raise RuntimeError(
            "W5.0 does not support conditioned OrthoDiffusion checkpoints"
        )

    ret_in = block.startswith("in_")
    ret_mid = block.startswith("mid_")
    ret_out = block.startswith("out_") or block == "final"

    kwargs = {}
    if getattr(diffusion, "class_cond", False):
        kwargs["y"] = None

    with autocast_context(device):
        _, info = diffusion.denoise_fn(
            x_noisy,
            t_tensor,
            ret_in=ret_in,
            ret_mid=ret_mid,
            ret_out=ret_out,
            **kwargs,
        )

    if block.startswith("in_"):
        feature = info["in"][block]
    elif block.startswith("mid_"):
        feature = info["mid"][block]
    elif block.startswith("out_"):
        feature = info["out"][block]
    elif block == "final":
        feature = info["out"]["final"]
    else:
        raise ValueError(f"Unsupported feature block: {block}")

    expected = (
        image.shape[0],
        ORTHO_FEATURE_CHANNELS,
        1,
        ORTHO_FEATURE_HEIGHT,
        ORTHO_FEATURE_WIDTH,
    )

    if tuple(feature.shape) != expected:
        raise RuntimeError(
            f"{plane} feature shape {tuple(feature.shape)} != expected {expected}. "
            "W5.0 is locked to the validated t=100 / mid_2 representation."
        )

    return feature[:, :, 0].detach()


# ============================================================
# 9. ENCODE ONE STUDY
# ============================================================


def encode_study(
    study_uid: str,
    study_series: pd.DataFrame,
    models: Dict[str, nn.Module],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ordered = study_series.copy()
    ordered["_plane_order"] = ordered["Anatomical_Plane"].map(PLANE_TO_INDEX)
    ordered = ordered.sort_values(["_plane_order", SERIES_UID_COLUMN]).reset_index(
        drop=True
    )

    prepared_series: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for _, row in ordered.iterrows():
        series_uid = str(row[SERIES_UID_COLUMN])
        metadata_plane = str(row["Anatomical_Plane"])

        try:
            windows, starts, centers, audit = prepare_series_windows(
                study_uid,
                series_uid,
                metadata_plane,
            )

            if not windows:
                audit_rows.append(audit)
                continue

            effective_plane = str(audit["EffectivePlane"])
            n_slices = int(audit["SourceSliceCount"])
            stack_span = float(audit["StackSpanMM"])
            pixel_spacing = float(audit["PixelSpacingMeanMM"])

            series_cont = np.asarray(
                [
                    np.clip(
                        math.log1p(max(n_slices, 0)) / math.log1p(320.0),
                        0.0,
                        2.0,
                    ),
                    (
                        np.clip(stack_span / 300.0, 0.0, 2.0)
                        if np.isfinite(stack_span)
                        else 0.0
                    ),
                    (
                        np.clip(pixel_spacing, 0.0, 2.0)
                        if np.isfinite(pixel_spacing)
                        else 0.0
                    ),
                ],
                dtype=np.float32,
            )

            prepared_series.append(
                {
                    "series_uid": series_uid,
                    "plane": effective_plane,
                    "windows": windows,
                    "starts": starts,
                    "centers": centers,
                    "series_meta": np.asarray(
                        [
                            PLANE_TO_INDEX[effective_plane],
                            int(row["_fluid"]),
                            int(row["_fs"]),
                        ],
                        dtype=np.int64,
                    ),
                    "series_cont": series_cont,
                    "audit": audit,
                }
            )

        except Exception as exc:
            audit_rows.append(
                {
                    UID_COLUMN: study_uid,
                    SERIES_UID_COLUMN: series_uid,
                    "MetadataPlane": metadata_plane,
                    "GeometryPlane": None,
                    "EffectivePlane": None,
                    "PlaneMatch": False,
                    "GeometryPlaneConfidence": float("nan"),
                    "MetadataGeometryMismatch": False,
                    "SourceSliceCount": float("nan"),
                    "Status": "ERROR",
                    "WindowCount": 0,
                    "WindowStarts": "",
                    "WindowCenters": "",
                    "StackSpanMM": float("nan"),
                    "PixelSpacingMeanMM": float("nan"),
                    "Error": repr(exc),
                }
            )
            raise RuntimeError(
                f"Failed preparing series {series_uid} in study {study_uid}: {exc}"
            ) from exc

    if not prepared_series:
        raise RuntimeError(
            f"Study {study_uid} has no OrthoDiffusion-eligible series with >=16 slices"
        )

    # Encode all windows, grouped by plane so every forward uses the correct
    # orientation-specific backbone.
    feature_lookup: Dict[Tuple[int, int], torch.Tensor] = {}

    for plane in PLANES:
        references: List[Tuple[int, int, torch.Tensor, int]] = []

        for series_index, item in enumerate(prepared_series):
            if item["plane"] != plane:
                continue

            for window_index, (volume, start) in enumerate(
                zip(item["windows"], item["starts"])
            ):
                seed = stable_seed(
                    "W5.0",
                    study_uid,
                    item["series_uid"],
                    start,
                    plane,
                    FEATURE_TIMESTEP_BY_PLANE[plane],
                    FEATURE_BLOCK_BY_PLANE[plane],
                    "diffusion-noise-v1",
                )
                references.append((series_index, window_index, volume, seed))

        batch_size = ORTHO_BATCH_BY_PLANE[plane]
        for start_pos in range(0, len(references), batch_size):
            part = references[start_pos : start_pos + batch_size]
            if not part:
                continue

            volume_batch = torch.stack([item[2] for item in part], dim=0)
            seeds = [item[3] for item in part]

            feature_batch = deterministic_ortho_feature(
                models[plane],
                volume_batch,
                seeds,
                plane,
                device,
            )

            for local_index, (series_index, window_index, _, _) in enumerate(part):
                feature_lookup[(series_index, window_index)] = (
                    feature_batch[local_index]
                    .to(dtype=torch.float16)
                    .cpu()
                    .contiguous()
                )

            del volume_batch, feature_batch

    n_series = len(prepared_series)
    features = torch.zeros(
        n_series,
        MAX_WINDOWS_PER_SERIES,
        ORTHO_FEATURE_CHANNELS,
        ORTHO_FEATURE_HEIGHT,
        ORTHO_FEATURE_WIDTH,
        dtype=torch.float16,
    )
    window_mask = torch.zeros(
        n_series,
        MAX_WINDOWS_PER_SERIES,
        dtype=torch.bool,
    )
    window_center = torch.zeros(
        n_series,
        MAX_WINDOWS_PER_SERIES,
        dtype=torch.float32,
    )
    series_meta = torch.zeros(n_series, 3, dtype=torch.long)
    series_cont = torch.zeros(n_series, 3, dtype=torch.float32)

    series_uids = []
    window_starts_serialized = []

    for series_index, item in enumerate(prepared_series):
        series_uids.append(item["series_uid"])
        series_meta[series_index] = torch.from_numpy(item["series_meta"])
        series_cont[series_index] = torch.from_numpy(item["series_cont"])

        starts_for_series = []
        for window_index, (start, center) in enumerate(
            zip(item["starts"], item["centers"])
        ):
            feature = feature_lookup.get((series_index, window_index))
            if feature is None:
                raise RuntimeError(
                    f"Missing encoded feature for {study_uid} / "
                    f"{item['series_uid']} window {window_index}"
                )

            features[series_index, window_index] = feature
            window_mask[series_index, window_index] = True
            window_center[series_index, window_index] = float(center)
            starts_for_series.append(int(start))

        window_starts_serialized.append(starts_for_series)

        audit = dict(item["audit"])
        audit["Status"] = "OK"
        audit["EncodedFeatureShape"] = (
            f"[{len(item['starts'])},{ORTHO_FEATURE_CHANNELS},"
            f"{ORTHO_FEATURE_HEIGHT},{ORTHO_FEATURE_WIDTH}]"
        )
        audit_rows.append(audit)

    payload = {
        "cache_version": CACHE_VERSION,
        "study_uid": study_uid,
        "features": features.contiguous(),
        "window_mask": window_mask.contiguous(),
        "window_center": window_center.contiguous(),
        "series_meta": series_meta.contiguous(),
        "series_cont": series_cont.contiguous(),
        "series_uids": series_uids,
        "window_starts": window_starts_serialized,
        "series_audit": audit_rows,
    }

    return payload, audit_rows


# ============================================================
# 10. DUAL-T4 RESUMABLE FEATURE CACHE
# ============================================================


def cache_file_is_usable(path: Path, uid: str) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        features = payload.get("features")
        window_mask = payload.get("window_mask")
        window_center = payload.get("window_center")
        series_meta = payload.get("series_meta")
        series_cont = payload.get("series_cont")

        return bool(
            payload.get("cache_version") == CACHE_VERSION
            and str(payload.get("study_uid")) == str(uid)
            and isinstance(features, torch.Tensor)
            and features.ndim == 5
            and tuple(features.shape[1:])
            == (
                MAX_WINDOWS_PER_SERIES,
                ORTHO_FEATURE_CHANNELS,
                ORTHO_FEATURE_HEIGHT,
                ORTHO_FEATURE_WIDTH,
            )
            and features.dtype == torch.float16
            and isinstance(window_mask, torch.Tensor)
            and tuple(window_mask.shape) == tuple(features.shape[:2])
            and isinstance(window_center, torch.Tensor)
            and tuple(window_center.shape) == tuple(features.shape[:2])
            and isinstance(series_meta, torch.Tensor)
            and tuple(series_meta.shape) == (features.shape[0], 3)
            and isinstance(series_cont, torch.Tensor)
            and tuple(series_cont.shape) == (features.shape[0], 3)
            and int(window_mask.sum()) >= 1
        )
    except Exception:
        return False


def _cache_worker(
    worker_id: int,
    device_id: int,
    study_uids: List[str],
    grouped_series: Dict[str, pd.DataFrame],
    code_root: Path,
    weights_root: Path,
    shared_progress: Dict[str, int],
    progress_lock: threading.Lock,
    total_requested: int,
    started: float,
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    log(
        f"Cache worker {worker_id}: loading three OrthoDiffusion models "
        f"on {device}..."
    )
    models = load_all_ortho_models(code_root, weights_root, device)
    log(f"Cache worker {worker_id}: models ready; studies={len(study_uids)}")

    failures = []
    encoded = 0
    skipped = 0

    for local_index, uid in enumerate(study_uids, start=1):
        path = study_cache_path(uid)

        if cache_file_is_usable(path, uid):
            skipped += 1
        else:
            try:
                study_series = grouped_series.get(uid)
                if study_series is None or len(study_series) == 0:
                    raise RuntimeError("No train_series.csv rows for study")

                payload, _ = encode_study(
                    uid,
                    study_series,
                    models,
                    device,
                )
                atomic_torch_save(payload, path)
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
            rate = done_now / max(elapsed_seconds, 1e-6)
            eta = (total_requested - done_now) / max(rate, 1e-6)
            log(
                f"W5 cache {done_now:4d}/{total_requested} "
                f"({100.0 * done_now / total_requested:5.1f}%) "
                f"elapsed={elapsed_string(elapsed_seconds)} "
                f"ETA={elapsed_string(eta)} "
                f"worker={worker_id} encoded={encoded} "
                f"failures={len(failures)}"
            )

        if local_index % 20 == 0:
            gc.collect()

    del models
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "worker_id": worker_id,
        "encoded": encoded,
        "skipped": skipped,
        "failures": failures,
    }


def summarize_cache(train_df: pd.DataFrame, identity: Dict[str, Any]) -> Dict[str, Any]:
    uids = sorted(train_df[UID_COLUMN].astype(str).tolist())
    cached = 0
    encoded_series = 0
    encoded_windows = 0
    feature_bytes = 0
    audit_rows: List[Dict[str, Any]] = []

    manifest_rows = []

    for uid in uids:
        path = study_cache_path(uid)
        usable = cache_file_is_usable(path, uid)
        manifest_rows.append(
            {
                UID_COLUMN: uid,
                "CachePath": str(path),
                "Usable": bool(usable),
                "FileSizeBytes": path.stat().st_size if path.exists() else 0,
            }
        )

        if not usable:
            continue

        cached += 1
        payload = torch.load(path, map_location="cpu", weights_only=False)
        encoded_series += int(payload["features"].shape[0])
        encoded_windows += int(payload["window_mask"].sum().item())
        feature_bytes += int(
            payload["features"].numel() * payload["features"].element_size()
        )
        audit_rows.extend(payload.get("series_audit", []))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(RESULT_ROOT / "feature_cache_manifest.csv", index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(RESULT_ROOT / "series_cache_audit.csv", index=False)

    if len(audit_df) and "Status" in audit_df.columns:
        status_counts = {
            str(key): int(value)
            for key, value in audit_df["Status"].value_counts(dropna=False).items()
        }
        ok_audit = audit_df[audit_df["Status"] == "OK"].copy()
    else:
        status_counts = {}
        ok_audit = pd.DataFrame()

    if len(ok_audit):
        encoded_series_by_plane = {
            plane: int((ok_audit["EffectivePlane"].astype(str) == plane).sum())
            for plane in PLANES
        }
        encoded_windows_by_plane = {
            plane: int(
                pd.to_numeric(
                    ok_audit.loc[
                        ok_audit["EffectivePlane"].astype(str) == plane,
                        "WindowCount",
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )
            for plane in PLANES
        }
        geometry_confidence = pd.to_numeric(
            ok_audit.get("GeometryPlaneConfidence"),
            errors="coerce",
        )
        minimum_geometry_confidence = (
            float(geometry_confidence.min())
            if geometry_confidence.notna().any()
            else float("nan")
        )
        metadata_geometry_matches = int(
            ok_audit.get("PlaneMatch", pd.Series(False, index=ok_audit.index))
            .fillna(False)
            .astype(bool)
            .sum()
        )
        metadata_geometry_mismatches = int(
            ok_audit.get(
                "MetadataGeometryMismatch",
                pd.Series(False, index=ok_audit.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        )
    else:
        encoded_series_by_plane = {plane: 0 for plane in PLANES}
        encoded_windows_by_plane = {plane: 0 for plane in PLANES}
        minimum_geometry_confidence = float("nan")
        metadata_geometry_matches = 0
        metadata_geometry_mismatches = 0

    summary = {
        "cache_version": CACHE_VERSION,
        "cache_complete": cached == len(uids),
        "train_studies": len(uids),
        "cached_studies": cached,
        "encoded_series": encoded_series,
        "encoded_windows": encoded_windows,
        "mean_encoded_series_per_study": encoded_series / max(cached, 1),
        "mean_encoded_windows_per_study": encoded_windows / max(cached, 1),
        "raw_feature_payload_gb": feature_bytes / (1024**3),
        "series_status_counts": status_counts,
        "encoded_series_by_plane": encoded_series_by_plane,
        "encoded_windows_by_plane": encoded_windows_by_plane,
        "metadata_geometry_matches": metadata_geometry_matches,
        "metadata_geometry_mismatches": metadata_geometry_mismatches,
        "minimum_geometry_plane_confidence": minimum_geometry_confidence,
        "orthodiffusion_identity": identity,
        "feature_timestep_by_plane": FEATURE_TIMESTEP_BY_PLANE,
        "feature_block_by_plane": FEATURE_BLOCK_BY_PLANE,
        "window_center_fractions": list(WINDOW_CENTER_FRACTIONS),
        "max_windows_per_series": MAX_WINDOWS_PER_SERIES,
    }

    (RESULT_ROOT / "feature_cache_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log("\nW5 feature-cache summary:")
    log(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def build_feature_cache(
    train_df: pd.DataFrame,
    series_df: pd.DataFrame,
) -> Dict[str, Any]:
    if GPU_COUNT <= 0:
        raise RuntimeError("W5 OrthoDiffusion cache requires CUDA")

    code_root = discover_ortho_code_root()
    weights_root = discover_ortho_weights_root()
    identity = validate_ortho_identity(code_root, weights_root)
    prepare_cache_identity(identity)

    uids = sorted(train_df[UID_COLUMN].astype(str).tolist())
    grouped_series = {
        str(uid): group.copy() for uid, group in series_df.groupby(UID_COLUMN)
    }

    missing_series = [uid for uid in uids if uid not in grouped_series]
    if missing_series:
        raise RuntimeError(f"train_series.csv missing studies: {missing_series[:10]}")

    already_cached = [
        uid for uid in uids if cache_file_is_usable(study_cache_path(uid), uid)
    ]
    already_set = set(already_cached)
    need = [uid for uid in uids if uid not in already_set]

    log("\n" + "=" * 88)
    log("W5.0 ORTHODIFFUSION 3-D SPATIAL FEATURE CACHE")
    log("=" * 88)
    log(f"Code root             : {code_root}")
    log(f"Weights root          : {weights_root}")
    log(f"CUDA GPU count        : {GPU_COUNT}")
    log(f"Cache GPU IDs         : {CACHE_GPU_IDS}")
    log(f"Train studies         : {len(uids)}")
    log(f"Already cached        : {len(already_cached)}")
    log(f"Need encoding         : {len(need)}")
    log(f"Max windows / series  : {MAX_WINDOWS_PER_SERIES}")
    log(f"Window centers        : {WINDOW_CENTER_FRACTIONS}")
    log(f"Plane batches         : {ORTHO_BATCH_BY_PLANE}")
    log(f"Feature timestep      : {FEATURE_TIMESTEP_BY_PLANE}")
    log(f"Feature block         : {FEATURE_BLOCK_BY_PLANE}")
    log("Cache representation  : frozen Ortho mid_2 [256,16,16], FP16")

    if not need:
        return summarize_cache(train_df, identity)

    worker_gpu_ids = CACHE_GPU_IDS if CACHE_GPU_IDS else [0]
    n_workers = len(worker_gpu_ids)
    shards = [need[i::n_workers] for i in range(n_workers)]
    shared_progress = {"done": 0}
    progress_lock = threading.Lock()
    started = time.time()
    worker_results = []

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
                    code_root,
                    weights_root,
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

    failure_path = RESULT_ROOT / "feature_cache_failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    else:
        failure_path.write_text("", encoding="utf-8")

    summary = summarize_cache(train_df, identity)
    summary["runtime_seconds_this_call"] = time.time() - started
    (RESULT_ROOT / "feature_cache_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    if failures:
        raise RuntimeError(
            f"{len(failures)} W5 study cache failures. Inspect {failure_path}. "
            "Training is stopped rather than silently dropping studies."
        )

    if not summary["cache_complete"]:
        raise RuntimeError("W5 feature cache is incomplete")

    return summary


# ============================================================
# 11. W2.3 DISCOVERY + UNCHANGED FOLD-SAFE PSEUDO TARGETS
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
        "W2.3 fold-safe outputs not found. Set W50_W23_ROOT to the directory "
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
        raise RuntimeError("W2.3 outer folds do not match the exact W3/W4/W5 folds")

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
        raise RuntimeError(f"W2.3 fold {fold} wide-file UID mismatch")

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
            raise RuntimeError(f"W2.3 fold {fold} high-mask UID mismatch")

        high_bool = pd.DataFrame(index=high_df.index)
        for label in LABEL_COLUMNS:
            high_bool[label] = coerce_bool_series(high_df[label])

        if not np.array_equal(
            high_bool[LABEL_COLUMNS].values.astype(bool),
            mask[LABEL_COLUMNS].values.astype(bool),
        ):
            raise RuntimeError(
                f"Reconstructed W2.3 fold {fold} high-selection mask differs "
                "from high_selection_candidate_mask_wide.csv"
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
# 12. LOAD COMPACT W5 CACHE INTO CPU RAM
# ============================================================


class OrthoFeatureStore:
    def __init__(self, train_df: pd.DataFrame):
        self.uids = sorted(train_df[UID_COLUMN].astype(str).tolist())
        self.uid_to_index = {uid: i for i, uid in enumerate(self.uids)}
        self.records: List[Dict[str, Any]] = []

        started = time.time()
        total_feature_bytes = 0

        log("\nLoading W5 Ortho spatial feature cache into CPU RAM...")

        for n, uid in enumerate(self.uids, start=1):
            path = study_cache_path(uid)
            if not cache_file_is_usable(path, uid):
                raise RuntimeError(
                    f"Missing/incompatible W5 cache for study {uid}. "
                    "Run run_w50_ortho('cache') first."
                )

            payload = torch.load(path, map_location="cpu", weights_only=False)
            record = {
                "uid": uid,
                "features": payload["features"].contiguous(),
                "window_mask": payload["window_mask"].contiguous(),
                "window_center": payload["window_center"].contiguous(),
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

        log(
            f"Ortho feature payload : "
            f"{total_feature_bytes / (1024 ** 3):.3f} GB FP16"
        )

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

        features = torch.zeros(
            batch_size,
            max_series,
            MAX_WINDOWS_PER_SERIES,
            ORTHO_FEATURE_CHANNELS,
            ORTHO_FEATURE_HEIGHT,
            ORTHO_FEATURE_WIDTH,
            dtype=torch.float16,
        )

        window_mask = torch.zeros(
            batch_size,
            max_series,
            MAX_WINDOWS_PER_SERIES,
            dtype=torch.bool,
        )

        window_center = torch.zeros(
            batch_size,
            max_series,
            MAX_WINDOWS_PER_SERIES,
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
            3,
            dtype=torch.float32,
        )

        series_mask = torch.zeros(
            batch_size,
            max_series,
            dtype=torch.bool,
        )

        for batch_index, record in enumerate(records):
            n_series = record["features"].shape[0]
            features[batch_index, :n_series] = record["features"]
            window_mask[batch_index, :n_series] = record["window_mask"]
            window_center[batch_index, :n_series] = record["window_center"].float()
            series_meta[batch_index, :n_series] = record["series_meta"].long()
            series_cont[batch_index, :n_series] = record["series_cont"].float()
            series_mask[batch_index, :n_series] = True

        return {
            "features": features.to(device, non_blocking=True),
            "window_mask": window_mask.to(device, non_blocking=True),
            "window_center": window_center.to(device, non_blocking=True),
            "series_meta": series_meta.to(device, non_blocking=True),
            "series_cont": series_cont.to(device, non_blocking=True),
            "series_mask": series_mask.to(device, non_blocking=True),
        }


# ============================================================
# 13. DIAGNOSIS-SPECIFIC SPATIAL -> WINDOW -> SERIES HEAD
# ============================================================


def _drop_mask_tokens(mask: torch.Tensor, drop_probability: float) -> torch.Tensor:
    if drop_probability <= 0.0:
        return mask

    original = mask.bool()
    keep = original & (torch.rand_like(original.float()) >= drop_probability)

    flat_original = original.reshape(-1, original.shape[-1])
    flat_keep = keep.reshape(-1, keep.shape[-1])

    had_any = flat_original.any(dim=1)
    lost_all = had_any & (~flat_keep.any(dim=1))

    if lost_all.any():
        rows = torch.where(lost_all)[0]
        # Deterministic relative to PyTorch RNG: choose one original valid token
        # using a random score masked to valid positions.
        random_score = torch.rand(
            len(rows),
            flat_original.shape[1],
            device=mask.device,
        )
        random_score = random_score.masked_fill(~flat_original[rows], -1.0)
        chosen = random_score.argmax(dim=1)
        flat_keep[rows, chosen] = True

    return flat_keep.reshape_as(original)


class OrthoSpatialHierarchicalDiagnosisHead(nn.Module):
    def __init__(self):
        super().__init__()

        h = HEAD_HIDDEN_DIM

        self.spatial_projection = nn.Sequential(
            nn.Conv2d(ORTHO_FEATURE_CHANNELS, h, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, math.gcd(h, 16)), num_channels=h),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h, bias=False),
            nn.GELU(),
        )

        self.row_position = nn.Parameter(torch.randn(ORTHO_FEATURE_HEIGHT, h) * 0.02)
        self.column_position = nn.Parameter(torch.randn(ORTHO_FEATURE_WIDTH, h) * 0.02)
        self.spatial_norm = nn.LayerNorm(h)

        self.spatial_queries = nn.Parameter(torch.randn(NUM_LABELS, h) / math.sqrt(h))

        # center + 4 Fourier frequencies -> 1 + 8 = 9 values.
        self.window_position_projection = nn.Sequential(
            nn.Linear(9, h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        self.window_norm = nn.LayerNorm(h)
        self.window_queries = nn.Parameter(torch.randn(NUM_LABELS, h) / math.sqrt(h))

        self.plane_embedding = nn.Embedding(3, 24)
        self.fluid_embedding = nn.Embedding(2, 8)
        self.fs_embedding = nn.Embedding(2, 8)

        self.metadata_projection = nn.Sequential(
            nn.Linear(24 + 8 + 8 + 3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        self.series_norm = nn.LayerNorm(h)
        self.series_queries = nn.Parameter(torch.randn(NUM_LABELS, h) / math.sqrt(h))

        self.final_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(HEAD_DROPOUT)

        self.classifier_weight = nn.Parameter(torch.randn(NUM_LABELS, h) / math.sqrt(h))
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    @staticmethod
    def _window_fourier(position: torch.Tensor) -> torch.Tensor:
        values = [position.unsqueeze(-1)]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            angle = math.pi * frequency * position
            values.append(torch.sin(angle).unsqueeze(-1))
            values.append(torch.cos(angle).unsqueeze(-1))
        return torch.cat(values, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        window_mask: torch.Tensor,
        window_center: torch.Tensor,
        series_meta: torch.Tensor,
        series_cont: torch.Tensor,
        series_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if features.ndim != 6:
            raise ValueError(f"Expected [B,S,M,C,H,W], got {features.shape}")

        batch_size, max_series, max_windows, channels, height, width = features.shape
        if channels != ORTHO_FEATURE_CHANNELS or height != 16 or width != 16:
            raise ValueError(f"Unexpected Ortho feature shape {features.shape}")

        if self.training and SERIES_DROPOUT > 0:
            series_mask_effective = _drop_mask_tokens(series_mask, SERIES_DROPOUT)
        else:
            series_mask_effective = series_mask.bool()

        window_mask_effective = window_mask.bool() & series_mask_effective.unsqueeze(-1)

        if self.training and WINDOW_DROPOUT > 0:
            # Each B,S row keeps at least one window if it originally had one.
            window_mask_effective = _drop_mask_tokens(
                window_mask_effective,
                WINDOW_DROPOUT,
            )
            window_mask_effective = (
                window_mask_effective & series_mask_effective.unsqueeze(-1)
            )

        valid_window_flat = window_mask_effective.reshape(-1)
        if not valid_window_flat.any():
            raise RuntimeError("No valid Ortho windows in batch after dropout")

        flat_features = features.reshape(
            batch_size * max_series * max_windows,
            channels,
            height,
            width,
        )[valid_window_flat]

        flat_center = window_center.reshape(-1)[valid_window_flat]

        hidden_map = self.spatial_projection(flat_features.float())
        tokens = hidden_map.permute(0, 2, 3, 1).reshape(
            hidden_map.shape[0],
            height * width,
            HEAD_HIDDEN_DIM,
        )

        position = (
            self.row_position[:, None, :] + self.column_position[None, :, :]
        ).reshape(height * width, HEAD_HIDDEN_DIM)

        tokens = self.spatial_norm(tokens + position.unsqueeze(0))

        spatial_scores = torch.einsum(
            "rth,lh->rlt",
            tokens,
            self.spatial_queries,
        ) / math.sqrt(HEAD_HIDDEN_DIM)

        spatial_attention = torch.softmax(spatial_scores, dim=-1)
        pooled_window = torch.einsum(
            "rlt,rth->rlh",
            spatial_attention,
            tokens,
        )

        window_position_hidden = self.window_position_projection(
            self._window_fourier(flat_center.float())
        )
        pooled_window = self.window_norm(
            pooled_window + window_position_hidden.unsqueeze(1)
        )

        window_representation = torch.zeros(
            batch_size * max_series * max_windows,
            NUM_LABELS,
            HEAD_HIDDEN_DIM,
            device=features.device,
            dtype=pooled_window.dtype,
        )
        window_representation[valid_window_flat] = pooled_window
        window_representation = window_representation.reshape(
            batch_size,
            max_series,
            max_windows,
            NUM_LABELS,
            HEAD_HIDDEN_DIM,
        )

        window_scores = torch.einsum(
            "bsmlh,lh->bsml",
            window_representation,
            self.window_queries,
        ) / math.sqrt(HEAD_HIDDEN_DIM)

        window_scores = window_scores.masked_fill(
            ~window_mask_effective.unsqueeze(-1),
            -1e4,
        )
        window_attention = torch.softmax(window_scores, dim=2)

        series_representation = torch.einsum(
            "bsml,bsmlh->bslh",
            window_attention,
            window_representation,
        )

        metadata_hidden = self.metadata_projection(
            torch.cat(
                [
                    self.plane_embedding(series_meta[..., 0]),
                    self.fluid_embedding(series_meta[..., 1]),
                    self.fs_embedding(series_meta[..., 2]),
                    series_cont.float(),
                ],
                dim=-1,
            )
        )

        series_representation = self.series_norm(
            series_representation + metadata_hidden.unsqueeze(2)
        )

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

        output = {"logits": logits}
        if return_attention:
            output.update(
                {
                    "spatial_attention": spatial_attention,
                    "window_attention": window_attention,
                    "series_attention": series_attention,
                    "window_mask_effective": window_mask_effective,
                    "series_mask_effective": series_mask_effective,
                }
            )
        return output


# ============================================================
# 14. MACRO-AWARE LOSSES + METRICS
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
    return per_cell.mean(dim=0).mean()


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
        raise RuntimeError("Pseudo batch contains no active target label cells")

    per_label = (per_cell * mask_float).sum(dim=0) / counts.clamp_min(1.0)
    return per_label[active_labels].mean(), counts


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
        "macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return pd.DataFrame(rows), summary


# ============================================================
# 15. TRAINING SUPERVISION OBJECTS + CONFIG HASH
# ============================================================


def build_gold_arrays(
    store: OrthoFeatureStore,
    gold_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = store.indices_for_uids(gold_df[UID_COLUMN].astype(str).tolist())
    targets = gold_df[LABEL_COLUMNS].values.astype(np.float32)
    return indices, targets


def pseudo_dataframe_to_arrays(
    store: OrthoFeatureStore,
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
        "experiment": "W5.0",
        "backbone": "OrthoDiffusion",
        "backbone_frozen": True,
        "cache_version": CACHE_VERSION,
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "feature_timestep_by_plane": FEATURE_TIMESTEP_BY_PLANE,
        "feature_block_by_plane": FEATURE_BLOCK_BY_PLANE,
        "feature_shape": [
            ORTHO_FEATURE_CHANNELS,
            ORTHO_FEATURE_HEIGHT,
            ORTHO_FEATURE_WIDTH,
        ],
        "deterministic_diffusion_noise": True,
        "max_windows_per_series": MAX_WINDOWS_PER_SERIES,
        "window_center_fractions": list(WINDOW_CENTER_FRACTIONS),
        "all_eligible_series": True,
        "min_slices_per_series": MIN_SLICES_PER_SERIES,
        "spatial_attention": True,
        "window_attention": True,
        "series_attention": True,
        "hidden_dim": HEAD_HIDDEN_DIM,
        "head_dropout": HEAD_DROPOUT,
        "window_dropout": WINDOW_DROPOUT,
        "series_dropout": SERIES_DROPOUT,
        "epochs": HEAD_EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "gold_batch_size": GOLD_BATCH_SIZE,
        "pseudo_batch_size": PSEUDO_BATCH_SIZE,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
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
    store: OrthoFeatureStore,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []

    for start in range(0, len(indices), batch_size):
        part = indices[start : start + batch_size]
        batch = store.make_batch(part, TRAIN_DEVICE)
        with autocast_context(TRAIN_DEVICE):
            logits = model(**batch)["logits"]
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
        del batch, logits

    if not outputs:
        return np.zeros((0, NUM_LABELS), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32)


# ============================================================
# 17. TRAIN ONE OUTER FOLD
# ============================================================


def _sample_rows(
    rng: np.random.Generator,
    available: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(available) == 0:
        raise RuntimeError("Cannot sample from an empty index set")
    positions = rng.integers(0, len(available), size=batch_size)
    return available[positions]


def train_one_fold(
    variant: str,
    fold: int,
    store: OrthoFeatureStore,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
    w23_root: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    variant_result_root = RESULT_ROOT / variant
    variant_checkpoint_root = CHECKPOINT_ROOT / variant
    variant_result_root.mkdir(parents=True, exist_ok=True)
    variant_checkpoint_root.mkdir(parents=True, exist_ok=True)

    history_path = variant_result_root / f"fold_{fold}_history.csv"
    prediction_path = variant_result_root / f"fold_{fold}_epoch_predictions.csv"
    pseudo_audit_path = variant_result_root / f"fold_{fold}_pseudo_audit.csv"
    checkpoint_path = variant_checkpoint_root / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"

    config_hash = training_config_hash()

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("config_hash") != config_hash:
            raise RuntimeError(
                f"Existing {checkpoint_path} has a different W5 config hash. "
                "Use a new W50_WORK_ROOT or deliberately remove stale outputs."
            )
        if history_path.exists() and prediction_path.exists():
            log(
                f"{variant} fold {fold}: complete checkpoint found -> skipping retrain."
            )
            history_df = pd.read_csv(history_path)
            prediction_df = pd.read_csv(prediction_path)
            pseudo_audit = (
                pd.read_csv(pseudo_audit_path) if pseudo_audit_path.exists() else None
            )
            return history_df, prediction_df, pseudo_audit

    gold_indices, gold_targets_all = build_gold_arrays(store, gold_df)
    val_rows = np.where(fold_zero == fold - 1)[0]
    train_rows = np.where(fold_zero != fold - 1)[0]

    gold_train_indices = gold_indices[train_rows]
    val_indices = gold_indices[val_rows]

    gold_train_targets = gold_targets_all[train_rows]
    val_targets = gold_targets_all[val_rows].astype(np.int64)
    gold_val_uids = gold_df.iloc[val_rows][UID_COLUMN].astype(str).tolist()

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
            raise RuntimeError("fold_safe_weak requires W2.3 outputs")

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
            raise RuntimeError(f"W2.3 fold {fold} produced no selected pseudo studies")

        pseudo_audit.to_csv(pseudo_audit_path, index=False)

    elif variant != "gold_only":
        raise ValueError(f"Unknown training variant: {variant}")

    seed = 50_000 + fold
    seed_everything(seed)

    model = OrthoSpatialHierarchicalDiagnosisHead().to(TRAIN_DEVICE)
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

    gold_rng = np.random.default_rng(seed + 1_000)
    pseudo_rng = np.random.default_rng(seed + 2_000)

    history_rows: List[Dict[str, Any]] = []
    epoch_prediction_rows: List[Dict[str, Any]] = []

    log("\n" + "-" * 88)
    log(f"{variant} | W5 fold {fold}/{NUM_FOLDS}")
    log("-" * 88)
    log(f"Gold train studies     : {len(gold_train_indices)}")
    log(f"Gold validation studies: {len(val_indices)}")
    log(f"Gold batch             : {GOLD_BATCH_SIZE}")

    if variant == "fold_safe_weak":
        log(f"Pseudo selected studies: {len(pseudo_indices)}")
        log(f"Pseudo selected cells  : {int(pseudo_masks.sum())}")
        log(f"Pseudo batch           : {PSEUDO_BATCH_SIZE}")
        log(
            f"Loss authority         : gold={GOLD_AUTHORITY:g}, "
            f"pseudo={PSEUDO_AUTHORITY:g}"
        )

    started = time.time()

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        epoch_gold_loss = 0.0
        epoch_pseudo_loss = 0.0
        epoch_total_loss = 0.0
        epoch_pseudo_label_counts = np.zeros(NUM_LABELS, dtype=np.int64)

        for _step in range(STEPS_PER_EPOCH):
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

            epoch_gold_loss += float(gold_loss.detach().cpu())
            epoch_pseudo_loss += float(pseudo_loss.detach().cpu())
            epoch_total_loss += float(total_loss.detach().cpu())
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

        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        pd.DataFrame(epoch_prediction_rows).to_csv(prediction_path, index=False)

        if epoch == 1 or epoch % 4 == 0 or epoch == HEAD_EPOCHS:
            memory_text = ""
            if TRAIN_DEVICE.type == "cuda":
                memory_text = f" gpu={torch.cuda.max_memory_allocated(TRAIN_DEVICE) / (1024**3):.2f}GB"
            log(
                f"{variant:14s} fold={fold} "
                f"epoch={epoch:02d}/{HEAD_EPOCHS} "
                f"gold={history_row['GoldLoss']:.4f} "
                f"pseudo={history_row['PseudoLoss']:.4f} "
                f"total={history_row['TotalLoss']:.4f} "
                f"val_auc={history_row['ValMacroAUROC']:.4f} "
                f"val_ap={history_row['ValMacroAP']:.4f} "
                f"elapsed={elapsed_string(time.time() - started)}"
                f"{memory_text}"
            )

    checkpoint = {
        "experiment": "W5.0",
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
# 18. AGGREGATE FIVE-FOLD OOF
# ============================================================


def aggregate_variant_oof(
    variant: str,
    fold_prediction_frames: List[pd.DataFrame],
    fold_history_frames: List[pd.DataFrame],
) -> Dict[str, Any]:
    variant_root = RESULT_ROOT / variant
    variant_root.mkdir(parents=True, exist_ok=True)

    prediction_df = pd.concat(fold_prediction_frames, ignore_index=True)
    history_df = pd.concat(fold_history_frames, ignore_index=True)

    prediction_df[UID_COLUMN] = prediction_df[UID_COLUMN].astype(str)
    prediction_df.to_csv(variant_root / "epoch_oof_predictions.csv", index=False)
    history_df.to_csv(variant_root / "all_fold_history.csv", index=False)

    epoch_summary_rows = []

    for epoch in range(1, HEAD_EPOCHS + 1):
        epoch_df = prediction_df[prediction_df["Epoch"] == epoch].copy()
        if len(epoch_df) != 58:
            raise RuntimeError(
                f"{variant} epoch {epoch}: expected 58 OOF rows, found {len(epoch_df)}"
            )
        if epoch_df[UID_COLUMN].duplicated().any():
            raise RuntimeError(f"{variant} epoch {epoch}: duplicate OOF UIDs")

        y_true = np.stack(
            [
                epoch_df[f"{label}__true"].values.astype(np.int64)
                for label in LABEL_COLUMNS
            ],
            axis=1,
        )
        y_prob = np.stack(
            [
                epoch_df[f"{label}__prob"].values.astype(np.float32)
                for label in LABEL_COLUMNS
            ],
            axis=1,
        )
        _, summary = metric_tables(y_true, y_prob)
        epoch_summary_rows.append(
            {
                "Epoch": epoch,
                **summary,
            }
        )

    epoch_summary_df = pd.DataFrame(epoch_summary_rows)
    epoch_summary_df.to_csv(variant_root / "epoch_oof_summary.csv", index=False)

    final_df = (
        prediction_df[prediction_df["Epoch"] == HEAD_EPOCHS]
        .copy()
        .sort_values(UID_COLUMN)
        .reset_index(drop=True)
    )

    y_true = np.stack(
        [final_df[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
        axis=1,
    )
    y_prob = np.stack(
        [
            final_df[f"{label}__prob"].values.astype(np.float32)
            for label in LABEL_COLUMNS
        ],
        axis=1,
    )

    per_label, summary = metric_tables(y_true, y_prob)
    summary.update(
        {
            "final_epoch": HEAD_EPOCHS,
            "n_oof_studies": len(final_df),
        }
    )

    final_df.to_csv(variant_root / "oof_predictions.csv", index=False)
    per_label.to_csv(variant_root / "oof_per_label_metrics.csv", index=False)

    fold_metric_rows = []
    for fold in range(1, NUM_FOLDS + 1):
        fold_df = final_df[final_df["Fold"] == fold]
        fold_true = np.stack(
            [
                fold_df[f"{label}__true"].values.astype(np.int64)
                for label in LABEL_COLUMNS
            ],
            axis=1,
        )
        fold_prob = np.stack(
            [
                fold_df[f"{label}__prob"].values.astype(np.float32)
                for label in LABEL_COLUMNS
            ],
            axis=1,
        )
        _, fold_summary = metric_tables(fold_true, fold_prob)
        fold_metric_rows.append({"Fold": fold, **fold_summary})

    pd.DataFrame(fold_metric_rows).to_csv(
        variant_root / "fold_metrics.csv",
        index=False,
    )

    (variant_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log(f"\n{variant} final pooled OOF:")
    log(json.dumps(summary, indent=2, allow_nan=True))

    return {
        "summary": summary,
        "per_label": per_label,
        "oof": final_df,
        "epoch_summary": epoch_summary_df,
    }


# ============================================================
# 19. PAIRED STUDY BOOTSTRAP
# ============================================================


def macro_auc_from_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for label_index in range(NUM_LABELS):
        auc = safe_auc(y_true[:, label_index], y_prob[:, label_index])
        if np.isfinite(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_macro_auc_delta(
    y_true: np.ndarray,
    first_prob: np.ndarray,
    second_prob: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 20260823,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        first_auc = macro_auc_from_arrays(y_true[indices], first_prob[indices])
        second_auc = macro_auc_from_arrays(y_true[indices], second_prob[indices])
        if np.isfinite(first_auc) and np.isfinite(second_auc):
            deltas.append(first_auc - second_auc)

    if not deltas:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_delta_gt_0": float("nan"),
            "valid_iterations": 0,
        }

    values = np.asarray(deltas, dtype=np.float64)
    return {
        "delta_mean": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_delta_gt_0": float(np.mean(values > 0.0)),
        "valid_iterations": int(len(values)),
    }


# ============================================================
# 20. TRAIN VARIANT / BOTH VARIANTS
# ============================================================


def train_variant(
    variant: str,
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
    store: OrthoFeatureStore,
    w23_root: Optional[Path],
) -> Dict[str, Any]:
    log("\n" + "=" * 88)
    log(f"W5.0 TRAINING VARIANT: {variant}")
    log("=" * 88)

    fold_histories = []
    fold_predictions = []
    pseudo_audits = []

    for fold in range(1, NUM_FOLDS + 1):
        history, prediction, pseudo_audit = train_one_fold(
            variant,
            fold,
            store,
            gold_df,
            fold_zero,
            w23_root,
        )
        fold_histories.append(history)
        fold_predictions.append(prediction)
        if pseudo_audit is not None:
            pseudo_audits.append(pseudo_audit)

    result = aggregate_variant_oof(
        variant,
        fold_predictions,
        fold_histories,
    )

    if pseudo_audits:
        pd.concat(pseudo_audits, ignore_index=True).to_csv(
            RESULT_ROOT / variant / "pseudo_audit_all_folds.csv",
            index=False,
        )

    return result


def train_all_variants(
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
) -> Dict[str, Any]:
    w23_root = discover_w23_root()
    verify_w23_outer_folds(w23_root, gold_df, fold_zero)
    log(f"W2.3 root             : {w23_root}")

    store = OrthoFeatureStore(train_df)

    gold_result = train_variant(
        "gold_only",
        train_df,
        gold_df,
        fold_zero,
        store,
        None,
    )
    weak_result = train_variant(
        "fold_safe_weak",
        train_df,
        gold_df,
        fold_zero,
        store,
        w23_root,
    )

    gold_oof = gold_result["oof"]
    weak_oof = weak_result["oof"]

    if gold_oof[UID_COLUMN].tolist() != weak_oof[UID_COLUMN].tolist():
        raise RuntimeError("W5 gold/weak OOF UID ordering mismatch")

    y_true = np.stack(
        [gold_oof[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
        axis=1,
    )
    gold_prob = np.stack(
        [
            gold_oof[f"{label}__prob"].values.astype(np.float32)
            for label in LABEL_COLUMNS
        ],
        axis=1,
    )
    weak_prob = np.stack(
        [
            weak_oof[f"{label}__prob"].values.astype(np.float32)
            for label in LABEL_COLUMNS
        ],
        axis=1,
    )

    bootstrap = bootstrap_macro_auc_delta(
        y_true,
        weak_prob,
        gold_prob,
    )

    gold_per_label = gold_result["per_label"].set_index("Label")
    weak_per_label = weak_result["per_label"].set_index("Label")

    comparison_rows = []
    for label in LABEL_COLUMNS:
        comparison_rows.append(
            {
                "Label": label,
                "GoldAUROC": float(gold_per_label.loc[label, "AUROC"]),
                "WeakAUROC": float(weak_per_label.loc[label, "AUROC"]),
                "WeakMinusGold": float(
                    weak_per_label.loc[label, "AUROC"]
                    - gold_per_label.loc[label, "AUROC"]
                ),
                "GoldAP": float(gold_per_label.loc[label, "AveragePrecision"]),
                "WeakAP": float(weak_per_label.loc[label, "AveragePrecision"]),
            }
        )

    pd.DataFrame(comparison_rows).to_csv(
        RESULT_ROOT / "W5_0_PER_LABEL_COMPARISON.csv",
        index=False,
    )

    comparison = {
        "gold_only": gold_result["summary"],
        "fold_safe_weak": weak_result["summary"],
        "weak_minus_gold_AUROC": (
            weak_result["summary"]["macro_AUROC"]
            - gold_result["summary"]["macro_AUROC"]
        ),
        "paired_bootstrap_weak_minus_gold": bootstrap,
        "W4_weak_OOF_reference": W4_WEAK_REFERENCE_AUROC,
        "W4_public_weak_LB_reference": W4_PUBLIC_WEAK_LB,
    }

    (RESULT_ROOT / "W5_0_COMPARISON.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log("\n" + "=" * 88)
    log("W5.0 TRAINING COMPLETE")
    log("=" * 88)
    log(f"Gold-only Ortho AUROC : {gold_result['summary']['macro_AUROC']:.6f}")
    log(f"Weak Ortho AUROC      : {weak_result['summary']['macro_AUROC']:.6f}")
    log(f"Weak - Gold AUROC     : " f"{comparison['weak_minus_gold_AUROC']:+.6f}")
    log(
        f"Bootstrap 95% CI      : "
        f"[{bootstrap['ci_low']:+.6f}, {bootstrap['ci_high']:+.6f}]"
    )
    log(f"P(delta>0)            : {bootstrap['p_delta_gt_0']:.3f}")

    # Automatic complementarity comparison is optional.  It is performed
    # only when a W4 reference artifact can be found.
    try:
        complementarity = compare_w5_to_w4(weak_oof)
    except FileNotFoundError as exc:
        log(f"W4 comparison skipped : {exc}")
        complementarity = None

    return {
        "gold_only": gold_result,
        "fold_safe_weak": weak_result,
        "comparison": comparison,
        "w4_complementarity": complementarity,
    }


def train_single_variant(
    variant: str,
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    fold_zero: np.ndarray,
) -> Dict[str, Any]:
    store = OrthoFeatureStore(train_df)

    if variant == "gold_only":
        w23_root = None
    elif variant == "fold_safe_weak":
        w23_root = discover_w23_root()
        verify_w23_outer_folds(w23_root, gold_df, fold_zero)
        log(f"W2.3 root             : {w23_root}")
    else:
        raise ValueError(variant)

    return train_variant(
        variant,
        train_df,
        gold_df,
        fold_zero,
        store,
        w23_root,
    )


# ============================================================
# 21. W4 CURIA REFERENCE + COMPLEMENTARITY ANALYSIS
# ============================================================


def looks_like_w4_training_root(path: Path) -> bool:
    """Full W4 training root containing the weak OOF predictions."""
    return (path / "results" / "fold_safe_weak" / "oof_predictions.csv").exists()


def discover_w4_training_root() -> Path:
    candidates: List[Path] = []

    if EXPLICIT_W4_ROOT:
        explicit = Path(EXPLICIT_W4_ROOT)
        candidates.extend([explicit, explicit / "rsna_w4_0_curia2"])

    candidates.append(Path("/kaggle/working/rsna_w4_0_curia2"))

    for root in shallow_kaggle_dirs(max_depth=4):
        candidates.extend([root, root / "rsna_w4_0_curia2"])

    for candidate in dict.fromkeys(candidates):
        if looks_like_w4_training_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Full W4 training output was not found. Attach the W4 training artifact "
        "and set W50_W4_ROOT to a directory containing "
        "results/fold_safe_weak/oof_predictions.csv."
    )


def _validate_oof_schema(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {UID_COLUMN, "Fold"}
    for label in LABEL_COLUMNS:
        required.add(f"{label}__true")
        required.add(f"{label}__prob")

    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} OOF file missing columns: {sorted(missing)}")

    output = frame.copy()
    output[UID_COLUMN] = output[UID_COLUMN].astype(str)

    if len(output) != 58:
        raise RuntimeError(f"{name}: expected 58 OOF rows, found {len(output)}")
    if output[UID_COLUMN].duplicated().any():
        raise RuntimeError(f"{name}: duplicate OOF StudyInstanceUID")

    return output.sort_values(UID_COLUMN).reset_index(drop=True)


def _pairwise_ranking_disagreement(
    y_true: np.ndarray,
    first_prob: np.ndarray,
    second_prob: np.ndarray,
) -> Tuple[float, int]:
    """
    Fraction of positive-negative pairs for which two models disagree on
    which study should rank higher. Ties are excluded rather than arbitrarily
    broken. This is directly relevant to AUROC complementarity.
    """
    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(first_prob, dtype=np.float64)
    b = np.asarray(second_prob, dtype=np.float64)

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    if len(positive) == 0 or len(negative) == 0:
        return float("nan"), 0

    a_diff = a[positive][:, None] - a[negative][None, :]
    b_diff = b[positive][:, None] - b[negative][None, :]

    valid = (
        np.isfinite(a_diff)
        & np.isfinite(b_diff)
        & (np.abs(a_diff) > 1e-12)
        & (np.abs(b_diff) > 1e-12)
    )

    if not valid.any():
        return float("nan"), 0

    disagreement = np.sign(a_diff[valid]) != np.sign(b_diff[valid])
    return float(np.mean(disagreement)), int(valid.sum())


def compare_w5_to_w4(
    w5_oof: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compare W5 fold_safe_weak OOF against W4 Curia fold_safe_weak OOF.

    This is deliberately diagnostic. It computes a fixed 50/50 blend but
    does NOT search per-label weights on the 58-study gold set.
    """
    if w5_oof is None:
        w5_path = RESULT_ROOT / "fold_safe_weak" / "oof_predictions.csv"
        if not w5_path.exists():
            raise FileNotFoundError(
                "W5 weak OOF not found. Run run_w50_ortho('train_weak') or "
                "run_w50_ortho('train') first."
            )
        w5_oof = pd.read_csv(w5_path)

    w5 = _validate_oof_schema(w5_oof, "W5")

    w4_root = discover_w4_training_root()
    w4_path = w4_root / "results" / "fold_safe_weak" / "oof_predictions.csv"
    w4 = _validate_oof_schema(pd.read_csv(w4_path), "W4")

    if w5[UID_COLUMN].tolist() != w4[UID_COLUMN].tolist():
        raise RuntimeError("W4/W5 OOF StudyInstanceUID sets or ordering differ")

    if not np.array_equal(
        w5["Fold"].astype(int).values,
        w4["Fold"].astype(int).values,
    ):
        raise RuntimeError("W4/W5 outer-fold assignments differ")

    y_true = np.stack(
        [w5[f"{label}__true"].values.astype(np.int64) for label in LABEL_COLUMNS],
        axis=1,
    )
    w5_prob = np.stack(
        [w5[f"{label}__prob"].values.astype(np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )
    w4_prob = np.stack(
        [w4[f"{label}__prob"].values.astype(np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    for label_index, label in enumerate(LABEL_COLUMNS):
        w4_truth = w4[f"{label}__true"].values.astype(np.int64)
        if not np.array_equal(y_true[:, label_index], w4_truth):
            raise RuntimeError(f"W4/W5 gold truth differs for {label}")

    blend_prob = 0.5 * w4_prob + 0.5 * w5_prob

    w5_metrics, w5_summary = metric_tables(y_true, w5_prob)
    w4_metrics, w4_summary = metric_tables(y_true, w4_prob)
    blend_metrics, blend_summary = metric_tables(y_true, blend_prob)

    rows = []
    for label_index, label in enumerate(LABEL_COLUMNS):
        truth = y_true[:, label_index]
        p4 = w4_prob[:, label_index]
        p5 = w5_prob[:, label_index]
        residual4 = p4 - truth.astype(np.float32)
        residual5 = p5 - truth.astype(np.float32)
        ranking_disagreement, pair_count = _pairwise_ranking_disagreement(truth, p4, p5)

        rows.append(
            {
                "Label": label,
                "W4AUROC": float(w4_metrics.loc[label_index, "AUROC"]),
                "W5AUROC": float(w5_metrics.loc[label_index, "AUROC"]),
                "EqualBlendAUROC": float(blend_metrics.loc[label_index, "AUROC"]),
                "W5MinusW4AUROC": float(
                    w5_metrics.loc[label_index, "AUROC"]
                    - w4_metrics.loc[label_index, "AUROC"]
                ),
                "BlendMinusBestSingleAUROC": float(
                    blend_metrics.loc[label_index, "AUROC"]
                    - max(
                        w4_metrics.loc[label_index, "AUROC"],
                        w5_metrics.loc[label_index, "AUROC"],
                    )
                ),
                "W4AP": float(w4_metrics.loc[label_index, "AveragePrecision"]),
                "W5AP": float(w5_metrics.loc[label_index, "AveragePrecision"]),
                "EqualBlendAP": float(
                    blend_metrics.loc[label_index, "AveragePrecision"]
                ),
                "PredictionPearson": finite_corr(p4, p5, spearman=False),
                "PredictionSpearman": finite_corr(p4, p5, spearman=True),
                "ResidualPearson": finite_corr(residual4, residual5, spearman=False),
                "MeanAbsolutePredictionDifference": float(np.mean(np.abs(p4 - p5))),
                "PairwiseRankingDisagreement": ranking_disagreement,
                "RankingPairCount": pair_count,
            }
        )

    per_label = pd.DataFrame(rows)
    per_label.to_csv(
        RESULT_ROOT / "W5_0_W4_COMPLEMENTARITY_PER_LABEL.csv",
        index=False,
    )

    w5_minus_w4_bootstrap = bootstrap_macro_auc_delta(
        y_true, w5_prob, w4_prob, seed=20260824
    )
    blend_minus_w4_bootstrap = bootstrap_macro_auc_delta(
        y_true, blend_prob, w4_prob, seed=20260825
    )
    blend_minus_w5_bootstrap = bootstrap_macro_auc_delta(
        y_true, blend_prob, w5_prob, seed=20260826
    )

    macro = {
        "W4_macro_AUROC": float(w4_summary["macro_AUROC"]),
        "W5_macro_AUROC": float(w5_summary["macro_AUROC"]),
        "equal_blend_macro_AUROC": float(blend_summary["macro_AUROC"]),
        "W5_minus_W4_AUROC": float(
            w5_summary["macro_AUROC"] - w4_summary["macro_AUROC"]
        ),
        "blend_minus_best_single_AUROC": float(
            blend_summary["macro_AUROC"]
            - max(w4_summary["macro_AUROC"], w5_summary["macro_AUROC"])
        ),
        "mean_prediction_Pearson": float(
            np.nanmean(per_label["PredictionPearson"].values)
        ),
        "mean_prediction_Spearman": float(
            np.nanmean(per_label["PredictionSpearman"].values)
        ),
        "mean_residual_Pearson": float(np.nanmean(per_label["ResidualPearson"].values)),
        "mean_pairwise_ranking_disagreement": float(
            np.nanmean(per_label["PairwiseRankingDisagreement"].values)
        ),
        "mean_absolute_prediction_difference": float(
            np.nanmean(per_label["MeanAbsolutePredictionDifference"].values)
        ),
    }

    payload = {
        "w4_root": str(w4_root),
        "w4_oof_path": str(w4_path),
        "w5_oof_path": str(RESULT_ROOT / "fold_safe_weak" / "oof_predictions.csv"),
        "note": (
            "Equal blend is a fixed diagnostic only. No per-label or scalar "
            "blend-weight search is performed on the 58 gold studies."
        ),
        "macro": macro,
        "W5_minus_W4_bootstrap": w5_minus_w4_bootstrap,
        "equal_blend_minus_W4_bootstrap": blend_minus_w4_bootstrap,
        "equal_blend_minus_W5_bootstrap": blend_minus_w5_bootstrap,
    }

    (RESULT_ROOT / "W5_0_W4_COMPLEMENTARITY.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    blend_oof = w5[[UID_COLUMN, "Fold"]].copy()
    for label_index, label in enumerate(LABEL_COLUMNS):
        blend_oof[f"{label}__true"] = y_true[:, label_index]
        blend_oof[f"{label}__prob"] = blend_prob[:, label_index]
    blend_oof.to_csv(
        RESULT_ROOT / "W5_0_W4_EQUAL_BLEND_OOF.csv",
        index=False,
    )

    log("\n" + "=" * 88)
    log("W5.0 ORTHO vs W4 CURIA COMPLEMENTARITY")
    log("=" * 88)
    log(f"W4 weak macro AUROC   : {macro['W4_macro_AUROC']:.6f}")
    log(f"W5 weak macro AUROC   : {macro['W5_macro_AUROC']:.6f}")
    log(f"50/50 blend AUROC     : {macro['equal_blend_macro_AUROC']:.6f}")
    log(f"Blend - best single   : {macro['blend_minus_best_single_AUROC']:+.6f}")
    log(f"Mean prediction rho   : {macro['mean_prediction_Pearson']:.4f}")
    log(f"Mean residual rho     : {macro['mean_residual_Pearson']:.4f}")
    log(
        f"Ranking disagreement  : " f"{macro['mean_pairwise_ranking_disagreement']:.4f}"
    )
    log(
        "No label-wise blend tuning was performed; this comparison is "
        "for architecture selection only."
    )

    return payload


# ============================================================
# 22. STATUS / INTEGRITY REPORT
# ============================================================


def _safe_discovery(function) -> Tuple[Optional[Path], Optional[str]]:
    try:
        return function(), None
    except Exception as exc:
        return None, repr(exc)


def w50_status() -> Dict[str, Any]:
    train_df, series_df, gold_df, fold_zero = load_training_tables()

    code_root, code_error = _safe_discovery(discover_ortho_code_root)
    weights_root, weights_error = _safe_discovery(discover_ortho_weights_root)

    ortho_identity = None
    ortho_identity_error = None
    if code_root is not None and weights_root is not None:
        try:
            ortho_identity = validate_ortho_identity(code_root, weights_root)
        except Exception as exc:
            ortho_identity_error = repr(exc)

    cached = 0
    cached_series = 0
    cached_windows = 0
    cache_bytes = 0

    for uid in train_df[UID_COLUMN].astype(str):
        path = study_cache_path(uid)
        if cache_file_is_usable(path, uid):
            cached += 1
            cache_bytes += path.stat().st_size
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                cached_series += int(payload["features"].shape[0])
                cached_windows += int(payload["window_mask"].sum().item())
            except Exception:
                pass

    w23_root, w23_error = _safe_discovery(discover_w23_root)
    w4_root, w4_error = _safe_discovery(discover_w4_training_root)

    payload = {
        "experiment": "W5.0 OrthoDiffusion spatial hierarchical",
        "device": str(TRAIN_DEVICE),
        "gpu_count": GPU_COUNT,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(GPU_COUNT)],
        "train_studies": int(len(train_df)),
        "train_series": int(len(series_df)),
        "gold_studies": int(len(gold_df)),
        "fold_sha256": fold_assignment_sha256(
            pd.DataFrame(
                {
                    UID_COLUMN: gold_df[UID_COLUMN].astype(str),
                    "OuterFold": fold_zero + 1,
                }
            )
        ),
        "cached_studies": cached,
        "cache_complete": cached == len(train_df),
        "cached_series": cached_series,
        "cached_windows": cached_windows,
        "cache_file_bytes_gb": cache_bytes / (1024**3),
        "cache_version": CACHE_VERSION,
        "ortho_code_root": str(code_root) if code_root else None,
        "ortho_code_error": code_error,
        "ortho_weights_root": str(weights_root) if weights_root else None,
        "ortho_weights_error": weights_error,
        "ortho_identity": ortho_identity,
        "ortho_identity_error": ortho_identity_error,
        "w23_root": str(w23_root) if w23_root else None,
        "w23_error": w23_error,
        "w4_root": str(w4_root) if w4_root else None,
        "w4_error": w4_error,
        "work_root": str(WORK_ROOT),
        "head_config_hash": training_config_hash(),
        "feature_timestep_by_plane": FEATURE_TIMESTEP_BY_PLANE,
        "feature_block_by_plane": FEATURE_BLOCK_BY_PLANE,
        "window_center_fractions": list(WINDOW_CENTER_FRACTIONS),
        "max_windows_per_series": MAX_WINDOWS_PER_SERIES,
    }

    log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


# ============================================================
# 23. NOTEBOOK / CLI ORCHESTRATION
# ============================================================


def _require_complete_cache(train_df: pd.DataFrame) -> None:
    missing = [
        str(uid)
        for uid in train_df[UID_COLUMN].astype(str)
        if not cache_file_is_usable(study_cache_path(str(uid)), str(uid))
    ]
    if missing:
        raise RuntimeError(
            f"W5 feature cache is incomplete: {len(missing)} studies missing or "
            f"incompatible. Run run_w50_ortho('cache') first. "
            f"Examples: {missing[:5]}"
        )


def run_w50_ortho(mode: str = "status"):
    mode = str(mode).strip().lower()
    valid_modes = {
        "status",
        "cache",
        "train_gold",
        "train_weak",
        "train",
        "compare_w4",
        "all",
    }

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    if mode == "status":
        return w50_status()

    train_df, series_df, gold_df, fold_zero = load_training_tables()

    if mode == "cache":
        return build_feature_cache(train_df, series_df)

    if mode == "all":
        cache_summary = build_feature_cache(train_df, series_df)
        _require_complete_cache(train_df)
        training = train_all_variants(train_df, gold_df, fold_zero)
        return {
            "cache": cache_summary,
            "training": training,
        }

    if mode == "compare_w4":
        return compare_w5_to_w4()

    _require_complete_cache(train_df)

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

    return train_all_variants(train_df, gold_df, fold_zero)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RSNA W5.0 OrthoDiffusion 3-D spatial fold-safe training"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "status",
            "cache",
            "train_gold",
            "train_weak",
            "train",
            "compare_w4",
            "all",
        ],
        default="status",
    )
    return parser.parse_args()


def main():
    # args = parse_args()
    # run_w50_ortho(args.mode)
    run_w50_ortho("status")
    run_w50_ortho("cache")
    run_w50_ortho("train_gold")
    run_w50_ortho("train_weak")
    run_w50_ortho("train")
    run_w50_ortho("compare_w4")
    run_w50_ortho("all")


if __name__ == "__main__":
    main()
