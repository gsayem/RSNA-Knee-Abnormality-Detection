#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection
W6 Curia — W2.6 production-teacher supervision + intra-slice spatial Curia

Purpose
-------
This is the W6 successor to the canonical W4 Curia pipeline.

W6.0 (controlled supervision update)
    Exact W4 Curia slice-CLS representation and W4 hierarchical head.
    The old W2.3 weak labels are replaced by the finalized W2.6-P production
    teacher (16/17/18 files from rsna_w2_6p_fast).

W6.1 (spatial Curia)
    Exact W4 DICOM preprocessing, slice selection, orientation, and Curia
    encoder are retained. In addition to each slice CLS token, Curia patch
    tokens are compacted to a configurable GxG grid (default 3x3). A
    label-conditioned spatial attention stage pools those grid tokens before
    the existing slice -> series -> study hierarchy.

Standalone design
-----------------
W6 is fully self-contained. It does NOT import, execute, or shell out to W4 or
any other project script. The canonical W4 preprocessing logic needed by W6
(DICOM decode/repair, physical slice ordering, anatomical orientation, slice
selection, Curia identity validation, Curia loading, and study encoding) is
implemented directly in this file.

The only W4 artifact W6 may reuse is the existing feature cache. By default:

  * local:  output/results/rsna_w4_0_curia2/feature_cache[/studies]
  * Kaggle: /kaggle/working/... or an attached Kaggle Dataset containing the
            same feature_cache[/studies] tree

Set W6_W4_CACHE_ROOT to override that cache path. No W4 Python file is needed.

Leakage / interpretation warning
--------------------------------
The W2.6-P teacher used all 58 gold reports as final production exemplars.
Therefore any diagnostic predictions on those 58 studies after training with
W2.6-P are NOT pristine OOF evidence. This script labels them explicitly as
non-pristine diagnostics. Final hidden-test inference remains MRI-only.

Portable layout
---------------
Local default (same convention as W2.6-P fast):

    PROJECT_ROOT/
      input/
        train.csv
        train_series.csv
        train_series/<StudyUID>/<SeriesUID>/*.dcm
        test.csv
        test_series.csv
        test_series/<StudyUID>/<SeriesUID>/*.dcm
        sample_submission.csv
      models/
        curia-2/
      output/results/
        rsna_w2_6p_fast/results/{16,17,18,...}
        rsna_w4_0_curia2/feature_cache/studies/*.pt   # optional reuse
        rsna_w6_curia/...

Kaggle default:
    competition data from /kaggle/input/competitions/rsna-knee-abnormality-detection
    output under /kaggle/working/rsna_w6_curia
    Curia weights, W2.6-P outputs, and an optional W4 feature cache are
    discovered shallowly under /kaggle/input or /kaggle/working, or can be set
    explicitly with environment variables. No external Python script is used.

Recommended sequence
--------------------
    run_w6("status")
    run_w6("validate_teacher")

    # W6.0 — first controlled experiment
    run_w6("cache_w60", accelerator="localGPU")
    run_w6("train_w60_cv", accelerator="localGPU")
    run_w6("train_w60_full", accelerator="localGPU")

    # W6.1 — only after W6.0
    run_w6("cache_w61", accelerator="localGPU")
    run_w6("train_w61_cv", accelerator="localGPU")
    run_w6("train_w61_full", accelerator="localGPU")

    # Final test prediction. If both variants exist, the default is rank-mean
    # across their full-fit seed ensembles. Use W6_FINAL_VARIANTS=w60 to submit
    # W6.0 alone for the clean ablation.
    run_w6("submit", accelerator="localGPU")

CLI examples
------------
    python rsna_w6_curia_teacher_spatial.py status
    python rsna_w6_curia_teacher_spatial.py check_curia --accelerator localGPU
    python rsna_w6_curia_teacher_spatial.py cache_w60 --accelerator localGPU
    python rsna_w6_curia_teacher_spatial.py train_w60_cv --accelerator localGPU
    python rsna_w6_curia_teacher_spatial.py cache_w61 --accelerator localGPU
    python rsna_w6_curia_teacher_spatial.py submit --accelerator localGPU

Important
---------
No code can guarantee a particular Kaggle leaderboard score. W6.0 and W6.1 are
implemented as controlled project stages; use their diagnostics plus actual
leaderboard submissions to decide whether W6.2 / final ensembling is warranted.
"""

# ============================================================
# 0. IMPORTS / EARLY ALLOCATOR SETTINGS
# ============================================================

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import gc
import hashlib
import inspect
import json
import math
import random
import shutil
import sys
import tempfile
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# ============================================================
# 1. CONSTANTS / PORTABLE PATHS
# ============================================================

UID = "StudyInstanceUID"
SERIES_UID = "SeriesInstanceUID"

LABELS = [
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
NUM_LABELS = len(LABELS)

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

EXPECTED_CURIA_HIDDEN = 768
EXPECTED_CURIA_IMAGE = 512
EXPECTED_CURIA_PATCH = 16
EXPECTED_CURIA_PATCHES = (EXPECTED_CURIA_IMAGE // EXPECTED_CURIA_PATCH) ** 2
EXPECTED_W4_CACHE_VERSION = (
    "w4_0_curia2_cls_allseries_24slice_canonical_orientation_slow_processor_v1"
)

# Canonical Curia/W4 identity and preprocessing constants copied into W6 so
# this script is completely independent from the W4 Python implementation.
CURIA_REPO_ID = "raidium/curia-2"
CURIA_REVISION = os.environ.get(
    "W6_CURIA_REVISION", "645f566dd9e002505691178917cee265b491c7f7"
).strip()
EXPECTED_CURIA_MODEL_SHA256 = (
    "403a02e27531d2858ecd1e9b1ec2d5ea" "7bfa909f10ff0d9e8416090a6a8c96ef"
)
ALLOW_CURIA_SHA_MISMATCH = os.environ.get(
    "W6_ALLOW_CURIA_SHA_MISMATCH", "0"
).strip().lower() in {"1", "true", "yes"}

# Local fallback for safetensors mmap/filesystem incompatibilities. The canonical
# Curia weight is only ~344 MB, so staging it on the OS-native temporary filesystem
# is inexpensive compared with MRI feature extraction.
CURIA_STAGING_ROOT = Path(
    os.environ.get(
        "W6_CURIA_STAGING_ROOT",
        str(Path(tempfile.gettempdir()) / "rsna_w6_curia_staging"),
    )
).expanduser()

CURIA_HIDDEN_DIM = EXPECTED_CURIA_HIDDEN
CURIA_IMAGE_SIZE = EXPECTED_CURIA_IMAGE
CURIA_PATCH_SIZE = EXPECTED_CURIA_PATCH
CURIA_NUM_CHANNELS = 1
MAX_SLICES_PER_SERIES = int(os.environ.get("W6_MAX_SLICES_PER_SERIES", "24"))
ORIENTATION_MIN_ALIGNMENT = float(
    os.environ.get("W6_ORIENTATION_MIN_ALIGNMENT", "0.70")
)
GEOMETRY_PLANE_CONFIDENCE = float(
    os.environ.get("W6_GEOMETRY_PLANE_CONFIDENCE", "0.80")
)
UID_COLUMN = UID
SERIES_UID_COLUMN = SERIES_UID

PLANE_TO_INDEX = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
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

W26_PROB_FILE = "16_final_hybrid_probabilities_wide.csv"
W26_WEIGHT_FILE = "17_recommended_teacher_weights_wide.csv"
W26_MASK_FILE = "18_recommended_teacher_mask_wide.csv"
W26_PROD_SUMMARY = "19_production_summary.json"
W26_VALIDATION_SUMMARY = "20_validation_summary.json"


def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


SCRIPT_DIR = _script_dir()
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    PROJECT_ROOT = (
        Path(os.environ.get("W6_PROJECT_ROOT", "/kaggle/working"))
        .expanduser()
        .resolve()
    )
    DATA_ROOT = (
        Path(
            os.environ.get(
                "W6_DATA_ROOT",
                "/kaggle/input/competitions/rsna-knee-abnormality-detection",
            )
        )
        .expanduser()
        .resolve()
    )
    OUTPUT_ROOT = (
        Path(os.environ.get("W6_OUTPUT_ROOT", "/kaggle/working/rsna_w6_curia"))
        .expanduser()
        .resolve()
    )
else:
    PROJECT_ROOT = (
        Path(
            os.environ.get("W6_PROJECT_ROOT", str((SCRIPT_DIR / ".." / "..").resolve()))
        )
        .expanduser()
        .resolve()
    )
    DATA_ROOT = (
        Path(os.environ.get("W6_DATA_ROOT", str(PROJECT_ROOT / "input")))
        .expanduser()
        .resolve()
    )
    OUTPUT_ROOT = (
        Path(
            os.environ.get(
                "W6_OUTPUT_ROOT",
                str(PROJECT_ROOT / "output" / "results" / "rsna_w6_curia"),
            )
        )
        .expanduser()
        .resolve()
    )

TRAIN_CSV = (
    Path(os.environ.get("W6_TRAIN_CSV", str(DATA_ROOT / "train.csv")))
    .expanduser()
    .resolve()
)
TRAIN_SERIES_CSV = (
    Path(os.environ.get("W6_TRAIN_SERIES_CSV", str(DATA_ROOT / "train_series.csv")))
    .expanduser()
    .resolve()
)
TRAIN_SERIES_ROOT = (
    Path(os.environ.get("W6_TRAIN_SERIES_ROOT", str(DATA_ROOT / "train_series")))
    .expanduser()
    .resolve()
)

TEST_CSV = (
    Path(os.environ.get("W6_TEST_CSV", str(DATA_ROOT / "test.csv")))
    .expanduser()
    .resolve()
)
TEST_SERIES_CSV = (
    Path(os.environ.get("W6_TEST_SERIES_CSV", str(DATA_ROOT / "test_series.csv")))
    .expanduser()
    .resolve()
)
TEST_SERIES_ROOT = (
    Path(os.environ.get("W6_TEST_SERIES_ROOT", str(DATA_ROOT / "test_series")))
    .expanduser()
    .resolve()
)
SAMPLE_SUBMISSION = (
    Path(
        os.environ.get("W6_SAMPLE_SUBMISSION", str(DATA_ROOT / "sample_submission.csv"))
    )
    .expanduser()
    .resolve()
)

RESULT_ROOT = OUTPUT_ROOT / "results"
CACHE_ROOT = OUTPUT_ROOT / "cache"
W60_CACHE_ROOT = CACHE_ROOT / "w60_cls"
W60_CACHE_STUDIES = W60_CACHE_ROOT / "studies"
W61_CACHE_ROOT = CACHE_ROOT / "w61_spatial"
W61_CACHE_STUDIES = W61_CACHE_ROOT / "studies"
TEST_CACHE_ROOT = CACHE_ROOT / "test"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
SUBMISSION_ROOT = OUTPUT_ROOT / "submission"

for _path in (
    OUTPUT_ROOT,
    RESULT_ROOT,
    CACHE_ROOT,
    W60_CACHE_STUDIES,
    W61_CACHE_STUDIES,
    TEST_CACHE_ROOT,
    CHECKPOINT_ROOT,
    SUBMISSION_ROOT,
):
    _path.mkdir(parents=True, exist_ok=True)

# W6.0 exact-W4 hyperparameters unless intentionally overridden.
HEAD_HIDDEN_DIM = int(os.environ.get("W6_HEAD_HIDDEN_DIM", "384"))
HEAD_NUM_HEADS = int(os.environ.get("W6_HEAD_NUM_HEADS", "8"))
HEAD_TRANSFORMER_LAYERS = int(os.environ.get("W6_HEAD_TRANSFORMER_LAYERS", "2"))
HEAD_DROPOUT = float(os.environ.get("W6_HEAD_DROPOUT", "0.15"))
SLICE_DROPOUT = float(os.environ.get("W6_SLICE_DROPOUT", "0.05"))
SERIES_DROPOUT = float(os.environ.get("W6_SERIES_DROPOUT", "0.05"))

HEAD_EPOCHS = int(os.environ.get("W6_HEAD_EPOCHS", "24"))
STEPS_PER_EPOCH = int(os.environ.get("W6_STEPS_PER_EPOCH", "32"))
GOLD_BATCH_SIZE = int(os.environ.get("W6_GOLD_BATCH_SIZE", "16"))
PSEUDO_BATCH_SIZE_W60 = int(os.environ.get("W6_PSEUDO_BATCH_SIZE_W60", "64"))
PSEUDO_BATCH_SIZE_W61 = int(os.environ.get("W6_PSEUDO_BATCH_SIZE_W61", "16"))
VALIDATION_BATCH_SIZE = int(os.environ.get("W6_VALIDATION_BATCH_SIZE", "8"))
HEAD_MAX_LR = float(os.environ.get("W6_HEAD_MAX_LR", "0.001"))
HEAD_WEIGHT_DECAY = float(os.environ.get("W6_HEAD_WEIGHT_DECAY", "0.001"))
GRAD_CLIP_NORM = float(os.environ.get("W6_GRAD_CLIP_NORM", "5.0"))
GOLD_AUTHORITY = float(os.environ.get("W6_GOLD_AUTHORITY", "8.0"))
PSEUDO_AUTHORITY = float(os.environ.get("W6_PSEUDO_AUTHORITY", "1.0"))

NUM_FOLDS = 5
RANDOM_SEED = 42
FULLFIT_SEEDS = [
    int(x.strip())
    for x in os.environ.get("W6_FULLFIT_SEEDS", "6001,6002,6003").split(",")
    if x.strip()
]

# Teacher weights are a deliberate W2.6-P output. Set "binary" for the strictest
# W4-style ablation where every selected pseudo cell has unit weight.
TEACHER_WEIGHT_MODE = (
    os.environ.get("W6_TEACHER_WEIGHT_MODE", "recommended").strip().lower()
)
if TEACHER_WEIGHT_MODE not in {"recommended", "binary"}:
    raise ValueError("W6_TEACHER_WEIGHT_MODE must be recommended or binary")

# W6.1: compact 32x32 Curia patch grid -> 3x3 regions by default.
SPATIAL_GRID = int(os.environ.get("W6_SPATIAL_GRID", "3"))
if SPATIAL_GRID < 2 or SPATIAL_GRID > 8:
    raise ValueError("W6_SPATIAL_GRID should be in [2,8]")
SPATIAL_TOKENS = SPATIAL_GRID * SPATIAL_GRID
SPATIAL_CACHE_VERSION = f"w6_1_curia2_cls_patchgrid{SPATIAL_GRID}_exact_w4_preproc_v1"
SPATIAL_HIDDEN_DIM = int(os.environ.get("W6_SPATIAL_HIDDEN_DIM", "256"))
SPATIAL_NUM_HEADS = int(os.environ.get("W6_SPATIAL_NUM_HEADS", "8"))
SPATIAL_SLICE_LAYERS = int(os.environ.get("W6_SPATIAL_SLICE_LAYERS", "1"))
SPATIAL_DROPOUT = float(os.environ.get("W6_SPATIAL_DROPOUT", "0.12"))
SPATIAL_ENCODER_BATCH = int(os.environ.get("W6_SPATIAL_ENCODER_BATCH", "8"))

# Cache loading. W60 is small enough to eager-load on a normal workstation;
# W61 defaults to a bounded LRU because the spatial cache is much larger.
FEATURE_LRU_SIZE = int(os.environ.get("W6_FEATURE_LRU_SIZE", "96"))

ACCELERATOR = os.environ.get("W6_ACCELERATOR", "auto").strip().lower()
PRECISION = os.environ.get("W6_PRECISION", "auto").strip().lower()
GPU_ID = int(os.environ.get("W6_GPU_ID", "0"))
CACHE_GPU_IDS_TEXT = os.environ.get("W6_CACHE_GPU_IDS", "").strip()
ENCODER_BATCH = int(os.environ.get("W6_CURIA_BATCH", "16"))

FINAL_VARIANTS = [
    x.strip().lower()
    for x in os.environ.get("W6_FINAL_VARIANTS", "w60,w61").split(",")
    if x.strip()
]
FINAL_ENSEMBLE_MODE = (
    os.environ.get("W6_FINAL_ENSEMBLE_MODE", "rankmean").strip().lower()
)
if FINAL_ENSEMBLE_MODE not in {"mean", "rankmean"}:
    raise ValueError("W6_FINAL_ENSEMBLE_MODE must be mean or rankmean")


# ============================================================
# 2. LOGGING / UTILITIES
# ============================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def stable_uid_hash(uid: str) -> str:
    return hashlib.md5(str(uid).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(obj: Mapping[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "", "nan", "none"}
    bad = set(text.unique()) - true_values - false_values
    if bad:
        raise ValueError(f"Unrecognized boolean values: {sorted(bad)[:10]}")
    return text.isin(true_values)


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


def metric_tables(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    aucs: List[float] = []
    aps: List[float] = []
    for j, label in enumerate(LABELS):
        auc = safe_auc(y_true[:, j], y_prob[:, j])
        ap = safe_ap(y_true[:, j], y_prob[:, j])
        rows.append(
            {
                "Label": label,
                "PositiveCount": int(y_true[:, j].sum()),
                "AUROC": auc,
                "AveragePrecision": ap,
            }
        )
        if np.isfinite(auc):
            aucs.append(auc)
        if np.isfinite(ap):
            aps.append(ap)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return pd.DataFrame(rows), {
        "macro_AUROC": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_AP": float(np.mean(aps)) if aps else float("nan"),
        "macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    ordered = assignments.sort_values(UID).reset_index(drop=True)
    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(ordered[UID].astype(str), ordered["OuterFold"].astype(int))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def greedy_multilabel_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    # Exact canonical W4 implementation; hash is mandatory below.
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


# ============================================================
# 3. PATH DISCOVERY
# ============================================================


def shallow_dirs(root: Path, max_depth: int = 4) -> Iterable[Path]:
    if not root.exists():
        return []
    base_parts = len(root.parts)
    out: List[Path] = []
    queue = [root]
    seen = set()
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
        depth = len(path.parts) - base_parts
        if depth >= max_depth:
            continue
        try:
            for child in path.iterdir():
                if child.is_dir():
                    queue.append(child)
        except Exception:
            pass
    return out


def discover_file(
    explicit_env: str, names: Sequence[str], local_candidates: Sequence[Path]
) -> Path:
    explicit = os.environ.get(explicit_env, "").strip()
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(local_candidates)
    if IS_KAGGLE:
        for root in shallow_dirs(Path("/kaggle/input"), max_depth=4):
            for name in names:
                candidates.append(root / name)
    for path in dict.fromkeys(p.resolve() if p.exists() else p for p in candidates):
        if path.exists() and path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not find any of {list(names)}. Set {explicit_env} explicitly."
    )


def _is_w26_root(path: Path) -> bool:
    result_dir = path / "results" if (path / "results").is_dir() else path
    return all(
        (result_dir / name).exists()
        for name in (W26_PROB_FILE, W26_WEIGHT_FILE, W26_MASK_FILE)
    )


def discover_w26_root() -> Path:
    explicit = os.environ.get("W6_W26_ROOT", "").strip()
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "output" / "results" / "rsna_w2_6p_fast",
            PROJECT_ROOT / "output" / "rsna_w2_6p_fast",
            DATA_ROOT / "rsna_w2_6p_fast",
            SCRIPT_DIR / "rsna_w2_6p_fast",
        ]
    )
    if IS_KAGGLE:
        candidates.append(Path("/kaggle/working/rsna_w2_6p_fast"))
        candidates.extend(shallow_dirs(Path("/kaggle/input"), max_depth=4))
    for path in dict.fromkeys(p.resolve() if p.exists() else p for p in candidates):
        if _is_w26_root(path):
            return path.resolve()
    raise FileNotFoundError(
        "Final W2.6-P FAST outputs not found. Set W6_W26_ROOT to the directory "
        "containing results/16_final_hybrid_probabilities_wide.csv (or the "
        "results directory itself)."
    )


def looks_like_curia_root(path: Path) -> bool:
    required = [
        path / "config.json",
        path / "model.safetensors",
        path / "preprocessor_config.json",
        path / "curia_image_processor.py",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        return (
            cfg.get("model_type") == "dinov2"
            and int(cfg.get("hidden_size", -1)) == EXPECTED_CURIA_HIDDEN
            and int(cfg.get("image_size", -1)) == EXPECTED_CURIA_IMAGE
            and int(cfg.get("patch_size", -1)) == EXPECTED_CURIA_PATCH
            and int(cfg.get("num_channels", -1)) == 1
        )
    except Exception:
        return False


def discover_curia_root() -> Path:
    explicit = os.environ.get("W6_CURIA_ROOT", "").strip()
    candidates: List[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        candidates.extend([p, p / "curia-2", p / "curia-2-model"])
    candidates.extend(
        [
            PROJECT_ROOT / "models" / "curia-2-model",
            PROJECT_ROOT / "models" / "raidium-curia-2",
            DATA_ROOT / "models" / "curia-2",
            Path("/kaggle/working/curia-2"),
        ]
    )
    if IS_KAGGLE:
        candidates.extend(shallow_dirs(Path("/kaggle/input"), max_depth=4))
    for path in dict.fromkeys(p.resolve() if p.exists() else p for p in candidates):
        if looks_like_curia_root(path):
            return path.resolve()
    raise FileNotFoundError(
        "Curia-2 weights not found. Set W6_CURIA_ROOT to the folder containing "
        "config.json, model.safetensors, preprocessor_config.json and "
        "curia_image_processor.py."
    )


def discover_external_w4_cache() -> Optional[Path]:
    """Locate reusable W4 study caches only; no W4 source script is required."""
    explicit = os.environ.get("W6_W4_CACHE_ROOT", "").strip()
    candidates: List[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        candidates.extend(
            [p, p / "feature_cache", p / "feature_cache" / "studies", p / "studies"]
        )

    # Canonical local project layout requested by the user.
    candidates.extend(
        [
            PROJECT_ROOT / "output" / "results" / "rsna_w4_0_curia2" / "feature_cache",
            PROJECT_ROOT
            / "output"
            / "results"
            / "rsna_w4_0_curia2"
            / "feature_cache"
            / "studies",
            PROJECT_ROOT / "output" / "rsna_w4_0_curia2" / "feature_cache",
            PROJECT_ROOT / "rsna_w4_0_curia2" / "feature_cache",
        ]
    )

    if IS_KAGGLE:
        # Working-folder variants.
        candidates.extend(
            [
                Path("/kaggle/working/rsna_w4_0_curia2/feature_cache"),
                Path("/kaggle/working/rsna_w4_0_curia2/feature_cache/studies"),
                Path("/kaggle/working/output/results/rsna_w4_0_curia2/feature_cache"),
            ]
        )
        # Attached datasets: accept either a dataset rooted at rsna_w4_0_curia2,
        # at feature_cache, or directly at studies.
        for root in shallow_dirs(Path("/kaggle/input"), max_depth=5):
            if root.name == "rsna_w4_0_curia2":
                candidates.extend(
                    [root / "feature_cache", root / "feature_cache" / "studies"]
                )
            elif root.name == "feature_cache":
                candidates.extend([root, root / "studies"])
            elif root.name == "studies":
                candidates.append(root)

    for path in dict.fromkeys(p.resolve() if p.exists() else p for p in candidates):
        studies = path / "studies" if (path / "studies").is_dir() else path
        if studies.is_dir() and any(studies.glob("*.pt")):
            return studies.resolve()
    return None


# ============================================================
# 4. STANDALONE CURIA + CANONICAL W4 DICOM PREPROCESSING
# ============================================================
#
# IMPORTANT: No project script is imported or executed here. The functions in
# this section are self-contained copies of the preprocessing behavior used by
# canonical W4, retained in W6 to prevent image-pipeline drift.


def curia_autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return contextlib.nullcontext()


def _require_pydicom():
    try:
        import pydicom as _pydicom
    except Exception as exc:
        raise RuntimeError(
            "pydicom is required only when W6 must build MRI feature caches from DICOM. "
            "Install pydicom or reuse an existing compatible W4/W6 cache."
        ) from exc
    return _pydicom


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _mount_info(path: Path) -> Dict[str, Any]:
    """Best-effort Linux mount/filesystem diagnostics for mmap-related failures."""
    result: Dict[str, Any] = {"path": str(path)}
    try:
        resolved = path.resolve()
        best = None
        proc = Path("/proc/mounts")
        if proc.exists():
            for line in proc.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = Path(parts[1].replace("\\040", " "))
                try:
                    resolved.relative_to(mount_point)
                except Exception:
                    continue
                score = len(mount_point.parts)
                if best is None or score > best[0]:
                    best = (score, parts[0], str(mount_point), parts[2])
        if best is not None:
            _, device, mount_point, fs_type = best
            result.update(
                {"mount_device": device, "mount_point": mount_point, "fs_type": fs_type}
            )
    except Exception as exc:
        result["mount_error"] = repr(exc)
    return result


def inspect_safetensors_header(weights_path: Path) -> Dict[str, Any]:
    """Parse the safetensors envelope without mmap or the Rust extension."""
    info: Dict[str, Any] = {
        "path": str(weights_path),
        "exists": weights_path.exists(),
    }
    if not weights_path.exists():
        return info

    size = weights_path.stat().st_size
    info["file_size_bytes"] = int(size)
    info["file_size_mb"] = float(size / (1024**2))
    try:
        with weights_path.open("rb") as handle:
            prefix = handle.read(16)
            info["first_16_hex"] = prefix.hex()
            info["first_16_ascii"] = prefix.decode("ascii", errors="replace")
            if len(prefix) < 8:
                info.update(
                    {"header_valid": False, "header_error": "file shorter than 8 bytes"}
                )
                return info
            header_len = int.from_bytes(prefix[:8], byteorder="little", signed=False)
            info["declared_header_bytes"] = int(header_len)
            if header_len <= 0:
                info.update(
                    {
                        "header_valid": False,
                        "header_error": "non-positive header length",
                    }
                )
                return info
            if header_len > 100_000_000:
                info.update(
                    {
                        "header_valid": False,
                        "header_error": "declared header exceeds safetensors 100MB guard",
                    }
                )
                return info
            if 8 + header_len > size:
                info.update(
                    {
                        "header_valid": False,
                        "header_error": "declared header extends past EOF",
                    }
                )
                return info
            handle.seek(8)
            header = handle.read(header_len)
        text = header.decode("utf-8")
        parsed = json.loads(text)
        info["header_valid"] = True
        info["tensor_entries"] = int(
            sum(1 for k in parsed.keys() if k != "__metadata__")
        )
        info["has_metadata"] = "__metadata__" in parsed
        info["header_first_char"] = text[:1]
    except Exception as exc:
        info.update({"header_valid": False, "header_error": repr(exc)})
    return info


def _safetensors_probe(weights_path: Path) -> Dict[str, Any]:
    """Probe mmap and, when supported, pread backends without materializing tensors."""
    out: Dict[str, Any] = {}
    try:
        import safetensors
        from safetensors import safe_open

        out["safetensors_version"] = getattr(
            safetensors, "__version__", _package_version("safetensors")
        )
    except Exception as exc:
        out["import_error"] = repr(exc)
        return out

    for backend in ("mmap", "pread"):
        key = f"safe_open_{backend}"
        try:
            kwargs = {"framework": "pt", "device": "cpu"}
            try:
                if "backend" in inspect.signature(safe_open).parameters:
                    kwargs["backend"] = backend
                elif backend == "pread":
                    out[key] = "unsupported_by_installed_safetensors"
                    continue
            except Exception:
                if backend == "pread":
                    out[key] = "backend_signature_unknown"
                    continue
            with safe_open(str(weights_path), **kwargs) as handle:
                keys = list(handle.keys())
            out[key] = {"ok": True, "tensor_count": len(keys)}
        except Exception as exc:
            out[key] = {"ok": False, "error": repr(exc)}
    return out


def curia_environment_diagnostics(curia_root: Path) -> Dict[str, Any]:
    weights_path = curia_root / "model.safetensors"
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "transformers": _package_version("transformers"),
        "safetensors": _package_version("safetensors"),
        "curia_root": str(curia_root),
        "weights": inspect_safetensors_header(weights_path),
        "filesystem": _mount_info(weights_path),
    }
    try:
        info["weights_sha256"] = sha256_file(weights_path)
    except Exception as exc:
        info["weights_sha256_error"] = repr(exc)
    info["safe_open_probe"] = _safetensors_probe(weights_path)
    return info


def _stage_curia_root(curia_root: Path, model_sha: str) -> Path:
    """Copy Curia assets onto the native temp filesystem and verify the weight hash."""
    dest = CURIA_STAGING_ROOT / model_sha[:16]
    dest.mkdir(parents=True, exist_ok=True)
    required = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "curia_image_processor.py",
    ]
    for name in required:
        src = curia_root / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dest / name
        if name == "model.safetensors" and dst.exists():
            try:
                if (
                    dst.stat().st_size == src.stat().st_size
                    and sha256_file(dst) == model_sha
                ):
                    continue
            except Exception:
                pass
        elif dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        tmp = dst.with_suffix(
            dst.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    staged_sha = sha256_file(dest / "model.safetensors")
    if staged_sha != model_sha:
        raise RuntimeError(f"Staged Curia SHA mismatch: {staged_sha} vs {model_sha}")
    return dest


def validate_curia_identity(curia_root: Path) -> Tuple[str, Dict[str, Any]]:
    weights_path = curia_root / "model.safetensors"
    model_sha = sha256_file(weights_path)

    if model_sha != EXPECTED_CURIA_MODEL_SHA256 and not ALLOW_CURIA_SHA_MISMATCH:
        raise RuntimeError(
            "Curia-2 weight SHA256 mismatch.\n"
            f"Found   : {model_sha}\n"
            f"Expected: {EXPECTED_CURIA_MODEL_SHA256}\n"
            "Set W6_ALLOW_CURIA_SHA_MISMATCH=1 only if this is intentional."
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
        from transformers import AutoConfig, AutoImageProcessor, AutoModel
    except Exception as exc:
        raise RuntimeError("transformers is required for Curia-2.") from exc

    weights_path = curia_root / "model.safetensors"
    header = inspect_safetensors_header(weights_path)
    if not header.get("header_valid", False):
        raise RuntimeError(
            "Curia model.safetensors failed the raw-file header check before Transformers loading.\n"
            + json.dumps(
                curia_environment_diagnostics(curia_root), indent=2, default=str
            )
        )

    # Explicit slow processor is intentional for W4/W6 reproducibility.
    processor = AutoImageProcessor.from_pretrained(
        str(curia_root),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    primary_error: Optional[Exception] = None
    try:
        model = AutoModel.from_pretrained(str(curia_root), local_files_only=True)
    except Exception as exc:
        primary_error = exc
        log(f"[Curia] standard Transformers load failed: {type(exc).__name__}: {exc}")
        model = None

    # Fallback 1: bypass Transformers' safetensors mmap path. safetensors >=0.8
    # exposes backend='pread', which is safer on some mounted/external filesystems.
    if model is None:
        try:
            from safetensors.torch import load_file as st_load_file

            cfg = AutoConfig.from_pretrained(str(curia_root), local_files_only=True)
            model = AutoModel.from_config(cfg)
            kwargs: Dict[str, Any] = {"device": "cpu"}
            try:
                if "backend" in inspect.signature(st_load_file).parameters:
                    kwargs["backend"] = "pread"
            except Exception:
                pass
            state = st_load_file(str(weights_path), **kwargs)
            model.load_state_dict(state, strict=True)
            del state
            gc.collect()
            log(
                "[Curia] loaded through explicit CPU safetensors fallback"
                + (" (pread)" if kwargs.get("backend") == "pread" else "")
            )
        except Exception as fallback_error:
            log(
                f"[Curia] explicit safetensors fallback failed: {type(fallback_error).__name__}: {fallback_error}"
            )
            model = None

    # Fallback 2: if the project/model lives under /media, NTFS/exFAT, FUSE, etc.,
    # stage the exact 344MB model onto /tmp and retry mmap from the native filesystem.
    if model is None:
        try:
            model_sha = sha256_file(weights_path)
            staged_root = _stage_curia_root(curia_root, model_sha)
            log(f"[Curia] retrying from staged native-filesystem copy: {staged_root}")
            model = AutoModel.from_pretrained(str(staged_root), local_files_only=True)
        except Exception as staged_error:
            diagnostics = curia_environment_diagnostics(curia_root)
            raise RuntimeError(
                "Curia-2 could not be loaded after standard, explicit-CPU/pread, and staged-file retries.\n"
                f"Standard load error: {primary_error!r}\n"
                f"Staged retry error : {staged_error!r}\n"
                "Diagnostics:\n" + json.dumps(diagnostics, indent=2, default=str)
            ) from staged_error

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


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


def read_series_headers(
    study_uid: str, series_uid: str, series_root: Path
) -> List[Dict[str, Any]]:
    pydicom = _require_pydicom()
    series_dir = series_root / study_uid / series_uid
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
    pydicom = _require_pydicom()
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


def encode_study(
    study_uid: str,
    study_series: pd.DataFrame,
    processor,
    model: nn.Module,
    device: torch.device,
    encoder_batch_size: int,
    series_root: Path,
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

        records = read_series_headers(study_uid, series_uid, series_root)

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

        with torch.inference_mode(), curia_autocast_context(device):
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
# 5. ACCELERATOR / PRECISION
# ============================================================


@dataclass
class Runtime:
    device: torch.device
    accelerator: str
    amp_dtype: Optional[torch.dtype]


def resolve_runtime(accelerator: Optional[str] = None) -> Runtime:
    requested = (accelerator or ACCELERATOR or "auto").strip().lower()
    aliases = {
        "localgpu": "cuda",
        "local_gpu": "cuda",
        "kaggle_t4": "cuda",
        "gpu": "cuda",
        "apple_mps": "mps",
        "apple": "mps",
    }
    requested = aliases.get(requested, requested)

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device(f"cuda:{GPU_ID}")
        precision = PRECISION
        if precision == "auto":
            major, _minor = torch.cuda.get_device_capability(device)
            amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        elif precision in {"bf16", "bfloat16"}:
            amp_dtype = torch.bfloat16
        elif precision in {"fp16", "float16"}:
            amp_dtype = torch.float16
        elif precision in {"fp32", "float32", "none"}:
            amp_dtype = None
        else:
            raise ValueError(f"Unsupported W6_PRECISION={PRECISION!r}")
        return Runtime(device, "cuda", amp_dtype)

    if requested == "mps":
        if (
            getattr(torch.backends, "mps", None) is None
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but unavailable")
        # MPS autocast support differs by PyTorch release. Keep the head/model in
        # float32 for correctness/portability; frozen Curia still runs on MPS.
        return Runtime(torch.device("mps"), "mps", None)

    if requested == "cpu":
        return Runtime(torch.device("cpu"), "cpu", None)

    raise ValueError(
        f"Unsupported accelerator {requested!r}. Use auto/localGPU/apple_mps/cpu."
    )


def cache_gpu_ids(runtime: Runtime) -> List[int]:
    """GPU IDs used for Curia feature extraction; defaults to both Kaggle T4s."""
    if runtime.accelerator != "cuda":
        return []
    count = int(torch.cuda.device_count())
    if count <= 0:
        return []
    if CACHE_GPU_IDS_TEXT:
        ids: List[int] = []
        for token in CACHE_GPU_IDS_TEXT.split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value < 0 or value >= count:
                raise ValueError(
                    f"W6_CACHE_GPU_IDS contains unavailable GPU {value}; cuda device count={count}"
                )
            if value not in ids:
                ids.append(value)
        if ids:
            return ids
    if IS_KAGGLE and count >= 2:
        return [0, 1]
    index = runtime.device.index if runtime.device.index is not None else GPU_ID
    return [int(index)]


@contextlib.contextmanager
def autocast_context(runtime: Runtime):
    if runtime.accelerator == "cuda" and runtime.amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=runtime.amp_dtype):
            yield
    else:
        yield


def make_grad_scaler(runtime: Runtime):
    if runtime.accelerator == "cuda" and runtime.amp_dtype == torch.float16:
        try:
            return torch.amp.GradScaler("cuda")
        except Exception:
            return torch.cuda.amp.GradScaler()
    return None


# ============================================================
# 6. TRAIN / SERIES TABLES + LOCKED FOLDS
# ============================================================


def load_train_tables(
    require_dicom: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)
    if not TRAIN_SERIES_CSV.exists():
        raise FileNotFoundError(TRAIN_SERIES_CSV)
    if require_dicom and not TRAIN_SERIES_ROOT.exists():
        raise FileNotFoundError(TRAIN_SERIES_ROOT)

    train_df = pd.read_csv(TRAIN_CSV)
    series_df = pd.read_csv(TRAIN_SERIES_CSV)
    train_df[UID] = train_df[UID].astype(str)
    series_df[UID] = series_df[UID].astype(str)
    series_df[SERIES_UID] = series_df[SERIES_UID].astype(str)

    missing_labels = [label for label in LABELS if label not in train_df.columns]
    if missing_labels:
        raise RuntimeError(f"train.csv missing labels: {missing_labels}")
    if len(train_df) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} train studies, found {len(train_df)}"
        )

    required_series = {
        UID,
        SERIES_UID,
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }
    missing = sorted(required_series - set(series_df.columns))
    if missing:
        raise RuntimeError(f"train_series.csv missing columns: {missing}")

    # These private columns are exactly what W4.encode_study expects. Keep the
    # coercion local so status/teacher validation does not require importing
    # the DICOM-heavy canonical W4 module.
    series_df["_fluid"] = bool_series(series_df["Fluid_Sensitive"]).astype(int)
    series_df["_fs"] = bool_series(series_df["Fat_Suppression"]).astype(int)

    gold_df = (
        train_df[train_df[LABELS].notna().all(axis=1)]
        .copy()
        .sort_values(UID)
        .reset_index(drop=True)
    )
    if len(gold_df) != EXPECTED_GOLD:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD} gold studies, found {len(gold_df)}"
        )

    gold_y = gold_df[LABELS].to_numpy(dtype=np.int64)
    fold_zero = greedy_multilabel_folds(gold_y, NUM_FOLDS, RANDOM_SEED)
    fold_df = gold_df[[UID]].copy()
    fold_df["OuterFold"] = fold_zero + 1
    fold_sha = fold_assignment_sha256(fold_df)
    if fold_sha != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Locked fold checksum mismatch. Refusing to continue.\n"
            f"Found   : {fold_sha}\nExpected: {EXPECTED_FOLD_SHA256}"
        )
    fold_df.to_csv(RESULT_ROOT / "00_outer_fold_assignments.csv", index=False)
    return train_df, series_df, gold_df, fold_zero


# ============================================================
# 7. W2.6-P FINAL TEACHER
# ============================================================


@dataclass
class TeacherData:
    uids: List[str]
    probability: np.ndarray
    weight: np.ndarray
    mask: np.ndarray
    root: Path
    summary: Dict[str, Any]


def _w26_result_dir(root: Path) -> Path:
    return root / "results" if (root / "results").is_dir() else root


def load_teacher(
    train_df: Optional[pd.DataFrame] = None, strict: bool = True
) -> TeacherData:
    root = discover_w26_root()
    result_dir = _w26_result_dir(root)

    prob_df = pd.read_csv(result_dir / W26_PROB_FILE)
    weight_df = pd.read_csv(result_dir / W26_WEIGHT_FILE)
    mask_df = pd.read_csv(result_dir / W26_MASK_FILE)

    for frame_name, frame in (
        ("probability", prob_df),
        ("weight", weight_df),
        ("mask", mask_df),
    ):
        if UID not in frame.columns:
            raise RuntimeError(f"{frame_name} teacher file missing {UID}")
        frame[UID] = frame[UID].astype(str)
        missing = [label for label in LABELS if label not in frame.columns]
        if missing:
            raise RuntimeError(f"{frame_name} teacher file missing labels: {missing}")
        if frame[UID].duplicated().any():
            raise RuntimeError(
                f"{frame_name} teacher file has duplicate StudyInstanceUIDs"
            )

    base_uids = prob_df[UID].tolist()
    if set(weight_df[UID]) != set(base_uids) or set(mask_df[UID]) != set(base_uids):
        raise RuntimeError("Teacher probability/weight/mask UID sets differ")
    weight_df = weight_df.set_index(UID).loc[base_uids].reset_index()
    mask_df = mask_df.set_index(UID).loc[base_uids].reset_index()

    probability = prob_df[LABELS].to_numpy(dtype=np.float32)
    weight = weight_df[LABELS].to_numpy(dtype=np.float32)
    mask = np.column_stack(
        [bool_series(mask_df[label]).to_numpy() for label in LABELS]
    ).astype(bool)

    if len(base_uids) != EXPECTED_UNLABELED:
        raise RuntimeError(
            f"Expected {EXPECTED_UNLABELED} pseudo-label rows, found {len(base_uids)}"
        )
    if probability.shape != (EXPECTED_UNLABELED, NUM_LABELS):
        raise RuntimeError(f"Unexpected teacher probability shape {probability.shape}")
    if (
        not np.isfinite(probability).all()
        or not ((probability >= 0.0) & (probability <= 1.0)).all()
    ):
        raise RuntimeError("Teacher probabilities contain invalid values")
    if not np.isfinite(weight).all() or not ((weight >= 0.0) & (weight <= 1.0)).all():
        raise RuntimeError("Teacher weights contain invalid values")
    if strict and (weight[mask] <= 0).any():
        raise RuntimeError(
            "Teacher mask contains selected cells with zero/non-positive weight"
        )

    prod_summary: Dict[str, Any] = {}
    validation_summary: Dict[str, Any] = {}
    prod_path = result_dir / W26_PROD_SUMMARY
    validation_path = result_dir / W26_VALIDATION_SUMMARY
    if prod_path.exists():
        prod_summary = json.loads(prod_path.read_text(encoding="utf-8"))
    if validation_path.exists():
        validation_summary = json.loads(validation_path.read_text(encoding="utf-8"))
        if strict and validation_summary.get("overall_pass") is not True:
            raise RuntimeError(
                "W2.6-P validation summary does not have overall_pass=true"
            )
        # Be defensive about provenance without depending on one exact schema.
        validation_text = json.dumps(validation_summary).lower()
        if (
            strict
            and "pilkwang" in validation_text
            and not any(
                marker in validation_text
                for marker in [
                    "pilkwang_not_used",
                    '"pilkwang_used": false',
                    '"pilkwang": false',
                ]
            )
        ):
            warnings.warn(
                "Validation summary mentions Pilkwang; manually confirm it remains audit-only.",
                RuntimeWarning,
            )

    if train_df is not None:
        unlabeled_df = train_df[~train_df[LABELS].notna().all(axis=1)].copy()
        unlabeled_uids = set(unlabeled_df[UID].astype(str))
        teacher_uids = set(base_uids)
        if unlabeled_uids != teacher_uids:
            missing = sorted(unlabeled_uids - teacher_uids)[:10]
            extra = sorted(teacher_uids - unlabeled_uids)[:10]
            raise RuntimeError(
                f"Teacher/train unlabeled UID mismatch. missing={missing}, extra={extra}"
            )
        gold_uids = set(train_df[train_df[LABELS].notna().all(axis=1)][UID].astype(str))
        overlap = teacher_uids & gold_uids
        if overlap:
            raise RuntimeError(
                f"Teacher file unexpectedly contains gold UIDs: {sorted(overlap)[:5]}"
            )

    if TEACHER_WEIGHT_MODE == "binary":
        effective_weight = mask.astype(np.float32)
    else:
        effective_weight = weight.copy()
        effective_weight[~mask] = 0.0

    summary = {
        "teacher_root": str(root),
        "teacher_weight_mode": TEACHER_WEIGHT_MODE,
        "rows": len(base_uids),
        "selected_cells": int(mask.sum()),
        "selected_cells_per_label": {
            label: int(mask[:, j].sum()) for j, label in enumerate(LABELS)
        },
        "mean_positive_weight_per_label": {
            label: float(weight[mask[:, j], j].mean()) if mask[:, j].any() else 0.0
            for j, label in enumerate(LABELS)
        },
        "production_summary": prod_summary,
        "validation_summary": validation_summary,
        "warning": (
            "Production teacher uses all 58 gold reports as exemplars; downstream 58-study "
            "diagnostics are not pristine OOF validation."
        ),
    }
    atomic_json_dump(summary, RESULT_ROOT / "01_teacher_validation.json")
    return TeacherData(base_uids, probability, effective_weight, mask, root, summary)


# ============================================================
# 8. FEATURE CACHE HELPERS
# ============================================================


def w60_cache_path(uid: str) -> Path:
    return W60_CACHE_STUDIES / f"{stable_uid_hash(uid)}.pt"


def w61_cache_path(uid: str) -> Path:
    return W61_CACHE_STUDIES / f"{stable_uid_hash(uid)}.pt"


def external_w4_cache_path(root: Path, uid: str) -> Path:
    return root / f"{stable_uid_hash(uid)}.pt"


def w4_payload_usable(payload: Mapping[str, Any], uid: str) -> bool:
    try:
        f = payload["features"]
        return (
            str(payload.get("cache_version")) == EXPECTED_W4_CACHE_VERSION
            and str(payload.get("study_uid")) == str(uid)
            and isinstance(f, torch.Tensor)
            and f.ndim == 3
            and f.shape[-1] == EXPECTED_CURIA_HIDDEN
            and f.dtype == torch.float16
            and payload["slice_mask"].shape == f.shape[:2]
            and payload["slice_position"].shape == f.shape[:2]
            and payload["series_meta"].shape[0] == f.shape[0]
            and payload["series_cont"].shape[0] == f.shape[0]
        )
    except Exception:
        return False


def w61_payload_usable(payload: Mapping[str, Any], uid: str) -> bool:
    try:
        cls = payload["features"]
        patch = payload["patch_grid"]
        return (
            str(payload.get("cache_version")) == SPATIAL_CACHE_VERSION
            and str(payload.get("study_uid")) == str(uid)
            and isinstance(cls, torch.Tensor)
            and isinstance(patch, torch.Tensor)
            and cls.ndim == 3
            and cls.shape[-1] == EXPECTED_CURIA_HIDDEN
            and patch.shape == (*cls.shape[:2], SPATIAL_TOKENS, EXPECTED_CURIA_HIDDEN)
            and cls.dtype == torch.float16
            and patch.dtype == torch.float16
            and payload["slice_mask"].shape == cls.shape[:2]
        )
    except Exception:
        return False


def w60_compatible_payload(payload: Mapping[str, Any], uid: str) -> bool:
    if w4_payload_usable(payload, uid):
        return True
    # W6.1 contains the exact W4 CLS tensors plus patch_grid. Reuse those CLS
    # tensors rather than re-running Curia when a spatial cache already exists.
    try:
        return (
            w61_payload_usable(payload, uid)
            and str(payload.get("w4_cache_version")) == EXPECTED_W4_CACHE_VERSION
        )
    except Exception:
        return False


class CuriaPatchCapture(nn.Module):
    """Wrap frozen Curia and capture compact patch-grid tokens in call order."""

    def __init__(self, base_model: nn.Module, grid: int):
        super().__init__()
        self.base_model = base_model
        self.grid = int(grid)
        self.captured: List[torch.Tensor] = []

    def reset_capture(self) -> None:
        self.captured.clear()

    def forward(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != EXPECTED_CURIA_HIDDEN:
            raise RuntimeError(
                f"Unexpected Curia hidden-state shape: {tuple(hidden.shape)}"
            )
        if hidden.shape[1] < 1 + EXPECTED_CURIA_PATCHES:
            raise RuntimeError(
                f"Curia returned only {hidden.shape[1]} tokens; need at least "
                f"{1 + EXPECTED_CURIA_PATCHES} for a 32x32 patch grid."
            )
        # Use the final 1024 tokens so this remains correct if a model variant
        # inserts register tokens between CLS and patch tokens.
        patches = hidden[:, -EXPECTED_CURIA_PATCHES:, :]
        patches = patches.reshape(-1, 32, 32, EXPECTED_CURIA_HIDDEN).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(patches.float(), (self.grid, self.grid))
        pooled = pooled.permute(0, 2, 3, 1).reshape(
            -1, self.grid * self.grid, EXPECTED_CURIA_HIDDEN
        )
        self.captured.append(pooled.detach().cpu().to(torch.float16).contiguous())
        return output


class FeatureStore:
    def __init__(
        self,
        uids: Sequence[str],
        variant: str,
        external_w4_cache: Optional[Path] = None,
        lru_size: int = FEATURE_LRU_SIZE,
    ):
        self.uids = [str(x) for x in uids]
        self.uid_to_index = {uid: i for i, uid in enumerate(self.uids)}
        if len(self.uid_to_index) != len(self.uids):
            raise RuntimeError("Duplicate UID in feature store")
        self.variant = variant
        self.external_w4_cache = external_w4_cache
        self.lru_size = max(0, int(lru_size))
        self._lru: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def indices_for_uids(self, uids: Sequence[str]) -> np.ndarray:
        missing = [str(uid) for uid in uids if str(uid) not in self.uid_to_index]
        if missing:
            raise KeyError(f"Feature store missing UIDs: {missing[:5]}")
        return np.asarray([self.uid_to_index[str(uid)] for uid in uids], dtype=np.int64)

    def path_for_uid(self, uid: str) -> Path:
        uid = str(uid)
        if self.variant == "w61":
            return w61_cache_path(uid)
        own = w60_cache_path(uid)
        if own.exists():
            return own
        if self.external_w4_cache is not None:
            external = external_w4_cache_path(self.external_w4_cache, uid)
            if external.exists():
                return external
        spatial = w61_cache_path(uid)
        if spatial.exists():
            return spatial
        return own

    def _load(self, uid: str) -> Dict[str, Any]:
        uid = str(uid)
        if uid in self._lru:
            payload = self._lru.pop(uid)
            self._lru[uid] = payload
            return payload
        path = self.path_for_uid(uid)
        if not path.exists():
            raise FileNotFoundError(f"Missing {self.variant} cache for {uid}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        usable = (
            w61_payload_usable(payload, uid)
            if self.variant == "w61"
            else w60_compatible_payload(payload, uid)
        )
        if not usable:
            raise RuntimeError(f"Invalid/stale {self.variant} cache: {path}")
        if (
            self.variant == "w60"
            and str(payload.get("cache_version")) == SPATIAL_CACHE_VERSION
        ):
            # Drop the large patch tensor immediately for W6.0 head training.
            payload = {k: v for k, v in payload.items() if k != "patch_grid"}
            payload["cache_version"] = EXPECTED_W4_CACHE_VERSION
        if self.lru_size > 0:
            self._lru[uid] = payload
            while len(self._lru) > self.lru_size:
                self._lru.popitem(last=False)
        return payload

    def validate_all(self) -> Dict[str, int]:
        missing = 0
        invalid = 0
        for uid in self.uids:
            path = self.path_for_uid(uid)
            if not path.exists():
                missing += 1
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                okay = (
                    w61_payload_usable(payload, uid)
                    if self.variant == "w61"
                    else w60_compatible_payload(payload, uid)
                )
                if not okay:
                    invalid += 1
            except Exception:
                invalid += 1
        return {
            "total": len(self.uids),
            "missing": missing,
            "invalid": invalid,
            "usable": len(self.uids) - missing - invalid,
        }

    def make_batch(
        self, indices: np.ndarray, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        payloads = [self._load(self.uids[int(i)]) for i in indices]
        max_s = max(int(p["features"].shape[0]) for p in payloads)
        max_k = max(int(p["features"].shape[1]) for p in payloads)
        b = len(payloads)

        features = torch.zeros(
            b, max_s, max_k, EXPECTED_CURIA_HIDDEN, dtype=torch.float16
        )
        slice_mask = torch.zeros(b, max_s, max_k, dtype=torch.bool)
        slice_position = torch.zeros(b, max_s, max_k, dtype=torch.float16)
        series_meta = torch.zeros(b, max_s, 3, dtype=torch.long)
        series_cont = torch.zeros(b, max_s, 2, dtype=torch.float32)
        series_mask = torch.zeros(b, max_s, dtype=torch.bool)

        patch_grid = None
        if self.variant == "w61":
            patch_grid = torch.zeros(
                b,
                max_s,
                max_k,
                SPATIAL_TOKENS,
                EXPECTED_CURIA_HIDDEN,
                dtype=torch.float16,
            )

        for bi, p in enumerate(payloads):
            s, k, _h = p["features"].shape
            features[bi, :s, :k] = p["features"]
            slice_mask[bi, :s, :k] = p["slice_mask"]
            slice_position[bi, :s, :k] = p["slice_position"]
            series_meta[bi, :s] = p["series_meta"].long()
            series_cont[bi, :s] = p["series_cont"].float()
            series_mask[bi, :s] = p["slice_mask"].any(dim=-1)
            if patch_grid is not None:
                patch_grid[bi, :s, :k] = p["patch_grid"]

        output = {
            "features": features.to(device, non_blocking=False),
            "slice_mask": slice_mask.to(device, non_blocking=False),
            "slice_position": slice_position.to(device, non_blocking=False),
            "series_meta": series_meta.to(device, non_blocking=False),
            "series_cont": series_cont.to(device, non_blocking=False),
            "series_mask": series_mask.to(device, non_blocking=False),
        }
        if patch_grid is not None:
            output["patch_grid"] = patch_grid.to(device, non_blocking=False)
        return output


# ============================================================
# 9. CACHE BUILDING — STANDALONE CANONICAL PREPROCESSING
# ============================================================


def _load_curia(runtime: Runtime):
    curia_root = discover_curia_root()
    model_sha, config = validate_curia_identity(curia_root)
    processor, model = load_curia_processor_and_model(curia_root, runtime.device)
    return curia_root, model_sha, config, processor, model


def cache_w60(accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    # W6.0 can train entirely from the canonical W4 CLS cache. Do not require the
    # 570GB train_series DICOM tree unless one or more study caches are actually missing.
    train_df, series_df, _gold_df, _fold_zero = load_train_tables(require_dicom=False)
    external = discover_external_w4_cache()
    all_uids = sorted(train_df[UID].astype(str).tolist())

    reusable = 0
    if external is not None:
        for uid in all_uids:
            path = external_w4_cache_path(external, uid)
            if path.exists():
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    reusable += int(w4_payload_usable(payload, uid))
                except Exception:
                    pass
    log(f"External canonical W4 cache: {external}")
    log(f"Reusable W4 study caches     : {reusable}/{len(all_uids)}")

    own_usable = 0
    for uid in all_uids:
        p = w60_cache_path(uid)
        if p.exists():
            try:
                own_usable += int(
                    w4_payload_usable(
                        torch.load(p, map_location="cpu", weights_only=False), uid
                    )
                )
            except Exception:
                pass
    missing = len(all_uids) - reusable - own_usable
    if missing <= 0:
        summary = {
            "status": "complete_via_reuse",
            "external_cache": str(external) if external else None,
            "reusable": reusable,
            "own_usable": own_usable,
        }
        atomic_json_dump(summary, RESULT_ROOT / "10_w60_cache_summary.json")
        return summary

    if not TRAIN_SERIES_ROOT.exists():
        raise RuntimeError(
            f"W6.0 is missing {missing} CLS study caches, but train_series DICOM is unavailable at {TRAIN_SERIES_ROOT}. "
            "Do not download the 570GB train_series tree just for W6.0. Attach/run on Kaggle to fill missing caches, "
            "or provide the complete W4 feature_cache via W6_W4_CACHE_ROOT."
        )

    curia_root, model_sha, config, processor, model = _load_curia(runtime)
    started = time.time()
    encoded = 0
    failures: List[Dict[str, Any]] = []
    uid_groups = {uid: g.copy() for uid, g in series_df.groupby(UID, sort=False)}

    for idx, uid in enumerate(all_uids, 1):
        own = w60_cache_path(uid)
        if own.exists():
            try:
                if w4_payload_usable(
                    torch.load(own, map_location="cpu", weights_only=False), uid
                ):
                    continue
            except Exception:
                pass
        if external is not None:
            ep = external_w4_cache_path(external, uid)
            if ep.exists():
                try:
                    if w4_payload_usable(
                        torch.load(ep, map_location="cpu", weights_only=False), uid
                    ):
                        continue
                except Exception:
                    pass
        try:
            if uid not in uid_groups:
                raise RuntimeError("No train_series.csv rows")
            payload, _audit = encode_study(
                uid,
                uid_groups[uid],
                processor,
                model,
                runtime.device,
                ENCODER_BATCH,
                TRAIN_SERIES_ROOT,
            )
            if not w4_payload_usable(payload, uid):
                raise RuntimeError("Newly encoded W4 payload failed validation")
            atomic_torch_save(payload, own)
            encoded += 1
        except Exception as exc:
            failures.append({UID: uid, "error": repr(exc)})
            log(f"[W60 cache] ERROR {uid}: {exc}")
        if idx % 50 == 0 or idx == len(all_uids):
            elapsed = max(time.time() - started, 1e-6)
            log(
                f"[W60 cache] {idx}/{len(all_uids)} new={encoded} rate={encoded/elapsed:.3f} studies/s"
            )

    if failures:
        pd.DataFrame(failures).to_csv(
            RESULT_ROOT / "10_w60_cache_failures.csv", index=False
        )
        raise RuntimeError(f"W60 cache completed with {len(failures)} failures")

    summary = {
        "status": "complete",
        "created_at": now_iso(),
        "runtime": runtime.accelerator,
        "curia_root": str(curia_root),
        "curia_sha256": model_sha,
        "curia_config": config,
        "external_w4_cache": str(external) if external else None,
        "newly_encoded": encoded,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(summary, RESULT_ROOT / "10_w60_cache_summary.json")
    return summary


def _w61_cache_worker(
    worker_id: int,
    device: torch.device,
    study_uids: Sequence[str],
    uid_groups: Mapping[str, pd.DataFrame],
    curia_root: Path,
    progress: Dict[str, int],
    progress_lock: threading.Lock,
    total_requested: int,
    started: float,
) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    processor, base_model = load_curia_processor_and_model(curia_root, device)
    capture_model = CuriaPatchCapture(base_model, SPATIAL_GRID).to(device).eval()
    for parameter in capture_model.parameters():
        parameter.requires_grad = False

    encoded = 0
    total_bytes = 0
    failures: List[Dict[str, Any]] = []
    log(
        f"[W61 cache worker {worker_id}] device={device} studies={len(study_uids)} batch={SPATIAL_ENCODER_BATCH}"
    )

    for uid in study_uids:
        path = w61_cache_path(uid)
        try:
            if path.exists():
                try:
                    if w61_payload_usable(
                        torch.load(path, map_location="cpu", weights_only=False), uid
                    ):
                        continue
                except Exception:
                    pass
            if uid not in uid_groups:
                raise RuntimeError("No train_series.csv rows")
            capture_model.reset_capture()
            payload, _audit = encode_study(
                uid,
                uid_groups[uid],
                processor,
                capture_model,
                device,
                SPATIAL_ENCODER_BATCH,
                TRAIN_SERIES_ROOT,
            )
            captured = (
                torch.cat(capture_model.captured, dim=0)
                if capture_model.captured
                else None
            )
            expected_slices = int(payload["slice_mask"].sum().item())
            expected_shape = (expected_slices, SPATIAL_TOKENS, EXPECTED_CURIA_HIDDEN)
            if captured is None or tuple(captured.shape) != expected_shape:
                raise RuntimeError(
                    f"Patch capture mismatch: captured={None if captured is None else tuple(captured.shape)}, "
                    f"expected={expected_shape}"
                )
            series_count, slice_count, _hidden = payload["features"].shape
            patch_grid = torch.zeros(
                series_count,
                slice_count,
                SPATIAL_TOKENS,
                EXPECTED_CURIA_HIDDEN,
                dtype=torch.float16,
            )
            cursor = 0
            for series_index in range(series_count):
                for slice_index in range(slice_count):
                    if bool(payload["slice_mask"][series_index, slice_index]):
                        patch_grid[series_index, slice_index] = captured[cursor]
                        cursor += 1
            if cursor != expected_slices:
                raise RuntimeError(
                    f"Patch scatter cursor {cursor} != {expected_slices}"
                )
            payload["w4_cache_version"] = payload.get("cache_version")
            payload["cache_version"] = SPATIAL_CACHE_VERSION
            payload["patch_grid"] = patch_grid.contiguous()
            payload["patch_grid_shape"] = [SPATIAL_GRID, SPATIAL_GRID]
            payload["patch_token_source"] = (
                f"last {EXPECTED_CURIA_PATCHES} hidden tokens -> adaptive_avg_pool2d({SPATIAL_GRID})"
            )
            if not w61_payload_usable(payload, uid):
                raise RuntimeError("New W61 payload failed validation")
            atomic_torch_save(payload, path)
            encoded += 1
            total_bytes += path.stat().st_size
        except Exception as exc:
            failures.append(
                {
                    UID: uid,
                    "Worker": worker_id,
                    "Device": str(device),
                    "error": repr(exc),
                }
            )
            log(f"[W61 cache worker {worker_id}] ERROR {uid}: {exc}")
        finally:
            with progress_lock:
                progress["done"] += 1
                done = int(progress["done"])
            if done <= 10 or done % 100 == 0 or done == total_requested:
                elapsed = max(time.time() - started, 1e-6)
                rate = done / elapsed
                eta = (total_requested - done) / max(rate, 1e-6)
                log(
                    f"[W61 cache] {done}/{total_requested} ({100.0*done/max(total_requested,1):.1f}%) "
                    f"elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h worker={worker_id} "
                    f"encoded={encoded} failures={len(failures)}"
                )

    del capture_model, base_model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "worker_id": worker_id,
        "device": str(device),
        "encoded": encoded,
        "total_bytes": total_bytes,
        "failures": failures,
    }


def cache_w61(accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    train_df, series_df, _gold_df, _fold_zero = load_train_tables(require_dicom=True)
    all_uids = sorted(train_df[UID].astype(str).tolist())
    uid_groups = {uid: g.copy() for uid, g in series_df.groupby(UID, sort=False)}

    existing = 0
    need: List[str] = []
    for uid in all_uids:
        path = w61_cache_path(uid)
        okay = False
        if path.exists():
            try:
                okay = w61_payload_usable(
                    torch.load(path, map_location="cpu", weights_only=False), uid
                )
            except Exception:
                okay = False
        if okay:
            existing += 1
        else:
            need.append(uid)

    if not need:
        summary = {
            "status": "complete_reused",
            "usable": existing,
            "spatial_grid": SPATIAL_GRID,
        }
        atomic_json_dump(summary, RESULT_ROOT / "20_w61_cache_summary.json")
        return summary

    curia_root = discover_curia_root()
    model_sha, config = validate_curia_identity(curia_root)
    gpu_ids = cache_gpu_ids(runtime)
    if runtime.accelerator == "cuda":
        worker_devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
    else:
        worker_devices = [runtime.device]
    if not worker_devices:
        worker_devices = [runtime.device]

    log(f"[W61 cache] Curia root={curia_root}")
    log(f"[W61 cache] Need encoding={len(need)}/{len(all_uids)}")
    log(f"[W61 cache] Worker devices={[str(x) for x in worker_devices]}")

    shards = [need[i :: len(worker_devices)] for i in range(len(worker_devices))]
    progress = {"done": 0}
    progress_lock = threading.Lock()
    started = time.time()
    worker_results: List[Dict[str, Any]] = []

    if len(worker_devices) == 1:
        worker_results.append(
            _w61_cache_worker(
                0,
                worker_devices[0],
                shards[0],
                uid_groups,
                curia_root,
                progress,
                progress_lock,
                len(need),
                started,
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
            futures = [
                executor.submit(
                    _w61_cache_worker,
                    worker_id,
                    device,
                    shards[worker_id],
                    uid_groups,
                    curia_root,
                    progress,
                    progress_lock,
                    len(need),
                    started,
                )
                for worker_id, device in enumerate(worker_devices)
            ]
            for future in as_completed(futures):
                worker_results.append(future.result())

    failures: List[Dict[str, Any]] = []
    encoded = 0
    total_bytes = 0
    for result in worker_results:
        failures.extend(result["failures"])
        encoded += int(result["encoded"])
        total_bytes += int(result["total_bytes"])

    if failures:
        pd.DataFrame(failures).to_csv(
            RESULT_ROOT / "20_w61_cache_failures.csv", index=False
        )
        raise RuntimeError(f"W61 cache completed with {len(failures)} failures")

    final_usable = 0
    for uid in all_uids:
        path = w61_cache_path(uid)
        if path.exists():
            try:
                final_usable += int(
                    w61_payload_usable(
                        torch.load(path, map_location="cpu", weights_only=False), uid
                    )
                )
            except Exception:
                pass
    if final_usable != len(all_uids):
        raise RuntimeError(
            f"W61 cache incomplete after workers: {final_usable}/{len(all_uids)} usable"
        )

    summary = {
        "status": "complete",
        "created_at": now_iso(),
        "runtime": runtime.accelerator,
        "worker_devices": [str(x) for x in worker_devices],
        "curia_root": str(curia_root),
        "curia_sha256": model_sha,
        "curia_config": config,
        "spatial_grid": SPATIAL_GRID,
        "spatial_tokens": SPATIAL_TOKENS,
        "newly_encoded": encoded,
        "usable": final_usable,
        "new_cache_bytes": total_bytes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(summary, RESULT_ROOT / "20_w61_cache_summary.json")
    return summary


# ============================================================
# 10. W6.1 SPATIAL HIERARCHICAL HEAD
# ============================================================


def _drop_mask_tokens(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    output = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    need_restore = mask.any(dim=-1) & ~output.any(dim=-1)
    if need_restore.any():
        first_valid = mask.float().argmax(dim=-1)
        for coordinate in torch.nonzero(need_restore, as_tuple=False):
            prefix = tuple(int(x) for x in coordinate.tolist())
            token = int(first_valid[prefix].item())
            output[prefix + (token,)] = True
    return output


class W4ExactCuriaHierarchicalDiagnosisHead(nn.Module):
    """Local copy of the canonical W4 head for cache-only W6.0 training.

    This mirrors the canonical W4 head at the architecture level. W6.0 can
    train directly from an existing W4 feature cache, and this W6 script also
    contains its own standalone DICOM/Curia encoder when cache generation is
    required. No external project script is imported.
    """

    def __init__(self):
        super().__init__()
        if HEAD_HIDDEN_DIM % HEAD_NUM_HEADS != 0:
            raise ValueError(
                "W6_HEAD_HIDDEN_DIM must be divisible by W6_HEAD_NUM_HEADS"
            )
        self.input_norm = nn.LayerNorm(EXPECTED_CURIA_HIDDEN)
        self.input_projection = nn.Linear(EXPECTED_CURIA_HIDDEN, HEAD_HIDDEN_DIM)
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
            [1.0, 2.0, 4.0, 8.0], device=position.device, dtype=position.dtype
        )
        angles = math.pi * position.unsqueeze(-1) * frequencies
        return torch.cat(
            [position.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1
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
        if self.training and SERIES_DROPOUT > 0:
            series_mask_effective = _drop_mask_tokens(series_mask, SERIES_DROPOUT)
        else:
            series_mask_effective = series_mask
        slice_mask_effective = slice_mask & series_mask_effective.unsqueeze(-1)
        if self.training and SLICE_DROPOUT > 0:
            b, s, k = slice_mask_effective.shape
            flat = _drop_mask_tokens(
                slice_mask_effective.reshape(b * s, k), SLICE_DROPOUT
            )
            slice_mask_effective = flat.reshape(b, s, k)

        batch_size, max_series, max_slices, _ = features.shape
        real_series_flat = series_mask_effective.reshape(-1)
        flat_features = features.reshape(
            batch_size * max_series, max_slices, EXPECTED_CURIA_HIDDEN
        )[real_series_flat]
        flat_slice_mask = slice_mask_effective.reshape(
            batch_size * max_series, max_slices
        )[real_series_flat]
        flat_position = slice_position.reshape(batch_size * max_series, max_slices)[
            real_series_flat
        ]

        hidden = self.input_projection(self.input_norm(flat_features.float()))
        hidden = hidden + self.position_projection(
            self._fourier_position(flat_position.float())
        )
        hidden = self.slice_transformer(hidden, src_key_padding_mask=(~flat_slice_mask))
        slice_scores = torch.einsum(
            "rkh,lh->rlk", hidden, self.slice_queries
        ) / math.sqrt(HEAD_HIDDEN_DIM)
        slice_scores = slice_scores.masked_fill(~flat_slice_mask.unsqueeze(1), -1e4)
        slice_attention = torch.softmax(slice_scores, dim=-1)
        pooled_series = torch.einsum("rlk,rkh->rlh", slice_attention, hidden)

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
        series_representation = torch.zeros(
            batch_size * max_series,
            NUM_LABELS,
            HEAD_HIDDEN_DIM,
            device=features.device,
            dtype=pooled_series.dtype,
        )
        series_representation[real_series_flat] = pooled_series
        series_representation = series_representation.reshape(
            batch_size, max_series, NUM_LABELS, HEAD_HIDDEN_DIM
        )
        series_scores = torch.einsum(
            "bslh,lh->bsl", series_representation, self.series_queries
        ) / math.sqrt(HEAD_HIDDEN_DIM)
        series_scores = series_scores.masked_fill(
            ~series_mask_effective.unsqueeze(-1), -1e4
        )
        series_attention = torch.softmax(series_scores, dim=1)
        study_representation = torch.einsum(
            "bsl,bslh->blh", series_attention, series_representation
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


class SpatialCuriaHierarchicalDiagnosisHead(nn.Module):
    """
    W6.1 head.

    Per real slice:
      3x3 pooled Curia patch tokens -> label-specific spatial attention
      + projected CLS residual -> label-specific slice representation.

    Per label, the same small Transformer is then applied over slice order for
    each series. W4-style label-specific slice attention, metadata fusion,
    series attention and final classifier follow.
    """

    def __init__(self):
        super().__init__()
        h = SPATIAL_HIDDEN_DIM
        if h % SPATIAL_NUM_HEADS != 0:
            raise ValueError(
                "W6_SPATIAL_HIDDEN_DIM must be divisible by W6_SPATIAL_NUM_HEADS"
            )

        self.cls_norm = nn.LayerNorm(EXPECTED_CURIA_HIDDEN)
        self.cls_projection = nn.Linear(EXPECTED_CURIA_HIDDEN, h)
        self.patch_norm = nn.LayerNorm(EXPECTED_CURIA_HIDDEN)
        self.patch_projection = nn.Linear(EXPECTED_CURIA_HIDDEN, h)

        # [x, y, sin/cos at 1,2,4,8*pi for each axis] = 18 dims.
        self.patch_position_projection = nn.Sequential(
            nn.Linear(18, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.spatial_queries = nn.Parameter(torch.randn(NUM_LABELS, h) * 0.02)
        self.cls_gate = nn.Parameter(torch.zeros(NUM_LABELS))
        self.spatial_norm = nn.LayerNorm(h)

        self.slice_position_projection = nn.Sequential(
            nn.Linear(9, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=SPATIAL_NUM_HEADS,
            dim_feedforward=4 * h,
            dropout=SPATIAL_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slice_transformer = nn.TransformerEncoder(
            layer,
            num_layers=SPATIAL_SLICE_LAYERS,
            norm=nn.LayerNorm(h),
        )
        self.slice_queries = nn.Parameter(torch.randn(NUM_LABELS, h) * 0.02)

        self.plane_embedding = nn.Embedding(3, 32)
        self.fluid_embedding = nn.Embedding(2, 16)
        self.fs_embedding = nn.Embedding(2, 16)
        self.metadata_projection = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 2, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )
        self.series_norm = nn.LayerNorm(h)
        self.series_queries = nn.Parameter(torch.randn(NUM_LABELS, h) * 0.02)
        self.final_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(SPATIAL_DROPOUT)
        self.classifier_weight = nn.Parameter(torch.randn(NUM_LABELS, h) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

        coords = []
        for yi in range(SPATIAL_GRID):
            for xi in range(SPATIAL_GRID):
                x = -1.0 + 2.0 * xi / max(SPATIAL_GRID - 1, 1)
                y = -1.0 + 2.0 * yi / max(SPATIAL_GRID - 1, 1)
                coords.append((x, y))
        self.register_buffer(
            "patch_coords", torch.tensor(coords, dtype=torch.float32), persistent=False
        )

    @staticmethod
    def _fourier_1d(position: torch.Tensor) -> torch.Tensor:
        position = position.clamp(-1.0, 1.0)
        frequencies = torch.tensor(
            [1.0, 2.0, 4.0, 8.0], device=position.device, dtype=position.dtype
        )
        angles = math.pi * position.unsqueeze(-1) * frequencies
        return torch.cat(
            [position.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1
        )

    def _fourier_2d(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._fourier_1d(coords[..., 0]), self._fourier_1d(coords[..., 1])], dim=-1
        )

    def forward(
        self,
        features: torch.Tensor,
        patch_grid: torch.Tensor,
        slice_mask: torch.Tensor,
        slice_position: torch.Tensor,
        series_meta: torch.Tensor,
        series_cont: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hdim = SPATIAL_HIDDEN_DIM
        if self.training and SERIES_DROPOUT > 0:
            series_mask_eff = _drop_mask_tokens(series_mask, SERIES_DROPOUT)
        else:
            series_mask_eff = series_mask
        slice_mask_eff = slice_mask & series_mask_eff.unsqueeze(-1)
        if self.training and SLICE_DROPOUT > 0:
            b, s, k = slice_mask_eff.shape
            flat = _drop_mask_tokens(slice_mask_eff.reshape(b * s, k), SLICE_DROPOUT)
            slice_mask_eff = flat.reshape(b, s, k)

        b, s, k, _ = features.shape
        real_series_flat = series_mask_eff.reshape(-1)
        cls = features.reshape(b * s, k, EXPECTED_CURIA_HIDDEN)[real_series_flat]
        patches = patch_grid.reshape(b * s, k, SPATIAL_TOKENS, EXPECTED_CURIA_HIDDEN)[
            real_series_flat
        ]
        smask = slice_mask_eff.reshape(b * s, k)[real_series_flat]
        spos = slice_position.reshape(b * s, k)[real_series_flat]
        r = cls.shape[0]

        # Project patch tokens; add fixed 2-D region position.
        patch_hidden = self.patch_projection(self.patch_norm(patches.float()))
        ppos = self.patch_position_projection(
            self._fourier_2d(self.patch_coords.to(patch_hidden.dtype))
        )
        patch_hidden = patch_hidden + ppos.view(1, 1, SPATIAL_TOKENS, hdim)

        # [R,K,L,T]
        spatial_scores = torch.einsum(
            "rkth,lh->rklt", patch_hidden, self.spatial_queries
        ) / math.sqrt(hdim)
        spatial_attention = torch.softmax(spatial_scores, dim=-1)
        spatial = torch.einsum("rklt,rkth->rklh", spatial_attention, patch_hidden)

        cls_hidden = self.cls_projection(self.cls_norm(cls.float()))
        gate = torch.sigmoid(self.cls_gate).view(1, 1, NUM_LABELS, 1)
        hidden = self.spatial_norm(cls_hidden.unsqueeze(2) + gate * spatial)
        hidden = hidden + self.slice_position_projection(
            self._fourier_1d(spos.float())
        ).unsqueeze(2)

        # Apply the SAME slice transformer independently to each label sequence:
        # [R,K,L,H] -> [R*L,K,H]. Shared weights preserve parameter efficiency.
        hidden = hidden.permute(0, 2, 1, 3).reshape(r * NUM_LABELS, k, hdim)
        transformer_mask = (
            ~smask.unsqueeze(1).expand(r, NUM_LABELS, k).reshape(r * NUM_LABELS, k)
        )
        hidden = self.slice_transformer(hidden, src_key_padding_mask=transformer_mask)
        hidden = hidden.reshape(r, NUM_LABELS, k, hdim)

        slice_scores = torch.einsum(
            "rlkh,lh->rlk", hidden, self.slice_queries
        ) / math.sqrt(hdim)
        slice_scores = slice_scores.masked_fill(~smask.unsqueeze(1), -1e4)
        slice_attention = torch.softmax(slice_scores, dim=-1)
        pooled_series = torch.einsum("rlk,rlkh->rlh", slice_attention, hidden)

        flat_meta = series_meta.reshape(b * s, 3)[real_series_flat]
        flat_cont = series_cont.reshape(b * s, 2)[real_series_flat]
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

        series_repr = torch.zeros(
            b * s, NUM_LABELS, hdim, device=features.device, dtype=pooled_series.dtype
        )
        series_repr[real_series_flat] = pooled_series
        series_repr = series_repr.reshape(b, s, NUM_LABELS, hdim)

        series_scores = torch.einsum(
            "bslh,lh->bsl", series_repr, self.series_queries
        ) / math.sqrt(hdim)
        series_scores = series_scores.masked_fill(~series_mask_eff.unsqueeze(-1), -1e4)
        series_attention = torch.softmax(series_scores, dim=1)
        study_repr = torch.einsum("bsl,bslh->blh", series_attention, series_repr)
        study_repr = self.dropout(self.final_norm(study_repr))
        logits = (study_repr * self.classifier_weight.unsqueeze(0)).sum(
            dim=-1
        ) + self.classifier_bias
        return {
            "logits": logits,
            "spatial_attention": spatial_attention,
            "slice_attention": slice_attention,
            "series_attention": series_attention,
        }


# ============================================================
# 11. LOSSES / SUPERVISION ARRAYS
# ============================================================


def compute_gold_pos_weight(
    train_labels: np.ndarray, device: torch.device
) -> torch.Tensor:
    positives = train_labels.sum(axis=0).astype(np.float64)
    negatives = len(train_labels) - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 5.0).astype(np.float32)
    return torch.tensor(weights, device=device, dtype=torch.float32)


def macro_gold_bce(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none", pos_weight=pos_weight
    )
    return cell.mean(dim=0).mean()


def macro_weighted_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    effective = mask.float() * weight.float()
    weight_sum = effective.sum(dim=0)
    active = weight_sum > 0
    if not active.any():
        raise RuntimeError("Pseudo batch contains no active weighted target cells")
    per_label = (cell * effective).sum(dim=0) / weight_sum.clamp_min(1e-6)
    return per_label[active].mean(), weight_sum


def _sample_rows(
    rng: np.random.Generator, available: np.ndarray, size: int
) -> np.ndarray:
    if len(available) == 0:
        raise RuntimeError("Cannot sample from empty index set")
    return available[rng.integers(0, len(available), size=size)]


def teacher_arrays(
    store: FeatureStore, teacher: TeacherData
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = store.indices_for_uids(teacher.uids)
    return indices, teacher.probability, teacher.mask, teacher.weight


def gold_arrays(
    store: FeatureStore, gold_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    return store.indices_for_uids(gold_df[UID].astype(str).tolist()), gold_df[
        LABELS
    ].to_numpy(dtype=np.float32)


# ============================================================
# 12. MODEL FACTORY / TRAINING CONFIG
# ============================================================


def make_model(variant: str, device: torch.device) -> nn.Module:
    if variant == "w60":
        return W4ExactCuriaHierarchicalDiagnosisHead().to(device)
    if variant == "w61":
        return SpatialCuriaHierarchicalDiagnosisHead().to(device)
    raise ValueError(variant)


def training_config(variant: str, scope: str, seed: int) -> Dict[str, Any]:
    base = {
        "stage": variant,
        "scope": scope,
        "seed": seed,
        "fold_sha256": EXPECTED_FOLD_SHA256,
        "teacher": "W2.6-P fast final hybrid",
        "teacher_probability_file": W26_PROB_FILE,
        "teacher_weight_file": W26_WEIGHT_FILE,
        "teacher_mask_file": W26_MASK_FILE,
        "teacher_weight_mode": TEACHER_WEIGHT_MODE,
        "gold_authority": GOLD_AUTHORITY,
        "pseudo_authority": PSEUDO_AUTHORITY,
        "epochs": HEAD_EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "gold_batch": GOLD_BATCH_SIZE,
        "pseudo_batch": (
            PSEUDO_BATCH_SIZE_W60 if variant == "w60" else PSEUDO_BATCH_SIZE_W61
        ),
        "max_lr": HEAD_MAX_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP_NORM,
        "slice_dropout": SLICE_DROPOUT,
        "series_dropout": SERIES_DROPOUT,
        "pristine_oof": False,
        "validation_warning": "All-58-derived W2.6-P pseudo labels make gold diagnostics non-pristine.",
    }
    if variant == "w60":
        base.update(
            {
                "representation": "canonical W4 per-slice Curia CLS",
                "head_hidden": HEAD_HIDDEN_DIM,
                "head_heads": HEAD_NUM_HEADS,
                "head_layers": HEAD_TRANSFORMER_LAYERS,
                "head_dropout": HEAD_DROPOUT,
            }
        )
    else:
        base.update(
            {
                "representation": "canonical W4 CLS + compact Curia patch grid",
                "spatial_grid": SPATIAL_GRID,
                "spatial_hidden": SPATIAL_HIDDEN_DIM,
                "spatial_heads": SPATIAL_NUM_HEADS,
                "spatial_slice_layers": SPATIAL_SLICE_LAYERS,
                "spatial_dropout": SPATIAL_DROPOUT,
            }
        )
    return base


# ============================================================
# 13. PREDICTION + TRAINING CORE
# ============================================================


def predict_indices(
    model: nn.Module,
    store: FeatureStore,
    indices: np.ndarray,
    batch_size: int,
    runtime: Runtime,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            batch = store.make_batch(part, runtime.device)
            with autocast_context(runtime):
                logits = model(**batch)["logits"]
            outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
            del batch, logits
    return (
        np.concatenate(outputs, axis=0).astype(np.float32)
        if outputs
        else np.zeros((0, NUM_LABELS), dtype=np.float32)
    )


def train_model(
    variant: str,
    scope: str,
    seed: int,
    store: FeatureStore,
    gold_train_indices: np.ndarray,
    gold_train_targets: np.ndarray,
    teacher: TeacherData,
    runtime: Runtime,
    checkpoint_path: Path,
    history_path: Path,
    val_indices: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None,
    val_uids: Optional[Sequence[str]] = None,
) -> Tuple[nn.Module, pd.DataFrame, Optional[pd.DataFrame]]:
    config = training_config(variant, scope, seed)
    config_hash = sha256_json(config)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("config_hash") != config_hash:
            raise RuntimeError(
                f"Stale/incompatible checkpoint {checkpoint_path}. Config hash differs; use a new output root or delete it deliberately."
            )
        model = make_model(variant, runtime.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        hist = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        pred_path = checkpoint_path.with_name(
            checkpoint_path.stem + "_diagnostic_predictions.csv"
        )
        pred_df = pd.read_csv(pred_path) if pred_path.exists() else None
        log(f"{variant} {scope}: compatible checkpoint exists -> reuse")
        return model, hist, pred_df

    seed_everything(seed)
    model = make_model(variant, runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HEAD_MAX_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HEAD_MAX_LR,
        total_steps=HEAD_EPOCHS * STEPS_PER_EPOCH,
        pct_start=0.10,
        anneal_strategy="cos",
    )
    scaler = make_grad_scaler(runtime)
    gold_pos_weight = compute_gold_pos_weight(gold_train_targets, runtime.device)

    pseudo_indices, pseudo_targets, pseudo_masks, pseudo_weights = teacher_arrays(
        store, teacher
    )
    gold_target_map = {
        int(index): gold_train_targets[row]
        for row, index in enumerate(gold_train_indices)
    }
    gold_rng = np.random.default_rng(seed + 1000)
    pseudo_rng = np.random.default_rng(seed + 2000)
    pseudo_batch_size = (
        PSEUDO_BATCH_SIZE_W60 if variant == "w60" else PSEUDO_BATCH_SIZE_W61
    )

    history_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    started = time.time()

    log("\n" + "-" * 92)
    log(f"{variant.upper()} | {scope} | seed={seed}")
    log("-" * 92)
    log(f"Gold train studies     : {len(gold_train_indices)}")
    log(f"Pseudo studies         : {len(pseudo_indices)}")
    log(f"Teacher selected cells : {int(pseudo_masks.sum())}")
    log(f"Teacher weight mode    : {TEACHER_WEIGHT_MODE}")
    log(
        f"Loss authority         : gold={GOLD_AUTHORITY:g}, pseudo={PSEUDO_AUTHORITY:g}"
    )

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        total_gold = total_pseudo = total_loss_value = 0.0
        pseudo_mass = np.zeros(NUM_LABELS, dtype=np.float64)

        for _step in range(STEPS_PER_EPOCH):
            sampled_gold = _sample_rows(gold_rng, gold_train_indices, GOLD_BATCH_SIZE)
            gold_batch = store.make_batch(sampled_gold, runtime.device)
            gold_target_np = np.stack(
                [gold_target_map[int(i)] for i in sampled_gold], axis=0
            )
            gold_target = torch.tensor(
                gold_target_np, device=runtime.device, dtype=torch.float32
            )

            pseudo_positions = pseudo_rng.integers(
                0, len(pseudo_indices), size=pseudo_batch_size
            )
            sampled_pseudo = pseudo_indices[pseudo_positions]
            pseudo_batch = store.make_batch(sampled_pseudo, runtime.device)
            pseudo_target = torch.tensor(
                pseudo_targets[pseudo_positions],
                device=runtime.device,
                dtype=torch.float32,
            )
            pseudo_mask = torch.tensor(
                pseudo_masks[pseudo_positions], device=runtime.device, dtype=torch.bool
            )
            pseudo_weight = torch.tensor(
                pseudo_weights[pseudo_positions],
                device=runtime.device,
                dtype=torch.float32,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                gold_logits = model(**gold_batch)["logits"]
                gold_loss = macro_gold_bce(gold_logits, gold_target, gold_pos_weight)
                pseudo_logits = model(**pseudo_batch)["logits"]
                pseudo_loss, mass = macro_weighted_soft_bce(
                    pseudo_logits, pseudo_target, pseudo_mask, pseudo_weight
                )
                loss = (GOLD_AUTHORITY * gold_loss + PSEUDO_AUTHORITY * pseudo_loss) / (
                    GOLD_AUTHORITY + PSEUDO_AUTHORITY
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            scheduler.step()

            total_gold += float(gold_loss.detach().cpu())
            total_pseudo += float(pseudo_loss.detach().cpu())
            total_loss_value += float(loss.detach().cpu())
            pseudo_mass += mass.detach().cpu().numpy()
            del (
                gold_batch,
                pseudo_batch,
                gold_target,
                pseudo_target,
                pseudo_mask,
                pseudo_weight,
                gold_logits,
                pseudo_logits,
            )

        row: Dict[str, Any] = {
            "Variant": variant,
            "Scope": scope,
            "Seed": seed,
            "Epoch": epoch,
            "GoldLoss": total_gold / STEPS_PER_EPOCH,
            "PseudoLoss": total_pseudo / STEPS_PER_EPOCH,
            "TotalLoss": total_loss_value / STEPS_PER_EPOCH,
            "LearningRate": optimizer.param_groups[0]["lr"],
            "ElapsedSeconds": time.time() - started,
        }
        for j, label in enumerate(LABELS):
            row[f"PseudoMass::{label}"] = float(pseudo_mass[j])

        if val_indices is not None and val_targets is not None:
            pred = predict_indices(
                model, store, val_indices, VALIDATION_BATCH_SIZE, runtime
            )
            _table, summary = metric_tables(val_targets.astype(np.int64), pred)
            row.update(
                {
                    "DiagnosticMacroAUROC": summary["macro_AUROC"],
                    "DiagnosticMacroAP": summary["macro_AP"],
                    "DiagnosticMacroF1": summary["macro_F1"],
                    "DiagnosticIsPristineOOF": False,
                }
            )
            if epoch == HEAD_EPOCHS and val_uids is not None:
                for uid, y, p in zip(val_uids, val_targets, pred):
                    out = {
                        UID: str(uid),
                        "Variant": variant,
                        "Scope": scope,
                        "Seed": seed,
                        "Epoch": epoch,
                        "PristineOOF": False,
                    }
                    for j, label in enumerate(LABELS):
                        out[f"Truth::{label}"] = int(y[j])
                        out[f"Probability::{label}"] = float(p[j])
                    diagnostic_rows.append(out)
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        metric_text = (
            f" diag_auc={row.get('DiagnosticMacroAUROC', float('nan')):.5f}"
            if "DiagnosticMacroAUROC" in row
            else ""
        )
        log(
            f"epoch {epoch:02d}/{HEAD_EPOCHS} loss={row['TotalLoss']:.5f} "
            f"gold={row['GoldLoss']:.5f} pseudo={row['PseudoLoss']:.5f}{metric_text}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "variant": variant,
        "scope": scope,
        "seed": seed,
        "config": config,
        "config_hash": config_hash,
        "created_at": now_iso(),
        "pristine_oof": False,
        "warning": "Model trained with all-58-derived production pseudo labels.",
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    pred_df = pd.DataFrame(diagnostic_rows) if diagnostic_rows else None
    if pred_df is not None:
        pred_df.to_csv(
            checkpoint_path.with_name(
                checkpoint_path.stem + "_diagnostic_predictions.csv"
            ),
            index=False,
        )
    return model, pd.DataFrame(history_rows), pred_df


# ============================================================
# 14. CV / FULL-FIT ORCHESTRATION
# ============================================================


def _store_for_variant(train_df: pd.DataFrame, variant: str) -> FeatureStore:
    external = discover_external_w4_cache() if variant == "w60" else None
    store = FeatureStore(
        train_df[UID].astype(str).tolist(), variant, external_w4_cache=external
    )
    summary = store.validate_all()
    if summary["missing"] or summary["invalid"]:
        mode = "cache_w60" if variant == "w60" else "cache_w61"
        raise RuntimeError(
            f"{variant} feature cache incomplete: {summary}. Run {mode} first."
        )
    return store


def train_cv(variant: str, accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    train_df, _series_df, gold_df, fold_zero = load_train_tables(require_dicom=False)
    teacher = load_teacher(train_df, strict=True)
    store = _store_for_variant(train_df, variant)
    gold_idx, gold_y = gold_arrays(store, gold_df)

    fold_predictions: List[pd.DataFrame] = []
    fold_summaries = []
    root = CHECKPOINT_ROOT / variant / "cv"
    result_root = RESULT_ROOT / variant / "cv"
    root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    for fold in range(1, NUM_FOLDS + 1):
        train_rows = np.where(fold_zero != fold - 1)[0]
        val_rows = np.where(fold_zero == fold - 1)[0]
        seed = 60_000 + (1000 if variant == "w61" else 0) + fold
        ckpt = root / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"
        history = result_root / f"fold_{fold}_history.csv"
        _model, hist, pred_df = train_model(
            variant,
            f"cv_fold_{fold}",
            seed,
            store,
            gold_idx[train_rows],
            gold_y[train_rows],
            teacher,
            runtime,
            ckpt,
            history,
            val_indices=gold_idx[val_rows],
            val_targets=gold_y[val_rows],
            val_uids=gold_df.iloc[val_rows][UID].astype(str).tolist(),
        )
        if pred_df is not None:
            pred_df["OuterFold"] = fold
            fold_predictions.append(pred_df)
        if not hist.empty:
            fold_summaries.append(hist.iloc[-1].to_dict())
        del _model
        if runtime.accelerator == "cuda":
            torch.cuda.empty_cache()

    combined = (
        pd.concat(fold_predictions, ignore_index=True)
        if fold_predictions
        else pd.DataFrame()
    )
    summary: Dict[str, Any] = {
        "variant": variant,
        "diagnostic_is_pristine_oof": False,
        "warning": "These fold-held-out gold predictions are contaminated by all-58-derived production pseudo labels and are diagnostic only.",
        "folds": fold_summaries,
    }
    if not combined.empty:
        truth = np.column_stack(
            [combined[f"Truth::{label}"].to_numpy(dtype=np.int64) for label in LABELS]
        )
        prob = np.column_stack(
            [
                combined[f"Probability::{label}"].to_numpy(dtype=np.float32)
                for label in LABELS
            ]
        )
        table, metrics = metric_tables(truth, prob)
        combined.to_csv(
            result_root / "diagnostic_gold_predictions_all_folds.csv", index=False
        )
        table.to_csv(result_root / "diagnostic_per_label_metrics.csv", index=False)
        summary["diagnostic_metrics"] = metrics
        log(
            f"{variant.upper()} diagnostic macro AUROC = {metrics['macro_AUROC']:.6f} (NON-PRISTINE)"
        )
    atomic_json_dump(summary, result_root / "diagnostic_summary.json")
    return summary


def train_full(variant: str, accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    train_df, _series_df, gold_df, _fold_zero = load_train_tables(require_dicom=False)
    teacher = load_teacher(train_df, strict=True)
    store = _store_for_variant(train_df, variant)
    gold_idx, gold_y = gold_arrays(store, gold_df)

    root = CHECKPOINT_ROOT / variant / "full"
    result_root = RESULT_ROOT / variant / "full"
    root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    completed = []

    for seed in FULLFIT_SEEDS:
        ckpt = root / f"seed_{seed}_epoch_{HEAD_EPOCHS}.pt"
        history = result_root / f"seed_{seed}_history.csv"
        model, hist, _ = train_model(
            variant,
            "full_all58",
            seed,
            store,
            gold_idx,
            gold_y,
            teacher,
            runtime,
            ckpt,
            history,
        )
        completed.append(
            {
                "seed": seed,
                "checkpoint": str(ckpt),
                "final_loss": (
                    float(hist.iloc[-1]["TotalLoss"]) if not hist.empty else None
                ),
            }
        )
        del model
        if runtime.accelerator == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "variant": variant,
        "scope": "full_all58",
        "seeds": FULLFIT_SEEDS,
        "models": completed,
        "use": "production hidden-test inference",
        "validation_note": "Full-fit uses all 58 gold labels and all W2.6-P pseudo labels; no OOF claim.",
    }
    atomic_json_dump(summary, result_root / "fullfit_summary.json")
    return summary


# ============================================================
# 15. TEST TABLES / TEST CACHE
# ============================================================


def load_test_tables(
    require_dicom: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (TEST_CSV, TEST_SERIES_CSV, SAMPLE_SUBMISSION):
        if not path.exists():
            raise FileNotFoundError(path)
    if require_dicom and not TEST_SERIES_ROOT.exists():
        raise FileNotFoundError(TEST_SERIES_ROOT)
    test_df = pd.read_csv(TEST_CSV)
    series_df = pd.read_csv(TEST_SERIES_CSV)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    test_df[UID] = test_df[UID].astype(str)
    series_df[UID] = series_df[UID].astype(str)
    series_df[SERIES_UID] = series_df[SERIES_UID].astype(str)
    sample[UID] = sample[UID].astype(str)
    if list(sample.columns) != [UID, *LABELS]:
        raise RuntimeError(
            f"sample_submission.csv schema mismatch: {list(sample.columns)}"
        )
    if set(sample[UID]) != set(test_df[UID]):
        raise RuntimeError("sample_submission.csv UID set differs from test.csv")
    series_df["_fluid"] = bool_series(series_df["Fluid_Sensitive"]).astype(int)
    series_df["_fs"] = bool_series(series_df["Fat_Suppression"]).astype(int)
    return test_df, series_df, sample


def _test_cache_variant_root(variant: str) -> Path:
    root = TEST_CACHE_ROOT / variant / "studies"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _test_cache_path(variant: str, uid: str) -> Path:
    return _test_cache_variant_root(variant) / f"{stable_uid_hash(uid)}.pt"


def _test_cache_worker(
    worker_id: int,
    device: torch.device,
    variant: str,
    study_uids: Sequence[str],
    uid_groups: Mapping[str, pd.DataFrame],
    curia_root: Path,
    progress: Dict[str, int],
    progress_lock: threading.Lock,
    total_requested: int,
    started: float,
) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    processor, base_model = load_curia_processor_and_model(curia_root, device)
    if variant == "w61":
        model: nn.Module = CuriaPatchCapture(base_model, SPATIAL_GRID).to(device).eval()
    else:
        model = base_model
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoded = 0
    failures: List[Dict[str, Any]] = []
    log(
        f"[test cache worker {worker_id}] variant={variant} device={device} studies={len(study_uids)}"
    )

    for uid in study_uids:
        path = _test_cache_path(variant, uid)
        try:
            if path.exists():
                try:
                    existing = torch.load(path, map_location="cpu", weights_only=False)
                    okay = (
                        w61_payload_usable(existing, uid)
                        if variant == "w61"
                        else w4_payload_usable(existing, uid)
                    )
                    if okay:
                        continue
                except Exception:
                    pass

            if uid not in uid_groups:
                raise RuntimeError(f"No test_series rows for {uid}")

            if variant == "w60":
                payload, _audit = encode_study(
                    uid,
                    uid_groups[uid],
                    processor,
                    model,
                    device,
                    ENCODER_BATCH,
                    TEST_SERIES_ROOT,
                )
                if not w4_payload_usable(payload, uid):
                    raise RuntimeError(f"Invalid W60 test cache for {uid}")
            else:
                if not isinstance(model, CuriaPatchCapture):
                    raise RuntimeError("Internal W61 test-cache model type mismatch")
                model.reset_capture()
                payload, _audit = encode_study(
                    uid,
                    uid_groups[uid],
                    processor,
                    model,
                    device,
                    SPATIAL_ENCODER_BATCH,
                    TEST_SERIES_ROOT,
                )
                captured = torch.cat(model.captured, dim=0) if model.captured else None
                expected = int(payload["slice_mask"].sum().item())
                expected_shape = (expected, SPATIAL_TOKENS, EXPECTED_CURIA_HIDDEN)
                if captured is None or tuple(captured.shape) != expected_shape:
                    raise RuntimeError(
                        f"Spatial capture mismatch for {uid}: "
                        f"{None if captured is None else tuple(captured.shape)} vs {expected_shape}"
                    )
                series_count, slice_count, _hidden = payload["features"].shape
                patch_grid = torch.zeros(
                    series_count,
                    slice_count,
                    SPATIAL_TOKENS,
                    EXPECTED_CURIA_HIDDEN,
                    dtype=torch.float16,
                )
                cursor = 0
                for series_index in range(series_count):
                    for slice_index in range(slice_count):
                        if bool(payload["slice_mask"][series_index, slice_index]):
                            patch_grid[series_index, slice_index] = captured[cursor]
                            cursor += 1
                payload["w4_cache_version"] = payload.get("cache_version")
                payload["cache_version"] = SPATIAL_CACHE_VERSION
                payload["patch_grid"] = patch_grid.contiguous()
                payload["patch_grid_shape"] = [SPATIAL_GRID, SPATIAL_GRID]
                if not w61_payload_usable(payload, uid):
                    raise RuntimeError(f"Invalid W61 test cache for {uid}")

            atomic_torch_save(payload, path)
            encoded += 1
        except Exception as exc:
            failures.append(
                {
                    UID: uid,
                    "Worker": worker_id,
                    "Device": str(device),
                    "error": repr(exc),
                }
            )
            log(f"[test cache worker {worker_id}] ERROR {uid}: {exc}")
        finally:
            with progress_lock:
                progress["done"] += 1
                done = int(progress["done"])
            if done <= 10 or done % 50 == 0 or done == total_requested:
                elapsed = max(time.time() - started, 1e-6)
                rate = done / elapsed
                eta = (total_requested - done) / max(rate, 1e-6)
                log(
                    f"[test cache {variant}] {done}/{total_requested} "
                    f"({100.0*done/max(total_requested,1):.1f}%) elapsed={elapsed/60:.1f}m "
                    f"ETA={eta/60:.1f}m worker={worker_id} encoded={encoded} failures={len(failures)}"
                )

    del model, base_model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "worker_id": worker_id,
        "device": str(device),
        "encoded": encoded,
        "failures": failures,
    }


def build_test_cache(variant: str, accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    test_df, series_df, _sample = load_test_tables(require_dicom=True)
    uid_groups = {
        uid: group.copy() for uid, group in series_df.groupby(UID, sort=False)
    }
    uids = test_df[UID].astype(str).tolist()

    need: List[str] = []
    for uid in uids:
        path = _test_cache_path(variant, uid)
        okay = False
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                okay = (
                    w61_payload_usable(payload, uid)
                    if variant == "w61"
                    else w4_payload_usable(payload, uid)
                )
            except Exception:
                okay = False
        if not okay:
            need.append(uid)

    if not need:
        return {
            "variant": variant,
            "test_studies": len(uids),
            "encoded": 0,
            "status": "reused",
        }

    curia_root = discover_curia_root()
    model_sha, config = validate_curia_identity(curia_root)
    gpu_ids = cache_gpu_ids(runtime)
    if runtime.accelerator == "cuda":
        worker_devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
    else:
        worker_devices = [runtime.device]
    if not worker_devices:
        worker_devices = [runtime.device]

    log(f"[test cache {variant}] Curia root={curia_root}")
    log(f"[test cache {variant}] Need encoding={len(need)}/{len(uids)}")
    log(f"[test cache {variant}] Worker devices={[str(x) for x in worker_devices]}")

    shards = [need[i :: len(worker_devices)] for i in range(len(worker_devices))]
    progress = {"done": 0}
    progress_lock = threading.Lock()
    started = time.time()
    worker_results: List[Dict[str, Any]] = []

    if len(worker_devices) == 1:
        worker_results.append(
            _test_cache_worker(
                0,
                worker_devices[0],
                variant,
                shards[0],
                uid_groups,
                curia_root,
                progress,
                progress_lock,
                len(need),
                started,
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
            futures = [
                executor.submit(
                    _test_cache_worker,
                    worker_id,
                    device,
                    variant,
                    shards[worker_id],
                    uid_groups,
                    curia_root,
                    progress,
                    progress_lock,
                    len(need),
                    started,
                )
                for worker_id, device in enumerate(worker_devices)
            ]
            for future in as_completed(futures):
                worker_results.append(future.result())

    failures: List[Dict[str, Any]] = []
    encoded = 0
    for result in worker_results:
        failures.extend(result["failures"])
        encoded += int(result["encoded"])

    if failures:
        failure_path = RESULT_ROOT / f"test_cache_{variant}_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        raise RuntimeError(
            f"{variant} test cache completed with {len(failures)} failures; see {failure_path}"
        )

    for uid in uids:
        path = _test_cache_path(variant, uid)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        okay = (
            w61_payload_usable(payload, uid)
            if variant == "w61"
            else w4_payload_usable(payload, uid)
        )
        if not okay:
            raise RuntimeError(f"Invalid final {variant} test cache for {uid}")

    summary = {
        "variant": variant,
        "test_studies": len(uids),
        "encoded": encoded,
        "status": "complete",
        "worker_devices": [str(x) for x in worker_devices],
        "curia_root": str(curia_root),
        "curia_sha256": model_sha,
        "curia_config": config,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(summary, RESULT_ROOT / f"test_cache_{variant}_summary.json")
    return summary


class TestFeatureStore(FeatureStore):
    def __init__(self, uids: Sequence[str], variant: str):
        super().__init__(
            uids, variant, external_w4_cache=None, lru_size=FEATURE_LRU_SIZE
        )

    def path_for_uid(self, uid: str) -> Path:
        uid = str(uid)
        own = _test_cache_path(self.variant, uid)
        if own.exists():
            return own
        if self.variant == "w60":
            spatial = _test_cache_path("w61", uid)
        if spatial.exists():
            return spatial
        return own


def _full_checkpoints(variant: str) -> List[Path]:
    root = CHECKPOINT_ROOT / variant / "full"
    paths = [root / f"seed_{seed}_epoch_{HEAD_EPOCHS}.pt" for seed in FULLFIT_SEEDS]
    return [p for p in paths if p.exists()]


def normalized_ranks(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float32)
    for j in range(matrix.shape[1]):
        order = np.argsort(matrix[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32)
        out[:, j] = (ranks + 0.5) / max(n, 1)
    return out


def submit(accelerator: Optional[str] = None) -> Path:
    runtime = resolve_runtime(accelerator)
    if not IS_KAGGLE:
        log(
            "[W6 submit] LOCAL MODE: this is a visible-test/smoke-test submission only. "
            "The competition hidden test is supplied only during Kaggle notebook rerun; run the final submission on Kaggle."
        )
    test_df, _series_df, sample = load_test_tables(require_dicom=True)
    wanted = [v for v in FINAL_VARIANTS if v in {"w60", "w61"}]
    if not wanted:
        raise RuntimeError("W6_FINAL_VARIANTS selected no valid variants")

    # If both variants are requested, build only the richer W61 test cache. It
    # contains the exact W4 CLS tensors, so W60 can read its CLS view from it.
    if "w61" in wanted:
        build_test_cache("w61", accelerator)
    elif "w60" in wanted:
        build_test_cache("w60", accelerator)

    model_predictions: List[np.ndarray] = []
    model_ids: List[str] = []
    for variant in wanted:
        checkpoints = _full_checkpoints(variant)
        if not checkpoints:
            warnings.warn(f"No full-fit checkpoints for {variant}; skipping it.")
            continue
        store = TestFeatureStore(test_df[UID].astype(str).tolist(), variant)
        indices = np.arange(len(test_df), dtype=np.int64)
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = make_model(variant, runtime.device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            pred = predict_indices(
                model, store, indices, VALIDATION_BATCH_SIZE, runtime
            )
            model_predictions.append(pred)
            model_ids.append(f"{variant}:{ckpt_path.name}")
            del model
            if runtime.accelerator == "cuda":
                torch.cuda.empty_cache()

    if not model_predictions:
        raise RuntimeError("No full-fit W60/W61 models are available for submission")

    if FINAL_ENSEMBLE_MODE == "rankmean":
        ensemble = np.mean([normalized_ranks(p) for p in model_predictions], axis=0)
    else:
        ensemble = np.mean(model_predictions, axis=0)
    ensemble = np.clip(ensemble, 1e-6, 1.0 - 1e-6)

    # Preserve sample_submission row order exactly.
    pred_df = pd.DataFrame(ensemble, columns=LABELS)
    pred_df.insert(0, UID, test_df[UID].astype(str).values)
    pred_df = pred_df.set_index(UID).loc[sample[UID].astype(str)].reset_index()
    if list(pred_df.columns) != list(sample.columns):
        raise RuntimeError("Submission schema differs from sample_submission.csv")
    if pred_df.isna().any().any():
        raise RuntimeError("Submission contains NaN")

    tag = "_".join(sorted(set(x.split(":")[0] for x in model_ids)))
    path = SUBMISSION_ROOT / f"rsna_w6_{tag}_{FINAL_ENSEMBLE_MODE}_submission.csv"
    pred_df.to_csv(path, index=False)
    summary = {
        "created_at": now_iso(),
        "submission": str(path),
        "models": model_ids,
        "ensemble_mode": FINAL_ENSEMBLE_MODE,
        "rows": len(pred_df),
        "columns": list(pred_df.columns),
        "min_probability": float(pred_df[LABELS].to_numpy().min()),
        "max_probability": float(pred_df[LABELS].to_numpy().max()),
        "hidden_test_inputs": "MRI only; reports are not read during submit mode",
    }
    atomic_json_dump(summary, SUBMISSION_ROOT / "submission_summary.json")
    log(f"Submission written: {path}")
    return path


# ============================================================
# 17. STATUS / VALIDATION / ORCHESTRATOR
# ============================================================


def check_curia(accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    root = discover_curia_root()
    model_sha, config = validate_curia_identity(root)
    result = curia_environment_diagnostics(root)
    result.update(
        {
            "identity_sha256": model_sha,
            "identity_config": config,
            "requested_device": str(runtime.device),
            "expected_remote_size_note": "raidium/curia-2 model.safetensors is ~344 MB",
        }
    )
    return result


def status(accelerator: Optional[str] = None) -> Dict[str, Any]:
    runtime = resolve_runtime(accelerator)
    info: Dict[str, Any] = {
        "created_at": now_iso(),
        "is_kaggle": IS_KAGGLE,
        "script_dir": str(SCRIPT_DIR),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "accelerator": runtime.accelerator,
        "device": str(runtime.device),
        "precision": str(runtime.amp_dtype),
        "curia_cache_gpu_ids": (
            cache_gpu_ids(runtime) if runtime.accelerator == "cuda" else []
        ),
        "teacher_weight_mode": TEACHER_WEIGHT_MODE,
        "spatial_grid": SPATIAL_GRID,
        "paths": {},
    }

    def probe(name: str, fn):
        try:
            value = fn()
            info["paths"][name] = {"ok": True, "value": str(value)}
            return value
        except Exception as exc:
            info["paths"][name] = {"ok": False, "error": repr(exc)}
            return None

    probe("w26_root", discover_w26_root)
    probe("external_w4_cache", lambda: discover_external_w4_cache() or "none")
    probe("curia_root", discover_curia_root)
    for name, path in (
        ("train_csv", TRAIN_CSV),
        ("train_series_csv", TRAIN_SERIES_CSV),
        ("train_series_root", TRAIN_SERIES_ROOT),
        ("sample_submission", SAMPLE_SUBMISSION),
    ):
        info["paths"][name] = {"ok": path.exists(), "value": str(path)}

    try:
        train_df, _series_df, _gold_df, _fold = load_train_tables(require_dicom=False)
        info["train_table_validation"] = {
            "ok": True,
            "rows": len(train_df),
            "fold_sha256": EXPECTED_FOLD_SHA256,
        }
        teacher = load_teacher(train_df, strict=True)
        info["teacher_validation"] = {
            "ok": True,
            "rows": len(teacher.uids),
            "selected_cells": int(teacher.mask.sum()),
        }
    except Exception as exc:
        info["train_or_teacher_validation"] = {"ok": False, "error": repr(exc)}

    atomic_json_dump(info, RESULT_ROOT / "00_status.json")
    log(json.dumps(info, indent=2))
    return info


def validate_teacher() -> Dict[str, Any]:
    train_df, _series_df, _gold_df, _fold = load_train_tables(require_dicom=False)
    teacher = load_teacher(train_df, strict=True)
    log(json.dumps(teacher.summary, indent=2))
    return teacher.summary


def run_w6(mode: str, accelerator: Optional[str] = None):
    mode = mode.strip().lower()
    if mode == "status":
        return status(accelerator)
    if mode == "check_curia":
        return check_curia(accelerator)
    if mode == "validate_teacher":
        return validate_teacher()
    if mode == "cache_w60":
        return cache_w60(accelerator)
    if mode == "train_w60_cv":
        return train_cv("w60", accelerator)
    if mode == "train_w60_full":
        return train_full("w60", accelerator)
    if mode == "cache_w61":
        return cache_w61(accelerator)
    if mode == "train_w61_cv":
        return train_cv("w61", accelerator)
    if mode == "train_w61_full":
        return train_full("w61", accelerator)
    if mode == "submit":
        return submit(accelerator)
    raise ValueError(
        f"Unknown mode {mode!r}. Expected status, validate_teacher, cache_w60, "
        "train_w60_cv, train_w60_full, cache_w61, train_w61_cv, "
        "train_w61_full, submit."
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="RSNA W6 Curia pipeline")
    parser.add_argument(
        "mode",
        choices=[
            "status",
            "check_curia",
            "validate_teacher",
            "cache_w60",
            "train_w60_cv",
            "train_w60_full",
            "cache_w61",
            "train_w61_cv",
            "train_w61_full",
            "submit",
        ],
    )
    parser.add_argument(
        "--accelerator", default=None, help="auto/localGPU/kaggle_t4/apple_mps/cpu"
    )
    args = parser.parse_args()
    result = run_w6(args.mode, args.accelerator)
    # print(result)
    if isinstance(result, Path):
        print(result)


if __name__ == "__main__":
    _cli()
