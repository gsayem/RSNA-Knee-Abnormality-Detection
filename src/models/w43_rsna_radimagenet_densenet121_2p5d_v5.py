%%writefile /kaggle/working/w43_rsna_radimagenet_densenet121_2p5d_v5.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W43 — RadImageNet DenseNet-121 2.5D Multi-Plane v5
==========================================================================

PRIMARY HYPOTHESIS
------------------
Replace the frozen Curia CLS representation with a fully trainable medical-image
backbone initialized from RadImageNet DenseNet-121, while preserving W40
as the common pseudo-supervision source.

This is intentionally a new image-model family. It is NOT a W6/W41 derivative.

Backbone
--------
    DenseNet-121
    RadImageNet pretrained weights
    224 x 224 input (matches RadImageNet source training resolution)
    2.5D input: previous / center / next physical MRI slice as 3 channels

Study representation
--------------------
    all MRI series
        -> five physical positions per series
        -> 2.5D stacks
        -> RadImageNet DenseNet-121
        -> token projection + plane / FS / fluid / position metadata
        -> 2-layer study Transformer
        -> 12 label-specific attention queries
        -> soft anatomy-plane prior
        -> 12 diagnosis logits

Supervision
-----------
Stage P: W40 pseudo-label representation learning on 4,349 report-only studies.
Stage C: diagnostic 5-fold gold adaptation from the SAME Stage-P checkpoint.
Stage F: production all-58 gold adaptation + one joint stabilization epoch.

W40 remains relevant and is reused exactly:
    16_final_probabilities_wide.csv
    17_teacher_weights_wide.csv
    18_teacher_mask_wide.csv

The script does NOT regenerate W40.

Official PyTorch availability
-----------------------------
The current official PyTorch bundle provides DenseNet121.pt, ResNet50.pt,
and InceptionV3.pt. W43 therefore uses DenseNet121.pt directly; no
TensorFlow conversion or unofficial InceptionResNetV2 port is used.

Important RadImageNet guard
---------------------------
The official RadImageNet PyTorch example preprocesses pixels as:

    (pixel - 127.5) * 2 / 255

which is approximately [-1, +1]. W43 uses that exact PyTorch-demo convention,
NOT ImageNet mean/std.

RadImageNet's public repository has an unresolved preprocessing-consistency
issue between TensorFlow and PyTorch examples. W43 therefore records the chosen
convention in every checkpoint and never silently changes it.

The RadImageNet checkpoint loader is fail-loud:
    - no strict=False-and-continue behavior
    - classifier keys may be excluded
    - missing num_batches_tracked buffers may be tolerated
    - every other backbone tensor must match by name and shape
    - >= 99.5% parameter-numel coverage is required

DenseNet / Kaggle-T4 precision policy
--------------------------------------
W43 v5 forces the RadImageNet DenseNet-121 backbone to FP32.

The previous v4 allowed the entire model to enter CUDA autocast. On Kaggle T4,
that means FP16 rather than BF16. With this RadImageNet DenseNet checkpoint's
large internal activation/BN scales, forward FP16 overflow can occur before
GradScaler has any opportunity to protect training.

v5 therefore uses:
    DenseNet backbone          -> FP32
    Study Transformer / heads -> CUDA autocast when available
    BCE loss                   -> FP32
    gradients                  -> GradScaler when FP16 is active

A real cached-knee numerical forward preflight must pass before epoch 1.

DenseNet BatchNorm policy
-------------------------
The official DenseNet121.pt checkpoint contains unusually large BatchNorm
running variances (also reported publicly for this checkpoint). W43 v5 does
NOT train against those source-domain running statistics unchanged.

Before Stage-P pseudo training, W43 v5:
    1. resets DenseNet backbone BN running_mean/running_var,
    2. recalibrates them on deterministic W40-unlabeled knee 2.5D stacks,
    3. validates the recalibrated buffers are finite/positive,
    4. freezes BN running statistics for ALL subsequent training stages.

BN affine gamma/beta parameters may still learn when their DenseNet blocks are
unfrozen; only running-stat updates are frozen. This avoids small-study-batch
BN drift while preserving the official convolutional and affine weights.

If the official DenseNet-121 PyTorch checkpoint uses a different key
layout, `status` will report the checkpoint audit and training will remain
blocked rather than silently using random weights.

DICOM decoder preflight
-----------------------
The competition mixes uncompressed DICOM, JPEG Lossless and JPEG 2000 transfer
syntaxes. Having `pydicom` installed is NOT sufficient to decode every study.

W43 v5 performs a real pixel-decode preflight before cache construction:
    - samples actual train DICOM files,
    - records TransferSyntaxUID values,
    - attempts pixel decode for every observed sampled syntax,
    - fails immediately if any required codec is unavailable.

Recommended local decoder packages:
    python -m pip install -U pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg

The cache will never again process all 4,407 studies before reporting a missing
compressed-pixel decoder.

Backbone-independent 2.5D cache
-------------------------------
W43 creates:

    output/results/rsna_2p5d_mri_cache_v1/studies/<md5(uid)>.npz

The cache is intentionally NOT named after DenseNet-121. W44
Inception-v3 must reuse the exact same cache for a clean backbone comparison.

CLI
---
    python w43_rsna_radimagenet_densenet121_2p5d_v5.py status --accelerator localGPU

    python w43_rsna_radimagenet_densenet121_2p5d_v5.py cache \
        --accelerator localGPU --cache-workers 8

    python w43_rsna_radimagenet_densenet121_2p5d_v5.py train_pseudo \
        --accelerator localGPU

    python w43_rsna_radimagenet_densenet121_2p5d_v5.py train_cv \
        --accelerator localGPU

    python w43_rsna_radimagenet_densenet121_2p5d_v5.py train_full \
        --accelerator localGPU

    python w43_rsna_radimagenet_densenet121_2p5d_v5.py train_all \
        --accelerator localGPU

    python w43_rsna_radimagenet_densenet121_2p5d_v5.py validate \
        --accelerator localGPU

Explicit RadImageNet weights:
    --radimagenet-weights /path/to/DenseNet121.pt

No project .py imports are used.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pydicom
except Exception:
    pydicom = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# =============================================================================
# VERSION / COMPETITION CONSTANTS
# =============================================================================

SCRIPT_VERSION = "w43_radimagenet_densenet121_2p5d_v5"
DISPLAY_VERSION = "W43 | RadImageNet DenseNet-121 2.5D Multi-Plane v5"
OUTPUT_DIR_NAME = "rsna_w43_radimagenet_densenet121_2p5d_v5"

UID_COLUMN = "StudyInstanceUID"
SERIES_UID_COLUMN = "SeriesInstanceUID"
REPORT_COLUMN = "Report"

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

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_SELECTED_CELLS = 32027

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

W40_CHANGED_LABELS = ["Contusion", "Effusion"]


# =============================================================================
# 2.5D CACHE — SHARED WITH FUTURE W44 INCEPTION-V3
# =============================================================================

CACHE_VERSION = "rsna_2p5d_mri_cache_v1"
CACHE_IMAGE_SIZE = 224
STACKS_PER_SERIES = 5
STACK_OFFSETS = (-1, 0, +1)

# Same geometry guard values used by the canonical W4 preprocessing.
ORIENTATION_MIN_ALIGNMENT = 0.70
GEOMETRY_PLANE_CONFIDENCE = 0.80

# DICOM decoder preflight. We inspect one real DICOM from a deterministic sample
# of series and then pixel-decode one representative file per observed syntax.
DICOM_PREFLIGHT_SERIES = 256
DICOM_PREFLIGHT_REQUIRED_DECODE_PER_SYNTAX = 1
DICOM_DECODER_INSTALL_COMMAND = (
    "python -m pip install -U pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
)

# This global is intentionally assigned by ProjectPaths.configure_globals().
TRAIN_SERIES_ROOT = Path(".")


# =============================================================================
# MODEL / TRAINING CONSTANTS
# =============================================================================

BACKBONE_NAME = "densenet121"
BACKBONE_FEATURE_DIM = 1024
TOKEN_DIM = 512
TOKEN_HEADS = 8
TOKEN_LAYERS = 2
TOKEN_DROPOUT = 0.15

TRAIN_TOKENS_PER_STUDY = 6
EVAL_MAX_TOKENS = 64

PSEUDO_EPOCHS = 3
PSEUDO_STAGES = ["frozen", "top", "full"]
PSEUDO_BACKBONE_LR = {
    "frozen": 0.0,
    "top": 2e-5,
    "full": 5e-6,
}
PSEUDO_HEAD_LR = 2e-4

CV_ADAPT_EPOCHS = 3
CV_STAGE_SCHEDULE = ["top", "full", "full"]

FULL_GOLD_EPOCHS = 4
FULL_STAGE_SCHEDULE = ["top", "full", "full", "full"]
FULL_JOINT_EPOCHS = 1

GOLD_BACKBONE_LR_TOP = 1e-5
GOLD_BACKBONE_LR_FULL = 3e-6
GOLD_HEAD_LR = 1e-4
JOINT_BACKBONE_LR = 2e-6
JOINT_HEAD_LR = 5e-5

WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 2.0

JOINT_GOLD_AUTHORITY = 4.0
JOINT_PSEUDO_AUTHORITY = 1.0
JOINT_STEPS_PER_EPOCH = 32

CV_SEED_BASE = 43000
PSEUDO_SEED = 4300
FULL_SEEDS = [4301, 4302, 4303]

# Diagnostic gate is intentionally conservative. W40 production pseudo labels use
# all 58 gold reports as exemplars, so the image CV is NON-PRISTINE and must not be
# treated as an unbiased performance estimate.
CV_GO_MIN_MACRO_AUC = 0.76
CV_GO_MAX_WEAK_LABELS = 3
CV_WEAK_LABEL_AUC = 0.55

RADIMAGENET_PREPROCESSING = "pytorch_demo_minus1_plus1"
RADIMAGENET_MIN_NUMEL_COVERAGE = 0.995

# Official DenseNet121.pt carries extreme source-domain BN running variances.
# Re-estimate ONLY BN running buffers on our knee-MRI domain before training.
BN_STRATEGY = "reset_recalibrate_fp32_on_w40_unlabeled_then_freeze_running_stats"
BN_CALIBRATION_STUDIES = 256
BN_CALIBRATION_MAX_TOKENS_PER_STUDY = 32
BN_CALIBRATION_MAX_STACKS = 8192
BN_CALIBRATION_MIN_STACKS = 4096
BN_CALIBRATION_IMAGE_BATCH = 32
BN_CALIBRATION_SEED = 4317

# T4 mixed precision is FP16. Keep the RadImageNet DenseNet forward in FP32
# because this checkpoint carries unusually large internal activation/BN scales.
BACKBONE_FORCE_FP32 = True
NUMERICAL_PREFLIGHT_BATCHES = 2


# =============================================================================
# SOFT ANATOMY PLANE PRIORS
# =============================================================================

# Plane order: Axial, Coronal, Sagittal.
# These are soft positive preferences, not hard exclusions.
_PLANE_PRIOR_WEIGHTS = {
    "ACL": [0.55, 0.85, 1.00],
    "MCL": [0.50, 1.00, 0.75],
    "Medial Meniscus": [0.55, 0.95, 1.00],
    "Lateral Meniscus": [0.55, 0.95, 1.00],
    "Medial OA": [0.60, 1.00, 0.80],
    "Lateral OA": [0.60, 1.00, 0.80],
    "PF OA": [1.00, 0.55, 0.85],
    "Effusion": [0.95, 0.65, 1.00],
    "Synovitis": [0.95, 0.65, 1.00],
    "Baker's": [0.55, 0.65, 1.00],
    "Contusion": [0.90, 0.90, 1.00],
    "Fracture": [0.90, 0.90, 1.00],
}

PLANE_PRIOR = torch.tensor(
    [_PLANE_PRIOR_WEIGHTS[label] for label in LABEL_COLUMNS],
    dtype=torch.float32,
)
PLANE_PRIOR_LOG = torch.log(PLANE_PRIOR.clamp_min(1e-3))


# =============================================================================
# GENERIC UTILITIES
# =============================================================================


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
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
            default=default,
        ),
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float) > 0.5

    values = series.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "yes", "y", "t"}
    false_values = {"0", "false", "no", "n", "f", "", "nan", "none"}

    unexpected = set(values.unique()) - true_values - false_values
    if unexpected:
        raise RuntimeError(f"Unexpected boolean-like values: {sorted(unexpected)}")

    return values.isin(true_values)


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    frame = assignments[[UID_COLUMN, "OuterFold"]].copy()
    frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
    frame["OuterFold"] = frame["OuterFold"].astype(int)
    frame = frame.sort_values(UID_COLUMN)
    payload = "".join(
        f"{uid},{fold}\n" for uid, fold in zip(frame[UID_COLUMN], frame["OuterFold"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


# =============================================================================
# PROJECT PATHS
# =============================================================================


def _script_project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        if here.parent.name == "models" and here.parent.parent.name == "src":
            return here.parent.parent.parent
        return here.parent
    except NameError:
        return Path.cwd().resolve()


def _kaggle_dataset_roots() -> List[Path]:
    """
    Return top-level attached Kaggle Dataset roots, excluding the competition
    mount. Kaggle already exposes Dataset contents as ordinary files/directories;
    no archive extraction is performed.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        return []

    output: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "competitions":
            continue
        output.append(child)
    return output


def _bounded_candidates(root: Path, relative_patterns: Sequence[str]) -> List[Path]:
    """
    Search only a few known nesting depths. Avoid recursive scans through the
    very large competition DICOM tree.
    """
    found: List[Path] = []
    for pattern in relative_patterns:
        try:
            found.extend(root.glob(pattern))
        except Exception:
            pass
    return [path for path in found if path.exists()]


def discover_w40_root(project_root: Path) -> Optional[Path]:
    """
    Locate an already-extracted W40 production artifact directory by its exact
    required result files. This works with Kaggle Dataset mounts regardless of
    the Dataset slug/top-level folder name.
    """
    required = (
        "16_final_probabilities_wide.csv",
        "17_teacher_weights_wide.csv",
        "18_teacher_mask_wide.csv",
        "19_production_summary.json",
        "20_validation_summary.json",
    )

    local = project_root / "output" / "results" / "rsna_w40_fs2_production_teacher_v1"
    candidates: List[Path] = [local]

    if Path("/kaggle/input").exists():
        for dataset_root in _kaggle_dataset_roots():
            candidates.extend(
                _bounded_candidates(
                    dataset_root,
                    (
                        ".",
                        "rsna_w40_fs2_production_teacher_v1",
                        "*/rsna_w40_fs2_production_teacher_v1",
                        "*/*/rsna_w40_fs2_production_teacher_v1",
                    ),
                )
            )

            # Also derive the W40 root from a uniquely named validation file.
            for validation_path in _bounded_candidates(
                dataset_root,
                (
                    "results/20_validation_summary.json",
                    "*/results/20_validation_summary.json",
                    "*/*/results/20_validation_summary.json",
                    "*/*/*/results/20_validation_summary.json",
                ),
            ):
                candidates.append(validation_path.parent.parent)

    dedup: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(resolved)

    valid: List[Path] = []
    for candidate in dedup:
        results = candidate / "results"
        if all((results / name).is_file() for name in required):
            valid.append(candidate)

    if not valid:
        return None

    # Prefer canonical folder name, then shortest path.
    valid.sort(
        key=lambda p: (
            0 if p.name == "rsna_w40_fs2_production_teacher_v1" else 1,
            len(str(p)),
            str(p),
        )
    )
    return valid[0]


def discover_fold_csv(project_root: Path) -> Optional[Path]:
    """
    Locate the exact locked 58-study fold CSV. Identity is still enforced later
    by the locked SHA256 guard, so discovery cannot silently substitute another
    split.
    """
    local = (
        project_root
        / "output"
        / "results"
        / "rsna_w2_3"
        / "results"
        / "00_outer_fold_assignments.csv"
    )
    candidates: List[Path] = [local]

    if Path("/kaggle/input").exists():
        for dataset_root in _kaggle_dataset_roots():
            candidates.extend(
                _bounded_candidates(
                    dataset_root,
                    (
                        "00_outer_fold_assignments.csv",
                        "results/00_outer_fold_assignments.csv",
                        "*/00_outer_fold_assignments.csv",
                        "*/results/00_outer_fold_assignments.csv",
                        "*/*/00_outer_fold_assignments.csv",
                        "*/*/results/00_outer_fold_assignments.csv",
                        "*/*/*/results/00_outer_fold_assignments.csv",
                    ),
                )
            )

    files = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda p: (len(str(p)), str(p)),
    )
    return files[0] if files else None


@dataclass
class ProjectPaths:
    project_root: Path
    train_csv: Path
    train_series_csv: Path
    train_series_root: Path
    w40_root: Path
    fold_csv: Path
    cache_root: Path
    output_root: Path
    checkpoint_root: Path
    result_root: Path
    radimagenet_weights: Optional[Path]

    @classmethod
    def discover(cls, args) -> "ProjectPaths":
        is_kaggle = Path("/kaggle/input").exists()

        if args.project_root:
            project_root = Path(args.project_root).expanduser().resolve()
        elif is_kaggle:
            project_root = Path("/kaggle/working")
        else:
            project_root = _script_project_root()

        if args.data_root:
            data_root = Path(args.data_root).expanduser().resolve()
        elif is_kaggle:
            data_root = Path(
                "/kaggle/input/competitions/rsna-knee-abnormality-detection"
            )
        else:
            data_root = project_root / "input"

        train_csv = data_root / "train.csv"
        train_series_csv = data_root / "train_series.csv"
        train_series_root = data_root / "train_series"

        if args.w40_root:
            w40_root = Path(args.w40_root).expanduser().resolve()
        else:
            discovered_w40 = discover_w40_root(project_root)
            w40_root = (
                discovered_w40
                if discovered_w40 is not None
                else (
                    project_root
                    / "output"
                    / "results"
                    / "rsna_w40_fs2_production_teacher_v1"
                )
            )

        if args.fold_csv:
            fold_csv = Path(args.fold_csv).expanduser().resolve()
        else:
            discovered_fold = discover_fold_csv(project_root)
            fold_csv = (
                discovered_fold
                if discovered_fold is not None
                else (
                    project_root
                    / "output"
                    / "results"
                    / "rsna_w2_3"
                    / "results"
                    / "00_outer_fold_assignments.csv"
                )
            )

        if args.cache_root:
            cache_root = Path(args.cache_root).expanduser().resolve()
        else:
            cache_root = (
                Path(
                    "/kaggle/input/datasets/isayem/w43-rsna-radimagenet-densenet121-2p5d-v4-cache/w43_rsna_radimagenet_densenet121_2p5d_v4_cache"
                )
                / "output"
                / "results"
                / "rsna_2p5d_mri_cache_v1"
            )

        if args.output_root:
            output_root = Path(args.output_root).expanduser().resolve()
        else:
            output_root = project_root / "output" / "results" / OUTPUT_DIR_NAME

        checkpoint_root = output_root / "checkpoints"
        result_root = output_root / "results"

        if args.radimagenet_weights:
            rad_weights = Path(args.radimagenet_weights).expanduser().resolve()
        else:
            rad_weights = discover_radimagenet_weights(project_root)

        result = cls(
            project_root=project_root,
            train_csv=train_csv,
            train_series_csv=train_series_csv,
            train_series_root=train_series_root,
            w40_root=w40_root,
            fold_csv=fold_csv,
            cache_root=cache_root,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            result_root=result_root,
            radimagenet_weights=rad_weights,
        )

        result.configure_globals()
        return result

    def configure_globals(self) -> None:
        global TRAIN_SERIES_ROOT
        TRAIN_SERIES_ROOT = self.train_series_root

    def ensure_output_dirs(self) -> None:
        (self.cache_root / "studies").mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)


def discover_radimagenet_weights(project_root: Path) -> Optional[Path]:
    """
    Search only ordinary mounted/extracted files. No archive extraction occurs.
    """
    candidates: List[Path] = []

    local_roots = [
        project_root / "models" / "radimagenet",
        project_root / "models" / "RadImageNet",
        project_root / "models",
    ]

    if Path("/kaggle/input").exists():
        local_roots.extend(
            [path for path in Path("/kaggle/input").iterdir() if path.is_dir()]
        )

    patterns = [
        "DenseNet121.pt",
        "DenseNet121.pth",
        "*DenseNet*121*.pt",
        "*DenseNet*121*.pth",
        "*densenet*121*.pt",
        "*densenet*121*.pth",
    ]

    for root in local_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                candidates.extend(root.glob(pattern))
                candidates.extend(root.glob(f"*/{pattern}"))
                candidates.extend(root.glob(f"*/*/{pattern}"))
            except Exception:
                pass

    candidates = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda p: (len(str(p)), str(p)),
    )

    return candidates[0] if candidates else None


# =============================================================================
# DATA TABLES / W40 / FOLDS
# =============================================================================


def load_train_tables(
    paths: ProjectPaths,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for required in (
        paths.train_csv,
        paths.train_series_csv,
        paths.train_series_root,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    train = pd.read_csv(paths.train_csv)
    series = pd.read_csv(paths.train_series_csv)

    train[UID_COLUMN] = train[UID_COLUMN].astype(str)
    series[UID_COLUMN] = series[UID_COLUMN].astype(str)
    series[SERIES_UID_COLUMN] = series[SERIES_UID_COLUMN].astype(str)

    required_train = {UID_COLUMN, REPORT_COLUMN, *LABEL_COLUMNS}
    missing_train = required_train - set(train.columns)
    if missing_train:
        raise RuntimeError(f"train.csv missing columns: {sorted(missing_train)}")

    required_series = {
        UID_COLUMN,
        SERIES_UID_COLUMN,
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }
    missing_series = required_series - set(series.columns)
    if missing_series:
        raise RuntimeError(
            f"train_series.csv missing columns: {sorted(missing_series)}"
        )

    series["_fluid"] = coerce_bool_series(series["Fluid_Sensitive"]).astype(int)
    series["_fs"] = coerce_bool_series(series["Fat_Suppression"]).astype(int)

    valid_planes = {"Axial", "Coronal", "Sagittal"}
    bad_planes = (
        set(series["Anatomical_Plane"].dropna().astype(str).unique()) - valid_planes
    )
    if bad_planes:
        raise RuntimeError(f"Unexpected Anatomical_Plane values: {sorted(bad_planes)}")

    is_gold = train[LABEL_COLUMNS].notna().all(axis=1)
    is_unlabeled = train[LABEL_COLUMNS].isna().all(axis=1)
    partial = ~(is_gold | is_unlabeled)
    if partial.any():
        raise RuntimeError(f"Found {int(partial.sum())} partially labeled rows.")

    gold = train[is_gold].copy().sort_values(UID_COLUMN).reset_index(drop=True)
    unlabeled = (
        train[is_unlabeled].copy().sort_values(UID_COLUMN).reset_index(drop=True)
    )

    observed = (len(train), len(gold), len(unlabeled))
    expected = (EXPECTED_TRAIN, EXPECTED_GOLD, EXPECTED_UNLABELED)
    if observed != expected:
        raise RuntimeError(f"Unexpected train split {observed}; expected={expected}")

    return train, series, gold, unlabeled


def load_locked_folds(paths: ProjectPaths, gold: pd.DataFrame) -> pd.DataFrame:
    if not paths.fold_csv.exists():
        raise FileNotFoundError(paths.fold_csv)

    folds = pd.read_csv(paths.fold_csv)
    folds[UID_COLUMN] = folds[UID_COLUMN].astype(str)
    folds = folds[[UID_COLUMN, "OuterFold"]].copy()
    folds["OuterFold"] = folds["OuterFold"].astype(int)

    if set(folds[UID_COLUMN]) != set(gold[UID_COLUMN]):
        raise RuntimeError("Locked fold UID set does not match the 58 gold studies.")

    digest = fold_assignment_sha256(folds)
    if digest != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            f"Locked fold SHA mismatch: {digest} != {EXPECTED_FOLD_SHA256}"
        )

    counts = folds["OuterFold"].value_counts().sort_index().to_dict()
    if counts != {1: 11, 2: 12, 3: 11, 4: 11, 5: 13}:
        raise RuntimeError(f"Unexpected fold counts: {counts}")

    return folds.sort_values(UID_COLUMN).reset_index(drop=True)


def _w40_result(paths: ProjectPaths, name: str) -> Path:
    return paths.w40_root / "results" / name


def load_w40_teacher(
    paths: ProjectPaths,
    unlabeled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    prob_path = _w40_result(paths, "16_final_probabilities_wide.csv")
    weight_path = _w40_result(paths, "17_teacher_weights_wide.csv")
    mask_path = _w40_result(paths, "18_teacher_mask_wide.csv")
    validation_path = _w40_result(paths, "20_validation_summary.json")
    summary_path = _w40_result(paths, "19_production_summary.json")

    for path in (prob_path, weight_path, mask_path, validation_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("overall_pass") is not True:
        raise RuntimeError("W40 validation_summary.json is not overall_pass=true.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    changed = summary.get("controlled_changes", {}).get("probabilities_changed")
    if list(changed or []) != W40_CHANGED_LABELS:
        raise RuntimeError(f"Unexpected W40 changed labels: {changed}")

    probs = pd.read_csv(prob_path)
    weights = pd.read_csv(weight_path)
    masks = pd.read_csv(mask_path)

    for frame, name in (
        (probs, "probabilities"),
        (weights, "weights"),
        (masks, "masks"),
    ):
        if UID_COLUMN not in frame.columns:
            raise RuntimeError(f"W40 {name} missing {UID_COLUMN}")
        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
        missing = set(LABEL_COLUMNS) - set(frame.columns)
        if missing:
            raise RuntimeError(f"W40 {name} missing labels: {sorted(missing)}")
        if len(frame) != EXPECTED_UNLABELED:
            raise RuntimeError(
                f"W40 {name} rows={len(frame)}, expected={EXPECTED_UNLABELED}"
            )
        if frame[UID_COLUMN].duplicated().any():
            raise RuntimeError(f"W40 {name} has duplicate UIDs.")
        if set(frame[UID_COLUMN]) != set(unlabeled[UID_COLUMN].astype(str)):
            raise RuntimeError(f"W40 {name} UID set mismatch.")

    p = probs[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    w = weights[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    m = masks[LABEL_COLUMNS].apply(coerce_bool_series).to_numpy(dtype=bool)

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("W40 probabilities are invalid.")
    if not np.isfinite(w).all() or ((w < 0) | (w > 1)).any():
        raise RuntimeError("W40 weights are invalid.")
    if int(m.sum()) != EXPECTED_SELECTED_CELLS:
        raise RuntimeError(
            f"W40 selected cells={int(m.sum())}, expected={EXPECTED_SELECTED_CELLS}"
        )

    return probs, weights, masks, summary


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


def _transfer_syntax_name(uid_value: Any) -> str:
    if uid_value is None:
        return "UNKNOWN"
    try:
        return str(uid_value.name)
    except Exception:
        return str(uid_value)


def _first_dicom_in_series(
    paths: ProjectPaths,
    study_uid: str,
    series_uid: str,
) -> Optional[Path]:
    series_dir = paths.train_series_root / str(study_uid) / str(series_uid)
    candidates = sorted(series_dir.glob("*.dcm"))
    return candidates[0] if candidates else None


def inspect_dicom_decoder_support(
    paths: ProjectPaths,
    series: pd.DataFrame,
    max_series: int = DICOM_PREFLIGHT_SERIES,
) -> Dict[str, Any]:
    """
    Inspect and actually decode representative training DICOMs.

    This is intentionally stronger than importing pydicom. The competition
    contains mixed transfer syntaxes and compressed syntaxes require external
    decoder plugins.
    """
    if pydicom is None:
        return {
            "overall_pass": False,
            "error": "pydicom is not installed.",
            "install_command": DICOM_DECODER_INSTALL_COMMAND,
        }

    universe = (
        series[[UID_COLUMN, SERIES_UID_COLUMN]]
        .drop_duplicates()
        .sort_values([UID_COLUMN, SERIES_UID_COLUMN])
        .reset_index(drop=True)
    )

    if len(universe) <= int(max_series):
        sample = universe.copy()
    else:
        # Deterministically spread probes over the entire 24k-series index so
        # the preflight is not dominated by a few lexicographically early studies.
        indices = np.linspace(
            0,
            len(universe) - 1,
            num=int(max_series),
            dtype=np.int64,
        )
        sample = universe.iloc[np.unique(indices)].reset_index(drop=True)

    syntax_counts: Dict[str, int] = {}
    syntax_names: Dict[str, str] = {}
    representative: Dict[str, str] = {}
    missing_series_dirs: List[Dict[str, str]] = []
    header_failures: List[Dict[str, str]] = []

    for row in sample.itertuples(index=False):
        study_uid = str(getattr(row, UID_COLUMN))
        series_uid = str(getattr(row, SERIES_UID_COLUMN))
        dicom_path = _first_dicom_in_series(paths, study_uid, series_uid)

        if dicom_path is None:
            missing_series_dirs.append(
                {
                    UID_COLUMN: study_uid,
                    SERIES_UID_COLUMN: series_uid,
                }
            )
            continue

        try:
            ds = pydicom.dcmread(
                str(dicom_path),
                stop_before_pixels=True,
                force=True,
            )
            ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
            ts_key = str(ts) if ts is not None else "UNKNOWN"
            syntax_counts[ts_key] = syntax_counts.get(ts_key, 0) + 1
            syntax_names[ts_key] = _transfer_syntax_name(ts)
            representative.setdefault(ts_key, str(dicom_path))
        except Exception as exc:
            header_failures.append(
                {
                    "path": str(dicom_path),
                    "error": repr(exc),
                }
            )

    decode_results: Dict[str, Dict[str, Any]] = {}
    for ts_key, dicom_path in representative.items():
        try:
            ds = pydicom.dcmread(
                dicom_path,
                force=True,
            )
            array = decode_dicom_pixel_array(ds, dicom_path)
            array = np.asarray(array)
            decode_results[ts_key] = {
                "transfer_syntax_uid": ts_key,
                "transfer_syntax_name": syntax_names.get(ts_key),
                "path": dicom_path,
                "decoded": True,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "error": None,
            }
        except Exception as exc:
            decode_results[ts_key] = {
                "transfer_syntax_uid": ts_key,
                "transfer_syntax_name": syntax_names.get(ts_key),
                "path": dicom_path,
                "decoded": False,
                "shape": None,
                "dtype": None,
                "error": repr(exc),
            }

    failed_syntaxes = [
        value for value in decode_results.values() if value.get("decoded") is not True
    ]

    overall = bool(
        len(sample) > 0
        and not header_failures
        and not missing_series_dirs
        and bool(decode_results)
        and not failed_syntaxes
    )

    return {
        "sampled_series": int(len(sample)),
        "series_with_representative_dicom": int(sum(syntax_counts.values())),
        "transfer_syntax_counts": syntax_counts,
        "transfer_syntax_names": syntax_names,
        "decode_results": decode_results,
        "failed_transfer_syntax_count": int(len(failed_syntaxes)),
        "failed_transfer_syntaxes": failed_syntaxes,
        "missing_series_directory_count": int(len(missing_series_dirs)),
        "missing_series_directory_examples": missing_series_dirs[:10],
        "header_failure_count": int(len(header_failures)),
        "header_failure_examples": header_failures[:10],
        "install_command": DICOM_DECODER_INSTALL_COMMAND,
        "overall_pass": overall,
    }


def require_dicom_decoder_preflight(
    paths: ProjectPaths,
    series: pd.DataFrame,
) -> Dict[str, Any]:
    audit = inspect_dicom_decoder_support(paths, series)

    if not audit.get("overall_pass"):
        missing_count = int(audit.get("missing_series_directory_count", 0))
        if missing_count > 0:
            examples = audit.get("missing_series_directory_examples") or []
            raise RuntimeError(
                "DICOM data-completeness preflight FAILED.\n"
                f"{missing_count} of the sampled train_series.csv rows have no matching "
                "series directory under the current train_series root.\n"
                f"Current root: {paths.train_series_root}\n"
                f"Examples: {json.dumps(examples[:5], indent=2)}\n"
                "Run W43 where the FULL competition train_series tree is mounted. "
                "Kaggle is the recommended environment. Existing valid cache files "
                "are reusable; do NOT use --reset-cache."
            )

        failed = audit.get("failed_transfer_syntaxes") or []
        failed_text = "\n".join(
            f"  - {item.get('transfer_syntax_name')} "
            f"({item.get('transfer_syntax_uid')}): {item.get('error')}"
            for item in failed
        )
        raise RuntimeError(
            "DICOM pixel-decoder preflight FAILED.\n"
            "The full DICOM tree is present for sampled rows, but at least one "
            "transfer syntax cannot be decoded.\n"
            + (f"Failed syntaxes:\n{failed_text}\n" if failed_text else "")
            + f"Recommended decoder install:\n  {DICOM_DECODER_INSTALL_COMMAND}\n"
            "After installing codecs, rerun `status` and then `cache`."
        )

    return audit


# =============================================================================
# 2.5D CACHE BUILD
# =============================================================================


def study_cache_path(paths: ProjectPaths, uid: str) -> Path:
    return paths.cache_root / "studies" / f"{stable_uid_hash(uid)}.npz"


def _resize_uint8(image: np.ndarray, size: int = CACHE_IMAGE_SIZE) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if max(image.shape) >= size else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interpolation)

    # Dependency-light fallback through torch.
    tensor = torch.from_numpy(image.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.clamp(0, 255).round().byte().numpy()


def _robust_triplet_to_uint8(images: Sequence[np.ndarray]) -> np.ndarray:
    if len(images) != 3:
        raise ValueError("2.5D triplet must contain exactly three slices.")

    stack = np.stack([np.asarray(image, dtype=np.float32) for image in images], axis=0)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        raise RuntimeError("MRI triplet contains no finite pixels.")

    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))

    if high <= low:
        scaled = np.zeros_like(stack, dtype=np.uint8)
    else:
        scaled = np.clip((stack - low) / (high - low), 0.0, 1.0)
        scaled = np.round(scaled * 255.0).astype(np.uint8)

    resized = np.stack([_resize_uint8(channel) for channel in scaled], axis=0)
    return resized


def _decode_record(
    records: List[Dict[str, Any]],
    target_index: int,
    orientation_spec: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, Any, int]:
    if pydicom is None:
        raise RuntimeError("pydicom is required for cache construction.")

    n = len(records)
    candidates = [target_index]
    for radius in range(1, min(8, n)):
        candidates.extend([target_index - radius, target_index + radius])

    last_error: Optional[Exception] = None

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
                    plane, confidence = geometry_plane_from_iop(iop)
                    if confidence >= GEOMETRY_PLANE_CONFIDENCE:
                        spec = orientation_transform_spec(iop, plane)

            if spec is not None:
                image = apply_orientation_transform(image, spec)

            return image, ds, index

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"No decodable slice near target index {target_index}. Last error={last_error}"
    )


def _series_centers(n_slices: int) -> List[int]:
    if n_slices <= 0:
        return []
    if n_slices == 1:
        return [0]

    quantiles = np.linspace(0.10, 0.90, STACKS_PER_SERIES)
    indices = [int(round(q * (n_slices - 1))) for q in quantiles]

    unique: List[int] = []
    for index in indices:
        index = int(np.clip(index, 0, n_slices - 1))
        if index not in unique:
            unique.append(index)

    # Very short series can yield fewer than five unique centers; that is fine.
    return unique


def build_study_2p5d_cache(
    paths: ProjectPaths,
    study_uid: str,
    study_series: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    study_series = study_series.copy()
    study_series["_plane_idx"] = study_series["Anatomical_Plane"].map(PLANE_TO_INDEX)
    study_series = study_series.sort_values(
        ["_plane_idx", "_fluid", "_fs", SERIES_UID_COLUMN],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    images: List[np.ndarray] = []
    plane_ids: List[int] = []
    fluid_ids: List[int] = []
    fs_ids: List[int] = []
    positions: List[float] = []
    series_ordinals: List[int] = []
    audit_rows: List[Dict[str, Any]] = []

    encoded_series = 0

    for series_ordinal, (_, row) in enumerate(study_series.iterrows()):
        series_uid = str(row[SERIES_UID_COLUMN])
        metadata_plane = str(row["Anatomical_Plane"])
        fluid = int(row["_fluid"])
        fat_suppression = int(row["_fs"])

        records = read_series_headers(study_uid, series_uid)
        if not records:
            audit_rows.append(
                {
                    UID_COLUMN: study_uid,
                    SERIES_UID_COLUMN: series_uid,
                    "Status": "NO_DICOMS",
                    "MetadataPlane": metadata_plane,
                }
            )
            continue

        valid_iop = next(
            (item["iop"] for item in records if item["iop"] is not None), None
        )
        geometry_plane = metadata_plane
        geometry_confidence = float("nan")
        plane_used = metadata_plane
        orientation_spec = None

        if valid_iop is not None:
            geometry_plane, geometry_confidence = geometry_plane_from_iop(valid_iop)
            if (
                geometry_plane != metadata_plane
                and geometry_confidence >= GEOMETRY_PLANE_CONFIDENCE
            ):
                plane_used = geometry_plane
            orientation_spec = orientation_transform_spec(valid_iop, plane_used)

        centers = _series_centers(len(records))
        emitted = 0
        failures = 0

        for center in centers:
            triplet: List[np.ndarray] = []
            actual_center = center

            try:
                for offset in STACK_OFFSETS:
                    target = int(np.clip(center + offset, 0, len(records) - 1))
                    image, _ds, actual = _decode_record(
                        records, target, orientation_spec
                    )
                    triplet.append(image)
                    if offset == 0:
                        actual_center = actual

                stack = _robust_triplet_to_uint8(triplet)
                images.append(stack)
                plane_ids.append(
                    int(PLANE_TO_INDEX.get(plane_used, PLANE_TO_INDEX[metadata_plane]))
                )
                fluid_ids.append(fluid)
                fs_ids.append(fat_suppression)
                positions.append(
                    float(normalized_slice_position(records, actual_center))
                )
                series_ordinals.append(int(encoded_series))
                emitted += 1

            except Exception:
                failures += 1

        if emitted > 0:
            encoded_series += 1

        audit_rows.append(
            {
                UID_COLUMN: study_uid,
                SERIES_UID_COLUMN: series_uid,
                "Status": "OK" if emitted > 0 else "NO_STACKS",
                "MetadataPlane": metadata_plane,
                "GeometryPlane": geometry_plane,
                "GeometryConfidence": geometry_confidence,
                "PlaneUsed": plane_used,
                "FluidSensitive": fluid,
                "FatSuppression": fat_suppression,
                "OriginalSliceCount": len(records),
                "RequestedStacks": len(centers),
                "EncodedStacks": emitted,
                "StackFailures": failures,
            }
        )

    if not images:
        raise RuntimeError(f"Study {study_uid} produced no 2.5D stacks.")

    payload = {
        "cache_version": np.asarray(CACHE_VERSION),
        "study_uid": np.asarray(study_uid),
        "images": np.stack(images, axis=0).astype(np.uint8),
        "plane": np.asarray(plane_ids, dtype=np.int8),
        "fluid": np.asarray(fluid_ids, dtype=np.int8),
        "fat_suppression": np.asarray(fs_ids, dtype=np.int8),
        "slice_position": np.asarray(positions, dtype=np.float32),
        "series_ordinal": np.asarray(series_ordinals, dtype=np.int16),
        "image_size": np.asarray(CACHE_IMAGE_SIZE, dtype=np.int16),
        "stacks_per_series": np.asarray(STACKS_PER_SERIES, dtype=np.int16),
    }

    return payload, audit_rows


def cache_file_is_usable(path: Path, uid: str) -> bool:
    if not path.is_file():
        return False

    try:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["cache_version"].item()) != CACHE_VERSION:
                return False
            if str(payload["study_uid"].item()) != str(uid):
                return False
            images = payload["images"]
            n = images.shape[0]
            if images.ndim != 4 or images.shape[1:] != (
                3,
                CACHE_IMAGE_SIZE,
                CACHE_IMAGE_SIZE,
            ):
                return False
            if images.dtype != np.uint8 or n <= 0:
                return False
            for key in (
                "plane",
                "fluid",
                "fat_suppression",
                "slice_position",
                "series_ordinal",
            ):
                if len(payload[key]) != n:
                    return False
            if not np.isfinite(payload["slice_position"]).all():
                return False
        return True
    except Exception:
        return False


def summarize_cache(paths: ProjectPaths, train: pd.DataFrame) -> Dict[str, Any]:
    usable = 0
    missing: List[str] = []
    invalid: List[str] = []
    total_tokens = 0
    i = 0
    for uid in train[UID_COLUMN].astype(str):
        path = study_cache_path(paths, uid)
        i += 1
        print(f"\rProcessing: {i}/{train.shape[0]}", end="")
        if not path.exists():
            missing.append(uid)
            continue
        if not cache_file_is_usable(path, uid):
            invalid.append(uid)
            continue
        usable += 1
        try:
            with np.load(path, allow_pickle=False) as payload:
                total_tokens += int(payload["images"].shape[0])
        except Exception:
            pass

    return {
        "cache_version": CACHE_VERSION,
        "total_studies": int(len(train)),
        "usable_studies": int(usable),
        "missing_studies": int(len(missing)),
        "invalid_studies": int(len(invalid)),
        "missing_examples": missing[:10],
        "invalid_examples": invalid[:10],
        "total_2p5d_stacks": int(total_tokens),
        "mean_stacks_per_study": float(total_tokens / max(usable, 1)),
        "cache_root": str(paths.cache_root),
        "shared_with_w44": True,
    }


def run_cache(paths: ProjectPaths, args) -> Dict[str, Any]:
    if pydicom is None:
        raise RuntimeError(
            "pydicom is required. Install pydicom before running cache mode."
        )

    paths.ensure_output_dirs()
    train, series, _gold, _unlabeled = load_train_tables(paths)

    dicom_decoder_audit = require_dicom_decoder_preflight(paths, series)
    write_json(
        paths.result_root / "00_dicom_decoder_preflight.json",
        dicom_decoder_audit,
    )

    log("DICOM decoder preflight   : PASS")
    for ts_uid, result in dicom_decoder_audit.get("decode_results", {}).items():
        log(
            "  "
            + str(result.get("transfer_syntax_name"))
            + f" [{ts_uid}] -> decoded={result.get('decoded')}"
        )

    grouped = {str(uid): group.copy() for uid, group in series.groupby(UID_COLUMN)}

    missing_series = [
        uid for uid in train[UID_COLUMN].astype(str) if uid not in grouped
    ]
    if missing_series:
        raise RuntimeError(f"Studies with no train_series rows: {missing_series[:10]}")

    todo = [
        uid
        for uid in train[UID_COLUMN].astype(str)
        if args.reset_cache
        or not cache_file_is_usable(study_cache_path(paths, uid), uid)
    ]

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | SHARED 2.5D CACHE")
    log("=" * 100)
    log(f"Studies total            : {len(train)}")
    log(f"Already usable           : {len(train) - len(todo)}")
    log(f"Need cache               : {len(todo)}")
    log(f"Workers                  : {args.cache_workers}")
    log(f"Stacks / series          : {STACKS_PER_SERIES}")
    log(f"Image size               : {CACHE_IMAGE_SIZE}")
    log("Cache is backbone-independent and reusable by W44 Inception-v3.")

    audit_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()

    def worker(uid: str):
        payload, audit = build_study_2p5d_cache(paths, uid, grouped[uid])
        path = study_cache_path(paths, uid)
        atomic_npz_save(path, **payload)
        if not cache_file_is_usable(path, uid):
            raise RuntimeError(f"Cache validation failed after write: {path}")
        return uid, audit

    if todo:
        with ThreadPoolExecutor(
            max_workers=max(1, int(args.cache_workers))
        ) as executor:
            futures = {executor.submit(worker, uid): uid for uid in todo}
            completed = 0
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    _uid, audit = future.result()
                    audit_rows.extend(audit)
                except Exception as exc:
                    failures.append({UID_COLUMN: uid, "Error": repr(exc)})
                    if len(failures) == 1:
                        log("")
                        log("FIRST CACHE FAILURE:")
                        log(f"  Study: {uid}")
                        log(f"  Error: {repr(exc)}")
                        log("")
                completed += 1
                if completed <= 10 or completed % 100 == 0 or completed == len(todo):
                    rate = completed / max(time.time() - started, 1e-9)
                    log(
                        f"  cached {completed:>4}/{len(todo)} "
                        f"fail={len(failures):>3} rate={rate*60:6.1f} studies/min"
                    )

                # Systemic failures should never consume the whole dataset.
                if completed >= 32 and len(failures) / completed >= 0.90:
                    pd.DataFrame(failures).to_csv(
                        paths.result_root / "02_cache_failures.csv",
                        index=False,
                    )
                    raise RuntimeError(
                        "Aborting cache early: >=90% of the first "
                        f"{completed} studies failed. Inspect "
                        f"{paths.result_root / '02_cache_failures.csv'} and "
                        f"{paths.result_root / '00_dicom_decoder_preflight.json'}."
                    )

    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(
            paths.result_root / "01_cache_series_audit.csv",
            index=False,
        )
    if failures:
        pd.DataFrame(failures).to_csv(
            paths.result_root / "02_cache_failures.csv",
            index=False,
        )

    summary = summarize_cache(paths, train)
    summary.update(
        {
            "created_at": now_iso(),
            "script_version": SCRIPT_VERSION,
            "new_failures": int(len(failures)),
            "dicom_decoder_preflight": dicom_decoder_audit,
        }
    )
    write_json(paths.result_root / "03_cache_summary.json", summary)

    log(json.dumps(summary, indent=2))

    if summary["usable_studies"] != EXPECTED_TRAIN:
        raise RuntimeError(
            f"2.5D cache incomplete: {summary['usable_studies']}/{EXPECTED_TRAIN} usable."
        )

    return summary


# =============================================================================
# RADIMAGENET DENSENET-121 IDENTITY / LOAD
# =============================================================================


def require_torchvision():
    try:
        import torchvision
        from torchvision.models import densenet121
    except Exception as exc:
        raise RuntimeError(
            "W43 requires torchvision with densenet121 support."
        ) from exc
    return torchvision, densenet121


class DenseNet121RadImageNetBackbone(nn.Module):
    """
    Mirror the structure used by the official RadImageNet PyTorch example:

        base_model = densenet121()
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])

    For torchvision DenseNet-121, encoder_layers[:-1] contains the `features`
    module. Keeping the Sequential wrapper is intentional because the released
    RadImageNet PyTorch state_dict may use `backbone.0.*` keys.

    Forward adds the standard DenseNet final ReLU + global average pooling and
    returns a [N,1024] feature vector.
    """

    def __init__(self):
        super().__init__()
        _torchvision, densenet121 = require_torchvision()
        base_model = densenet121(weights=None)
        encoder_layers = list(base_model.children())
        if len(encoder_layers) < 2:
            raise RuntimeError("Unexpected torchvision DenseNet-121 child structure.")
        self.backbone = nn.Sequential(*encoder_layers[:-1])
        self.num_features = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = F.relu(features, inplace=False)
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        return torch.flatten(features, 1)


def forward_backbone_fp32(
    backbone: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Run RadImageNet DenseNet-121 in FP32 even inside CUDA autocast."""
    x = images.float()
    if x.device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            return backbone(x)
    return backbone(x)


def create_densenet121_backbone() -> nn.Module:
    model = DenseNet121RadImageNetBackbone()
    if int(model.num_features) != BACKBONE_FEATURE_DIM:
        raise RuntimeError(
            f"Unexpected DenseNet-121 feature dim={model.num_features}; "
            f"expected={BACKBONE_FEATURE_DIM}"
        )
    return model


def _extract_checkpoint_state(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, nn.Module):
        return dict(payload.state_dict())

    if isinstance(payload, Mapping):
        # Direct tensor dictionary.
        if payload and all(
            isinstance(value, torch.Tensor) for value in payload.values()
        ):
            return dict(payload)

        for key in (
            "state_dict",
            "model_state_dict",
            "backbone_state_dict",
            "backbone",
            "model",
        ):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, nn.Module):
                return dict(value.state_dict())
            if (
                isinstance(value, Mapping)
                and value
                and all(isinstance(item, torch.Tensor) for item in value.values())
            ):
                return dict(value)

    raise RuntimeError("Could not identify a PyTorch state_dict inside DenseNet121.pt.")


def _strip_outer_checkpoint_prefixes(key: str) -> str:
    """
    Remove only generic serialization wrappers.

    Do NOT blindly remove `backbone.` because the official RadImageNet example
    itself stores the encoder under self.backbone and thus legitimately emits
    `backbone.0.*` keys.
    """
    result = str(key)
    prefixes = ("module.", "model.", "encoder.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix) :]
                changed = True
    return result


def _densenet_key_candidates(raw_key: str) -> List[str]:
    """
    Support the known/likely RadImageNet DenseNet checkpoint layouts while
    remaining fail-loud on tensor shape or coverage mismatch.

    Canonical W43 model keys:
        backbone.0.conv0.weight
        backbone.0.denseblock1.denselayer1.norm1.weight
        ...

    Accepted source layouts include:
        backbone.0.*
        0.*
        features.*
        backbone.features.*
        backbone.<feature-name>.*
        <feature-name>.*

    Classifier keys are ignored separately and can never satisfy backbone
    coverage.
    """
    key = _strip_outer_checkpoint_prefixes(raw_key)
    candidates = [key]

    feature_roots = (
        "conv0.",
        "norm0.",
        "relu0.",
        "pool0.",
        "denseblock1.",
        "transition1.",
        "denseblock2.",
        "transition2.",
        "denseblock3.",
        "transition3.",
        "denseblock4.",
        "norm5.",
    )

    if key.startswith("features."):
        candidates.append("backbone.0." + key[len("features.") :])

    if key.startswith("backbone.features."):
        candidates.append("backbone.0." + key[len("backbone.features.") :])

    if key.startswith("0."):
        candidates.append("backbone." + key)

    if key.startswith("backbone.") and not key.startswith("backbone.0."):
        suffix = key[len("backbone.") :]
        if suffix.startswith(feature_roots):
            candidates.append("backbone.0." + suffix)

    if key.startswith(feature_roots):
        candidates.append("backbone.0." + key)

    # Older torchvision DenseNet checkpoints occasionally serialized dotted
    # denselayer names (norm.1, conv.2, ...). Convert those to current names.
    legacy = []
    legacy_pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12])\.(weight|bias|running_mean|running_var)$"
    )
    for candidate in list(candidates):
        match = legacy_pattern.match(candidate)
        if match:
            legacy.append(match.group(1) + match.group(2) + "." + match.group(3))

    candidates.extend(legacy)

    # Stable unique order.
    output = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def _map_checkpoint_to_densenet(
    raw_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    mapped: Dict[str, torch.Tensor] = {}
    unmapped_raw: List[str] = []

    for raw_key, tensor in raw_state.items():
        chosen = None
        for candidate in _densenet_key_candidates(raw_key):
            if candidate in model_state:
                chosen = candidate
                break

        if chosen is None:
            unmapped_raw.append(str(raw_key))
            continue

        if chosen in mapped:
            # Ambiguous duplicate mappings are unsafe.
            raise RuntimeError(
                f"Multiple RadImageNet checkpoint tensors map to {chosen!r}."
            )

        mapped[chosen] = tensor

    return mapped, unmapped_raw


def _densenet_bn_running_var_summary(
    mapped_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Surface the known DenseNet RadImageNet BN-running-var concern as audit data.
    We do not mutate these official weights here.
    """
    arrays = []
    extreme_examples = []

    for key, tensor in mapped_state.items():
        if not key.endswith("running_var"):
            continue
        value = tensor.detach().float().cpu()
        if value.numel() == 0:
            continue
        arrays.append(value.reshape(-1))
        vmax = float(value.max())
        if vmax > 1.0e4:
            extreme_examples.append(
                {
                    "key": key,
                    "min": float(value.min()),
                    "max": vmax,
                    "mean": float(value.mean()),
                }
            )

    if not arrays:
        return {
            "count": 0,
            "global_min": None,
            "global_max": None,
            "global_mean": None,
            "extreme_gt_1e4_count": 0,
            "extreme_examples": [],
        }

    joined = torch.cat(arrays)
    return {
        "count": int(joined.numel()),
        "global_min": float(joined.min()),
        "global_max": float(joined.max()),
        "global_mean": float(joined.mean()),
        "extreme_gt_1e4_count": int((joined > 1.0e4).sum().item()),
        "extreme_examples": extreme_examples[:10],
    }


def audit_radimagenet_weights(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    backbone = create_densenet121_backbone()
    model_state = backbone.state_dict()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = _extract_checkpoint_state(payload)
    checkpoint_state, unmapped_raw = _map_checkpoint_to_densenet(
        raw_state,
        model_state,
    )

    ignored_classifier_prefixes = (
        "classif.",
        "classifier.",
        "last_linear.",
        "fc.",
        "head.",
        "module.classif.",
        "module.classifier.",
        "module.fc.",
        "module.head.",
    )

    matched: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[Dict[str, Any]] = []
    missing: List[str] = []

    total_parameter_numel = 0
    matched_parameter_numel = 0
    parameter_keys = {name for name, _ in backbone.named_parameters()}

    for key, model_tensor in model_state.items():
        if key in parameter_keys:
            total_parameter_numel += int(model_tensor.numel())

        checkpoint_tensor = checkpoint_state.get(key)

        if checkpoint_tensor is None:
            if key.endswith("num_batches_tracked"):
                continue
            missing.append(key)
            continue

        if tuple(checkpoint_tensor.shape) != tuple(model_tensor.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint_tensor.shape),
                    "model_shape": list(model_tensor.shape),
                }
            )
            continue

        matched[key] = checkpoint_tensor
        if key in parameter_keys:
            matched_parameter_numel += int(model_tensor.numel())

    unexpected = []
    for key in unmapped_raw:
        normalized = _strip_outer_checkpoint_prefixes(key)
        if not normalized.startswith(ignored_classifier_prefixes):
            unexpected.append(key)

    coverage = matched_parameter_numel / max(total_parameter_numel, 1)
    disallowed_missing = [
        key for key in missing if not key.endswith("num_batches_tracked")
    ]

    passed = bool(
        coverage >= RADIMAGENET_MIN_NUMEL_COVERAGE
        and not shape_mismatch
        and not disallowed_missing
    )

    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "backbone": BACKBONE_NAME,
        "checkpoint_format_target": "official RadImageNet PyTorch DenseNet121",
        "model_state_keys": int(len(model_state)),
        "raw_checkpoint_state_keys": int(len(raw_state)),
        "mapped_checkpoint_state_keys": int(len(checkpoint_state)),
        "matched_keys": int(len(matched)),
        "parameter_numel_coverage": float(coverage),
        "min_required_numel_coverage": RADIMAGENET_MIN_NUMEL_COVERAGE,
        "missing_non_bn_tracking_count": int(len(disallowed_missing)),
        "missing_examples": disallowed_missing[:20],
        "shape_mismatch_count": int(len(shape_mismatch)),
        "shape_mismatch_examples": shape_mismatch[:10],
        "unexpected_non_classifier_count": int(len(unexpected)),
        "unexpected_examples": unexpected[:20],
        "checkpoint_first_keys": list(raw_state.keys())[:20],
        "mapped_first_keys": list(checkpoint_state.keys())[:20],
        "batchnorm_running_var": _densenet_bn_running_var_summary(checkpoint_state),
        "preprocessing": RADIMAGENET_PREPROCESSING,
        "overall_pass": passed,
    }

    del backbone, payload, raw_state, checkpoint_state
    gc.collect()
    return audit


def load_radimagenet_backbone(path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
    audit = audit_radimagenet_weights(path)
    if not audit["overall_pass"]:
        raise RuntimeError(
            "RadImageNet DenseNet-121 checkpoint audit FAILED. "
            "Refusing to train with partial/random initialization.\n"
            + json.dumps(audit, indent=2)
        )

    backbone = create_densenet121_backbone()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = _extract_checkpoint_state(payload)
    checkpoint_state, _unmapped = _map_checkpoint_to_densenet(
        raw_state,
        backbone.state_dict(),
    )

    result = backbone.load_state_dict(checkpoint_state, strict=False)
    disallowed_missing = [
        key for key in result.missing_keys if not key.endswith("num_batches_tracked")
    ]

    if disallowed_missing:
        raise RuntimeError(
            f"RadImageNet DenseNet-121 load left unmatched tensors: "
            f"{disallowed_missing[:20]}"
        )

    return backbone, audit


# =============================================================================
# STUDY MODEL
# =============================================================================


class RadImageNetDenseNet121StudyModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(BACKBONE_FEATURE_DIM),
            nn.Linear(BACKBONE_FEATURE_DIM, TOKEN_DIM),
            nn.GELU(),
        )

        self.plane_embedding = nn.Embedding(3, 32)
        self.fluid_embedding = nn.Embedding(2, 8)
        self.fs_embedding = nn.Embedding(2, 8)
        self.position_projection = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
        )

        self.metadata_projection = nn.Sequential(
            nn.Linear(32 + 8 + 8 + 16, TOKEN_DIM),
            nn.GELU(),
            nn.LayerNorm(TOKEN_DIM),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TOKEN_DIM,
            nhead=TOKEN_HEADS,
            dim_feedforward=TOKEN_DIM * 4,
            dropout=TOKEN_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.study_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TOKEN_LAYERS,
        )

        self.label_queries = nn.Parameter(torch.empty(NUM_LABELS, TOKEN_DIM))
        nn.init.normal_(self.label_queries, std=0.02)

        # Small fixed anatomical preference. It is deliberately not allowed to
        # exclude any plane. A single learnable scalar controls global strength.
        self.register_buffer("plane_prior_log", PLANE_PRIOR_LOG.clone())
        self.plane_prior_scale = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))

        self.final_norm = nn.LayerNorm(TOKEN_DIM)
        self.dropout = nn.Dropout(0.20)
        self.classifier_weight = nn.Parameter(torch.empty(NUM_LABELS, TOKEN_DIM))
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))
        nn.init.normal_(self.classifier_weight, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        token_mask: torch.Tensor,
        plane: torch.Tensor,
        fluid: torch.Tensor,
        fat_suppression: torch.Tensor,
        slice_position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, max_tokens = images.shape[:2]

        flat_images = images.reshape(
            batch_size * max_tokens, 3, images.shape[-2], images.shape[-1]
        )
        flat_mask = token_mask.reshape(batch_size * max_tokens)

        valid_images = flat_images[flat_mask]
        if valid_images.numel() == 0:
            raise RuntimeError("Study batch has no valid MRI tokens.")

        valid_features = forward_backbone_fp32(self.backbone, valid_images)
        if valid_features.ndim != 2 or valid_features.shape[-1] != BACKBONE_FEATURE_DIM:
            raise RuntimeError(
                f"Unexpected DenseNet-121 feature shape {tuple(valid_features.shape)}"
            )

        projected = self.feature_projection(valid_features)

        token_features = projected.new_zeros(batch_size * max_tokens, TOKEN_DIM)
        token_features[flat_mask] = projected
        token_features = token_features.reshape(batch_size, max_tokens, TOKEN_DIM)

        metadata = torch.cat(
            [
                self.plane_embedding(plane.clamp(0, 2)),
                self.fluid_embedding(fluid.clamp(0, 1)),
                self.fs_embedding(fat_suppression.clamp(0, 1)),
                self.position_projection(slice_position.unsqueeze(-1)),
            ],
            dim=-1,
        )
        token_features = token_features + self.metadata_projection(metadata)
        token_features = token_features.masked_fill(~token_mask.unsqueeze(-1), 0.0)

        hidden = self.study_transformer(
            token_features,
            src_key_padding_mask=~token_mask,
        )

        scores = torch.einsum(
            "bth,lh->btl",
            hidden,
            self.label_queries,
        ) / math.sqrt(TOKEN_DIM)

        # plane_prior_log is [L,3]; gather -> [B,T,L]
        plane_prior = self.plane_prior_log[:, plane].permute(1, 2, 0)
        scores = scores + self.plane_prior_scale.clamp(0.0, 2.0) * plane_prior
        scores = scores.masked_fill(~token_mask.unsqueeze(-1), -1e4)

        attention = torch.softmax(scores, dim=1)
        study_rep = torch.einsum("btl,bth->blh", attention, hidden)
        study_rep = self.final_norm(study_rep)
        study_rep = self.dropout(study_rep)

        logits = (study_rep * self.classifier_weight.unsqueeze(0)).sum(
            dim=-1
        ) + self.classifier_bias.unsqueeze(0)

        return {
            "logits": logits,
            "token_attention": attention,
        }


def unwrap_model(model: nn.Module) -> RadImageNetDenseNet121StudyModel:
    return model.module if isinstance(model, nn.DataParallel) else model


def set_backbone_train_stage(model: nn.Module, stage: str) -> None:
    core = unwrap_model(model)

    for parameter in core.backbone.parameters():
        parameter.requires_grad = False

    if stage == "frozen":
        return

    if stage == "top":
        # DenseNet-121 official-wrapper layout:
        #   core.backbone.backbone[0] == torchvision DenseNet.features
        try:
            features = core.backbone.backbone[0]
        except Exception as exc:
            raise RuntimeError(
                "Could not locate DenseNet-121 features module for progressive unfreezing."
            ) from exc

        found = 0
        for name in ("transition3", "denseblock4", "norm5"):
            module = getattr(features, name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True
            found += 1

        if found != 3:
            raise RuntimeError(
                f"DenseNet-121 top-stage modules incomplete: found={found}/3."
            )
        return

    if stage == "full":
        for parameter in core.backbone.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unknown backbone stage: {stage}")


def set_backbone_batchnorm_eval(model: nn.Module) -> None:
    """
    Freeze backbone BatchNorm RUNNING STATISTICS during optimization.

    Calling eval() on BatchNorm does NOT disable gradients for affine weight/bias.
    Therefore gamma/beta can still learn in unfrozen DenseNet stages while
    running_mean/running_var remain fixed at the knee-domain recalibrated values.
    """
    core = unwrap_model(model)
    for module in core.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _backbone_bn_modules(model: nn.Module) -> List[nn.modules.batchnorm._BatchNorm]:
    core = unwrap_model(model)
    return [
        module
        for module in core.backbone.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def backbone_bn_buffer_summary(model: nn.Module) -> Dict[str, Any]:
    modules = _backbone_bn_modules(model)

    running_vars = []
    running_means = []
    tracked = []

    for module in modules:
        if module.running_var is not None:
            running_vars.append(module.running_var.detach().float().cpu().reshape(-1))
        if module.running_mean is not None:
            running_means.append(module.running_mean.detach().float().cpu().reshape(-1))
        if module.num_batches_tracked is not None:
            tracked.append(int(module.num_batches_tracked.detach().cpu().item()))

    if running_vars:
        var = torch.cat(running_vars)
        var_finite = bool(torch.isfinite(var).all().item())
        var_positive = bool((var > 0).all().item())
        var_summary = {
            "count": int(var.numel()),
            "min": float(var.min()),
            "max": float(var.max()),
            "mean": float(var.mean()),
            "finite": var_finite,
            "strictly_positive": var_positive,
            "gt_1e4_count": int((var > 1.0e4).sum().item()),
        }
    else:
        var_summary = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "finite": False,
            "strictly_positive": False,
            "gt_1e4_count": 0,
        }

    if running_means:
        mean = torch.cat(running_means)
        mean_summary = {
            "count": int(mean.numel()),
            "min": float(mean.min()),
            "max": float(mean.max()),
            "mean": float(mean.mean()),
            "finite": bool(torch.isfinite(mean).all().item()),
        }
    else:
        mean_summary = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "finite": False,
        }

    return {
        "bn_module_count": int(len(modules)),
        "running_var": var_summary,
        "running_mean": mean_summary,
        "num_batches_tracked_min": int(min(tracked)) if tracked else None,
        "num_batches_tracked_max": int(max(tracked)) if tracked else None,
    }


def recalibrate_backbone_batchnorm(
    model: nn.Module,
    loader: DataLoader,
    runtime: Runtime,
    max_stacks: int = BN_CALIBRATION_MAX_STACKS,
) -> Dict[str, Any]:
    """
    Reset DenseNet BN running buffers and estimate new knee-MRI domain statistics.

    Only the backbone is forwarded. No labels, losses, gradients, Transformer
    tokens, or classifier heads participate in calibration.
    """
    core = unwrap_model(model)
    bn_modules = _backbone_bn_modules(model)

    if not bn_modules:
        raise RuntimeError("DenseNet backbone exposes no BatchNorm modules.")

    source_summary = backbone_bn_buffer_summary(model)

    original_momentum = {}
    for module in bn_modules:
        original_momentum[id(module)] = module.momentum
        module.reset_running_stats()
        # Cumulative moving average across equal-size image chunks.
        module.momentum = None

    # Keep every non-BN module deterministic while allowing BN buffers to update.
    core.eval()
    for module in bn_modules:
        module.train()

    processed = 0
    started = time.time()

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"]
            token_mask = batch["token_mask"]
            valid_images = images[token_mask]

            if valid_images.numel() == 0:
                continue

            cursor = 0
            while cursor < len(valid_images) and processed < int(max_stacks):
                remaining = int(max_stacks) - processed
                take = min(
                    BN_CALIBRATION_IMAGE_BATCH,
                    len(valid_images) - cursor,
                    remaining,
                )
                if take <= 0:
                    break

                chunk = valid_images[cursor : cursor + take].to(
                    runtime.device,
                    non_blocking=False,
                )

                _ = forward_backbone_fp32(core.backbone, chunk)

                processed += int(take)
                cursor += int(take)
                del chunk

            if processed >= int(max_stacks):
                break

    for module in bn_modules:
        module.momentum = original_momentum[id(module)]
        module.eval()

    recalibrated_summary = backbone_bn_buffer_summary(model)

    var_info = recalibrated_summary["running_var"]
    mean_info = recalibrated_summary["running_mean"]

    passed = bool(
        processed >= BN_CALIBRATION_MIN_STACKS
        and var_info["count"] > 0
        and var_info["finite"]
        and var_info["strictly_positive"]
        and mean_info["finite"]
        and recalibrated_summary["num_batches_tracked_min"] is not None
        and recalibrated_summary["num_batches_tracked_min"] > 0
    )

    summary = {
        "strategy": BN_STRATEGY,
        "created_at": now_iso(),
        "source_checkpoint_buffers": source_summary,
        "recalibrated_buffers": recalibrated_summary,
        "calibration_stacks": int(processed),
        "minimum_required_stacks": int(BN_CALIBRATION_MIN_STACKS),
        "maximum_requested_stacks": int(max_stacks),
        "image_batch": int(BN_CALIBRATION_IMAGE_BATCH),
        "preprocessing": RADIMAGENET_PREPROCESSING,
        "elapsed_seconds": float(time.time() - started),
        "overall_pass": passed,
    }

    if not passed:
        raise RuntimeError(
            "DenseNet BatchNorm recalibration failed validation.\n"
            + json.dumps(summary, indent=2, allow_nan=True)
        )

    return summary


# =============================================================================
# CACHE DATASET / COLLATION
# =============================================================================


class StudyCacheDataset(Dataset):
    def __init__(
        self,
        paths: ProjectPaths,
        uids: Sequence[str],
        targets: Mapping[str, np.ndarray],
        target_masks: Mapping[str, np.ndarray],
        target_weights: Mapping[str, np.ndarray],
        training: bool,
        max_tokens: int,
        seed: int,
    ):
        self.paths = paths
        self.uids = [str(uid) for uid in uids]
        self.targets = targets
        self.target_masks = target_masks
        self.target_weights = target_weights
        self.training = bool(training)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.uids)

    def _select_indices(self, uid: str, plane: np.ndarray, n: int) -> np.ndarray:
        if n <= self.max_tokens:
            return np.arange(n, dtype=np.int64)

        if not self.training:
            return (
                np.linspace(0, n - 1, min(self.max_tokens, n)).round().astype(np.int64)
            )

        # Worker-safe per-sample randomness that changes across calls while still
        # deriving from the configured seed and UID.
        entropy = random.getrandbits(32)
        uid_seed = int(stable_uid_hash(uid)[:8], 16)
        rng = np.random.default_rng(self.seed ^ uid_seed ^ entropy)

        selected: List[int] = []

        # Guarantee plane diversity when available.
        for plane_id in (0, 1, 2):
            candidates = np.where(plane == plane_id)[0]
            if len(candidates):
                selected.append(int(rng.choice(candidates)))

        selected = list(dict.fromkeys(selected))
        remaining = [index for index in range(n) if index not in selected]
        need = self.max_tokens - len(selected)
        if need > 0:
            chosen = rng.choice(remaining, size=need, replace=False)
            selected.extend(int(index) for index in chosen)

        selected.sort()
        return np.asarray(selected, dtype=np.int64)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        uid = self.uids[index]
        path = study_cache_path(self.paths, uid)
        if not cache_file_is_usable(path, uid):
            raise RuntimeError(f"Unusable 2.5D cache: {path}")

        with np.load(path, allow_pickle=False) as payload:
            images = payload["images"]
            plane = payload["plane"].astype(np.int64)
            fluid = payload["fluid"].astype(np.int64)
            fs = payload["fat_suppression"].astype(np.int64)
            position = payload["slice_position"].astype(np.float32)

        selected = self._select_indices(uid, plane, len(images))

        images = images[selected].astype(np.float32)
        # Exact official RadImageNet PyTorch-demo convention:
        #     (image - 127.5) * 2 / 255
        images = (images - 127.5) * (2.0 / 255.0)

        if self.training:
            # Intensity-only augmentation. Horizontal flips are deliberately
            # forbidden because medial/lateral diagnoses are separate targets.
            gain = np.random.uniform(0.92, 1.08)
            bias = np.random.uniform(-0.04, 0.04)
            images = images * gain + bias
            if np.random.random() < 0.35:
                images = images + np.random.normal(
                    0.0, 0.015, size=images.shape
                ).astype(np.float32)
            images = np.clip(images, -1.25, 1.25)

        return {
            UID_COLUMN: uid,
            "images": torch.from_numpy(images.astype(np.float32)),
            "plane": torch.from_numpy(plane[selected]),
            "fluid": torch.from_numpy(fluid[selected]),
            "fat_suppression": torch.from_numpy(fs[selected]),
            "slice_position": torch.from_numpy(position[selected]),
            "targets": torch.from_numpy(self.targets[uid].astype(np.float32)),
            "target_mask": torch.from_numpy(self.target_masks[uid].astype(bool)),
            "target_weight": torch.from_numpy(
                self.target_weights[uid].astype(np.float32)
            ),
        }


def collate_studies(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_tokens = max(int(item["images"].shape[0]) for item in batch)
    height = int(batch[0]["images"].shape[-2])
    width = int(batch[0]["images"].shape[-1])

    images = torch.zeros(batch_size, max_tokens, 3, height, width, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    plane = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    fluid = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    fs = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    position = torch.zeros(batch_size, max_tokens, dtype=torch.float32)

    targets = torch.stack([item["targets"] for item in batch])
    target_mask = torch.stack([item["target_mask"] for item in batch])
    target_weight = torch.stack([item["target_weight"] for item in batch])

    uids = []
    for batch_index, item in enumerate(batch):
        n = int(item["images"].shape[0])
        images[batch_index, :n] = item["images"]
        token_mask[batch_index, :n] = True
        plane[batch_index, :n] = item["plane"]
        fluid[batch_index, :n] = item["fluid"]
        fs[batch_index, :n] = item["fat_suppression"]
        position[batch_index, :n] = item["slice_position"]
        uids.append(str(item[UID_COLUMN]))

    return {
        UID_COLUMN: uids,
        "images": images,
        "token_mask": token_mask,
        "plane": plane,
        "fluid": fluid,
        "fat_suppression": fs,
        "slice_position": position,
        "targets": targets,
        "target_mask": target_mask,
        "target_weight": target_weight,
    }


# =============================================================================
# TARGET DICTIONARIES
# =============================================================================


def build_gold_target_maps(
    gold: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    targets = {}
    masks = {}
    weights = {}
    for _, row in gold.iterrows():
        uid = str(row[UID_COLUMN])
        targets[uid] = row[LABEL_COLUMNS].to_numpy(dtype=np.float32)
        masks[uid] = np.ones(NUM_LABELS, dtype=bool)
        weights[uid] = np.ones(NUM_LABELS, dtype=np.float32)
    return targets, masks, weights


def build_pseudo_target_maps(
    probs: pd.DataFrame,
    weights: pd.DataFrame,
    masks: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    p = probs.set_index(UID_COLUMN)
    w = weights.set_index(UID_COLUMN)
    m = masks.set_index(UID_COLUMN)

    targets = {}
    target_masks = {}
    target_weights = {}

    for uid in p.index.astype(str):
        targets[uid] = p.loc[uid, LABEL_COLUMNS].to_numpy(dtype=np.float32)
        target_weights[uid] = w.loc[uid, LABEL_COLUMNS].to_numpy(dtype=np.float32)
        target_masks[uid] = np.asarray(
            [
                bool(coerce_bool_series(pd.Series([m.loc[uid, label]])).iloc[0])
                for label in LABEL_COLUMNS
            ],
            dtype=bool,
        )

    return targets, target_masks, target_weights


# =============================================================================
# RUNTIME / MIXED PRECISION
# =============================================================================


@dataclass
class Runtime:
    accelerator_requested: str
    backend: str
    device: torch.device
    visible_cuda_devices: int
    autocast_dtype: Optional[torch.dtype]
    use_data_parallel: bool


def resolve_runtime(accelerator: str) -> Runtime:
    requested = str(accelerator or "auto")
    value = requested.strip().lower().replace("-", "_")
    aliases = {
        "localgpu": "local_gpu",
        "gpu": "local_gpu",
        "cuda": "local_gpu",
        "t4": "kaggle_t4",
        "kagglegpu": "kaggle_t4",
        "mps": "apple_mps",
        "apple": "apple_mps",
        "kaggle_tpu": "tpu",
    }
    value = aliases.get(value, value)

    if value == "auto":
        if torch.cuda.is_available():
            value = "kaggle_t4" if Path("/kaggle/input").exists() else "local_gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            value = "apple_mps"
        else:
            value = "cpu"

    if value == "tpu":
        raise RuntimeError("W43 TPU/XLA training is not implemented or validated.")

    if value in {"local_gpu", "kaggle_t4"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{value} requested but CUDA is unavailable.")
        count = int(torch.cuda.device_count())
        device = torch.device("cuda:0")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return Runtime(requested, "cuda", device, count, dtype, count > 1)

    if value == "apple_mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("apple_mps requested but MPS is unavailable.")
        return Runtime(requested, "mps", torch.device("mps"), 0, None, False)

    if value == "cpu":
        return Runtime(requested, "cpu", torch.device("cpu"), 0, None, False)

    raise ValueError(f"Unknown accelerator: {accelerator}")


def autocast_context(runtime: Runtime):
    if runtime.backend == "cuda" and runtime.autocast_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=runtime.autocast_dtype)
    return contextlib.nullcontext()


def make_grad_scaler(runtime: Runtime):
    enabled = runtime.backend == "cuda" and runtime.autocast_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    result = dict(batch)
    for key in (
        "images",
        "token_mask",
        "plane",
        "fluid",
        "fat_suppression",
        "slice_position",
        "targets",
        "target_mask",
        "target_weight",
    ):
        result[key] = batch[key].to(device, non_blocking=True)
    return result


# =============================================================================
# LOSSES / OPTIMIZER
# =============================================================================


def masked_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    target_weight: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits32 = logits.float()
    targets32 = targets.float()
    mask32 = target_mask.float()
    weights32 = target_weight.float()

    if not torch.isfinite(logits32).all():
        raise RuntimeError("Non-finite logits reached masked_soft_bce.")
    if not torch.isfinite(targets32).all():
        raise RuntimeError("Targets contain NaN/Inf.")
    if not torch.isfinite(weights32).all():
        raise RuntimeError("Target weights contain NaN/Inf.")
    if ((targets32 < 0) | (targets32 > 1)).any():
        raise RuntimeError("Targets outside [0,1].")
    if ((weights32 < 0) | (weights32 > 1)).any():
        raise RuntimeError("Target weights outside [0,1].")

    if pos_weight is not None:
        pos_weight = pos_weight.float()

    per_cell = F.binary_cross_entropy_with_logits(
        logits32,
        targets32,
        reduction="none",
        pos_weight=pos_weight,
    )

    effective = mask32 * weights32
    denominator = effective.sum().clamp_min(1.0)
    result = (per_cell * effective).sum() / denominator

    if not torch.isfinite(result):
        raise RuntimeError("masked_soft_bce produced NaN/Inf.")

    return result


def gold_positive_weight(gold: pd.DataFrame) -> torch.Tensor:
    y = gold[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    values = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 5.0)
    return torch.tensor(values, dtype=torch.float32)


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> torch.optim.Optimizer:
    core = unwrap_model(model)

    backbone_parameters = [
        parameter for parameter in core.backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in core.backbone.parameters()}
    head_parameters = [
        parameter
        for parameter in core.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]

    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": float(backbone_lr)})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": float(head_lr)})

    if not groups:
        raise RuntimeError("No trainable parameters in optimizer.")

    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def stage_backbone_lr(stage: str, gold: bool = False) -> float:
    if gold:
        if stage == "top":
            return GOLD_BACKBONE_LR_TOP
        if stage == "full":
            return GOLD_BACKBONE_LR_FULL
        return 0.0
    return float(PSEUDO_BACKBONE_LR[stage])


# =============================================================================
# MODEL CREATION / CHECKPOINTS
# =============================================================================


def create_new_training_model(
    paths: ProjectPaths, runtime: Runtime
) -> Tuple[nn.Module, Dict[str, Any]]:
    if paths.radimagenet_weights is None:
        raise RuntimeError(
            "RadImageNet DenseNet-121 weights were not discovered. "
            "Pass --radimagenet-weights explicitly."
        )

    backbone, audit = load_radimagenet_backbone(paths.radimagenet_weights)
    model = RadImageNetDenseNet121StudyModel(backbone).to(runtime.device)

    if runtime.use_data_parallel:
        model = nn.DataParallel(
            model, device_ids=list(range(runtime.visible_cuda_devices))
        )

    return model, audit


def create_empty_model(runtime: Runtime) -> nn.Module:
    backbone = create_densenet121_backbone()
    model = RadImageNetDenseNet121StudyModel(backbone).to(runtime.device)
    if runtime.use_data_parallel:
        model = nn.DataParallel(
            model, device_ids=list(range(runtime.visible_cuda_devices))
        )
    return model


def checkpoint_config(_paths=None) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "backbone": BACKBONE_NAME,
        "radimagenet_preprocessing": RADIMAGENET_PREPROCESSING,
        "bn_strategy": BN_STRATEGY,
        "bn_calibration_studies": BN_CALIBRATION_STUDIES,
        "bn_calibration_max_stacks": BN_CALIBRATION_MAX_STACKS,
        "bn_calibration_min_stacks": BN_CALIBRATION_MIN_STACKS,
        "bn_calibration_image_batch": BN_CALIBRATION_IMAGE_BATCH,
        "backbone_force_fp32": BACKBONE_FORCE_FP32,
        "numerical_preflight_batches": NUMERICAL_PREFLIGHT_BATCHES,
        "cache_version": CACHE_VERSION,
        "cache_image_size": CACHE_IMAGE_SIZE,
        "stacks_per_series": STACKS_PER_SERIES,
        "stack_offsets": list(STACK_OFFSETS),
        "token_dim": TOKEN_DIM,
        "token_heads": TOKEN_HEADS,
        "token_layers": TOKEN_LAYERS,
        "train_tokens_per_study": TRAIN_TOKENS_PER_STUDY,
        "w40_changed_probabilities": W40_CHANGED_LABELS,
        "w40_selected_cells": EXPECTED_SELECTED_CELLS,
        "fold_sha256": EXPECTED_FOLD_SHA256,
    }


def save_model_checkpoint(
    model: nn.Module,
    path: Path,
    scope: str,
    seed: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": scope,
        "seed": int(seed),
        "created_at": now_iso(),
        "config": checkpoint_config(None),
        "model_state_dict": unwrap_model(model).state_dict(),
        "extra": dict(extra or {}),
    }
    atomic_torch_save(payload, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "scope": scope,
        "seed": int(seed),
    }


def load_model_checkpoint(
    model: nn.Module,
    path: Path,
    expected_scope: Optional[str] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("script_version") != SCRIPT_VERSION:
        raise RuntimeError(
            f"Checkpoint version mismatch: {payload.get('script_version')} != {SCRIPT_VERSION}"
        )
    if expected_scope is not None and payload.get("scope") != expected_scope:
        raise RuntimeError(
            f"Checkpoint scope mismatch: {payload.get('scope')} != {expected_scope}"
        )

    config = payload.get("config", {})
    expected = checkpoint_config(None)
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"Checkpoint config mismatch {key}: {config.get(key)!r} != {value!r}"
            )

    unwrap_model(model).load_state_dict(payload["model_state_dict"], strict=True)
    return payload


# =============================================================================
# DATA LOADERS
# =============================================================================


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers > 0),
        collate_fn=collate_studies,
        drop_last=False,
    )


def global_study_batch(args, runtime: Runtime) -> int:
    per_device = max(1, int(args.study_batch_per_gpu))
    devices = runtime.visible_cuda_devices if runtime.backend == "cuda" else 1
    return per_device * max(devices, 1)


def tensor_numeric_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    finite_count = int(finite.sum().item())
    total_count = int(x.numel())

    if finite_count == 0:
        return {
            "shape": list(x.shape),
            "dtype": str(tensor.dtype),
            "finite_count": 0,
            "total_count": total_count,
            "all_finite": False,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
        }

    values = x[finite]
    return {
        "shape": list(x.shape),
        "dtype": str(tensor.dtype),
        "finite_count": finite_count,
        "total_count": total_count,
        "all_finite": finite_count == total_count,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "abs_max": float(values.abs().max().item()),
    }


@torch.inference_mode()
def numerical_forward_preflight(
    model: nn.Module,
    loader: DataLoader,
    runtime: Runtime,
    max_batches: int = NUMERICAL_PREFLIGHT_BATCHES,
) -> Dict[str, Any]:
    model.eval()
    set_backbone_batchnorm_eval(model)
    core = unwrap_model(model)
    records: List[Dict[str, Any]] = []

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= int(max_batches):
            break

        uids = [str(uid) for uid in raw_batch[UID_COLUMN]]
        batch = move_batch(raw_batch, runtime.device)

        images = batch["images"]
        token_mask = batch["token_mask"]
        flat_images = images.reshape(
            images.shape[0] * images.shape[1],
            3,
            images.shape[-2],
            images.shape[-1],
        )
        flat_mask = token_mask.reshape(-1)
        valid_images = flat_images[flat_mask]

        backbone_features = forward_backbone_fp32(
            core.backbone,
            valid_images,
        )

        with autocast_context(runtime):
            output = model(
                images=batch["images"],
                token_mask=batch["token_mask"],
                plane=batch["plane"],
                fluid=batch["fluid"],
                fat_suppression=batch["fat_suppression"],
                slice_position=batch["slice_position"],
            )

        logits = output["logits"]
        loss = masked_soft_bce(
            logits,
            batch["targets"],
            batch["target_mask"],
            batch["target_weight"],
            pos_weight=None,
        )

        record = {
            "batch_index": int(batch_index),
            "uids": uids,
            "images": tensor_numeric_summary(valid_images),
            "backbone_features": tensor_numeric_summary(backbone_features),
            "logits": tensor_numeric_summary(logits),
            "loss": float(loss.detach().cpu()),
        }
        records.append(record)

        if not (
            record["images"]["all_finite"]
            and record["backbone_features"]["all_finite"]
            and record["logits"]["all_finite"]
            and math.isfinite(record["loss"])
        ):
            raise RuntimeError(
                "W43 numerical forward preflight FAILED.\n"
                + json.dumps(record, indent=2, allow_nan=True)
            )

    if len(records) < int(max_batches):
        raise RuntimeError(
            f"Numerical forward preflight checked only {len(records)}/{max_batches} batches."
        )

    return {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "backbone_force_fp32": BACKBONE_FORCE_FP32,
        "outer_autocast_dtype": str(runtime.autocast_dtype),
        "batches_checked": len(records),
        "records": records,
        "overall_pass": True,
    }


# =============================================================================
# TRAINING / PREDICTION
# =============================================================================


def train_one_loader_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    runtime: Runtime,
    pos_weight: Optional[torch.Tensor],
) -> float:
    model.train()
    set_backbone_batchnorm_eval(model)
    losses = []

    if pos_weight is not None:
        pos_weight = pos_weight.to(runtime.device)

    for step_index, raw_batch in enumerate(loader, start=1):
        batch_uids = [str(uid) for uid in raw_batch[UID_COLUMN]]
        batch = move_batch(raw_batch, runtime.device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(runtime):
            output = model(
                images=batch["images"],
                token_mask=batch["token_mask"],
                plane=batch["plane"],
                fluid=batch["fluid"],
                fat_suppression=batch["fat_suppression"],
                slice_position=batch["slice_position"],
            )

        logits = output["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"Non-finite logits at train step={step_index}, uids={batch_uids}. "
                f"stats={json.dumps(tensor_numeric_summary(logits))}"
            )

        loss = masked_soft_bce(
            logits,
            batch["targets"],
            batch["target_mask"],
            batch["target_weight"],
            pos_weight=pos_weight,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at train step={step_index}, uids={batch_uids}."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def train_joint_epoch(
    model: nn.Module,
    gold_loader: DataLoader,
    pseudo_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    runtime: Runtime,
    gold_pos_weight: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    set_backbone_batchnorm_eval(model)

    gold_iterator = iter(gold_loader)
    pseudo_iterator = iter(pseudo_loader)

    total_losses = []
    gold_losses = []
    pseudo_losses = []

    gold_pos_weight = gold_pos_weight.to(runtime.device)

    for _step in range(JOINT_STEPS_PER_EPOCH):
        try:
            gold_batch = next(gold_iterator)
        except StopIteration:
            gold_iterator = iter(gold_loader)
            gold_batch = next(gold_iterator)

        try:
            pseudo_batch = next(pseudo_iterator)
        except StopIteration:
            pseudo_iterator = iter(pseudo_loader)
            pseudo_batch = next(pseudo_iterator)

        gold_batch = move_batch(gold_batch, runtime.device)
        pseudo_batch = move_batch(pseudo_batch, runtime.device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(runtime):
            gold_output = model(
                images=gold_batch["images"],
                token_mask=gold_batch["token_mask"],
                plane=gold_batch["plane"],
                fluid=gold_batch["fluid"],
                fat_suppression=gold_batch["fat_suppression"],
                slice_position=gold_batch["slice_position"],
            )
            pseudo_output = model(
                images=pseudo_batch["images"],
                token_mask=pseudo_batch["token_mask"],
                plane=pseudo_batch["plane"],
                fluid=pseudo_batch["fluid"],
                fat_suppression=pseudo_batch["fat_suppression"],
                slice_position=pseudo_batch["slice_position"],
            )

        if not torch.isfinite(gold_output["logits"]).all():
            raise RuntimeError("Non-finite gold logits during joint training.")
        if not torch.isfinite(pseudo_output["logits"]).all():
            raise RuntimeError("Non-finite pseudo logits during joint training.")

        gold_loss = masked_soft_bce(
            gold_output["logits"],
            gold_batch["targets"],
            gold_batch["target_mask"],
            gold_batch["target_weight"],
            pos_weight=gold_pos_weight,
        )
        pseudo_loss = masked_soft_bce(
            pseudo_output["logits"],
            pseudo_batch["targets"],
            pseudo_batch["target_mask"],
            pseudo_batch["target_weight"],
            pos_weight=None,
        )

        loss = (
            JOINT_GOLD_AUTHORITY * gold_loss + JOINT_PSEUDO_AUTHORITY * pseudo_loss
        ) / (JOINT_GOLD_AUTHORITY + JOINT_PSEUDO_AUTHORITY)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite joint loss.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_losses.append(float(loss.detach().cpu()))
        gold_losses.append(float(gold_loss.detach().cpu()))
        pseudo_losses.append(float(pseudo_loss.detach().cpu()))

    return {
        "loss": float(np.mean(total_losses)),
        "gold_loss": float(np.mean(gold_losses)),
        "pseudo_loss": float(np.mean(pseudo_losses)),
    }


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    runtime: Runtime,
) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in loader:
        uids = list(batch[UID_COLUMN])
        targets = batch["targets"].numpy()
        batch = move_batch(batch, runtime.device)

        with autocast_context(runtime):
            output = model(
                images=batch["images"],
                token_mask=batch["token_mask"],
                plane=batch["plane"],
                fluid=batch["fluid"],
                fat_suppression=batch["fat_suppression"],
                slice_position=batch["slice_position"],
            )
            probabilities = torch.sigmoid(output["logits"]).float().cpu().numpy()

        for i, uid in enumerate(uids):
            for label_index, label in enumerate(LABEL_COLUMNS):
                rows.append(
                    {
                        UID_COLUMN: uid,
                        "Label": label,
                        "Gold": float(targets[i, label_index]),
                        "Probability": float(probabilities[i, label_index]),
                    }
                )

    return pd.DataFrame(rows)


def diagnostic_metrics(oof: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for label in LABEL_COLUMNS:
        subset = oof[oof["Label"] == label]
        y = subset["Gold"].to_numpy(dtype=int)
        p = subset["Probability"].to_numpy(dtype=float)
        rows.append(
            {
                "Label": label,
                "N": len(subset),
                "Positive": int(y.sum()),
                "AUROC": safe_auc(y, p),
                "AP": safe_ap(y, p),
                "Brier": float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            }
        )

    per_label = pd.DataFrame(rows)
    macro_auc = float(per_label["AUROC"].mean())
    macro_ap = float(per_label["AP"].mean())
    macro_brier = float(per_label["Brier"].mean())
    weak = per_label.loc[per_label["AUROC"] < CV_WEAK_LABEL_AUC, "Label"].tolist()

    if macro_auc >= CV_GO_MIN_MACRO_AUC and len(weak) <= CV_GO_MAX_WEAK_LABELS:
        verdict = "GO_TO_FULL"
    elif macro_auc >= 0.72 and len(weak) <= 4:
        verdict = "REVIEW"
    else:
        verdict = "STOP"

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "diagnostic_is_pristine": False,
        "non_pristine_reason": (
            "W40 production pseudo labels use all 58 gold reports as exemplars; "
            "this CV is a regression/architecture diagnostic, not an unbiased estimate."
        ),
        "macro_AUROC": macro_auc,
        "macro_AP": macro_ap,
        "macro_Brier": macro_brier,
        "weak_labels_threshold": CV_WEAK_LABEL_AUC,
        "weak_labels": weak,
        "gate": {
            "min_macro_AUROC": CV_GO_MIN_MACRO_AUC,
            "max_weak_labels": CV_GO_MAX_WEAK_LABELS,
        },
        "verdict": verdict,
    }

    return per_label, summary


# =============================================================================
# STAGE P — W40 PSEUDO REPRESENTATION LEARNING
# =============================================================================


def pseudo_checkpoint_path(paths: ProjectPaths) -> Path:
    return paths.checkpoint_root / "pseudo" / "pseudo_epoch_3.pt"


def run_train_pseudo(paths: ProjectPaths, args, runtime: Runtime) -> Dict[str, Any]:
    paths.ensure_output_dirs()
    train, _series, gold, unlabeled = load_train_tables(paths)
    probs, weights, masks, w40_summary = load_w40_teacher(paths, unlabeled)

    cache = summarize_cache(paths, train)
    if cache["usable_studies"] != EXPECTED_TRAIN:
        raise RuntimeError("Build the complete shared 2.5D cache before training.")

    pseudo_targets, pseudo_masks, pseudo_weights = build_pseudo_target_maps(
        probs, weights, masks
    )

    seed_everything(PSEUDO_SEED)
    model, rad_audit = create_new_training_model(paths, runtime)

    batch_size = global_study_batch(args, runtime)

    # ------------------------------------------------------------------
    # DenseNet BN domain recalibration BEFORE any Stage-P optimization.
    # Deterministic subset, no augmentation, no labels/losses involved.
    # ------------------------------------------------------------------
    all_pseudo_uids = unlabeled[UID_COLUMN].astype(str).tolist()
    calibration_rng = np.random.default_rng(BN_CALIBRATION_SEED)
    calibration_count = min(BN_CALIBRATION_STUDIES, len(all_pseudo_uids))
    calibration_uids = calibration_rng.choice(
        np.asarray(all_pseudo_uids, dtype=object),
        size=calibration_count,
        replace=False,
    ).tolist()

    calibration_dataset = StudyCacheDataset(
        paths,
        calibration_uids,
        pseudo_targets,
        pseudo_masks,
        pseudo_weights,
        training=False,
        max_tokens=BN_CALIBRATION_MAX_TOKENS_PER_STUDY,
        seed=BN_CALIBRATION_SEED,
    )
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=min(max(1, batch_size), 2),
        shuffle=False,
        workers=args.loader_workers,
    )

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | DENSENET BN RECALIBRATION")
    log("=" * 100)
    log(f"Calibration studies        : {len(calibration_uids)}")
    log(f"Max stacks requested       : {BN_CALIBRATION_MAX_STACKS}")
    log(f"Image batch                : {BN_CALIBRATION_IMAGE_BATCH}")
    log(f"Strategy                   : {BN_STRATEGY}")

    bn_summary = recalibrate_backbone_batchnorm(
        model,
        calibration_loader,
        runtime,
        max_stacks=BN_CALIBRATION_MAX_STACKS,
    )
    write_json(paths.result_root / "09_bn_recalibration_summary.json", bn_summary)

    log(
        "BN source running_var     : "
        f"max={bn_summary['source_checkpoint_buffers']['running_var']['max']:.4f} "
        f"gt1e4={bn_summary['source_checkpoint_buffers']['running_var']['gt_1e4_count']}"
    )
    log(
        "BN recalibrated running_var: "
        f"max={bn_summary['recalibrated_buffers']['running_var']['max']:.4f} "
        f"gt1e4={bn_summary['recalibrated_buffers']['running_var']['gt_1e4_count']}"
    )
    log(f"BN calibration stacks      : {bn_summary['calibration_stacks']}")
    log("BN calibration             : PASS")

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | NUMERICAL FORWARD PREFLIGHT")
    log("=" * 100)
    numeric_summary = numerical_forward_preflight(
        model,
        calibration_loader,
        runtime,
        max_batches=NUMERICAL_PREFLIGHT_BATCHES,
    )
    write_json(
        paths.result_root / "09b_numerical_forward_preflight.json",
        numeric_summary,
    )
    for record in numeric_summary["records"]:
        log(
            f"preflight batch={record['batch_index']} "
            f"feature_abs_max={record['backbone_features']['abs_max']:.6g} "
            f"logit_abs_max={record['logits']['abs_max']:.6g} "
            f"loss={record['loss']:.6f}"
        )
    log("Numerical forward preflight: PASS")

    del calibration_loader, calibration_dataset

    dataset = StudyCacheDataset(
        paths,
        all_pseudo_uids,
        pseudo_targets,
        pseudo_masks,
        pseudo_weights,
        training=True,
        max_tokens=TRAIN_TOKENS_PER_STUDY,
        seed=PSEUDO_SEED,
    )
    loader = make_loader(dataset, batch_size, True, args.loader_workers)
    scaler = make_grad_scaler(runtime)

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | STAGE P: W40 PSEUDO PRETRAIN")
    log("=" * 100)
    log(f"Pseudo studies             : {len(unlabeled)}")
    log(f"Selected pseudo cells      : {EXPECTED_SELECTED_CELLS}")
    log(f"Global study batch         : {batch_size}")
    log(f"Train 2.5D tokens/study    : {TRAIN_TOKENS_PER_STUDY}")
    log(f"RadImageNet weights        : {paths.radimagenet_weights}")
    log(f"RadImageNet SHA256         : {rad_audit['sha256']}")
    log(f"Preprocessing              : {RADIMAGENET_PREPROCESSING}")
    log(f"BatchNorm policy           : {BN_STRATEGY}")

    history = []
    for epoch, stage in enumerate(PSEUDO_STAGES, start=1):
        set_backbone_train_stage(model, stage)
        optimizer = build_optimizer(
            model,
            backbone_lr=stage_backbone_lr(stage, gold=False),
            head_lr=PSEUDO_HEAD_LR,
        )

        started = time.time()
        loss = train_one_loader_epoch(
            model,
            loader,
            optimizer,
            scaler,
            runtime,
            pos_weight=None,
        )
        row = {
            "epoch": epoch,
            "stage": stage,
            "loss": loss,
            "seconds": time.time() - started,
            "backbone_lr": stage_backbone_lr(stage, gold=False),
            "head_lr": PSEUDO_HEAD_LR,
        }
        history.append(row)
        log(
            f"pseudo epoch {epoch:02d}/{PSEUDO_EPOCHS} "
            f"stage={stage:<6} loss={loss:.6f} time={row['seconds']:.1f}s"
        )

    checkpoint = pseudo_checkpoint_path(paths)
    checkpoint_record = save_model_checkpoint(
        model,
        checkpoint,
        scope="pseudo_pretrain",
        seed=PSEUDO_SEED,
        extra={
            "radimagenet_audit": rad_audit,
            "bn_recalibration": bn_summary,
            "numerical_forward_preflight": numeric_summary,
            "w40_summary_status": w40_summary.get("status"),
            "history": history,
        },
    )

    pd.DataFrame(history).to_csv(
        paths.result_root / "10_pseudo_history.csv", index=False
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "stage": "pseudo_pretrain",
        "checkpoint": checkpoint_record,
        "history": history,
        "radimagenet_audit": rad_audit,
        "bn_recalibration": bn_summary,
        "numerical_forward_preflight": numeric_summary,
        "cache": cache,
    }
    write_json(paths.result_root / "11_pseudo_summary.json", summary)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# =============================================================================
# STAGE C — NON-PRISTINE 5-FOLD DIAGNOSTIC
# =============================================================================


def run_train_cv(paths: ProjectPaths, args, runtime: Runtime) -> Dict[str, Any]:
    paths.ensure_output_dirs()
    train, _series, gold, unlabeled = load_train_tables(paths)
    folds = load_locked_folds(paths, gold)
    probs, weights, masks, _w40_summary = load_w40_teacher(paths, unlabeled)

    cache = summarize_cache(paths, train)
    if cache["usable_studies"] != EXPECTED_TRAIN:
        raise RuntimeError("Shared 2.5D cache is incomplete.")

    pseudo_ckpt = pseudo_checkpoint_path(paths)
    if not pseudo_ckpt.exists():
        raise FileNotFoundError(
            f"Stage-P checkpoint missing: {pseudo_ckpt}. Run train_pseudo first."
        )

    gold_targets, gold_masks, gold_weights = build_gold_target_maps(gold)
    gold_pos_weight_all = gold_positive_weight(gold)

    fold_map = folds.set_index(UID_COLUMN)["OuterFold"].astype(int).to_dict()
    all_oof = []
    fold_summaries = []
    batch_size = global_study_batch(args, runtime)

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | STAGE C: 5-FOLD DIAGNOSTIC")
    log("=" * 100)
    log("IMPORTANT: this diagnostic is NON-PRISTINE because W40 production teacher")
    log("used all 58 gold reports as final production exemplars.")

    for fold in range(1, 6):
        seed = CV_SEED_BASE + fold
        seed_everything(seed)

        train_uids = [
            uid for uid in gold[UID_COLUMN].astype(str) if fold_map[uid] != fold
        ]
        val_uids = [
            uid for uid in gold[UID_COLUMN].astype(str) if fold_map[uid] == fold
        ]

        model = create_empty_model(runtime)
        load_model_checkpoint(model, pseudo_ckpt, expected_scope="pseudo_pretrain")
        scaler = make_grad_scaler(runtime)

        train_dataset = StudyCacheDataset(
            paths,
            train_uids,
            gold_targets,
            gold_masks,
            gold_weights,
            training=True,
            max_tokens=TRAIN_TOKENS_PER_STUDY,
            seed=seed,
        )
        val_dataset = StudyCacheDataset(
            paths,
            val_uids,
            gold_targets,
            gold_masks,
            gold_weights,
            training=False,
            max_tokens=EVAL_MAX_TOKENS,
            seed=seed,
        )
        train_loader = make_loader(train_dataset, batch_size, True, args.loader_workers)
        val_loader = make_loader(val_dataset, 1, False, args.loader_workers)

        # Fold-specific positive weights from outer-train only.
        fold_gold = gold[gold[UID_COLUMN].astype(str).isin(train_uids)]
        fold_pos_weight = gold_positive_weight(fold_gold)

        history = []
        log("-" * 96)
        log(
            f"CV fold {fold} | seed={seed} | train={len(train_uids)} val={len(val_uids)}"
        )

        for epoch, stage in enumerate(CV_STAGE_SCHEDULE, start=1):
            set_backbone_train_stage(model, stage)
            optimizer = build_optimizer(
                model,
                backbone_lr=stage_backbone_lr(stage, gold=True),
                head_lr=GOLD_HEAD_LR,
            )
            loss = train_one_loader_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                runtime,
                pos_weight=fold_pos_weight,
            )
            history.append({"epoch": epoch, "stage": stage, "loss": loss})
            log(
                f"  epoch {epoch:02d}/{CV_ADAPT_EPOCHS} "
                f"stage={stage:<4} gold_loss={loss:.6f}"
            )

        predictions = predict_loader(model, val_loader, runtime)
        predictions["OuterFold"] = fold
        all_oof.append(predictions)

        checkpoint = paths.checkpoint_root / "cv" / f"fold_{fold}_final.pt"
        checkpoint_record = save_model_checkpoint(
            model,
            checkpoint,
            scope=f"cv_fold_{fold}",
            seed=seed,
            extra={"history": history},
        )
        fold_summaries.append(
            {
                "fold": fold,
                "seed": seed,
                "train_studies": len(train_uids),
                "val_studies": len(val_uids),
                "checkpoint": checkpoint_record,
                "history": history,
            }
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof = pd.concat(all_oof, ignore_index=True)
    expected_rows = EXPECTED_GOLD * NUM_LABELS
    if len(oof) != expected_rows:
        raise RuntimeError(f"OOF rows={len(oof)}, expected={expected_rows}")
    if oof[[UID_COLUMN, "Label"]].duplicated().any():
        raise RuntimeError("OOF contains duplicate UID/Label rows.")

    per_label, summary = diagnostic_metrics(oof)
    summary.update(
        {
            "fold_sha256": fold_assignment_sha256(folds),
            "fold_summaries": fold_summaries,
            "pseudo_checkpoint_sha256": sha256_file(pseudo_ckpt),
        }
    )

    oof.to_csv(paths.result_root / "20_cv_oof_predictions.csv", index=False)
    per_label.to_csv(paths.result_root / "21_cv_per_label_metrics.csv", index=False)
    write_json(paths.result_root / "22_cv_summary.json", summary)

    log("=" * 100)
    log("W43 DIAGNOSTIC")
    log("=" * 100)
    log(f"Macro AUROC          : {summary['macro_AUROC']:.6f}")
    log(f"Macro AP             : {summary['macro_AP']:.6f}")
    log(f"Macro Brier          : {summary['macro_Brier']:.6f}")
    log(
        f"Weak labels < .55    : {len(summary['weak_labels'])} -> {summary['weak_labels']}"
    )
    log(f"VERDICT              : {summary['verdict']}")

    return summary


# =============================================================================
# STAGE F — FULL PRODUCTION MODELS
# =============================================================================


def run_train_full(
    paths: ProjectPaths, args, runtime: Runtime, force: bool = False
) -> Dict[str, Any]:
    paths.ensure_output_dirs()
    train, _series, gold, unlabeled = load_train_tables(paths)
    probs, weights, masks, _w40_summary = load_w40_teacher(paths, unlabeled)

    cache = summarize_cache(paths, train)
    if cache["usable_studies"] != EXPECTED_TRAIN:
        raise RuntimeError("Shared 2.5D cache is incomplete.")

    pseudo_ckpt = pseudo_checkpoint_path(paths)
    if not pseudo_ckpt.exists():
        raise FileNotFoundError(pseudo_ckpt)

    cv_summary_path = paths.result_root / "22_cv_summary.json"
    cv_summary = {}
    if cv_summary_path.exists():
        cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))

    if not force:
        verdict = cv_summary.get("verdict")
        if verdict != "GO_TO_FULL":
            raise RuntimeError(
                f"W43 full training requires CV verdict GO_TO_FULL; found {verdict!r}. "
                "Use --force-full only for an intentional override."
            )

    gold_targets, gold_masks, gold_weights = build_gold_target_maps(gold)
    pseudo_targets, pseudo_masks, pseudo_weights = build_pseudo_target_maps(
        probs, weights, masks
    )
    gold_pos_weight_all = gold_positive_weight(gold)

    batch_size = global_study_batch(args, runtime)
    full_records = []

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | STAGE F: FULL PRODUCTION FIT")
    log("=" * 100)
    log(f"Full seeds                : {FULL_SEEDS}")
    log(f"Global study batch        : {batch_size}")
    log(
        f"Joint authority           : gold={JOINT_GOLD_AUTHORITY:g}, pseudo={JOINT_PSEUDO_AUTHORITY:g}"
    )

    for seed in FULL_SEEDS:
        seed_everything(seed)
        model = create_empty_model(runtime)
        load_model_checkpoint(model, pseudo_ckpt, expected_scope="pseudo_pretrain")
        scaler = make_grad_scaler(runtime)

        gold_dataset = StudyCacheDataset(
            paths,
            gold[UID_COLUMN].astype(str).tolist(),
            gold_targets,
            gold_masks,
            gold_weights,
            training=True,
            max_tokens=TRAIN_TOKENS_PER_STUDY,
            seed=seed,
        )
        pseudo_dataset = StudyCacheDataset(
            paths,
            unlabeled[UID_COLUMN].astype(str).tolist(),
            pseudo_targets,
            pseudo_masks,
            pseudo_weights,
            training=True,
            max_tokens=TRAIN_TOKENS_PER_STUDY,
            seed=seed + 100000,
        )

        gold_loader = make_loader(gold_dataset, batch_size, True, args.loader_workers)
        pseudo_loader = make_loader(
            pseudo_dataset, batch_size, True, args.loader_workers
        )

        history = []
        log("-" * 96)
        log(f"FULL seed={seed}")

        for epoch, stage in enumerate(FULL_STAGE_SCHEDULE, start=1):
            set_backbone_train_stage(model, stage)
            optimizer = build_optimizer(
                model,
                backbone_lr=stage_backbone_lr(stage, gold=True),
                head_lr=GOLD_HEAD_LR,
            )
            loss = train_one_loader_epoch(
                model,
                gold_loader,
                optimizer,
                scaler,
                runtime,
                pos_weight=gold_pos_weight_all,
            )
            row = {
                "phase": "gold",
                "epoch": epoch,
                "stage": stage,
                "loss": loss,
            }
            history.append(row)
            log(
                f"  gold epoch {epoch:02d}/{FULL_GOLD_EPOCHS} "
                f"stage={stage:<4} loss={loss:.6f}"
            )

        set_backbone_train_stage(model, "full")
        joint_optimizer = build_optimizer(
            model,
            backbone_lr=JOINT_BACKBONE_LR,
            head_lr=JOINT_HEAD_LR,
        )

        for joint_epoch in range(1, FULL_JOINT_EPOCHS + 1):
            joint = train_joint_epoch(
                model,
                gold_loader,
                pseudo_loader,
                joint_optimizer,
                scaler,
                runtime,
                gold_pos_weight_all,
            )
            history.append(
                {
                    "phase": "joint",
                    "epoch": joint_epoch,
                    "stage": "full",
                    **joint,
                }
            )
            log(
                f"  joint epoch {joint_epoch:02d}/{FULL_JOINT_EPOCHS} "
                f"loss={joint['loss']:.6f} gold={joint['gold_loss']:.6f} "
                f"pseudo={joint['pseudo_loss']:.6f}"
            )

        checkpoint = paths.checkpoint_root / "full" / f"seed_{seed}_final.pt"
        record = save_model_checkpoint(
            model,
            checkpoint,
            scope="full_all58",
            seed=seed,
            extra={"history": history},
        )
        record["history"] = history
        full_records.append(record)

        pd.DataFrame(history).to_csv(
            paths.result_root / f"30_full_seed_{seed}_history.csv",
            index=False,
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "script_version": SCRIPT_VERSION,
        "stage": "full_all58",
        "created_at": now_iso(),
        "seeds": FULL_SEEDS,
        "models": full_records,
        "cv_verdict": cv_summary.get("verdict"),
        "use": "future hidden-test inference after validation",
        "teacher": "W40",
        "backbone": "RadImageNet DenseNet-121",
        "cache_version": CACHE_VERSION,
    }
    write_json(paths.result_root / "31_fullfit_summary.json", summary)

    return summary


# =============================================================================
# STATUS / VALIDATION / TRAIN-ALL
# =============================================================================


def run_status(paths: ProjectPaths, args, runtime: Runtime) -> Dict[str, Any]:
    paths.ensure_output_dirs()

    dependency = {
        "pydicom": pydicom is not None,
        "opencv": cv2 is not None,
    }
    try:
        import torchvision  # noqa: F401
        from torchvision.models import densenet121  # noqa: F401

        dependency["torchvision"] = True
    except Exception:
        dependency["torchvision"] = False

    train = series = gold = unlabeled = None
    data_error = None
    try:
        train, series, gold, unlabeled = load_train_tables(paths)
    except Exception as exc:
        data_error = repr(exc)

    fold_info = {}
    fold_error = None
    if gold is not None:
        try:
            folds = load_locked_folds(paths, gold)
            fold_info = {
                "sha256": fold_assignment_sha256(folds),
                "counts": folds["OuterFold"].value_counts().sort_index().to_dict(),
            }
        except Exception as exc:
            fold_error = repr(exc)

    dicom_decoder_audit = None
    dicom_decoder_error = None
    if series is not None and dependency["pydicom"]:
        try:
            dicom_decoder_audit = inspect_dicom_decoder_support(paths, series)
            if not dicom_decoder_audit.get("overall_pass"):
                if (
                    int(dicom_decoder_audit.get("missing_series_directory_count", 0))
                    > 0
                ):
                    dicom_decoder_error = (
                        "Training DICOM tree is incomplete for sampled train_series.csv "
                        "rows. Run W43 where the full competition train_series directory "
                        "is mounted (recommended: Kaggle)."
                    )
                elif (
                    int(dicom_decoder_audit.get("failed_transfer_syntax_count", 0)) > 0
                ):
                    dicom_decoder_error = (
                        "One or more sampled DICOM transfer syntaxes cannot be "
                        "pixel-decoded. Install codecs with: "
                        f"{DICOM_DECODER_INSTALL_COMMAND}"
                    )
                else:
                    dicom_decoder_error = (
                        "DICOM decoder/data preflight failed; inspect audit details."
                    )
        except Exception as exc:
            dicom_decoder_error = repr(exc)

    teacher_info = {}
    teacher_error = None
    if unlabeled is not None:
        try:
            probs, weights, masks, w40_summary = load_w40_teacher(paths, unlabeled)
            teacher_info = {
                "rows": len(probs),
                "selected_cells": int(
                    masks[LABEL_COLUMNS]
                    .apply(coerce_bool_series)
                    .to_numpy(dtype=bool)
                    .sum()
                ),
                "changed_probabilities": w40_summary.get("controlled_changes", {}).get(
                    "probabilities_changed"
                ),
                "weights_preserved": w40_summary.get("controlled_changes", {}).get(
                    "all_teacher_weights_preserved"
                ),
                "masks_preserved": w40_summary.get("controlled_changes", {}).get(
                    "all_teacher_masks_preserved"
                ),
            }
        except Exception as exc:
            teacher_error = repr(exc)

    rad_audit = None
    rad_error = None
    if paths.radimagenet_weights is not None and dependency.get("torchvision"):
        try:
            rad_audit = audit_radimagenet_weights(paths.radimagenet_weights)
        except Exception as exc:
            rad_error = repr(exc)
    elif paths.radimagenet_weights is None:
        rad_error = (
            "RadImageNet DenseNet-121 checkpoint not discovered. "
            "Pass --radimagenet-weights /path/to/checkpoint.pt"
        )
    elif not dependency.get("torchvision"):
        rad_error = "torchvision DenseNet-121 is not available."

    cache_info = summarize_cache(paths, train) if train is not None else {}

    payload = {
        "script_version": SCRIPT_VERSION,
        "experiment": DISPLAY_VERSION,
        "runtime": {
            "requested": runtime.accelerator_requested,
            "backend": runtime.backend,
            "device": str(runtime.device),
            "visible_cuda_devices": runtime.visible_cuda_devices,
            "autocast_dtype": str(runtime.autocast_dtype),
            "data_parallel": runtime.use_data_parallel,
            "accelerator_env_variables_required": False,
        },
        "dependencies": dependency,
        "paths": {
            "project_root": str(paths.project_root),
            "train_csv": str(paths.train_csv),
            "train_series_csv": str(paths.train_series_csv),
            "train_series_root": str(paths.train_series_root),
            "w40_root": str(paths.w40_root),
            "fold_csv": str(paths.fold_csv),
            "shared_cache_root": str(paths.cache_root),
            "output_root": str(paths.output_root),
            "radimagenet_weights": (
                str(paths.radimagenet_weights) if paths.radimagenet_weights else None
            ),
        },
        "data": {
            "error": data_error,
            "train_rows": len(train) if train is not None else None,
            "series_rows": len(series) if series is not None else None,
            "gold_rows": len(gold) if gold is not None else None,
            "unlabeled_rows": len(unlabeled) if unlabeled is not None else None,
        },
        "folds": {"info": fold_info, "error": fold_error},
        "dicom_decoder": {
            "audit": dicom_decoder_audit,
            "error": dicom_decoder_error,
            "recommended_install": DICOM_DECODER_INSTALL_COMMAND,
        },
        "teacher": {"info": teacher_info, "error": teacher_error},
        "radimagenet": {
            "audit": rad_audit,
            "error": rad_error,
            "preprocessing": RADIMAGENET_PREPROCESSING,
            "batchnorm_policy": {
                "strategy": BN_STRATEGY,
                "source_extreme_running_var_detected": bool(
                    rad_audit is not None
                    and rad_audit.get("batchnorm_running_var", {}).get(
                        "extreme_gt_1e4_count", 0
                    )
                    > 0
                ),
                "calibration_studies": BN_CALIBRATION_STUDIES,
                "calibration_max_stacks": BN_CALIBRATION_MAX_STACKS,
                "calibration_min_stacks": BN_CALIBRATION_MIN_STACKS,
                "calibration_image_batch": BN_CALIBRATION_IMAGE_BATCH,
                "running_stats_frozen_after_calibration": True,
                "affine_gamma_beta_can_still_train": True,
            },
        },
        "cache": cache_info,
        "architecture": {
            "backbone": "RadImageNet DenseNet-121",
            "input": "2.5D previous/center/next MRI slices",
            "image_size": CACHE_IMAGE_SIZE,
            "all_series": True,
            "stacks_per_series": STACKS_PER_SERIES,
            "study_transformer_layers": TOKEN_LAYERS,
            "label_specific_attention": True,
            "soft_plane_prior": True,
            "horizontal_flip_augmentation": False,
        },
        "training": {
            "pseudo_epochs": PSEUDO_EPOCHS,
            "cv_adapt_epochs": CV_ADAPT_EPOCHS,
            "full_gold_epochs": FULL_GOLD_EPOCHS,
            "full_joint_epochs": FULL_JOINT_EPOCHS,
            "full_seeds": FULL_SEEDS,
            "w40_reused": True,
            "bn_strategy": BN_STRATEGY,
            "precision_policy": {
                "backbone": "float32_forced",
                "study_transformer_and_head": (
                    str(runtime.autocast_dtype)
                    if runtime.autocast_dtype is not None
                    else "float32"
                ),
                "loss": "float32",
                "grad_scaler_enabled": bool(
                    runtime.backend == "cuda"
                    and runtime.autocast_dtype == torch.float16
                ),
            },
        },
        "ready_for_cache": bool(
            data_error is None
            and dependency["pydicom"]
            and dicom_decoder_audit is not None
            and dicom_decoder_audit.get("overall_pass") is True
        ),
        "ready_for_training": bool(
            data_error is None
            and fold_error is None
            and teacher_error is None
            and dependency.get("torchvision")
            and rad_audit is not None
            and rad_audit.get("overall_pass") is True
            and dicom_decoder_audit is not None
            and dicom_decoder_audit.get("overall_pass") is True
            and cache_info.get("usable_studies") == EXPECTED_TRAIN
        ),
        "scope": {
            "backbone_family": "RadImageNet CNN",
            "resnet50": False,
            "resnet152": False,
            "dino": False,
            "curia": False,
            "w40_teacher": True,
            "project_python_imports": False,
            "submission_mode": False,
            "kaggle_dataset_auto_discovery": {
                "competition_data": True,
                "w40": True,
                "locked_fold_csv": True,
                "radimagenet_weights": True,
                "zip_extraction": False,
            },
        },
    }

    write_json(paths.result_root / "00_status.json", payload)
    log(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True))
    return payload


def run_validate(paths: ProjectPaths, args, runtime: Runtime) -> Dict[str, Any]:
    paths.ensure_output_dirs()
    train, _series, gold, unlabeled = load_train_tables(paths)
    folds = load_locked_folds(paths, gold)
    probs, weights, masks, _summary = load_w40_teacher(paths, unlabeled)
    cache = summarize_cache(paths, train)
    dicom_decoder_audit = inspect_dicom_decoder_support(paths, _series)

    checks: Dict[str, bool] = {
        "train_4407": len(train) == EXPECTED_TRAIN,
        "gold_58": len(gold) == EXPECTED_GOLD,
        "unlabeled_4349": len(unlabeled) == EXPECTED_UNLABELED,
        "dicom_decoder_preflight": bool(
            dicom_decoder_audit.get("overall_pass") is True
        ),
        "fold_sha_match": fold_assignment_sha256(folds) == EXPECTED_FOLD_SHA256,
        "w40_selected_cells_32027": int(
            masks[LABEL_COLUMNS].apply(coerce_bool_series).to_numpy(dtype=bool).sum()
        )
        == EXPECTED_SELECTED_CELLS,
        "shared_cache_4407": cache["usable_studies"] == EXPECTED_TRAIN,
    }

    rad_audit = None
    if paths.radimagenet_weights is not None:
        try:
            rad_audit = audit_radimagenet_weights(paths.radimagenet_weights)
            checks["radimagenet_checkpoint_exact_enough"] = bool(
                rad_audit["overall_pass"]
            )
        except Exception:
            checks["radimagenet_checkpoint_exact_enough"] = False
    else:
        # Once Stage-P exists, final validation does not require the original
        # pretrained checkpoint file to remain mounted.
        checks["radimagenet_checkpoint_exact_enough"] = pseudo_checkpoint_path(
            paths
        ).exists()

    pseudo_ckpt = pseudo_checkpoint_path(paths)
    checks["pseudo_checkpoint_exists"] = pseudo_ckpt.exists()

    if pseudo_ckpt.exists():
        try:
            model = create_empty_model(runtime)
            pseudo_payload = load_model_checkpoint(
                model,
                pseudo_ckpt,
                expected_scope="pseudo_pretrain",
            )
            checks["pseudo_checkpoint_strict_load"] = True
            bn_info = pseudo_payload.get("extra", {}).get("bn_recalibration", {})
            checks["pseudo_bn_recalibration_pass"] = bool(
                bn_info.get("overall_pass") is True
                and int(bn_info.get("calibration_stacks", 0))
                >= BN_CALIBRATION_MIN_STACKS
                and bn_info.get("strategy") == BN_STRATEGY
            )
            numeric_info = pseudo_payload.get("extra", {}).get(
                "numerical_forward_preflight", {}
            )
            checks["pseudo_numerical_preflight_pass"] = bool(
                numeric_info.get("overall_pass") is True
                and int(numeric_info.get("batches_checked", 0))
                >= NUMERICAL_PREFLIGHT_BATCHES
                and numeric_info.get("backbone_force_fp32") is True
            )
            del model
        except Exception:
            checks["pseudo_checkpoint_strict_load"] = False
    else:
        checks["pseudo_checkpoint_strict_load"] = False
        checks["pseudo_bn_recalibration_pass"] = False
        checks["pseudo_numerical_preflight_pass"] = False

    cv_summary_path = paths.result_root / "22_cv_summary.json"
    if cv_summary_path.exists():
        cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
        checks["cv_summary_version"] = (
            cv_summary.get("script_version") == SCRIPT_VERSION
        )
        checks["cv_verdict_valid"] = cv_summary.get("verdict") in {
            "GO_TO_FULL",
            "REVIEW",
            "STOP",
        }
    else:
        cv_summary = {}
        checks["cv_summary_version"] = False
        checks["cv_verdict_valid"] = False

    full_summary_path = paths.result_root / "31_fullfit_summary.json"
    if full_summary_path.exists():
        full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
        checks["full_summary_version"] = (
            full_summary.get("script_version") == SCRIPT_VERSION
        )
        for seed in FULL_SEEDS:
            checkpoint = paths.checkpoint_root / "full" / f"seed_{seed}_final.pt"
            checks[f"full_seed_{seed}_exists"] = checkpoint.exists()
            if checkpoint.exists():
                try:
                    model = create_empty_model(runtime)
                    payload = load_model_checkpoint(
                        model, checkpoint, expected_scope="full_all58"
                    )
                    checks[f"full_seed_{seed}_strict_load"] = (
                        int(payload.get("seed", -1)) == seed
                    )
                    del model
                except Exception:
                    checks[f"full_seed_{seed}_strict_load"] = False
            else:
                checks[f"full_seed_{seed}_strict_load"] = False
    else:
        full_summary = {}
        checks["full_summary_version"] = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Validation can be considered complete at CV stage when the gate is STOP;
    # full-model checks become mandatory only when GO_TO_FULL has been reached.
    cv_verdict = cv_summary.get("verdict")
    if cv_verdict != "GO_TO_FULL":
        mandatory = {
            key: value for key, value in checks.items() if not key.startswith("full_")
        }
    else:
        mandatory = checks

    payload = {
        "script_version": SCRIPT_VERSION,
        "overall_pass": bool(all(mandatory.values())),
        "checks": checks,
        "cv_verdict": cv_verdict,
        "cache": cache,
        "radimagenet_audit": rad_audit,
        "results_root": str(paths.result_root),
    }
    write_json(paths.result_root / "40_validation_summary.json", payload)
    log(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True))

    if not payload["overall_pass"]:
        raise RuntimeError("W43 validation failed.")

    return payload


def run_train_all(paths: ProjectPaths, args, runtime: Runtime) -> Dict[str, Any]:
    if (
        summarize_cache(paths, load_train_tables(paths)[0])["usable_studies"]
        != EXPECTED_TRAIN
    ):
        log("Shared cache incomplete -> building cache first.")
        run_cache(paths, args)

    pseudo_summary = run_train_pseudo(paths, args, runtime)
    cv_summary = run_train_cv(paths, args, runtime)

    full_summary = None
    if cv_summary["verdict"] == "GO_TO_FULL":
        full_summary = run_train_full(paths, args, runtime, force=False)
    else:
        log("")
        log(
            f"W43 CV verdict is {cv_summary['verdict']}. Full production training is not started."
        )

    return {
        "pseudo": pseudo_summary,
        "cv": cv_summary,
        "full": full_summary,
    }


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DISPLAY_VERSION)

    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "status",
            "cache",
            "train_pseudo",
            "train_cv",
            "train_full",
            "train_all",
            "validate",
        ],
        default=None,
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=[
            "status",
            "cache",
            "train_pseudo",
            "train_cv",
            "train_full",
            "train_all",
            "validate",
        ],
        default=None,
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="auto | localGPU | kaggle_t4 | apple_mps | cpu | kaggle_tpu",
    )

    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--w40-root", default=None)
    parser.add_argument("--fold-csv", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--radimagenet-weights", default=None)

    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--study-batch-per-gpu", type=int, default=1)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--force-full", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mode = args.mode_flag or args.mode or "status"

    paths = ProjectPaths.discover(args)
    runtime = resolve_runtime(args.accelerator)

    if mode == "status":
        run_status(paths, args, runtime)
    elif mode == "cache":
        run_cache(paths, args)
    elif mode == "train_pseudo":
        run_train_pseudo(paths, args, runtime)
    elif mode == "train_cv":
        run_train_cv(paths, args, runtime)
    elif mode == "train_full":
        run_train_full(paths, args, runtime, force=bool(args.force_full))
    elif mode == "train_all":
        run_train_all(paths, args, runtime)
    elif mode == "validate":
        run_validate(paths, args, runtime)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
