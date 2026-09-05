# %%writefile w45_rsna_mi2_production_submission_v1.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W45 — MedImageInsight Production Submission v1
================================================

PURPOSE
-------
Promote the successful W44 MedImageInsight (MI2) representation into a
production hidden-test submission.

W44 diagnostic:
    MI2 Macro AUROC = 0.742447
    Curia reference = 0.702997
    delta           = +0.039450

This script deliberately does NOT start another representation experiment.

PRODUCTION PIPELINE
-------------------
TRAIN
    saved W44 MI2 train embeddings
        -> exact W44 MI2 label-aware head
        -> W40 pseudo pretraining on 4,349 studies
        -> all-58 gold adaptation
        -> 5 independent production seeds

TEST
    competition test DICOM
        -> exact W43 2.5D geometry/preprocessing
        -> center slice from each 2.5D stack
        -> grayscale replicated to RGB
        -> frozen MedImageInsight
        -> reusable test MI2 embedding cache
        -> 5 production heads
        -> probability mean
        -> submission.csv

IMPORTANT
---------
* NO DINO.
* NO ResNet.
* NO RadImageNet.
* NO BiomedCLIP in the primary W45 production path.
* NO Curia retraining.
* NO project .py imports.
* NO ZIP extraction.
* Large MedImageInsight encoder remains frozen.
* Train MI2 cache is reused; it is never regenerated here.
* Test MI2 cache is reusable and should be saved as a Kaggle Dataset.
* Two visible CUDA GPUs are used as independent extraction workers.
* nn.DataParallel is NOT used.
* sample_submission.csv determines final UID and column order.
* No post-hoc calibration is used for the first leaderboard measurement.
* An optional W41 probability blend is provided, but it never replaces the
  MI2 baseline unless --promote-blend is explicitly supplied.

PLANE-MAPPING COMPATIBILITY
---------------------------
The W43 shared 2.5D cache encoded planes as:
    Axial=0, Coronal=1, Sagittal=2

W44's head used its plane-prior tensor with a conceptual ordering of:
    Sagittal, Coronal, Axial

Therefore W44's validated 0.742447 result contains that historical mapping
behavior. W45 intentionally preserves it. We do NOT silently change the
mapping in production without re-running CV.

CLI
---
Status:
    python w45_rsna_mi2_production_submission_v1.py status \
        --accelerator kaggle_t4

Train 5 full heads:
    python w45_rsna_mi2_production_submission_v1.py train_full \
        --accelerator kaggle_t4

Extract hidden-test MI2 embeddings:
    python w45_rsna_mi2_production_submission_v1.py extract_test \
        --accelerator kaggle_t4

Create MI2 submission:
    python w45_rsna_mi2_production_submission_v1.py submit \
        --accelerator kaggle_t4

All:
    python w45_rsna_mi2_production_submission_v1.py run_all \
        --accelerator kaggle_t4

Validate:
    python w45_rsna_mi2_production_submission_v1.py validate \
        --accelerator kaggle_t4

Optional later W41 blend:
    python w45_rsna_mi2_production_submission_v1.py blend_w41 \
        --accelerator kaggle_t4 \
        --w41-submission /path/to/w41_submission.csv \
        --mi2-weight 0.80

"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pydicom
except Exception:
    pydicom = None

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# 1. IDENTITY / LOCKS
# =============================================================================

SCRIPT_VERSION = "w45_mi2_production_submission_v1"
DISPLAY_VERSION = "W45 | MedImageInsight Production Submission v1"
OUTPUT_DIR_NAME = "rsna_w45_mi2_production_submission_v1"

UID_COLUMN = "StudyInstanceUID"
SERIES_UID_COLUMN = "SeriesInstanceUID"

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
EXPECTED_SELECTED_CELLS = 32027

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

# W44 train representation identity.
TRAIN_MI2_CACHE_VERSION = "w44_frozen_medical_embedding_cache_v1"
VLM_INPUT_MODE = "center_slice_replicated_rgb"

MI2_DIM = 1024
MI2_IMAGE_SIZE = 480

MI2_WEIGHT_SHA256 = "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"

EXPECTED_TRAIN_MI2_TOKENS = 121855

# New hidden-test representation cache.
TEST_MI2_CACHE_VERSION = "w45_mi2_test_embedding_cache_v1"

# W43 2.5D preprocessing contract.
SHARED_CACHE_IMAGE_SIZE = 224
STACKS_PER_SERIES = 5
STACK_OFFSETS = (-1, 0, +1)

GEOMETRY_PLANE_CONFIDENCE = 0.80
ORIENTATION_MIN_ALIGNMENT = 0.70

# IMPORTANT: exact W43 cache plane IDs.
CACHE_PLANE_TO_INDEX = {
    "Axial": 0,
    "Coronal": 1,
    "Sagittal": 2,
}

# Canonical image orientation.
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

DICOM_PREFLIGHT_SERIES_STATUS = 64
DICOM_PREFLIGHT_SERIES_EXTRACT = 256

DICOM_DECODER_INSTALL_COMMAND = (
    "python -m pip install -U " "pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
)


# =============================================================================
# 2. EXACT W44 MI2 HEAD / TRAINING RECIPE
# =============================================================================

HIDDEN_DIM = 192
PLANE_EMBED_DIM = 16
BINARY_META_DIM = 8
POSITION_DIM = 32

HEAD_DROPOUT = 0.18

MAX_VLM_TOKENS = 64

PSEUDO_EPOCHS = 5
GOLD_ADAPT_EPOCHS = 10

PSEUDO_BATCH = 48
GOLD_BATCH = 16
INFERENCE_BATCH = 64

PSEUDO_LR = 3e-4
GOLD_LR = 8e-5

WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 3.0

# W44 MI2 pseudo initialization seed.
PSEUDO_SEED = 44100

# Five cheap all-58 production heads.
FULL_SEEDS = [
    45101,
    45102,
    45103,
    45104,
    45105,
]

# Exact W44 tensor. Conceptual comments there were sagittal/coronal/axial,
# while the underlying W43 cache IDs were axial/coronal/sagittal.
# Do not "correct" this in W45 without new CV.
PLANE_PRIOR = torch.tensor(
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
PLANE_PRIOR_LOG = torch.log(PLANE_PRIOR.clamp_min(1e-3))


# =============================================================================
# 3. PATHOLOGY TEXT PROMPTS — EXACT W44 CONTENT
# =============================================================================

PROMPTS: Dict[str, Dict[str, List[str]]] = {
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


# =============================================================================
# 4. GENERIC HELPERS
# =============================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_uid_hash(uid: str) -> str:
    return hashlib.md5(str(uid).encode("utf-8")).hexdigest()


def sha256_file(
    path: Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
            default=convert,
        ),
        encoding="utf-8",
    )


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float) > 0.5

    values = series.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "yes", "y", "t"}
    false_values = {
        "0",
        "false",
        "no",
        "n",
        "f",
        "",
        "nan",
        "none",
    }

    unexpected = set(values.unique()) - true_values - false_values
    if unexpected:
        raise RuntimeError(f"Unexpected boolean-like values: {sorted(unexpected)}")

    return values.isin(true_values)


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    frame = assignments[[UID_COLUMN, "OuterFold"]].copy()
    frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
    frame["OuterFold"] = frame["OuterFold"].astype(int)
    frame = frame.sort_values(UID_COLUMN).reset_index(drop=True)

    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(
            frame[UID_COLUMN],
            frame["OuterFold"],
        )
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _script_project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def kaggle_dataset_roots() -> List[Path]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []

    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and child.name != "competitions"
    )


def bounded_glob(
    root: Path,
    patterns: Sequence[str],
) -> List[Path]:
    output: List[Path] = []

    if not root.exists():
        return output

    for pattern in patterns:
        try:
            output.extend(root.glob(pattern))
        except Exception:
            pass

    return output


def is_kaggle_input_path(path: Path) -> bool:
    try:
        return str(path.resolve()).startswith("/kaggle/input/")
    except Exception:
        return str(path).startswith("/kaggle/input/")


def choose_device(accelerator: str) -> torch.device:
    if accelerator == "apple_mps":
        raise RuntimeError("apple_mps is not implemented for W45.")

    if accelerator == "cpu":
        return torch.device("cpu")

    if accelerator == "kaggle_t4":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--accelerator kaggle_t4 requested " "but CUDA is unavailable."
            )
        return torch.device("cuda:0")

    if accelerator in {"localGPU", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    raise ValueError(accelerator)


def native_cuda_bf16_supported(
    device_index: int,
) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device_index)
    return int(major) >= 8


def runtime_autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()

    dtype = (
        torch.bfloat16
        if native_cuda_bf16_supported(device.index or 0)
        else torch.float16
    )

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


# =============================================================================
# 5. DISCOVERY
# =============================================================================


def discover_w40_root(
    project_root: Path,
) -> Optional[Path]:
    required = [
        "16_final_probabilities_wide.csv",
        "17_teacher_weights_wide.csv",
        "18_teacher_mask_wide.csv",
    ]

    candidates: List[Path] = [
        project_root / "output" / "results" / "rsna_w40_fs2_production_teacher_v1"
    ]

    patterns = [
        "rsna_w40_fs2_production_teacher_v1",
        "*/rsna_w40_fs2_production_teacher_v1",
        "*/*/rsna_w40_fs2_production_teacher_v1",
        "*/*/*/rsna_w40_fs2_production_teacher_v1",
        "*/*/*/*/rsna_w40_fs2_production_teacher_v1",
    ]

    validation_patterns = [
        "results/20_validation_summary.json",
        "*/results/20_validation_summary.json",
        "*/*/results/20_validation_summary.json",
        "*/*/*/results/20_validation_summary.json",
        "*/*/*/*/results/20_validation_summary.json",
    ]

    for root in kaggle_dataset_roots():
        candidates.extend(bounded_glob(root, patterns))

        for path in bounded_glob(
            root,
            validation_patterns,
        ):
            candidates.append(path.parent.parent)

    valid = []
    for candidate in candidates:
        result_dir = candidate / "results"
        if all((result_dir / name).is_file() for name in required):
            valid.append(candidate.resolve())

    if not valid:
        return None

    return sorted(
        set(valid),
        key=lambda p: (
            0 if p.name == "rsna_w40_fs2_production_teacher_v1" else 1,
            len(str(p)),
            str(p),
        ),
    )[0]


def discover_fold_csv(
    project_root: Path,
) -> Optional[Path]:
    candidates = [
        project_root
        / "output"
        / "results"
        / "rsna_w2_3"
        / "results"
        / "00_outer_fold_assignments.csv"
    ]

    patterns = [
        "00_outer_fold_assignments.csv",
        "results/00_outer_fold_assignments.csv",
        "*/00_outer_fold_assignments.csv",
        "*/results/00_outer_fold_assignments.csv",
        "*/*/00_outer_fold_assignments.csv",
        "*/*/results/00_outer_fold_assignments.csv",
        "*/*/*/00_outer_fold_assignments.csv",
        "*/*/*/results/00_outer_fold_assignments.csv",
        "*/*/*/*/results/00_outer_fold_assignments.csv",
    ]

    for root in kaggle_dataset_roots():
        candidates.extend(bounded_glob(root, patterns))

    files = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda p: (
            len(str(p)),
            str(p),
        ),
    )

    return files[0] if files else None


def _cache_root_has_version(
    root: Path,
    expected_version: str,
) -> bool:
    studies = root / "studies"

    if not studies.is_dir():
        return False

    sample = next(
        iter(studies.glob("*.npz")),
        None,
    )

    if sample is None:
        return False

    try:
        with np.load(
            sample,
            allow_pickle=False,
        ) as payload:
            return str(payload["cache_version"].item()) == expected_version
    except Exception:
        return False


def discover_train_mi2_cache_root(
    project_root: Path,
) -> Optional[Path]:
    candidates: List[Path] = [
        project_root
        / "rsna_w44_multimodel_medical_embedding_fusion_v3"
        / "embedding_cache"
        / "mi2",
        project_root
        / "output"
        / "results"
        / "rsna_w44_multimodel_medical_embedding_fusion_v3"
        / "embedding_cache"
        / "mi2",
    ]

    patterns = [
        "embedding_cache/mi2",
        "*/embedding_cache/mi2",
        "*/*/embedding_cache/mi2",
        "*/*/*/embedding_cache/mi2",
        "*/*/*/*/embedding_cache/mi2",
        "mi2",
        "*/mi2",
        "*/*/mi2",
        "*/*/*/mi2",
        "*/*/*/*/mi2",
    ]

    for root in kaggle_dataset_roots():
        candidates.extend(bounded_glob(root, patterns))

    valid = [
        candidate.resolve()
        for candidate in candidates
        if _cache_root_has_version(
            candidate,
            TRAIN_MI2_CACHE_VERSION,
        )
    ]

    if not valid:
        return None

    return sorted(
        set(valid),
        key=lambda p: (
            0 if "w44" in str(p).lower() else 1,
            len(str(p)),
            str(p),
        ),
    )[0]


def discover_test_mi2_cache_root(
    project_root: Path,
) -> Optional[Path]:
    patterns = [
        "test_embedding_cache/mi2",
        "*/test_embedding_cache/mi2",
        "*/*/test_embedding_cache/mi2",
        "*/*/*/test_embedding_cache/mi2",
        "*/*/*/*/test_embedding_cache/mi2",
    ]

    candidates: List[Path] = []

    for root in kaggle_dataset_roots():
        candidates.extend(bounded_glob(root, patterns))

    valid = [
        candidate.resolve()
        for candidate in candidates
        if _cache_root_has_version(
            candidate,
            TEST_MI2_CACHE_VERSION,
        )
    ]

    if not valid:
        return None

    return sorted(
        set(valid),
        key=lambda p: (
            len(str(p)),
            str(p),
        ),
    )[0]


def discover_mi2_root(
    project_root: Path,
) -> Optional[Path]:
    candidates: List[Path] = [
        project_root / "models" / "MedImageInsights",
        project_root / "models" / "medimageinsight",
    ]

    patterns = [
        "MedImageInsights",
        "*/MedImageInsights",
        "*/*/MedImageInsights",
        "*/*/*/MedImageInsights",
        "*/*/*/*/MedImageInsights",
        "medimageinsight",
        "*/medimageinsight",
        "*/*/medimageinsight",
        "*/*/*/medimageinsight",
    ]

    for root in kaggle_dataset_roots():
        candidates.extend(bounded_glob(root, patterns))

    valid = []

    for root in candidates:
        if (
            (root / "MedImageInsight" / "UniCLModel.py").is_file()
            and (root / "2024.09.27" / "config.yaml").is_file()
            and (
                root / "2024.09.27" / "vision_model" / "medimageinsigt-v1.0.0.pt"
            ).is_file()
            and (
                root / "2024.09.27" / "language_model" / "clip_tokenizer_4.16.2"
            ).is_dir()
        ):
            valid.append(root.resolve())

    if not valid:
        return None

    return sorted(
        set(valid),
        key=lambda p: (
            len(str(p)),
            str(p),
        ),
    )[0]


# =============================================================================
# 6. PATHS
# =============================================================================


@dataclass
class Paths:
    project_root: Path
    data_root: Path

    train_csv: Path

    w40_root: Path
    fold_csv: Path

    train_mi2_cache_root: Optional[Path]
    mi2_root: Optional[Path]

    output_root: Path
    result_root: Path
    checkpoint_root: Path

    test_csv: Path
    test_series_csv: Path
    test_series_root: Path
    sample_submission_csv: Path

    test_mi2_cache_root: Path

    @classmethod
    def discover(
        cls,
        args,
    ) -> "Paths":
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
                "/kaggle/input/competitions/" "rsna-knee-abnormality-detection"
            )
        else:
            data_root = project_root / "input"

        if args.output_root:
            output_root = Path(args.output_root).expanduser().resolve()
        elif is_kaggle:
            output_root = Path("/kaggle/working") / OUTPUT_DIR_NAME
        else:
            output_root = project_root / "output" / "results" / OUTPUT_DIR_NAME

        if args.w40_root:
            w40_root = Path(args.w40_root).expanduser().resolve()
        else:
            w40_root = discover_w40_root(project_root) or (
                project_root
                / "output"
                / "results"
                / "rsna_w40_fs2_production_teacher_v1"
            )

        if args.fold_csv:
            fold_csv = Path(args.fold_csv).expanduser().resolve()
        else:
            fold_csv = discover_fold_csv(project_root) or (
                project_root / "00_outer_fold_assignments.csv"
            )

        train_mi2_cache_root = (
            Path(args.train_mi2_cache_root).expanduser().resolve()
            if args.train_mi2_cache_root
            else discover_train_mi2_cache_root(project_root)
        )

        mi2_root = (
            Path(args.mi2_root).expanduser().resolve()
            if args.mi2_root
            else discover_mi2_root(project_root)
        )

        if args.test_mi2_cache_root:
            test_mi2_cache_root = Path(args.test_mi2_cache_root).expanduser().resolve()
        else:
            discovered_test = discover_test_mi2_cache_root(project_root)
            test_mi2_cache_root = (
                discovered_test
                if discovered_test is not None
                else (output_root / "test_embedding_cache" / "mi2")
            )

        return cls(
            project_root=project_root,
            data_root=data_root,
            train_csv=(data_root / "train.csv"),
            w40_root=w40_root,
            fold_csv=fold_csv,
            train_mi2_cache_root=(train_mi2_cache_root),
            mi2_root=mi2_root,
            output_root=output_root,
            result_root=(output_root / "results"),
            checkpoint_root=(output_root / "checkpoints"),
            test_csv=(data_root / "test.csv"),
            test_series_csv=(data_root / "test_series.csv"),
            test_series_root=(data_root / "test_series"),
            sample_submission_csv=(data_root / "sample_submission.csv"),
            test_mi2_cache_root=(test_mi2_cache_root),
        )

    def ensure_dirs(self) -> None:
        self.result_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.checkpoint_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not is_kaggle_input_path(self.test_mi2_cache_root):
            (self.test_mi2_cache_root / "studies").mkdir(
                parents=True,
                exist_ok=True,
            )


def paths_to_dict(
    paths: Paths,
) -> Dict[str, Optional[str]]:
    output: Dict[
        str,
        Optional[str],
    ] = {}

    for field in dataclasses.fields(Paths):
        value = getattr(
            paths,
            field.name,
        )
        output[field.name] = str(value) if value is not None else None

    return output


def paths_from_dict(
    payload: Mapping[
        str,
        Optional[str],
    ],
) -> Paths:
    kwargs = {}

    for field in dataclasses.fields(Paths):
        value = payload.get(field.name)
        kwargs[field.name] = Path(value) if value is not None else None

    return Paths(**kwargs)


# =============================================================================
# 7. TRAIN / W40 / FOLD VALIDATION
# =============================================================================


def load_train(
    paths: Paths,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not paths.train_csv.is_file():
        raise FileNotFoundError(paths.train_csv)

    train = pd.read_csv(paths.train_csv)

    train[UID_COLUMN] = train[UID_COLUMN].astype(str)

    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} " f"train studies; got {len(train)}"
        )

    missing = [label for label in LABELS if label not in train.columns]

    if missing:
        raise RuntimeError(f"train.csv missing labels: " f"{missing}")

    gold_mask = train[LABELS].notna().all(axis=1)

    unlabeled_mask = train[LABELS].isna().all(axis=1)

    partial = ~(gold_mask | unlabeled_mask)

    if partial.any():
        raise RuntimeError(f"Partially-labelled studies: " f"{int(partial.sum())}")

    gold = train.loc[gold_mask].copy().reset_index(drop=True)

    unlabeled = train.loc[unlabeled_mask].copy().reset_index(drop=True)

    if len(gold) != EXPECTED_GOLD or len(unlabeled) != EXPECTED_UNLABELED:
        raise RuntimeError(
            "Unexpected gold/unlabeled split: " f"{len(gold)}/{len(unlabeled)}"
        )

    for label in LABELS:
        gold[label] = pd.to_numeric(
            gold[label],
            errors="raise",
        ).astype(np.float32)

    return train, gold, unlabeled


def load_folds(
    paths: Paths,
    gold: pd.DataFrame,
) -> pd.DataFrame:
    if not paths.fold_csv.is_file():
        raise FileNotFoundError(paths.fold_csv)

    folds = pd.read_csv(paths.fold_csv)

    folds[UID_COLUMN] = folds[UID_COLUMN].astype(str)

    if "OuterFold" not in folds.columns:
        raise RuntimeError("Fold CSV missing OuterFold.")

    folds = folds[
        [
            UID_COLUMN,
            "OuterFold",
        ]
    ].copy()

    folds["OuterFold"] = folds["OuterFold"].astype(int)

    if set(folds[UID_COLUMN]) != set(gold[UID_COLUMN]):
        raise RuntimeError("Locked fold UID set does not " "match gold 58.")

    digest = fold_assignment_sha256(folds)

    if digest != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Locked fold SHA mismatch:\n"
            f"  observed={digest}\n"
            f"  expected={EXPECTED_FOLD_SHA256}"
        )

    counts = folds["OuterFold"].value_counts().sort_index().to_dict()

    expected = {
        1: 11,
        2: 12,
        3: 11,
        4: 11,
        5: 13,
    }

    if counts != expected:
        raise RuntimeError(f"Fold counts mismatch: {counts}")

    return folds


def w40_result(
    paths: Paths,
    name: str,
) -> Path:
    return paths.w40_root / "results" / name


def load_w40_teacher(
    paths: Paths,
    unlabeled: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    probability_path = w40_result(
        paths,
        "16_final_probabilities_wide.csv",
    )

    weight_path = w40_result(
        paths,
        "17_teacher_weights_wide.csv",
    )

    mask_path = w40_result(
        paths,
        "18_teacher_mask_wide.csv",
    )

    validation_path = w40_result(
        paths,
        "20_validation_summary.json",
    )

    for required in (
        probability_path,
        weight_path,
        mask_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))

        if validation.get("overall_pass") is not True:
            raise RuntimeError("W40 validation is not " "overall_pass=true.")

    probabilities = pd.read_csv(probability_path)
    weights = pd.read_csv(weight_path)
    masks = pd.read_csv(mask_path)

    for frame, name in (
        (probabilities, "probabilities"),
        (weights, "weights"),
        (masks, "masks"),
    ):
        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)

        if frame[UID_COLUMN].duplicated().any():
            raise RuntimeError(f"W40 {name} contains " "duplicate UIDs.")

        if set(frame[UID_COLUMN]) != set(unlabeled[UID_COLUMN]):
            raise RuntimeError(
                f"W40 {name} UID set " "does not match 4,349 " "unlabeled studies."
            )

    probabilities = (
        probabilities.set_index(UID_COLUMN).reindex(unlabeled[UID_COLUMN]).reset_index()
    )

    weights = weights.set_index(UID_COLUMN).reindex(unlabeled[UID_COLUMN]).reset_index()

    masks = masks.set_index(UID_COLUMN).reindex(unlabeled[UID_COLUMN]).reset_index()

    for label in LABELS:
        probabilities[label] = pd.to_numeric(
            probabilities[label],
            errors="raise",
        ).astype(np.float32)

        weights[label] = pd.to_numeric(
            weights[label],
            errors="raise",
        ).astype(np.float32)

        masks[label] = coerce_bool_series(masks[label])

    p = probabilities[LABELS].to_numpy(np.float64)

    w = weights[LABELS].to_numpy(np.float64)

    m = masks[LABELS].to_numpy(bool)

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("Invalid W40 probabilities.")

    if not np.isfinite(w).all() or ((w < 0) | (w > 1)).any():
        raise RuntimeError("Invalid W40 weights.")

    selected = int(m.sum())

    if selected != EXPECTED_SELECTED_CELLS:
        raise RuntimeError(
            "W40 selected-cell mismatch: " f"{selected}/" f"{EXPECTED_SELECTED_CELLS}"
        )

    return (
        probabilities,
        weights,
        masks,
        {
            "rows": len(probabilities),
            "selected_cells": selected,
            "probability_sha256": sha256_file(probability_path),
            "weight_sha256": sha256_file(weight_path),
            "mask_sha256": sha256_file(mask_path),
            "validation_pass": True,
        },
    )


# =============================================================================
# 8. TRAIN MI2 CACHE
# =============================================================================


def train_mi2_cache_path(
    root: Path,
    uid: str,
) -> Path:
    return root / "studies" / f"{stable_uid_hash(uid)}.npz"


def train_mi2_cache_file_usable(
    path: Path,
    uid: str,
) -> bool:
    if not path.is_file():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as payload:
            if str(payload["cache_version"].item()) != TRAIN_MI2_CACHE_VERSION:
                return False

            if str(payload["study_uid"].item()) != str(uid):
                return False

            if str(payload["encoder"].item()) != "mi2":
                return False

            if str(payload["input_mode"].item()) != VLM_INPUT_MODE:
                return False

            if (
                "encoder_sha256" not in payload
                or str(payload["encoder_sha256"].item()) != MI2_WEIGHT_SHA256
            ):
                return False

            embeddings = payload["embeddings"]
            semantic = payload["semantic"]

            n = len(embeddings)

            if embeddings.ndim != 2 or embeddings.shape[1] != MI2_DIM or n <= 0:
                return False

            if semantic.shape != (
                n,
                NUM_LABELS,
            ):
                return False

            for key in (
                "plane",
                "fluid",
                "fat_suppression",
                "slice_position",
            ):
                if key not in payload or len(payload[key]) != n:
                    return False

            if not np.isfinite(embeddings).all():
                return False

            if not np.isfinite(semantic).all():
                return False

            if not np.isfinite(payload["slice_position"]).all():
                return False

        return True

    except Exception:
        return False


def summarize_train_mi2_cache(
    paths: Paths,
    train: pd.DataFrame,
) -> Dict[str, Any]:
    root = paths.train_mi2_cache_root

    if root is None:
        return {
            "root": None,
            "cache_version": TRAIN_MI2_CACHE_VERSION,
            "usable_studies": 0,
            "expected_studies": len(train),
            "total_tokens": 0,
            "expected_tokens": EXPECTED_TRAIN_MI2_TOKENS,
            "complete": False,
            "missing_examples": train[UID_COLUMN].head(10).tolist(),
            "invalid_examples": [],
        }

    usable = 0
    tokens = 0
    missing = []
    invalid = []

    for uid in train[UID_COLUMN].astype(str):
        path = train_mi2_cache_path(
            root,
            uid,
        )

        if not path.is_file():
            if len(missing) < 10:
                missing.append(uid)
            continue

        if not train_mi2_cache_file_usable(
            path,
            uid,
        ):
            if len(invalid) < 10:
                invalid.append(uid)
            continue

        usable += 1

        with np.load(
            path,
            allow_pickle=False,
        ) as payload:
            tokens += int(len(payload["embeddings"]))

    return {
        "root": str(root),
        "cache_version": TRAIN_MI2_CACHE_VERSION,
        "encoder_sha256": MI2_WEIGHT_SHA256,
        "usable_studies": int(usable),
        "expected_studies": int(len(train)),
        "total_tokens": int(tokens),
        "expected_tokens": EXPECTED_TRAIN_MI2_TOKENS,
        "token_count_matches": int(tokens) == EXPECTED_TRAIN_MI2_TOKENS,
        "complete": bool(usable == len(train) and tokens == EXPECTED_TRAIN_MI2_TOKENS),
        "missing_examples": missing,
        "invalid_examples": invalid,
    }


def select_balanced_indices(
    plane: np.ndarray,
    n: int,
    max_tokens: int,
) -> np.ndarray:
    if n <= max_tokens:
        return np.arange(
            n,
            dtype=np.int64,
        )

    selected: List[int] = []
    per_plane = max(
        1,
        max_tokens // 3,
    )

    for plane_id in (0, 1, 2):
        indices = np.where(plane == plane_id)[0]

        if len(indices):
            take = min(
                per_plane,
                len(indices),
            )

            chosen = (
                np.linspace(
                    0,
                    len(indices) - 1,
                    take,
                )
                .round()
                .astype(int)
            )

            selected.extend(indices[chosen].tolist())

    selected = list(dict.fromkeys(selected))

    if len(selected) < max_tokens:
        used = set(selected)

        remainder = np.asarray(
            [i for i in range(n) if i not in used],
            dtype=np.int64,
        )

        need = min(
            max_tokens - len(selected),
            len(remainder),
        )

        if need:
            chosen = (
                np.linspace(
                    0,
                    len(remainder) - 1,
                    need,
                )
                .round()
                .astype(int)
            )

            selected.extend(remainder[chosen].tolist())

    selected = sorted(selected[:max_tokens])

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def load_mi2_cache_record(
    path: Path,
    uid: str,
    cache_kind: str,
) -> Dict[str, np.ndarray]:
    if cache_kind == "train":
        usable = train_mi2_cache_file_usable(
            path,
            uid,
        )
    elif cache_kind == "test":
        usable = test_mi2_cache_file_usable(
            path,
            uid,
        )
    else:
        raise ValueError(cache_kind)

    if not usable:
        raise RuntimeError(f"Invalid {cache_kind} MI2 " f"cache for {uid}: {path}")

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:
        plane = payload["plane"].astype(np.int64)

        selected = select_balanced_indices(
            plane,
            len(plane),
            MAX_VLM_TOKENS,
        )

        return {
            "embeddings": payload["embeddings"][selected].copy(),
            "semantic": payload["semantic"][selected].copy(),
            "plane": payload["plane"][selected].astype(np.int8),
            "fluid": payload["fluid"][selected].astype(np.int8),
            "fat_suppression": payload["fat_suppression"][selected].astype(np.int8),
            "slice_position": payload["slice_position"][selected].astype(np.float16),
        }


class MI2FeatureStore:
    def __init__(
        self,
        root: Path,
        uids: Sequence[str],
        cache_kind: str,
    ):
        self.records: Dict[
            str,
            Dict[
                str,
                np.ndarray,
            ],
        ] = {}

        self.cache_kind = cache_kind

        started = time.time()

        log(f"Loading {cache_kind} " "MI2 cache into CPU RAM...")

        for index, uid in enumerate(
            [str(x) for x in uids],
            start=1,
        ):
            if cache_kind == "train":
                path = train_mi2_cache_path(
                    root,
                    uid,
                )
            elif cache_kind == "test":
                path = test_mi2_cache_path(
                    root,
                    uid,
                )
            else:
                raise ValueError(cache_kind)

            self.records[uid] = load_mi2_cache_record(
                path,
                uid,
                cache_kind,
            )

            if index % 500 == 0 or index == len(uids):
                log(f"  loaded " f"{index}/{len(uids)}")

        log("MI2 FeatureStore ready " f"in {time.time()-started:.1f}s")

    def get(
        self,
        uid: str,
    ) -> Dict[str, np.ndarray]:
        return self.records[str(uid)]


# =============================================================================
# 9. TRAINING TARGETS / DATASETS
# =============================================================================


def build_target_maps(
    gold: pd.DataFrame,
    unlabeled: pd.DataFrame,
    probabilities: pd.DataFrame,
    weights: pd.DataFrame,
    masks: pd.DataFrame,
):
    gold_t = {}
    gold_m = {}
    gold_w = {}

    for _, row in gold.iterrows():
        uid = str(row[UID_COLUMN])

        gold_t[uid] = row[LABELS].to_numpy(np.float32)

        gold_m[uid] = np.ones(
            NUM_LABELS,
            dtype=bool,
        )

        gold_w[uid] = np.ones(
            NUM_LABELS,
            dtype=np.float32,
        )

    p_index = probabilities.set_index(UID_COLUMN)
    w_index = weights.set_index(UID_COLUMN)
    m_index = masks.set_index(UID_COLUMN)

    pseudo_t = {}
    pseudo_m = {}
    pseudo_w = {}

    for uid in unlabeled[UID_COLUMN].astype(str):
        pseudo_t[uid] = p_index.loc[
            uid,
            LABELS,
        ].to_numpy(np.float32)

        pseudo_m[uid] = m_index.loc[
            uid,
            LABELS,
        ].to_numpy(bool)

        pseudo_w[uid] = w_index.loc[
            uid,
            LABELS,
        ].to_numpy(np.float32)

    return (
        gold_t,
        gold_m,
        gold_w,
        pseudo_t,
        pseudo_m,
        pseudo_w,
    )


class MI2SupervisedDataset(Dataset):
    def __init__(
        self,
        store: MI2FeatureStore,
        uids: Sequence[str],
        targets: Mapping[
            str,
            np.ndarray,
        ],
        masks: Mapping[
            str,
            np.ndarray,
        ],
        weights: Mapping[
            str,
            np.ndarray,
        ],
    ):
        self.store = store
        self.uids = [str(uid) for uid in uids]
        self.targets = targets
        self.masks = masks
        self.weights = weights

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        uid = self.uids[index]

        return {
            UID_COLUMN: uid,
            "features": self.store.get(uid),
            "targets": self.targets[uid],
            "target_mask": self.masks[uid],
            "target_weight": self.weights[uid],
        }


class MI2InferenceDataset(Dataset):
    def __init__(
        self,
        store: MI2FeatureStore,
        uids: Sequence[str],
    ):
        self.store = store
        self.uids = [str(uid) for uid in uids]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        uid = self.uids[index]

        return {
            UID_COLUMN: uid,
            "features": self.store.get(uid),
        }


def pad_mi2_branch(
    batch: Sequence[Mapping[str, Any]],
) -> Dict[str, torch.Tensor]:
    records = [item["features"] for item in batch]

    max_tokens = max(len(record["embeddings"]) for record in records)

    batch_size = len(records)

    embeddings = torch.zeros(
        batch_size,
        max_tokens,
        MI2_DIM,
        dtype=torch.float32,
    )

    semantic = torch.zeros(
        batch_size,
        max_tokens,
        NUM_LABELS,
        dtype=torch.float32,
    )

    mask = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.bool,
    )

    plane = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.long,
    )

    fluid = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.long,
    )

    fat_suppression = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.long,
    )

    slice_position = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.float32,
    )

    for i, record in enumerate(records):
        n = len(record["embeddings"])

        embeddings[
            i,
            :n,
        ] = torch.from_numpy(record["embeddings"].astype(np.float32))

        semantic[
            i,
            :n,
        ] = torch.from_numpy(record["semantic"].astype(np.float32))

        mask[
            i,
            :n,
        ] = True

        plane[
            i,
            :n,
        ] = torch.from_numpy(record["plane"].astype(np.int64))

        fluid[
            i,
            :n,
        ] = torch.from_numpy(record["fluid"].astype(np.int64))

        fat_suppression[
            i,
            :n,
        ] = torch.from_numpy(record["fat_suppression"].astype(np.int64))

        slice_position[
            i,
            :n,
        ] = torch.from_numpy(record["slice_position"].astype(np.float32))

    return {
        "embeddings": embeddings,
        "semantic": semantic,
        "mask": mask,
        "plane": plane,
        "fluid": fluid,
        "fat_suppression": fat_suppression,
        "slice_position": slice_position,
    }


def collate_supervised(
    batch: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        UID_COLUMN: [str(item[UID_COLUMN]) for item in batch],
        "mi2": pad_mi2_branch(batch),
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


def collate_inference(
    batch: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        UID_COLUMN: [str(item[UID_COLUMN]) for item in batch],
        "mi2": pad_mi2_branch(batch),
    }


def move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if isinstance(
        value,
        torch.Tensor,
    ):
        return value.to(
            device,
            non_blocking=True,
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: move_to_device(
                item,
                device,
            )
            for key, item in value.items()
        }

    return value


def make_supervised_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(torch.cuda.is_available()),
        collate_fn=collate_supervised,
        drop_last=False,
    )


def make_inference_loader(
    dataset: Dataset,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(torch.cuda.is_available()),
        collate_fn=collate_inference,
        drop_last=False,
    )


# =============================================================================
# 10. EXACT W44 MI2 HEAD
# =============================================================================


def position_features(
    position: torch.Tensor,
) -> torch.Tensor:
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


class MI2BranchAggregator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_norm = nn.LayerNorm(MI2_DIM)

        self.projection = nn.Linear(
            MI2_DIM,
            HIDDEN_DIM,
        )

        self.plane_embedding = nn.Embedding(
            3,
            PLANE_EMBED_DIM,
        )

        self.fluid_embedding = nn.Embedding(
            2,
            BINARY_META_DIM,
        )

        self.fs_embedding = nn.Embedding(
            2,
            BINARY_META_DIM,
        )

        self.position_projection = nn.Linear(
            9,
            POSITION_DIM,
        )

        self.meta_projection = nn.Linear(
            PLANE_EMBED_DIM + 2 * BINARY_META_DIM + POSITION_DIM,
            HIDDEN_DIM,
        )

        self.token_norm = nn.LayerNorm(HIDDEN_DIM)

        self.queries = nn.Parameter(
            torch.randn(
                NUM_LABELS,
                HIDDEN_DIM,
            )
            * 0.02
        )

        self.dropout = nn.Dropout(HEAD_DROPOUT)

    def forward(
        self,
        branch: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        embeddings = branch["embeddings"]

        mask = branch["mask"]

        plane = branch["plane"].clamp(0, 2)

        fluid = branch["fluid"].clamp(0, 1)

        fat_suppression = branch["fat_suppression"].clamp(0, 1)

        position = branch["slice_position"]

        visual = self.projection(self.input_norm(embeddings))

        metadata = torch.cat(
            [
                self.plane_embedding(plane),
                self.fluid_embedding(fluid),
                self.fs_embedding(fat_suppression),
                self.position_projection(position_features(position)),
            ],
            dim=-1,
        )

        tokens = self.token_norm(visual + self.meta_projection(metadata))

        tokens = self.dropout(tokens)

        scores = torch.einsum(
            "bth,lh->blt",
            tokens,
            self.queries,
        ) / math.sqrt(HIDDEN_DIM)

        plane_prior = PLANE_PRIOR_LOG.to(scores.device)

        # Intentional W44 compatibility.
        scores = scores + (
            plane_prior[
                :,
                plane,
            ].permute(
                1,
                0,
                2,
            )
        )

        scores = scores.masked_fill(
            ~mask[:, None, :],
            -1e4,
        )

        attention = torch.softmax(
            scores.float(),
            dim=-1,
        ).to(tokens.dtype)

        pooled = torch.einsum(
            "blt,bth->blh",
            attention,
            tokens,
        )

        semantic = branch["semantic"]

        semantic_by_label = semantic.permute(
            0,
            2,
            1,
        )

        semantic_pooled = (attention.float() * semantic_by_label.float()).sum(dim=-1)

        return (
            pooled,
            semantic_pooled,
        )


class MI2ProductionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.mi2 = MI2BranchAggregator()

        classifier_dim = HIDDEN_DIM + 1

        self.classifier_weight = nn.Parameter(
            torch.randn(
                NUM_LABELS,
                classifier_dim,
            )
            * 0.02
        )

        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    def forward(
        self,
        batch: Mapping[
            str,
            Any,
        ],
    ) -> torch.Tensor:
        pooled, semantic = self.mi2(batch["mi2"])

        features = torch.cat(
            [
                pooled,
                semantic[
                    ...,
                    None,
                ].to(pooled.dtype),
            ],
            dim=-1,
        )

        logits = (
            features
            * self.classifier_weight[
                None,
                :,
                :,
            ]
        ).sum(dim=-1)

        logits = (
            logits
            + self.classifier_bias[
                None,
                :,
            ]
        )

        return logits


# =============================================================================
# 11. LOSSES / HEAD TRAINING
# =============================================================================


def pseudo_macro_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise RuntimeError("Pseudo logits non-finite.")

    cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
    )

    effective = mask.float() * weights.float()

    denominator = effective.sum(dim=0)

    active = denominator > 0

    if not active.any():
        raise RuntimeError("Pseudo batch has no " "active label cells.")

    per_label = (cell * effective).sum(dim=0) / denominator.clamp_min(1e-6)

    loss = per_label[active].mean()

    if not torch.isfinite(loss):
        raise RuntimeError("Pseudo loss non-finite.")

    return loss


def gold_pos_weight(
    gold: pd.DataFrame,
) -> torch.Tensor:
    y = gold[LABELS].to_numpy(np.float64)

    positives = y.sum(axis=0)
    negatives = len(y) - positives

    weights = np.clip(
        negatives
        / np.maximum(
            positives,
            1.0,
        ),
        1.0,
        5.0,
    )

    return torch.tensor(weights.astype(np.float32))


def gold_macro_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise RuntimeError("Gold logits non-finite.")

    cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
        pos_weight=pos_weight.float(),
    )

    loss = cell.mean(dim=0).mean()

    if not torch.isfinite(loss):
        raise RuntimeError("Gold loss non-finite.")

    return loss


def train_pseudo_head(
    model: MI2ProductionHead,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PSEUDO_LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    for epoch in range(
        1,
        PSEUDO_EPOCHS + 1,
    ):
        model.train()

        total = 0.0
        count = 0
        started = time.time()

        for raw in loader:
            batch = move_to_device(
                raw,
                device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)

            loss = pseudo_macro_loss(
                logits,
                batch["targets"],
                batch["target_mask"],
                batch["target_weight"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM,
            )

            optimizer.step()

            n = len(raw[UID_COLUMN])

            total += float(loss.detach().cpu()) * n

            count += n

        value = total / max(count, 1)

        history.append(
            {
                "epoch": epoch,
                "loss": value,
                "seconds": time.time() - started,
            }
        )

        log(f"pseudo epoch " f"{epoch:02d}/" f"{PSEUDO_EPOCHS} " f"loss={value:.6f}")

    return history


def adapt_gold_head(
    model: MI2ProductionHead,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> List[Dict[str, Any]]:
    model.to(device)
    pos_weight = pos_weight.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GOLD_LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    for epoch in range(
        1,
        GOLD_ADAPT_EPOCHS + 1,
    ):
        model.train()

        total = 0.0
        count = 0

        for raw in loader:
            batch = move_to_device(
                raw,
                device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)

            loss = gold_macro_loss(
                logits,
                batch["targets"],
                pos_weight,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM,
            )

            optimizer.step()

            n = len(raw[UID_COLUMN])

            total += float(loss.detach().cpu()) * n
            count += n

        value = total / max(count, 1)

        history.append(
            {
                "epoch": epoch,
                "loss": value,
            }
        )

        if epoch in {
            1,
            GOLD_ADAPT_EPOCHS,
        }:
            log(
                f"gold epoch "
                f"{epoch:02d}/"
                f"{GOLD_ADAPT_EPOCHS} "
                f"loss={value:.6f}"
            )

    return history


@torch.inference_mode()
def predict_head(
    model: MI2ProductionHead,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[
    List[str],
    np.ndarray,
]:
    model.to(device)
    model.eval()

    output_uids = []
    output_probabilities = []

    for raw in loader:
        batch = move_to_device(
            raw,
            device,
        )

        logits = model(batch)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite inference " "logits.")

        probabilities = torch.sigmoid(logits.float()).cpu().numpy()

        output_uids.extend(raw[UID_COLUMN])

        output_probabilities.append(probabilities)

    if not output_probabilities:
        raise RuntimeError("Inference loader " "produced no batches.")

    return (
        output_uids,
        np.concatenate(
            output_probabilities,
            axis=0,
        ),
    )


# =============================================================================
# 12. PRODUCTION HEAD CHECKPOINTS
# =============================================================================


def pseudo_checkpoint_path(
    paths: Paths,
) -> Path:
    return paths.checkpoint_root / "pseudo_mi2.pt"


def full_checkpoint_path(
    paths: Paths,
    seed: int,
) -> Path:
    return paths.checkpoint_root / f"full_seed_{seed}.pt"


def pseudo_config(
    train_cache: Mapping[
        str,
        Any,
    ],
    teacher: Mapping[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "scope": "mi2_pseudo_pretrain",
        "head": "w44_mi2_label_aware",
        "mi2_dim": MI2_DIM,
        "hidden_dim": HIDDEN_DIM,
        "head_dropout": HEAD_DROPOUT,
        "max_vlm_tokens": MAX_VLM_TOKENS,
        "plane_prior": PLANE_PRIOR.tolist(),
        "plane_mapping_behavior": "preserve_w44_validated_behavior",
        "train_cache_version": TRAIN_MI2_CACHE_VERSION,
        "train_cache_tokens": int(train_cache["total_tokens"]),
        "mi2_weight_sha256": MI2_WEIGHT_SHA256,
        "input_mode": VLM_INPUT_MODE,
        "prompt_hash": stable_json_hash(PROMPTS),
        "pseudo_epochs": PSEUDO_EPOCHS,
        "pseudo_batch": PSEUDO_BATCH,
        "pseudo_lr": PSEUDO_LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP_NORM,
        "pseudo_seed": PSEUDO_SEED,
        "teacher": dict(teacher),
        "fold_sha256": EXPECTED_FOLD_SHA256,
    }


def full_config(
    pseudo_config_hash: str,
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "scope": "all58_gold_adaptation",
        "pseudo_config_hash": pseudo_config_hash,
        "gold_studies": EXPECTED_GOLD,
        "gold_adapt_epochs": GOLD_ADAPT_EPOCHS,
        "gold_batch": GOLD_BATCH,
        "gold_lr": GOLD_LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP_NORM,
        "full_seeds": FULL_SEEDS,
        "seed_ensemble": "arithmetic_probability_mean",
    }


def load_checkpoint(
    path: Path,
) -> Dict[str, Any]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def checkpoint_report(
    paths: Paths,
) -> Dict[str, Any]:
    rows = []

    for seed in FULL_SEEDS:
        path = full_checkpoint_path(
            paths,
            seed,
        )

        valid = False
        error = None

        if path.is_file():
            try:
                payload = load_checkpoint(path)

                valid = bool(
                    payload.get("script_version") == SCRIPT_VERSION
                    and payload.get("scope") == "full_all58"
                    and int(
                        payload.get(
                            "seed",
                            -1,
                        )
                    )
                    == seed
                    and isinstance(
                        payload.get("model_state"),
                        dict,
                    )
                )

            except Exception as exc:
                error = repr(exc)

        rows.append(
            {
                "seed": seed,
                "path": str(path),
                "exists": path.is_file(),
                "valid": valid,
                "error": error,
            }
        )

    return {
        "expected_seeds": FULL_SEEDS,
        "valid_checkpoints": int(sum(row["valid"] for row in rows)),
        "expected_checkpoints": len(FULL_SEEDS),
        "complete": all(row["valid"] for row in rows),
        "checkpoints": rows,
    }


def run_train_full(
    paths: Paths,
    args,
) -> Dict[str, Any]:
    paths.ensure_dirs()

    train, gold, unlabeled = load_train(paths)

    # Provenance lock.
    folds = load_folds(
        paths,
        gold,
    )

    (
        probabilities,
        teacher_weights,
        teacher_masks,
        teacher_info,
    ) = load_w40_teacher(
        paths,
        unlabeled,
    )

    train_cache = summarize_train_mi2_cache(
        paths,
        train,
    )

    if not train_cache["complete"]:
        raise RuntimeError(
            "Saved W44 MI2 train cache "
            "is incomplete or has the "
            "wrong identity:\n"
            + json.dumps(
                train_cache,
                indent=2,
            )
        )

    device = choose_device(args.accelerator)

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | " "FULL PRODUCTION TRAINING")
    log("=" * 100)
    log(f"Training device       : " f"{device}")
    log(f"Train MI2 cache       : " f"{train_cache['root']}")
    log(f"MI2 train tokens      : " f"{train_cache['total_tokens']}")
    log(f"Pseudo studies        : " f"{len(unlabeled)}")
    log(f"Selected pseudo cells : " f"{teacher_info['selected_cells']}")
    log(f"Gold studies          : " f"{len(gold)}")
    log(f"Production seeds      : " f"{FULL_SEEDS}")
    log("Large MI2 encoder      : " "FROZEN / not loaded")

    store = MI2FeatureStore(
        paths.train_mi2_cache_root,
        train[UID_COLUMN].astype(str).tolist(),
        cache_kind="train",
    )

    (
        gold_t,
        gold_m,
        gold_w,
        pseudo_t,
        pseudo_m,
        pseudo_w,
    ) = build_target_maps(
        gold,
        unlabeled,
        probabilities,
        teacher_weights,
        teacher_masks,
    )

    p_config = pseudo_config(
        train_cache,
        teacher_info,
    )

    p_hash = stable_json_hash(p_config)

    p_path = pseudo_checkpoint_path(paths)

    if p_path.is_file():
        pseudo_payload = load_checkpoint(p_path)

        if pseudo_payload.get("config_hash") != p_hash:
            raise RuntimeError(
                "Existing W45 pseudo "
                "checkpoint has a "
                "different configuration. "
                "Use a new --output-root."
            )

        log("Reusable W45 pseudo " "checkpoint found.")

    else:
        seed_everything(PSEUDO_SEED)

        pseudo_model = MI2ProductionHead()

        pseudo_dataset = MI2SupervisedDataset(
            store,
            unlabeled[UID_COLUMN].astype(str).tolist(),
            pseudo_t,
            pseudo_m,
            pseudo_w,
        )

        pseudo_loader = make_supervised_loader(
            pseudo_dataset,
            PSEUDO_BATCH,
            True,
        )

        log("-" * 100)
        log("STAGE P | " "W40 PSEUDO PRETRAIN")
        log("-" * 100)

        pseudo_history = train_pseudo_head(
            pseudo_model,
            pseudo_loader,
            device,
        )

        pseudo_payload = {
            "script_version": SCRIPT_VERSION,
            "scope": "pseudo_mi2",
            "seed": PSEUDO_SEED,
            "config": p_config,
            "config_hash": p_hash,
            "model_state": {
                key: value.detach().cpu()
                for key, value in pseudo_model.state_dict().items()
            },
            "history": pseudo_history,
            "created_at": now_iso(),
        }

        atomic_torch_save(
            pseudo_payload,
            p_path,
        )

        del (
            pseudo_model,
            pseudo_loader,
            pseudo_dataset,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gold_dataset = MI2SupervisedDataset(
        store,
        gold[UID_COLUMN].astype(str).tolist(),
        gold_t,
        gold_m,
        gold_w,
    )

    position_weight = gold_pos_weight(gold)

    f_config = full_config(p_hash)

    f_hash = stable_json_hash(f_config)

    seed_summaries = []

    for seed in FULL_SEEDS:
        path = full_checkpoint_path(
            paths,
            seed,
        )

        if path.is_file():
            payload = load_checkpoint(path)

            if (
                payload.get("config_hash") != f_hash
                or int(
                    payload.get(
                        "seed",
                        -1,
                    )
                )
                != seed
            ):
                raise RuntimeError(
                    f"Existing {path} "
                    "has incompatible "
                    "production config. "
                    "Use a new output root."
                )

            log(f"seed={seed}: " "reusable full checkpoint")

            seed_summaries.append(
                {
                    "seed": seed,
                    "checkpoint": str(path),
                    "reused": True,
                    "final_gold_loss": float(payload["history"][-1]["loss"]),
                }
            )
            continue

        log("-" * 100)
        log(f"FULL ALL-58 | seed={seed}")
        log("-" * 100)

        seed_everything(seed)

        model = MI2ProductionHead()

        model.load_state_dict(
            pseudo_payload["model_state"],
            strict=True,
        )

        gold_loader = make_supervised_loader(
            gold_dataset,
            GOLD_BATCH,
            True,
        )

        history = adapt_gold_head(
            model,
            gold_loader,
            device,
            position_weight,
        )

        payload = {
            "script_version": SCRIPT_VERSION,
            "scope": "full_all58",
            "seed": seed,
            "config": f_config,
            "config_hash": f_hash,
            "pseudo_config_hash": p_hash,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "history": history,
            "created_at": now_iso(),
        }

        atomic_torch_save(
            payload,
            path,
        )

        seed_summaries.append(
            {
                "seed": seed,
                "checkpoint": str(path),
                "reused": False,
                "final_gold_loss": float(history[-1]["loss"]),
            }
        )

        del model, gold_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint_status = checkpoint_report(paths)

    if not checkpoint_status["complete"]:
        raise RuntimeError("Production checkpoint set " "is incomplete.")

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "scope": "full_all58_5seed",
        "fold_sha256": fold_assignment_sha256(folds),
        "train_cache": train_cache,
        "teacher": teacher_info,
        "pseudo_config_hash": p_hash,
        "full_config_hash": f_hash,
        "full_seeds": FULL_SEEDS,
        "seed_summaries": seed_summaries,
        "checkpoint_status": checkpoint_status,
        "overall_pass": checkpoint_status["complete"],
    }

    write_json(
        paths.result_root / "01_full_training_summary.json",
        summary,
    )

    log("=" * 100)
    log("W45 FULL TRAINING COMPLETE")
    log("=" * 100)

    for row in seed_summaries:
        log(f"seed={row['seed']} " f"final_gold_loss=" f"{row['final_gold_loss']:.6f}")

    return summary


# =============================================================================
# 13. TEST TABLES
# =============================================================================


def load_test_tables(
    paths: Paths,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    for required in (
        paths.test_csv,
        paths.test_series_csv,
        paths.test_series_root,
        paths.sample_submission_csv,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    test = pd.read_csv(paths.test_csv)

    series = pd.read_csv(paths.test_series_csv)

    sample = pd.read_csv(paths.sample_submission_csv)

    for frame, name in (
        (test, "test.csv"),
        (series, "test_series.csv"),
        (
            sample,
            "sample_submission.csv",
        ),
    ):
        if UID_COLUMN not in frame.columns:
            raise RuntimeError(f"{name} missing " f"{UID_COLUMN}")

        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)

    if SERIES_UID_COLUMN not in series.columns:
        raise RuntimeError("test_series.csv missing " "SeriesInstanceUID.")

    series[SERIES_UID_COLUMN] = series[SERIES_UID_COLUMN].astype(str)

    required_series = {
        UID_COLUMN,
        SERIES_UID_COLUMN,
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }

    missing_series = required_series - set(series.columns)

    if missing_series:
        raise RuntimeError("test_series.csv missing: " f"{sorted(missing_series)}")

    if test[UID_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate test UID.")

    if sample[UID_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate sample submission UID.")

    sample_labels = [column for column in sample.columns if column != UID_COLUMN]

    if set(sample_labels) != set(LABELS):
        raise RuntimeError(
            "sample_submission label set "
            "does not match W45 labels.\n"
            f"sample={sample_labels}\n"
            f"W45={LABELS}"
        )

    if set(sample[UID_COLUMN]) != set(test[UID_COLUMN]):
        raise RuntimeError("test.csv and " "sample_submission.csv UID " "sets differ.")

    found_planes = set(series["Anatomical_Plane"].dropna().astype(str).unique())

    bad_planes = found_planes - set(CACHE_PLANE_TO_INDEX)

    if bad_planes:
        raise RuntimeError("Unexpected anatomical " f"planes: {sorted(bad_planes)}")

    series["_fluid"] = coerce_bool_series(series["Fluid_Sensitive"]).astype(int)

    series["_fs"] = coerce_bool_series(series["Fat_Suppression"]).astype(int)

    # Exact canonical final order.
    test = sample[[UID_COLUMN]].merge(
        test,
        on=UID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    return (
        test,
        series,
        sample,
    )


# =============================================================================
# 14. W43 DICOM GEOMETRY / ORIENTATION
# =============================================================================


def unit_vector(
    vector: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = float(np.linalg.norm(vector))

    if norm < 1e-8:
        raise ValueError("Zero orientation vector.")

    return vector / norm


def array_axis_vectors_from_iop(
    image_orientation_patient,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    iop = np.asarray(
        image_orientation_patient,
        dtype=np.float64,
    )

    if iop.shape != (6,):
        raise ValueError("Expected six IOP values, " f"got {iop.shape}.")

    # pixel_array axis 0 follows
    # DICOM column direction.
    axis0 = unit_vector(iop[3:6])

    # pixel_array axis 1 follows
    # DICOM row direction.
    axis1 = unit_vector(iop[:3])

    return axis0, axis1


def geometry_plane_from_iop(
    image_orientation_patient,
) -> Tuple[str, float]:
    axis0, axis1 = array_axis_vectors_from_iop(image_orientation_patient)

    normal = unit_vector(
        np.cross(
            axis1,
            axis0,
        )
    )

    scores = {
        "Axial": abs(float(normal[2])),
        "Coronal": abs(float(normal[1])),
        "Sagittal": abs(float(normal[0])),
    }

    plane = max(
        scores,
        key=scores.get,
    )

    return (
        plane,
        float(scores[plane]),
    )


def orientation_transform_spec(
    image_orientation_patient,
    target_plane: str,
) -> Dict[str, Any]:
    if target_plane not in (PLANE_TARGET_AXES):
        raise ValueError(f"Unsupported target " f"plane: {target_plane}")

    target_letters = PLANE_TARGET_AXES[target_plane]

    target0 = LPS_VECTOR[target_letters[0]]

    target1 = LPS_VECTOR[target_letters[1]]

    current0, current1 = array_axis_vectors_from_iop(image_orientation_patient)

    identity_score = abs(
        float(
            np.dot(
                current0,
                target0,
            )
        )
    ) + abs(
        float(
            np.dot(
                current1,
                target1,
            )
        )
    )

    transpose_score = abs(
        float(
            np.dot(
                current1,
                target0,
            )
        )
    ) + abs(
        float(
            np.dot(
                current0,
                target1,
            )
        )
    )

    transpose = transpose_score > identity_score

    if transpose:
        new0 = current1.copy()
        new1 = current0.copy()
    else:
        new0 = current0.copy()
        new1 = current1.copy()

    flip0 = (
        float(
            np.dot(
                new0,
                target0,
            )
        )
        < 0
    )

    if flip0:
        new0 *= -1.0

    flip1 = (
        float(
            np.dot(
                new1,
                target1,
            )
        )
        < 0
    )

    if flip1:
        new1 *= -1.0

    alignment0 = float(
        np.dot(
            unit_vector(new0),
            target0,
        )
    )

    alignment1 = float(
        np.dot(
            unit_vector(new1),
            target1,
        )
    )

    return {
        "target_plane": target_plane,
        "target_orientation": "".join(target_letters),
        "transpose": bool(transpose),
        "flip0": bool(flip0),
        "flip1": bool(flip1),
        "alignment0": alignment0,
        "alignment1": alignment1,
        "min_alignment": min(
            alignment0,
            alignment1,
        ),
    }


def apply_orientation_transform(
    image: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    result = np.asarray(image)

    if spec["transpose"]:
        result = result.T

    if spec["flip0"]:
        result = np.flip(
            result,
            axis=0,
        )

    if spec["flip1"]:
        result = np.flip(
            result,
            axis=1,
        )

    return np.ascontiguousarray(result)


def scalar_slice_position(
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

        column = np.asarray(
            orientation[3:],
            dtype=np.float64,
        )

        normal = np.cross(
            row,
            column,
        )

        return float(
            np.dot(
                np.asarray(
                    position,
                    dtype=np.float64,
                ),
                normal,
            )
        )

    except Exception:
        return None


def read_series_headers(
    series_root: Path,
    study_uid: str,
    series_uid: str,
) -> List[Dict[str, Any]]:
    if pydicom is None:
        raise RuntimeError("pydicom is required.")

    series_dir = series_root / str(study_uid) / str(series_uid)

    dicom_paths = sorted(series_dir.glob("*.dcm"))

    records: List[Dict[str, Any]] = []

    for path in dicom_paths:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )

            iop = getattr(
                ds,
                "ImageOrientationPatient",
                None,
            )

            records.append(
                {
                    "path": str(path),
                    "position": scalar_slice_position(ds),
                    "instance": getattr(
                        ds,
                        "InstanceNumber",
                        0,
                    ),
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


def decode_dicom_pixel_array(
    ds,
    dicom_path: str,
) -> np.ndarray:
    """
    W43 safe native-DICOM repair path.

    Compressed transfer syntaxes are never
    guessed or rewritten.
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
                if (transfer_syntax is not None)
                else False
            )
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

        def restore_metadata():
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

            # Repair NumberOfFrames.
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
                            return repaired

                        restore_metadata()

            # Repair native MONOCHROME
            # storage width.
            restore_metadata()

            if not photometric.startswith("MONOCHROME"):
                raise first_error

            base_pixels = rows * columns * samples

            usable_bytes = actual_bytes

            if (
                base_pixels > 0
                and usable_bytes % base_pixels != 0
                and usable_bytes > 0
                and (usable_bytes - 1) % base_pixels == 0
            ):
                usable_bytes -= 1

            if base_pixels <= 0 or usable_bytes <= 0 or usable_bytes % base_pixels != 0:
                raise first_error

            inferred_bits_allocated = (usable_bytes // base_pixels) * 8

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
                "DICOM pixel decode failed "
                "and safe native metadata "
                "repair was not possible: "
                f"{dicom_path}. "
                f"Original={first_error}; "
                f"repair={repair_error}"
            ) from first_error


def normalized_slice_position(
    records: List[Dict[str, Any]],
    index: int,
) -> float:
    positions = [item["position"] for item in records]

    if positions and all(value is not None for value in positions):
        values = np.asarray(
            positions,
            dtype=np.float64,
        )

        low = float(np.min(values))

        high = float(np.max(values))

        if np.isfinite(low) and np.isfinite(high) and high > low:
            return float(2.0 * ((values[index] - low) / (high - low)) - 1.0)

    if len(records) <= 1:
        return 0.0

    return float(2.0 * (index / (len(records) - 1)) - 1.0)


def resize_uint8(
    image: np.ndarray,
    size: int = SHARED_CACHE_IMAGE_SIZE,
) -> np.ndarray:
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if max(image.shape) >= size else cv2.INTER_LINEAR

        return cv2.resize(
            image,
            (size, size),
            interpolation=interpolation,
        )

    tensor = torch.from_numpy(image.astype(np.float32))[None, None]

    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return resized.clamp(0, 255).round().byte().numpy()


def robust_triplet_to_uint8(
    images: Sequence[np.ndarray],
) -> np.ndarray:
    if len(images) != 3:
        raise ValueError("2.5D triplet requires " "three slices.")

    stack = np.stack(
        [
            np.asarray(
                image,
                dtype=np.float32,
            )
            for image in images
        ],
        axis=0,
    )

    finite = stack[np.isfinite(stack)]

    if finite.size == 0:
        raise RuntimeError("MRI triplet has no " "finite pixels.")

    low, high = np.percentile(
        finite,
        [1.0, 99.0],
    )

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))

    if high <= low:
        scaled = np.zeros_like(
            stack,
            dtype=np.uint8,
        )
    else:
        scaled = np.clip(
            (stack - low) / (high - low),
            0.0,
            1.0,
        )

        scaled = np.round(scaled * 255.0).astype(np.uint8)

    return np.stack(
        [resize_uint8(channel) for channel in scaled],
        axis=0,
    )


def decode_record(
    records: List[Dict[str, Any]],
    target_index: int,
    orientation_spec: Optional[Dict[str, Any]],
) -> Tuple[
    np.ndarray,
    Any,
    int,
]:
    if pydicom is None:
        raise RuntimeError("pydicom is required.")

    n = len(records)

    candidates = [target_index]

    for radius in range(
        1,
        min(8, n),
    ):
        candidates.extend(
            [
                target_index - radius,
                target_index + radius,
            ]
        )

    last_error = None

    for index in candidates:
        if index < 0 or index >= n:
            continue

        path = records[index]["path"]

        try:
            ds = pydicom.dcmread(
                path,
                force=True,
            )

            image = decode_dicom_pixel_array(
                ds,
                path,
            )

            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]

            if image.ndim != 2:
                raise RuntimeError("Expected 2-D MRI " f"slice, got {image.shape}")

            image = image.astype(np.float32)

            slope = float(
                getattr(
                    ds,
                    "RescaleSlope",
                    1.0,
                )
                or 1.0
            )

            intercept = float(
                getattr(
                    ds,
                    "RescaleIntercept",
                    0.0,
                )
                or 0.0
            )

            image = image * slope + intercept

            spec = orientation_spec

            if spec is None:
                iop = getattr(
                    ds,
                    "ImageOrientationPatient",
                    None,
                )

                if iop is not None:
                    (
                        plane,
                        confidence,
                    ) = geometry_plane_from_iop(iop)

                    if confidence >= GEOMETRY_PLANE_CONFIDENCE:
                        spec = orientation_transform_spec(
                            iop,
                            plane,
                        )

            if spec is not None:
                image = apply_orientation_transform(
                    image,
                    spec,
                )

            return (
                image,
                ds,
                index,
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "No decodable slice near " f"index {target_index}. " f"Last error={last_error}"
    )


def series_centers(
    n_slices: int,
) -> List[int]:
    if n_slices <= 0:
        return []

    if n_slices == 1:
        return [0]

    quantiles = np.linspace(
        0.10,
        0.90,
        STACKS_PER_SERIES,
    )

    indices = [int(round(q * (n_slices - 1))) for q in quantiles]

    unique = []

    for index in indices:
        index = int(
            np.clip(
                index,
                0,
                n_slices - 1,
            )
        )

        if index not in unique:
            unique.append(index)

    return unique


def build_study_2p5d_arrays(
    series_root: Path,
    study_uid: str,
    study_series: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    study_series = study_series.copy()

    study_series["_plane_idx"] = study_series["Anatomical_Plane"].map(
        CACHE_PLANE_TO_INDEX
    )

    study_series = study_series.sort_values(
        [
            "_plane_idx",
            "_fluid",
            "_fs",
            SERIES_UID_COLUMN,
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    images = []
    plane_ids = []
    fluid_ids = []
    fs_ids = []
    positions = []

    for _, row in study_series.iterrows():
        series_uid = str(row[SERIES_UID_COLUMN])

        metadata_plane = str(row["Anatomical_Plane"])

        fluid = int(row["_fluid"])

        fat_suppression = int(row["_fs"])

        records = read_series_headers(
            series_root,
            study_uid,
            series_uid,
        )

        if not records:
            continue

        valid_iop = next(
            (item["iop"] for item in records if item["iop"] is not None),
            None,
        )

        plane_used = metadata_plane

        orientation_spec = None

        if valid_iop is not None:
            (
                geometry_plane,
                geometry_confidence,
            ) = geometry_plane_from_iop(valid_iop)

            if (
                geometry_plane != metadata_plane
                and geometry_confidence >= GEOMETRY_PLANE_CONFIDENCE
            ):
                plane_used = geometry_plane

            orientation_spec = orientation_transform_spec(
                valid_iop,
                plane_used,
            )

        for center in series_centers(len(records)):
            try:
                triplet = []
                actual_center = center

                for offset in STACK_OFFSETS:
                    target = int(
                        np.clip(
                            center + offset,
                            0,
                            len(records) - 1,
                        )
                    )

                    (
                        image,
                        _ds,
                        actual,
                    ) = decode_record(
                        records,
                        target,
                        orientation_spec,
                    )

                    triplet.append(image)

                    if offset == 0:
                        actual_center = actual

                stack = robust_triplet_to_uint8(triplet)

                images.append(stack)

                plane_ids.append(int(CACHE_PLANE_TO_INDEX[plane_used]))

                fluid_ids.append(fluid)

                fs_ids.append(fat_suppression)

                positions.append(
                    float(
                        normalized_slice_position(
                            records,
                            actual_center,
                        )
                    )
                )

            except Exception:
                # Exact W43 behavior:
                # individual stack failure
                # does not kill a study.
                continue

    if not images:
        raise RuntimeError(f"Study {study_uid} " "produced no valid " "2.5D stacks.")

    return {
        "images": np.stack(
            images,
            axis=0,
        ).astype(np.uint8),
        "plane": np.asarray(
            plane_ids,
            dtype=np.int8,
        ),
        "fluid": np.asarray(
            fluid_ids,
            dtype=np.int8,
        ),
        "fat_suppression": np.asarray(
            fs_ids,
            dtype=np.int8,
        ),
        "slice_position": np.asarray(
            positions,
            dtype=np.float32,
        ),
    }


# =============================================================================
# 15. DICOM PREFLIGHT
# =============================================================================


def first_dicom_in_series(
    series_root: Path,
    study_uid: str,
    series_uid: str,
) -> Optional[Path]:
    series_dir = series_root / str(study_uid) / str(series_uid)

    files = sorted(series_dir.glob("*.dcm"))

    return files[0] if files else None


def inspect_dicom_decoder_support(
    series_root: Path,
    series: pd.DataFrame,
    max_series: int,
) -> Dict[str, Any]:
    if pydicom is None:
        return {
            "overall_pass": False,
            "error": "pydicom is not installed.",
            "install_command": DICOM_DECODER_INSTALL_COMMAND,
        }

    universe = (
        series[
            [
                UID_COLUMN,
                SERIES_UID_COLUMN,
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                UID_COLUMN,
                SERIES_UID_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )

    if len(universe) <= max_series:
        sample = universe.copy()
    else:
        indices = np.linspace(
            0,
            len(universe) - 1,
            num=max_series,
            dtype=np.int64,
        )

        sample = universe.iloc[np.unique(indices)].reset_index(drop=True)

    syntax_counts = {}
    syntax_names = {}
    representative = {}

    missing_series_dirs = []
    header_failures = []

    for row in sample.itertuples(index=False):
        study_uid = str(
            getattr(
                row,
                UID_COLUMN,
            )
        )

        series_uid = str(
            getattr(
                row,
                SERIES_UID_COLUMN,
            )
        )

        path = first_dicom_in_series(
            series_root,
            study_uid,
            series_uid,
        )

        if path is None:
            missing_series_dirs.append(
                {
                    UID_COLUMN: study_uid,
                    SERIES_UID_COLUMN: series_uid,
                }
            )
            continue

        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )

            ts = getattr(
                getattr(
                    ds,
                    "file_meta",
                    None,
                ),
                "TransferSyntaxUID",
                None,
            )

            key = str(ts) if ts is not None else "UNKNOWN"

            syntax_counts[key] = (
                syntax_counts.get(
                    key,
                    0,
                )
                + 1
            )

            try:
                syntax_names[key] = str(ts.name)
            except Exception:
                syntax_names[key] = str(ts)

            representative.setdefault(
                key,
                str(path),
            )

        except Exception as exc:
            header_failures.append(
                {
                    "path": str(path),
                    "error": repr(exc),
                }
            )

    decode_results = {}

    for key, path in representative.items():
        try:
            ds = pydicom.dcmread(
                path,
                force=True,
            )

            array = decode_dicom_pixel_array(
                ds,
                path,
            )

            array = np.asarray(array)

            decode_results[key] = {
                "transfer_syntax_uid": key,
                "transfer_syntax_name": syntax_names.get(key),
                "path": path,
                "decoded": True,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "error": None,
            }

        except Exception as exc:
            decode_results[key] = {
                "transfer_syntax_uid": key,
                "transfer_syntax_name": syntax_names.get(key),
                "path": path,
                "decoded": False,
                "shape": None,
                "dtype": None,
                "error": repr(exc),
            }

    failed = [
        value for value in decode_results.values() if value.get("decoded") is not True
    ]

    overall = bool(
        len(sample) > 0
        and not header_failures
        and not missing_series_dirs
        and bool(decode_results)
        and not failed
    )

    return {
        "sampled_series": int(len(sample)),
        "series_with_representative_dicom": int(sum(syntax_counts.values())),
        "transfer_syntax_counts": syntax_counts,
        "transfer_syntax_names": syntax_names,
        "decode_results": decode_results,
        "failed_transfer_syntax_count": int(len(failed)),
        "failed_transfer_syntaxes": failed,
        "missing_series_directory_count": int(len(missing_series_dirs)),
        "missing_series_directory_examples": missing_series_dirs[:10],
        "header_failure_count": int(len(header_failures)),
        "header_failure_examples": header_failures[:10],
        "install_command": DICOM_DECODER_INSTALL_COMMAND,
        "overall_pass": overall,
    }


def require_dicom_preflight(
    paths: Paths,
    series: pd.DataFrame,
) -> Dict[str, Any]:
    audit = inspect_dicom_decoder_support(
        paths.test_series_root,
        series,
        DICOM_PREFLIGHT_SERIES_EXTRACT,
    )

    if audit.get("overall_pass") is True:
        return audit

    missing = int(
        audit.get(
            "missing_series_directory_count",
            0,
        )
    )

    if missing:
        raise RuntimeError(
            "Hidden-test DICOM tree "
            "is incomplete.\n"
            f"Missing sampled series: "
            f"{missing}\n"
            f"Root: "
            f"{paths.test_series_root}\n"
            "Use the full Kaggle "
            "competition mount."
        )

    failed = audit.get(
        "failed_transfer_syntaxes",
        [],
    )

    if failed:
        raise RuntimeError(
            "DICOM codec preflight "
            "failed with the full "
            "test tree present.\n"
            + json.dumps(
                failed,
                indent=2,
            )
            + "\nInstall:\n  "
            + DICOM_DECODER_INSTALL_COMMAND
        )

    raise RuntimeError(
        "DICOM preflight failed:\n"
        + json.dumps(
            audit,
            indent=2,
        )
    )


# =============================================================================
# 16. MEDIMAGEINSIGHT
# =============================================================================


def mi2_asset_report(
    root: Optional[Path],
) -> Dict[str, Any]:
    if root is None:
        return {
            "root": None,
            "complete": False,
            "expected_weight_sha256": MI2_WEIGHT_SHA256,
        }

    code = root / "MedImageInsight" / "UniCLModel.py"

    config = root / "2024.09.27" / "config.yaml"

    vision = root / "2024.09.27" / "vision_model" / "medimageinsigt-v1.0.0.pt"

    tokenizer = root / "2024.09.27" / "language_model" / "clip_tokenizer_4.16.2"

    return {
        "root": str(root),
        "complete": bool(
            code.is_file()
            and config.is_file()
            and vision.is_file()
            and tokenizer.is_dir()
        ),
        "code": str(code),
        "config": str(config),
        "vision_weight": str(vision),
        "vision_weight_size_gb": (
            vision.stat().st_size / (1024**3) if vision.is_file() else None
        ),
        "expected_weight_sha256": MI2_WEIGHT_SHA256,
        "native_input_size": MI2_IMAGE_SIZE,
        "embedding_dim": MI2_DIM,
    }


def load_mi2_model(
    root: Path,
    device: torch.device,
):
    if str(root) not in sys.path:
        sys.path.insert(
            0,
            str(root),
        )

    from MedImageInsight.UniCLModel import (
        build_unicl_model,
    )
    from MedImageInsight.Utils.Arguments import (
        load_opt_from_config_files,
    )
    from MedImageInsight.ImageDataLoader import (
        build_transforms,
    )
    from MedImageInsight.LangEncoder import (
        build_tokenizer,
    )

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

    preprocess = build_transforms(
        opt,
        False,
    )

    model = build_unicl_model(opt)

    model.to(device)
    model.eval()

    tokenizer = build_tokenizer(opt["LANG_ENCODER"])

    context_length = int(opt["LANG_ENCODER"]["CONTEXT_LENGTH"])

    return (
        model,
        preprocess,
        tokenizer,
        context_length,
        vision_path,
    )


def normalize_rows(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(
        tensor.float(),
        dim=-1,
        eps=1e-8,
    )


def prompt_lists():
    positive = []
    negative = []

    for label in LABELS:
        positive.append(PROMPTS[label]["positive"])

        negative.append(PROMPTS[label]["negative"])

    return positive, negative


def mi2_text_prototypes(
    model,
    tokenizer,
    context_length: int,
    device: torch.device,
):
    positive_sets, negative_sets = prompt_lists()

    def encode(
        prompts: List[str],
    ) -> torch.Tensor:
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=context_length,
            truncation=True,
            return_tensors="pt",
        )

        tokens = {key: value.to(device) for key, value in tokens.items()}

        with torch.inference_mode():
            features = model.encode_text(tokens)

        return normalize_rows(features)

    positive_prototypes = []
    negative_prototypes = []

    for positive, negative in zip(
        positive_sets,
        negative_sets,
    ):
        p = normalize_rows(
            encode(positive).mean(
                dim=0,
                keepdim=True,
            )
        )[0]

        n = normalize_rows(
            encode(negative).mean(
                dim=0,
                keepdim=True,
            )
        )[0]

        positive_prototypes.append(p)

        negative_prototypes.append(n)

    return (
        torch.stack(positive_prototypes),
        torch.stack(negative_prototypes),
    )


def center_rgb_pil(
    images_chw_uint8: np.ndarray,
) -> List[Image.Image]:
    output = []

    for stack in images_chw_uint8:
        # Channel 1 is the physical
        # center slice.
        center = np.asarray(
            stack[1],
            dtype=np.uint8,
        )

        rgb = np.repeat(
            center[:, :, None],
            3,
            axis=2,
        )

        output.append(
            Image.fromarray(
                rgb,
                mode="RGB",
            )
        )

    return output


def encode_images_mi2(
    model,
    preprocess,
    images: List[Image.Image],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks = []

    with torch.inference_mode():
        for start in range(
            0,
            len(images),
            batch_size,
        ):
            batch_images = images[start : start + batch_size]

            batch = torch.stack([preprocess(image) for image in batch_images]).to(
                device
            )

            with runtime_autocast(device):
                features = model.encode_image(batch)

            features = normalize_rows(features).cpu()

            if not torch.isfinite(features).all():
                raise RuntimeError(
                    "MedImageInsight " "produced non-finite " "embeddings."
                )

            chunks.append(features)

            del batch, features

    return torch.cat(
        chunks,
        dim=0,
    )


# =============================================================================
# 17. TEST MI2 CACHE
# =============================================================================


def test_mi2_cache_path(
    root: Path,
    uid: str,
) -> Path:
    return root / "studies" / f"{stable_uid_hash(uid)}.npz"


def test_mi2_cache_file_usable(
    path: Path,
    uid: str,
) -> bool:
    if not path.is_file():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as payload:
            if str(payload["cache_version"].item()) != TEST_MI2_CACHE_VERSION:
                return False

            if str(payload["study_uid"].item()) != str(uid):
                return False

            if str(payload["encoder"].item()) != "mi2":
                return False

            if str(payload["encoder_sha256"].item()) != MI2_WEIGHT_SHA256:
                return False

            if str(payload["input_mode"].item()) != VLM_INPUT_MODE:
                return False

            embeddings = payload["embeddings"]

            semantic = payload["semantic"]

            n = len(embeddings)

            if embeddings.ndim != 2 or embeddings.shape[1] != MI2_DIM or n <= 0:
                return False

            if semantic.shape != (
                n,
                NUM_LABELS,
            ):
                return False

            for key in (
                "plane",
                "fluid",
                "fat_suppression",
                "slice_position",
            ):
                if key not in payload or len(payload[key]) != n:
                    return False

            if not np.isfinite(embeddings).all():
                return False

            if not np.isfinite(semantic).all():
                return False

            if not np.isfinite(payload["slice_position"]).all():
                return False

        return True

    except Exception:
        return False


def summarize_test_mi2_cache(
    paths: Paths,
    test: pd.DataFrame,
) -> Dict[str, Any]:
    root = paths.test_mi2_cache_root

    usable = 0
    tokens = 0
    missing = []
    invalid = []

    for uid in test[UID_COLUMN].astype(str):
        path = test_mi2_cache_path(
            root,
            uid,
        )

        if not path.is_file():
            if len(missing) < 10:
                missing.append(uid)
            continue

        if not test_mi2_cache_file_usable(
            path,
            uid,
        ):
            if len(invalid) < 10:
                invalid.append(uid)
            continue

        usable += 1

        with np.load(
            path,
            allow_pickle=False,
        ) as payload:
            tokens += int(len(payload["embeddings"]))

    return {
        "root": str(root),
        "cache_version": TEST_MI2_CACHE_VERSION,
        "encoder_sha256": MI2_WEIGHT_SHA256,
        "usable_studies": int(usable),
        "expected_studies": int(len(test)),
        "total_tokens": int(tokens),
        "complete": bool(usable == len(test)),
        "missing_examples": missing,
        "invalid_examples": invalid,
    }


def extract_test_worker(
    paths_payload: Mapping[
        str,
        Optional[str],
    ],
    gpu_index: Optional[int],
    study_uids: Sequence[str],
    mi2_batch: int,
    worker_id: int,
) -> None:
    paths = paths_from_dict(paths_payload)

    paths.ensure_dirs()

    if gpu_index is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    _test, series, _sample = load_test_tables(paths)

    grouped = {str(uid): group.copy() for uid, group in series.groupby(UID_COLUMN)}

    if paths.mi2_root is None:
        raise RuntimeError("MedImageInsight assets " "were not discovered.")

    (
        model,
        preprocess,
        tokenizer,
        context_length,
        _vision_path,
    ) = load_mi2_model(
        paths.mi2_root,
        device,
    )

    (
        positive_prototype,
        negative_prototype,
    ) = mi2_text_prototypes(
        model,
        tokenizer,
        context_length,
        device,
    )

    positive_cpu = positive_prototype.float().cpu()

    negative_cpu = negative_prototype.float().cpu()

    log("=" * 100)
    log(
        f"W45 TEST MI2 WORKER "
        f"{worker_id} | "
        f"device={device} | "
        f"studies={len(study_uids)}"
    )
    log("=" * 100)

    started = time.time()
    completed = 0
    failures = []

    for local_index, uid in enumerate(
        [str(value) for value in study_uids],
        start=1,
    ):
        path = test_mi2_cache_path(
            paths.test_mi2_cache_root,
            uid,
        )

        if test_mi2_cache_file_usable(
            path,
            uid,
        ):
            completed += 1
            continue

        try:
            study_series = grouped.get(uid)

            if study_series is None or len(study_series) == 0:
                raise RuntimeError("No test_series rows " "for study.")

            arrays = build_study_2p5d_arrays(
                paths.test_series_root,
                uid,
                study_series,
            )

            pil_images = center_rgb_pil(arrays["images"])

            features = encode_images_mi2(
                model,
                preprocess,
                pil_images,
                device,
                mi2_batch,
            )

            if features.shape != (
                len(arrays["images"]),
                MI2_DIM,
            ):
                raise RuntimeError(
                    "Unexpected MI2 feature " f"shape: " f"{tuple(features.shape)}"
                )

            semantic = (
                features.float() @ positive_cpu.T - features.float() @ negative_cpu.T
            )

            if semantic.shape != (
                len(arrays["images"]),
                NUM_LABELS,
            ):
                raise RuntimeError(
                    "Unexpected semantic " f"shape: " f"{tuple(semantic.shape)}"
                )

            if not torch.isfinite(semantic).all():
                raise RuntimeError("Semantic scores " "non-finite.")

            atomic_npz_save(
                path,
                cache_version=np.asarray(TEST_MI2_CACHE_VERSION),
                study_uid=np.asarray(uid),
                encoder=np.asarray("mi2"),
                encoder_sha256=np.asarray(MI2_WEIGHT_SHA256),
                source_train_cache_version=np.asarray(TRAIN_MI2_CACHE_VERSION),
                input_mode=np.asarray(VLM_INPUT_MODE),
                embeddings=features.numpy().astype(np.float16),
                semantic=semantic.numpy().astype(np.float16),
                plane=arrays["plane"],
                fluid=arrays["fluid"],
                fat_suppression=arrays["fat_suppression"],
                slice_position=arrays["slice_position"],
            )

            if not test_mi2_cache_file_usable(
                path,
                uid,
            ):
                raise RuntimeError(
                    "Test MI2 cache " "failed validation " "after write."
                )

            completed += 1

        except Exception as exc:
            failures.append(
                {
                    UID_COLUMN: uid,
                    "worker": worker_id,
                    "device": str(device),
                    "error": repr(exc),
                }
            )

            if len(failures) == 1:
                log(
                    f"FIRST FAILURE "
                    f"worker={worker_id} "
                    f"uid={uid}: "
                    f"{repr(exc)}"
                )

        processed = completed + len(failures)

        if local_index <= 5 or local_index % 50 == 0 or local_index == len(study_uids):
            elapsed = time.time() - started

            rate = (
                processed
                / max(
                    elapsed,
                    1e-9,
                )
                * 60.0
            )

            log(
                f"worker={worker_id} "
                f"{local_index:4d}/"
                f"{len(study_uids)} "
                f"ok={completed:4d} "
                f"fail={len(failures):3d} "
                f"rate={rate:5.1f} "
                "studies/min"
            )

        if processed >= 16 and (len(failures) / processed) >= 0.50:
            pd.DataFrame(failures).to_csv(
                paths.result_root
                / ("12_test_extract_" f"worker_{worker_id}" "_failures.csv"),
                index=False,
            )

            raise RuntimeError("Systemic test MI2 " "extraction failures.")

    if failures:
        pd.DataFrame(failures).to_csv(
            paths.result_root
            / ("12_test_extract_" f"worker_{worker_id}" "_failures.csv"),
            index=False,
        )

        raise RuntimeError(f"Worker {worker_id} " f"failed {len(failures)} " "studies.")

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_extract_test(
    paths: Paths,
    args,
) -> Dict[str, Any]:
    paths.ensure_dirs()

    test, series, _sample = load_test_tables(paths)

    existing = summarize_test_mi2_cache(
        paths,
        test,
    )

    if existing["complete"]:
        log("Reusable W45 test MI2 " "cache already complete.")
        return existing

    if is_kaggle_input_path(paths.test_mi2_cache_root):
        raise RuntimeError(
            "Attached test MI2 cache "
            "is incomplete and read-only. "
            "Do not attempt to modify "
            "/kaggle/input. Omit "
            "--test-mi2-cache-root to "
            "create a new cache under "
            "/kaggle/working."
        )

    if args.reset_test_cache:
        for path in (paths.test_mi2_cache_root / "studies").glob("*.npz"):
            path.unlink()

        existing = summarize_test_mi2_cache(
            paths,
            test,
        )

    if paths.mi2_root is None:
        raise RuntimeError("MedImageInsight model " "assets not found.")

    asset = mi2_asset_report(paths.mi2_root)

    if not asset["complete"]:
        raise RuntimeError(
            "MedImageInsight assets "
            "are incomplete:\n"
            + json.dumps(
                asset,
                indent=2,
            )
        )

    # Expensive SHA is performed once in
    # the parent, not once per GPU.
    vision_path = Path(asset["vision_weight"])

    log("Validating MedImageInsight " "weight SHA256...")

    actual_sha = sha256_file(vision_path)

    if actual_sha != (MI2_WEIGHT_SHA256):
        raise RuntimeError(
            "MedImageInsight weight "
            "SHA mismatch:\n"
            f"observed={actual_sha}\n"
            f"expected="
            f"{MI2_WEIGHT_SHA256}"
        )

    audit = require_dicom_preflight(
        paths,
        series,
    )

    write_json(
        paths.result_root / "10_test_dicom_preflight.json",
        audit,
    )

    log("Hidden-test DICOM preflight: PASS")

    for key, value in audit.get("decode_results", {}).items():
        log(
            "  "
            + str(value.get("transfer_syntax_name"))
            + f" [{key}]"
            + " -> "
            + f"decoded="
            + str(value.get("decoded"))
        )

    remaining = [
        uid
        for uid in test[UID_COLUMN].astype(str)
        if not test_mi2_cache_file_usable(
            test_mi2_cache_path(
                paths.test_mi2_cache_root,
                uid,
            ),
            uid,
        )
    ]

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | " "HIDDEN TEST MI2 EXTRACTION")
    log("=" * 100)
    log(f"Test studies         : " f"{len(test)}")
    log(f"Already cached       : " f"{len(test)-len(remaining)}")
    log(f"Need extraction      : " f"{len(remaining)}")
    log(f"MI2 batch            : " f"{args.mi2_batch}")

    if not remaining:
        return summarize_test_mi2_cache(
            paths,
            test,
        )

    device = choose_device(args.accelerator)

    if device.type == "cuda":
        visible = torch.cuda.device_count()

        if visible <= 0:
            raise RuntimeError("CUDA extraction requested " "but no GPU visible.")

        max_workers = (
            visible
            if args.max_extract_gpus <= 0
            else min(
                visible,
                args.max_extract_gpus,
            )
        )

        worker_count = max(
            1,
            min(
                max_workers,
                len(remaining),
            ),
        )

        log(f"Visible CUDA GPUs     : " f"{visible}")
        log(f"Extraction workers    : " f"{worker_count}")
        log("Policy                : " "one independent MI2 model " "per GPU")

        shards = [remaining[index::worker_count] for index in range(worker_count)]

        context = mp.get_context("spawn")

        payload = paths_to_dict(paths)

        workers = []

        for worker_id in range(worker_count):
            process = context.Process(
                target=extract_test_worker,
                args=(
                    payload,
                    worker_id,
                    shards[worker_id],
                    args.mi2_batch,
                    worker_id,
                ),
            )

            workers.append(process)

        for process in workers:
            process.start()

        for process in workers:
            process.join()

        exitcodes = [process.exitcode for process in workers]

        if any(code != 0 for code in exitcodes):
            raise RuntimeError(
                "At least one test " "extraction worker failed: " f"{exitcodes}"
            )

    else:
        # CPU remains available for
        # debugging only.
        extract_test_worker(
            paths_to_dict(paths),
            None,
            remaining,
            args.mi2_batch,
            0,
        )

    summary = summarize_test_mi2_cache(
        paths,
        test,
    )

    summary.update(
        {
            "script_version": SCRIPT_VERSION,
            "created_at": now_iso(),
            "mi2_weight_sha256": actual_sha,
            "dicom_preflight": audit,
        }
    )

    write_json(
        paths.result_root / "11_test_mi2_cache_summary.json",
        summary,
    )

    if not summary["complete"]:
        raise RuntimeError(
            "Hidden-test MI2 cache "
            "is incomplete:\n"
            + json.dumps(
                summary,
                indent=2,
            )
        )

    log("=" * 100)
    log("W45 HIDDEN TEST MI2 " "EXTRACTION COMPLETE")
    log("=" * 100)
    log(f"Studies: " f"{summary['usable_studies']}/" f"{summary['expected_studies']}")
    log(f"Tokens : " f"{summary['total_tokens']}")

    return summary


# =============================================================================
# 18. SUBMISSION
# =============================================================================


def build_submission_dataframe(
    sample: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    if probabilities.shape != (
        len(sample),
        NUM_LABELS,
    ):
        raise RuntimeError(
            "Probability matrix shape " f"mismatch: " f"{probabilities.shape}"
        )

    output = sample.copy()

    for label_index, label in enumerate(LABELS):
        output[label] = probabilities[
            :,
            label_index,
        ]

    # Preserve exact competition order.
    return output[sample.columns]


def validate_submission_dataframe(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    name: str,
) -> None:
    if list(submission.columns) != list(sample.columns):
        raise RuntimeError(f"{name}: column/order mismatch.")

    if (
        submission[UID_COLUMN].astype(str).tolist()
        != sample[UID_COLUMN].astype(str).tolist()
    ):
        raise RuntimeError(f"{name}: UID order mismatch.")

    if submission[UID_COLUMN].duplicated().any():
        raise RuntimeError(f"{name}: duplicate UID.")

    values = submission[LABELS].to_numpy(np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: non-finite " "probabilities.")

    if values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError(f"{name}: probabilities " "outside [0,1].")


def load_full_seed_model(
    paths: Paths,
    seed: int,
) -> MI2ProductionHead:
    path = full_checkpoint_path(
        paths,
        seed,
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    payload = load_checkpoint(path)

    if (
        payload.get("script_version") != SCRIPT_VERSION
        or payload.get("scope") != "full_all58"
        or int(
            payload.get(
                "seed",
                -1,
            )
        )
        != seed
    ):
        raise RuntimeError("Invalid W45 full " f"checkpoint: {path}")

    model = MI2ProductionHead()

    model.load_state_dict(
        payload["model_state"],
        strict=True,
    )

    return model


def run_submit(
    paths: Paths,
    args,
) -> Dict[str, Any]:
    paths.ensure_dirs()

    test, _series, sample = load_test_tables(paths)

    checkpoint_status = checkpoint_report(paths)

    if not checkpoint_status["complete"]:
        raise RuntimeError("W45 production heads are " "incomplete. Run train_full.")

    test_cache = summarize_test_mi2_cache(
        paths,
        test,
    )

    if not test_cache["complete"]:
        raise RuntimeError(
            "W45 hidden-test MI2 cache " "is incomplete. " "Run extract_test."
        )

    device = choose_device(args.accelerator)

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | " "5-SEED TEST PREDICTION")
    log("=" * 100)
    log(f"Prediction device     : " f"{device}")
    log(f"Test studies          : " f"{len(test)}")
    log(f"Production seeds      : " f"{FULL_SEEDS}")

    uids = test[UID_COLUMN].astype(str).tolist()

    store = MI2FeatureStore(
        paths.test_mi2_cache_root,
        uids,
        cache_kind="test",
    )

    dataset = MI2InferenceDataset(
        store,
        uids,
    )

    loader = make_inference_loader(
        dataset,
        INFERENCE_BATCH,
    )

    seed_probabilities = []

    for seed in FULL_SEEDS:
        log(f"Predicting seed={seed}")

        model = load_full_seed_model(
            paths,
            seed,
        )

        predicted_uids, probabilities = predict_head(
            model,
            loader,
            device,
        )

        if predicted_uids != uids:
            raise RuntimeError(f"seed={seed}: " "prediction UID order " "changed.")

        if probabilities.shape != (
            len(test),
            NUM_LABELS,
        ):
            raise RuntimeError(
                f"seed={seed}: " "prediction shape " f"{probabilities.shape}"
            )

        seed_probabilities.append(probabilities.astype(np.float32))

        seed_frame = build_submission_dataframe(
            sample,
            probabilities,
        )

        validate_submission_dataframe(
            seed_frame,
            sample,
            f"seed_{seed}",
        )

        seed_frame.to_csv(
            paths.output_root / ("submission_w45_mi2_" f"seed_{seed}.csv"),
            index=False,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked = np.stack(
        seed_probabilities,
        axis=0,
    )

    mean_probabilities = stacked.mean(axis=0)

    std_probabilities = stacked.std(axis=0)

    baseline = build_submission_dataframe(
        sample,
        mean_probabilities,
    )

    validate_submission_dataframe(
        baseline,
        sample,
        "w45_mi2_5seed",
    )

    baseline_copy = paths.output_root / "submission_w45_mi2_5seed.csv"

    baseline.to_csv(
        baseline_copy,
        index=False,
    )

    if Path("/kaggle/working").exists():
        kaggle_submission = Path("/kaggle/working/submission.csv")
    else:
        kaggle_submission = paths.output_root / "submission.csv"

    baseline.to_csv(
        kaggle_submission,
        index=False,
    )

    np.savez_compressed(
        paths.result_root / "20_test_seed_probabilities.npz",
        seeds=np.asarray(
            FULL_SEEDS,
            dtype=np.int64,
        ),
        study_uids=np.asarray(uids),
        probabilities=stacked.astype(np.float32),
        mean_probabilities=mean_probabilities.astype(np.float32),
        std_probabilities=std_probabilities.astype(np.float32),
    )

    disagreement_rows = []

    for label_index, label in enumerate(LABELS):
        values = std_probabilities[
            :,
            label_index,
        ]

        probabilities = mean_probabilities[
            :,
            label_index,
        ]

        disagreement_rows.append(
            {
                "Label": label,
                "MeanSeedStd": float(values.mean()),
                "MedianSeedStd": float(np.median(values)),
                "MaxSeedStd": float(values.max()),
                "MeanProbability": float(probabilities.mean()),
                "MinProbability": float(probabilities.min()),
                "MaxProbability": float(probabilities.max()),
            }
        )

    disagreement = pd.DataFrame(disagreement_rows)

    disagreement.to_csv(
        paths.result_root / "21_test_seed_disagreement.csv",
        index=False,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "submission": str(kaggle_submission),
        "submission_copy": str(baseline_copy),
        "test_studies": len(test),
        "seeds": FULL_SEEDS,
        "ensemble": "arithmetic_probability_mean",
        "calibration": "none",
        "probability_min": float(mean_probabilities.min()),
        "probability_max": float(mean_probabilities.max()),
        "probability_mean": float(mean_probabilities.mean()),
        "mean_seed_std": float(std_probabilities.mean()),
        "max_seed_std": float(std_probabilities.max()),
        "test_cache": test_cache,
        "overall_pass": True,
    }

    write_json(
        paths.result_root / "22_submission_summary.json",
        summary,
    )

    log("=" * 100)
    log("W45 MI2 SUBMISSION READY")
    log("=" * 100)
    log(f"Kaggle submission : " f"{kaggle_submission}")
    log(f"Saved copy        : " f"{baseline_copy}")
    log(
        f"Probability range : "
        f"[{summary['probability_min']:.6f}, "
        f"{summary['probability_max']:.6f}]"
    )
    log(f"Mean seed std     : " f"{summary['mean_seed_std']:.6f}")
    log(f"Max seed std      : " f"{summary['max_seed_std']:.6f}")

    return summary


# =============================================================================
# 19. OPTIONAL W41 PROBABILITY BLEND
# =============================================================================


def run_blend_w41(
    paths: Paths,
    args,
) -> Dict[str, Any]:
    _test, _series, sample = load_test_tables(paths)

    if not args.w41_submission:
        raise RuntimeError("blend_w41 requires " "--w41-submission.")

    w41_path = Path(args.w41_submission).expanduser().resolve()

    if args.mi2_submission:
        mi2_path = Path(args.mi2_submission).expanduser().resolve()
    else:
        mi2_path = paths.output_root / "submission_w45_mi2_5seed.csv"

    if not mi2_path.is_file():
        raise FileNotFoundError(mi2_path)

    if not w41_path.is_file():
        raise FileNotFoundError(w41_path)

    mi2 = pd.read_csv(mi2_path)

    w41 = pd.read_csv(w41_path)

    mi2[UID_COLUMN] = mi2[UID_COLUMN].astype(str)
    w41[UID_COLUMN] = w41[UID_COLUMN].astype(str)

    validate_submission_dataframe(
        mi2,
        sample,
        "MI2 submission",
    )

    validate_submission_dataframe(
        w41,
        sample,
        "W41 submission",
    )

    weight = float(args.mi2_weight)

    if weight < 0.0 or weight > 1.0:
        raise ValueError("--mi2-weight must be " "within [0,1].")

    mi2_values = mi2[LABELS].to_numpy(np.float64)

    w41_values = w41[LABELS].to_numpy(np.float64)

    blend_values = weight * mi2_values + (1.0 - weight) * w41_values

    blend = build_submission_dataframe(
        sample,
        blend_values,
    )

    validate_submission_dataframe(
        blend,
        sample,
        "W45/W41 blend",
    )

    suffix = f"{weight:.2f}".replace(".", "p")

    output_path = paths.output_root / (
        "submission_w45_mi2_"
        f"w{suffix}_w41_"
        f"w{1-weight:.2f}".replace(".", "p") + ".csv"
    )

    blend.to_csv(
        output_path,
        index=False,
    )

    promoted = False

    if args.promote_blend:
        if Path("/kaggle/working").exists():
            promoted_path = Path("/kaggle/working/submission.csv")
        else:
            promoted_path = paths.output_root / "submission.csv"

        blend.to_csv(
            promoted_path,
            index=False,
        )

        promoted = True

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "mi2_submission": str(mi2_path),
        "w41_submission": str(w41_path),
        "mi2_weight": weight,
        "w41_weight": 1.0 - weight,
        "output": str(output_path),
        "promoted_to_submission_csv": promoted,
        "note": "This fixed blend is optional. "
        "MI2-only should be measured "
        "on the leaderboard first.",
    }

    write_json(
        paths.result_root / "23_w41_blend_summary.json",
        summary,
    )

    log(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return summary


# =============================================================================
# 20. STATUS / VALIDATION
# =============================================================================


def dependency_report() -> Dict[
    str,
    bool,
]:
    names = [
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
    ]

    return {name: dependency_available(name) for name in names}


def run_status(
    paths: Paths,
    args,
) -> Dict[str, Any]:
    paths.ensure_dirs()

    dependencies = dependency_report()

    data_error = None
    train = None
    gold = None
    unlabeled = None
    fold_info = None
    teacher_info = None

    try:
        (
            train,
            gold,
            unlabeled,
        ) = load_train(paths)

        folds = load_folds(
            paths,
            gold,
        )

        (
            _probabilities,
            _weights,
            _masks,
            teacher_info,
        ) = load_w40_teacher(
            paths,
            unlabeled,
        )

        fold_info = {
            "sha256": fold_assignment_sha256(folds),
            "counts": folds["OuterFold"].value_counts().sort_index().to_dict(),
        }

    except Exception as exc:
        data_error = repr(exc)

    if train is not None:
        train_cache = summarize_train_mi2_cache(
            paths,
            train,
        )
    else:
        train_cache = {
            "complete": False,
        }

    test_error = None
    test = None
    series = None
    sample = None

    try:
        test, series, sample = load_test_tables(paths)

        test_cache = summarize_test_mi2_cache(
            paths,
            test,
        )

    except Exception as exc:
        test_error = repr(exc)

        test_cache = {
            "complete": False,
        }

    dicom_preflight = None

    if (
        test_error is None
        and test_cache.get("complete") is not True
        and pydicom is not None
    ):
        try:
            dicom_preflight = inspect_dicom_decoder_support(
                paths.test_series_root,
                series,
                DICOM_PREFLIGHT_SERIES_STATUS,
            )
        except Exception as exc:
            dicom_preflight = {
                "overall_pass": False,
                "error": repr(exc),
            }

    elif test_cache.get("complete") is True:
        dicom_preflight = {
            "overall_pass": True,
            "skipped": True,
            "reason": "test MI2 cache already complete",
        }

    else:
        dicom_preflight = {
            "overall_pass": False,
            "error": "pydicom unavailable",
        }

    asset = mi2_asset_report(paths.mi2_root)

    checkpoints = checkpoint_report(paths)

    required_mi2_dependencies = [
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
    ]

    mi2_dependencies_ok = all(
        dependencies.get(
            name,
            False,
        )
        for name in required_mi2_dependencies
    )

    ready_train = bool(data_error is None and train_cache.get("complete") is True)

    ready_extract = bool(
        test_error is None
        and (
            test_cache.get("complete") is True
            or (
                asset.get("complete") is True
                and mi2_dependencies_ok
                and dependencies.get(
                    "pydicom",
                    False,
                )
                and dicom_preflight.get("overall_pass") is True
            )
        )
    )

    ready_submit = bool(
        test_error is None
        and test_cache.get("complete") is True
        and checkpoints.get("complete") is True
    )

    runtime = {
        "requested_accelerator": args.accelerator,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "cuda_devices": (
            [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        ),
        "test_extraction_policy": "one independent MI2 worker per visible GPU",
        "data_parallel": False,
        "head_training": "single_gpu_fp32",
    }

    payload = {
        "script_version": SCRIPT_VERSION,
        "experiment": DISPLAY_VERSION,
        "runtime": runtime,
        "dependencies": dependencies,
        "paths": {
            "data_root": str(paths.data_root),
            "w40_root": str(paths.w40_root),
            "fold_csv": str(paths.fold_csv),
            "train_mi2_cache_root": (
                str(paths.train_mi2_cache_root)
                if paths.train_mi2_cache_root is not None
                else None
            ),
            "mi2_root": (str(paths.mi2_root) if paths.mi2_root is not None else None),
            "test_mi2_cache_root": str(paths.test_mi2_cache_root),
            "output_root": str(paths.output_root),
        },
        "training_data": {
            "train": (len(train) if train is not None else None),
            "gold": (len(gold) if gold is not None else None),
            "unlabeled": (len(unlabeled) if unlabeled is not None else None),
            "fold": fold_info,
            "teacher": teacher_info,
            "error": data_error,
        },
        "train_mi2_cache": train_cache,
        "medimageinsight": {
            "assets": asset,
            "dependencies_ok": mi2_dependencies_ok,
            "expected_weight_sha256": MI2_WEIGHT_SHA256,
        },
        "hidden_test": {
            "studies": (len(test) if test is not None else None),
            "series_rows": (len(series) if series is not None else None),
            "sample_rows": (len(sample) if sample is not None else None),
            "error": test_error,
            "dicom_preflight": dicom_preflight,
        },
        "test_mi2_cache": test_cache,
        "production_checkpoints": checkpoints,
        "production_recipe": {
            "pseudo_epochs": PSEUDO_EPOCHS,
            "all58_gold_epochs": GOLD_ADAPT_EPOCHS,
            "full_seeds": FULL_SEEDS,
            "encoder_frozen": True,
            "probability_ensemble": "mean",
            "calibration": "none",
            "plane_mapping": "preserve_W44_behavior",
        },
        "ready_for_train_full": ready_train,
        "ready_for_extract_test": ready_extract,
        "ready_for_submit": ready_submit,
        "install_hint_if_needed": (
            "python -m pip install -q "
            "ftfy fvcore mup "
            "sentencepiece safetensors "
            "pydicom pylibjpeg "
            "pylibjpeg-libjpeg "
            "pylibjpeg-openjpeg"
        ),
    }

    print(
        json.dumps(
            payload,
            indent=2,
            allow_nan=True,
        )
    )

    write_json(
        paths.result_root / "00_status.json",
        payload,
    )

    return payload


def run_validate(
    paths: Paths,
    args,
) -> Dict[str, Any]:
    train, gold, unlabeled = load_train(paths)

    folds = load_folds(
        paths,
        gold,
    )

    (
        _probabilities,
        _weights,
        _masks,
        teacher,
    ) = load_w40_teacher(
        paths,
        unlabeled,
    )

    test, _series, sample = load_test_tables(paths)

    train_cache = summarize_train_mi2_cache(
        paths,
        train,
    )

    test_cache = summarize_test_mi2_cache(
        paths,
        test,
    )

    checkpoints = checkpoint_report(paths)

    baseline_path = paths.output_root / "submission_w45_mi2_5seed.csv"

    submission_valid = False
    submission_error = None

    if baseline_path.is_file():
        try:
            frame = pd.read_csv(baseline_path)

            frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)

            validate_submission_dataframe(
                frame,
                sample,
                "W45 baseline",
            )

            submission_valid = True

        except Exception as exc:
            submission_error = repr(exc)

    checks = {
        "train_4407": len(train) == EXPECTED_TRAIN,
        "gold_58": len(gold) == EXPECTED_GOLD,
        "unlabeled_4349": len(unlabeled) == EXPECTED_UNLABELED,
        "fold_sha_locked": fold_assignment_sha256(folds) == EXPECTED_FOLD_SHA256,
        "teacher_32027": teacher["selected_cells"] == EXPECTED_SELECTED_CELLS,
        "train_mi2_cache_complete": train_cache["complete"],
        "train_mi2_token_count": train_cache.get("total_tokens")
        == EXPECTED_TRAIN_MI2_TOKENS,
        "production_5seed_complete": checkpoints["complete"],
        "test_mi2_cache_complete": test_cache["complete"],
        "submission_valid": submission_valid,
    }

    payload = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "checks": checks,
        "submission_error": submission_error,
        "train_cache": train_cache,
        "test_cache": test_cache,
        "checkpoints": checkpoints,
        "overall_pass": all(bool(value) for value in checks.values()),
    }

    write_json(
        paths.result_root / "99_validation.json",
        payload,
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    return payload


# =============================================================================
# 21. CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DISPLAY_VERSION)

    parser.add_argument(
        "mode",
        choices=[
            "status",
            "train_full",
            "extract_test",
            "submit",
            "blend_w41",
            "run_all",
            "validate",
        ],
    )

    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=[
            "auto",
            "localGPU",
            "kaggle_t4",
            "apple_mps",
            "cpu",
        ],
    )

    parser.add_argument(
        "--project-root",
        default=None,
    )

    parser.add_argument(
        "--data-root",
        default=None,
    )

    parser.add_argument(
        "--w40-root",
        default=None,
    )

    parser.add_argument(
        "--fold-csv",
        default=None,
    )

    parser.add_argument(
        "--train-mi2-cache-root",
        default=None,
    )

    parser.add_argument(
        "--mi2-root",
        default=None,
    )

    parser.add_argument(
        "--test-mi2-cache-root",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        default=None,
    )

    parser.add_argument(
        "--mi2-batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-extract-gpus",
        type=int,
        default=0,
        help=("0 = use all visible CUDA " "devices independently."),
    )

    parser.add_argument(
        "--reset-test-cache",
        action="store_true",
    )

    parser.add_argument(
        "--w41-submission",
        default=None,
    )

    parser.add_argument(
        "--mi2-submission",
        default=None,
    )

    parser.add_argument(
        "--mi2-weight",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--promote-blend",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.accelerator == "kaggle_t4" and not torch.cuda.is_available():
        raise RuntimeError(
            "--accelerator kaggle_t4 " "requested but CUDA is " "unavailable."
        )

    if args.accelerator == ("apple_mps"):
        raise RuntimeError("apple_mps is not " "implemented in W45.")

    paths = Paths.discover(args)

    paths.ensure_dirs()

    if args.mode == "status":
        run_status(
            paths,
            args,
        )

    elif args.mode == "train_full":
        run_train_full(
            paths,
            args,
        )

    elif args.mode == "extract_test":
        run_extract_test(
            paths,
            args,
        )

    elif args.mode == "submit":
        run_submit(
            paths,
            args,
        )

    elif args.mode == "blend_w41":
        run_blend_w41(
            paths,
            args,
        )

    elif args.mode == "run_all":
        status = run_status(
            paths,
            args,
        )

        if not status["train_mi2_cache"].get(
            "complete",
            False,
        ):
            raise RuntimeError(
                "Cannot continue: " "saved W44 MI2 train " "cache is not complete."
            )

        checkpoints = checkpoint_report(paths)

        if not checkpoints["complete"]:
            run_train_full(
                paths,
                args,
            )

        test, _series, _sample = load_test_tables(paths)

        test_cache = summarize_test_mi2_cache(
            paths,
            test,
        )

        if not test_cache["complete"]:
            run_extract_test(
                paths,
                args,
            )

        run_submit(
            paths,
            args,
        )

        validation = run_validate(
            paths,
            args,
        )

        if not validation["overall_pass"]:
            raise RuntimeError("W45 final validation " "failed.")

    elif args.mode == "validate":
        run_validate(
            paths,
            args,
        )

    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    mp.freeze_support()
    main()
