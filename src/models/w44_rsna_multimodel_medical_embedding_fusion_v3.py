%%writefile /kaggle/working/w44_rsna_multimodel_medical_embedding_fusion_v3.py
#!/usr/bin/env python3
"""
W44 — Multi-Model Medical Embedding Fusion v1
==============================================

Goal
----
Evaluate the user's original multi-model fusion hypothesis on the RSNA 2026
Knee Abnormality Detection task without spending another cycle fine-tuning a
single large image backbone.

Frozen visual-language representations:
    1. MedImageInsight (DaViT / UniCL), 1024-D, native 480x480 preprocessing.
    2. BiomedCLIP (ViT-B/16 + PubMedBERT), 512-D, native 224x224 preprocessing.
    3. Existing Curia-2 W4 CLS cache, 768-D (optional but strongly preferred).

The expensive encoders are frozen and extracted exactly once. W40 remains the
pseudo-supervision source. Lightweight label-aware heads are then compared under
one locked 5-fold diagnostic:

    mi2
    biomedclip
    mean2
    weighted2
    concat2
    concat3_curia       (only when the complete canonical W4 Curia cache exists)

The script also reports:
    - fixed text/image semantic diagnostics,
    - per-label metrics for every variant,
    - a clearly-labelled non-deployable per-label ORACLE upper bound,
    - a fixed probability-mean ensemble of the two best global variants.

Important design choices
------------------------
* NO DINO / DINOv2 / DINOv3.
* NO ResNet branch.
* NO RadImageNet training.
* NO large-encoder fine-tuning.
* NO imports/calls to another project .py file.
* Third-party model package code shipped with the attached model assets is
  allowed; it is not project code.
* No ZIP extraction. Kaggle attached datasets are already mounted.
* The existing W43 backbone-independent 2.5D cache is reused.
* VLM encoders consume the CENTER MRI slice replicated to RGB, rather than
  treating adjacent MRI slices as semantic RGB channels. Through-plane context
  is recovered by study-level aggregation across multiple physical positions.
* Each VLM uses its OWN native preprocessing. Do not force a shared transform.
* Both Kaggle T4 GPUs are used independently during extraction when available:
      GPU0 -> MedImageInsight
      GPU1 -> BiomedCLIP
  The lightweight fusion-head training uses one GPU and is cheap.

Expected external model assets
------------------------------
MedImageInsight repository root must contain:
    MedImageInsight/...
    2024.09.27/config.yaml
    2024.09.27/vision_model/medimageinsigt-v1.0.0.pt
    2024.09.27/language_model/clip_tokenizer_4.16.2/

BiomedCLIP local directory must contain:
    open_clip_config.json
    open_clip_pytorch_model.bin
    tokenizer.json / vocab.txt / tokenizer configs

The script discovers mounted Kaggle datasets by file structure, not dataset slug.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# =============================================================================
# IDENTITY / LOCKS
# =============================================================================

SCRIPT_VERSION = "w44_multimodel_medical_embedding_fusion_v3"
DISPLAY_VERSION = "W44 | Multi-Model Medical Embedding Fusion v3"
OUTPUT_DIR_NAME = "rsna_w44_multimodel_medical_embedding_fusion_v3"

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
EXPECTED_FOLD_SHA256 = "1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4"

SHARED_CACHE_VERSION = "rsna_2p5d_mri_cache_v1"
SHARED_CACHE_IMAGE_SIZE = 224
CURIA_CACHE_VERSION = (
    "w4_0_curia2_cls_allseries_24slice_" "canonical_orientation_slow_processor_v1"
)

MI2_DIM = 1024
BIOMEDCLIP_DIM = 512
CURIA_DIM = 768
MI2_IMAGE_SIZE = 480
BIOMEDCLIP_IMAGE_SIZE = 224

EMBED_CACHE_VERSION = "w44_frozen_medical_embedding_cache_v1"
VLM_INPUT_MODE = "center_slice_replicated_rgb"

# Lightweight fusion head.
HIDDEN_DIM = 192
PLANE_EMBED_DIM = 16
BINARY_META_DIM = 8
POSITION_DIM = 32
HEAD_DROPOUT = 0.18

MAX_VLM_TOKENS = 64
MAX_CURIA_TOKENS = 96

PSEUDO_EPOCHS = 5
GOLD_ADAPT_EPOCHS = 10
PSEUDO_BATCH = 48
GOLD_BATCH = 16
PSEUDO_LR = 3e-4
GOLD_LR = 8e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 3.0

BASE_SEED = 44000
CURIA_REFERENCE_AUC = 0.702997
GO_AUC = 0.760
STRONG_GO_AUC = 0.800
REVIEW_AUC = 0.720
WEAK_LABEL_AUC = 0.55

MI2_DEFAULT_BATCH = 4
BIOMEDCLIP_DEFAULT_BATCH = 48

VARIANTS_BASE = [
    "mi2",
    "biomedclip",
    "mean2",
    "weighted2",
    "concat2",
]
CURIA_VARIANT = "concat3_curia"

# Plane order: sagittal, coronal, axial.
PLANE_TO_INDEX = {"Sagittal": 0, "Coronal": 1, "Axial": 2}

# Soft anatomy priors only; no plane is hard-excluded.
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
# PATHOLOGY PROMPTS
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
# BASIC HELPERS
# =============================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def stable_uid_hash(uid: str) -> str:
    return hashlib.md5(str(uid).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(x: Any):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        raise TypeError(type(x).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True, default=conv),
        encoding="utf-8",
    )


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise RuntimeError("Boolean column contains unrecognized values.")
    return out.astype(bool)


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    x = assignments[[UID_COLUMN, "OuterFold"]].copy()
    x[UID_COLUMN] = x[UID_COLUMN].astype(str)
    x = x.sort_values(UID_COLUMN).reset_index(drop=True)
    payload = "".join(
        f"{uid},{int(fold)}\n" for uid, fold in zip(x[UID_COLUMN], x["OuterFold"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _script_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, here.parent.parent.parent]
    for candidate in candidates:
        if (candidate / "input" / "train.csv").is_file():
            return candidate
    return here.parent


def kaggle_dataset_roots() -> List[Path]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name != "competitions"]
    )


def bounded_glob(root: Path, patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    if not root.exists():
        return found
    for pattern in patterns:
        try:
            found.extend(root.glob(pattern))
        except Exception:
            pass
    return found


# =============================================================================
# DISCOVERY
# =============================================================================


def discover_w40_root(project_root: Path) -> Optional[Path]:
    required = [
        "16_final_probabilities_wide.csv",
        "17_teacher_weights_wide.csv",
        "18_teacher_mask_wide.csv",
    ]
    candidates = [
        project_root / "output" / "results" / "rsna_w40_fs2_production_teacher_v1"
    ]
    for root in kaggle_dataset_roots():
        candidates.extend(
            bounded_glob(
                root,
                [
                    "rsna_w40_fs2_production_teacher_v1",
                    "*/rsna_w40_fs2_production_teacher_v1",
                    "*/*/rsna_w40_fs2_production_teacher_v1",
                ],
            )
        )
        for path in bounded_glob(
            root,
            [
                "results/16_final_probabilities_wide.csv",
                "*/results/16_final_probabilities_wide.csv",
                "*/*/results/16_final_probabilities_wide.csv",
            ],
        ):
            candidates.append(path.parent.parent)

    valid = []
    for candidate in candidates:
        result_dir = candidate / "results"
        if all((result_dir / name).is_file() for name in required):
            valid.append(candidate.resolve())
    if not valid:
        return None
    valid = sorted(set(valid), key=lambda p: (len(str(p)), str(p)))
    return valid[0]


def discover_fold_csv(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root
        / "output"
        / "results"
        / "rsna_w2_3"
        / "results"
        / "00_outer_fold_assignments.csv"
    ]
    for root in kaggle_dataset_roots():
        candidates.extend(
            bounded_glob(
                root,
                [
                    "00_outer_fold_assignments.csv",
                    "results/00_outer_fold_assignments.csv",
                    "*/00_outer_fold_assignments.csv",
                    "*/results/00_outer_fold_assignments.csv",
                    "*/*/00_outer_fold_assignments.csv",
                    "*/*/results/00_outer_fold_assignments.csv",
                    "*/*/*/results/00_outer_fold_assignments.csv",
                ],
            )
        )
    valid = sorted(
        {p.resolve() for p in candidates if p.is_file()},
        key=lambda p: (len(str(p)), str(p)),
    )
    return valid[0] if valid else None


def _shared_cache_candidate_usable(root: Path) -> bool:
    study_dir = root / "studies"
    if not study_dir.is_dir():
        return False
    # Cheap structural probe; full 4407 validation is performed in status/train.
    return any(study_dir.glob("*.npz"))


def discover_shared_cache(project_root: Path) -> Optional[Path]:
    candidates = [
        Path(
            "/kaggle/input/datasets/isayem/w43-rsna-radimagenet-densenet121-2p5d-v4-cache/w43_rsna_radimagenet_densenet121_2p5d_v4_cache"
        )
        / "output"
        / "results"
        / SHARED_CACHE_VERSION,
        project_root / "output" / "results" / SHARED_CACHE_VERSION,
        Path("/kaggle/working") / SHARED_CACHE_VERSION,
        Path("/kaggle/working/output/results") / SHARED_CACHE_VERSION,
    ]
    for root in kaggle_dataset_roots():
        candidates.extend(
            bounded_glob(
                root,
                [
                    SHARED_CACHE_VERSION,
                    f"*/{SHARED_CACHE_VERSION}",
                    f"*/*/{SHARED_CACHE_VERSION}",
                ],
            )
        )
        for studies in bounded_glob(
            root,
            [
                "studies",
                "*/studies",
                "*/*/studies",
            ],
        ):
            if any(studies.glob("*.npz")):
                parent = studies.parent
                if parent.name == SHARED_CACHE_VERSION:
                    candidates.append(parent)
    valid = [p.resolve() for p in candidates if _shared_cache_candidate_usable(p)]
    if not valid:
        return None
    valid = sorted(
        set(valid),
        key=lambda p: (0 if "/kaggle/working" in str(p) else 1, len(str(p)), str(p)),
    )
    return valid[0]


def discover_mi2_root(project_root: Path) -> Optional[Path]:
    candidates: List[Path] = [
        Path("/kaggle/input/datasets/isayem/models/MedImageInsights"),
        project_root / "models" / "MedImageInsights",
        project_root / "models" / "medimageinsight",
    ]
    for root in kaggle_dataset_roots():
        candidates.extend([root])
        candidates.extend(
            bounded_glob(root, ["MedImageInsights", "*/MedImageInsights"])
        )

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
    return sorted(set(valid), key=lambda p: (len(str(p)), str(p)))[0]


def discover_biomedclip_root(project_root: Path) -> Optional[Path]:
    candidates = [
        Path("/kaggle/input/datasets/isayem/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        project_root / "models" / "biomedclip",
        project_root / "models" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ]
    for root in kaggle_dataset_roots():
        candidates.extend([root])
        candidates.extend(
            bounded_glob(
                root,
                [
                    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    "*/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    "biomedclip",
                    "*/biomedclip",
                ],
            )
        )
    valid = []
    for root in candidates:
        if (root / "open_clip_config.json").is_file() and (
            root / "open_clip_pytorch_model.bin"
        ).is_file():
            valid.append(root.resolve())
    if not valid:
        return None
    return sorted(set(valid), key=lambda p: (len(str(p)), str(p)))[0]


def _biomedbert_candidate_usable(root: Path) -> bool:
    if not root.is_dir():
        return False
    config_ok = (root / "config.json").is_file()
    vocab_ok = (root / "vocab.txt").is_file()
    weight_ok = any(
        (root / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")
    )
    return bool(config_ok and vocab_ok and weight_ok)


def discover_biomedbert_root(project_root: Path) -> Optional[Path]:
    """
    Discover the separate PubMedBERT/BiomedBERT text-tower repository required
    by BiomedCLIP's OpenCLIP config.

    Exact upstream identifier:
        microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
    """
    names = (
        "BiomedNLP-BiomedBERT-base-uncased-abstract",
        "microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract",
        "biomedbert",
        "BiomedBERT",
    )
    candidates: List[Path] = [
        Path("/kaggle/input/datasets/isayem/models/BiomedNLP-BiomedBERT-base-uncased-abstract"),
        project_root / "models" / names[0],
        project_root / "models" / "biomedbert",
    ]

    for root in kaggle_dataset_roots():
        candidates.append(root)
        for name in names:
            candidates.extend(
                bounded_glob(
                    root,
                    [name, f"*/{name}", f"*/*/{name}"],
                )
            )
        if _biomedbert_candidate_usable(root):
            candidates.append(root)

    valid = [
        candidate.resolve()
        for candidate in candidates
        if _biomedbert_candidate_usable(candidate)
    ]
    if not valid:
        return None

    return sorted(
        set(valid),
        key=lambda p: (
            0 if p.name == "BiomedNLP-BiomedBERT-base-uncased-abstract" else 1,
            len(str(p)),
            str(p),
        ),
    )[0]


def _curia_study_dir_usable(study_dir: Path) -> bool:
    if not study_dir.is_dir():
        return False
    return any(study_dir.glob("*.pt"))


def discover_curia_study_root(project_root: Path) -> Optional[Path]:
    candidates = [
        Path("/kaggle/input/datasets/isayem/rsna-w4-0-curia2")        
        / "feature_cache"
        / "studies",
        project_root / "rsna_w4_0_curia2" / "feature_cache" / "studies",
        Path("/kaggle/working/rsna_w4_0_curia2/feature_cache/studies"),
    ]
    for root in kaggle_dataset_roots():
        candidates.extend(
            bounded_glob(
                root,
                [
                    "feature_cache/studies",
                    "rsna_w4_0_curia2/feature_cache/studies",
                    "*/feature_cache/studies",
                    "*/rsna_w4_0_curia2/feature_cache/studies",
                    "*/*/feature_cache/studies",
                ],
            )
        )
    valid = [p.resolve() for p in candidates if _curia_study_dir_usable(p)]
    if not valid:
        return None
    return sorted(set(valid), key=lambda p: (len(str(p)), str(p)))[0]


@dataclass
class Paths:
    project_root: Path
    data_root: Path
    train_csv: Path
    train_series_csv: Path
    w40_root: Path
    fold_csv: Path
    shared_cache_root: Path
    mi2_root: Optional[Path]
    biomedclip_root: Optional[Path]
    biomedbert_root: Optional[Path]
    curia_study_root: Optional[Path]
    output_root: Path
    result_root: Path
    checkpoint_root: Path
    embedding_root: Path

    @classmethod
    def discover(cls, args) -> "Paths":
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

        if args.w40_root:
            w40_root = Path(args.w40_root).expanduser().resolve()
        else:
            w40_root = discover_w40_root(project_root) or (
                project_root / "output/results/rsna_w40_fs2_production_teacher_v1"
            )

        if args.fold_csv:
            fold_csv = Path(args.fold_csv).expanduser().resolve()
        else:
            fold_csv = discover_fold_csv(project_root) or (
                project_root
                / "output/results/rsna_w2_3/results/00_outer_fold_assignments.csv"
            )

        if args.shared_cache_root:
            shared_cache_root = Path(args.shared_cache_root).expanduser().resolve()
        else:
            shared_cache_root = discover_shared_cache(project_root) or (
                project_root / "output/results" / SHARED_CACHE_VERSION
            )

        mi2_root = (
            Path(args.mi2_root).expanduser().resolve()
            if args.mi2_root
            else discover_mi2_root(project_root)
        )
        biomedclip_root = (
            Path(args.biomedclip_root).expanduser().resolve()
            if args.biomedclip_root
            else discover_biomedclip_root(project_root)
        )
        biomedbert_root = (
            Path(args.biomedbert_root).expanduser().resolve()
            if args.biomedbert_root
            else discover_biomedbert_root(project_root)
        )
        curia_study_root = (
            Path(args.curia_cache_root).expanduser().resolve()
            if args.curia_cache_root
            else discover_curia_study_root(project_root)
        )

        if args.output_root:
            output_root = Path(args.output_root).expanduser().resolve()
        elif is_kaggle:
            output_root = Path("/kaggle/working") / OUTPUT_DIR_NAME
        else:
            output_root = project_root / "output/results" / OUTPUT_DIR_NAME

        return cls(
            project_root=project_root,
            data_root=data_root,
            train_csv=data_root / "train.csv",
            train_series_csv=data_root / "train_series.csv",
            w40_root=w40_root,
            fold_csv=fold_csv,
            shared_cache_root=shared_cache_root,
            mi2_root=mi2_root,
            biomedclip_root=biomedclip_root,
            biomedbert_root=biomedbert_root,
            curia_study_root=curia_study_root,
            output_root=output_root,
            result_root=output_root / "results",
            checkpoint_root=output_root / "checkpoints",
            embedding_root=output_root / "embedding_cache",
        )

    def ensure_dirs(self) -> None:
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        (self.embedding_root / "mi2" / "studies").mkdir(parents=True, exist_ok=True)
        (self.embedding_root / "biomedclip" / "studies").mkdir(
            parents=True, exist_ok=True
        )


# =============================================================================
# DATA / W40 / FOLDS
# =============================================================================


def load_train(paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(paths.train_csv)
    train[UID_COLUMN] = train[UID_COLUMN].astype(str)
    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN} train rows, got {len(train)}")
    missing = [label for label in LABELS if label not in train.columns]
    if missing:
        raise RuntimeError(f"train.csv missing labels: {missing}")

    gold_mask = train[LABELS].notna().all(axis=1)
    gold = train.loc[gold_mask].copy().reset_index(drop=True)
    unlabeled = train.loc[~gold_mask].copy().reset_index(drop=True)
    if len(gold) != EXPECTED_GOLD or len(unlabeled) != EXPECTED_UNLABELED:
        raise RuntimeError(f"Gold/unlabeled mismatch: {len(gold)}/{len(unlabeled)}")
    return train, gold, unlabeled


def load_folds(paths: Paths, gold: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_csv(paths.fold_csv)
    folds[UID_COLUMN] = folds[UID_COLUMN].astype(str)
    if "OuterFold" not in folds.columns:
        raise RuntimeError("Fold CSV missing OuterFold")
    folds = folds[[UID_COLUMN, "OuterFold"]].copy()
    folds["OuterFold"] = folds["OuterFold"].astype(int)
    if set(folds[UID_COLUMN]) != set(gold[UID_COLUMN]):
        raise RuntimeError("Locked fold UID set does not match gold 58")
    sha = fold_assignment_sha256(folds)
    if sha != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"Locked fold SHA mismatch: {sha}")
    counts = folds["OuterFold"].value_counts().sort_index().to_dict()
    expected = {1: 11, 2: 12, 3: 11, 4: 11, 5: 13}
    if counts != expected:
        raise RuntimeError(f"Fold counts mismatch: {counts}")
    return folds


def w40_result(paths: Paths, name: str) -> Path:
    return paths.w40_root / "results" / name


def load_w40_teacher(
    paths: Paths,
    unlabeled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    prob_path = w40_result(paths, "16_final_probabilities_wide.csv")
    weight_path = w40_result(paths, "17_teacher_weights_wide.csv")
    mask_path = w40_result(paths, "18_teacher_mask_wide.csv")

    probs = pd.read_csv(prob_path)
    weights = pd.read_csv(weight_path)
    masks = pd.read_csv(mask_path)

    for frame in (probs, weights, masks):
        frame[UID_COLUMN] = frame[UID_COLUMN].astype(str)
        if frame[UID_COLUMN].duplicated().any():
            raise RuntimeError("W40 wide table contains duplicate UIDs")
        if set(frame[UID_COLUMN]) != set(unlabeled[UID_COLUMN]):
            raise RuntimeError("W40 UID set mismatch")

    probs = probs.set_index(UID_COLUMN).reindex(unlabeled[UID_COLUMN]).reset_index()
    weights = weights.set_index(UID_COLUMN).reindex(unlabeled[UID_COLUMN]).reset_index()
    masks = masks.set_index(UID_COLUMN).reindex(unlabeled[UID_COLUMN]).reset_index()

    for label in LABELS:
        probs[label] = pd.to_numeric(probs[label], errors="raise").astype(np.float32)
        weights[label] = pd.to_numeric(weights[label], errors="raise").astype(
            np.float32
        )
        masks[label] = coerce_bool_series(masks[label])

    selected = int(masks[LABELS].to_numpy(bool).sum())
    if selected != EXPECTED_SELECTED_CELLS:
        raise RuntimeError(
            f"W40 selected cell mismatch: {selected}/{EXPECTED_SELECTED_CELLS}"
        )
    if not np.isfinite(probs[LABELS].to_numpy(float)).all():
        raise RuntimeError("W40 probabilities contain non-finite values")
    if not np.isfinite(weights[LABELS].to_numpy(float)).all():
        raise RuntimeError("W40 weights contain non-finite values")

    return (
        probs,
        weights,
        masks,
        {
            "rows": len(probs),
            "selected_cells": selected,
            "probability_sha256": sha256_file(prob_path),
            "weight_sha256": sha256_file(weight_path),
            "mask_sha256": sha256_file(mask_path),
        },
    )


# =============================================================================
# SHARED 2.5D CACHE / CURIA CACHE
# =============================================================================


def shared_study_path(paths: Paths, uid: str) -> Path:
    return paths.shared_cache_root / "studies" / f"{stable_uid_hash(uid)}.npz"


def shared_cache_file_usable(path: Path, uid: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as x:
            if str(x["cache_version"].item()) != SHARED_CACHE_VERSION:
                return False
            if str(x["study_uid"].item()) != str(uid):
                return False
            images = x["images"]
            if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
                return False
            if images.dtype != np.uint8 or len(images) <= 0:
                return False
            n = len(images)
            for key in ("plane", "fluid", "fat_suppression", "slice_position"):
                if len(x[key]) != n:
                    return False
            if not np.isfinite(x["slice_position"]).all():
                return False
        return True
    except Exception:
        return False


def summarize_shared_cache(paths: Paths, train: pd.DataFrame) -> Dict[str, Any]:
    usable = 0
    total_tokens = 0
    missing = []
    for uid in train[UID_COLUMN].astype(str):
        path = shared_study_path(paths, uid)
        if not shared_cache_file_usable(path, uid):
            if len(missing) < 10:
                missing.append(uid)
            continue
        usable += 1
        with np.load(path, allow_pickle=False) as x:
            total_tokens += int(len(x["images"]))
    return {
        "cache_version": SHARED_CACHE_VERSION,
        "root": str(paths.shared_cache_root),
        "usable_studies": usable,
        "expected_studies": len(train),
        "total_tokens": total_tokens,
        "mean_tokens_per_study": float(total_tokens / max(usable, 1)),
        "complete": usable == len(train),
        "missing_examples": missing,
    }


def curia_study_path(paths: Paths, uid: str) -> Optional[Path]:
    if paths.curia_study_root is None:
        return None
    return paths.curia_study_root / f"{stable_uid_hash(uid)}.pt"


def curia_cache_file_usable(path: Optional[Path], uid: str) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return bool(
            payload.get("cache_version") == CURIA_CACHE_VERSION
            and str(payload.get("study_uid")) == str(uid)
            and isinstance(payload.get("features"), torch.Tensor)
            and payload["features"].ndim == 3
            and payload["features"].shape[-1] == CURIA_DIM
            and payload["slice_mask"].shape == payload["features"].shape[:2]
            and payload["slice_position"].shape == payload["features"].shape[:2]
            and payload["series_meta"].shape[0] == payload["features"].shape[0]
        )
    except Exception:
        return False


def summarize_curia_cache(paths: Paths, train: pd.DataFrame) -> Dict[str, Any]:
    if paths.curia_study_root is None:
        return {
            "root": None,
            "usable_studies": 0,
            "expected_studies": len(train),
            "complete": False,
            "missing_examples": train[UID_COLUMN].head(10).tolist(),
        }
    usable = 0
    missing = []
    for uid in train[UID_COLUMN].astype(str):
        if curia_cache_file_usable(curia_study_path(paths, uid), uid):
            usable += 1
        elif len(missing) < 10:
            missing.append(uid)
    return {
        "root": str(paths.curia_study_root),
        "cache_version": CURIA_CACHE_VERSION,
        "usable_studies": usable,
        "expected_studies": len(train),
        "complete": usable == len(train),
        "missing_examples": missing,
    }


# =============================================================================
# ENCODER CACHE
# =============================================================================


def encoder_cache_path(paths: Paths, encoder: str, uid: str) -> Path:
    return paths.embedding_root / encoder / "studies" / f"{stable_uid_hash(uid)}.npz"


def expected_encoder_dim(encoder: str) -> int:
    if encoder == "mi2":
        return MI2_DIM
    if encoder == "biomedclip":
        return BIOMEDCLIP_DIM
    raise ValueError(encoder)


def embedding_cache_file_usable(path: Path, uid: str, encoder: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as x:
            if str(x["cache_version"].item()) != EMBED_CACHE_VERSION:
                return False
            if str(x["study_uid"].item()) != str(uid):
                return False
            if str(x["encoder"].item()) != encoder:
                return False
            if str(x["input_mode"].item()) != VLM_INPUT_MODE:
                return False
            emb = x["embeddings"]
            sem = x["semantic"]
            n = len(emb)
            if emb.ndim != 2 or emb.shape[1] != expected_encoder_dim(encoder):
                return False
            if sem.shape != (n, NUM_LABELS):
                return False
            for key in ("plane", "fluid", "fat_suppression", "slice_position"):
                if len(x[key]) != n:
                    return False
            if n <= 0:
                return False
            if not np.isfinite(emb).all() or not np.isfinite(sem).all():
                return False
        return True
    except Exception:
        return False


def summarize_embedding_cache(
    paths: Paths, train: pd.DataFrame, encoder: str
) -> Dict[str, Any]:
    usable = 0
    tokens = 0
    missing = []
    for uid in train[UID_COLUMN].astype(str):
        path = encoder_cache_path(paths, encoder, uid)
        if embedding_cache_file_usable(path, uid, encoder):
            usable += 1
            with np.load(path, allow_pickle=False) as x:
                tokens += int(len(x["embeddings"]))
        elif len(missing) < 10:
            missing.append(uid)
    return {
        "encoder": encoder,
        "cache_version": EMBED_CACHE_VERSION,
        "root": str(paths.embedding_root / encoder),
        "usable_studies": usable,
        "expected_studies": len(train),
        "total_tokens": tokens,
        "complete": usable == len(train),
        "missing_examples": missing,
    }


# =============================================================================
# MODEL LOADING / PREPROCESSING
# =============================================================================


def center_rgb_pil(images_chw_uint8: np.ndarray) -> List[Image.Image]:
    # Input is [T,3,H,W], where channel 1 is the physical center slice.
    output: List[Image.Image] = []
    for stack in images_chw_uint8:
        center = np.asarray(stack[1], dtype=np.uint8)
        rgb = np.repeat(center[:, :, None], 3, axis=2)
        output.append(Image.fromarray(rgb, mode="RGB"))
    return output


def prompt_lists() -> Tuple[List[List[str]], List[List[str]]]:
    pos, neg = [], []
    for label in LABELS:
        pos.append(PROMPTS[label]["positive"])
        neg.append(PROMPTS[label]["negative"])
    return pos, neg


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def load_mi2_model(root: Path, device: torch.device):
    # Third-party model package, not project Python.
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
    # Inference only. This changes no weights/shapes.
    try:
        opt["IMAGE_ENCODER"]["SPEC"]["ENABLE_CHECKPOINT"] = False
    except Exception:
        pass
    opt["VERBOSE"] = False

    preprocess = build_transforms(opt, False)
    model = build_unicl_model(opt)
    model.to(device)
    model.eval()
    tokenizer = build_tokenizer(opt["LANG_ENCODER"])
    context_length = int(opt["LANG_ENCODER"]["CONTEXT_LENGTH"])
    return model, preprocess, tokenizer, context_length, vision_path


def mi2_text_prototypes(model, tokenizer, context_length: int, device: torch.device):
    pos_sets, neg_sets = prompt_lists()

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
        return _normalize_rows(x)

    pos_proto, neg_proto = [], []
    for pos, neg in zip(pos_sets, neg_sets):
        pos_proto.append(_normalize_rows(encode(pos).mean(dim=0, keepdim=True))[0])
        neg_proto.append(_normalize_rows(encode(neg).mean(dim=0, keepdim=True))[0])
    return torch.stack(pos_proto), torch.stack(neg_proto)


def load_biomedclip_model(
    root: Path,
    biomedbert_root: Path,
    device: torch.device,
):
    import open_clip
    from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

    config_path = root / "open_clip_config.json"
    weight_path = root / "open_clip_pytorch_model.bin"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]
    model_name = "w44_biomedclip_local"
    if not model_name.startswith(HF_HUB_PREFIX) and model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = model_cfg

    # The text HF model/tokenizer normally resolves from the local repo files if
    # transformers is pointed at the local config by OpenCLIP. To make offline
    # Kaggle robust, temporarily set local-files-only HF flags in-process.
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        # The official BiomedCLIP OpenCLIP config references a separate
        # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract text tower.
        # Point both HF model and tokenizer at the attached local snapshot so
        # Kaggle works fully offline.
        if not _biomedbert_candidate_usable(biomedbert_root):
            raise RuntimeError(
                "BiomedBERT snapshot incomplete. Expected config.json, vocab.txt, "
                "and pytorch_model.bin or model.safetensors under "
                f"{biomedbert_root}"
            )
        local_cfg = copy.deepcopy(model_cfg)
        local_cfg["text_cfg"]["hf_model_name"] = str(biomedbert_root)
        local_cfg["text_cfg"]["hf_tokenizer_name"] = str(biomedbert_root)
        _MODEL_CONFIGS[model_name] = local_cfg

        tokenizer = open_clip.get_tokenizer(model_name)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=str(weight_path),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
    finally:
        if old_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline
        if old_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf_offline

    model.to(device)
    model.eval()
    return model, preprocess, tokenizer, weight_path


def biomedclip_text_prototypes(model, tokenizer, device: torch.device):
    pos_sets, neg_sets = prompt_lists()

    def encode(prompts: List[str]) -> torch.Tensor:
        tokens = tokenizer(prompts, context_length=256).to(device)
        with torch.inference_mode():
            try:
                x = model.encode_text(tokens, normalize=True)
            except TypeError:
                x = model.encode_text(tokens)
        return _normalize_rows(x)

    pos_proto, neg_proto = [], []
    for pos, neg in zip(pos_sets, neg_sets):
        pos_proto.append(_normalize_rows(encode(pos).mean(dim=0, keepdim=True))[0])
        neg_proto.append(_normalize_rows(encode(neg).mean(dim=0, keepdim=True))[0])
    return torch.stack(pos_proto), torch.stack(neg_proto)


# =============================================================================
# EXTRACTION
# =============================================================================


def _runtime_autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    # T4 => FP16; Ampere+ => BF16 when native.
    major, _minor = torch.cuda.get_device_capability(device.index or 0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def encode_images_mi2(
    model, preprocess, images: List[Image.Image], device: torch.device, batch_size: int
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_pil = images[start : start + batch_size]
            batch = torch.stack([preprocess(img) for img in batch_pil]).to(device)
            with _runtime_autocast(device):
                features = model.encode_image(batch)
            features = _normalize_rows(features).cpu()
            if not torch.isfinite(features).all():
                raise RuntimeError("MedImageInsight produced non-finite embeddings")
            chunks.append(features)
            del batch, features
    return torch.cat(chunks, dim=0)


def encode_images_biomedclip(
    model, preprocess, images: List[Image.Image], device: torch.device, batch_size: int
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_pil = images[start : start + batch_size]
            batch = torch.stack([preprocess(img) for img in batch_pil]).to(device)
            with _runtime_autocast(device):
                try:
                    features = model.encode_image(batch, normalize=True)
                except TypeError:
                    features = model.encode_image(batch)
            features = _normalize_rows(features).cpu()
            if not torch.isfinite(features).all():
                raise RuntimeError("BiomedCLIP produced non-finite embeddings")
            chunks.append(features)
            del batch, features
    return torch.cat(chunks, dim=0)


def _extract_one_encoder(
    encoder: str,
    paths_dict: Dict[str, Any],
    gpu_index: Optional[int],
    mi2_batch: int,
    biomed_batch: int,
) -> None:
    paths = Paths(
        **{
            k: Path(v) if isinstance(v, str) and k.endswith(("root", "csv")) else v
            for k, v in paths_dict.items()
        }
    )
    paths.ensure_dirs()

    if gpu_index is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    train, _gold, _unlabeled = load_train(paths)

    if encoder == "mi2":
        if paths.mi2_root is None:
            raise RuntimeError("MedImageInsight root not discovered")
        model, preprocess, tokenizer, context_length, weight_path = load_mi2_model(
            paths.mi2_root, device
        )
        pos_proto, neg_proto = mi2_text_prototypes(
            model, tokenizer, context_length, device
        )
        batch_size = int(mi2_batch)
        encode_fn = encode_images_mi2
    elif encoder == "biomedclip":
        if paths.biomedclip_root is None:
            raise RuntimeError("BiomedCLIP root not discovered")
        if paths.biomedbert_root is None:
            raise RuntimeError(
                "BiomedBERT text-tower root not discovered. Attach "
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract."
            )
        model, preprocess, tokenizer, weight_path = load_biomedclip_model(
            paths.biomedclip_root,
            paths.biomedbert_root,
            device,
        )
        pos_proto, neg_proto = biomedclip_text_prototypes(model, tokenizer, device)
        batch_size = int(biomed_batch)
        encode_fn = encode_images_biomedclip
    else:
        raise ValueError(encoder)

    weight_sha = sha256_file(weight_path)
    log("=" * 100)
    log(f"{DISPLAY_VERSION} | EXTRACT {encoder.upper()} | device={device}")
    log("=" * 100)
    log(f"Weight path : {weight_path}")
    log(f"Weight SHA  : {weight_sha}")
    log(f"Batch       : {batch_size}")
    log(f"Input mode  : {VLM_INPUT_MODE}")

    # Fail-fast model/input preflight on first uncached study.
    uids = train[UID_COLUMN].astype(str).tolist()
    remaining = [
        uid
        for uid in uids
        if not embedding_cache_file_usable(
            encoder_cache_path(paths, encoder, uid), uid, encoder
        )
    ]
    log(f"Already cached: {len(uids)-len(remaining)}/{len(uids)}")
    if not remaining:
        return

    started = time.time()
    done = 0
    failed: List[Dict[str, str]] = []

    for uid in remaining:
        try:
            shared_path = shared_study_path(paths, uid)
            if not shared_cache_file_usable(shared_path, uid):
                raise RuntimeError(f"Missing shared cache: {shared_path}")
            with np.load(shared_path, allow_pickle=False) as x:
                images = x["images"].copy()
                plane = x["plane"].astype(np.int8)
                fluid = x["fluid"].astype(np.int8)
                fs = x["fat_suppression"].astype(np.int8)
                position = x["slice_position"].astype(np.float32)

            pil_images = center_rgb_pil(images)
            features = encode_fn(model, preprocess, pil_images, device, batch_size)
            if features.shape != (len(images), expected_encoder_dim(encoder)):
                raise RuntimeError(
                    f"Unexpected {encoder} shape {tuple(features.shape)}"
                )

            pos_cpu = pos_proto.float().cpu()
            neg_cpu = neg_proto.float().cpu()
            semantic = features.float() @ pos_cpu.T - features.float() @ neg_cpu.T
            if semantic.shape != (len(images), NUM_LABELS):
                raise RuntimeError("Semantic score shape mismatch")
            if not torch.isfinite(semantic).all():
                raise RuntimeError("Semantic scores non-finite")

            atomic_npz_save(
                encoder_cache_path(paths, encoder, uid),
                cache_version=np.asarray(EMBED_CACHE_VERSION),
                study_uid=np.asarray(uid),
                encoder=np.asarray(encoder),
                encoder_sha256=np.asarray(weight_sha),
                input_mode=np.asarray(VLM_INPUT_MODE),
                embeddings=features.numpy().astype(np.float16),
                semantic=semantic.numpy().astype(np.float16),
                plane=plane,
                fluid=fluid,
                fat_suppression=fs,
                slice_position=position,
            )
            done += 1
        except Exception as exc:
            failed.append({UID_COLUMN: uid, "Error": repr(exc)})
            if len(failed) == 1:
                log(f"FIRST FAILURE uid={uid}: {repr(exc)}")
            if done + len(failed) >= 16 and len(failed) / (done + len(failed)) >= 0.5:
                raise RuntimeError(
                    f"Aborting {encoder} extraction: systemic failures. First={failed[0]}"
                ) from exc

        total_processed = done + len(failed)
        if (
            total_processed <= 5
            or total_processed % 100 == 0
            or total_processed == len(remaining)
        ):
            rate = total_processed / max(time.time() - started, 1e-9) * 60.0
            log(
                f"{encoder:10s} {total_processed:4d}/{len(remaining)} "
                f"ok={done:4d} fail={len(failed):3d} rate={rate:5.1f} studies/min"
            )

    if failed:
        pd.DataFrame(failed).to_csv(
            paths.result_root / f"02_{encoder}_extract_failures.csv", index=False
        )
        raise RuntimeError(f"{encoder} extraction failed for {len(failed)} studies")

    summary = summarize_embedding_cache(paths, train, encoder)
    summary.update(
        {
            "script_version": SCRIPT_VERSION,
            "created_at": now_iso(),
            "weight_path": str(weight_path),
            "weight_sha256": weight_sha,
            "input_mode": VLM_INPUT_MODE,
        }
    )
    write_json(paths.result_root / f"03_{encoder}_embedding_summary.json", summary)
    if not summary["complete"]:
        raise RuntimeError(f"{encoder} embedding cache incomplete")
    log(f"{encoder} extraction COMPLETE: {summary['usable_studies']}/{EXPECTED_TRAIN}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def paths_to_worker_dict(paths: Paths) -> Dict[str, Any]:
    return {
        field.name: (
            str(getattr(paths, field.name))
            if isinstance(getattr(paths, field.name), Path)
            else getattr(paths, field.name)
        )
        for field in dataclasses.fields(Paths)
    }


def run_extract(paths: Paths, args) -> Dict[str, Any]:
    paths.ensure_dirs()
    train, _gold, _unlabeled = load_train(paths)
    shared = summarize_shared_cache(paths, train)
    if not shared["complete"]:
        raise RuntimeError(
            f"Shared 2.5D cache incomplete: {shared['usable_studies']}/{EXPECTED_TRAIN}"
        )

    requested = args.encoder
    if requested not in {"mi2", "biomedclip", "both"}:
        raise ValueError(requested)

    if requested in {"mi2", "both"} and paths.mi2_root is None:
        raise RuntimeError(
            "MedImageInsight model root not found. Attach the model dataset or pass --mi2-root."
        )
    if requested in {"biomedclip", "both"} and paths.biomedclip_root is None:
        raise RuntimeError(
            "BiomedCLIP model root not found. Attach the model dataset or "
            "pass --biomedclip-root."
        )
    if requested in {"biomedclip", "both"} and paths.biomedbert_root is None:
        raise RuntimeError(
            "BiomedBERT text-tower root not found. Attach the local snapshot "
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract or pass "
            "--biomedbert-root."
        )
    if requested in {"biomedclip", "both"} and not _biomedbert_candidate_usable(
        paths.biomedbert_root
    ):
        raise RuntimeError(
            "BiomedBERT snapshot is incomplete. Expected config.json, vocab.txt, "
            "and pytorch_model.bin or model.safetensors."
        )

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    pdict = paths_to_worker_dict(paths)

    if requested == "both" and visible >= 2:
        log("Using independent GPUs: GPU0=MedImageInsight, GPU1=BiomedCLIP")
        ctx = mp.get_context("spawn")
        workers = [
            ctx.Process(
                target=_extract_one_encoder,
                args=("mi2", pdict, 0, args.mi2_batch, args.biomedclip_batch),
            ),
            ctx.Process(
                target=_extract_one_encoder,
                args=("biomedclip", pdict, 1, args.mi2_batch, args.biomedclip_batch),
            ),
        ]
        for p in workers:
            p.start()
        for p in workers:
            p.join()
        if any(p.exitcode != 0 for p in workers):
            raise RuntimeError(
                f"Embedding worker failed: exitcodes={[p.exitcode for p in workers]}"
            )
    else:
        encoders = [requested] if requested != "both" else ["mi2", "biomedclip"]
        gpu = 0 if visible else None
        for encoder in encoders:
            _extract_one_encoder(
                encoder, pdict, gpu, args.mi2_batch, args.biomedclip_batch
            )

    result = {
        "mi2": summarize_embedding_cache(paths, train, "mi2"),
        "biomedclip": summarize_embedding_cache(paths, train, "biomedclip"),
    }
    write_json(paths.result_root / "04_embedding_cache_summary.json", result)
    return result


# =============================================================================
# IN-RAM FEATURE STORE
# =============================================================================


def _select_balanced_indices(plane: np.ndarray, n: int, max_tokens: int) -> np.ndarray:
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
        remainder = np.array([i for i in range(n) if i not in set(selected)], dtype=int)
        need = min(max_tokens - len(selected), len(remainder))
        if need:
            chosen = np.linspace(0, len(remainder) - 1, need).round().astype(int)
            selected.extend(remainder[chosen].tolist())
    selected = sorted(selected[:max_tokens])
    return np.asarray(selected, dtype=np.int64)


def load_curia_record(path: Path, uid: str) -> Dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    mask = payload["slice_mask"].bool().numpy()
    position = payload["slice_position"].float().numpy()
    meta = payload["series_meta"].long().numpy()

    feat_rows, plane_rows, fluid_rows, fs_rows, pos_rows = [], [], [], [], []
    for s in range(features.shape[0]):
        valid = np.where(mask[s])[0]
        if len(valid) == 0:
            continue
        feat_rows.append(features[s, valid])
        plane_rows.append(np.full(len(valid), int(meta[s, 0]), dtype=np.int64))
        fluid_rows.append(np.full(len(valid), int(meta[s, 1]), dtype=np.int64))
        fs_rows.append(np.full(len(valid), int(meta[s, 2]), dtype=np.int64))
        pos_rows.append(position[s, valid].astype(np.float32))

    if not feat_rows:
        raise RuntimeError(f"Curia study has no valid tokens: {uid}")
    feat = np.concatenate(feat_rows, axis=0)
    plane = np.concatenate(plane_rows)
    fluid = np.concatenate(fluid_rows)
    fs = np.concatenate(fs_rows)
    pos = np.concatenate(pos_rows)
    selected = _select_balanced_indices(plane, len(feat), MAX_CURIA_TOKENS)
    return {
        "embeddings": feat[selected].astype(np.float16),
        "plane": plane[selected].astype(np.int8),
        "fluid": fluid[selected].astype(np.int8),
        "fat_suppression": fs[selected].astype(np.int8),
        "slice_position": pos[selected].astype(np.float16),
    }


class FeatureStore:
    def __init__(self, paths: Paths, train: pd.DataFrame, include_curia: bool):
        self.records: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.include_curia = include_curia
        started = time.time()
        log("Loading frozen embedding caches into CPU RAM once...")
        for i, uid in enumerate(train[UID_COLUMN].astype(str), start=1):
            rec: Dict[str, Dict[str, np.ndarray]] = {}
            for encoder in ("mi2", "biomedclip"):
                path = encoder_cache_path(paths, encoder, uid)
                if not embedding_cache_file_usable(path, uid, encoder):
                    raise RuntimeError(f"Missing {encoder} embedding cache for {uid}")
                with np.load(path, allow_pickle=False) as x:
                    plane = x["plane"].astype(np.int64)
                    selected = _select_balanced_indices(
                        plane, len(plane), MAX_VLM_TOKENS
                    )
                    rec[encoder] = {
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
            if include_curia:
                cpath = curia_study_path(paths, uid)
                if cpath is None or not curia_cache_file_usable(cpath, uid):
                    raise RuntimeError(f"Missing Curia cache for {uid}")
                rec["curia"] = load_curia_record(cpath, uid)
            self.records[uid] = rec
            if i % 500 == 0 or i == len(train):
                log(f"  loaded {i}/{len(train)} studies")
        log(f"FeatureStore ready in {time.time()-started:.1f}s")

    def get(self, uid: str) -> Dict[str, Dict[str, np.ndarray]]:
        return self.records[str(uid)]


# =============================================================================
# TARGET MAPS / DATASET / COLLATE
# =============================================================================


def build_target_maps(
    gold: pd.DataFrame,
    unlabeled: pd.DataFrame,
    probs: pd.DataFrame,
    weights: pd.DataFrame,
    masks: pd.DataFrame,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    gold_t, gold_m, gold_w = {}, {}, {}
    for _, row in gold.iterrows():
        uid = str(row[UID_COLUMN])
        gold_t[uid] = row[LABELS].to_numpy(np.float32)
        gold_m[uid] = np.ones(NUM_LABELS, dtype=bool)
        gold_w[uid] = np.ones(NUM_LABELS, dtype=np.float32)

    pidx = probs.set_index(UID_COLUMN)
    widx = weights.set_index(UID_COLUMN)
    midx = masks.set_index(UID_COLUMN)
    pseudo_t, pseudo_m, pseudo_w = {}, {}, {}
    for uid in unlabeled[UID_COLUMN].astype(str):
        pseudo_t[uid] = pidx.loc[uid, LABELS].to_numpy(np.float32)
        pseudo_m[uid] = midx.loc[uid, LABELS].to_numpy(bool)
        pseudo_w[uid] = widx.loc[uid, LABELS].to_numpy(np.float32)
    return gold_t, gold_m, gold_w, pseudo_t, pseudo_m, pseudo_w


class FusionDataset(Dataset):
    def __init__(
        self,
        store: FeatureStore,
        uids: Sequence[str],
        targets: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        weights: Mapping[str, np.ndarray],
        include_curia: bool,
    ):
        self.store = store
        self.uids = [str(u) for u in uids]
        self.targets = targets
        self.masks = masks
        self.weights = weights
        self.include_curia = include_curia

    def __len__(self):
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


def _pad_branch(
    batch: Sequence[Mapping[str, Any]], encoder: str
) -> Dict[str, torch.Tensor]:
    records = [item["features"][encoder] for item in batch]
    max_t = max(len(r["embeddings"]) for r in records)
    dim = records[0]["embeddings"].shape[1]
    b = len(records)
    embeddings = torch.zeros(b, max_t, dim, dtype=torch.float32)
    mask = torch.zeros(b, max_t, dtype=torch.bool)
    plane = torch.zeros(b, max_t, dtype=torch.long)
    fluid = torch.zeros(b, max_t, dtype=torch.long)
    fs = torch.zeros(b, max_t, dtype=torch.long)
    pos = torch.zeros(b, max_t, dtype=torch.float32)
    semantic = None
    if encoder in {"mi2", "biomedclip"}:
        semantic = torch.zeros(b, max_t, NUM_LABELS, dtype=torch.float32)

    for i, rec in enumerate(records):
        n = len(rec["embeddings"])
        embeddings[i, :n] = torch.from_numpy(rec["embeddings"].astype(np.float32))
        mask[i, :n] = True
        plane[i, :n] = torch.from_numpy(rec["plane"].astype(np.int64))
        fluid[i, :n] = torch.from_numpy(rec["fluid"].astype(np.int64))
        fs[i, :n] = torch.from_numpy(rec["fat_suppression"].astype(np.int64))
        pos[i, :n] = torch.from_numpy(rec["slice_position"].astype(np.float32))
        if semantic is not None:
            semantic[i, :n] = torch.from_numpy(rec["semantic"].astype(np.float32))

    out = {
        "embeddings": embeddings,
        "mask": mask,
        "plane": plane,
        "fluid": fluid,
        "fat_suppression": fs,
        "slice_position": pos,
    }
    if semantic is not None:
        out["semantic"] = semantic
    return out


def collate_fusion(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    include_curia = "curia" in batch[0]["features"]
    out: Dict[str, Any] = {
        UID_COLUMN: [str(item[UID_COLUMN]) for item in batch],
        "mi2": _pad_branch(batch, "mi2"),
        "biomedclip": _pad_branch(batch, "biomedclip"),
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
    if include_curia:
        out["curia"] = _pad_branch(batch, "curia")
    return out


def move_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


# =============================================================================
# LABEL-AWARE FUSION HEAD
# =============================================================================


def position_features(position: torch.Tensor) -> torch.Tensor:
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


class BranchAggregator(nn.Module):
    def __init__(self, input_dim: int, has_semantic: bool):
        super().__init__()
        self.has_semantic = has_semantic
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, HIDDEN_DIM)
        self.plane_embedding = nn.Embedding(3, PLANE_EMBED_DIM)
        self.fluid_embedding = nn.Embedding(2, BINARY_META_DIM)
        self.fs_embedding = nn.Embedding(2, BINARY_META_DIM)
        self.position_projection = nn.Linear(9, POSITION_DIM)
        self.meta_projection = nn.Linear(
            PLANE_EMBED_DIM + 2 * BINARY_META_DIM + POSITION_DIM,
            HIDDEN_DIM,
        )
        self.token_norm = nn.LayerNorm(HIDDEN_DIM)
        self.queries = nn.Parameter(torch.randn(NUM_LABELS, HIDDEN_DIM) * 0.02)
        self.dropout = nn.Dropout(HEAD_DROPOUT)

    def forward(
        self, branch: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                self.position_projection(position_features(pos)),
            ],
            dim=-1,
        )
        tokens = self.token_norm(visual + self.meta_projection(meta))
        tokens = self.dropout(tokens)

        scores = torch.einsum("bth,lh->blt", tokens, self.queries) / math.sqrt(
            HIDDEN_DIM
        )
        plane_prior = PLANE_PRIOR_LOG.to(scores.device)
        scores = scores + plane_prior[:, plane].permute(1, 0, 2)
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        attention = torch.softmax(scores.float(), dim=-1).to(tokens.dtype)
        pooled = torch.einsum("blt,bth->blh", attention, tokens)

        semantic_pooled = None
        if self.has_semantic:
            semantic = branch["semantic"]  # [B,T,L]
            semantic_by_label = semantic.permute(0, 2, 1)  # [B,L,T]
            semantic_pooled = (attention.float() * semantic_by_label.float()).sum(
                dim=-1
            )
        return pooled, semantic_pooled


class FusionHead(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.mi2 = BranchAggregator(MI2_DIM, True)
        self.biomed = BranchAggregator(BIOMEDCLIP_DIM, True)
        self.curia = (
            BranchAggregator(CURIA_DIM, False) if variant == CURIA_VARIANT else None
        )

        if variant == "mi2":
            classifier_dim = HIDDEN_DIM + 1
            self.fusion = None
        elif variant == "biomedclip":
            classifier_dim = HIDDEN_DIM + 1
            self.fusion = None
        elif variant == "mean2":
            classifier_dim = HIDDEN_DIM + 1
            self.fusion = None
        elif variant == "weighted2":
            classifier_dim = HIDDEN_DIM + 1
            self.fusion_logits = nn.Parameter(torch.zeros(NUM_LABELS, 2))
            self.fusion = None
        elif variant == "concat2":
            self.fusion = nn.Sequential(
                nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(HEAD_DROPOUT),
                nn.LayerNorm(HIDDEN_DIM),
            )
            classifier_dim = HIDDEN_DIM + 2
        elif variant == CURIA_VARIANT:
            self.fusion = nn.Sequential(
                nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(HEAD_DROPOUT),
                nn.LayerNorm(HIDDEN_DIM),
            )
            classifier_dim = HIDDEN_DIM + 2
        else:
            raise ValueError(variant)

        self.classifier_weight = nn.Parameter(
            torch.randn(NUM_LABELS, classifier_dim) * 0.02
        )
        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        mi, mi_sem = self.mi2(batch["mi2"])
        bc, bc_sem = self.biomed(batch["biomedclip"])

        if self.variant == "mi2":
            fused = mi
            sem = mi_sem.unsqueeze(-1)
        elif self.variant == "biomedclip":
            fused = bc
            sem = bc_sem.unsqueeze(-1)
        elif self.variant == "mean2":
            fused = 0.5 * (mi + bc)
            sem = (0.5 * (mi_sem + bc_sem)).unsqueeze(-1)
        elif self.variant == "weighted2":
            weights = torch.softmax(self.fusion_logits, dim=-1)  # [L,2]
            fused = mi * weights[None, :, 0, None] + bc * weights[None, :, 1, None]
            sem_scalar = mi_sem * weights[None, :, 0] + bc_sem * weights[None, :, 1]
            sem = sem_scalar.unsqueeze(-1)
        elif self.variant == "concat2":
            fused = self.fusion(torch.cat([mi, bc], dim=-1))
            sem = torch.stack([mi_sem, bc_sem], dim=-1)
        elif self.variant == CURIA_VARIANT:
            if "curia" not in batch:
                raise RuntimeError("concat3_curia requested but Curia branch absent")
            cu, _ = self.curia(batch["curia"])
            fused = self.fusion(torch.cat([mi, bc, cu], dim=-1))
            sem = torch.stack([mi_sem, bc_sem], dim=-1)
        else:
            raise AssertionError(self.variant)

        features = torch.cat([fused, sem.to(fused.dtype)], dim=-1)
        logits = (features * self.classifier_weight[None, :, :]).sum(dim=-1)
        logits = logits + self.classifier_bias[None, :]
        return logits


# =============================================================================
# LOSSES / TRAIN / PREDICT
# =============================================================================


def pseudo_macro_loss(logits, targets, mask, weights) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    eff = mask.float() * weights.float()
    denom = eff.sum(dim=0)
    active = denom > 0
    if not active.any():
        raise RuntimeError("Pseudo batch has no active cells")
    per_label = (cell * eff).sum(dim=0) / denom.clamp_min(1e-6)
    loss = per_label[active].mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Pseudo loss is non-finite")
    return loss


def gold_pos_weight(frame: pd.DataFrame) -> torch.Tensor:
    y = frame[LABELS].to_numpy(np.float64)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = np.clip(neg / np.maximum(pos, 1.0), 1.0, 5.0)
    return torch.tensor(w.astype(np.float32))


def gold_macro_loss(logits, targets, pos_weight) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none", pos_weight=pos_weight.float()
    )
    loss = cell.mean(dim=0).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Gold loss is non-finite")
    return loss


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fusion,
        drop_last=False,
    )


def train_pseudo_head(
    model: FusionHead,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> List[Dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PSEUDO_LR, weight_decay=WEIGHT_DECAY
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        started = time.time()
        for raw in loader:
            batch = move_to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits in pseudo head training")
            loss = pseudo_macro_loss(
                logits, batch["targets"], batch["target_mask"], batch["target_weight"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(raw[UID_COLUMN])
            n += len(raw[UID_COLUMN])
        value = total / max(n, 1)
        history.append(
            {"epoch": epoch, "loss": value, "seconds": time.time() - started}
        )
        log(f"    pseudo epoch {epoch:02d}/{epochs} loss={value:.6f}")
    return history


def adapt_gold_head(
    model: FusionHead,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    epochs: int,
) -> List[Dict[str, Any]]:
    model.to(device)
    pos_weight = pos_weight.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GOLD_LR, weight_decay=WEIGHT_DECAY
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for raw in loader:
            batch = move_to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = gold_macro_loss(logits, batch["targets"], pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(raw[UID_COLUMN])
            n += len(raw[UID_COLUMN])
        value = total / max(n, 1)
        history.append({"epoch": epoch, "loss": value})
        if epoch in {1, epochs}:
            log(f"      gold epoch {epoch:02d}/{epochs} loss={value:.6f}")
    return history


@torch.inference_mode()
def predict_head(
    model: FusionHead, loader: DataLoader, device: torch.device
) -> Tuple[List[str], np.ndarray]:
    model.to(device)
    model.eval()
    uids, probs = [], []
    for raw in loader:
        batch = move_to_device(raw, device)
        logits = model(batch)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        uids.extend(raw[UID_COLUMN])
        probs.append(p)
    return uids, np.concatenate(probs, axis=0)


# =============================================================================
# METRICS
# =============================================================================


def metrics_from_long(
    oof: pd.DataFrame, variant: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for label in LABELS:
        x = oof[oof["Label"] == label]
        y = x["Gold"].to_numpy(int)
        p = x["Probability"].to_numpy(float)
        rows.append(
            {
                "Variant": variant,
                "Label": label,
                "N": len(x),
                "Positive": int(y.sum()),
                "AUROC": safe_auc(y, p),
                "AP": safe_ap(y, p),
                "Brier": float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            }
        )
    table = pd.DataFrame(rows)
    finite_auc = table["AUROC"].dropna().to_numpy(float)
    finite_ap = table["AP"].dropna().to_numpy(float)
    summary = {
        "Variant": variant,
        "Macro_AUROC": float(np.mean(finite_auc)),
        "Macro_AP": float(np.mean(finite_ap)),
        "Macro_Brier": float(table["Brier"].mean()),
        "WeakLabelsLT055": int((table["AUROC"] < WEAK_LABEL_AUC).sum()),
        "WeakLabels": table.loc[table["AUROC"] < WEAK_LABEL_AUC, "Label"].tolist(),
    }
    return table, summary


def oof_matrix(oof: pd.DataFrame, gold: pd.DataFrame) -> np.ndarray:
    wide = oof.pivot(index=UID_COLUMN, columns="Label", values="Probability")
    wide = wide.reindex(index=gold[UID_COLUMN].astype(str), columns=LABELS)
    return wide.to_numpy(float)


def build_long_predictions(
    uids: Sequence[str],
    fold: int,
    probs: np.ndarray,
    gold_lookup: Mapping[str, np.ndarray],
    variant: str,
) -> pd.DataFrame:
    rows = []
    for uid, p in zip(uids, probs):
        y = gold_lookup[str(uid)]
        for j, label in enumerate(LABELS):
            rows.append(
                {
                    UID_COLUMN: str(uid),
                    "OuterFold": int(fold),
                    "Variant": variant,
                    "Label": label,
                    "Gold": int(y[j]),
                    "Probability": float(p[j]),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# CV EXPERIMENT
# =============================================================================


def pseudo_checkpoint_path(paths: Paths, variant: str) -> Path:
    return paths.checkpoint_root / variant / "pseudo_head.pt"


def variant_config(
    variant: str, include_curia: bool, teacher_info: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "variant": variant,
        "embed_cache_version": EMBED_CACHE_VERSION,
        "shared_cache_version": SHARED_CACHE_VERSION,
        "curia_cache_version": CURIA_CACHE_VERSION if include_curia else None,
        "input_mode": VLM_INPUT_MODE,
        "hidden_dim": HIDDEN_DIM,
        "pseudo_epochs": PSEUDO_EPOCHS,
        "gold_adapt_epochs": GOLD_ADAPT_EPOCHS,
        "pseudo_lr": PSEUDO_LR,
        "gold_lr": GOLD_LR,
        "teacher": dict(teacher_info),
        "fold_sha256": EXPECTED_FOLD_SHA256,
    }


def train_or_load_pseudo_variant(
    paths: Paths,
    variant: str,
    include_curia: bool,
    store: FeatureStore,
    unlabeled: pd.DataFrame,
    pseudo_t: Mapping[str, np.ndarray],
    pseudo_m: Mapping[str, np.ndarray],
    pseudo_w: Mapping[str, np.ndarray],
    teacher_info: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = pseudo_checkpoint_path(paths, variant)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    config = variant_config(variant, include_curia, teacher_info)
    config_hash = stable_json_hash(config)

    if ckpt.is_file():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if payload.get("config_hash") == config_hash:
            log(f"  {variant}: reusable pseudo checkpoint found")
            return payload
        raise RuntimeError(
            f"Existing {ckpt} has incompatible config. Use a new W44 output root."
        )

    seed_everything(
        BASE_SEED + 100 + VARIANTS_BASE.index(variant)
        if variant in VARIANTS_BASE
        else BASE_SEED + 199
    )
    model = FusionHead(variant)
    dataset = FusionDataset(
        store,
        unlabeled[UID_COLUMN].astype(str).tolist(),
        pseudo_t,
        pseudo_m,
        pseudo_w,
        include_curia,
    )
    loader = make_loader(dataset, PSEUDO_BATCH, True)
    log(f"  {variant}: pseudo pretrain")
    history = train_pseudo_head(model, loader, device, PSEUDO_EPOCHS)
    payload = {
        "script_version": SCRIPT_VERSION,
        "variant": variant,
        "config": config,
        "config_hash": config_hash,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "history": history,
        "created_at": now_iso(),
    }
    torch.save(payload, ckpt)
    return payload


def run_train_cv(paths: Paths, args) -> Dict[str, Any]:
    paths.ensure_dirs()
    train, gold, unlabeled = load_train(paths)
    folds = load_folds(paths, gold)
    probs, weights, masks, teacher_info = load_w40_teacher(paths, unlabeled)

    mi_summary = summarize_embedding_cache(paths, train, "mi2")
    bc_summary = summarize_embedding_cache(paths, train, "biomedclip")
    if not mi_summary["complete"] or not bc_summary["complete"]:
        raise RuntimeError(
            "Frozen embedding caches are incomplete. Run `extract --encoder both` first."
        )

    curia_summary = summarize_curia_cache(paths, train)
    include_curia = bool(curia_summary["complete"])
    variants = list(VARIANTS_BASE) + ([CURIA_VARIANT] if include_curia else [])

    log("=" * 100)
    log(f"{DISPLAY_VERSION} | LOCKED 5-FOLD FUSION DIAGNOSTIC")
    log("=" * 100)
    log(
        "IMPORTANT: NON-PRISTINE diagnostic because W40 production teacher used all 58 gold exemplars."
    )
    log(f"Variants: {variants}")
    log(f"Curia complete: {include_curia}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu"
    )
    store = FeatureStore(paths, train, include_curia=include_curia)

    gold_t, gold_m, gold_w, pseudo_t, pseudo_m, pseudo_w = build_target_maps(
        gold, unlabeled, probs, weights, masks
    )
    gold_lookup = {
        str(row[UID_COLUMN]): row[LABELS].to_numpy(np.int64)
        for _, row in gold.iterrows()
    }

    # Pseudo-pretrain each variant exactly once.
    pseudo_payloads: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        pseudo_payloads[variant] = train_or_load_pseudo_variant(
            paths,
            variant,
            include_curia=(variant == CURIA_VARIANT),
            store=store,
            unlabeled=unlabeled,
            pseudo_t=pseudo_t,
            pseudo_m=pseudo_m,
            pseudo_w=pseudo_w,
            teacher_info=teacher_info,
            device=device,
        )

    fold_map = folds.set_index(UID_COLUMN)["OuterFold"].to_dict()
    all_variant_oof: Dict[str, pd.DataFrame] = {}
    all_metric_tables = []
    summaries = []

    for variant in variants:
        variant_root = paths.result_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        oof_parts = []
        log("-" * 100)
        log(f"VARIANT: {variant}")
        log("-" * 100)

        for fold in range(1, 6):
            pred_path = variant_root / f"fold_{fold}_predictions.csv"
            hist_path = variant_root / f"fold_{fold}_history.csv"
            if pred_path.is_file():
                cached = pd.read_csv(pred_path)
                if len(cached) == int((folds["OuterFold"] == fold).sum()) * NUM_LABELS:
                    log(f"  fold {fold}: reusable predictions found")
                    oof_parts.append(cached)
                    continue

            val_uids = [
                uid
                for uid in gold[UID_COLUMN].astype(str)
                if int(fold_map[uid]) == fold
            ]
            train_uids = [
                uid
                for uid in gold[UID_COLUMN].astype(str)
                if int(fold_map[uid]) != fold
            ]
            gold_train_df = gold[gold[UID_COLUMN].isin(train_uids)].copy()

            seed = BASE_SEED + fold + (1000 * variants.index(variant))
            seed_everything(seed)
            model = FusionHead(variant)
            model.load_state_dict(pseudo_payloads[variant]["model_state"], strict=True)

            train_ds = FusionDataset(
                store, train_uids, gold_t, gold_m, gold_w, variant == CURIA_VARIANT
            )
            val_ds = FusionDataset(
                store, val_uids, gold_t, gold_m, gold_w, variant == CURIA_VARIANT
            )
            train_loader = make_loader(train_ds, GOLD_BATCH, True)
            val_loader = make_loader(val_ds, GOLD_BATCH, False)
            pos_weight = gold_pos_weight(gold_train_df)

            log(
                f"  fold {fold} train={len(train_uids)} val={len(val_uids)} seed={seed}"
            )
            history = adapt_gold_head(
                model, train_loader, device, pos_weight, GOLD_ADAPT_EPOCHS
            )
            uids_out, pred = predict_head(model, val_loader, device)
            long = build_long_predictions(uids_out, fold, pred, gold_lookup, variant)
            long.to_csv(pred_path, index=False)
            pd.DataFrame(history).to_csv(hist_path, index=False)
            oof_parts.append(long)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oof = pd.concat(oof_parts, ignore_index=True)
        if len(oof) != EXPECTED_GOLD * NUM_LABELS:
            raise RuntimeError(f"OOF size mismatch for {variant}: {len(oof)}")
        all_variant_oof[variant] = oof
        oof.to_csv(variant_root / "oof_predictions.csv", index=False)
        table, summary = metrics_from_long(oof, variant)
        table.to_csv(variant_root / "per_label_metrics.csv", index=False)
        write_json(variant_root / "summary.json", summary)
        all_metric_tables.append(table)
        summaries.append(summary)
        log(
            f"  {variant}: macro AUC={summary['Macro_AUROC']:.6f} "
            f"AP={summary['Macro_AP']:.6f} Brier={summary['Macro_Brier']:.6f} "
            f"weak={summary['WeakLabelsLT055']}"
        )

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values("Macro_AUROC", ascending=False)
        .reset_index(drop=True)
    )
    metrics_df = pd.concat(all_metric_tables, ignore_index=True)
    summary_df.to_csv(paths.result_root / "30_variant_summary.csv", index=False)
    metrics_df.to_csv(paths.result_root / "31_all_per_label_metrics.csv", index=False)

    # Fixed top-2 probability mean. Selection is by global OOF macro AUC, so this
    # is diagnostic, not an unbiased estimate.
    top2 = summary_df["Variant"].head(2).tolist()
    ensemble_rows = None
    if len(top2) == 2:
        a = all_variant_oof[top2[0]].copy()
        b = all_variant_oof[top2[1]].copy()
        key = [UID_COLUMN, "OuterFold", "Label", "Gold"]
        merged = a[key + ["Probability"]].merge(
            b[key + ["Probability"]],
            on=key,
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        merged["Probability"] = 0.5 * (
            merged["Probability_a"] + merged["Probability_b"]
        )
        merged["Variant"] = f"probmean({top2[0]}+{top2[1]})"
        ensemble_rows = merged[key + ["Variant", "Probability"]]
        etable, esummary = metrics_from_long(
            ensemble_rows, str(ensemble_rows["Variant"].iloc[0])
        )
        ensemble_rows.to_csv(
            paths.result_root / "32_top2_probmean_oof.csv", index=False
        )
        etable.to_csv(paths.result_root / "33_top2_probmean_per_label.csv", index=False)
        write_json(paths.result_root / "34_top2_probmean_summary.json", esummary)

    # Per-label oracle upper bound across variants. Explicitly NOT deployable and
    # not a valid leaderboard estimate; it tells us whether representation
    # complementarity exists.
    oracle_rows = []
    oracle_selection = []
    for label in LABELS:
        candidates = metrics_df[metrics_df["Label"] == label].sort_values(
            "AUROC", ascending=False
        )
        best_variant = str(candidates.iloc[0]["Variant"])
        best_auc = float(candidates.iloc[0]["AUROC"])
        oracle_selection.append(
            {"Label": label, "BestVariant": best_variant, "AUROC": best_auc}
        )
        oracle_rows.append(
            all_variant_oof[best_variant][
                all_variant_oof[best_variant]["Label"] == label
            ]
        )
    oracle_oof = pd.concat(oracle_rows, ignore_index=True)
    oracle_oof["Variant"] = "ORACLE_per_label_NOT_DEPLOYABLE"
    oracle_table, oracle_summary = metrics_from_long(
        oracle_oof, "ORACLE_per_label_NOT_DEPLOYABLE"
    )
    pd.DataFrame(oracle_selection).to_csv(
        paths.result_root / "35_oracle_label_selection.csv", index=False
    )
    oracle_oof.to_csv(
        paths.result_root / "36_oracle_oof_NOT_DEPLOYABLE.csv", index=False
    )
    write_json(
        paths.result_root / "37_oracle_summary_NOT_DEPLOYABLE.json", oracle_summary
    )

    best = summary_df.iloc[0].to_dict()
    best_auc = float(best["Macro_AUROC"])
    delta_curia = best_auc - CURIA_REFERENCE_AUC
    if best_auc >= STRONG_GO_AUC:
        verdict = "STRONG_GO_TO_W45"
    elif best_auc >= GO_AUC:
        verdict = "GO_TO_W45"
    elif best_auc >= REVIEW_AUC:
        verdict = "REVIEW"
    else:
        verdict = "STOP_MULTIMODEL_BRANCH"

    final = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "diagnostic_non_pristine": True,
        "variants": variants,
        "curia_included": include_curia,
        "best_variant": str(best["Variant"]),
        "best_macro_AUROC": best_auc,
        "best_macro_AP": float(best["Macro_AP"]),
        "best_macro_Brier": float(best["Macro_Brier"]),
        "curia_reference_AUROC": CURIA_REFERENCE_AUC,
        "delta_vs_curia_reference": delta_curia,
        "top2_variants": top2,
        "oracle_macro_AUROC_NOT_DEPLOYABLE": float(oracle_summary["Macro_AUROC"]),
        "verdict": verdict,
    }
    write_json(paths.result_root / "38_w44_final_diagnostic.json", final)

    log("=" * 100)
    log("W44 FINAL DIAGNOSTIC")
    log("=" * 100)
    print(summary_df.to_string(index=False))
    log(f"Best variant          : {final['best_variant']}")
    log(f"Best macro AUROC      : {best_auc:.6f}")
    log(f"Delta vs Curia .702997: {delta_curia:+.6f}")
    log(
        f"Oracle upper bound    : {final['oracle_macro_AUROC_NOT_DEPLOYABLE']:.6f} (NOT DEPLOYABLE)"
    )
    log(f"VERDICT               : {verdict}")
    return final


# =============================================================================
# STATUS / VALIDATE
# =============================================================================


def dependency_report() -> Dict[str, Any]:
    modules = [
        "torch",
        "torchvision",
        "transformers",
        "open_clip",
        "timm",
        "yaml",
        "einops",
        "ftfy",
        "fvcore",
        "mup",
        "sentencepiece",
        "safetensors",
    ]
    return {name: dependency_available(name) for name in modules}


def mi2_asset_report(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {"root": None, "complete": False}
    required = {
        "code": root / "MedImageInsight" / "UniCLModel.py",
        "config": root / "2024.09.27" / "config.yaml",
        "vision_weight": root
        / "2024.09.27"
        / "vision_model"
        / "medimageinsigt-v1.0.0.pt",
        "tokenizer": root / "2024.09.27" / "language_model" / "clip_tokenizer_4.16.2",
    }
    return {
        "root": str(root),
        "complete": all(p.exists() for p in required.values()),
        "files": {k: str(v) for k, v in required.items()},
        "vision_weight_size_gb": (
            required["vision_weight"].stat().st_size / (1024**3)
            if required["vision_weight"].is_file()
            else None
        ),
        "native_input_size": 480,
        "embedding_dim": MI2_DIM,
    }


def biomed_asset_report(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {"root": None, "complete": False}
    config = root / "open_clip_config.json"
    weight = root / "open_clip_pytorch_model.bin"
    required_tokenizer = [root / "tokenizer.json", root / "vocab.txt"]
    complete = (
        config.is_file()
        and weight.is_file()
        and any(p.is_file() for p in required_tokenizer)
    )
    return {
        "root": str(root),
        "complete": complete,
        "config": str(config),
        "weight": str(weight),
        "weight_size_gb": (
            weight.stat().st_size / (1024**3) if weight.is_file() else None
        ),
        "native_input_size": 224,
        "embedding_dim": BIOMEDCLIP_DIM,
    }


def biomedbert_asset_report(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {
            "root": None,
            "complete": False,
            "required_repo": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        }

    config = root / "config.json"
    vocab = root / "vocab.txt"
    weight_candidates = [
        root / "pytorch_model.bin",
        root / "model.safetensors",
    ]
    weight = next((p for p in weight_candidates if p.is_file()), None)

    return {
        "root": str(root),
        "complete": _biomedbert_candidate_usable(root),
        "required_repo": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "config": str(config),
        "vocab": str(vocab),
        "weight": str(weight) if weight is not None else None,
        "weight_size_gb": (
            weight.stat().st_size / (1024**3) if weight is not None else None
        ),
    }


def run_status(paths: Paths, args) -> Dict[str, Any]:
    deps = dependency_report()
    data_info: Dict[str, Any] = {}
    data_error = None
    try:
        train, gold, unlabeled = load_train(paths)
        folds = load_folds(paths, gold)
        _p, _w, _m, teacher = load_w40_teacher(paths, unlabeled)
        shared = summarize_shared_cache(paths, train)
        curia = summarize_curia_cache(paths, train)
        mi_cache = summarize_embedding_cache(paths, train, "mi2")
        bc_cache = summarize_embedding_cache(paths, train, "biomedclip")
        data_info = {
            "train": len(train),
            "gold": len(gold),
            "unlabeled": len(unlabeled),
            "fold_sha256": fold_assignment_sha256(folds),
            "teacher": teacher,
        }
    except Exception as exc:
        data_error = repr(exc)
        train = pd.DataFrame({UID_COLUMN: []})
        shared = {"complete": False}
        curia = {"complete": False}
        mi_cache = {"complete": False}
        bc_cache = {"complete": False}

    mi_assets = mi2_asset_report(paths.mi2_root)
    bc_assets = biomed_asset_report(paths.biomedclip_root)
    bert_assets = biomedbert_asset_report(paths.biomedbert_root)

    mi_dep_names = [
        "torch",
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
    bc_dep_names = ["torch", "transformers", "open_clip"]
    mi_deps_ok = all(deps.get(name, False) for name in mi_dep_names)
    bc_deps_ok = all(deps.get(name, False) for name in bc_dep_names)

    payload = {
        "script_version": SCRIPT_VERSION,
        "experiment": DISPLAY_VERSION,
        "runtime": {
            "requested_accelerator": args.accelerator,
            "cuda_available": torch.cuda.is_available(),
            "visible_cuda_devices": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "cuda_devices": (
                [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "compute_capability": list(torch.cuda.get_device_capability(i)),
                    }
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else []
            ),
            "extraction_policy": "gpu0_mi2_gpu1_biomedclip_when_two_gpus",
            "head_training_policy": "single_gpu_fp32",
        },
        "dependencies": deps,
        "paths": {
            "project_root": str(paths.project_root),
            "data_root": str(paths.data_root),
            "w40_root": str(paths.w40_root),
            "fold_csv": str(paths.fold_csv),
            "shared_cache_root": str(paths.shared_cache_root),
            "mi2_root": str(paths.mi2_root) if paths.mi2_root else None,
            "biomedclip_root": (
                str(paths.biomedclip_root) if paths.biomedclip_root else None
            ),
            "biomedbert_root": (
                str(paths.biomedbert_root) if paths.biomedbert_root else None
            ),
            "curia_study_root": (
                str(paths.curia_study_root) if paths.curia_study_root else None
            ),
            "output_root": str(paths.output_root),
        },
        "data": {"info": data_info, "error": data_error},
        "shared_2p5d_cache": shared,
        "curia_cache": curia,
        "medimageinsight": {
            "assets": mi_assets,
            "dependencies_ok": mi_deps_ok,
            "native_input_size": 480,
            "embedding_dim": 1024,
            "source_model_package": "attached open-weight MedImageInsight repository",
        },
        "biomedclip": {
            "assets": bc_assets,
            "biomedbert_text_tower_assets": bert_assets,
            "dependencies_ok": bc_deps_ok,
            "offline_text_tower_ready": bert_assets.get("complete") is True,
            "native_input_size": 224,
            "embedding_dim": 512,
            "source_model_package": "Microsoft BiomedCLIP OpenCLIP local files",
            "text_tower_repo": ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"),
        },
        "embedding_cache": {
            "mi2": mi_cache,
            "biomedclip": bc_cache,
            "input_mode": VLM_INPUT_MODE,
        },
        "fusion": {
            "base_variants": VARIANTS_BASE,
            "curia_variant": CURIA_VARIANT,
            "concat3_enabled": bool(curia.get("complete")),
            "large_encoders_frozen": True,
            "w40_reused": True,
            "dino": False,
            "resnet": False,
            "radimagenet": False,
        },
        "ready_for_extract": bool(
            data_error is None
            and shared.get("complete") is True
            and mi_assets.get("complete") is True
            and bc_assets.get("complete") is True
            and bert_assets.get("complete") is True
            and mi_deps_ok
            and bc_deps_ok
        ),
        "ready_for_train_cv": bool(
            data_error is None
            and mi_cache.get("complete") is True
            and bc_cache.get("complete") is True
        ),
        "install_hint": (
            "python -m pip install -q open_clip_torch ftfy fvcore mup sentencepiece safetensors"
        ),
    }
    print(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def run_validate(paths: Paths, args) -> Dict[str, Any]:
    train, gold, unlabeled = load_train(paths)
    folds = load_folds(paths, gold)
    _p, _w, _m, teacher = load_w40_teacher(paths, unlabeled)
    checks = {
        "train_4407": len(train) == EXPECTED_TRAIN,
        "gold_58": len(gold) == EXPECTED_GOLD,
        "unlabeled_4349": len(unlabeled) == EXPECTED_UNLABELED,
        "fold_sha": fold_assignment_sha256(folds) == EXPECTED_FOLD_SHA256,
        "teacher_32027": teacher["selected_cells"] == EXPECTED_SELECTED_CELLS,
        "shared_cache_complete": summarize_shared_cache(paths, train)["complete"],
        "mi2_embeddings_complete": summarize_embedding_cache(paths, train, "mi2")[
            "complete"
        ],
        "biomedclip_embeddings_complete": summarize_embedding_cache(
            paths, train, "biomedclip"
        )["complete"],
    }
    final_path = paths.result_root / "38_w44_final_diagnostic.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text())
        checks["cv_final_present"] = True
        checks["cv_best_auc_finite"] = math.isfinite(float(final["best_macro_AUROC"]))
    else:
        checks["cv_final_present"] = False
        checks["cv_best_auc_finite"] = False
    payload = {
        "script_version": SCRIPT_VERSION,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }
    write_json(paths.result_root / "99_validation.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DISPLAY_VERSION)
    parser.add_argument(
        "mode", choices=["status", "extract", "train_cv", "run_all", "validate"]
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "localGPU", "kaggle_t4", "apple_mps", "cpu"],
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--w40-root", default=None)
    parser.add_argument("--fold-csv", default=None)
    parser.add_argument("--shared-cache-root", default=None)
    parser.add_argument("--mi2-root", default=None)
    parser.add_argument("--biomedclip-root", default=None)
    parser.add_argument("--biomedbert-root", default=None)
    parser.add_argument("--curia-cache-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--encoder", default="both", choices=["mi2", "biomedclip", "both"]
    )
    parser.add_argument("--mi2-batch", type=int, default=MI2_DEFAULT_BATCH)
    parser.add_argument(
        "--biomedclip-batch", type=int, default=BIOMEDCLIP_DEFAULT_BATCH
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accelerator == "kaggle_t4" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator kaggle_t4 requested but CUDA is unavailable")
    if args.accelerator == "apple_mps":
        raise RuntimeError(
            "W44 extraction currently supports CUDA or CPU; apple_mps is not implemented"
        )

    paths = Paths.discover(args)
    paths.ensure_dirs()

    if args.mode == "status":
        run_status(paths, args)
    elif args.mode == "extract":
        run_extract(paths, args)
    elif args.mode == "train_cv":
        run_train_cv(paths, args)
    elif args.mode == "run_all":
        status = run_status(paths, args)
        if not status["ready_for_extract"] and not status["ready_for_train_cv"]:
            raise RuntimeError("W44 status is not ready for extraction/training")
        if not status["ready_for_train_cv"]:
            run_extract(paths, args)
        run_train_cv(paths, args)
    elif args.mode == "validate":
        run_validate(paths, args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    mp.freeze_support()
    main()
