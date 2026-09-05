#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W7_v1 — Label-Aware Evidence-Routed MIL

Purpose
-------
A single controlled experiment using ONLY the canonical Curia CLS feature cache.

W7_v1 deliberately does NOT:
- read train_series DICOMs
- run Curia
- use W6.1 spatial patch caches
- implement hidden-test submission
- call/import another project Python script

It consumes data artifacts only:
- train.csv
- canonical W4 Curia CLS cache
- W2.6-P production teacher outputs
- an existing locked-fold artifact

Experiment hypothesis
---------------------
Weak study-level supervision is better matched by sparse evidence MIL than by
high-capacity spatial patch attention.

File version
------------
rsna_w7_label_aware_mil_v1.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score

# =============================================================================
# VERSION / LOCKED PROJECT CONSTANTS
# =============================================================================

SCRIPT_VERSION = "w7_label_aware_mil_v1"

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

EXPECTED_W4_CACHE_VERSION = (
    "w4_0_curia2_cls_allseries_24slice_" "canonical_orientation_slow_processor_v1"
)

LOCKED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

N_LABELS = len(LABEL_COLUMNS)
CURIA_DIM = 768

# -------------------------------------------------------------------------
# One experiment. Fixed architecture/training defaults.
# -------------------------------------------------------------------------

HIDDEN_DIM = 256
DROPOUT = 0.10

SLICE_TOPK = 4
SERIES_TOPK = 2

EPOCHS = 24
STEPS_PER_EPOCH = 32

GOLD_BATCH_SIZE = 16
PSEUDO_BATCH_SIZE = 64

MAX_LR = 8e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0

GOLD_AUTHORITY = 8.0
PSEUDO_AUTHORITY = 1.0

FEATURE_LRU_SIZE = 512

CV_FOLDS = [1, 2, 3, 4, 5]
CV_SEED_BASE = 70000
FULL_SEEDS = [7001, 7002, 7003]

# -------------------------------------------------------------------------
# Conservative experiment gate.
#
# W2.6 teacher leakage means the diagnostic is not pristine OOF.
# Therefore W7 must show a material improvement before spending a
# leaderboard submission.
# -------------------------------------------------------------------------

GO_MIN_MACRO_AUC_WITHOUT_REFERENCE = 0.720
GO_MIN_DELTA_VS_W60 = 0.015
REVIEW_MIN_DELTA_VS_W60 = 0.005

MAX_WEAK_LABELS_FOR_GO = 2
WEAK_LABEL_AUC = 0.55


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def utc_timestamp() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def safe_json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p.resolve()

    return None


# =============================================================================
# PATH DISCOVERY
# =============================================================================


def discover_project_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")

    script_dir = Path(__file__).resolve().parent

    for p in [script_dir] + list(script_dir.parents):
        if (p / "input" / "train.csv").exists():
            return p

    # Same local layout used throughout this project:
    # PROJECT_ROOT/src/models/script.py
    candidate = (script_dir / ".." / "..").resolve()

    if candidate.exists():
        return candidate

    return script_dir


def kaggle_dataset_roots() -> List[Path]:
    """
    Enumerate dataset roots WITHOUT recursively traversing the 570GB
    competition train_series tree.
    """

    base = Path("/kaggle/input")
    roots: List[Path] = []

    if not base.exists():
        return roots

    try:
        for p in base.iterdir():
            if p.name in {"competitions", "datasets"}:
                continue

            if p.is_dir():
                roots.append(p)
    except Exception:
        pass

    datasets_root = base / "datasets"

    if datasets_root.exists():
        try:
            for owner in datasets_root.iterdir():
                if not owner.is_dir():
                    continue

                for dataset in owner.iterdir():
                    if dataset.is_dir():
                        roots.append(dataset)
        except Exception:
            pass

    return roots


def discover_w4_cache(project_root: Path) -> Path:
    candidates = [
        project_root
        / "output"
        / "results"
        / "rsna_w4_0_curia2"
        / "feature_cache"
        / "studies",
        project_root / "output" / "results" / "rsna_w4_0_curia2" / "feature_cache",
    ]

    found = _first_existing(candidates)

    if found:
        if (found / "studies").is_dir():
            found = found / "studies"

        return found.resolve()

    if is_kaggle():
        scored = []

        for root in kaggle_dataset_roots():
            options = [
                root / "feature_cache" / "studies",
                root / "rsna_w4_0_curia2" / "feature_cache" / "studies",
            ]

            for p in options:
                if p.is_dir():
                    score = 0
                    s = str(p).lower()

                    if "w4" in s:
                        score += 10

                    if "curia" in s:
                        score += 5

                    scored.append((score, p))

        if scored:
            scored.sort(key=lambda x: (-x[0], len(str(x[1]))))

            return scored[0][1].resolve()

    raise FileNotFoundError(
        "Could not locate canonical W4 Curia feature cache. "
        "Expected local path: "
        "output/results/rsna_w4_0_curia2/feature_cache/studies"
    )


def _teacher_has_required_files(root: Path) -> bool:
    names = [
        "16_final_hybrid_probabilities_wide.csv",
        "17_recommended_teacher_weights_wide.csv",
        "18_recommended_teacher_mask_wide.csv",
    ]

    return all((root / "results" / n).exists() or (root / n).exists() for n in names)


def discover_w26_root(project_root: Path) -> Path:
    local = project_root / "output" / "results" / "rsna_w2_6p_fast"

    if local.exists() and _teacher_has_required_files(local):
        return local.resolve()

    if is_kaggle():
        scored = []

        for root in kaggle_dataset_roots():
            options = [
                root,
                root / "rsna_w2_6p_fast",
            ]

            for p in options:
                if p.is_dir() and _teacher_has_required_files(p):
                    score = 0
                    s = str(p).lower()

                    if "w2" in s:
                        score += 10

                    if "fast" in s:
                        score += 5

                    scored.append((score, p))

        if scored:
            scored.sort(key=lambda x: (-x[0], len(str(x[1]))))

            return scored[0][1].resolve()

    raise FileNotFoundError("Could not locate W2.6-P production teacher outputs.")


@dataclass
class ProjectPaths:
    project_root: str
    train_csv: str
    w4_cache: str
    w26_root: str
    output_root: str

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = discover_project_root()

        if is_kaggle():
            train_csv = Path(
                "/kaggle/input/competitions/"
                "rsna-knee-abnormality-detection/"
                "train.csv"
            )
        else:
            train_csv = root / "input" / "train.csv"

        if not train_csv.exists():
            raise FileNotFoundError(f"train.csv not found: {train_csv}")

        w4_cache = discover_w4_cache(root)
        w26_root = discover_w26_root(root)

        if is_kaggle():
            output_root = Path("/kaggle/working") / "rsna_w7_label_aware_mil_v1"
        else:
            output_root = root / "output" / "results" / "rsna_w7_label_aware_mil_v1"

        output_root.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=str(root),
            train_csv=str(train_csv.resolve()),
            w4_cache=str(w4_cache.resolve()),
            w26_root=str(w26_root.resolve()),
            output_root=str(output_root.resolve()),
        )


# =============================================================================
# ACCELERATOR — NO HARD-CODED GPU / TPU COUNT
# =============================================================================


@dataclass
class Runtime:
    requested: str
    backend: str
    devices: List[str]
    precision: str


CUDA_ALIASES = {
    "cuda",
    "gpu",
    "localgpu",
    "local_gpu",
    "kaggle_t4",
    "t4",
}

TPU_ALIASES = {
    "tpu",
    "xla",
    "kaggle_tpu",
}

MPS_ALIASES = {
    "mps",
    "apple",
    "apple_mps",
}


def resolve_runtime(accelerator: str) -> Runtime:
    requested = accelerator or "auto"
    key = requested.strip().lower()

    # ------------------------------------------------------------------
    # AUTO
    # ------------------------------------------------------------------

    if key == "auto":
        if torch.cuda.is_available():
            key = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            key = "mps"
        else:
            key = "cpu"

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------

    if key in CUDA_ALIASES:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--accelerator {requested} requested CUDA, " "but CUDA is unavailable."
            )

        count = torch.cuda.device_count()

        if count < 1:
            raise RuntimeError("No visible CUDA devices.")

        devices = [f"cuda:{i}" for i in range(count)]

        if torch.cuda.is_bf16_supported():
            precision = "bfloat16"
        else:
            precision = "float16"

        return Runtime(
            requested=requested,
            backend="cuda",
            devices=devices,
            precision=precision,
        )

    # ------------------------------------------------------------------
    # APPLE MPS
    # ------------------------------------------------------------------

    if key in MPS_ALIASES:
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Apple MPS requested but unavailable.")

        return Runtime(
            requested=requested,
            backend="mps",
            devices=["mps"],
            precision="float32",
        )

    # ------------------------------------------------------------------
    # TPU / XLA
    # ------------------------------------------------------------------

    if key in TPU_ALIASES:
        try:
            import torch_xla.runtime as xr

            try:
                count = int(xr.global_runtime_device_count())
            except Exception:
                count = 0

        except Exception as exc:
            raise RuntimeError(
                "TPU/XLA requested but torch_xla is unavailable."
            ) from exc

        devices = [f"xla:{i}" for i in range(count)] if count > 0 else ["xla:auto"]

        return Runtime(
            requested=requested,
            backend="xla",
            devices=devices,
            precision="float32",
        )

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    if key == "cpu":
        return Runtime(
            requested=requested,
            backend="cpu",
            devices=["cpu"],
            precision="float32",
        )

    raise ValueError(f"Unknown accelerator: {accelerator}")


# =============================================================================
# FEATURE CACHE
# =============================================================================
def stable_uid_hash(uid: str) -> str:
    return hashlib.md5(str(uid).encode("utf-8")).hexdigest()


class FeatureStore:
    def __init__(
        self,
        cache_root: Path,
        required_uids: Sequence[str],
        lru_size: int = FEATURE_LRU_SIZE,
    ):
        self.cache_root = Path(cache_root)
        self.lru_size = int(lru_size)
        self._lru: OrderedDict[str, dict] = OrderedDict()

        if not self.cache_root.exists():
            raise FileNotFoundError(f"Feature cache missing: {self.cache_root}")

        self.index: Dict[str, Path] = {}

        missing = []

        for uid in required_uids:
            uid = str(uid)

            filename = f"{stable_uid_hash(uid)}.pt"

            # Normal canonical W4 location:
            # feature_cache/studies/<md5>.pt
            path = self.cache_root / filename

            if not path.exists():
                # Defensive support if caller passed feature_cache/
                # instead of feature_cache/studies/
                alternate = self.cache_root / "studies" / filename

                if alternate.exists():
                    path = alternate

            if path.exists():
                self.index[uid] = path
            else:
                missing.append(uid)

        if missing:
            raise RuntimeError(
                f"Missing feature caches: {len(missing)}. " f"Example: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.index)

    def load(self, uid: str) -> dict:
        uid = str(uid)

        if uid in self._lru:
            payload = self._lru.pop(uid)
            self._lru[uid] = payload

            return payload

        path = self.index[uid]

        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                path,
                map_location="cpu",
            )

        version = str(payload.get("cache_version", ""))

        if version != EXPECTED_W4_CACHE_VERSION:
            raise RuntimeError(
                f"{uid}: unexpected cache_version={version!r}. "
                f"Expected {EXPECTED_W4_CACHE_VERSION!r}"
            )

        features = payload.get("features")

        if not torch.is_tensor(features):
            raise RuntimeError(f"{uid}: features missing.")

        if features.ndim != 3 or int(features.shape[-1]) != CURIA_DIM:
            raise RuntimeError(
                f"{uid}: invalid features shape " f"{tuple(features.shape)}"
            )

        required = [
            "slice_mask",
            "slice_position",
            "series_meta",
            "series_cont",
        ]

        for key in required:
            if key not in payload:
                raise RuntimeError(f"{uid}: missing cache field {key}")

        self._lru[uid] = payload

        while len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)

        return payload


def collate_features(
    store: FeatureStore,
    uids: Sequence[str],
) -> dict:
    payloads = [store.load(uid) for uid in uids]

    b = len(payloads)

    max_s = max(int(x["features"].shape[0]) for x in payloads)

    max_k = max(int(x["features"].shape[1]) for x in payloads)

    features = torch.zeros(
        (b, max_s, max_k, CURIA_DIM),
        dtype=torch.float16,
    )

    slice_mask = torch.zeros(
        (b, max_s, max_k),
        dtype=torch.bool,
    )

    slice_position = torch.zeros(
        (b, max_s, max_k),
        dtype=torch.float32,
    )

    series_meta = torch.zeros(
        (b, max_s, 3),
        dtype=torch.float32,
    )

    series_cont = torch.zeros(
        (b, max_s, 2),
        dtype=torch.float32,
    )

    for i, p in enumerate(payloads):
        f = p["features"]

        s = int(f.shape[0])
        k = int(f.shape[1])

        features[i, :s, :k] = f.to(dtype=torch.float16)

        slice_mask[i, :s, :k] = p["slice_mask"].bool()

        slice_position[i, :s, :k] = p["slice_position"].float()

        series_meta[i, :s] = p["series_meta"].float()

        series_cont[i, :s] = p["series_cont"].float()

    return {
        "features": features,
        "slice_mask": slice_mask,
        "slice_position": slice_position,
        "series_meta": series_meta,
        "series_cont": series_cont,
    }


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device,
) -> dict:
    return {
        k: v.to(
            device,
            non_blocking=(isinstance(device, torch.device) and device.type == "cuda"),
        )
        for k, v in batch.items()
    }


# =============================================================================
# LOCKED GOLD FOLDS
# =============================================================================


def _find_column(
    columns: Sequence[str],
    choices: Sequence[str],
) -> Optional[str]:
    mapping = {str(c).lower(): c for c in columns}

    for x in choices:
        if x.lower() in mapping:
            return mapping[x.lower()]

    return None


def _normalize_folds(
    fold_series: pd.Series,
) -> pd.Series:
    vals = pd.to_numeric(
        fold_series,
        errors="raise",
    ).astype(int)

    unique = sorted(vals.dropna().unique().tolist())

    if unique == [0, 1, 2, 3, 4]:
        vals = vals + 1
        unique = [1, 2, 3, 4, 5]

    if unique != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Unexpected fold values: {unique}")

    return vals


def discover_fold_map(
    train_df: pd.DataFrame,
    paths: ProjectPaths,
) -> Tuple[Dict[str, int], str]:
    gold = train_df[train_df[LABEL_COLUMNS].notna().all(axis=1)]

    gold_uids = set(gold[UID_COLUMN].astype(str))

    # train.csv itself may already contain a fold field.
    fold_col = _find_column(
        train_df.columns,
        ["Fold", "fold"],
    )

    if fold_col:
        tmp = train_df[[UID_COLUMN, fold_col]].dropna()

        tmp[fold_col] = _normalize_folds(tmp[fold_col])

        mapping = dict(
            zip(
                tmp[UID_COLUMN].astype(str),
                tmp[fold_col].astype(int),
            )
        )

        if gold_uids.issubset(mapping):
            return (
                {u: mapping[u] for u in gold_uids},
                str(Path(paths.train_csv)),
            )

    roots = [
        Path(paths.project_root) / "output" / "results" / "rsna_w6_curia",
        Path(paths.project_root) / "output" / "results" / "rsna_w4_0_curia2",
        Path(paths.project_root) / "output" / "results",
    ]

    candidates: List[Path] = []

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            name = p.name.lower()

            if any(
                token in name
                for token in [
                    "fold",
                    "diagnostic",
                    "oof",
                    "prediction",
                ]
            ):
                candidates.append(p)

    # Prefer W6/W60 diagnostic artifacts, because these definitely use
    # the already locked project folds.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if "w60" in str(p).lower() else 1,
            0 if "diagnostic" in p.name.lower() else 1,
            len(str(p)),
        ),
    )

    for p in candidates:
        try:
            head = pd.read_csv(
                p,
                nrows=5,
            )
        except Exception:
            continue

        uid_col = _find_column(
            head.columns,
            [
                UID_COLUMN,
                "study_uid",
                "uid",
            ],
        )

        fold_col = _find_column(
            head.columns,
            ["Fold", "fold"],
        )

        if not uid_col or not fold_col:
            continue

        try:
            df = pd.read_csv(
                p,
                usecols=[uid_col, fold_col],
            ).dropna()

            df[fold_col] = _normalize_folds(df[fold_col])

        except Exception:
            continue

        mapping = dict(
            zip(
                df[uid_col].astype(str),
                df[fold_col].astype(int),
            )
        )

        if gold_uids.issubset(mapping):
            return (
                {u: mapping[u] for u in gold_uids},
                str(p.resolve()),
            )

    raise RuntimeError(
        "Could not find an existing locked-fold artifact. "
        "W7_v1 intentionally refuses to invent/re-split folds. "
        "Keep your W6/W4 CV output under output/results so the "
        "existing fold assignments can be consumed as a data artifact."
    )


# =============================================================================
# W2.6-P PRODUCTION TEACHER
# =============================================================================


def _teacher_file(
    root: Path,
    name: str,
) -> Path:
    options = [
        root / "results" / name,
        root / name,
    ]

    found = _first_existing(options)

    if found is None:
        raise FileNotFoundError(f"Teacher file missing: {name}")

    return found


def _parse_bool_column(
    s: pd.Series,
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)

    numeric = pd.to_numeric(
        s,
        errors="coerce",
    )

    if numeric.notna().all():
        return numeric.to_numpy(dtype=float) > 0.5

    normalized = s.astype(str).str.strip().str.lower()

    return normalized.isin(["true", "1", "yes", "y", "t"]).to_numpy(dtype=bool)


@dataclass
class TrainingData:
    gold_uids: List[str]
    gold_targets: Dict[str, np.ndarray]

    pseudo_uids: List[str]
    teacher_prob: Dict[str, np.ndarray]
    teacher_weight: Dict[str, np.ndarray]

    fold_map: Dict[str, int]
    fold_source: str


def load_training_data(
    paths: ProjectPaths,
) -> TrainingData:
    train = pd.read_csv(paths.train_csv)

    missing_cols = [c for c in [UID_COLUMN] + LABEL_COLUMNS if c not in train.columns]

    if missing_cols:
        raise RuntimeError(f"train.csv missing columns: {missing_cols}")

    train[UID_COLUMN] = train[UID_COLUMN].astype(str)

    gold_mask = train[LABEL_COLUMNS].notna().all(axis=1)

    gold = train.loc[
        gold_mask,
        [UID_COLUMN] + LABEL_COLUMNS,
    ].copy()

    if len(gold) != 58:
        raise RuntimeError(f"Expected 58 gold studies, found {len(gold)}")

    gold_targets = {
        str(row[UID_COLUMN]): row[LABEL_COLUMNS].to_numpy(dtype=np.float32)
        for _, row in gold.iterrows()
    }

    fold_map, fold_source = discover_fold_map(
        train,
        paths,
    )

    root = Path(paths.w26_root)

    prob_path = _teacher_file(
        root,
        "16_final_hybrid_probabilities_wide.csv",
    )

    weight_path = _teacher_file(
        root,
        "17_recommended_teacher_weights_wide.csv",
    )

    mask_path = _teacher_file(
        root,
        "18_recommended_teacher_mask_wide.csv",
    )

    prob_df = pd.read_csv(prob_path)
    weight_df = pd.read_csv(weight_path)
    mask_df = pd.read_csv(mask_path)

    for df, name in [
        (prob_df, "probabilities"),
        (weight_df, "weights"),
        (mask_df, "mask"),
    ]:
        missing = [c for c in [UID_COLUMN] + LABEL_COLUMNS if c not in df.columns]

        if missing:
            raise RuntimeError(f"Teacher {name} missing: {missing}")

        df[UID_COLUMN] = df[UID_COLUMN].astype(str)

    if len(prob_df) != 4349:
        raise RuntimeError(f"Expected 4349 teacher rows, " f"found {len(prob_df)}")

    prob_df = prob_df.set_index(UID_COLUMN)

    weight_df = weight_df.set_index(UID_COLUMN)

    mask_df = mask_df.set_index(UID_COLUMN)

    pseudo_uids = list(prob_df.index.astype(str))

    if set(pseudo_uids) != set(weight_df.index.astype(str)) or set(pseudo_uids) != set(
        mask_df.index.astype(str)
    ):
        raise RuntimeError("Teacher UID sets do not match.")

    teacher_prob = {}
    teacher_weight = {}

    for uid in pseudo_uids:
        p = prob_df.loc[
            uid,
            LABEL_COLUMNS,
        ].to_numpy(dtype=np.float32)

        w = weight_df.loc[
            uid,
            LABEL_COLUMNS,
        ].to_numpy(dtype=np.float32)

        mask_row = mask_df.loc[
            uid,
            LABEL_COLUMNS,
        ]

        m = _parse_bool_column(mask_row).astype(np.float32)

        if not np.isfinite(p).all():
            raise RuntimeError(f"{uid}: invalid teacher probabilities")

        if (p < 0).any() or (p > 1).any():
            raise RuntimeError(f"{uid}: teacher probability outside [0,1]")

        w = np.clip(
            w,
            0.0,
            None,
        )

        w = w * m

        teacher_prob[uid] = p
        teacher_weight[uid] = w.astype(np.float32)

    selected_cells = int(sum((w > 0).sum() for w in teacher_weight.values()))

    if selected_cells != 32027:
        raise RuntimeError(
            f"Expected 32027 selected teacher cells, " f"found {selected_cells}"
        )

    return TrainingData(
        gold_uids=list(gold[UID_COLUMN].astype(str)),
        gold_targets=gold_targets,
        pseudo_uids=pseudo_uids,
        teacher_prob=teacher_prob,
        teacher_weight=teacher_weight,
        fold_map=fold_map,
        fold_source=fold_source,
    )


# =============================================================================
# W7_v1 MODEL
# =============================================================================


def fourier_position(
    position: torch.Tensor,
    n_freq: int = 4,
) -> torch.Tensor:
    """
    position: [B,S,K], expected normalized roughly 0..1

    returns [B,S,K,2*n_freq]
    """

    freqs = 2.0 ** torch.arange(
        n_freq,
        device=position.device,
        dtype=torch.float32,
    )

    angle = position.float().unsqueeze(-1) * math.pi * freqs

    return torch.cat(
        [
            torch.sin(angle),
            torch.cos(angle),
        ],
        dim=-1,
    )


def masked_attention_pool(
    values: torch.Tensor,
    mask: torch.Tensor,
    temperature: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Attention where the evidence value itself drives soft selection.
    """

    mask = mask.expand_as(values)

    scaled = values / temperature.clamp_min(0.20)

    scaled = scaled.masked_fill(
        ~mask,
        -1e4,
    )

    weight = torch.softmax(
        scaled,
        dim=dim,
    )

    weight = weight * mask.to(weight.dtype)

    denom = weight.sum(
        dim=dim,
        keepdim=True,
    ).clamp_min(1e-6)

    weight = weight / denom

    return (weight * values).sum(dim=dim)


def masked_topk_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    dim: int,
) -> torch.Tensor:
    mask = mask.expand_as(values)

    kk = min(
        int(k),
        int(values.shape[dim]),
    )

    masked = values.masked_fill(
        ~mask,
        -1e4,
    )

    top = torch.topk(
        masked,
        k=kk,
        dim=dim,
    ).values

    counts = mask.sum(
        dim=dim,
    ).clamp(
        min=0,
        max=kk,
    )

    shape = [1] * values.ndim
    shape[dim] = kk

    rank = torch.arange(
        kk,
        device=values.device,
    ).view(shape)

    valid = rank < counts.unsqueeze(dim)

    numerator = (top * valid.to(top.dtype)).sum(dim=dim)

    denominator = valid.sum(dim=dim).clamp_min(1)

    result = numerator / denominator.to(numerator.dtype)

    return torch.where(
        counts > 0,
        result,
        torch.zeros_like(result),
    )


def masked_max(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    mask = mask.expand_as(values)

    out = (
        values.masked_fill(
            ~mask,
            -1e4,
        )
        .max(dim=dim)
        .values
    )

    valid = mask.any(dim=dim)

    return torch.where(
        valid,
        out,
        torch.zeros_like(out),
    )


class W7LabelAwareMIL(nn.Module):
    """
    Low-parameter weak-supervision MIL head.

    No spatial patch tokens.
    No slice Transformer.
    No series Transformer.

    It learns label-specific evidence directly from Curia slice CLS.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.slice_encoder = nn.Sequential(
            nn.LayerNorm(CURIA_DIM),
            nn.Linear(
                CURIA_DIM,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.position_encoder = nn.Sequential(
            nn.Linear(
                8,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        # plane/fluid/FS + two continuous fields
        self.series_encoder = nn.Sequential(
            nn.Linear(
                5,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.fused_norm = nn.LayerNorm(hidden_dim)

        # Direct label-specific slice evidence.
        self.slice_head = nn.Linear(
            hidden_dim,
            N_LABELS,
        )

        # Small label-specific metadata routing bias.
        self.metadata_label_bias = nn.Linear(
            5,
            N_LABELS,
        )

        # Learn how sparse/diffuse each label is.
        #
        # components:
        # 0 = attention
        # 1 = top-k mean
        # 2 = max
        self.slice_pool_mix = nn.Parameter(
            torch.tensor(
                [[1.0, 1.0, 0.0]] * N_LABELS,
                dtype=torch.float32,
            )
        )

        self.series_pool_mix = nn.Parameter(
            torch.tensor(
                [[1.0, 1.0, 0.0]] * N_LABELS,
                dtype=torch.float32,
            )
        )

        self.slice_temperature_raw = nn.Parameter(torch.zeros(N_LABELS))

        self.series_temperature_raw = nn.Parameter(torch.zeros(N_LABELS))

        # Diffuse/global context path.
        self.global_head = nn.Linear(
            hidden_dim,
            N_LABELS,
        )

        # Start with only ~25% global contribution.
        self.global_gate_logits = nn.Parameter(
            torch.full(
                (N_LABELS,),
                -1.10,
            )
        )

    def _pool(
        self,
        evidence: torch.Tensor,
        mask: torch.Tensor,
        *,
        dim: int,
        topk: int,
        mix_param: torch.Tensor,
        temperature_raw: torch.Tensor,
    ) -> torch.Tensor:
        temp = 0.50 + F.softplus(temperature_raw)

        # broadcast label temperature
        shape = [1] * evidence.ndim
        shape[-1] = N_LABELS

        temp = temp.view(shape)

        attention = masked_attention_pool(
            evidence,
            mask,
            temp,
            dim=dim,
        )

        topk_mean = masked_topk_mean(
            evidence,
            mask,
            k=topk,
            dim=dim,
        )

        maximum = masked_max(
            evidence,
            mask,
            dim=dim,
        )

        components = torch.stack(
            [
                attention,
                topk_mean,
                maximum,
            ],
            dim=-1,
        )

        mix = torch.softmax(
            mix_param,
            dim=-1,
        )

        return (components * mix).sum(dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        slice_position: torch.Tensor,
        series_meta: torch.Tensor,
        series_cont: torch.Tensor,
    ) -> torch.Tensor:
        # -------------------------------------------------------------
        # Slice representation
        # -------------------------------------------------------------

        x = self.slice_encoder(features.float())

        pos = fourier_position(
            slice_position,
            n_freq=4,
        )

        x = x + self.position_encoder(pos)

        series_raw = torch.cat(
            [
                series_meta.float(),
                series_cont.float(),
            ],
            dim=-1,
        )

        series_embedding = self.series_encoder(series_raw).unsqueeze(2)

        x = self.fused_norm(x + series_embedding)

        # -------------------------------------------------------------
        # Label-specific evidence per slice
        #
        # [B,S,K,L]
        # -------------------------------------------------------------

        slice_evidence = self.slice_head(x)

        slice_evidence = slice_evidence.masked_fill(
            ~slice_mask.unsqueeze(-1),
            -1e4,
        )

        # -------------------------------------------------------------
        # Slice -> series MIL
        #
        # [B,S,L]
        # -------------------------------------------------------------

        series_evidence = self._pool(
            slice_evidence,
            slice_mask.unsqueeze(-1),
            dim=2,
            topk=SLICE_TOPK,
            mix_param=self.slice_pool_mix,
            temperature_raw=self.slice_temperature_raw,
        )

        # Learned metadata routing — bounded so metadata cannot dominate.
        metadata_bias = 0.25 * torch.tanh(self.metadata_label_bias(series_raw))

        series_evidence = series_evidence + metadata_bias

        series_mask = slice_mask.any(dim=2)

        # -------------------------------------------------------------
        # Series -> study MIL
        #
        # [B,L]
        # -------------------------------------------------------------

        study_mil = self._pool(
            series_evidence,
            series_mask.unsqueeze(-1),
            dim=1,
            topk=SERIES_TOPK,
            mix_param=self.series_pool_mix,
            temperature_raw=self.series_temperature_raw,
        )

        # -------------------------------------------------------------
        # Small global-context residual
        # -------------------------------------------------------------

        valid = slice_mask.unsqueeze(-1).to(x.dtype)

        global_feature = (x * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(
            1.0
        )

        global_logits = self.global_head(global_feature)

        gate = torch.sigmoid(self.global_gate_logits).unsqueeze(0)

        logits = (1.0 - gate) * study_mil + gate * global_logits

        return logits


# =============================================================================
# LOSSES / METRICS
# =============================================================================


def positive_weights(
    targets: np.ndarray,
) -> torch.Tensor:
    pos = targets.sum(axis=0)
    neg = len(targets) - pos

    weight = neg / np.clip(pos, 1.0, None)

    weight = np.clip(
        weight,
        1.0,
        5.0,
    )

    return torch.tensor(
        weight,
        dtype=torch.float32,
    )


def gold_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
    )


def pseudo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    denom = weights.sum().clamp_min(1e-6)

    return (raw * weights).sum() / denom


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    per_label = []

    for i, label in enumerate(LABEL_COLUMNS):
        y = y_true[:, i]
        p = y_prob[:, i]

        auc = np.nan
        ap = np.nan

        if len(np.unique(y)) >= 2:
            auc = float(
                roc_auc_score(
                    y,
                    p,
                )
            )

        if y.sum() > 0:
            ap = float(
                average_precision_score(
                    y,
                    p,
                )
            )

        per_label.append(
            {
                "Label": label,
                "AUROC": auc,
                "AP": ap,
            }
        )

    aucs = [x["AUROC"] for x in per_label if np.isfinite(x["AUROC"])]

    aps = [x["AP"] for x in per_label if np.isfinite(x["AP"])]

    return {
        "macro_auroc": float(np.mean(aucs)),
        "macro_ap": float(np.mean(aps)),
        "per_label": per_label,
    }


# =============================================================================
# TRAINING HELPERS
# =============================================================================


def get_autocast(
    device,
    backend: str,
):
    if backend != "cuda":
        return nullcontext()

    if torch.cuda.is_bf16_supported():
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    )


def make_grad_scaler(
    backend: str,
):
    enabled = backend == "cuda" and not torch.cuda.is_bf16_supported()

    if not enabled:
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def sample_uids(
    rng: np.random.Generator,
    population: Sequence[str],
    batch_size: int,
) -> List[str]:
    replace = len(population) < batch_size

    selected = rng.choice(
        np.asarray(
            population,
            dtype=object,
        ),
        size=batch_size,
        replace=replace,
    )

    return [str(x) for x in selected.tolist()]


def batch_targets(
    uids: Sequence[str],
    mapping: Mapping[str, np.ndarray],
) -> torch.Tensor:
    return torch.tensor(
        np.stack([mapping[str(uid)] for uid in uids]),
        dtype=torch.float32,
    )


@torch.no_grad()
def predict_uids(
    model: nn.Module,
    store: FeatureStore,
    uids: Sequence[str],
    device,
    backend: str,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()

    outputs = []

    for start in range(
        0,
        len(uids),
        batch_size,
    ):
        batch_uids = list(uids[start : start + batch_size])

        batch = move_batch(
            collate_features(
                store,
                batch_uids,
            ),
            device,
        )

        with get_autocast(
            device,
            backend,
        ):
            logits = model(**batch)

        prob = torch.sigmoid(logits.float())

        outputs.append(prob.detach().cpu().numpy())

    return np.concatenate(
        outputs,
        axis=0,
    )


def _cpu_state_dict(
    model: nn.Module,
) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# =============================================================================
# ONE CV FOLD / ONE FULL SEED
# =============================================================================


def train_single_task(
    paths: ProjectPaths,
    *,
    mode: str,
    task: int,
    device,
    backend: str,
) -> dict:
    data = load_training_data(paths)

    all_uids = data.gold_uids + data.pseudo_uids

    store = FeatureStore(
        Path(paths.w4_cache),
        required_uids=all_uids,
    )

    if mode == "cv":
        fold = int(task)
        seed = CV_SEED_BASE + fold

        gold_train = [uid for uid in data.gold_uids if data.fold_map[uid] != fold]

        gold_val = [uid for uid in data.gold_uids if data.fold_map[uid] == fold]

        task_name = f"fold_{fold}"

    elif mode == "full":
        seed = int(task)

        gold_train = list(data.gold_uids)

        gold_val = []

        task_name = f"seed_{seed}"

    else:
        raise ValueError(mode)

    seed_everything(seed)

    rng = np.random.default_rng(seed)

    model = W7LabelAwareMIL().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        pct_start=0.20,
        anneal_strategy="cos",
    )

    gold_y_matrix = np.stack([data.gold_targets[uid] for uid in gold_train])

    pos_weight = positive_weights(gold_y_matrix).to(device)

    scaler = make_grad_scaler(backend)

    xm = None

    if backend == "xla":
        import torch_xla.core.xla_model as xm

    log("")
    log("=" * 96)
    log(f"W7_v1 | {mode} | {task_name} | " f"device={device}")
    log("=" * 96)
    log(f"Gold train studies : {len(gold_train)}")

    if gold_val:
        log(f"Gold diag studies  : {len(gold_val)}")

    log(f"Pseudo studies     : {len(data.pseudo_uids)}")
    log(f"Architecture       : label-aware evidence MIL")
    log(f"Gold:pseudo loss   : " f"{GOLD_AUTHORITY:g}:{PSEUDO_AUTHORITY:g}")

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        epoch_total = []
        epoch_gold = []
        epoch_pseudo = []

        for _ in range(STEPS_PER_EPOCH):
            gold_batch_uids = sample_uids(
                rng,
                gold_train,
                GOLD_BATCH_SIZE,
            )

            pseudo_batch_uids = sample_uids(
                rng,
                data.pseudo_uids,
                PSEUDO_BATCH_SIZE,
            )

            g_batch = move_batch(
                collate_features(
                    store,
                    gold_batch_uids,
                ),
                device,
            )

            p_batch = move_batch(
                collate_features(
                    store,
                    pseudo_batch_uids,
                ),
                device,
            )

            g_target = batch_targets(
                gold_batch_uids,
                data.gold_targets,
            ).to(device)

            p_target = batch_targets(
                pseudo_batch_uids,
                data.teacher_prob,
            ).to(device)

            p_weight = batch_targets(
                pseudo_batch_uids,
                data.teacher_weight,
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast(
                device,
                backend,
            ):
                g_logits = model(**g_batch)

                p_logits = model(**p_batch)

                g_loss = gold_loss(
                    g_logits,
                    g_target,
                    pos_weight,
                )

                p_loss = pseudo_loss(
                    p_logits,
                    p_target,
                    p_weight,
                )

                loss = (GOLD_AUTHORITY * g_loss + PSEUDO_AUTHORITY * p_loss) / (
                    GOLD_AUTHORITY + PSEUDO_AUTHORITY
                )

            if backend == "xla":
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                xm.optimizer_step(
                    optimizer,
                    barrier=False,
                )

                xm.mark_step()

            elif scaler is not None:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                scaler.step(optimizer)

                scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

            scheduler.step()

            epoch_total.append(float(loss.detach().float().cpu()))

            epoch_gold.append(float(g_loss.detach().float().cpu()))

            epoch_pseudo.append(float(p_loss.detach().float().cpu()))

        record = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_total)),
            "gold_loss": float(np.mean(epoch_gold)),
            "pseudo_loss": float(np.mean(epoch_pseudo)),
        }

        if gold_val:
            pred = predict_uids(
                model,
                store,
                gold_val,
                device,
                backend,
            )

            true = np.stack([data.gold_targets[u] for u in gold_val])

            metrics = compute_metrics(
                true,
                pred,
            )

            record["diagnostic_macro_auroc"] = metrics["macro_auroc"]

            log(
                f"epoch {epoch:02d}/{EPOCHS} "
                f"loss={record['loss']:.5f} "
                f"gold={record['gold_loss']:.5f} "
                f"pseudo={record['pseudo_loss']:.5f} "
                f"diag_auc="
                f"{metrics['macro_auroc']:.5f}"
            )

        else:
            log(
                f"epoch {epoch:02d}/{EPOCHS} "
                f"loss={record['loss']:.5f} "
                f"gold={record['gold_loss']:.5f} "
                f"pseudo={record['pseudo_loss']:.5f}"
            )

        history.append(record)

    # ------------------------------------------------------------------
    # Save fixed-final-epoch model.
    # No validation-based epoch cherry-picking.
    # ------------------------------------------------------------------

    output_root = Path(paths.output_root)

    checkpoint_dir = output_root / "checkpoints" / mode

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_dir / f"{task_name}_epoch_{EPOCHS}.pt"

    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "created_at": utc_timestamp(),
        "mode": mode,
        "task": task,
        "seed": seed,
        "epoch": EPOCHS,
        "model_state": _cpu_state_dict(model),
        "model_config": {
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "slice_topk": SLICE_TOPK,
            "series_topk": SERIES_TOPK,
        },
        "training_config": {
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "gold_batch_size": GOLD_BATCH_SIZE,
            "pseudo_batch_size": PSEUDO_BATCH_SIZE,
            "max_lr": MAX_LR,
            "weight_decay": WEIGHT_DECAY,
            "gold_authority": GOLD_AUTHORITY,
            "pseudo_authority": PSEUDO_AUTHORITY,
        },
        "expected_w4_cache_version": EXPECTED_W4_CACHE_VERSION,
        "locked_fold_sha256": LOCKED_FOLD_SHA256,
        "fold_source": data.fold_source,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    result = {
        "mode": mode,
        "task": task,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }

    # ------------------------------------------------------------------
    # Fold diagnostic predictions
    # ------------------------------------------------------------------

    if gold_val:
        pred = predict_uids(
            model,
            store,
            gold_val,
            device,
            backend,
        )

        true = np.stack([data.gold_targets[u] for u in gold_val])

        rows = []

        for row_idx, uid in enumerate(gold_val):
            row = {
                UID_COLUMN: uid,
                "Fold": int(task),
            }

            for j, label in enumerate(LABEL_COLUMNS):
                row[f"true__{label}"] = float(true[row_idx, j])

                row[f"pred__{label}"] = float(pred[row_idx, j])

            rows.append(row)

        pred_dir = output_root / "results" / "cv"

        pred_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pred_path = pred_dir / f"fold_{task}_predictions.csv"

        pd.DataFrame(rows).to_csv(
            pred_path,
            index=False,
        )

        result["predictions"] = str(pred_path)

    result_path = output_root / "results" / mode / f"{task_name}_result.json"

    safe_json_dump(
        result,
        result_path,
    )

    del model
    del store

    gc.collect()

    if backend == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# GENERIC MULTI-DEVICE TASK SCHEDULER
# =============================================================================


def _cuda_worker(
    device_index: int,
    mode: str,
    tasks: Sequence[int],
    paths_dict: dict,
) -> None:
    torch.cuda.set_device(device_index)

    device = torch.device(f"cuda:{device_index}")

    paths = ProjectPaths(**paths_dict)

    for task in tasks:
        train_single_task(
            paths,
            mode=mode,
            task=int(task),
            device=device,
            backend="cuda",
        )


def _xla_worker(
    process_index: int,
    mode: str,
    tasks: Sequence[int],
    paths_dict: dict,
) -> None:
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr

        ordinal = int(xr.global_ordinal())

        world = int(xr.world_size())

    except Exception:
        ordinal = int(process_index)

        world = max(
            1,
            len(tasks),
        )

    assigned = list(tasks[ordinal::world])

    if not assigned:
        return

    device = xm.xla_device()

    paths = ProjectPaths(**paths_dict)

    for task in assigned:
        train_single_task(
            paths,
            mode=mode,
            task=int(task),
            device=device,
            backend="xla",
        )


def run_xla_tasks(
    mode: str,
    tasks: Sequence[int],
    paths: ProjectPaths,
) -> None:
    args = (
        mode,
        list(tasks),
        asdict(paths),
    )

    try:
        import torch_xla

        launch = getattr(
            torch_xla,
            "launch",
            None,
        )

        if callable(launch):
            launch(
                _xla_worker,
                args=args,
                nprocs=None,
            )

            return

    except Exception:
        pass

    try:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(
            _xla_worker,
            args=args,
            nprocs=None,
            start_method="spawn",
        )

    except Exception as exc:
        raise RuntimeError("Unable to launch TPU/XLA workers.") from exc


def run_tasks(
    mode: str,
    tasks: Sequence[int],
    paths: ProjectPaths,
    runtime: Runtime,
) -> None:
    tasks = [int(x) for x in tasks]

    if runtime.backend == "xla":
        log("[W7] TPU/XLA: using available " "XLA worker world automatically.")

        run_xla_tasks(
            mode,
            tasks,
            paths,
        )

        return

    # ------------------------------------------------------------------
    # CUDA: use EVERY visible CUDA device automatically.
    # ------------------------------------------------------------------

    if runtime.backend == "cuda":
        device_count = len(runtime.devices)

        if device_count == 1:
            for task in tasks:
                train_single_task(
                    paths,
                    mode=mode,
                    task=task,
                    device=torch.device(runtime.devices[0]),
                    backend="cuda",
                )

            return

        assignments = [tasks[i::device_count] for i in range(device_count)]

        assignments = [x for x in assignments if x]

        log(f"[W7] automatic CUDA scheduling " f"across {device_count} visible GPU(s):")

        for i, assigned in enumerate(assignments):
            log(f"  cuda:{i} -> {assigned}")

        ctx = mp.get_context("spawn")

        processes = []

        for device_index, assigned in enumerate(assignments):
            p = ctx.Process(
                target=_cuda_worker,
                args=(
                    device_index,
                    mode,
                    assigned,
                    asdict(paths),
                ),
            )

            p.start()
            processes.append(p)

        failed = []

        for p in processes:
            p.join()

            if p.exitcode != 0:
                failed.append(p.exitcode)

        if failed:
            raise RuntimeError(
                "One or more CUDA training workers " f"failed: exit codes={failed}"
            )

        return

    # ------------------------------------------------------------------
    # CPU / MPS
    # ------------------------------------------------------------------

    device = torch.device(runtime.devices[0])

    for task in tasks:
        train_single_task(
            paths,
            mode=mode,
            task=task,
            device=device,
            backend=runtime.backend,
        )


# =============================================================================
# CV AGGREGATION / GO-NO-GO
# =============================================================================


def find_w60_reference(
    paths: ProjectPaths,
) -> Optional[dict]:
    base = Path(paths.project_root) / "output" / "results" / "rsna_w6_curia"

    if not base.exists():
        return None

    candidates = [
        p
        for p in base.rglob("*.json")
        if "w60" in str(p).lower()
        and ("diagnostic" in p.name.lower() or "summary" in p.name.lower())
    ]

    def find_auc(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                k = str(key).lower()

                if (
                    isinstance(
                        value,
                        (int, float),
                    )
                    and "macro" in k
                    and ("auc" in k or "auroc" in k)
                ):
                    return float(value)

            for value in obj.values():
                found = find_auc(value)

                if found is not None:
                    return found

        elif isinstance(obj, list):
            for value in obj:
                found = find_auc(value)

                if found is not None:
                    return found

        return None

    for p in sorted(
        candidates,
        key=lambda x: len(str(x)),
    ):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))

            auc = find_auc(obj)

            if auc is not None:
                return {
                    "macro_auroc": auc,
                    "source": str(p.resolve()),
                }

        except Exception:
            continue

    return None


def aggregate_cv(
    paths: ProjectPaths,
) -> dict:
    result_root = Path(paths.output_root) / "results" / "cv"

    files = [result_root / f"fold_{fold}_predictions.csv" for fold in CV_FOLDS]

    missing = [str(p) for p in files if not p.exists()]

    if missing:
        raise RuntimeError("Missing CV prediction files: " + ", ".join(missing))

    oof = pd.concat(
        [pd.read_csv(p) for p in files],
        ignore_index=True,
    )

    if len(oof) != 58:
        raise RuntimeError(f"Expected 58 OOF rows, got {len(oof)}")

    y_true = np.stack(
        [oof[f"true__{label}"].to_numpy(dtype=np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    y_prob = np.stack(
        [oof[f"pred__{label}"].to_numpy(dtype=np.float32) for label in LABEL_COLUMNS],
        axis=1,
    )

    metrics = compute_metrics(
        y_true,
        y_prob,
    )

    per_label_df = pd.DataFrame(metrics["per_label"])

    weak_labels = per_label_df[per_label_df["AUROC"] < WEAK_LABEL_AUC]["Label"].tolist()

    reference = find_w60_reference(paths)

    macro_auc = metrics["macro_auroc"]

    delta = None

    if reference is not None:
        delta = macro_auc - reference["macro_auroc"]

        if delta >= GO_MIN_DELTA_VS_W60 and len(weak_labels) <= MAX_WEAK_LABELS_FOR_GO:
            verdict = "GO_TO_FULL"

        elif delta >= REVIEW_MIN_DELTA_VS_W60:
            verdict = "REVIEW"

        else:
            verdict = "STOP"

    else:
        if (
            macro_auc >= GO_MIN_MACRO_AUC_WITHOUT_REFERENCE
            and len(weak_labels) <= MAX_WEAK_LABELS_FOR_GO
        ):
            verdict = "GO_TO_FULL"

        elif macro_auc >= 0.710:
            verdict = "REVIEW"

        else:
            verdict = "STOP"

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": utc_timestamp(),
        "diagnostic_is_pristine_oof": False,
        "warning": (
            "W2.6-P production teacher uses all 58 gold reports "
            "as exemplars. This diagnostic is not pristine OOF."
        ),
        "macro_auroc": metrics["macro_auroc"],
        "macro_ap": metrics["macro_ap"],
        "weak_label_threshold": WEAK_LABEL_AUC,
        "weak_labels": weak_labels,
        "w60_reference": reference,
        "delta_vs_w60": delta,
        "gate": {
            "verdict": verdict,
            "go_min_delta_vs_w60": GO_MIN_DELTA_VS_W60,
            "review_min_delta_vs_w60": REVIEW_MIN_DELTA_VS_W60,
            "go_min_macro_without_reference": GO_MIN_MACRO_AUC_WITHOUT_REFERENCE,
            "max_weak_labels_for_go": MAX_WEAK_LABELS_FOR_GO,
        },
        "locked_fold_sha256": LOCKED_FOLD_SHA256,
    }

    oof_path = result_root / "oof_predictions.csv"

    per_label_path = result_root / "per_label_metrics.csv"

    summary_path = result_root / "diagnostic_summary.json"

    oof.to_csv(
        oof_path,
        index=False,
    )

    per_label_df.to_csv(
        per_label_path,
        index=False,
    )

    safe_json_dump(
        summary,
        summary_path,
    )

    log("")
    log("=" * 96)
    log("W7_v1 DIAGNOSTIC")
    log("=" * 96)

    log(f"Macro AUROC : " f"{metrics['macro_auroc']:.6f}")

    log(f"Macro AP    : " f"{metrics['macro_ap']:.6f}")

    if reference is not None:
        log(f"W6.0 ref    : " f"{reference['macro_auroc']:.6f}")

        log(f"Delta       : " f"{delta:+.6f}")

    else:
        log("W6.0 reference artifact not auto-discovered.")

    log(f"Weak labels : " f"{len(weak_labels)} -> {weak_labels}")

    log("")
    log(f"VERDICT     : {verdict}")

    if verdict == "STOP":
        log("STOP: do not run full training and " "do not spend a Kaggle submission.")

    elif verdict == "REVIEW":
        log("REVIEW: inspect per-label results before " "running full training.")

    else:
        log("GO: W7_v1 has cleared the conservative " "local experiment gate.")

    return summary


# =============================================================================
# STATUS / VALIDATION
# =============================================================================


def status(
    accelerator: str,
) -> dict:
    paths = ProjectPaths.discover()
    runtime = resolve_runtime(accelerator)

    train = pd.read_csv(paths.train_csv)

    gold_rows = int(train[LABEL_COLUMNS].notna().all(axis=1).sum())

    result = {
        "script_version": SCRIPT_VERSION,
        "created_at": utc_timestamp(),
        "is_kaggle": is_kaggle(),
        "paths": asdict(paths),
        "accelerator": {
            "requested": runtime.requested,
            "backend": runtime.backend,
            "devices": runtime.devices,
            "device_count": len(runtime.devices),
            "precision": runtime.precision,
            "device_count_is_automatic": True,
            "accelerator_env_variables_required": False,
        },
        "train": {
            "rows": len(train),
            "gold_rows": gold_rows,
        },
        "locked_fold_sha256": LOCKED_FOLD_SHA256,
        "w7_scope": {
            "uses_w4_cls_cache": True,
            "uses_w26_teacher": True,
            "reads_train_series_dicom": False,
            "runs_curia": False,
            "uses_w61_patch_cache": False,
            "has_submission_mode": False,
        },
    }

    log(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


def validate_inputs(
    accelerator: str,
) -> dict:
    paths = ProjectPaths.discover()
    runtime = resolve_runtime(accelerator)

    data = load_training_data(paths)

    all_uids = data.gold_uids + data.pseudo_uids

    store = FeatureStore(
        Path(paths.w4_cache),
        required_uids=all_uids,
    )

    # Validate a sample plus all file existence through index construction.
    sample_uids = data.gold_uids[:5] + data.pseudo_uids[:5]

    shapes = {}

    for uid in sample_uids:
        p = store.load(uid)

        shapes[uid] = {
            "features": list(p["features"].shape),
            "slice_mask": list(p["slice_mask"].shape),
        }

    fold_counts = {
        str(fold): int(sum(data.fold_map[u] == fold for u in data.gold_uids))
        for fold in CV_FOLDS
    }

    result = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "runtime": asdict(runtime),
        "w4_cache": paths.w4_cache,
        "cache_files_indexed": len(store),
        "gold_studies": len(data.gold_uids),
        "pseudo_studies": len(data.pseudo_uids),
        "teacher_selected_cells": int(
            sum((x > 0).sum() for x in data.teacher_weight.values())
        ),
        "fold_source": data.fold_source,
        "locked_fold_sha256": LOCKED_FOLD_SHA256,
        "fold_counts": fold_counts,
        "sample_cache_shapes": shapes,
    }

    safe_json_dump(
        result,
        Path(paths.output_root) / "validation.json",
    )

    log(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


# =============================================================================
# TOP-LEVEL TRAIN MODES
# =============================================================================


def train_cv(
    accelerator: str,
) -> dict:
    paths = ProjectPaths.discover()
    runtime = resolve_runtime(accelerator)

    validate_inputs(accelerator)

    run_tasks(
        "cv",
        CV_FOLDS,
        paths,
        runtime,
    )

    return aggregate_cv(paths)


def train_full(
    accelerator: str,
    force: bool = False,
) -> dict:
    paths = ProjectPaths.discover()
    runtime = resolve_runtime(accelerator)

    decision_path = (
        Path(paths.output_root) / "results" / "cv" / "diagnostic_summary.json"
    )

    if not force:
        if not decision_path.exists():
            raise RuntimeError(
                "W7_v1 refuses full training before CV. " "Run train_cv first."
            )

        decision = json.loads(decision_path.read_text(encoding="utf-8"))

        verdict = decision.get(
            "gate",
            {},
        ).get("verdict")

        if verdict != "GO_TO_FULL":
            raise RuntimeError(
                f"W7_v1 CV verdict is {verdict!r}, not GO_TO_FULL. "
                "Full training was blocked to avoid unnecessary "
                "experimentation. Use --force only after manual review."
            )

    validate_inputs(accelerator)

    run_tasks(
        "full",
        FULL_SEEDS,
        paths,
        runtime,
    )

    checkpoints = sorted(
        (Path(paths.output_root) / "checkpoints" / "full").glob("seed_*_epoch_*.pt")
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "checkpoints": [str(p) for p in checkpoints],
    }

    safe_json_dump(
        result,
        Path(paths.output_root) / "results" / "full_summary.json",
    )

    log(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


# =============================================================================
# API / CLI
# =============================================================================


def run_w7(
    mode: str,
    accelerator: str = "auto",
    force: bool = False,
):
    mode = mode.strip().lower()

    if mode == "status":
        return status(accelerator)

    if mode == "validate":
        return validate_inputs(accelerator)

    if mode == "train_cv":
        return train_cv(accelerator)

    if mode == "train_full":
        return train_full(
            accelerator,
            force=force,
        )

    raise ValueError(f"Unknown mode: {mode}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=("RSNA W7_v1 Label-Aware Evidence-Routed MIL")
    )

    parser.add_argument(
        "mode",
        choices=[
            "status",
            "validate",
            "train_cv",
            "train_full",
        ],
    )

    parser.add_argument(
        "--accelerator",
        default="auto",
        help=("auto | localGPU | cuda | kaggle_t4 | " "apple_mps | cpu | kaggle_tpu"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=("Allow train_full despite CV gate. " "Normally leave this off."),
    )

    args = parser.parse_args()

    result = run_w7(
        args.mode,
        accelerator=args.accelerator,
        force=args.force,
    )

    if isinstance(
        result,
        (str, Path),
    ):
        print(result)


if __name__ == "__main__":
    _cli()
    # run_w7("validate", "localGPU")
