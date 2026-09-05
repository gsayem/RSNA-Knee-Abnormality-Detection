#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection
W41 — Exact W6.0 Curia-CLS Head + W40 Teacher v1
=================================================

Goal
----
Run ONE controlled image experiment:

    W6.0 anchor:
        canonical W4 Curia per-slice CLS cache
        exact W4/W6.0 hierarchical diagnosis head
        W2.6-P FAST production teacher

    W41:
        IDENTICAL image representation, head, folds, optimizer, schedule,
        gold:pseudo authority, masks, and weights
        +
        W40 teacher probabilities

W40 changes ONLY:
    Contusion probability = 0.50 * W2.6-P base + 0.50 * validated W39 FAST
    Effusion probability  = 0.50 * W2.6-P base + 0.50 * validated W39 FAST

All 10 other probabilities, all weights, and all masks are preserved.

Scientific validity
-------------------
W40 production teacher uses all 58 gold reports as production exemplars.
Therefore the 58-study fold-held-out diagnostic produced here is NON-PRISTINE.
It is only a regression / direction check. Kaggle remains the real model-selection
signal.

Standalone boundary
-------------------
This file:
    - DOES NOT import/call/execute another project .py file.
    - DOES NOT read DICOM.
    - DOES NOT load Curia.
    - consumes the already-built canonical W4 CLS cache only.
    - has no hidden-test submission mode.

A separate submission-capable script should be built only after W41 full-fit
models are accepted.

Modes
-----
    python w41_rsna_w60_curia_w40_teacher_v1.py status --accelerator localGPU
    python w41_rsna_w60_curia_w40_teacher_v1.py validate_teacher --accelerator localGPU
    python w41_rsna_w60_curia_w40_teacher_v1.py train_cv --accelerator localGPU
    python w41_rsna_w60_curia_w40_teacher_v1.py train_full --accelerator localGPU
    python w41_rsna_w60_curia_w40_teacher_v1.py train_all --accelerator localGPU
    python w41_rsna_w60_curia_w40_teacher_v1.py validate --accelerator localGPU

train_all runs CV first and automatically runs the three full-fit seeds only if
the diagnostic gate is GO_TO_FULL. No extra experiment is inserted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# =============================================================================
# 1. LOCKED EXPERIMENT
# =============================================================================

SCRIPT_VERSION = "w41_w60_curia_w40_teacher_v1"
DISPLAY_NAME = "W41 | Exact W6.0 Curia CLS + W40 teacher"

UID = "StudyInstanceUID"

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

FS2_LABELS = ["Contusion", "Effusion"]

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_TEACHER_CELLS = EXPECTED_UNLABELED * NUM_LABELS
EXPECTED_SELECTED_CELLS = 32027

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

EXPECTED_W4_CACHE_VERSION = (
    "w4_0_curia2_cls_allseries_24slice_canonical_orientation_" "slow_processor_v1"
)
EXPECTED_CURIA_HIDDEN = 768

EXPECTED_W40_SCRIPT_VERSION = "w40_fs2_production_teacher_v1"
EXPECTED_W39_GATE = "PASS_FS2_GOLD_GATE"
EXPECTED_W39_DELTA = 0.0504559894720078

# Successful public-submission run diagnostic. NON-PRISTINE reference only.
W60_DIAGNOSTIC_REFERENCE = 0.702997

# Conservative regression guard. Because both W6.0 and W41 diagnostics are
# non-pristine and n=58, do not demand a noisy +delta. W39 already validated the
# teacher change directly. We only stop if W41 materially breaks the image head.
GO_MACRO_AUC_MIN = 0.695
REVIEW_MACRO_AUC_MIN = 0.685
MAX_WEAK_LABELS = 3
WEAK_LABEL_AUC = 0.55

# Exact W6.0/W4 head hyperparameters.
HEAD_HIDDEN_DIM = 384
HEAD_NUM_HEADS = 8
HEAD_TRANSFORMER_LAYERS = 2
HEAD_DROPOUT = 0.15
SLICE_DROPOUT = 0.05
SERIES_DROPOUT = 0.05

HEAD_EPOCHS = 24
STEPS_PER_EPOCH = 32
GOLD_BATCH_SIZE = 16
PSEUDO_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 8
HEAD_MAX_LR = 0.001
HEAD_WEIGHT_DECAY = 0.001
GRAD_CLIP_NORM = 5.0
GOLD_AUTHORITY = 8.0
PSEUDO_AUTHORITY = 1.0

NUM_FOLDS = 5
RANDOM_SEED = 42
CV_SEED_BASE = 60000
FULLFIT_SEEDS = [6001, 6002, 6003]

FEATURE_LRU_SIZE = 96


# =============================================================================
# 2. PATHS
# =============================================================================


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def discover_project_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    here = script_dir()
    for candidate in [here, *here.parents]:
        if (candidate / "input" / "train.csv").exists():
            return candidate.resolve()

    candidate = (here / ".." / "..").resolve()
    if (candidate / "input" / "train.csv").exists():
        return candidate

    raise FileNotFoundError(
        "Could not discover project root containing input/train.csv. "
        "Use --project-root."
    )


@dataclass(frozen=True)
class Paths:
    project_root: Path
    train_csv: Path
    w4_cache_root: Path
    w40_root: Path
    result_root: Path
    checkpoint_root: Path

    @classmethod
    def discover(cls, args: argparse.Namespace) -> "Paths":
        project_root = discover_project_root(args.project_root)

        train_csv = (
            Path(args.train_csv).expanduser().resolve()
            if args.train_csv
            else project_root / "input" / "train.csv"
        )

        w4_cache_root = (
            Path(args.w4_cache).expanduser().resolve()
            if args.w4_cache
            else project_root
            / "output"
            / "results"
            / "rsna_w4_0_curia2"
            / "feature_cache"
            / "studies"
        )

        # Accept feature_cache/ as well as feature_cache/studies/.
        if (w4_cache_root / "studies").is_dir():
            w4_cache_root = w4_cache_root / "studies"

        w40_root = (
            Path(args.w40_root).expanduser().resolve()
            if args.w40_root
            else project_root
            / "output"
            / "results"
            / "rsna_w40_fs2_production_teacher_v1"
        )

        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else project_root
            / "output"
            / "results"
            / "rsna_w41_w60_curia_w40_teacher_v1"
        )

        return cls(
            project_root=project_root,
            train_csv=train_csv,
            w4_cache_root=w4_cache_root,
            w40_root=w40_root,
            result_root=output_root / "results",
            checkpoint_root=output_root / "checkpoints",
        )


def ensure_dirs(paths: Paths) -> None:
    paths.result_root.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_root.mkdir(parents=True, exist_ok=True)


def w40_result_root(paths: Paths) -> Path:
    candidate = paths.w40_root / "results"
    return candidate if candidate.is_dir() else paths.w40_root


def w40_prob_path(paths: Paths) -> Path:
    return w40_result_root(paths) / "16_final_probabilities_wide.csv"


def w40_weight_path(paths: Paths) -> Path:
    return w40_result_root(paths) / "17_teacher_weights_wide.csv"


def w40_mask_path(paths: Paths) -> Path:
    return w40_result_root(paths) / "18_teacher_mask_wide.csv"


def w40_summary_path(paths: Paths) -> Path:
    return w40_result_root(paths) / "19_production_summary.json"


def w40_validation_path(paths: Paths) -> Path:
    return w40_result_root(paths) / "20_validation_summary.json"


# =============================================================================
# 3. UTILITIES
# =============================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


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
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric > 0.5

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
    y_true: np.ndarray,
    y_prob: np.ndarray,
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
        "macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    ordered = assignments.sort_values(UID).reset_index(drop=True)
    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(
            ordered[UID].astype(str),
            ordered["OuterFold"].astype(int),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def greedy_multilabel_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Exact canonical W4/W6 fold implementation."""
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


# =============================================================================
# 4. ACCELERATOR
# =============================================================================


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    accelerator: str
    amp_dtype: Optional[torch.dtype]
    requested: str
    device_index: Optional[int]


def normalize_accelerator(value: Optional[str]) -> str:
    x = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "localgpu": "cuda",
        "local_gpu": "cuda",
        "gpu": "cuda",
        "kaggle_t4": "cuda",
        "t4": "cuda",
        "apple_mps": "mps",
        "apple": "mps",
        "kaggle_tpu": "tpu",
        "v5e": "tpu",
    }
    return aliases.get(x, x)


def resolve_runtime(
    accelerator: Optional[str],
    precision: str,
    device_index: Optional[int] = None,
) -> Runtime:
    requested_original = str(accelerator or "auto")
    requested = normalize_accelerator(accelerator)

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

    if requested == "tpu":
        raise RuntimeError(
            "kaggle_tpu is intentionally not implemented for W41. "
            "This cache-head experiment uses CUDA/MPS/CPU."
        )

    precision = str(precision or "auto").strip().lower()

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")

        count = torch.cuda.device_count()
        index = int(device_index or 0)
        if not 0 <= index < count:
            raise RuntimeError(
                f"CUDA device index {index} is invalid; visible devices={count}."
            )

        device = torch.device(f"cuda:{index}")

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
            raise ValueError(f"Unsupported precision={precision!r}")

        return Runtime(
            device=device,
            accelerator="cuda",
            amp_dtype=amp_dtype,
            requested=requested_original,
            device_index=index,
        )

    if requested == "mps":
        if (
            getattr(torch.backends, "mps", None) is None
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but unavailable.")
        return Runtime(
            device=torch.device("mps"),
            accelerator="mps",
            amp_dtype=None,
            requested=requested_original,
            device_index=None,
        )

    if requested == "cpu":
        return Runtime(
            device=torch.device("cpu"),
            accelerator="cpu",
            amp_dtype=None,
            requested=requested_original,
            device_index=None,
        )

    raise ValueError(
        f"Unsupported accelerator {accelerator!r}; use "
        "auto/localGPU/kaggle_t4/apple_mps/cpu/kaggle_tpu."
    )


@contextlib.contextmanager
def autocast_context(runtime: Runtime):
    if runtime.accelerator == "cuda" and runtime.amp_dtype is not None:
        with torch.autocast(
            device_type="cuda",
            dtype=runtime.amp_dtype,
        ):
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


def accelerator_device_count(accelerator: Optional[str]) -> int:
    resolved = normalize_accelerator(accelerator)
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved == "cuda":
        return max(1, torch.cuda.device_count())
    return 1


# =============================================================================
# 5. TRAIN TABLE + LOCKED FOLDS
# =============================================================================


def load_train_state(
    paths: Paths,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not paths.train_csv.exists():
        raise FileNotFoundError(paths.train_csv)

    train_df = pd.read_csv(paths.train_csv)
    if UID not in train_df.columns:
        raise RuntimeError("train.csv missing StudyInstanceUID")

    train_df[UID] = train_df[UID].astype(str)

    missing = [label for label in LABELS if label not in train_df.columns]
    if missing:
        raise RuntimeError(f"train.csv missing labels: {missing}")

    if len(train_df) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} train rows; found {len(train_df)}"
        )

    gold_mask = train_df[LABELS].notna().all(axis=1)
    unlabeled_mask = train_df[LABELS].isna().all(axis=1)

    if (~(gold_mask | unlabeled_mask)).any():
        raise RuntimeError("train.csv contains partially labeled rows.")

    gold_df = train_df[gold_mask].copy().sort_values(UID).reset_index(drop=True)

    if len(gold_df) != EXPECTED_GOLD:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD} gold studies; found {len(gold_df)}"
        )

    if int(unlabeled_mask.sum()) != EXPECTED_UNLABELED:
        raise RuntimeError(
            f"Expected {EXPECTED_UNLABELED} unlabeled studies; "
            f"found {int(unlabeled_mask.sum())}"
        )

    gold_y = gold_df[LABELS].to_numpy(dtype=np.int64)
    fold_zero = greedy_multilabel_folds(
        gold_y,
        NUM_FOLDS,
        RANDOM_SEED,
    )

    fold_df = gold_df[[UID]].copy()
    fold_df["OuterFold"] = fold_zero + 1
    digest = fold_assignment_sha256(fold_df)

    if digest != EXPECTED_FOLD_SHA256:
        raise RuntimeError(
            "Locked fold checksum mismatch.\n"
            f"Found   : {digest}\n"
            f"Expected: {EXPECTED_FOLD_SHA256}"
        )

    ensure_dirs(paths)
    fold_path = paths.result_root / "00_outer_fold_assignments.csv"
    temporary = fold_path.with_suffix(fold_path.suffix + f".tmp-{os.getpid()}")
    fold_df.to_csv(
        temporary,
        index=False,
    )
    os.replace(temporary, fold_path)

    return train_df, gold_df, fold_zero


# =============================================================================
# 6. W40 TEACHER
# =============================================================================


@dataclass
class TeacherData:
    uids: List[str]
    probability: np.ndarray
    weight: np.ndarray
    mask: np.ndarray
    root: Path
    summary: Dict[str, Any]


def validate_w40_provenance(paths: Paths) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary_path = w40_summary_path(paths)
    validation_path = w40_validation_path(paths)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    if summary.get("script_version") != EXPECTED_W40_SCRIPT_VERSION:
        raise RuntimeError(
            "Unexpected W40 script version: " f"{summary.get('script_version')!r}"
        )

    if summary.get("status") != "PRODUCTION_COMPLETE":
        raise RuntimeError("W40 production summary is not PRODUCTION_COMPLETE.")

    if validation.get("overall_pass") is not True:
        raise RuntimeError("W40 validation overall_pass is not true.")

    if summary.get("w39_gate", {}).get("verdict") != EXPECTED_W39_GATE:
        raise RuntimeError("W40 does not record PASS_FS2_GOLD_GATE.")

    delta = float(
        summary.get("w39_gate", {}).get(
            "target_mean_AUROC_delta",
            -999.0,
        )
    )
    if abs(delta - EXPECTED_W39_DELTA) > 1e-9:
        raise RuntimeError(
            f"W39 delta changed: {delta:.12f} != {EXPECTED_W39_DELTA:.12f}"
        )

    changes = summary.get("controlled_changes", {})
    if set(changes.get("probabilities_changed", [])) != set(FS2_LABELS):
        raise RuntimeError(
            "W40 controlled change set is not exactly Contusion + Effusion."
        )

    required_true = [
        "other_10_probabilities_preserved",
        "all_teacher_weights_preserved",
        "all_teacher_masks_preserved",
        "synovitis_probability_preserved",
        "synovitis_policy_preserved",
    ]
    bad = [name for name in required_true if changes.get(name) is not True]
    if bad:
        raise RuntimeError(f"W40 provenance guards failed: {bad}")

    return summary, validation


def load_teacher(
    paths: Paths,
    train_df: Optional[pd.DataFrame] = None,
    strict: bool = True,
) -> TeacherData:
    summary, validation = validate_w40_provenance(paths)

    prob_path = w40_prob_path(paths)
    weight_path = w40_weight_path(paths)
    mask_path = w40_mask_path(paths)

    for path in [prob_path, weight_path, mask_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    prob_df = pd.read_csv(prob_path)
    weight_df = pd.read_csv(weight_path)
    mask_df = pd.read_csv(mask_path)

    for name, frame in [
        ("probability", prob_df),
        ("weight", weight_df),
        ("mask", mask_df),
    ]:
        if UID not in frame.columns:
            raise RuntimeError(f"{name} file missing {UID}")
        frame[UID] = frame[UID].astype(str)
        missing = [label for label in LABELS if label not in frame.columns]
        if missing:
            raise RuntimeError(f"{name} file missing labels: {missing}")
        if frame[UID].duplicated().any():
            raise RuntimeError(f"{name} file has duplicate UIDs")

    base_uids = prob_df[UID].tolist()

    if len(base_uids) != EXPECTED_UNLABELED:
        raise RuntimeError(
            f"Expected {EXPECTED_UNLABELED} teacher rows; " f"found {len(base_uids)}"
        )

    if set(weight_df[UID]) != set(base_uids) or set(mask_df[UID]) != set(base_uids):
        raise RuntimeError("Teacher probability/weight/mask UID sets differ.")

    weight_df = weight_df.set_index(UID).loc[base_uids].reset_index()
    mask_df = mask_df.set_index(UID).loc[base_uids].reset_index()

    probability = prob_df[LABELS].to_numpy(dtype=np.float32)
    weight = weight_df[LABELS].to_numpy(dtype=np.float32)
    mask = np.column_stack(
        [bool_series(mask_df[label]).to_numpy(dtype=bool) for label in LABELS]
    ).astype(bool)

    if probability.shape != (EXPECTED_UNLABELED, NUM_LABELS):
        raise RuntimeError(f"Unexpected teacher probability shape {probability.shape}")
    if not np.isfinite(probability).all():
        raise RuntimeError("Teacher probabilities contain non-finite values.")
    if not ((probability >= 0) & (probability <= 1)).all():
        raise RuntimeError("Teacher probabilities outside [0,1].")
    if not np.isfinite(weight).all():
        raise RuntimeError("Teacher weights contain non-finite values.")
    if not ((weight >= 0) & (weight <= 1)).all():
        raise RuntimeError("Teacher weights outside [0,1].")
    if strict and (weight[mask] <= 0).any():
        raise RuntimeError("Selected teacher cells contain non-positive weight.")
    if int(mask.sum()) != EXPECTED_SELECTED_CELLS:
        raise RuntimeError(
            f"Selected teacher cells={int(mask.sum())}, "
            f"expected={EXPECTED_SELECTED_CELLS}"
        )

    if train_df is not None:
        unlabeled = train_df[train_df[LABELS].isna().all(axis=1)].copy()
        unlabeled_uids = set(unlabeled[UID].astype(str))
        teacher_uids = set(base_uids)
        if teacher_uids != unlabeled_uids:
            missing = sorted(unlabeled_uids - teacher_uids)[:5]
            extra = sorted(teacher_uids - unlabeled_uids)[:5]
            raise RuntimeError(
                "Teacher/train UID mismatch: " f"missing={missing}, extra={extra}"
            )

    effective_weight = weight.copy()
    effective_weight[~mask] = 0.0

    teacher_summary = {
        "teacher_root": str(paths.w40_root),
        "probability_file": str(prob_path),
        "weight_file": str(weight_path),
        "mask_file": str(mask_path),
        "probability_sha256": sha256_file(prob_path),
        "weight_sha256": sha256_file(weight_path),
        "mask_sha256": sha256_file(mask_path),
        "rows": len(base_uids),
        "selected_cells": int(mask.sum()),
        "selected_cells_per_label": {
            label: int(mask[:, j].sum()) for j, label in enumerate(LABELS)
        },
        "mean_selected_weight_per_label": {
            label: (float(weight[mask[:, j], j].mean()) if mask[:, j].any() else 0.0)
            for j, label in enumerate(LABELS)
        },
        "w40_summary": summary,
        "w40_validation": validation,
        "scientific_warning": (
            "W40 uses all 58 gold reports as final production exemplars. "
            "Downstream gold diagnostics are NON-PRISTINE."
        ),
    }

    ensure_dirs(paths)
    atomic_json_dump(
        teacher_summary,
        paths.result_root / "01_teacher_validation.json",
    )

    return TeacherData(
        uids=base_uids,
        probability=probability,
        weight=effective_weight,
        mask=mask,
        root=paths.w40_root,
        summary=teacher_summary,
    )


# =============================================================================
# 7. CANONICAL W4 CLS FEATURE STORE
# =============================================================================


def cache_path(paths: Paths, uid: str) -> Path:
    return paths.w4_cache_root / f"{stable_uid_hash(uid)}.pt"


def w4_payload_usable(
    payload: Mapping[str, Any],
    uid: str,
) -> bool:
    try:
        features = payload["features"]
        return (
            str(payload.get("cache_version")) == EXPECTED_W4_CACHE_VERSION
            and str(payload.get("study_uid")) == str(uid)
            and isinstance(features, torch.Tensor)
            and features.ndim == 3
            and features.shape[-1] == EXPECTED_CURIA_HIDDEN
            and features.dtype == torch.float16
            and payload["slice_mask"].shape == features.shape[:2]
            and payload["slice_position"].shape == features.shape[:2]
            and payload["series_meta"].shape[0] == features.shape[0]
            and payload["series_meta"].shape[1] == 3
            and payload["series_cont"].shape[0] == features.shape[0]
            and payload["series_cont"].shape[1] == 2
        )
    except Exception:
        return False


class FeatureStore:
    def __init__(
        self,
        paths: Paths,
        uids: Sequence[str],
        lru_size: int = FEATURE_LRU_SIZE,
    ):
        self.paths = paths
        self.uids = [str(uid) for uid in uids]
        self.uid_to_index = {uid: i for i, uid in enumerate(self.uids)}
        if len(self.uid_to_index) != len(self.uids):
            raise RuntimeError("Duplicate UID in feature store.")

        self.lru_size = max(0, int(lru_size))
        self._lru: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def indices_for_uids(self, uids: Sequence[str]) -> np.ndarray:
        missing = [str(uid) for uid in uids if str(uid) not in self.uid_to_index]
        if missing:
            raise KeyError(f"Feature store missing UIDs: {missing[:5]}")
        return np.asarray(
            [self.uid_to_index[str(uid)] for uid in uids],
            dtype=np.int64,
        )

    def path_for_uid(self, uid: str) -> Path:
        return cache_path(self.paths, str(uid))

    def _load(self, uid: str) -> Dict[str, Any]:
        uid = str(uid)

        if uid in self._lru:
            payload = self._lru.pop(uid)
            self._lru[uid] = payload
            return payload

        path = self.path_for_uid(uid)
        if not path.exists():
            raise FileNotFoundError(f"Missing W4 cache for {uid}: {path}")

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not w4_payload_usable(payload, uid):
            raise RuntimeError(f"Invalid/stale W4 cache: {path}")

        # Ensure no spatial branch can enter this controlled experiment.
        if "patch_grid" in payload:
            payload = {
                key: value for key, value in payload.items() if key != "patch_grid"
            }

        if self.lru_size > 0:
            self._lru[uid] = payload
            while len(self._lru) > self.lru_size:
                self._lru.popitem(last=False)

        return payload

    def validate_all(self) -> Dict[str, Any]:
        missing: List[str] = []
        invalid: List[str] = []

        for uid in self.uids:
            path = self.path_for_uid(uid)
            if not path.exists():
                missing.append(uid)
                continue
            try:
                payload = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
                if not w4_payload_usable(payload, uid):
                    invalid.append(uid)
            except Exception:
                invalid.append(uid)

        return {
            "total": len(self.uids),
            "usable": len(self.uids) - len(missing) - len(invalid),
            "missing": len(missing),
            "invalid": len(invalid),
            "missing_examples": missing[:5],
            "invalid_examples": invalid[:5],
        }

    def make_batch(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        payloads = [self._load(self.uids[int(index)]) for index in indices]

        max_series = max(int(payload["features"].shape[0]) for payload in payloads)
        max_slices = max(int(payload["features"].shape[1]) for payload in payloads)
        batch_size = len(payloads)

        features = torch.zeros(
            batch_size,
            max_series,
            max_slices,
            EXPECTED_CURIA_HIDDEN,
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
            dtype=torch.float16,
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

        for batch_index, payload in enumerate(payloads):
            series_count, slice_count, _ = payload["features"].shape
            features[
                batch_index,
                :series_count,
                :slice_count,
            ] = payload["features"]
            slice_mask[
                batch_index,
                :series_count,
                :slice_count,
            ] = payload["slice_mask"]
            slice_position[
                batch_index,
                :series_count,
                :slice_count,
            ] = payload["slice_position"]
            series_meta[
                batch_index,
                :series_count,
            ] = payload["series_meta"].long()
            series_cont[
                batch_index,
                :series_count,
            ] = payload["series_cont"].float()
            series_mask[
                batch_index,
                :series_count,
            ] = payload[
                "slice_mask"
            ].any(dim=-1)

        return {
            "features": features.to(device, non_blocking=False),
            "slice_mask": slice_mask.to(device, non_blocking=False),
            "slice_position": slice_position.to(device, non_blocking=False),
            "series_meta": series_meta.to(device, non_blocking=False),
            "series_cont": series_cont.to(device, non_blocking=False),
            "series_mask": series_mask.to(device, non_blocking=False),
        }


def validate_cache_once(
    paths: Paths,
    train_df: pd.DataFrame,
    force: bool = False,
) -> Dict[str, Any]:
    """Validate the 4,407-study canonical cache once per W41 output root.

    status writes the sentinel; subsequent fold/full workers reuse it instead of
    reopening all 4,407 tensors before every training job.
    """
    ensure_dirs(paths)
    summary_path = paths.result_root / "02_cache_validation.json"

    if summary_path.exists() and not force:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                existing.get("cache_root") == str(paths.w4_cache_root)
                and existing.get("expected_cache_version") == EXPECTED_W4_CACHE_VERSION
                and existing.get("total") == EXPECTED_TRAIN
                and existing.get("usable") == EXPECTED_TRAIN
                and existing.get("missing") == 0
                and existing.get("invalid") == 0
            ):
                return existing
        except Exception:
            pass

    store = FeatureStore(
        paths,
        train_df[UID].astype(str).tolist(),
    )
    validation = store.validate_all()

    payload = {
        **validation,
        "cache_root": str(paths.w4_cache_root),
        "expected_cache_version": EXPECTED_W4_CACHE_VERSION,
        "created_at": now_iso(),
    }
    atomic_json_dump(payload, summary_path)
    return payload


# =============================================================================
# 8. EXACT W4 / W6.0 HIERARCHICAL HEAD
# =============================================================================


def _drop_mask_tokens(
    mask: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    if probability <= 0:
        return mask

    output = mask & (
        torch.rand(
            mask.shape,
            device=mask.device,
        )
        >= probability
    )

    need_restore = mask.any(dim=-1) & ~output.any(dim=-1)

    if need_restore.any():
        first_valid = mask.float().argmax(dim=-1)

        for coordinate in torch.nonzero(
            need_restore,
            as_tuple=False,
        ):
            prefix = tuple(int(x) for x in coordinate.tolist())
            token = int(first_valid[prefix].item())
            output[prefix + (token,)] = True

    return output


class W4ExactCuriaHierarchicalDiagnosisHead(nn.Module):
    """Exact architecture-level copy of the W4/W6.0 CLS head."""

    def __init__(self):
        super().__init__()

        if HEAD_HIDDEN_DIM % HEAD_NUM_HEADS != 0:
            raise ValueError("HEAD_HIDDEN_DIM must be divisible by HEAD_NUM_HEADS")

        self.input_norm = nn.LayerNorm(EXPECTED_CURIA_HIDDEN)
        self.input_projection = nn.Linear(
            EXPECTED_CURIA_HIDDEN,
            HEAD_HIDDEN_DIM,
        )

        self.position_projection = nn.Sequential(
            nn.Linear(9, HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(
                HEAD_HIDDEN_DIM,
                HEAD_HIDDEN_DIM,
            ),
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
            torch.randn(
                NUM_LABELS,
                HEAD_HIDDEN_DIM,
            )
            * 0.02
        )

        self.plane_embedding = nn.Embedding(
            3,
            32,
        )
        self.fluid_embedding = nn.Embedding(
            2,
            16,
        )
        self.fs_embedding = nn.Embedding(
            2,
            16,
        )

        self.metadata_projection = nn.Sequential(
            nn.Linear(
                32 + 16 + 16 + 2,
                HEAD_HIDDEN_DIM,
            ),
            nn.LayerNorm(HEAD_HIDDEN_DIM),
            nn.GELU(),
        )

        self.series_norm = nn.LayerNorm(HEAD_HIDDEN_DIM)

        self.series_queries = nn.Parameter(
            torch.randn(
                NUM_LABELS,
                HEAD_HIDDEN_DIM,
            )
            * 0.02
        )

        self.final_norm = nn.LayerNorm(HEAD_HIDDEN_DIM)

        self.dropout = nn.Dropout(HEAD_DROPOUT)

        self.classifier_weight = nn.Parameter(
            torch.randn(
                NUM_LABELS,
                HEAD_HIDDEN_DIM,
            )
            * 0.02
        )

        self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    @staticmethod
    def _fourier_position(
        position: torch.Tensor,
    ) -> torch.Tensor:
        position = position.clamp(
            -1.0,
            1.0,
        )

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

        if self.training and SERIES_DROPOUT > 0:
            series_mask_effective = _drop_mask_tokens(
                series_mask,
                SERIES_DROPOUT,
            )
        else:
            series_mask_effective = series_mask

        slice_mask_effective = slice_mask & series_mask_effective.unsqueeze(-1)

        if self.training and SLICE_DROPOUT > 0:
            batch_size, max_series, max_slices = slice_mask_effective.shape

            flat = _drop_mask_tokens(
                slice_mask_effective.reshape(
                    batch_size * max_series,
                    max_slices,
                ),
                SLICE_DROPOUT,
            )

            slice_mask_effective = flat.reshape(
                batch_size,
                max_series,
                max_slices,
            )

        (
            batch_size,
            max_series,
            max_slices,
            _,
        ) = features.shape

        real_series_flat = series_mask_effective.reshape(-1)

        flat_features = features.reshape(
            batch_size * max_series,
            max_slices,
            EXPECTED_CURIA_HIDDEN,
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
            self._fourier_position(flat_position.float())
        )

        hidden = self.slice_transformer(
            hidden,
            src_key_padding_mask=(~flat_slice_mask),
        )

        slice_scores = torch.einsum(
            "rkh,lh->rlk",
            hidden,
            self.slice_queries,
        ) / math.sqrt(HEAD_HIDDEN_DIM)

        slice_scores = slice_scores.masked_fill(
            ~flat_slice_mask.unsqueeze(1),
            -1e4,
        )

        slice_attention = torch.softmax(
            slice_scores,
            dim=-1,
        )

        pooled_series = torch.einsum(
            "rlk,rkh->rlh",
            slice_attention,
            hidden,
        )

        flat_meta = series_meta.reshape(
            batch_size * max_series,
            3,
        )[real_series_flat]

        flat_cont = series_cont.reshape(
            batch_size * max_series,
            2,
        )[real_series_flat]

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
            batch_size,
            max_series,
            NUM_LABELS,
            HEAD_HIDDEN_DIM,
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

        series_attention = torch.softmax(
            series_scores,
            dim=1,
        )

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


# =============================================================================
# 9. LOSSES
# =============================================================================


def compute_gold_pos_weight(
    train_labels: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    positives = train_labels.sum(axis=0).astype(np.float64)
    negatives = len(train_labels) - positives
    weights = negatives / np.maximum(
        positives,
        1.0,
    )
    weights = np.clip(
        weights,
        1.0,
        5.0,
    ).astype(np.float32)
    return torch.tensor(
        weights,
        device=device,
        dtype=torch.float32,
    )


def macro_gold_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
        pos_weight=pos_weight,
    )
    return cell.mean(dim=0).mean()


def macro_weighted_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    cell = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
    )

    effective = mask.float() * weight.float()

    weight_sum = effective.sum(dim=0)

    active = weight_sum > 0

    if not active.any():
        raise RuntimeError("Pseudo batch contains no active weighted target cells.")

    per_label = (cell * effective).sum(dim=0) / weight_sum.clamp_min(1e-6)

    return (
        per_label[active].mean(),
        weight_sum,
    )


def sample_rows(
    rng: np.random.Generator,
    available: np.ndarray,
    size: int,
) -> np.ndarray:
    if len(available) == 0:
        raise RuntimeError("Cannot sample from empty index set.")
    return available[
        rng.integers(
            0,
            len(available),
            size=size,
        )
    ]


def teacher_arrays(
    store: FeatureStore,
    teacher: TeacherData,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    indices = store.indices_for_uids(teacher.uids)
    return (
        indices,
        teacher.probability,
        teacher.mask,
        teacher.weight,
    )


def gold_arrays(
    store: FeatureStore,
    gold_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    return (
        store.indices_for_uids(gold_df[UID].astype(str).tolist()),
        gold_df[LABELS].to_numpy(dtype=np.float32),
    )


# =============================================================================
# 10. TRAINING CORE
# =============================================================================


def training_config(
    paths: Paths,
    scope: str,
    seed: int,
    teacher: TeacherData,
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "stage": "w41",
        "anchor": "exact_w6_0_cls_head",
        "scope": scope,
        "seed": seed,
        "fold_sha256": EXPECTED_FOLD_SHA256,
        "teacher": "W40 FS2 production teacher",
        "teacher_probability_sha256": teacher.summary["probability_sha256"],
        "teacher_weight_sha256": teacher.summary["weight_sha256"],
        "teacher_mask_sha256": teacher.summary["mask_sha256"],
        "teacher_selected_cells": int(teacher.mask.sum()),
        "teacher_changes_only": FS2_LABELS,
        "gold_authority": GOLD_AUTHORITY,
        "pseudo_authority": PSEUDO_AUTHORITY,
        "epochs": HEAD_EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "gold_batch": GOLD_BATCH_SIZE,
        "pseudo_batch": PSEUDO_BATCH_SIZE,
        "max_lr": HEAD_MAX_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP_NORM,
        "representation": "canonical W4 per-slice Curia CLS cache",
        "w4_cache_version": EXPECTED_W4_CACHE_VERSION,
        "head_hidden": HEAD_HIDDEN_DIM,
        "head_heads": HEAD_NUM_HEADS,
        "head_layers": HEAD_TRANSFORMER_LAYERS,
        "head_dropout": HEAD_DROPOUT,
        "slice_dropout": SLICE_DROPOUT,
        "series_dropout": SERIES_DROPOUT,
        "diagnostic_pristine": False,
    }


def make_model(
    device: torch.device,
) -> nn.Module:
    return W4ExactCuriaHierarchicalDiagnosisHead().to(device)


def predict_indices(
    model: nn.Module,
    store: FeatureStore,
    indices: np.ndarray,
    runtime: Runtime,
    batch_size: int = VALIDATION_BATCH_SIZE,
) -> np.ndarray:

    model.eval()
    outputs: List[np.ndarray] = []

    with torch.inference_mode():

        for start in range(
            0,
            len(indices),
            batch_size,
        ):

            part = indices[start : start + batch_size]

            batch = store.make_batch(
                part,
                runtime.device,
            )

            with autocast_context(runtime):

                logits = model(**batch)["logits"]

            outputs.append(torch.sigmoid(logits.float()).cpu().numpy())

            del batch, logits

    if not outputs:
        return np.zeros(
            (
                0,
                NUM_LABELS,
            ),
            dtype=np.float32,
        )

    return np.concatenate(
        outputs,
        axis=0,
    ).astype(np.float32)


def train_model(
    paths: Paths,
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
) -> Tuple[
    nn.Module,
    pd.DataFrame,
    Optional[pd.DataFrame],
]:

    config = training_config(
        paths,
        scope,
        seed,
        teacher,
    )

    config_hash = sha256_json(config)

    if checkpoint_path.exists():

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("config_hash") != config_hash:

            raise RuntimeError("Stale/incompatible checkpoint: " f"{checkpoint_path}")

        model = make_model(runtime.device)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()

        prediction_path = checkpoint_path.with_name(
            checkpoint_path.stem + "_diagnostic_predictions.csv"
        )

        prediction = pd.read_csv(prediction_path) if prediction_path.exists() else None

        log(f"{scope}: compatible checkpoint exists -> reuse")

        return (
            model,
            history,
            prediction,
        )

    seed_everything(seed)

    model = make_model(runtime.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HEAD_MAX_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HEAD_MAX_LR,
        total_steps=(HEAD_EPOCHS * STEPS_PER_EPOCH),
        pct_start=0.10,
        anneal_strategy="cos",
    )

    scaler = make_grad_scaler(runtime)

    gold_pos_weight = compute_gold_pos_weight(
        gold_train_targets,
        runtime.device,
    )

    (
        pseudo_indices,
        pseudo_targets,
        pseudo_masks,
        pseudo_weights,
    ) = teacher_arrays(
        store,
        teacher,
    )

    gold_target_map = {
        int(index): gold_train_targets[row]
        for row, index in enumerate(gold_train_indices)
    }

    gold_rng = np.random.default_rng(seed + 1000)

    pseudo_rng = np.random.default_rng(seed + 2000)

    history_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    diagnostic_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    started = time.time()

    log("")
    log("-" * 96)
    log(f"W41 | {scope} | " f"seed={seed} | " f"device={runtime.device}")
    log("-" * 96)
    log(f"Gold train studies     : " f"{len(gold_train_indices)}")
    log(f"Pseudo studies         : " f"{len(pseudo_indices)}")
    log(f"Teacher selected cells : " f"{int(pseudo_masks.sum())}")
    log("Teacher probabilities  : " "W40; only Contusion/Effusion differ from W6.0")
    log(
        f"Loss authority         : "
        f"gold={GOLD_AUTHORITY:g}, "
        f"pseudo={PSEUDO_AUTHORITY:g}"
    )

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):

        model.train()

        total_gold = 0.0
        total_pseudo = 0.0
        total_loss_value = 0.0
        pseudo_mass = np.zeros(
            NUM_LABELS,
            dtype=np.float64,
        )

        for _step in range(STEPS_PER_EPOCH):

            sampled_gold = sample_rows(
                gold_rng,
                gold_train_indices,
                GOLD_BATCH_SIZE,
            )

            gold_batch = store.make_batch(
                sampled_gold,
                runtime.device,
            )

            gold_target_np = np.stack(
                [gold_target_map[int(index)] for index in sampled_gold],
                axis=0,
            )

            gold_target = torch.tensor(
                gold_target_np,
                device=runtime.device,
                dtype=torch.float32,
            )

            pseudo_positions = pseudo_rng.integers(
                0,
                len(pseudo_indices),
                size=PSEUDO_BATCH_SIZE,
            )

            sampled_pseudo = pseudo_indices[pseudo_positions]

            pseudo_batch = store.make_batch(
                sampled_pseudo,
                runtime.device,
            )

            pseudo_target = torch.tensor(
                pseudo_targets[pseudo_positions],
                device=runtime.device,
                dtype=torch.float32,
            )

            pseudo_mask = torch.tensor(
                pseudo_masks[pseudo_positions],
                device=runtime.device,
                dtype=torch.bool,
            )

            pseudo_weight = torch.tensor(
                pseudo_weights[pseudo_positions],
                device=runtime.device,
                dtype=torch.float32,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(runtime):

                gold_logits = model(**gold_batch)["logits"]

                gold_loss = macro_gold_bce(
                    gold_logits,
                    gold_target,
                    gold_pos_weight,
                )

                pseudo_logits = model(**pseudo_batch)["logits"]

                pseudo_loss, mass = macro_weighted_soft_bce(
                    pseudo_logits,
                    pseudo_target,
                    pseudo_mask,
                    pseudo_weight,
                )

                loss = (
                    (GOLD_AUTHORITY * gold_loss) + (PSEUDO_AUTHORITY * pseudo_loss)
                ) / (GOLD_AUTHORITY + PSEUDO_AUTHORITY)

            if scaler is not None:

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )

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
                loss,
            )

        row: Dict[
            str,
            Any,
        ] = {
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

            prediction = predict_indices(
                model,
                store,
                val_indices,
                runtime,
            )

            _table, summary = metric_tables(
                val_targets.astype(np.int64),
                prediction,
            )

            row.update(
                {
                    "DiagnosticMacroAUROC": summary["macro_AUROC"],
                    "DiagnosticMacroAP": summary["macro_AP"],
                    "DiagnosticMacroF1": summary["macro_F1"],
                    "DiagnosticIsPristineOOF": False,
                }
            )

            if epoch == HEAD_EPOCHS and val_uids is not None:

                for (
                    uid,
                    truth,
                    probability,
                ) in zip(
                    val_uids,
                    val_targets,
                    prediction,
                ):

                    output = {
                        UID: str(uid),
                        "Scope": scope,
                        "Seed": seed,
                        "Epoch": epoch,
                        "PristineOOF": False,
                    }

                    for j, label in enumerate(LABELS):

                        output[f"Truth::{label}"] = int(truth[j])

                        output[f"Probability::{label}"] = float(probability[j])

                    diagnostic_rows.append(output)

        history_rows.append(row)

        pd.DataFrame(history_rows).to_csv(
            history_path,
            index=False,
        )

        metric_text = (
            f" diag_auc=" f"{row.get('DiagnosticMacroAUROC', float('nan')):.5f}"
            if "DiagnosticMacroAUROC" in row
            else ""
        )

        log(
            f"epoch "
            f"{epoch:02d}/"
            f"{HEAD_EPOCHS} "
            f"loss="
            f"{row['TotalLoss']:.5f} "
            f"gold="
            f"{row['GoldLoss']:.5f} "
            f"pseudo="
            f"{row['PseudoLoss']:.5f}"
            f"{metric_text}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "script_version": SCRIPT_VERSION,
        "scope": scope,
        "seed": seed,
        "config": config,
        "config_hash": config_hash,
        "created_at": now_iso(),
        "pristine_oof": False,
        "warning": ("Model trained with all-58-derived W40 production pseudo labels."),
    }

    atomic_torch_save(
        checkpoint,
        checkpoint_path,
    )

    prediction_frame = pd.DataFrame(diagnostic_rows) if diagnostic_rows else None

    if prediction_frame is not None:

        prediction_frame.to_csv(
            checkpoint_path.with_name(
                checkpoint_path.stem + "_diagnostic_predictions.csv"
            ),
            index=False,
        )

    return (
        model,
        pd.DataFrame(history_rows),
        prediction_frame,
    )


# =============================================================================
# 11. ORCHESTRATION
# =============================================================================


def make_store(
    paths: Paths,
    train_df: pd.DataFrame,
) -> FeatureStore:

    store = FeatureStore(
        paths,
        train_df[UID].astype(str).tolist(),
    )

    validation = validate_cache_once(
        paths,
        train_df,
        force=False,
    )

    if validation["missing"] or validation["invalid"]:
        raise RuntimeError("Canonical W4 feature cache incomplete: " f"{validation}")

    return store


def cv_checkpoint_path(
    paths: Paths,
    fold: int,
) -> Path:
    return paths.checkpoint_root / "cv" / f"fold_{fold}_epoch_{HEAD_EPOCHS}.pt"


def cv_history_path(
    paths: Paths,
    fold: int,
) -> Path:
    return paths.result_root / "cv" / f"fold_{fold}_history.csv"


def full_checkpoint_path(
    paths: Paths,
    seed: int,
) -> Path:
    return paths.checkpoint_root / "full" / f"seed_{seed}_epoch_{HEAD_EPOCHS}.pt"


def full_history_path(
    paths: Paths,
    seed: int,
) -> Path:
    return paths.result_root / "full" / f"seed_{seed}_history.csv"


def train_one_cv_fold(
    paths: Paths,
    fold: int,
    accelerator: str,
    precision: str,
    device_index: int,
) -> Dict[str, Any]:

    runtime = resolve_runtime(
        accelerator,
        precision,
        device_index=device_index,
    )

    train_df, gold_df, fold_zero = load_train_state(paths)

    teacher = load_teacher(
        paths,
        train_df,
        strict=True,
    )

    store = make_store(
        paths,
        train_df,
    )

    gold_indices, gold_y = gold_arrays(
        store,
        gold_df,
    )

    train_rows = np.where(fold_zero != fold - 1)[0]

    val_rows = np.where(fold_zero == fold - 1)[0]

    seed = CV_SEED_BASE + fold

    checkpoint = cv_checkpoint_path(
        paths,
        fold,
    )

    history = cv_history_path(
        paths,
        fold,
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, hist, prediction = train_model(
        paths=paths,
        scope=f"cv_fold_{fold}",
        seed=seed,
        store=store,
        gold_train_indices=gold_indices[train_rows],
        gold_train_targets=gold_y[train_rows],
        teacher=teacher,
        runtime=runtime,
        checkpoint_path=checkpoint,
        history_path=history,
        val_indices=gold_indices[val_rows],
        val_targets=gold_y[val_rows],
        val_uids=gold_df.iloc[val_rows][UID].astype(str).tolist(),
    )

    if prediction is None:
        raise RuntimeError(f"Fold {fold} produced no diagnostic predictions.")

    prediction["OuterFold"] = fold

    prediction.to_csv(
        paths.result_root / "cv" / f"fold_{fold}_diagnostic_predictions.csv",
        index=False,
    )

    result = {
        "fold": fold,
        "seed": seed,
        "device": str(runtime.device),
        "checkpoint": str(checkpoint),
        "history": str(history),
        "final_total_loss": (
            float(hist.iloc[-1]["TotalLoss"]) if not hist.empty else None
        ),
        "final_fold_diag_auc": (
            float(hist.iloc[-1]["DiagnosticMacroAUROC"])
            if (not hist.empty and "DiagnosticMacroAUROC" in hist.columns)
            else None
        ),
    }

    del model

    if runtime.accelerator == "cuda":
        torch.cuda.empty_cache()

    return result


def combine_cv(
    paths: Paths,
) -> Dict[str, Any]:

    frames = []

    fold_summaries = []

    for fold in range(
        1,
        NUM_FOLDS + 1,
    ):

        prediction_path = (
            paths.result_root / "cv" / f"fold_{fold}_diagnostic_predictions.csv"
        )

        history_path = cv_history_path(
            paths,
            fold,
        )

        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        frames.append(pd.read_csv(prediction_path))

        if history_path.exists():
            history = pd.read_csv(history_path)

            if not history.empty:
                fold_summaries.append(history.iloc[-1].to_dict())

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(combined) != EXPECTED_GOLD:
        raise RuntimeError(
            f"Combined CV rows={len(combined)}, expected={EXPECTED_GOLD}"
        )

    if combined[UID].astype(str).duplicated().any():
        raise RuntimeError("Duplicate held-out gold UID in combined W41 CV.")

    truth = np.column_stack(
        [combined[f"Truth::{label}"].to_numpy(dtype=np.int64) for label in LABELS]
    )

    probability = np.column_stack(
        [
            combined[f"Probability::{label}"].to_numpy(dtype=np.float32)
            for label in LABELS
        ]
    )

    table, metrics = metric_tables(
        truth,
        probability,
    )

    combined_path = (
        paths.result_root / "cv" / "diagnostic_gold_predictions_all_folds.csv"
    )

    metric_path = paths.result_root / "cv" / "diagnostic_per_label_metrics.csv"

    combined.to_csv(
        combined_path,
        index=False,
    )

    table.to_csv(
        metric_path,
        index=False,
    )

    weak = table[table["AUROC"] < WEAK_LABEL_AUC]["Label"].tolist()

    macro_auc = float(metrics["macro_AUROC"])

    delta_vs_w60 = macro_auc - W60_DIAGNOSTIC_REFERENCE

    if macro_auc >= GO_MACRO_AUC_MIN and len(weak) <= MAX_WEAK_LABELS:
        verdict = "GO_TO_FULL"
    elif macro_auc >= REVIEW_MACRO_AUC_MIN:
        verdict = "REVIEW"
    else:
        verdict = "STOP"

    summary = {
        "script_version": SCRIPT_VERSION,
        "diagnostic_is_pristine_oof": False,
        "warning": (
            "W40 teacher used all 58 gold reports; "
            "this is a NON-PRISTINE diagnostic."
        ),
        "w60_public_lb": 0.730,
        "w60_reference_diagnostic_macro_AUROC": W60_DIAGNOSTIC_REFERENCE,
        "diagnostic_metrics": metrics,
        "delta_vs_w60_reference": delta_vs_w60,
        "weak_label_threshold": WEAK_LABEL_AUC,
        "weak_labels": weak,
        "gate": {
            "go_macro_auc_min": GO_MACRO_AUC_MIN,
            "review_macro_auc_min": REVIEW_MACRO_AUC_MIN,
            "max_weak_labels": MAX_WEAK_LABELS,
            "verdict": verdict,
        },
        "folds": fold_summaries,
        "next_step": (
            "train_full"
            if verdict == "GO_TO_FULL"
            else ("manual review" if verdict == "REVIEW" else "stop W41")
        ),
    }

    atomic_json_dump(
        summary,
        paths.result_root / "cv" / "diagnostic_summary.json",
    )

    log("")
    log("=" * 96)
    log("W41 DIAGNOSTIC")
    log("=" * 96)
    log(f"Macro AUROC          : " f"{macro_auc:.6f}")
    log(f"W6.0 reference       : " f"{W60_DIAGNOSTIC_REFERENCE:.6f}")
    log(f"Delta vs W6.0        : " f"{delta_vs_w60:+.6f}")
    log(f"Weak labels < .55    : " f"{len(weak)} -> {weak}")
    log(f"VERDICT              : " f"{verdict}")

    return summary


def hidden_worker_command(
    paths: Paths,
    mode: str,
    accelerator: str,
    precision: str,
    device_index: int,
    *,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        mode,
        "--accelerator",
        accelerator,
        "--precision",
        precision,
        "--device-index",
        str(device_index),
        "--project-root",
        str(paths.project_root),
        "--train-csv",
        str(paths.train_csv),
        "--w4-cache",
        str(paths.w4_cache_root),
        "--w40-root",
        str(paths.w40_root),
        "--output-root",
        str(paths.result_root.parent),
    ]

    if fold is not None:
        command.extend(
            [
                "--fold",
                str(fold),
            ]
        )

    if seed is not None:
        command.extend(
            [
                "--seed",
                str(seed),
            ]
        )

    return command


def run_parallel_jobs(
    jobs: Sequence[
        Tuple[
            str,
            List[str],
        ]
    ],
    max_workers: int,
) -> None:

    log(f"Parallel workers      : " f"{max_workers}")

    def execute(
        name: str,
        command: List[str],
    ) -> str:

        completed = subprocess.run(
            command,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Worker {name} failed " f"with code={completed.returncode}"
            )

        return name

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(
                execute,
                name,
                command,
            ): name
            for name, command in jobs
        }

        for future in as_completed(futures):
            name = futures[future]
            future.result()
            log(f"Completed worker       : " f"{name}")


def train_cv(
    paths: Paths,
    accelerator: str,
    precision: str,
) -> Dict[str, Any]:

    ensure_dirs(paths)

    device_count = accelerator_device_count(accelerator)

    resolved = normalize_accelerator(accelerator)

    if (
        resolved
        in {
            "auto",
            "cuda",
        }
        and torch.cuda.is_available()
        and device_count > 1
    ):

        jobs = []

        for index, fold in enumerate(
            range(
                1,
                NUM_FOLDS + 1,
            )
        ):

            device_index = index % device_count

            jobs.append(
                (
                    f"cv_fold_{fold}",
                    hidden_worker_command(
                        paths,
                        "_worker_cv",
                        accelerator,
                        precision,
                        device_index,
                        fold=fold,
                    ),
                )
            )

        run_parallel_jobs(
            jobs,
            max_workers=min(
                device_count,
                NUM_FOLDS,
            ),
        )

    else:

        for fold in range(
            1,
            NUM_FOLDS + 1,
        ):

            train_one_cv_fold(
                paths,
                fold,
                accelerator,
                precision,
                device_index=0,
            )

    return combine_cv(paths)


def train_one_full_seed(
    paths: Paths,
    seed: int,
    accelerator: str,
    precision: str,
    device_index: int,
) -> Dict[str, Any]:

    runtime = resolve_runtime(
        accelerator,
        precision,
        device_index=device_index,
    )

    train_df, gold_df, _fold_zero = load_train_state(paths)

    teacher = load_teacher(
        paths,
        train_df,
        strict=True,
    )

    store = make_store(
        paths,
        train_df,
    )

    gold_indices, gold_y = gold_arrays(
        store,
        gold_df,
    )

    checkpoint = full_checkpoint_path(
        paths,
        seed,
    )

    history = full_history_path(
        paths,
        seed,
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, hist, _prediction = train_model(
        paths=paths,
        scope="full_all58",
        seed=seed,
        store=store,
        gold_train_indices=gold_indices,
        gold_train_targets=gold_y,
        teacher=teacher,
        runtime=runtime,
        checkpoint_path=checkpoint,
        history_path=history,
    )

    result = {
        "seed": seed,
        "device": str(runtime.device),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "final_total_loss": (
            float(hist.iloc[-1]["TotalLoss"]) if not hist.empty else None
        ),
    }

    del model

    if runtime.accelerator == "cuda":
        torch.cuda.empty_cache()

    return result


def load_cv_gate(
    paths: Paths,
) -> Dict[str, Any]:

    path = paths.result_root / "cv" / "diagnostic_summary.json"

    if not path.exists():
        raise FileNotFoundError(
            "CV diagnostic summary does not exist. " "Run train_cv first."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def train_full(
    paths: Paths,
    accelerator: str,
    precision: str,
    force: bool,
) -> Dict[str, Any]:

    ensure_dirs(paths)

    gate = load_cv_gate(paths)

    verdict = gate.get("gate", {}).get("verdict")

    if verdict != "GO_TO_FULL" and not force:

        raise RuntimeError(
            f"W41 CV verdict is {verdict!r}; "
            "full training is blocked. "
            "Use --force only deliberately."
        )

    device_count = accelerator_device_count(accelerator)

    resolved = normalize_accelerator(accelerator)

    if (
        resolved
        in {
            "auto",
            "cuda",
        }
        and torch.cuda.is_available()
        and device_count > 1
    ):

        jobs = []

        for index, seed in enumerate(FULLFIT_SEEDS):

            device_index = index % device_count

            jobs.append(
                (
                    f"full_seed_{seed}",
                    hidden_worker_command(
                        paths,
                        "_worker_full",
                        accelerator,
                        precision,
                        device_index,
                        seed=seed,
                    ),
                )
            )

        run_parallel_jobs(
            jobs,
            max_workers=min(
                device_count,
                len(FULLFIT_SEEDS),
            ),
        )

    else:

        for seed in FULLFIT_SEEDS:

            train_one_full_seed(
                paths,
                seed,
                accelerator,
                precision,
                device_index=0,
            )

    models = []

    for seed in FULLFIT_SEEDS:

        checkpoint = full_checkpoint_path(
            paths,
            seed,
        )

        history_path = full_history_path(
            paths,
            seed,
        )

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()

        models.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "final_total_loss": (
                    float(history.iloc[-1]["TotalLoss"]) if not history.empty else None
                ),
            }
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "scope": "full_all58",
        "seeds": FULLFIT_SEEDS,
        "models": models,
        "cv_gate": gate["gate"],
        "use": "production hidden-test inference",
        "validation_note": (
            "Full-fit uses all 58 gold labels and all W40 pseudo labels; "
            "no OOF claim."
        ),
        "next_step": (
            "Build standalone submission/inference script using these three "
            "W41 checkpoints and the exact canonical W4 Curia preprocessing."
        ),
    }

    atomic_json_dump(
        summary,
        paths.result_root / "full" / "fullfit_summary.json",
    )

    log("")
    log("=" * 96)
    log("W41 FULL FIT COMPLETE")
    log("=" * 96)
    for model in models:
        log(
            f"seed={model['seed']} "
            f"sha256={model['checkpoint_sha256'][:16]}... "
            f"loss={model['final_total_loss']}"
        )

    return summary


# =============================================================================
# 12. STATUS / VALIDATION
# =============================================================================


def status(
    paths: Paths,
    accelerator: str,
    precision: str,
) -> Dict[str, Any]:

    ensure_dirs(paths)

    runtime = resolve_runtime(
        accelerator,
        precision,
        device_index=0,
    )

    train_df, gold_df, _fold_zero = load_train_state(paths)

    teacher = load_teacher(
        paths,
        train_df,
        strict=True,
    )

    cache = validate_cache_once(
        paths,
        train_df,
        force=False,
    )

    info = {
        "script_version": SCRIPT_VERSION,
        "experiment": DISPLAY_NAME,
        "project_root": str(paths.project_root),
        "train_csv": str(paths.train_csv),
        "w4_cache_root": str(paths.w4_cache_root),
        "w40_root": str(paths.w40_root),
        "output_root": str(paths.result_root.parent),
        "runtime": {
            "requested": accelerator,
            "backend": runtime.accelerator,
            "device": str(runtime.device),
            "precision": str(runtime.amp_dtype),
            "visible_cuda_devices": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "accelerator_env_variables_required": False,
        },
        "train_rows": len(train_df),
        "gold_rows": len(gold_df),
        "locked_fold_sha256": EXPECTED_FOLD_SHA256,
        "cache": cache,
        "teacher": {
            "rows": len(teacher.uids),
            "selected_cells": int(teacher.mask.sum()),
            "changed_probabilities": FS2_LABELS,
            "weights_changed": False,
            "masks_changed": False,
        },
        "architecture": {
            "representation": "canonical W4 Curia CLS cache",
            "hidden": HEAD_HIDDEN_DIM,
            "heads": HEAD_NUM_HEADS,
            "layers": HEAD_TRANSFORMER_LAYERS,
            "head_dropout": HEAD_DROPOUT,
            "slice_dropout": SLICE_DROPOUT,
            "series_dropout": SERIES_DROPOUT,
        },
        "training": {
            "epochs": HEAD_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "gold_batch": GOLD_BATCH_SIZE,
            "pseudo_batch": PSEUDO_BATCH_SIZE,
            "lr": HEAD_MAX_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "gold_authority": GOLD_AUTHORITY,
            "pseudo_authority": PSEUDO_AUTHORITY,
            "cv_seed_base": CV_SEED_BASE,
            "full_seeds": FULLFIT_SEEDS,
        },
        "scope": {
            "dicom": False,
            "curia": False,
            "spatial": False,
            "submission_mode": False,
            "project_python_imports": False,
        },
    }

    atomic_json_dump(
        info,
        paths.result_root / "00_status.json",
    )

    log(
        json.dumps(
            info,
            indent=2,
            default=str,
        )
    )

    return info


def validate_teacher(
    paths: Paths,
) -> Dict[str, Any]:

    train_df, _gold_df, _fold_zero = load_train_state(paths)

    teacher = load_teacher(
        paths,
        train_df,
        strict=True,
    )

    log(
        json.dumps(
            teacher.summary,
            indent=2,
            default=str,
        )
    )

    return teacher.summary


def validate(
    paths: Paths,
) -> Dict[str, Any]:

    ensure_dirs(paths)

    train_df, _gold_df, _fold_zero = load_train_state(paths)

    teacher = load_teacher(
        paths,
        train_df,
        strict=True,
    )

    cache = validate_cache_once(
        paths,
        train_df,
        force=False,
    )

    checks: Dict[
        str,
        bool,
    ] = {
        "cache_4407_usable": (cache["usable"] == EXPECTED_TRAIN),
        "teacher_4349_rows": (len(teacher.uids) == EXPECTED_UNLABELED),
        "teacher_selected_cells_32027": (
            int(teacher.mask.sum()) == EXPECTED_SELECTED_CELLS
        ),
        "teacher_probabilities_valid": bool(
            np.isfinite(teacher.probability).all()
            and (teacher.probability >= 0).all()
            and (teacher.probability <= 1).all()
        ),
        "w40_only_fs2_changed": (
            set(
                teacher.summary["w40_summary"]["controlled_changes"][
                    "probabilities_changed"
                ]
            )
            == set(FS2_LABELS)
        ),
    }

    cv_summary_path = paths.result_root / "cv" / "diagnostic_summary.json"

    if cv_summary_path.exists():

        cv = json.loads(cv_summary_path.read_text(encoding="utf-8"))

        checks["cv_summary_script_version"] = cv.get("script_version") == SCRIPT_VERSION

        checks["cv_has_valid_gate"] = cv.get("gate", {}).get("verdict") in {
            "GO_TO_FULL",
            "REVIEW",
            "STOP",
        }

    full_summary_path = paths.result_root / "full" / "fullfit_summary.json"

    if full_summary_path.exists():

        full = json.loads(full_summary_path.read_text(encoding="utf-8"))

        checks["full_summary_script_version"] = (
            full.get("script_version") == SCRIPT_VERSION
        )

        for seed in FULLFIT_SEEDS:

            checkpoint = full_checkpoint_path(
                paths,
                seed,
            )

            checks[f"full_seed_{seed}_checkpoint"] = checkpoint.exists()

            if checkpoint.exists():

                payload = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )

                checks[f"full_seed_{seed}_version"] = (
                    payload.get("script_version") == SCRIPT_VERSION
                )

    overall_pass = bool(all(checks.values()))

    payload = {
        "overall_pass": overall_pass,
        "checks": checks,
        "cache": cache,
        "results_root": str(paths.result_root),
    }

    atomic_json_dump(
        payload,
        paths.result_root / "validation_summary.json",
    )

    log(
        json.dumps(
            payload,
            indent=2,
        )
    )

    if not overall_pass:
        raise RuntimeError("W41 validation failed.")

    return payload


# =============================================================================
# 13. CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description=DISPLAY_NAME)

    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "status",
            "validate_teacher",
            "train_cv",
            "train_full",
            "train_all",
            "validate",
            "_worker_cv",
            "_worker_full",
        ],
        default="status",
    )

    parser.add_argument(
        "--accelerator",
        default="auto",
        help=("auto | localGPU | kaggle_t4 | " "apple_mps | cpu | kaggle_tpu"),
    )

    parser.add_argument(
        "--precision",
        default="auto",
        choices=[
            "auto",
            "bf16",
            "fp16",
            "fp32",
        ],
    )

    parser.add_argument(
        "--project-root",
        default=None,
    )

    parser.add_argument(
        "--train-csv",
        default=None,
    )

    parser.add_argument(
        "--w4-cache",
        default=None,
    )

    parser.add_argument(
        "--w40-root",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        default=None,
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    paths = Paths.discover(args)

    ensure_dirs(paths)

    if args.mode == "status":

        status(
            paths,
            args.accelerator,
            args.precision,
        )

    elif args.mode == "validate_teacher":

        validate_teacher(paths)

    elif args.mode == "_worker_cv":

        if args.fold is None or args.fold not in range(
            1,
            NUM_FOLDS + 1,
        ):
            raise ValueError("--fold must be 1..5")

        train_one_cv_fold(
            paths,
            args.fold,
            args.accelerator,
            args.precision,
            args.device_index,
        )

    elif args.mode == "train_cv":

        train_cv(
            paths,
            args.accelerator,
            args.precision,
        )

    elif args.mode == "_worker_full":

        if args.seed is None or args.seed not in FULLFIT_SEEDS:
            raise ValueError(f"--seed must be one of {FULLFIT_SEEDS}")

        train_one_full_seed(
            paths,
            args.seed,
            args.accelerator,
            args.precision,
            args.device_index,
        )

    elif args.mode == "train_full":

        train_full(
            paths,
            args.accelerator,
            args.precision,
            force=args.force,
        )

    elif args.mode == "train_all":

        cv = train_cv(
            paths,
            args.accelerator,
            args.precision,
        )

        verdict = cv.get("gate", {}).get("verdict")

        if verdict == "GO_TO_FULL":

            train_full(
                paths,
                args.accelerator,
                args.precision,
                force=False,
            )

        else:

            log("")
            log(f"train_all stopped after CV: " f"verdict={verdict}")

    elif args.mode == "validate":

        validate(paths)

    else:

        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
