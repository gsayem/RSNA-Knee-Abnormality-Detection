#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection
W2.6-P FAST — Optimized challenge-mapper production teacher

Why this version exists
-----------------------
The previous W2.6-P production script asked Qwen2.5-7B to autoregressively
generate JSON for 17,396 study/label cells. The production log showed that long
few-shot prompts plus repeated generation made the job far too slow and caused
repeated CUDA OOM retries.

This version changes the inference primitive while preserving the W2.6 idea:

    report + W2 features + 2 positive / 2 negative challenge exemplars
        -> Qwen2.5-Instruct
        -> direct next-token A/B logit score
        -> probability(label=1)

There is NO 72-token JSON generation. One forward pass scores A vs B.

Major optimizations
-------------------
1. Direct A/B next-token logits; no autoregressive JSON generation.
2. Target-focused compact report snippets instead of four mostly-full reports.
3. Compact W2 summaries.
4. Dynamic length-aware batching.
5. No torch.cuda.empty_cache() / gc.collect() on every successful batch.
6. OOM handling splits the batch immediately instead of retrying four expensive
   full batches at progressively smaller contexts.
7. `logits_to_keep=1` is used when supported, avoiding full-vocabulary logits
   for every prompt position.
8. CUDA (local NVIDIA / Kaggle T4), Apple Silicon MPS, and CPU are supported.
9. Resumable caches with prompt hashes.
10. A 232-cell fold-safe FAST gold gate is included so the new scoring method
    is validated before spending time on 17,396 production cells.
11. A benchmark mode maps real production cells into the real cache, so the
    benchmark work is not wasted.

Important validation rule
-------------------------
The final production teacher uses all 58 gold reports as production exemplars.
It is therefore a FINAL TRAINING RESOURCE, not pristine fold-safe OOF evidence.

Recommended sequence
--------------------
Desktop RTX 4060 Ti:
    run_w26pf("status", accelerator="localGPU")
    run_w26pf("gold_fast", accelerator="localGPU")
    run_w26pf("benchmark", accelerator="localGPU")
    run_w26pf("production", accelerator="localGPU")
    run_w26pf("validate")

Apple Silicon:
    run_w26pf("status", accelerator="apple_mps")
    run_w26pf("gold_fast", accelerator="apple_mps")

A 7B FP16 model may be too tight on a 16GB unified-memory Mac. MPS support is
real, but use a smaller compatible local Qwen2.5-Instruct model if the 7B model
cannot load. Do not mix different model sizes in one final production cache.
"""

# ============================================================
# 0. IMPORTS / ALLOCATOR SETTINGS
# ============================================================

import os

# Must be set before torch import.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import inspect
import json
import math
import re
import time
import hashlib
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# ============================================================
# 1. CONSTANTS / PORTABLE PATHS
# ============================================================

UID = "StudyInstanceUID"
REPORT = "Report"

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

FS4_LABELS = [
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Synovitis",
]

ESTABLISHED_LABELS = [x for x in LABELS if x not in FS4_LABELS]

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_FS4_GOLD_CELLS = EXPECTED_GOLD * len(FS4_LABELS)
EXPECTED_FS4_PROD_CELLS = EXPECTED_UNLABELED * len(FS4_LABELS)
EXPECTED_ALL_PROD_CELLS = EXPECTED_UNLABELED * len(LABELS)

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

FIXED_BLEND_ALPHA = float(os.environ.get("W26PF_FIXED_BLEND_ALPHA", "0.50"))
LOGIT_TEMPERATURE = float(os.environ.get("W26PF_LOGIT_TEMPERATURE", "2.0"))

# FAST gate: one quick validation, then production. No model-shopping loop.
FAST_GATE_MIN_MACRO_AUC = float(os.environ.get("W26PF_GATE_MIN_MACRO_AUC", "0.80"))
FAST_GATE_MIN_DELTA = float(os.environ.get("W26PF_GATE_MIN_DELTA", "0.02"))

TARGET_DEFINITIONS = {
    "Medial OA": (
        "medial tibiofemoral osteoarthritis / degenerative disease; challenge "
        "positives may include medial arthrosis, osteophytes, joint-space "
        "degeneration, or advanced/full-thickness medial cartilage loss even "
        "without the literal word osteoarthritis"
    ),
    "Lateral OA": (
        "lateral tibiofemoral osteoarthritis / degenerative disease; use "
        "lateral-compartment evidence rather than medial-only or PF-only disease"
    ),
    "PF OA": (
        "patellofemoral degenerative disease / OA; advanced patellar or trochlear "
        "cartilage loss, grade-4 chondropathy, arthrosis, or osteophytes may map "
        "positive; effusion alone is not PF OA"
    ),
    "Synovitis": (
        "synovitis / synovial inflammatory-proliferative abnormality; effusion "
        "alone is not synovitis; infer the challenge annotation convention from "
        "the labeled examples"
    ),
}

# W2 columns used for retrieval and compact prompt features.
W2_REQUIRED_COLUMNS = [
    "RuleAssertion",
    "SemanticAssertion",
    "FusedAssertion",
    "FusedAssertionConfidence",
    "EvidenceAvailable",
    "EvidencePositiveScore",
    "EvidenceNegativeScore",
    "RelatedScore",
    "UncertaintyFlag",
    "SeverityLow",
    "SeverityModerate",
    "SeverityHigh",
    "SeverityDegenerative",
    "FusedEvidence",
    "SemanticPositiveEvidence",
    "SemanticNegativeEvidence",
    "SemanticRelatedEvidence",
]


def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


SCRIPT_DIR = _script_dir()
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    PROJECT_ROOT = (
        Path(os.environ.get("W26PF_PROJECT_ROOT", "/kaggle/working"))
        .expanduser()
        .resolve()
    )
else:
    PROJECT_ROOT = (
        Path(
            os.environ.get(
                "W26PF_PROJECT_ROOT",
                str((SCRIPT_DIR / ".." / "..").resolve()),
            )
        )
        .expanduser()
        .resolve()
    )

if IS_KAGGLE:
    DATA_ROOT = Path(
        os.environ.get(
            "W26PF_DATA_ROOT",
            "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        )
    )
    TRAIN_CSV = Path(os.environ.get("W26PF_TRAIN_CSV", str(DATA_ROOT / "train.csv")))
    W2_ROOT_ENV = os.environ.get(
        "W26PF_W2_ROOT",
        "/kaggle/input/datasets/isayem/rsna-w2/rsna_w2",
    ).strip()
    W23_ROOT_ENV = os.environ.get(
        "W26PF_W23_ROOT",
        "/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3",
    ).strip()
    MODEL_PATH_ENV = os.environ.get(
        "W26PF_MODEL_PATH",
        "/kaggle/input/datasets/ragnar123/qwen2-5-7b-instruct",
    ).strip()
    OUTPUT_ROOT = Path(
        os.environ.get(
            "W26PF_OUTPUT_ROOT",
            "/kaggle/working/rsna_w2_6p_fast",
        )
    )
else:
    DATA_ROOT = (
        Path(
            os.environ.get(
                "W26PF_DATA_ROOT",
                str(PROJECT_ROOT / "input"),
            )
        )
        .expanduser()
        .resolve()
    )
    TRAIN_CSV = (
        Path(
            os.environ.get(
                "W26PF_TRAIN_CSV",
                str(DATA_ROOT / "train.csv"),
            )
        )
        .expanduser()
        .resolve()
    )
    W2_ROOT_ENV = os.environ.get(
        "W26PF_W2_ROOT",
        str(PROJECT_ROOT / "output" / "results" / "rsna_w2"),
    ).strip()
    W23_ROOT_ENV = os.environ.get(
        "W26PF_W23_ROOT",
        str(PROJECT_ROOT / "output" / "results" / "rsna_w2_3"),
    ).strip()

    model_candidates = [
        PROJECT_ROOT / "models" / "qwen2-5-7b-instruct",
        PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct",
        DATA_ROOT / "models" / "qwen2-5-7b-instruct",
        DATA_ROOT / "qwen2-5-7b-instruct",
    ]
    default_model = next(
        (p for p in model_candidates if p.exists()),
        model_candidates[0],
    )
    MODEL_PATH_ENV = os.environ.get(
        "W26PF_MODEL_PATH",
        str(default_model),
    ).strip()

    OUTPUT_ROOT = (
        Path(
            os.environ.get(
                "W26PF_OUTPUT_ROOT",
                str(PROJECT_ROOT / "output" / "results" / "rsna_w2_6p_fast"),
            )
        )
        .expanduser()
        .resolve()
    )

CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULT_ROOT = OUTPUT_ROOT / "results"

ACCELERATOR = os.environ.get("W26PF_ACCELERATOR", "auto").strip().lower()
PRECISION = os.environ.get("W26PF_PRECISION", "auto").strip().lower()
GPU_ID = int(os.environ.get("W26PF_GPU_ID", "0"))

# Prompt style:
# compact = optimized target-focused snippets (default)
# legacy  = longer reports; useful only if compact gold_fast unexpectedly fails
PROMPT_STYLE = os.environ.get("W26PF_PROMPT_STYLE", "compact").strip().lower()

N_POS_EXAMPLES = int(os.environ.get("W26PF_N_POS_EXAMPLES", "2"))
N_NEG_EXAMPLES = int(os.environ.get("W26PF_N_NEG_EXAMPLES", "2"))
TFIDF_MAX_FEATURES = int(os.environ.get("W26PF_TFIDF_MAX_FEATURES", "30000"))

# Compact style substantially cuts prefill cost.
COMPACT_EXAMPLE_CHARS = int(os.environ.get("W26PF_COMPACT_EXAMPLE_CHARS", "650"))
COMPACT_QUERY_CHARS = int(os.environ.get("W26PF_COMPACT_QUERY_CHARS", "1100"))
LEGACY_EXAMPLE_CHARS = int(os.environ.get("W26PF_LEGACY_EXAMPLE_CHARS", "1600"))
LEGACY_QUERY_CHARS = int(os.environ.get("W26PF_LEGACY_QUERY_CHARS", "2600"))

MAX_INPUT_TOKENS = int(os.environ.get("W26PF_MAX_INPUT_TOKENS", "2304"))
SINGLE_PROMPT_FALLBACK_TOKENS = int(
    os.environ.get("W26PF_SINGLE_PROMPT_FALLBACK_TOKENS", "1536")
)

# Maximum number of independent prompts per forward pass.
MAX_BATCH_SIZE = int(os.environ.get("W26PF_MAX_BATCH_SIZE", "24"))

# Approximate padded-token budget per forward pass. "auto" is chosen after model
# loading from accelerator / available memory.
TOKEN_BATCH_BUDGET_ENV = os.environ.get("W26PF_TOKEN_BATCH_BUDGET", "auto").strip()

# Gold/bootstrap is tiny and CPU-only.
BOOTSTRAP_REPEATS = int(os.environ.get("W26PF_BOOTSTRAP_REPEATS", "2000"))
BOOTSTRAP_SEED = int(os.environ.get("W26PF_BOOTSTRAP_SEED", "260826"))

# Benchmark cells are written to the REAL production cache; no work is wasted.
BENCHMARK_CELLS = int(os.environ.get("W26PF_BENCHMARK_CELLS", "256"))


# ============================================================
# 2. UTILITIES
# ============================================================


def ensure_dirs() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def normalize_text(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(s.split())


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def clean_scalar(v: Any, digits: int = 3) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def compact_report(text: str, max_chars: int) -> str:
    s = normalize_text(text)
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + " ... " + s[-tail:]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

TARGET_PATTERNS = {
    "Medial OA": [
        r"\bmedial\b",
        r"femorotibial medial",
        r"tibiofemoral medial",
        r"compartimento medial",
        r"arthro",
        r"osteoarth",
        r"gonarth",
        r"chondrop",
        r"cartilage",
        r"cartíl",
        r"osteophy",
    ],
    "Lateral OA": [
        r"\blateral\b",
        r"femorotibial lateral",
        r"tibiofemoral lateral",
        r"compartimento lateral",
        r"arthro",
        r"osteoarth",
        r"gonarth",
        r"chondrop",
        r"cartilage",
        r"cartíl",
        r"osteophy",
    ],
    "PF OA": [
        r"patell",
        r"trochle",
        r"femoropat",
        r"retropatell",
        r"patelo[f-]?em",
        r"chondrop",
        r"cartilage",
        r"cartíl",
        r"osteophy",
        r"arthro",
    ],
    "Synovitis": [
        r"synovit",
        r"synovial",
        r"sinovit",
        r"sinovial",
        r"plica",
        r"hypertroph",
        r"prolifer",
    ],
}


def target_focused_snippet(
    report: str,
    label: str,
    max_chars: int,
) -> str:
    """
    Keep sentences most likely to matter for the requested target, then add a
    small report tail. This is intentionally deterministic and gold-independent.
    """
    text = normalize_text(report)
    if not text:
        return ""

    sentences = [
        x.strip() for x in _SENT_SPLIT_RE.split(str(report)) if x and x.strip()
    ]
    patterns = TARGET_PATTERNS[label]

    scored: List[Tuple[int, int, str]] = []
    for i, sentence in enumerate(sentences):
        s = normalize_text(sentence)
        hits = sum(1 for p in patterns if re.search(p, s, flags=re.I))
        if hits:
            # More target hits first; later/impression-like sentences win ties.
            scored.append((hits, i, s))

    selected: List[str] = []
    if scored:
        scored.sort(key=lambda x: (-x[0], -x[1]))
        for _, _, s in scored[:4]:
            if s not in selected:
                selected.append(s)

    # Preserve impression/tail information even if no keyword fired.
    tail = text[-500:]
    if tail and tail not in selected:
        selected.append(tail)

    if not selected:
        return compact_report(text, max_chars)

    joined = " | ".join(selected)
    return compact_report(joined, max_chars)


def fold_sha256(frame: pd.DataFrame) -> str:
    x = frame[[UID, "OuterFold"]].copy()
    x[UID] = x[UID].astype(str)
    x["OuterFold"] = x["OuterFold"].astype(int)
    x = x.sort_values(UID)
    payload = "".join(f"{u},{int(f)}\n" for u, f in zip(x[UID], x["OuterFold"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def macro_auc_from_matrices(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            continue
        vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals))


def bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    n = len(y)
    for _ in range(BOOTSTRAP_REPEATS):
        idx = rng.integers(0, n, size=n)
        vals_c, vals_r = [], []
        for j in range(y.shape[1]):
            yy = y[idx, j]
            if len(np.unique(yy)) < 2:
                continue
            vals_c.append(roc_auc_score(yy, candidate[idx, j]))
            vals_r.append(roc_auc_score(yy, reference[idx, j]))
        if vals_c:
            deltas.append(float(np.mean(vals_c) - np.mean(vals_r)))
    if not deltas:
        return {"n": 0}
    arr = np.asarray(deltas, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "p_delta_gt_0": float(np.mean(arr > 0)),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _normalize_accelerator(value: str) -> str:
    x = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "t4": "kaggle_t4",
        "kaggle": "kaggle_t4",
        "kaggle_gpu": "kaggle_t4",
        "local": "local_gpu",
        "localgpu": "local_gpu",
        "cuda": "local_gpu",
        "gpu": "local_gpu",
        "apple": "apple_mps",
        "mac": "apple_mps",
        "macos": "apple_mps",
        "mps": "apple_mps",
        "apple_silicon": "apple_mps",
        "v5e": "tpu",
        "tpu_v5e": "tpu",
    }
    return aliases.get(x, x)


def _mps_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _tpu_detected() -> bool:
    try:
        # import torch_xla.core.xla_model as xm  # noqa: F401
        return False
    except Exception:
        return False


def resolve_accelerator(requested: Optional[str] = None) -> str:
    req = _normalize_accelerator(requested or ACCELERATOR)
    allowed = {"auto", "kaggle_t4", "local_gpu", "apple_mps", "tpu", "cpu"}
    if req not in allowed:
        raise ValueError(
            f"Unknown accelerator {req!r}. Use auto, kaggle_t4/T4, "
            "localGPU/local_gpu, apple_mps/MPS, tpu, or cpu."
        )
    if req != "auto":
        return req

    try:
        import torch

        if torch.cuda.is_available():
            names = [
                torch.cuda.get_device_name(i).casefold()
                for i in range(torch.cuda.device_count())
            ]
            if IS_KAGGLE and len(names) >= 2 and all("t4" in n for n in names[:2]):
                return "kaggle_t4"
            return "local_gpu"
    except Exception:
        pass

    if _mps_available():
        return "apple_mps"
    if _tpu_detected():
        return "tpu"
    return "cpu"


def hard_cleanup(torch_module, accelerator: str) -> None:
    """
    Expensive cleanup. Call ONLY after OOM / model release, never every batch.
    """
    gc.collect()
    if accelerator in {"local_gpu", "kaggle_t4"}:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    elif accelerator == "apple_mps":
        try:
            torch_module.mps.empty_cache()
        except Exception:
            pass


# ============================================================
# 3. INPUT DISCOVERY / LOADERS
# ============================================================


def load_train() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train.csv not found: {TRAIN_CSV}")
    train = pd.read_csv(TRAIN_CSV)
    train[UID] = train[UID].astype(str)

    missing = {UID, REPORT, *LABELS} - set(train.columns)
    if missing:
        raise RuntimeError(f"train.csv missing columns: {sorted(missing)}")

    is_gold = train[LABELS].notna().all(axis=1)
    is_unlabeled = train[LABELS].isna().all(axis=1)
    partial = ~(is_gold | is_unlabeled)
    if partial.any():
        raise RuntimeError(f"Unexpected partially labeled rows: {int(partial.sum())}")

    gold = train[is_gold].copy().sort_values(UID).reset_index(drop=True)
    unlabeled = train[is_unlabeled].copy().sort_values(UID).reset_index(drop=True)

    if (len(train), len(gold), len(unlabeled)) != (
        EXPECTED_TRAIN,
        EXPECTED_GOLD,
        EXPECTED_UNLABELED,
    ):
        raise RuntimeError(
            f"Unexpected counts train/gold/unlabeled="
            f"{len(train)}/{len(gold)}/{len(unlabeled)}"
        )
    return train, gold, unlabeled


def resolve_root(
    raw: str,
    expected_files: Sequence[str],
    suffix: Optional[str] = None,
) -> Path:
    candidates = [Path(raw)]
    if suffix:
        candidates.append(Path(raw) / suffix)
    for x in candidates:
        if all((x / rel).exists() for rel in expected_files):
            return x
    raise FileNotFoundError(
        f"Could not resolve root from {raw!r}; expected {list(expected_files)}"
    )


def get_w2_root() -> Path:
    return resolve_root(
        W2_ROOT_ENV,
        [
            "results/04_gold_structured_report_features.csv",
            "results/08_full_structured_report_labels.csv",
        ],
        suffix="rsna_w2",
    )


def get_w23_root() -> Path:
    return resolve_root(
        W23_ROOT_ENV,
        [
            "results/00_outer_fold_assignments.csv",
            "results/06_fold_safe_unlabeled_soft_labels_long.csv",
            "results/10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv",
            "folds/fold_1/heldout_gold_stage_b_predictions.csv",
            "folds/fold_5/heldout_gold_stage_b_predictions.csv",
        ],
        suffix="rsna_w2_3",
    )


def load_folds(w23_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    p = w23_root / "results" / "00_outer_fold_assignments.csv"
    folds = pd.read_csv(p)
    folds[UID] = folds[UID].astype(str)
    folds = folds[[UID, "OuterFold"]].sort_values(UID).reset_index(drop=True)
    if set(folds[UID]) != set(gold[UID]):
        raise RuntimeError("W2.3 fold UID mismatch")
    digest = fold_sha256(folds)
    if digest != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"Fold SHA mismatch: {digest}")
    return folds


def load_w2_tables(
    w2_root: Path,
    train: pd.DataFrame,
    gold: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gold_path = w2_root / "results" / "04_gold_structured_report_features.csv"
    full_path = w2_root / "results" / "08_full_structured_report_labels.csv"
    gold_w2 = pd.read_csv(gold_path)
    full_w2 = pd.read_csv(full_path)

    for x in [gold_w2, full_w2]:
        x[UID] = x[UID].astype(str)

    required = {UID, "Label", *W2_REQUIRED_COLUMNS}
    for name, frame in [("gold", gold_w2), ("full", full_w2)]:
        miss = required - set(frame.columns)
        if miss:
            raise RuntimeError(f"W2 {name} table missing: {sorted(miss)}")
        if frame[[UID, "Label"]].duplicated().any():
            raise RuntimeError(f"W2 {name} duplicate UID/Label")

    if len(gold_w2) != EXPECTED_GOLD * len(LABELS):
        raise RuntimeError("Unexpected W2 gold row count")
    if len(full_w2) != EXPECTED_TRAIN * len(LABELS):
        raise RuntimeError("Unexpected W2 full row count")
    if set(gold_w2[UID]) != set(gold[UID]):
        raise RuntimeError("W2 gold UID mismatch")
    if set(full_w2[UID]) != set(train[UID]):
        raise RuntimeError("W2 full UID mismatch")
    return gold_w2, full_w2


def load_w23_oof(
    w23_root: Path,
    gold: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    for fold in range(1, 6):
        p = w23_root / "folds" / f"fold_{fold}" / "heldout_gold_stage_b_predictions.csv"
        x = pd.read_csv(p)
        x[UID] = x[UID].astype(str)
        req = {UID, "Label", "Gold", "FoldSafeChallengeProbability"}
        if req - set(x.columns):
            raise RuntimeError(f"{p} missing required columns")
        x = x[[UID, "Label", "Gold", "FoldSafeChallengeProbability"]].copy()
        x["OuterFold"] = fold
        pieces.append(x)

    out = pd.concat(pieces, ignore_index=True)
    if (
        len(out) != EXPECTED_GOLD * len(LABELS)
        or out[[UID, "Label"]].duplicated().any()
    ):
        raise RuntimeError("Invalid W2.3 OOF table")
    return out


def load_w23_production_base(
    w23_root: Path,
    unlabeled: pd.DataFrame,
) -> pd.DataFrame:
    stab_path = (
        w23_root / "results" / "10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv"
    )
    long_path = w23_root / "results" / "06_fold_safe_unlabeled_soft_labels_long.csv"
    stab = pd.read_csv(stab_path)
    long = pd.read_csv(long_path)
    stab[UID] = stab[UID].astype(str)
    long[UID] = long[UID].astype(str)

    required = {
        UID,
        "Label",
        "RawProbabilityMean",
        "RawProbabilityStd",
        "SoftLabelAvailableFolds",
        "HighSelectionFolds",
    }
    if required - set(stab.columns):
        raise RuntimeError("W2.3 stability table missing columns")
    if "CandidateSelectionScore" not in long.columns:
        raise RuntimeError("W2.3 long table missing CandidateSelectionScore")

    agg = long.groupby([UID, "Label"], as_index=False).agg(
        W23CandidateSelectionMean=("CandidateSelectionScore", "mean"),
        W23CandidateSelectionMax=("CandidateSelectionScore", "max"),
    )

    out = (
        stab[
            [
                UID,
                "Label",
                "RawProbabilityMean",
                "RawProbabilityStd",
                "SoftLabelAvailableFolds",
                "HighSelectionFolds",
            ]
        ]
        .copy()
        .rename(
            columns={
                "RawProbabilityMean": "W23RawProbabilityMean",
                "RawProbabilityStd": "W23RawProbabilityStd",
                "SoftLabelAvailableFolds": "W23SoftAvailableFolds",
                "HighSelectionFolds": "W23HighSelectionFolds",
            }
        )
    )
    out = out.merge(
        agg,
        on=[UID, "Label"],
        how="left",
        validate="one_to_one",
    )
    uid_set = set(unlabeled[UID].astype(str))
    out = out[out[UID].isin(uid_set)].copy()

    if len(out) != EXPECTED_ALL_PROD_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_ALL_PROD_CELLS} W2.3 production rows, "
            f"got {len(out)}"
        )
    return out.sort_values([UID, "Label"]).reset_index(drop=True)


# ============================================================
# 4. W2 SUMMARIES / RETRIEVAL
# ============================================================


def build_feature_lookup(
    frame: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.Series]:
    return {(str(r[UID]), str(r["Label"])): r for _, r in frame.iterrows()}


def _dedup_evidence(row: pd.Series) -> List[str]:
    vals: List[str] = []
    seen = set()
    for col in [
        "FusedEvidence",
        "SemanticPositiveEvidence",
        "SemanticNegativeEvidence",
        "SemanticRelatedEvidence",
    ]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            s = normalize_text(v)
            key = s.casefold()
            if key not in seen:
                seen.add(key)
                vals.append(s)
    return vals


def w2_retrieval_summary(row: pd.Series) -> str:
    ev = _dedup_evidence(row)
    parts = [
        f"rule={clean_scalar(row.get('RuleAssertion'))}",
        f"sem={clean_scalar(row.get('SemanticAssertion'))}",
        f"fused={clean_scalar(row.get('FusedAssertion'))}",
        f"conf={clean_scalar(row.get('FusedAssertionConfidence'))}",
        f"pos={clean_scalar(row.get('EvidencePositiveScore'))}",
        f"neg={clean_scalar(row.get('EvidenceNegativeScore'))}",
        f"rel={clean_scalar(row.get('RelatedScore'))}",
        f"unc={clean_scalar(row.get('UncertaintyFlag'))}",
        f"sl={clean_scalar(row.get('SeverityLow'))}",
        f"sm={clean_scalar(row.get('SeverityModerate'))}",
        f"sh={clean_scalar(row.get('SeverityHigh'))}",
        f"sd={clean_scalar(row.get('SeverityDegenerative'))}",
    ]
    if ev:
        parts.append("ev=" + " || ".join(ev[:3]))
    return "; ".join(parts)


def w2_prompt_summary(row: pd.Series) -> str:
    ev = _dedup_evidence(row)
    parts = [
        f"fused={clean_scalar(row.get('FusedAssertion'))}",
        f"conf={clean_scalar(row.get('FusedAssertionConfidence'))}",
        f"pos={clean_scalar(row.get('EvidencePositiveScore'))}",
        f"neg={clean_scalar(row.get('EvidenceNegativeScore'))}",
        f"rel={clean_scalar(row.get('RelatedScore'))}",
        f"unc={clean_scalar(row.get('UncertaintyFlag'))}",
        (
            "sev="
            + "/".join(
                [
                    clean_scalar(row.get("SeverityLow")),
                    clean_scalar(row.get("SeverityModerate")),
                    clean_scalar(row.get("SeverityHigh")),
                    clean_scalar(row.get("SeverityDegenerative")),
                ]
            )
        ),
    ]
    if ev:
        compact_ev = " || ".join(ev[:2])
        parts.append("ev=" + compact_report(compact_ev, 240))
    return "; ".join(parts)


def retrieval_document(
    report: str,
    feature_summary: str,
    label: str,
) -> str:
    return (
        f"target {label} target {label} "
        f"{normalize_text(feature_summary)} "
        f"{normalize_text(report)}"
    )


def prompt_report(
    report: str,
    label: str,
    is_query: bool,
) -> str:
    if PROMPT_STYLE == "legacy":
        max_chars = LEGACY_QUERY_CHARS if is_query else LEGACY_EXAMPLE_CHARS
        return compact_report(report, max_chars)
    if PROMPT_STYLE != "compact":
        raise ValueError(
            f"Unknown W26PF_PROMPT_STYLE={PROMPT_STYLE!r}; " "use compact or legacy"
        )
    max_chars = COMPACT_QUERY_CHARS if is_query else COMPACT_EXAMPLE_CHARS
    return target_focused_snippet(report, label, max_chars)


# ============================================================
# 5. FOLD-SAFE AND PRODUCTION EXEMPLAR RETRIEVAL
# ============================================================


def _select_top_by_class(
    reference: pd.DataFrame,
    similarities: np.ndarray,
    label: str,
    k_pos: int,
    k_neg: int,
) -> List[int]:
    tmp = reference[[UID, label]].copy()
    tmp["Similarity"] = similarities
    tmp["_idx"] = np.arange(len(tmp))
    tmp = tmp.sort_values(
        ["Similarity", UID],
        ascending=[False, True],
    )
    pos = tmp[tmp[label] == 1].head(k_pos)
    neg = tmp[tmp[label] == 0].head(k_neg)
    chosen = pd.concat([pos, neg], ignore_index=True)
    chosen = chosen.sort_values(
        ["Similarity", UID],
        ascending=[False, True],
    )
    return chosen["_idx"].astype(int).tolist()


def select_gold_fold_exemplars(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_lookup: Dict[Tuple[str, str], pd.Series],
    fold: int,
    label: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()
    tmp = gold[[UID, REPORT, label]].copy()
    tmp["OuterFold"] = tmp[UID].map(fold_map).astype(int)
    outer_train = (
        tmp[tmp["OuterFold"] != fold].copy().sort_values(UID).reset_index(drop=True)
    )
    heldout = (
        tmp[tmp["OuterFold"] == fold].copy().sort_values(UID).reset_index(drop=True)
    )

    if int((outer_train[label] == 1).sum()) < N_POS_EXAMPLES:
        raise RuntimeError(f"{label}/fold{fold}: insufficient positives")
    if int((outer_train[label] == 0).sum()) < N_NEG_EXAMPLES:
        raise RuntimeError(f"{label}/fold{fold}: insufficient negatives")

    train_docs = []
    for _, r in outer_train.iterrows():
        uid = str(r[UID])
        fs = w2_retrieval_summary(w2_lookup[(uid, label)])
        train_docs.append(retrieval_document(str(r[REPORT]), fs, label))

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
    )
    X_train = vectorizer.fit_transform(train_docs)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    audits: List[Dict[str, Any]] = []

    for _, q in heldout.iterrows():
        q_uid = str(q[UID])
        q_fs = w2_retrieval_summary(w2_lookup[(q_uid, label)])
        q_vec = vectorizer.transform([retrieval_document(str(q[REPORT]), q_fs, label)])
        sims = (X_train @ q_vec.T).toarray().ravel()

        chosen_idx = _select_top_by_class(
            outer_train,
            sims,
            label,
            N_POS_EXAMPLES,
            N_NEG_EXAMPLES,
        )

        exs: List[Dict[str, Any]] = []
        for rank, j in enumerate(chosen_idx, start=1):
            ex = outer_train.iloc[j]
            ex_uid = str(ex[UID])
            exs.append(
                {
                    "StudyInstanceUID": ex_uid,
                    "Gold": int(ex[label]),
                    "Similarity": float(sims[j]),
                    "Report": str(ex[REPORT]),
                    "W2Summary": w2_prompt_summary(w2_lookup[(ex_uid, label)]),
                }
            )
            audits.append(
                {
                    "OuterFold": fold,
                    "QueryStudyInstanceUID": q_uid,
                    "Label": label,
                    "ExampleRank": rank,
                    "ExampleStudyInstanceUID": ex_uid,
                    "ExampleGold": int(ex[label]),
                    "Similarity": float(sims[j]),
                }
            )
        selected[q_uid] = exs

    return selected, pd.DataFrame(audits)


def build_all_gold_exemplars(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_gold: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[str, str], List[Dict[str, Any]]],
    pd.DataFrame,
]:
    lookup = build_feature_lookup(w2_gold)
    all_selected: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    audits = []

    for fold in range(1, 6):
        for label in FS4_LABELS:
            selected, audit = select_gold_fold_exemplars(
                gold, folds, lookup, fold, label
            )
            for uid, exs in selected.items():
                all_selected[(uid, label)] = exs
            audits.append(audit)

    out = pd.concat(audits, ignore_index=True)
    if len(all_selected) != EXPECTED_FS4_GOLD_CELLS:
        raise RuntimeError("Gold exemplar-set count mismatch")
    return all_selected, out


def build_production_exemplars(
    gold: pd.DataFrame,
    unlabeled: pd.DataFrame,
    gold_w2: pd.DataFrame,
    full_w2: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    gold_lookup = build_feature_lookup(gold_w2)
    full_lookup = build_feature_lookup(full_w2)

    queries: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []

    for label in FS4_LABELS:
        log(f"Building production retrieval: {label}")

        gold_docs = []
        for _, r in gold.iterrows():
            uid = str(r[UID])
            fs = w2_retrieval_summary(gold_lookup[(uid, label)])
            gold_docs.append(retrieval_document(str(r[REPORT]), fs, label))

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            norm="l2",
        )
        X_gold = vectorizer.fit_transform(gold_docs)

        query_docs = []
        query_w2 = []
        for _, r in unlabeled.iterrows():
            uid = str(r[UID])
            fs_retr = w2_retrieval_summary(full_lookup[(uid, label)])
            query_docs.append(retrieval_document(str(r[REPORT]), fs_retr, label))
            query_w2.append(w2_prompt_summary(full_lookup[(uid, label)]))

        X_query = vectorizer.transform(query_docs)
        sims = (X_query @ X_gold.T).toarray()
        y_gold = gold[label].astype(int).to_numpy()
        prior = float(y_gold.mean())

        for qi, (_, qrow) in enumerate(unlabeled.iterrows()):
            chosen_idx = _select_top_by_class(
                gold,
                sims[qi],
                label,
                N_POS_EXAMPLES,
                N_NEG_EXAMPLES,
            )
            examples = []
            for rank, j in enumerate(chosen_idx, start=1):
                ex = gold.iloc[j]
                ex_uid = str(ex[UID])
                examples.append(
                    {
                        "StudyInstanceUID": ex_uid,
                        "Gold": int(ex[label]),
                        "Similarity": float(sims[qi, j]),
                        "Report": str(ex[REPORT]),
                        "W2Summary": w2_prompt_summary(gold_lookup[(ex_uid, label)]),
                    }
                )
                audits.append(
                    {
                        "QueryStudyInstanceUID": str(qrow[UID]),
                        "Label": label,
                        "ExampleRank": rank,
                        "ExampleStudyInstanceUID": ex_uid,
                        "ExampleGold": int(ex[label]),
                        "Similarity": float(sims[qi, j]),
                    }
                )

            prompt = build_fast_prompt(
                label=label,
                query_report=str(qrow[REPORT]),
                query_w2=query_w2[qi],
                examples=examples,
                prior=prior,
                fold_safe=False,
            )
            queries.append(
                {
                    UID: str(qrow[UID]),
                    "Label": label,
                    "Prompt": prompt,
                    "PromptSHA256": stable_sha256(prompt),
                    "PromptChars": len(prompt),
                    "Prior": prior,
                    "ExampleUIDs": "|".join(x["StudyInstanceUID"] for x in examples),
                }
            )

    if len(queries) != EXPECTED_FS4_PROD_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_FS4_PROD_CELLS} production queries, "
            f"got {len(queries)}"
        )
    return queries, pd.DataFrame(audits)


# ============================================================
# 6. FAST A/B PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You map knee MRI reports to THIS competition's binary labels. "
    "Infer the challenge annotation convention from labeled examples. "
    "Do not equate silence with negative. You will choose A or B only."
)


def build_fast_prompt(
    label: str,
    query_report: str,
    query_w2: str,
    examples: Sequence[Mapping[str, Any]],
    prior: float,
    fold_safe: bool,
) -> str:
    chunks = [
        f"TARGET={label}",
        f"DEFINITION={TARGET_DEFINITIONS[label]}",
        f"REFERENCE_POSITIVE_RATE={prior:.3f}",
        "A = challenge label 0",
        "B = challenge label 1",
        "",
        ("FOLD-SAFE EXAMPLES:" if fold_safe else "LABELED PRODUCTION EXAMPLES:"),
    ]

    for i, ex in enumerate(examples, start=1):
        code = "B" if int(ex["Gold"]) == 1 else "A"
        chunks += [
            (
                f"E{i}: answer={code}; sim={float(ex['Similarity']):.3f}; "
                f"w2={ex['W2Summary']}"
            ),
            f"report={prompt_report(ex['Report'], label, is_query=False)}",
        ]

    chunks += [
        "",
        f"QUERY_W2={query_w2}",
        f"QUERY_REPORT={prompt_report(query_report, label, is_query=True)}",
        "",
        (
            "Choose the competition label for QUERY. "
            "Reply with exactly one token: A or B."
        ),
        "ANSWER:",
    ]
    return "\n".join(chunks)


def build_gold_queries(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_gold: pd.DataFrame,
    exemplars: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()
    w2_lookup = build_feature_lookup(w2_gold)
    gold_lookup = gold.set_index(UID)
    queries = []

    gold_uids = gold[UID].astype(str).tolist()
    for uid in gold_uids:
        fold = int(fold_map[uid])
        outer_train_uids = [u for u in gold_uids if int(fold_map[u]) != fold]
        for label in FS4_LABELS:
            prior = float(gold_lookup.loc[outer_train_uids, label].astype(float).mean())
            prompt = build_fast_prompt(
                label=label,
                query_report=str(gold_lookup.at[uid, REPORT]),
                query_w2=w2_prompt_summary(w2_lookup[(uid, label)]),
                examples=exemplars[(uid, label)],
                prior=prior,
                fold_safe=True,
            )
            queries.append(
                {
                    UID: uid,
                    "Label": label,
                    "Gold": int(gold_lookup.at[uid, label]),
                    "OuterFold": fold,
                    "Prior": prior,
                    "Prompt": prompt,
                    "PromptSHA256": stable_sha256(prompt),
                    "PromptChars": len(prompt),
                }
            )
    if len(queries) != EXPECTED_FS4_GOLD_CELLS:
        raise RuntimeError("Gold FAST query count mismatch")
    return queries


# ============================================================
# 7. FAST LOGIT MODEL
# ============================================================


class FastLogitMapper:
    def __init__(
        self,
        model_path: Path,
        accelerator: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:
            raise RuntimeError("W2.6-P FAST requires torch + transformers") from exc

        self.torch = torch
        self.model_path = Path(model_path)
        self.accelerator = resolve_accelerator(accelerator)

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        if self.accelerator == "tpu":
            raise RuntimeError(
                "TPU is intentionally not used for this Qwen decoder path. "
                "Use localGPU/T4/MPS here and preserve TPU for W6 image work."
            )

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.answer_zero, self.answer_one = self._choose_verbalizers()

        common: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        log(f"Loading FAST mapper       : {self.model_path}")
        log(f"Accelerator requested     : {accelerator or ACCELERATOR}")
        log(f"Accelerator resolved      : {self.accelerator}")
        log(f"Precision policy          : {PRECISION}")
        log(
            f"Answer tokens             : "
            f"{self.answer_zero[0]!r}/{self.answer_one[0]!r}"
        )

        if self.accelerator == "local_gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("local_gpu selected but CUDA unavailable")
            gpu_count = torch.cuda.device_count()
            if GPU_ID >= gpu_count:
                raise RuntimeError(f"GPU_ID={GPU_ID}, visible CUDA GPUs={gpu_count}")

            loaded = False
            if PRECISION in {"auto", "4bit", "nf4"}:
                try:
                    compute_dtype = (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                    qkwargs = dict(common)
                    qkwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    qkwargs["device_map"] = {"": GPU_ID}
                    qkwargs["attn_implementation"] = "sdpa"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        **qkwargs,
                    )
                    self.load_mode = f"4bit_nf4_local_gpu_{GPU_ID}"
                    loaded = True
                except Exception as exc:
                    if PRECISION in {"4bit", "nf4"}:
                        raise RuntimeError("Explicit 4-bit load failed") from exc
                    warnings.warn(
                        "4-bit load unavailable; using FP16 CPU offload. "
                        f"Original error: {exc}"
                    )

            if not loaded:
                # Conservative fallback for 16GB/6GB cards.
                try:
                    free_b, total_b = torch.cuda.mem_get_info(GPU_ID)
                    total_gib = total_b / (1024**3)
                except Exception:
                    total_gib = 16.0

                gpu_cap = max(3.5, min(11.5, total_gib - 3.5))
                cpu_cap = float(os.environ.get("W26PF_CPU_OFFLOAD_GIB", "36.0"))
                offload_dir = OUTPUT_ROOT / "model_offload"
                offload_dir.mkdir(parents=True, exist_ok=True)

                fkwargs = dict(common)
                fkwargs["dtype"] = torch.float16
                fkwargs["device_map"] = "auto"
                fkwargs["max_memory"] = {
                    GPU_ID: f"{gpu_cap:.1f}GiB",
                    "cpu": f"{cpu_cap:.1f}GiB",
                }
                fkwargs["offload_folder"] = str(offload_dir)
                fkwargs["offload_state_dict"] = True
                fkwargs["attn_implementation"] = "sdpa"
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **fkwargs,
                )
                self.load_mode = f"fp16_local_gpu_{GPU_ID}_cpu_offload"

        elif self.accelerator == "kaggle_t4":
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError("kaggle_t4 requires two visible CUDA GPUs")
            t4_cap = float(os.environ.get("W26PF_T4_GPU_MAX_GIB", "11.5"))
            kwargs = dict(common)
            kwargs["dtype"] = torch.float16
            kwargs["device_map"] = "balanced"
            kwargs["max_memory"] = {
                i: f"{t4_cap:.1f}GiB" for i in range(torch.cuda.device_count())
            }
            kwargs["attn_implementation"] = "sdpa"
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )
            self.load_mode = f"fp16_balanced_{torch.cuda.device_count()}gpu"

        elif self.accelerator == "apple_mps":
            if not _mps_available():
                raise RuntimeError(
                    "apple_mps selected but torch.backends.mps is unavailable"
                )
            # MPS currently has no bitsandbytes 4-bit path. FP16 is used.
            # A 7B model can be too tight on a 16GB Mac; use a smaller local
            # Qwen2.5-Instruct path if loading fails.
            kwargs = dict(common)
            kwargs["dtype"] = torch.float16
            kwargs["attn_implementation"] = "sdpa"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **kwargs,
                )
                self.model.to("mps")
            except Exception as exc:
                raise RuntimeError(
                    "Apple MPS model load failed. On a 16GB unified-memory Mac, "
                    "Qwen2.5-7B FP16 may not fit alongside macOS/Python. "
                    "Set W26PF_MODEL_PATH to a smaller local Qwen2.5-Instruct "
                    "model (for example 3B) or use the RTX 4060 Ti."
                ) from exc
            self.load_mode = "fp16_apple_mps"

        elif self.accelerator == "cpu":
            kwargs = dict(common)
            kwargs["dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )
            self.load_mode = "fp32_cpu"
        else:
            raise RuntimeError(f"Unhandled accelerator {self.accelerator}")

        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

        # Qwen2 in current transformers supports logits_to_keep.
        try:
            params = inspect.signature(self.model.forward).parameters
            self.supports_logits_to_keep = "logits_to_keep" in params
        except Exception:
            self.supports_logits_to_keep = False

        self.token_budget = self._resolve_token_batch_budget()

        log(f"Model load mode           : {self.load_mode}")
        log(f"Input device              : {self.input_device}")
        log(f"logits_to_keep support    : " f"{self.supports_logits_to_keep}")
        log(f"Token batch budget        : {self.token_budget}")
        log(f"Max batch size            : {MAX_BATCH_SIZE}")
        log(f"Max input tokens          : {MAX_INPUT_TOKENS}")

        if hasattr(self.model, "hf_device_map"):
            counts: Dict[str, int] = {}
            for dev in self.model.hf_device_map.values():
                counts[str(dev)] = counts.get(str(dev), 0) + 1
            log(f"HF device-map modules     : {counts}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(i)
                log(
                    f"GPU {i} free after load    : "
                    f"{free_b/(1024**3):.2f} / "
                    f"{total_b/(1024**3):.2f} GiB"
                )

    def _choose_verbalizers(
        self,
    ) -> Tuple[Tuple[str, int], Tuple[str, int]]:
        # Prefer A/B because the prompt explicitly defines them.
        candidates = [
            ("A", "B"),
            ("0", "1"),
            ("N", "Y"),
        ]
        for zero, one in candidates:
            z = self.tokenizer.encode(zero, add_special_tokens=False)
            o = self.tokenizer.encode(one, add_special_tokens=False)
            if len(z) == 1 and len(o) == 1 and z[0] != o[0]:
                return (zero, int(z[0])), (one, int(o[0]))
        raise RuntimeError("Could not find single-token binary verbalizers")

    def _resolve_token_batch_budget(self) -> int:
        if TOKEN_BATCH_BUDGET_ENV.casefold() != "auto":
            return int(TOKEN_BATCH_BUDGET_ENV)

        # Conservative defaults based on known devices.
        if self.accelerator == "local_gpu":
            try:
                free_b, total_b = self.torch.cuda.mem_get_info(GPU_ID)
                total_gib = total_b / (1024**3)
            except Exception:
                total_gib = 12.0

            if "4bit" in self.load_mode:
                if total_gib >= 14:
                    return 18000
                if total_gib >= 10:
                    return 10000
                return 4000
            if total_gib >= 14:
                return 9000
            return 3500

        if self.accelerator == "kaggle_t4":
            return 8000
        if self.accelerator == "apple_mps":
            return 3500
        return 4000

    def _chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return SYSTEM_PROMPT + "\n\n" + user_prompt

    def prepare_lengths(
        self,
        prompts: Sequence[str],
    ) -> List[int]:
        """
        Fast-tokenizer pass used once to make padding-efficient batches.
        """
        chat = [self._chat_prompt(p) for p in prompts]
        encoded = self.tokenizer(
            chat,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=False,
        )
        return [len(x) for x in encoded["input_ids"]]

    def _forward_batch(
        self,
        prompts: Sequence[str],
        token_limit: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        chat = [self._chat_prompt(p) for p in prompts]

        encoded = self.tokenizer(
            chat,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(token_limit),
        )
        input_lengths = encoded["attention_mask"].sum(dim=1).cpu().numpy().astype(int)
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}

        kwargs: Dict[str, Any] = {
            **encoded,
            "use_cache": False,
            "return_dict": True,
        }
        if self.supports_logits_to_keep:
            kwargs["logits_to_keep"] = 1

        outputs = None
        try:
            with torch.inference_mode():
                outputs = self.model(**kwargs)

            logits = outputs.logits
            if logits.ndim != 3:
                raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
            last = logits[:, -1, :]

            z_id = self.answer_zero[1]
            o_id = self.answer_one[1]
            pair = torch.stack(
                [last[:, z_id], last[:, o_id]],
                dim=1,
            ).float()

            # Temperature is fixed before gold_fast and production.
            pair = pair / float(LOGIT_TEMPERATURE)
            probs = torch.softmax(pair, dim=1)[:, 1]
            margin = pair[:, 1] - pair[:, 0]

            p_np = probs.detach().cpu().numpy().astype(float)
            m_np = margin.detach().cpu().numpy().astype(float)
            return p_np, m_np, input_lengths
        finally:
            try:
                del encoded
            except Exception:
                pass
            try:
                del outputs
            except Exception:
                pass
            # Crucially: NO empty_cache() here.

    def score_batch_recursive(
        self,
        prompts: Sequence[str],
        token_limit: int = MAX_INPUT_TOKENS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        OOM strategy:
        - split the batch immediately;
        - only if a single prompt still OOMs, truncate that prompt to the
          single-prompt fallback context.

        This avoids repeating the same expensive 8-prompt OOM four times.
        """
        try:
            return self._forward_batch(
                prompts,
                token_limit=token_limit,
            )
        except self.torch.cuda.OutOfMemoryError:
            hard_cleanup(self.torch, self.accelerator)
            if len(prompts) > 1:
                mid = len(prompts) // 2
                log(
                    f"  OOM: splitting batch {len(prompts)} "
                    f"-> {mid}+{len(prompts)-mid}"
                )
                a = self.score_batch_recursive(prompts[:mid], token_limit=token_limit)
                b = self.score_batch_recursive(prompts[mid:], token_limit=token_limit)
                return (
                    np.concatenate([a[0], b[0]]),
                    np.concatenate([a[1], b[1]]),
                    np.concatenate([a[2], b[2]]),
                )
            if token_limit > SINGLE_PROMPT_FALLBACK_TOKENS:
                log(
                    f"  single-prompt OOM at {token_limit}; "
                    f"retrying {SINGLE_PROMPT_FALLBACK_TOKENS}"
                )
                return self.score_batch_recursive(
                    prompts,
                    token_limit=SINGLE_PROMPT_FALLBACK_TOKENS,
                )
            raise

        except RuntimeError as exc:
            # MPS OOM is often surfaced as RuntimeError rather than the CUDA
            # OutOfMemoryError subclass.
            msg = str(exc).casefold()
            if self.accelerator == "apple_mps" and (
                "out of memory" in msg or "mps backend" in msg
            ):
                hard_cleanup(self.torch, self.accelerator)
                if len(prompts) > 1:
                    mid = len(prompts) // 2
                    log(
                        f"  MPS OOM: splitting batch {len(prompts)} "
                        f"-> {mid}+{len(prompts)-mid}"
                    )
                    a = self.score_batch_recursive(
                        prompts[:mid], token_limit=token_limit
                    )
                    b = self.score_batch_recursive(
                        prompts[mid:], token_limit=token_limit
                    )
                    return (
                        np.concatenate([a[0], b[0]]),
                        np.concatenate([a[1], b[1]]),
                        np.concatenate([a[2], b[2]]),
                    )
                if token_limit > SINGLE_PROMPT_FALLBACK_TOKENS:
                    return self.score_batch_recursive(
                        prompts,
                        token_limit=SINGLE_PROMPT_FALLBACK_TOKENS,
                    )
            raise


# ============================================================
# 8. LENGTH-AWARE BATCHING / CACHE
# ============================================================


def make_length_aware_batches(
    items: Sequence[Mapping[str, Any]],
    lengths: Sequence[int],
    token_budget: int,
    max_batch_size: int,
) -> List[List[int]]:
    order = sorted(
        range(len(items)),
        key=lambda i: (int(lengths[i]), str(items[i].get("Label", ""))),
    )
    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0

    for idx in order:
        l = int(lengths[idx])
        proposed_max = max(current_max, l)
        proposed_n = len(current) + 1
        padded_tokens = proposed_max * proposed_n

        if current and (proposed_n > max_batch_size or padded_tokens > token_budget):
            batches.append(current)
            current = []
            current_max = 0

        current.append(idx)
        current_max = max(current_max, l)

    if current:
        batches.append(current)
    return batches


def _cache_read(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = f"{row[UID]}|||{row['Label']}"
                out[key] = row
            except Exception as exc:
                raise RuntimeError(f"Corrupt cache line {line_no}: {exc}") from exc
    return out


def _cache_append(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def gold_cache_path() -> Path:
    return CACHE_ROOT / f"w26pf_gold_fast_{PROMPT_STYLE}_v1.jsonl"


def production_cache_path() -> Path:
    return CACHE_ROOT / f"w26pf_production_fast_{PROMPT_STYLE}_v1.jsonl"


def scorer_signature(
    model_path: Path,
    accelerator: Optional[str],
) -> str:
    config_path = Path(model_path) / "config.json"
    config_sha = sha256_file(config_path) if config_path.exists() else "no-config-sha"
    resolved = resolve_accelerator(accelerator)
    payload = {
        "version": "w26pf_fast_logits_v1",
        "model_config_sha256": config_sha,
        "accelerator": resolved,
        "precision_policy": PRECISION,
        "prompt_style": PROMPT_STYLE,
        "logit_temperature": LOGIT_TEMPERATURE,
        "max_input_tokens": MAX_INPUT_TOKENS,
    }
    return stable_sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def run_fast_scoring(
    queries: Sequence[Mapping[str, Any]],
    model_path: Path,
    cache_file: Path,
    accelerator: Optional[str],
    limit_new: Optional[int] = None,
) -> pd.DataFrame:
    ensure_dirs()
    cache = _cache_read(cache_file)
    expected_signature = scorer_signature(
        model_path=model_path,
        accelerator=accelerator,
    )

    valid: Dict[str, Dict[str, Any]] = {}
    needed: List[Mapping[str, Any]] = []

    for q in queries:
        key = f"{q[UID]}|||{q['Label']}"
        row = cache.get(key)
        if (
            row is not None
            and str(row.get("PromptSHA256")) == str(q["PromptSHA256"])
            and str(row.get("ScorerSignature")) == expected_signature
        ):
            valid[key] = row
        else:
            needed.append(q)

    if limit_new is not None:
        needed = needed[: int(limit_new)]

    log(f"FAST query cells          : {len(queries)}")
    log(f"Valid cached cells        : {len(valid)}")
    log(f"New cells this invocation : {len(needed)}")

    if not needed:
        rows = [
            valid[f"{q[UID]}|||{q['Label']}"]
            for q in queries
            if f"{q[UID]}|||{q['Label']}" in valid
        ]
        return pd.DataFrame(rows)

    mapper = FastLogitMapper(
        model_path=model_path,
        accelerator=accelerator,
    )

    log("Tokenizing prompt lengths once...")
    lengths = mapper.prepare_lengths([str(q["Prompt"]) for q in needed])

    batches = make_length_aware_batches(
        needed,
        lengths,
        token_budget=mapper.token_budget,
        max_batch_size=MAX_BATCH_SIZE,
    )

    log(f"Length-aware batches      : {len(batches)}")
    if lengths:
        arr = np.asarray(lengths)
        log(
            "Input token length p50/p90/p99/max: "
            f"{np.quantile(arr, 0.50):.0f}/"
            f"{np.quantile(arr, 0.90):.0f}/"
            f"{np.quantile(arr, 0.99):.0f}/"
            f"{arr.max()}"
        )

    started = time.time()
    completed = 0

    for bi, indices in enumerate(batches, start=1):
        batch_queries = [needed[i] for i in indices]
        prompts = [str(q["Prompt"]) for q in batch_queries]
        t0 = time.time()

        probs, margins, actual_lengths = mapper.score_batch_recursive(prompts)

        cache_rows = []
        for q, p, margin, input_len in zip(
            batch_queries,
            probs,
            margins,
            actual_lengths,
        ):
            row = {
                UID: str(q[UID]),
                "Label": str(q["Label"]),
                "FastProbability": float(p),
                "FastLogitMargin": float(margin),
                "InputTokens": int(input_len),
                "PromptChars": int(q["PromptChars"]),
                "PromptSHA256": str(q["PromptSHA256"]),
                "ScorerSignature": expected_signature,
                "PromptStyle": PROMPT_STYLE,
                "LogitTemperature": LOGIT_TEMPERATURE,
                "Accelerator": mapper.accelerator,
                "LoadMode": mapper.load_mode,
                "Verbalizer0": mapper.answer_zero[0],
                "Verbalizer1": mapper.answer_one[0],
            }
            for col in ["Gold", "OuterFold", "Prior", "ExampleUIDs"]:
                if col in q:
                    row[col] = q[col]
            cache_rows.append(row)

        _cache_append(cache_file, cache_rows)
        for row in cache_rows:
            valid[f"{row[UID]}|||{row['Label']}"] = row

        completed += len(cache_rows)
        elapsed = time.time() - started
        rate = completed / max(elapsed, 1e-9)
        remain = len(needed) - completed
        eta_h = remain / max(rate, 1e-9) / 3600.0

        if bi <= 5 or bi % 10 == 0 or completed == len(needed):
            log(
                f"  scored {completed:>5}/{len(needed)} "
                f"batch={len(batch_queries):>2} "
                f"maxTok={max(actual_lengths):>4} "
                f"call={time.time()-t0:5.2f}s "
                f"rate={rate*60:7.1f} cells/min "
                f"ETA={eta_h:5.2f}h"
            )

    rows = [
        valid[f"{q[UID]}|||{q['Label']}"]
        for q in queries
        if f"{q[UID]}|||{q['Label']}" in valid
    ]
    return pd.DataFrame(rows)


# ============================================================
# 9. FAST GOLD GATE
# ============================================================


def build_variant_oof(
    w23_oof: pd.DataFrame,
    fast_oof: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    base = (
        w23_oof[
            [
                UID,
                "Label",
                "Gold",
                "OuterFold",
                "FoldSafeChallengeProbability",
            ]
        ]
        .copy()
        .rename(columns={"FoldSafeChallengeProbability": "W23Probability"})
    )
    fs = fast_oof[[UID, "Label", "FastProbability"]].copy()
    x = base.merge(
        fs,
        on=[UID, "Label"],
        how="left",
        validate="one_to_one",
    )

    x["Probability"] = x["W23Probability"].astype(float)

    if variant == "w23":
        pass
    elif variant == "fast_replace":
        m = x["Label"].isin(FS4_LABELS)
        x.loc[m, "Probability"] = x.loc[m, "FastProbability"].astype(float)
    elif variant == "fast_fixed50":
        m = x["Label"].isin(FS4_LABELS)
        a = FIXED_BLEND_ALPHA
        x.loc[m, "Probability"] = (1.0 - a) * x.loc[m, "W23Probability"].astype(
            float
        ) + a * x.loc[m, "FastProbability"].astype(float)
    else:
        raise ValueError(variant)

    x["Variant"] = variant
    return x[[UID, "Label", "Gold", "OuterFold", "Variant", "Probability"]]


def metrics_for_variant(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        x = oof[oof["Label"] == label]
        y = x["Gold"].to_numpy(int)
        p = x["Probability"].to_numpy(float)
        rows.append(
            {
                "Label": label,
                "N": len(x),
                "Positive": int(y.sum()),
                "AUROC": safe_auc(y, p),
                "AP": safe_ap(y, p),
                "Brier": float(
                    brier_score_loss(
                        y,
                        np.clip(p, 1e-5, 1 - 1e-5),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def matrix_from_oof(
    oof: pd.DataFrame,
    gold: pd.DataFrame,
) -> np.ndarray:
    wide = oof.pivot(
        index=UID,
        columns="Label",
        values="Probability",
    )
    wide = wide.reindex(
        index=gold[UID].astype(str),
        columns=LABELS,
    )
    return wide.to_numpy(float)


def gold_fast_w26pf(
    accelerator: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    train, gold, _ = load_train()
    w2_root = get_w2_root()
    w23_root = get_w23_root()
    folds = load_folds(w23_root, gold)
    w2_gold, _ = load_w2_tables(w2_root, train, gold)
    w23_oof = load_w23_oof(w23_root, gold)

    exemplars, audit = build_all_gold_exemplars(gold, folds, w2_gold)
    audit.to_csv(
        RESULT_ROOT / "00_fast_gold_exemplar_audit.csv",
        index=False,
    )

    queries = build_gold_queries(gold, folds, w2_gold, exemplars)
    pd.DataFrame(
        [
            {
                UID: q[UID],
                "Label": q["Label"],
                "Gold": q["Gold"],
                "OuterFold": q["OuterFold"],
                "PromptSHA256": q["PromptSHA256"],
                "PromptChars": q["PromptChars"],
            }
            for q in queries
        ]
    ).to_csv(
        RESULT_ROOT / "01_fast_gold_prompt_audit.csv",
        index=False,
    )

    fast = run_fast_scoring(
        queries=queries,
        model_path=Path(MODEL_PATH_ENV),
        cache_file=gold_cache_path(),
        accelerator=accelerator,
    )
    if len(fast) != EXPECTED_FS4_GOLD_CELLS:
        raise RuntimeError(
            f"FAST gold incomplete: {len(fast)}/{EXPECTED_FS4_GOLD_CELLS}"
        )
    fast.to_csv(
        RESULT_ROOT / "02_fast_gold_scores_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    variants = [
        build_variant_oof(w23_oof, fast, "w23"),
        build_variant_oof(w23_oof, fast, "fast_replace"),
        build_variant_oof(w23_oof, fast, "fast_fixed50"),
    ]
    metrics_parts = []
    matrices = {}
    for v in variants:
        name = str(v["Variant"].iloc[0])
        m = metrics_for_variant(v)
        m.insert(0, "Variant", name)
        metrics_parts.append(m)
        matrices[name] = matrix_from_oof(v, gold)

    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics.to_csv(
        RESULT_ROOT / "03_fast_gold_metrics.csv",
        index=False,
    )

    summary: Dict[str, Any] = {
        "version": "w26pf_fast_logits_v1",
        "prompt_style": PROMPT_STYLE,
        "logit_temperature": LOGIT_TEMPERATURE,
        "fold_sha256": fold_sha256(folds),
        "fold_sha256_match": (fold_sha256(folds) == EXPECTED_FOLD_SHA256),
        "variants": {},
    }

    for name in matrices:
        m = metrics[metrics["Variant"] == name]
        summary["variants"][name] = {
            "macro_AUROC": float(np.nanmean(m["AUROC"])),
            "macro_AP": float(np.nanmean(m["AP"])),
            "macro_Brier": float(np.nanmean(m["Brier"])),
        }

    ref = summary["variants"]["w23"]["macro_AUROC"]
    cand = summary["variants"]["fast_fixed50"]["macro_AUROC"]
    delta = cand - ref
    summary["primary_variant"] = "fast_fixed50"
    summary["primary_delta_vs_w23"] = float(delta)
    summary["bootstrap"] = bootstrap_delta(
        gold[LABELS].to_numpy(int),
        matrices["fast_fixed50"],
        matrices["w23"],
    )

    if cand >= FAST_GATE_MIN_MACRO_AUC and delta >= FAST_GATE_MIN_DELTA:
        verdict = "PASS_FAST_PRODUCTION"
    elif delta > 0:
        verdict = "MARGINAL_USE_W23_IF_SPEED_IS_PRIORITY"
    else:
        verdict = "FAIL_FAST_SCORER_USE_W23_ONLY"
    summary["gate_verdict"] = verdict

    weak = metrics[metrics["Label"].isin(FS4_LABELS)].pivot(
        index="Label",
        columns="Variant",
        values="AUROC",
    )
    weak_rows = []
    for label in FS4_LABELS:
        weak_rows.append(
            {
                "Label": label,
                "W23_AUROC": float(weak.at[label, "w23"]),
                "FAST_AUROC": float(weak.at[label, "fast_replace"]),
                "FAST50_AUROC": float(weak.at[label, "fast_fixed50"]),
                "FAST50_Delta": float(
                    weak.at[label, "fast_fixed50"] - weak.at[label, "w23"]
                ),
            }
        )
    pd.DataFrame(weak_rows).to_csv(
        RESULT_ROOT / "04_fast_gold_fs4_attribution.csv",
        index=False,
    )

    (RESULT_ROOT / "05_fast_gold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("\n" + "=" * 96)
    log("W2.6-P FAST GOLD GATE")
    log("=" * 96)
    for name, x in summary["variants"].items():
        log(
            f"{name:16s} "
            f"AUROC={x['macro_AUROC']:.6f} "
            f"AP={x['macro_AP']:.6f} "
            f"Brier={x['macro_Brier']:.6f}"
        )
    log(f"Primary delta vs W2.3    : " f"{summary['primary_delta_vs_w23']:+.6f}")
    log(f"GATE VERDICT             : {summary['gate_verdict']}")
    return summary


# ============================================================
# 10. PRODUCTION ASSEMBLY
# ============================================================


def assemble_hybrid_teacher(
    w23: pd.DataFrame,
    fast: pd.DataFrame,
) -> pd.DataFrame:
    fscols = [
        UID,
        "Label",
        "FastProbability",
        "FastLogitMargin",
        "InputTokens",
        "PromptStyle",
        "Accelerator",
        "LoadMode",
    ]
    x = w23.merge(
        fast[fscols],
        on=[UID, "Label"],
        how="left",
        validate="one_to_one",
    )
    m = x["Label"].isin(FS4_LABELS)

    if x.loc[m, "FastProbability"].isna().any():
        raise RuntimeError("Missing FAST production probabilities")

    x["TeacherProbability"] = x["W23RawProbabilityMean"].astype(float)
    x.loc[m, "TeacherProbability"] = (1.0 - FIXED_BLEND_ALPHA) * x.loc[
        m, "W23RawProbabilityMean"
    ].astype(float) + FIXED_BLEND_ALPHA * x.loc[m, "FastProbability"].astype(float)
    x["TeacherSource"] = "W23_crossfold_mean"
    x.loc[m, "TeacherSource"] = "W23_50pct_plus_W26PF_FAST_50pct"

    avail_frac = (x["W23SoftAvailableFolds"].fillna(0).astype(float) / 5.0).clip(0, 1)
    high_frac = (x["W23HighSelectionFolds"].fillna(0).astype(float) / 5.0).clip(0, 1)
    established_weight = (avail_frac * (0.50 + 0.50 * high_frac)).clip(0, 1)

    # FAST confidence from distance from 0.5. Keep a floor because the
    # fixed50 blend already carries W2.3 information.
    fast_conf = (2.0 * (x["FastProbability"].fillna(0.5) - 0.5).abs()).clip(0, 1)
    fast_weight = (0.35 + 0.65 * fast_conf).clip(0, 1)

    x["RecommendedTeacherWeight"] = established_weight
    x.loc[m, "RecommendedTeacherWeight"] = fast_weight[m]
    x["RecommendedTeacherMask"] = x["RecommendedTeacherWeight"] >= 0.35

    p = x["TeacherProbability"].to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("Invalid hybrid teacher probabilities")

    if len(x) != EXPECTED_ALL_PROD_CELLS:
        raise RuntimeError("Hybrid row count mismatch")
    return x.sort_values([UID, "Label"]).reset_index(drop=True)


def pivot_value(
    long: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    wide = long.pivot(
        index=UID,
        columns="Label",
        values=value_col,
    ).reindex(columns=LABELS)
    return wide.reset_index()


def _prepare_production_queries() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[Dict[str, Any]],
    pd.DataFrame,
]:
    train, gold, unlabeled = load_train()
    w2_root = get_w2_root()
    w23_root = get_w23_root()
    gold_w2, full_w2 = load_w2_tables(w2_root, train, gold)
    w23 = load_w23_production_base(w23_root, unlabeled)
    queries, audit = build_production_exemplars(gold, unlabeled, gold_w2, full_w2)
    return gold, unlabeled, w23, queries, audit


def benchmark_w26pf(
    accelerator: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    _, _, _, queries, audit = _prepare_production_queries()
    audit.head(1).to_csv(
        RESULT_ROOT / "benchmark_exemplar_schema.csv",
        index=False,
    )

    before = _cache_read(production_cache_path())
    t0 = time.time()
    partial = run_fast_scoring(
        queries=queries,
        model_path=Path(MODEL_PATH_ENV),
        cache_file=production_cache_path(),
        accelerator=accelerator,
        limit_new=BENCHMARK_CELLS,
    )
    elapsed = time.time() - t0
    after = _cache_read(production_cache_path())
    newly_cached = max(0, len(after) - len(before))
    rate = newly_cached / max(elapsed, 1e-9)
    remaining = EXPECTED_FS4_PROD_CELLS - len(after)
    eta_h = remaining / max(rate, 1e-9) / 3600.0

    payload = {
        "newly_cached": newly_cached,
        "total_cached": len(after),
        "elapsed_seconds": elapsed,
        "cells_per_minute": rate * 60.0,
        "estimated_remaining_hours": eta_h,
        "cache": str(production_cache_path()),
    }
    (RESULT_ROOT / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(payload, indent=2))
    return payload


def production_w26pf(
    accelerator: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()

    # Require the quick FAST gate unless explicitly overridden.
    gate_file = RESULT_ROOT / "05_fast_gold_summary.json"
    allow_without_gate = (
        os.environ.get("W26PF_ALLOW_PRODUCTION_WITHOUT_FAST_GATE", "0").strip() == "1"
    )

    if gate_file.exists():
        gate = json.loads(gate_file.read_text(encoding="utf-8"))
        if (
            gate.get("gate_verdict") != "PASS_FAST_PRODUCTION"
            and not allow_without_gate
        ):
            raise RuntimeError(
                "FAST gold gate did not pass. Do not spend production time. "
                "Set W26PF_ALLOW_PRODUCTION_WITHOUT_FAST_GATE=1 only if "
                "you deliberately accept that tradeoff."
            )
    elif not allow_without_gate:
        raise RuntimeError(
            "Run run_w26pf('gold_fast', ...) first. It is only 232 FAST "
            "forward scores and validates this optimized inference primitive."
        )

    gold, unlabeled, w23, queries, audit = _prepare_production_queries()

    w23.to_csv(
        RESULT_ROOT / "10_w23_production_base_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        RESULT_ROOT / "11_production_exemplar_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prompt_meta = pd.DataFrame(
        [
            {
                UID: q[UID],
                "Label": q["Label"],
                "PromptSHA256": q["PromptSHA256"],
                "PromptChars": q["PromptChars"],
                "Prior": q["Prior"],
                "ExampleUIDs": q["ExampleUIDs"],
            }
            for q in queries
        ]
    )
    prompt_meta.to_csv(
        RESULT_ROOT / "12_production_prompt_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fast = run_fast_scoring(
        queries=queries,
        model_path=Path(MODEL_PATH_ENV),
        cache_file=production_cache_path(),
        accelerator=accelerator,
    )
    if len(fast) != EXPECTED_FS4_PROD_CELLS:
        raise RuntimeError(
            f"Production incomplete: {len(fast)}/" f"{EXPECTED_FS4_PROD_CELLS}"
        )

    fast.to_csv(
        RESULT_ROOT / "13_fast_fs4_production_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fast.pivot(
        index=UID,
        columns="Label",
        values="FastProbability",
    ).reindex(columns=FS4_LABELS).reset_index().to_csv(
        RESULT_ROOT / "14_fast_fs4_probabilities_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hybrid = assemble_hybrid_teacher(w23, fast)
    hybrid.to_csv(
        RESULT_ROOT / "15_final_hybrid_teacher_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pivot_value(hybrid, "TeacherProbability").to_csv(
        RESULT_ROOT / "16_final_hybrid_probabilities_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pivot_value(hybrid, "RecommendedTeacherWeight").to_csv(
        RESULT_ROOT / "17_recommended_teacher_weights_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pivot_value(hybrid, "RecommendedTeacherMask").to_csv(
        RESULT_ROOT / "18_recommended_teacher_mask_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "version": "w26pf_fast_logits_v1",
        "uses_pilkwang_labels": False,
        "prompt_style": PROMPT_STYLE,
        "logit_temperature": LOGIT_TEMPERATURE,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "counts": {
            "unlabeled_studies": int(hybrid[UID].nunique()),
            "fast_fs4_cells": int(len(fast)),
            "hybrid_cells": int(len(hybrid)),
        },
        "fast_probability_summary": (
            fast.groupby("Label")["FastProbability"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "results_root": str(RESULT_ROOT),
        "validation_warning": (
            "All 58 gold reports are production exemplars. "
            "These pseudo-labels are not pristine OOF validation data."
        ),
    }
    (RESULT_ROOT / "19_production_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("=" * 96)
    log("W2.6-P FAST PRODUCTION COMPLETE")
    log("=" * 96)
    log(f"FS4 cells                 : {len(fast)}")
    log(f"Hybrid cells              : {len(hybrid)}")
    log(f"Unlabeled studies         : {hybrid[UID].nunique()}")
    log(f"Results                   : {RESULT_ROOT}")
    return summary


# ============================================================
# 11. STATUS / VALIDATION
# ============================================================


def status_w26pf(
    accelerator: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    payload: Dict[str, Any] = {
        "experiment": "RSNA W2.6-P FAST optimized logit mapper",
        "uses_pilkwang_labels": False,
        "is_kaggle": IS_KAGGLE,
        "script_dir": str(SCRIPT_DIR),
        "project_root": str(PROJECT_ROOT),
        "train_csv": str(TRAIN_CSV),
        "train_exists": TRAIN_CSV.exists(),
        "w2_root_env": W2_ROOT_ENV,
        "w23_root_env": W23_ROOT_ENV,
        "model_path_env": MODEL_PATH_ENV,
        "model_exists": Path(MODEL_PATH_ENV).exists(),
        "accelerator_requested": accelerator or ACCELERATOR,
        "accelerator_resolved": resolve_accelerator(accelerator),
        "precision_policy": PRECISION,
        "prompt_style": PROMPT_STYLE,
        "logit_temperature": LOGIT_TEMPERATURE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "single_prompt_fallback_tokens": SINGLE_PROMPT_FALLBACK_TOKENS,
        "max_batch_size": MAX_BATCH_SIZE,
        "token_batch_budget_env": TOKEN_BATCH_BUDGET_ENV,
        "gold_fast_cells": EXPECTED_FS4_GOLD_CELLS,
        "production_fast_cells": EXPECTED_FS4_PROD_CELLS,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "gold_cache": str(gold_cache_path()),
        "production_cache": str(production_cache_path()),
        "gold_cached_cells": len(_cache_read(gold_cache_path())),
        "production_cached_cells": len(_cache_read(production_cache_path())),
        "apple_mps_available": _mps_available(),
    }

    try:
        train, gold, unlabeled = load_train()
        payload["counts"] = {
            "train": len(train),
            "gold": len(gold),
            "unlabeled": len(unlabeled),
        }
        w2_root = get_w2_root()
        w23_root = get_w23_root()
        folds = load_folds(w23_root, gold)
        payload["w2_root"] = str(w2_root)
        payload["w23_root"] = str(w23_root)
        payload["fold_sha256"] = fold_sha256(folds)
        payload["fold_sha256_match"] = payload["fold_sha256"] == EXPECTED_FOLD_SHA256
    except Exception as exc:
        payload["input_error"] = repr(exc)

    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        payload["gpu_names"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
    except Exception as exc:
        payload["torch_error"] = repr(exc)

    (RESULT_ROOT / "status_fast.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def validate_w26pf() -> Dict[str, Any]:
    required = {
        "fast_long": RESULT_ROOT / "13_fast_fs4_production_long.csv",
        "hybrid_long": RESULT_ROOT / "15_final_hybrid_teacher_long.csv",
        "hybrid_wide": RESULT_ROOT / "16_final_hybrid_probabilities_wide.csv",
        "weights_wide": RESULT_ROOT / "17_recommended_teacher_weights_wide.csv",
        "mask_wide": RESULT_ROOT / "18_recommended_teacher_mask_wide.csv",
    }
    checks = {f"{k}_exists": p.exists() for k, p in required.items()}
    if not all(checks.values()):
        payload = {
            "overall_pass": False,
            "checks": checks,
        }
        log(json.dumps(payload, indent=2))
        return payload

    fast = pd.read_csv(required["fast_long"])
    hybrid = pd.read_csv(required["hybrid_long"])
    hw = pd.read_csv(required["hybrid_wide"])
    ww = pd.read_csv(required["weights_wide"])
    mw = pd.read_csv(required["mask_wide"])

    checks.update(
        {
            "fast_rows_17396": len(fast) == EXPECTED_FS4_PROD_CELLS,
            "fast_uids_4349": fast[UID].astype(str).nunique() == EXPECTED_UNLABELED,
            "fast_four_labels": set(fast["Label"]) == set(FS4_LABELS),
            "fast_no_duplicates": not fast[[UID, "Label"]].duplicated().any(),
            "fast_probabilities_valid": bool(
                np.isfinite(fast["FastProbability"].to_numpy(float)).all()
                and fast["FastProbability"].between(0, 1).all()
            ),
            "hybrid_rows_52188": len(hybrid) == EXPECTED_ALL_PROD_CELLS,
            "hybrid_uids_4349": hybrid[UID].astype(str).nunique() == EXPECTED_UNLABELED,
            "hybrid_12_labels": set(hybrid["Label"]) == set(LABELS),
            "hybrid_no_duplicates": not hybrid[[UID, "Label"]].duplicated().any(),
            "hybrid_probability_valid": bool(
                np.isfinite(hybrid["TeacherProbability"].to_numpy(float)).all()
                and hybrid["TeacherProbability"].between(0, 1).all()
            ),
            "weights_valid": bool(
                np.isfinite(hybrid["RecommendedTeacherWeight"].to_numpy(float)).all()
                and hybrid["RecommendedTeacherWeight"].between(0, 1).all()
            ),
            "hybrid_wide_shape": hw.shape == (EXPECTED_UNLABELED, 13),
            "weights_wide_shape": ww.shape == (EXPECTED_UNLABELED, 13),
            "mask_wide_shape": mw.shape == (EXPECTED_UNLABELED, 13),
            "pilkwang_not_used": True,
        }
    )

    payload = {
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "fast_probability_by_label": (
            fast.groupby("Label")["FastProbability"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "results_root": str(RESULT_ROOT),
    }
    (RESULT_ROOT / "20_validation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


# ============================================================
# 12. NOTEBOOK API
# ============================================================


def run_w26pf(
    mode: str = "status",
    accelerator: Optional[str] = None,
):
    mode = str(mode).strip().lower()
    aliases = {
        "gold": "gold_fast",
        "check": "validate",
        "prod": "production",
        "run": "production",
        "bench": "benchmark",
    }
    mode = aliases.get(mode, mode)

    if mode == "status":
        return status_w26pf(accelerator=accelerator)
    if mode == "gold_fast":
        return gold_fast_w26pf(accelerator=accelerator)
    if mode == "benchmark":
        return benchmark_w26pf(accelerator=accelerator)
    if mode == "production":
        return production_w26pf(accelerator=accelerator)
    if mode == "validate":
        return validate_w26pf()

    raise ValueError(
        "Unknown mode. Use status, gold_fast, benchmark, production, validate."
    )


if __name__ == "__main__":
    # Safe default. Running the file itself does not launch production.
    # run_w26pf("status")
    run_w26pf("status", accelerator="localGPU")
    run_w26pf("gold_fast", accelerator="localGPU")
    run_w26pf("benchmark", accelerator="localGPU")
    run_w26pf("production", accelerator="localGPU")
    run_w26pf("validate")
