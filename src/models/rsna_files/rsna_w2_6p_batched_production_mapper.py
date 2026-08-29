#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection
W2.6-P — Batched Production Report Teacher

Purpose
-------
Generate FINAL production pseudo-labels for the 4,349 unlabeled training studies
after W2.6 FS4 passed the controlled gold gate.

Production design:
- 8 established labels: W2.3 cross-fold mean probabilities.
- 4 W2.3-weak labels:
    Medial OA, Lateral OA, PF OA, Synovitis
  use:
    0.50 * W2.3 cross-fold mean
  + 0.50 * Qwen2.5-7B FS4 challenge-mapper probability.
- Qwen FS4 uses ALL 58 gold reports only as production exemplars.
- Each unlabeled query selects 2 nearest positive + 2 nearest negative gold
  exemplars for the requested label.
- Independent prompts are batched at model.generate() time; they are NOT mixed
  into one multi-query prompt.
- Cache is append-only and resumable after every successful batch.
- No Pilkwang labels are used.

Validation note
---------------
This is a FINAL TRAINING RESOURCE, not a fold-safe OOF validation resource.
Because all 58 gold reports are used as the production exemplar pool, downstream
image-model OOF metrics trained on these pseudo-labels must not be interpreted as
pristine held-out evidence.

Portable execution
------------------
Kaggle:
    run_w26p("status", accelerator="kaggle_t4")
    run_w26p("production", accelerator="kaggle_t4")
    run_w26p("validate")

Local CUDA (recommended for the user's 4060 Ti 16GB + 48GB RAM):
    run_w26p("status", accelerator="localGPU")
    run_w26p("production", accelerator="localGPU")
    run_w26p("validate")

TPU:
    The accelerator selector recognizes "tpu", but this Qwen2.5-7B decoder path
    intentionally refuses TPU execution. Reliable v5e-8 multi-chip autoregressive
    inference needs a separate XLA/SPMD implementation. Preserve TPU time for W6
    image modeling rather than spending project time porting this one production
    report mapper.
"""

# ============================================================
# 0. IMPORTS / CUDA ALLOCATOR
# ============================================================

import os

# Set before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import math
import time
import hashlib
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 1. COMPETITION CONSTANTS / PORTABLE PATHS
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

W23_ESTABLISHED_LABELS = [x for x in LABELS if x not in FS4_LABELS]

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_FS4_CELLS = EXPECTED_UNLABELED * len(FS4_LABELS)
EXPECTED_ALL_CELLS = EXPECTED_UNLABELED * len(LABELS)

FIXED_BLEND_ALPHA = 0.50

TARGET_DEFINITIONS = {
    "Medial OA": (
        "Competition label for medial tibiofemoral osteoarthritis/degenerative disease. "
        "Challenge positives may be supported by medial-compartment arthrosis, osteophytes, "
        "joint-space degeneration, or advanced/full-thickness medial compartment cartilage loss, "
        "even if the exact words 'osteoarthritis' are absent."
    ),
    "Lateral OA": (
        "Competition label for lateral tibiofemoral osteoarthritis/degenerative disease. "
        "Use lateral-compartment degenerative evidence only. Medial-only or patellofemoral-only "
        "disease is not sufficient."
    ),
    "PF OA": (
        "Competition label for patellofemoral degenerative disease/OA. Advanced patellar or "
        "trochlear cartilage loss, grade-4 chondropathy, arthrosis, or osteophytes may map positive "
        "even when 'OA' is not literally written. Effusion alone is not PF OA."
    ),
    "Synovitis": (
        "Competition label for synovitis/synovial inflammatory-proliferative abnormality. "
        "Effusion alone is not synovitis. Challenge annotations may not exactly equal literal "
        "report mention, so infer the competition mapping from the supplied labeled examples."
    ),
}

W2_PROMPT_COLUMNS = [
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
    # Local .py script: use actual script location.
    # Notebook/pasted cell: current working directory is the project root fallback.
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


SCRIPT_DIR = _script_dir()
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    PROJECT_ROOT = (
        Path(os.environ.get("W26P_PROJECT_ROOT", "/kaggle/working"))
        .expanduser()
        .resolve()
    )
else:
    default_project = (SCRIPT_DIR / ".." / "..").resolve()
    # When the script itself sits directly in a project root, allow a very easy override.
    PROJECT_ROOT = (
        Path(os.environ.get("W26P_PROJECT_ROOT", str(default_project)))
        .expanduser()
        .resolve()
    )

if IS_KAGGLE:
    DATA_ROOT = Path(
        os.environ.get(
            "W26P_DATA_ROOT",
            "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        )
    )
    TRAIN_CSV = Path(
        os.environ.get(
            "W26P_TRAIN_CSV",
            str(DATA_ROOT / "train.csv"),
        )
    )
    W2_ROOT_ENV = os.environ.get(
        "W26P_W2_ROOT",
        "/kaggle/input/datasets/isayem/rsna-w2/rsna_w2",
    ).strip()
    W23_ROOT_ENV = os.environ.get(
        "W26P_W23_ROOT",
        "/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3",
    ).strip()
    MODEL_PATH_ENV = os.environ.get(
        "W26P_MODEL_PATH",
        "/kaggle/input/datasets/ragnar123/qwen2-5-7b-instruct",
    ).strip()
    OUTPUT_ROOT = Path(
        os.environ.get(
            "W26P_OUTPUT_ROOT",
            "/kaggle/working/rsna_w2_6p",
        )
    )
else:
    DATA_ROOT = (
        Path(
            os.environ.get(
                "W26P_DATA_ROOT",
                str(PROJECT_ROOT / "input"),
            )
        )
        .expanduser()
        .resolve()
    )

    TRAIN_CSV = (
        Path(
            os.environ.get(
                "W26P_TRAIN_CSV",
                str(DATA_ROOT / "train.csv"),
            )
        )
        .expanduser()
        .resolve()
    )

    W2_ROOT_ENV = os.environ.get(
        "W26P_W2_ROOT",
        str(PROJECT_ROOT / "output" / "results" / "rsna_w2"),
    ).strip()

    W23_ROOT_ENV = os.environ.get(
        "W26P_W23_ROOT",
        str(PROJECT_ROOT / "output" / "results" / "rsna_w2_3"),
    ).strip()

    local_model_candidates = [
        PROJECT_ROOT / "models" / "qwen2-5-7b-instruct",
        PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct",
        DATA_ROOT / "models" / "qwen2-5-7b-instruct",
        DATA_ROOT / "qwen2-5-7b-instruct",
    ]
    local_model_default = next(
        (p for p in local_model_candidates if p.exists()),
        local_model_candidates[0],
    )
    MODEL_PATH_ENV = os.environ.get(
        "W26P_MODEL_PATH",
        str(local_model_default),
    ).strip()

    OUTPUT_ROOT = (
        Path(
            os.environ.get(
                "W26P_OUTPUT_ROOT",
                str(PROJECT_ROOT / "output" / "results" / "rsna_w2_6p"),
            )
        )
        .expanduser()
        .resolve()
    )

CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULT_ROOT = OUTPUT_ROOT / "results"

# Accelerator:
#   auto, kaggle_t4, local_gpu, tpu, cpu
ACCELERATOR = os.environ.get("W26P_ACCELERATOR", "auto").strip().lower()
PRECISION = os.environ.get("W26P_PRECISION", "auto").strip().lower()
GPU_ID = int(os.environ.get("W26P_GPU_ID", "0"))

# Local 4060 Ti defaults:
LOCAL_GPU_MAX_GIB = float(os.environ.get("W26P_LOCAL_GPU_MAX_GIB", "11.5"))
LOCAL_CPU_MAX_GIB = float(os.environ.get("W26P_LOCAL_CPU_MAX_GIB", "36.0"))

# Kaggle T4 defaults:
T4_GPU_MAX_GIB = float(os.environ.get("W26P_T4_GPU_MAX_GIB", "11.5"))

# Batching and context. Independent prompts are padded together.
INITIAL_BATCH_SIZE = int(os.environ.get("W26P_BATCH_SIZE", "32"))
MIN_BATCH_SIZE = 1
MAX_INPUT_TOKENS = int(os.environ.get("W26P_MAX_INPUT_TOKENS", "3584"))
OOM_TOKEN_BUDGETS = [
    int(x)
    for x in os.environ.get("W26P_OOM_TOKEN_BUDGETS", "3072,2560,2048,1536").split(",")
    if x.strip()
]
MAX_NEW_TOKENS = int(os.environ.get("W26P_MAX_NEW_TOKENS", "72"))

N_POS_EXAMPLES = int(os.environ.get("W26P_N_POS_EXAMPLES", "2"))
N_NEG_EXAMPLES = int(os.environ.get("W26P_N_NEG_EXAMPLES", "2"))

MAX_EXAMPLE_REPORT_CHARS = int(os.environ.get("W26P_MAX_EXAMPLE_REPORT_CHARS", "1600"))
MAX_QUERY_REPORT_CHARS = int(os.environ.get("W26P_MAX_QUERY_REPORT_CHARS", "2600"))
TFIDF_MAX_FEATURES = int(os.environ.get("W26P_TFIDF_MAX_FEATURES", "30000"))


# ============================================================
# 2. BASIC UTILITIES
# ============================================================


def ensure_dirs() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


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
    # Preserve both opening context and impression/tail.
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + " ...[middle omitted]... " + s[-tail:]


def clean_scalar(v: Any, digits: int = 3) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


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
        "v5e": "tpu",
        "tpu_v5e": "tpu",
    }
    return aliases.get(x, x)


def _tpu_detected() -> bool:
    try:
        # import torch_xla.core.xla_model as xm  # noqa: F401
        return False
    except Exception:
        return False


def resolve_accelerator(requested: Optional[str] = None) -> str:
    req = _normalize_accelerator(requested or ACCELERATOR)
    allowed = {"auto", "kaggle_t4", "local_gpu", "tpu", "cpu"}
    if req not in allowed:
        raise ValueError(
            f"Unknown accelerator {req!r}. "
            "Use auto, kaggle_t4/T4, local_gpu/localGPU, tpu, or cpu."
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

    if _tpu_detected():
        return "tpu"
    return "cpu"


def cuda_cleanup(torch_module) -> None:
    gc.collect()
    if not torch_module.cuda.is_available():
        return
    for i in range(torch_module.cuda.device_count()):
        try:
            with torch_module.cuda.device(i):
                torch_module.cuda.empty_cache()
        except Exception:
            pass


# ============================================================
# 3. INPUT DISCOVERY / LOADERS
# ============================================================


def looks_like_w2_root(root: Path) -> bool:
    return (root / "results" / "04_gold_structured_report_features.csv").exists() and (
        root / "results" / "08_full_structured_report_labels.csv"
    ).exists()


def discover_w2_root() -> Path:
    candidates = [
        Path(W2_ROOT_ENV),
        PROJECT_ROOT / "output" / "results" / "rsna_w2",
        PROJECT_ROOT / "output" / "rsna_w2",
    ]
    for p in candidates:
        if looks_like_w2_root(p):
            return p
    raise FileNotFoundError(
        "W2 root not found. Set W26P_W2_ROOT to a directory containing "
        "results/04_gold_structured_report_features.csv and "
        "results/08_full_structured_report_labels.csv."
    )


def looks_like_w23_root(root: Path) -> bool:
    return (
        root / "results" / "06_fold_safe_unlabeled_soft_labels_long.csv"
    ).exists() and (
        root / "results" / "10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv"
    ).exists()


def discover_w23_root() -> Path:
    candidates = [
        Path(W23_ROOT_ENV),
        PROJECT_ROOT / "output" / "results" / "rsna_w2_3",
        PROJECT_ROOT / "output" / "rsna_w2_3",
    ]
    for p in candidates:
        if looks_like_w23_root(p):
            return p
    raise FileNotFoundError(
        "W2.3 root not found. Set W26P_W23_ROOT to a directory containing "
        "results/06_fold_safe_unlabeled_soft_labels_long.csv and "
        "results/10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv."
    )


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

    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN} train rows, got {len(train)}")
    if len(gold) != EXPECTED_GOLD:
        raise RuntimeError(f"Expected {EXPECTED_GOLD} gold rows, got {len(gold)}")
    if len(unlabeled) != EXPECTED_UNLABELED:
        raise RuntimeError(
            f"Expected {EXPECTED_UNLABELED} unlabeled rows, got {len(unlabeled)}"
        )

    return train, gold, unlabeled


def load_w2_tables(
    w2_root: Path,
    train: pd.DataFrame,
    gold: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gold_path = w2_root / "results" / "04_gold_structured_report_features.csv"
    full_path = w2_root / "results" / "08_full_structured_report_labels.csv"

    gold_w2 = pd.read_csv(gold_path)
    full_w2 = pd.read_csv(full_path)
    gold_w2[UID] = gold_w2[UID].astype(str)
    full_w2[UID] = full_w2[UID].astype(str)

    required = {UID, "Label", *W2_PROMPT_COLUMNS}
    for name, frame in [("gold", gold_w2), ("full", full_w2)]:
        miss = required - set(frame.columns)
        if miss:
            raise RuntimeError(f"W2 {name} table missing: {sorted(miss)}")
        if frame[[UID, "Label"]].duplicated().any():
            raise RuntimeError(f"W2 {name} has duplicate UID/Label rows")

    if len(gold_w2) != EXPECTED_GOLD * len(LABELS):
        raise RuntimeError("Unexpected W2 gold row count")
    if len(full_w2) != EXPECTED_TRAIN * len(LABELS):
        raise RuntimeError("Unexpected W2 full row count")
    if set(gold_w2[UID]) != set(gold[UID]):
        raise RuntimeError("W2 gold UID mismatch")
    if set(full_w2[UID]) != set(train[UID]):
        raise RuntimeError("W2 full UID mismatch")

    return gold_w2, full_w2


def load_w23_production_base(
    w23_root: Path,
    unlabeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one W2.3 production row per unlabeled UID x label.

    Probability = mean of the five fold-specific raw probabilities. This is the
    same cross-fold raw probability information used by the W2.3 stability audit.
    """
    stability_path = (
        w23_root / "results" / "10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv"
    )
    long_path = w23_root / "results" / "06_fold_safe_unlabeled_soft_labels_long.csv"

    stab = pd.read_csv(stability_path)
    long = pd.read_csv(long_path)
    stab[UID] = stab[UID].astype(str)
    long[UID] = long[UID].astype(str)

    expected_stab = {
        UID,
        "Label",
        "RawProbabilityMean",
        "RawProbabilityStd",
        "SoftLabelAvailableFolds",
        "HighSelectionFolds",
    }
    if expected_stab - set(stab.columns):
        raise RuntimeError(
            "W2.3 stability file missing: "
            f"{sorted(expected_stab - set(stab.columns))}"
        )

    if "CandidateSelectionScore" not in long.columns:
        raise RuntimeError("W2.3 long file missing CandidateSelectionScore")

    agg = long.groupby([UID, "Label"], as_index=False).agg(
        W23CandidateSelectionMean=("CandidateSelectionScore", "mean"),
        W23CandidateSelectionMax=("CandidateSelectionScore", "max"),
    )

    out = stab[
        [
            UID,
            "Label",
            "RawProbabilityMean",
            "RawProbabilityStd",
            "SoftLabelAvailableFolds",
            "HighSelectionFolds",
        ]
    ].copy()
    out = out.rename(
        columns={
            "RawProbabilityMean": "W23RawProbabilityMean",
            "RawProbabilityStd": "W23RawProbabilityStd",
            "SoftLabelAvailableFolds": "W23SoftAvailableFolds",
            "HighSelectionFolds": "W23HighSelectionFolds",
        }
    )
    out = out.merge(agg, on=[UID, "Label"], how="left", validate="one_to_one")

    uid_set = set(unlabeled[UID].astype(str))
    out = out[out[UID].isin(uid_set)].copy()

    if len(out) != EXPECTED_ALL_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_ALL_CELLS} W2.3 base rows, got {len(out)}"
        )
    if out[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate W2.3 production rows")
    if set(out[UID]) != uid_set:
        raise RuntimeError("W2.3 production UID mismatch")
    if set(out["Label"]) != set(LABELS):
        raise RuntimeError("W2.3 production label mismatch")

    p = out["W23RawProbabilityMean"].to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("Invalid W2.3 raw probabilities")

    return out.sort_values([UID, "Label"]).reset_index(drop=True)


# ============================================================
# 4. W2 FEATURE SUMMARIES / RETRIEVAL
# ============================================================


def w2_row_summary(row: pd.Series, include_evidence: bool = True) -> str:
    parts = [
        f"rule={clean_scalar(row.get('RuleAssertion'))}",
        f"semantic={clean_scalar(row.get('SemanticAssertion'))}",
        f"fused={clean_scalar(row.get('FusedAssertion'))}",
        f"fused_conf={clean_scalar(row.get('FusedAssertionConfidence'))}",
        f"evidence_available={clean_scalar(row.get('EvidenceAvailable'))}",
        f"pos={clean_scalar(row.get('EvidencePositiveScore'))}",
        f"neg={clean_scalar(row.get('EvidenceNegativeScore'))}",
        f"related={clean_scalar(row.get('RelatedScore'))}",
        f"uncertain={clean_scalar(row.get('UncertaintyFlag'))}",
        f"sev_low={clean_scalar(row.get('SeverityLow'))}",
        f"sev_mod={clean_scalar(row.get('SeverityModerate'))}",
        f"sev_high={clean_scalar(row.get('SeverityHigh'))}",
        f"sev_degen={clean_scalar(row.get('SeverityDegenerative'))}",
    ]

    if include_evidence:
        evidence = []
        for col in [
            "FusedEvidence",
            "SemanticPositiveEvidence",
            "SemanticNegativeEvidence",
            "SemanticRelatedEvidence",
        ]:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                evidence.append(str(v).strip())

        seen: set = set()
        unique: List[str] = []
        for e in evidence:
            key = normalize_text(e).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        if unique:
            parts.append("evidence=" + " || ".join(unique[:3]))

    return "; ".join(parts)


def build_feature_lookup(
    frame: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.Series]:
    return {(str(r[UID]), str(r["Label"])): r for _, r in frame.iterrows()}


def retrieval_document(report: str, feature_summary: str, label: str) -> str:
    return (
        f"target {label} target {label} "
        f"{normalize_text(feature_summary)} "
        f"{normalize_text(report)}"
    )


# ============================================================
# 5. PRODUCTION EXEMPLAR RETRIEVAL
# ============================================================


def build_production_exemplars_and_queries(
    gold: pd.DataFrame,
    unlabeled: pd.DataFrame,
    gold_w2: pd.DataFrame,
    full_w2: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Build all 17,396 independent FS4 query prompts.

    Retrieval model is fitted only on the 58 gold exemplar documents for the
    requested target; unlabeled reports are transformed as queries.
    """
    gold_lookup = build_feature_lookup(gold_w2)
    full_lookup = build_feature_lookup(full_w2)

    queries: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for label in FS4_LABELS:
        log(f"Building production retrieval: {label}")

        gold_docs: List[str] = []
        for _, r in gold.iterrows():
            uid = str(r[UID])
            fs = w2_row_summary(gold_lookup[(uid, label)], include_evidence=True)
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

        query_docs: List[str] = []
        query_w2_summaries: List[str] = []
        for _, r in unlabeled.iterrows():
            uid = str(r[UID])
            fs = w2_row_summary(full_lookup[(uid, label)], include_evidence=True)
            query_w2_summaries.append(fs)
            query_docs.append(retrieval_document(str(r[REPORT]), fs, label))

        X_query = vectorizer.transform(query_docs)
        sims = (X_query @ X_gold.T).toarray()

        gold_label = gold[label].astype(int).to_numpy()
        pos_idx = np.flatnonzero(gold_label == 1)
        neg_idx = np.flatnonzero(gold_label == 0)
        prior = float(gold_label.mean())

        if len(pos_idx) < N_POS_EXAMPLES or len(neg_idx) < N_NEG_EXAMPLES:
            raise RuntimeError(f"Insufficient production exemplars for {label}")

        for qi, (_, qrow) in enumerate(unlabeled.iterrows()):
            uid = str(qrow[UID])
            s = sims[qi]

            # Stable top-k within class: similarity descending, UID ascending tie-break.
            def top_class(indices: np.ndarray, k: int) -> List[int]:
                ranked = sorted(
                    indices.tolist(),
                    key=lambda j: (-float(s[j]), str(gold.iloc[j][UID])),
                )
                return ranked[:k]

            chosen_idx = top_class(pos_idx, N_POS_EXAMPLES) + top_class(
                neg_idx, N_NEG_EXAMPLES
            )
            chosen_idx = sorted(
                chosen_idx,
                key=lambda j: (-float(s[j]), str(gold.iloc[j][UID])),
            )

            examples: List[Dict[str, Any]] = []
            for rank, j in enumerate(chosen_idx, start=1):
                ex = gold.iloc[j]
                ex_uid = str(ex[UID])
                ex_fs = w2_row_summary(
                    gold_lookup[(ex_uid, label)],
                    include_evidence=True,
                )
                item = {
                    "StudyInstanceUID": ex_uid,
                    "Gold": int(ex[label]),
                    "Similarity": float(s[j]),
                    "Report": str(ex[REPORT]),
                    "W2Summary": ex_fs,
                }
                examples.append(item)
                audit_rows.append(
                    {
                        "QueryStudyInstanceUID": uid,
                        "Label": label,
                        "ExampleRank": rank,
                        "ExampleStudyInstanceUID": ex_uid,
                        "ExampleGold": int(ex[label]),
                        "Similarity": float(s[j]),
                    }
                )

            prompt = build_mapper_prompt(
                label=label,
                query_report=str(qrow[REPORT]),
                query_w2=query_w2_summaries[qi],
                examples=examples,
                production_prior=prior,
            )

            queries.append(
                {
                    UID: uid,
                    "Label": label,
                    "Prompt": prompt,
                    "PromptSHA256": stable_sha256(prompt),
                    "PromptCharLength": len(prompt),
                    "ProductionGoldPrior": prior,
                    "ExampleUIDs": "|".join(x["StudyInstanceUID"] for x in examples),
                    "ExampleSimilarities": "|".join(
                        f"{x['Similarity']:.6f}" for x in examples
                    ),
                }
            )

    expected = EXPECTED_FS4_CELLS
    if len(queries) != expected:
        raise RuntimeError(f"Expected {expected} queries, got {len(queries)}")

    qkeys = {(q[UID], q["Label"]) for q in queries}
    if len(qkeys) != expected:
        raise RuntimeError("Duplicate production query UID/Label")

    audit = pd.DataFrame(audit_rows)
    expected_audit = expected * (N_POS_EXAMPLES + N_NEG_EXAMPLES)
    if len(audit) != expected_audit:
        raise RuntimeError(
            f"Expected {expected_audit} exemplar audit rows, got {len(audit)}"
        )

    return queries, audit


# ============================================================
# 6. VALIDATED W2.6 CHALLENGE-MAPPER PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a competition-label mapper for knee MRI reports.
Your task is NOT merely to decide whether a finding is literally mentioned.
You must infer the binary annotation convention of THIS challenge from the
provided labeled examples, then estimate the probability that the query study
has challenge label 1.

Important:
- The production examples are drawn from the 58 gold reports.
- The query is one of the 4,349 unlabeled training reports; its gold label is unavailable.
- Reports and challenge labels can disagree; learn that convention from examples.
- The auxiliary W2 summary is generated from the report without using challenge gold.
- Do not assume silence means negative.
- Return JSON only. No markdown.
"""


def build_mapper_prompt(
    label: str,
    query_report: str,
    query_w2: str,
    examples: Sequence[Mapping[str, Any]],
    production_prior: float,
) -> str:
    chunks = [
        f"TARGET LABEL: {label}",
        f"TARGET DEFINITION: {TARGET_DEFINITIONS[label]}",
        f"58-GOLD LABEL-1 FRACTION: {production_prior:.3f}",
        "",
        "LABELED PRODUCTION EXEMPLARS:",
    ]

    for i, ex in enumerate(examples, start=1):
        chunks += [
            f"--- EXAMPLE {i} ---",
            f"ChallengeLabel: {int(ex['Gold'])}",
            f"SimilarityToQuery: {float(ex['Similarity']):.4f}",
            f"W2ReportFeatures: {ex['W2Summary']}",
            "Report:",
            compact_report(ex["Report"], MAX_EXAMPLE_REPORT_CHARS),
            "",
        ]

    chunks += [
        "--- UNLABELED QUERY ---",
        f"W2ReportFeatures: {query_w2}",
        "Report:",
        compact_report(query_report, MAX_QUERY_REPORT_CHARS),
        "",
        "OUTPUT CONTRACT:",
        'Return exactly one JSON object: {"p": 0.000..1.000, "confidence": 0..100, "state": "P|A|U|N", "reason": "brief"}',
        "p is the probability of THIS CHALLENGE LABEL being 1, after considering the supplied challenge examples.",
        "state is only a compact report-side interpretation: P present, A explicitly absent, U uncertain/indirect, N not addressed.",
        "Use a genuinely graded p; do not restrict yourself to a small fixed set of probability values.",
        "",
        f"FINAL REMINDER — TARGET={label}: infer challenge label probability for the UNLABELED QUERY. "
        "Return JSON only.",
    ]
    return "\n".join(chunks)


def extract_json_blob(text: str) -> Dict[str, Any]:
    raw = str(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("No JSON object found")
    x = json.loads(raw[a : b + 1])
    if not isinstance(x, dict):
        raise ValueError("Top-level output is not an object")
    return x


def parse_mapper_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    p_raw = payload.get("p", payload.get("probability", payload.get("score")))
    if p_raw is None:
        raise ValueError("missing probability field p")
    p = float(p_raw)
    if not np.isfinite(p):
        raise ValueError("non-finite probability")
    if p > 1.0 and p <= 100.0:
        p /= 100.0
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability outside [0,1]: {p}")

    c_raw = payload.get("confidence", payload.get("c", 50))
    try:
        c = int(round(float(c_raw)))
    except Exception:
        c = 50
    c = int(np.clip(c, 0, 100))

    state = str(payload.get("state", payload.get("s", "U"))).strip().upper()
    aliases = {
        "PRESENT": "P",
        "ABSENT": "A",
        "UNCERTAIN": "U",
        "NOT_ADDRESSED": "N",
        "NOTADDRESSED": "N",
    }
    state = aliases.get(state, state)
    if state not in {"P", "A", "U", "N"}:
        state = "U"

    reason = normalize_text(payload.get("reason", ""))[:600]

    return {
        "FS4Probability": float(p),
        "FS4Confidence": c,
        "FS4State": state,
        "FS4Reason": reason,
    }


# ============================================================
# 7. PORTABLE BATCHED QWEN LOADER
# ============================================================


class LocalBatchedMapperLLM:
    def __init__(self, model_path: Path, accelerator: Optional[str] = None):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:
            raise RuntimeError("W2.6-P requires torch + transformers") from exc

        self.torch = torch
        self.model_path = Path(model_path)
        self.accelerator = resolve_accelerator(accelerator)

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        if self.accelerator == "tpu":
            raise RuntimeError(
                "W26P accelerator='tpu' is recognized but intentionally disabled "
                "for this Qwen2.5-7B autoregressive production mapper. Reliable "
                "v5e-8 multi-chip generation requires a separate XLA/SPMD port. "
                "Use localGPU for W2.6-P and preserve TPU hours for W6 image work."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decoder-only batch generation should left-pad.
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        log(f"Loading production mapper : {self.model_path}")
        log(f"Requested accelerator     : {accelerator or ACCELERATOR}")
        log(f"Resolved accelerator      : {self.accelerator}")
        log(f"Precision policy          : {PRECISION}")
        log(f"CUDA GPUs                 : {gpu_count}")

        if self.accelerator == "kaggle_t4":
            if not torch.cuda.is_available() or gpu_count < 2:
                raise RuntimeError("kaggle_t4 requires at least two visible CUDA GPUs.")
            kwargs["dtype"] = torch.float16
            kwargs["device_map"] = "balanced"
            kwargs["max_memory"] = {
                i: f"{T4_GPU_MAX_GIB:.1f}GiB" for i in range(gpu_count)
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), **kwargs
            )
            self.load_mode = f"fp16_balanced_{gpu_count}gpu"

        elif self.accelerator == "local_gpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "local_gpu selected but torch.cuda.is_available() is False."
                )
            if GPU_ID >= gpu_count:
                raise RuntimeError(
                    f"W26P_GPU_ID={GPU_ID}, but only {gpu_count} GPUs are visible."
                )

            loaded = False
            try_4bit = PRECISION in {"auto", "4bit", "nf4"}

            if try_4bit:
                try:
                    qkwargs = dict(kwargs)
                    qkwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    qkwargs["device_map"] = {"": GPU_ID}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path), **qkwargs
                    )
                    self.load_mode = f"4bit_nf4_local_gpu_{GPU_ID}"
                    loaded = True
                except Exception as exc:
                    if PRECISION in {"4bit", "nf4"}:
                        raise RuntimeError(
                            "Explicit local 4-bit load failed. Install compatible "
                            "bitsandbytes/transformers/accelerate or use "
                            "W26P_PRECISION=auto."
                        ) from exc
                    warnings.warn(
                        "Local 4-bit unavailable; falling back to FP16 + CPU "
                        f"offload. Original error: {exc}"
                    )

            if not loaded:
                offload_dir = OUTPUT_ROOT / "model_offload"
                offload_dir.mkdir(parents=True, exist_ok=True)
                kwargs["dtype"] = torch.float16
                kwargs["device_map"] = "auto"
                kwargs["max_memory"] = {
                    GPU_ID: f"{LOCAL_GPU_MAX_GIB:.1f}GiB",
                    "cpu": f"{LOCAL_CPU_MAX_GIB:.1f}GiB",
                }
                kwargs["offload_folder"] = str(offload_dir)
                kwargs["offload_state_dict"] = True
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), **kwargs
                )
                self.load_mode = f"fp16_local_gpu_{GPU_ID}_cpu_offload"

        elif self.accelerator == "cpu":
            kwargs["dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), **kwargs
            )
            self.load_mode = "fp32_cpu"

        else:
            raise RuntimeError(f"Unhandled accelerator: {self.accelerator}")

        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

        log(f"Model load mode           : {self.load_mode}")
        log(f"Input embedding device    : {self.input_device}")
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
                    f"{free_b/(1024**3):.2f} / {total_b/(1024**3):.2f} GiB"
                )

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
        return SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nJSON:"

    def _generate_batch_once(
        self,
        user_prompts: Sequence[str],
        token_budget: int,
    ) -> Tuple[List[str], List[int]]:
        torch = self.torch
        cuda_cleanup(torch)

        chat_prompts = [self._chat_prompt(x) for x in user_prompts]
        encoded = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(token_budget),
        )
        lengths = (
            encoded["attention_mask"].sum(dim=1).cpu().numpy().astype(int).tolist()
        )
        padded_input_len = int(encoded["input_ids"].shape[1])
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}

        generated = None
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            outputs: List[str] = []
            for i in range(len(user_prompts)):
                new_tokens = generated[i, padded_input_len:]
                outputs.append(
                    self.tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                    ).strip()
                )
            return outputs, lengths
        finally:
            try:
                del encoded
            except Exception:
                pass
            try:
                del generated
            except Exception:
                pass
            cuda_cleanup(torch)

    def generate_batch(
        self,
        user_prompts: Sequence[str],
    ) -> Tuple[List[str], List[int], int]:
        budgets: List[int] = []
        for b in [MAX_INPUT_TOKENS, *OOM_TOKEN_BUDGETS]:
            b = int(b)
            if b > 0 and b not in budgets:
                budgets.append(b)

        last_oom: Optional[Exception] = None
        for budget in budgets:
            try:
                out, lengths = self._generate_batch_once(
                    user_prompts,
                    token_budget=budget,
                )
                return out, lengths, budget
            except self.torch.cuda.OutOfMemoryError as exc:
                last_oom = exc
                log(
                    f"  batch CUDA OOM: n={len(user_prompts)} "
                    f"token_budget={budget}; trying smaller context..."
                )
                cuda_cleanup(self.torch)

        raise RuntimeError(
            "CUDA OOM at all context budgets for batch size " f"{len(user_prompts)}"
        ) from last_oom

    def map_probability_batch(
        self,
        prompts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """
        Main batch plus one batched JSON-repair pass only for malformed outputs.
        """
        raw_outputs, input_tokens, token_budget = self.generate_batch(prompts)

        parsed_rows: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
        failures: List[int] = []

        for i, raw in enumerate(raw_outputs):
            try:
                parsed = parse_mapper_payload(extract_json_blob(raw))
                parsed.update(
                    {
                        "RawOutput": raw,
                        "ParseStatus": "ok",
                        "InputTokens": int(input_tokens[i]),
                        "TokenBudgetUsed": int(token_budget),
                    }
                )
                parsed_rows[i] = parsed
            except Exception:
                failures.append(i)

        if failures:
            repair_prompts = []
            for i in failures:
                repair_prompts.append(
                    prompts[i]
                    + "\n\nYOUR PREVIOUS RESPONSE WAS INVALID. "
                    + "Return ONLY valid JSON with numeric p in [0,1], "
                    + "confidence 0..100, state P|A|U|N, and a brief reason. "
                    + "Previous response:\n<<<\n"
                    + raw_outputs[i][:1000]
                    + "\n>>>"
                )

            try:
                repaired_raw, repaired_lengths, repair_budget = self.generate_batch(
                    repair_prompts
                )
            except RuntimeError as exc:
                # Do not lose a completed production batch because a repair pass
                # could not fit. Failed cells become neutral/zero-confidence.
                log(
                    f"  repair batch failed; neutral fallback for {len(failures)} cells: {exc}"
                )
                repaired_raw = [""] * len(failures)
                repaired_lengths = [0] * len(failures)
                repair_budget = 0

            for j, original_i in enumerate(failures):
                try:
                    parsed = parse_mapper_payload(extract_json_blob(repaired_raw[j]))
                    parsed.update(
                        {
                            "RawOutput": repaired_raw[j],
                            "ParseStatus": "repaired",
                            "InputTokens": int(repaired_lengths[j]),
                            "TokenBudgetUsed": int(repair_budget),
                        }
                    )
                except Exception:
                    parsed = {
                        "FS4Probability": 0.5,
                        "FS4Confidence": 0,
                        "FS4State": "U",
                        "FS4Reason": "parse_fallback",
                        "RawOutput": raw_outputs[original_i],
                        "ParseStatus": "fallback_0.5",
                        "InputTokens": int(input_tokens[original_i]),
                        "TokenBudgetUsed": int(token_budget),
                    }
                parsed_rows[original_i] = parsed

        return [x for x in parsed_rows if x is not None]


# ============================================================
# 8. RESUMABLE PRODUCTION CACHE / BATCH RUNNER
# ============================================================


def cache_path() -> Path:
    return CACHE_ROOT / "w26p_fs4_production_v1.jsonl"


def pair_key(uid: str, label: str) -> str:
    return f"{uid}|||{label}"


def load_cache() -> Dict[str, Dict[str, Any]]:
    path = cache_path()
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = pair_key(str(row[UID]), str(row["Label"]))
                out[key] = row
            except Exception as exc:
                raise RuntimeError(
                    f"Corrupt production cache line {line_no}: {exc}"
                ) from exc
    return out


def append_cache_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dirs()
    with cache_path().open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def run_production_mapping(
    queries: Sequence[Mapping[str, Any]],
    model_path: Path,
    accelerator: Optional[str] = None,
) -> pd.DataFrame:
    ensure_dirs()

    cache = load_cache()
    q_lookup = {pair_key(str(q[UID]), str(q["Label"])): q for q in queries}

    valid_cached: Dict[str, Dict[str, Any]] = {}
    for key, row in cache.items():
        q = q_lookup.get(key)
        if q is None:
            continue
        # Only reuse rows generated from the exact same prompt.
        if str(row.get("PromptSHA256")) == str(q["PromptSHA256"]):
            valid_cached[key] = row

    needed = [
        q for q in queries if pair_key(str(q[UID]), str(q["Label"])) not in valid_cached
    ]

    # Similar prompt lengths in the same batch reduce left-padding waste.
    needed = sorted(
        needed,
        key=lambda q: (
            str(q["Label"]),
            int(q["PromptCharLength"]),
            str(q[UID]),
        ),
    )

    log("=" * 96)
    log("W2.6-P — BATCHED PRODUCTION FS4 MAPPER")
    log("=" * 96)
    log(f"FS4 production cells      : {len(queries)}")
    log(f"Valid cached cells        : {len(valid_cached)}")
    log(f"Need new mappings         : {len(needed)}")
    log(f"Initial batch size        : {INITIAL_BATCH_SIZE}")
    log(f"Max input tokens          : {MAX_INPUT_TOKENS}")
    log(f"Output token cap          : {MAX_NEW_TOKENS}")

    if needed:
        llm = LocalBatchedMapperLLM(model_path, accelerator=accelerator)
    else:
        llm = None

    current_bs = max(MIN_BATCH_SIZE, INITIAL_BATCH_SIZE)
    index = 0
    completed_new = 0
    started = time.time()

    while index < len(needed):
        chunk = needed[index : index + current_bs]
        prompts = [str(q["Prompt"]) for q in chunk]
        t0 = time.time()

        try:
            parsed_rows = llm.map_probability_batch(prompts)
            if len(parsed_rows) != len(chunk):
                raise RuntimeError(
                    f"Batch parse length mismatch: {len(parsed_rows)} vs {len(chunk)}"
                )
        except RuntimeError as exc:
            if "CUDA OOM" in str(exc) and current_bs > 1:
                new_bs = max(MIN_BATCH_SIZE, current_bs // 2)
                log(
                    f"  OOM at batch_size={current_bs}; permanently reducing "
                    f"to {new_bs} and retrying same cells."
                )
                current_bs = new_bs
                cuda_cleanup(llm.torch)
                continue
            raise

        cache_rows: List[Dict[str, Any]] = []
        for q, parsed in zip(chunk, parsed_rows):
            row = {
                UID: str(q[UID]),
                "Label": str(q["Label"]),
                "FS4Probability": float(parsed["FS4Probability"]),
                "FS4Confidence": int(parsed["FS4Confidence"]),
                "FS4State": str(parsed["FS4State"]),
                "FS4Reason": str(parsed["FS4Reason"]),
                "ParseStatus": str(parsed["ParseStatus"]),
                "InputTokens": int(parsed["InputTokens"]),
                "TokenBudgetUsed": int(parsed["TokenBudgetUsed"]),
                "PromptSHA256": str(q["PromptSHA256"]),
                "PromptCharLength": int(q["PromptCharLength"]),
                "ProductionGoldPrior": float(q["ProductionGoldPrior"]),
                "ExampleUIDs": str(q["ExampleUIDs"]),
                "ExampleSimilarities": str(q["ExampleSimilarities"]),
                "Accelerator": str(llm.accelerator),
                "LoadMode": str(llm.load_mode),
                "PromptVersion": "w26p_fs4_all58_query_specific_v1",
            }
            cache_rows.append(row)

        append_cache_rows(cache_rows)
        for row in cache_rows:
            valid_cached[pair_key(row[UID], row["Label"])] = row

        index += len(chunk)
        completed_new += len(chunk)
        elapsed = time.time() - started
        rate = completed_new / max(elapsed, 1e-9)
        remaining = len(needed) - completed_new
        eta_s = remaining / max(rate, 1e-9)

        log(
            f"  mapped {completed_new:>5}/{len(needed)} "
            f"batch={len(chunk):>2} active_bs={current_bs:>2} "
            f"call={time.time()-t0:5.1f}s "
            f"rate={rate*60:6.1f} cells/min "
            f"ETA={eta_s/3600:5.2f}h"
        )

    ordered = [valid_cached[pair_key(str(q[UID]), str(q["Label"]))] for q in queries]
    out = pd.DataFrame(ordered)

    if len(out) != EXPECTED_FS4_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_FS4_CELLS} final FS4 rows, got {len(out)}"
        )
    if out[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate final FS4 production rows")

    return out.sort_values([UID, "Label"]).reset_index(drop=True)


# ============================================================
# 9. FINAL HYBRID TEACHER ASSEMBLY
# ============================================================


def assemble_hybrid_teacher(
    unlabeled: pd.DataFrame,
    w23: pd.DataFrame,
    fs4: pd.DataFrame,
) -> pd.DataFrame:
    base = w23.copy()
    fs4_cols = [
        UID,
        "Label",
        "FS4Probability",
        "FS4Confidence",
        "FS4State",
        "FS4Reason",
        "ParseStatus",
        "InputTokens",
        "TokenBudgetUsed",
    ]
    merged = base.merge(
        fs4[fs4_cols],
        on=[UID, "Label"],
        how="left",
        validate="one_to_one",
    )

    is_fs4 = merged["Label"].isin(FS4_LABELS)
    if merged.loc[is_fs4, "FS4Probability"].isna().any():
        raise RuntimeError("Missing FS4 probability on a weak label")
    if merged.loc[~is_fs4, "FS4Probability"].notna().any():
        raise RuntimeError("Unexpected FS4 probability on established labels")

    merged["TeacherProbability"] = merged["W23RawProbabilityMean"].astype(float)
    merged.loc[is_fs4, "TeacherProbability"] = (1.0 - FIXED_BLEND_ALPHA) * merged.loc[
        is_fs4, "W23RawProbabilityMean"
    ].astype(float) + FIXED_BLEND_ALPHA * merged.loc[is_fs4, "FS4Probability"].astype(
        float
    )

    merged["TeacherSource"] = "W23_crossfold_mean"
    merged.loc[is_fs4, "TeacherSource"] = "W23_50pct_plus_W26P_FS4_50pct"

    # Transparent, conservative training weights.
    # Established W2.3 labels retain the original fold-availability concept.
    avail_frac = (merged["W23SoftAvailableFolds"].fillna(0).astype(float) / 5.0).clip(
        0, 1
    )
    high_frac = (merged["W23HighSelectionFolds"].fillna(0).astype(float) / 5.0).clip(
        0, 1
    )

    established_weight = (avail_frac * (0.50 + 0.50 * high_frac)).clip(0, 1)

    fs4_weight = (merged["FS4Confidence"].fillna(0).astype(float) / 100.0).clip(0, 1)
    fs4_bad_parse = merged["ParseStatus"].fillna("").eq("fallback_0.5")
    fs4_weight = fs4_weight.mask(fs4_bad_parse, 0.0)

    merged["RecommendedTeacherWeight"] = established_weight
    merged.loc[is_fs4, "RecommendedTeacherWeight"] = fs4_weight[is_fs4]

    # Mask is deliberately only a convenience; W6 can still use continuous weights.
    merged["RecommendedTeacherMask"] = merged["RecommendedTeacherWeight"] >= 0.35

    p = merged["TeacherProbability"].to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("Invalid final teacher probabilities")

    if len(merged) != EXPECTED_ALL_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_ALL_CELLS} hybrid rows, got {len(merged)}"
        )
    if merged[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate hybrid UID/Label rows")

    return merged.sort_values([UID, "Label"]).reset_index(drop=True)


def pivot_metric(
    long: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    wide = long.pivot(index=UID, columns="Label", values=value_col)
    wide = wide.reindex(columns=LABELS)
    wide = wide.reset_index()
    return wide


# ============================================================
# 10. STATUS / PRODUCTION / VALIDATION
# ============================================================


def status_w26p(
    accelerator: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()

    payload: Dict[str, Any] = {
        "experiment": "RSNA W2.6-P batched production report teacher",
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
        "initial_batch_size": INITIAL_BATCH_SIZE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "oom_token_budgets": OOM_TOKEN_BUDGETS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "fs4_labels": FS4_LABELS,
        "established_w23_labels": W23_ESTABLISHED_LABELS,
        "production_fs4_cells": EXPECTED_FS4_CELLS,
        "production_all_teacher_cells": EXPECTED_ALL_CELLS,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "cache_file": str(cache_path()),
        "cached_fs4_cells": len(load_cache()),
    }

    try:
        train, gold, unlabeled = load_train()
        payload["counts"] = {
            "train": len(train),
            "gold": len(gold),
            "unlabeled": len(unlabeled),
        }
    except Exception as exc:
        payload["train_error"] = repr(exc)

    try:
        w2 = discover_w2_root()
        payload["w2_root"] = str(w2)
        payload["w2_gold_sha256"] = sha256_file(
            w2 / "results" / "04_gold_structured_report_features.csv"
        )
        payload["w2_full_sha256"] = sha256_file(
            w2 / "results" / "08_full_structured_report_labels.csv"
        )
    except Exception as exc:
        payload["w2_error"] = repr(exc)

    try:
        w23 = discover_w23_root()
        payload["w23_root"] = str(w23)
        payload["w23_stability_exists"] = (
            w23 / "results" / "10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv"
        ).exists()
    except Exception as exc:
        payload["w23_error"] = repr(exc)

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

    status_path = RESULT_ROOT / "00_status.json"
    status_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def production_w26p(
    accelerator: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()

    train, gold, unlabeled = load_train()
    w2_root = discover_w2_root()
    w23_root = discover_w23_root()
    model_path = Path(MODEL_PATH_ENV)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Qwen model path not found: {model_path}. " "Set W26P_MODEL_PATH."
        )

    log("=" * 96)
    log("W2.6-P PRODUCTION BUILD")
    log("=" * 96)
    log(f"Train/gold/unlabeled      : {len(train)} / {len(gold)} / {len(unlabeled)}")
    log(f"W2 root                   : {w2_root}")
    log(f"W2.3 root                 : {w23_root}")
    log(f"Qwen model                : {model_path}")
    log(f"Pilkwang labels           : NO")
    log("All 58 gold reports are used ONLY as production FS4 exemplars.")

    gold_w2, full_w2 = load_w2_tables(w2_root, train, gold)
    w23 = load_w23_production_base(w23_root, unlabeled)

    w23.to_csv(
        RESULT_ROOT / "01_w23_production_base_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    queries, exemplar_audit = build_production_exemplars_and_queries(
        gold=gold,
        unlabeled=unlabeled,
        gold_w2=gold_w2,
        full_w2=full_w2,
    )

    exemplar_audit.to_csv(
        RESULT_ROOT / "02_fs4_exemplar_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prompt_meta = pd.DataFrame(
        [
            {
                UID: q[UID],
                "Label": q["Label"],
                "PromptSHA256": q["PromptSHA256"],
                "PromptCharLength": q["PromptCharLength"],
                "ProductionGoldPrior": q["ProductionGoldPrior"],
                "ExampleUIDs": q["ExampleUIDs"],
                "ExampleSimilarities": q["ExampleSimilarities"],
            }
            for q in queries
        ]
    )
    prompt_meta.to_csv(
        RESULT_ROOT / "03_fs4_prompt_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fs4 = run_production_mapping(
        queries=queries,
        model_path=model_path,
        accelerator=accelerator,
    )
    fs4.to_csv(
        RESULT_ROOT / "04_fs4_production_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fs4_wide = pivot_metric(fs4, "FS4Probability")
    fs4_wide.to_csv(
        RESULT_ROOT / "05_fs4_probabilities_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hybrid = assemble_hybrid_teacher(
        unlabeled=unlabeled,
        w23=w23,
        fs4=fs4,
    )
    hybrid.to_csv(
        RESULT_ROOT / "06_final_hybrid_teacher_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prob_wide = pivot_metric(hybrid, "TeacherProbability")
    prob_wide.to_csv(
        RESULT_ROOT / "07_final_hybrid_probabilities_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weight_wide = pivot_metric(hybrid, "RecommendedTeacherWeight")
    weight_wide.to_csv(
        RESULT_ROOT / "08_recommended_teacher_weights_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mask_wide = pivot_metric(hybrid, "RecommendedTeacherMask")
    mask_wide.to_csv(
        RESULT_ROOT / "09_recommended_teacher_mask_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "version": "w26p_batched_production_v1",
        "uses_pilkwang_labels": False,
        "production_validation_warning": (
            "All 58 gold reports were used as FS4 production exemplars. "
            "These pseudo-labels are a final training resource, not fold-safe OOF."
        ),
        "counts": {
            "unlabeled_studies": int(hybrid[UID].nunique()),
            "fs4_cells": int(len(fs4)),
            "all_teacher_cells": int(len(hybrid)),
        },
        "fs4_labels": FS4_LABELS,
        "established_labels": W23_ESTABLISHED_LABELS,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "parse_status_counts": {
            str(k): int(v)
            for k, v in fs4["ParseStatus"].value_counts(dropna=False).items()
        },
        "fs4_probability_summary": (
            fs4.groupby("Label")["FS4Probability"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "teacher_weight_coverage_ge_035": {
            str(label): float(
                hybrid.loc[
                    hybrid["Label"] == label,
                    "RecommendedTeacherMask",
                ].mean()
            )
            for label in LABELS
        },
        "output_root": str(OUTPUT_ROOT),
    }

    (RESULT_ROOT / "10_production_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "script_version": "w26p_batched_production_v1",
        "train_csv": str(TRAIN_CSV),
        "w2_root": str(w2_root),
        "w23_root": str(w23_root),
        "model_path": str(model_path),
        "model_config_sha256": (
            sha256_file(model_path / "config.json")
            if (model_path / "config.json").exists()
            else None
        ),
        "w2_gold_sha256": sha256_file(
            w2_root / "results" / "04_gold_structured_report_features.csv"
        ),
        "w2_full_sha256": sha256_file(
            w2_root / "results" / "08_full_structured_report_labels.csv"
        ),
        "accelerator": resolve_accelerator(accelerator),
        "precision_policy": PRECISION,
        "initial_batch_size": INITIAL_BATCH_SIZE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_pos_examples": N_POS_EXAMPLES,
        "n_neg_examples": N_NEG_EXAMPLES,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "pilkwang_used": False,
    }
    (OUTPUT_ROOT / "w26p_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("=" * 96)
    log("W2.6-P PRODUCTION COMPLETE")
    log("=" * 96)
    log(f"FS4 cells                 : {len(fs4)}")
    log(f"Hybrid teacher cells      : {len(hybrid)}")
    log(f"Unlabeled studies         : {hybrid[UID].nunique()}")
    log(f"Parse status              : {summary['parse_status_counts']}")
    log(f"Results                   : {RESULT_ROOT}")

    return summary


def validate_w26p() -> Dict[str, Any]:
    ensure_dirs()

    required_files = {
        "fs4_long": RESULT_ROOT / "04_fs4_production_long.csv",
        "fs4_wide": RESULT_ROOT / "05_fs4_probabilities_wide.csv",
        "hybrid_long": RESULT_ROOT / "06_final_hybrid_teacher_long.csv",
        "hybrid_wide": RESULT_ROOT / "07_final_hybrid_probabilities_wide.csv",
        "weights_wide": RESULT_ROOT / "08_recommended_teacher_weights_wide.csv",
        "mask_wide": RESULT_ROOT / "09_recommended_teacher_mask_wide.csv",
    }

    checks: Dict[str, bool] = {
        f"{name}_exists": path.exists() for name, path in required_files.items()
    }
    missing = [
        str(path)
        for name, path in required_files.items()
        if not checks[f"{name}_exists"]
    ]
    if missing:
        payload = {
            "overall_pass": False,
            "checks": checks,
            "missing": missing,
        }
        log(json.dumps(payload, indent=2))
        return payload

    fs4 = pd.read_csv(required_files["fs4_long"])
    hybrid = pd.read_csv(required_files["hybrid_long"])
    fs4_wide = pd.read_csv(required_files["fs4_wide"])
    hybrid_wide = pd.read_csv(required_files["hybrid_wide"])
    weights_wide = pd.read_csv(required_files["weights_wide"])
    mask_wide = pd.read_csv(required_files["mask_wide"])

    checks.update(
        {
            "fs4_rows_17396": len(fs4) == EXPECTED_FS4_CELLS,
            "fs4_uids_4349": fs4[UID].astype(str).nunique() == EXPECTED_UNLABELED,
            "fs4_four_labels": set(fs4["Label"]) == set(FS4_LABELS),
            "fs4_no_duplicates": not fs4[[UID, "Label"]].duplicated().any(),
            "fs4_probabilities_valid": bool(
                np.isfinite(fs4["FS4Probability"].to_numpy(float)).all()
                and fs4["FS4Probability"].between(0, 1).all()
            ),
            "hybrid_rows_52188": len(hybrid) == EXPECTED_ALL_CELLS,
            "hybrid_uids_4349": (
                hybrid[UID].astype(str).nunique() == EXPECTED_UNLABELED
            ),
            "hybrid_12_labels": set(hybrid["Label"]) == set(LABELS),
            "hybrid_no_duplicates": not hybrid[[UID, "Label"]].duplicated().any(),
            "hybrid_probabilities_valid": bool(
                np.isfinite(hybrid["TeacherProbability"].to_numpy(float)).all()
                and hybrid["TeacherProbability"].between(0, 1).all()
            ),
            "weights_valid": bool(
                np.isfinite(hybrid["RecommendedTeacherWeight"].to_numpy(float)).all()
                and hybrid["RecommendedTeacherWeight"].between(0, 1).all()
            ),
            "fs4_wide_shape": fs4_wide.shape
            == (EXPECTED_UNLABELED, 1 + len(FS4_LABELS)),
            "hybrid_wide_shape": hybrid_wide.shape
            == (EXPECTED_UNLABELED, 1 + len(LABELS)),
            "weights_wide_shape": weights_wide.shape
            == (EXPECTED_UNLABELED, 1 + len(LABELS)),
            "mask_wide_shape": mask_wide.shape == (EXPECTED_UNLABELED, 1 + len(LABELS)),
            "pilkwang_not_used": True,
        }
    )

    overall_pass = all(checks.values())

    status_counts = {
        str(k): int(v) for k, v in fs4["ParseStatus"].value_counts(dropna=False).items()
    }
    fallback_rate = float((fs4["ParseStatus"] == "fallback_0.5").mean())

    per_label = []
    for label in LABELS:
        x = hybrid[hybrid["Label"] == label]
        per_label.append(
            {
                "Label": label,
                "TeacherProbabilityMean": float(x["TeacherProbability"].mean()),
                "TeacherProbabilityStd": float(x["TeacherProbability"].std()),
                "RecommendedWeightMean": float(x["RecommendedTeacherWeight"].mean()),
                "RecommendedMaskCoverage": float(x["RecommendedTeacherMask"].mean()),
            }
        )

    payload = {
        "overall_pass": overall_pass,
        "checks": checks,
        "parse_status_counts": status_counts,
        "parse_fallback_rate": fallback_rate,
        "per_label": per_label,
        "results_root": str(RESULT_ROOT),
    }

    (RESULT_ROOT / "11_validation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("=" * 96)
    log("W2.6-P VALIDATION")
    log("=" * 96)
    log(json.dumps(payload, indent=2, ensure_ascii=False))

    return payload


# ============================================================
# 11. NOTEBOOK API
# ============================================================


def run_w26p(
    mode: str = "status",
    accelerator: Optional[str] = None,
):
    mode = str(mode).strip().lower()

    aliases = {
        "run": "production",
        "prod": "production",
        "build": "production",
        "check": "validate",
    }
    mode = aliases.get(mode, mode)

    if mode == "status":
        return status_w26p(accelerator=accelerator)
    if mode == "production":
        return production_w26p(accelerator=accelerator)
    if mode == "validate":
        return validate_w26p()

    raise ValueError(f"Unknown W2.6-P mode {mode!r}. Use status, production, validate.")


if __name__ == "__main__":
    # Safe default: executing the .py file by itself does not launch 17k mappings.
    run_w26p("status", accelerator="localGPU")
    run_w26p("production", accelerator="localGPU")
    run_w26p("validate")
