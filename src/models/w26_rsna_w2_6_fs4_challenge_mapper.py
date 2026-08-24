#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA Knee Abnormality Detection — W2.6 FS4
OUR OWN Fold-Safe Few-Shot Challenge Mapper
============================================

Purpose
-------
W2.5 showed that a generic report extractor adds complementary signal but is
not strong enough to mass-label all 4,349 reports. W2.6 therefore attacks the
four labels that our pre-existing W2.3 gate could not pseudo-label reliably:

    Medial OA, Lateral OA, PF OA, Synovitis

For every held-out gold study, W2.6 retrieves 2 positive and 2 negative
examples ONLY from that study's outer-training fold, then asks our local
Qwen2.5-7B-Instruct model to map the report to the competition label.

No Pilkwang labels are used.
No held-out gold label enters its prompt.
No W2.5 pseudo-target is used as input.
The other 8 labels remain exactly W2.3 for attribution.

Notebook API
------------
    run_w26("status")
    run_w26("gold")
    run_w26("validate")

Recommended: execute the script, inspect status, then run gold directly.
This gold gate is only 58 studies x 4 labels = 232 primary LLM calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# ============================================================
# 1. CONFIG
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

FS4_LABELS = ["Medial OA", "Lateral OA", "PF OA", "Synovitis"]
KEEP_W23_LABELS = [x for x in LABELS if x not in FS4_LABELS]

EXPECTED_TOTAL = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

DATA_ROOT = Path(
    os.environ.get(
        "W26_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)
TRAIN_CSV = Path(os.environ.get("W26_TRAIN_CSV", str(DATA_ROOT / "train.csv")))

W2_ROOT_ENV = os.environ.get(
    "W26_W2_ROOT", "/kaggle/input/datasets/isayem/rsna-w2/rsna_w2"
).strip()

W23_ROOT_ENV = os.environ.get(
    "W26_W23_ROOT", "/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3"
).strip()

MODEL_PATH_ENV = os.environ.get(
    "W26_MODEL_PATH", "/kaggle/input/datasets/ragnar123/qwen2-5-7b-instruct"
).strip()

OUTPUT_ROOT = Path(os.environ.get("W26_OUTPUT_ROOT", "/kaggle/working/rsna_w2_6"))
CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULT_ROOT = OUTPUT_ROOT / "results"

USE_4BIT = os.environ.get("W26_USE_4BIT", "0").strip() == "1"
ALLOW_FP16_FALLBACK = os.environ.get("W26_ALLOW_FP16_FALLBACK", "1").strip() == "1"
FP16_SHARD_ALL_GPUS = os.environ.get("W26_FP16_SHARD_ALL_GPUS", "1").strip() == "1"
FP16_GPU_MAX_GIB = float(os.environ.get("W26_FP16_GPU_MAX_GIB", "13.0"))
GPU_ID = int(os.environ.get("W26_GPU_ID", "0"))

MAX_INPUT_TOKENS = int(os.environ.get("W26_MAX_INPUT_TOKENS", "6144"))
MAX_NEW_TOKENS = int(os.environ.get("W26_MAX_NEW_TOKENS", "160"))

N_POS_EXAMPLES = int(os.environ.get("W26_N_POS_EXAMPLES", "2"))
N_NEG_EXAMPLES = int(os.environ.get("W26_N_NEG_EXAMPLES", "2"))
MAX_EXAMPLE_REPORT_CHARS = int(os.environ.get("W26_MAX_EXAMPLE_REPORT_CHARS", "2600"))
MAX_QUERY_REPORT_CHARS = int(os.environ.get("W26_MAX_QUERY_REPORT_CHARS", "3600"))
TFIDF_MAX_FEATURES = int(os.environ.get("W26_TFIDF_MAX_FEATURES", "30000"))

BOOTSTRAP_REPEATS = int(os.environ.get("W26_BOOTSTRAP_REPEATS", "3000"))
BOOTSTRAP_SEED = int(os.environ.get("W26_BOOTSTRAP_SEED", "260824"))

# Fixed, predeclared comparison variants. No post-hoc alpha search.
FIXED_BLEND_ALPHA = float(os.environ.get("W26_FIXED_BLEND_ALPHA", "0.50"))

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


def fold_sha256(frame: pd.DataFrame) -> str:
    x = frame[[UID, "OuterFold"]].copy()
    x[UID] = x[UID].astype(str)
    x = x.sort_values(UID).reset_index(drop=True)
    payload = "".join(f"{u},{int(f)}\n" for u, f in zip(x[UID], x["OuterFold"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True, default=conv),
        encoding="utf-8",
    )


def compact_report(text: Any, max_chars: int) -> str:
    s = normalize_text(text)
    if len(s) <= max_chars:
        return s
    # Preserve both the beginning and conclusion-like tail.
    head = max_chars * 3 // 5
    tail = max_chars - head
    return s[:head] + " ... [TRUNCATED] ... " + s[-tail:]


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


def fast_binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def fast_macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.nanmean([fast_binary_auc(y[:, j], p[:, j]) for j in range(y.shape[1])])
    )


def bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values: List[float] = []
    attempts = 0
    while len(values) < repeats and attempts < repeats * 30:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if any(len(np.unique(ys[:, j])) < 2 for j in range(ys.shape[1])):
            continue
        values.append(
            fast_macro_auc(ys, candidate[idx]) - fast_macro_auc(ys, reference[idx])
        )
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_gt_0": np.nan,
        }
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "ci_low": float(np.quantile(a, 0.025)),
        "ci_high": float(np.quantile(a, 0.975)),
        "p_gt_0": float(np.mean(a > 0)),
    }


# ============================================================
# 3. INPUTS / FOLD INTEGRITY
# ============================================================


def load_train() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)
    train = pd.read_csv(TRAIN_CSV)
    train[UID] = train[UID].astype(str)
    missing = {UID, REPORT, *LABELS} - set(train.columns)
    if missing:
        raise RuntimeError(f"train.csv missing columns: {sorted(missing)}")

    gold = (
        train[train[LABELS].notna().all(axis=1)]
        .copy()
        .sort_values(UID)
        .reset_index(drop=True)
    )
    unlabeled = (
        train[train[LABELS].isna().all(axis=1)]
        .copy()
        .sort_values(UID)
        .reset_index(drop=True)
    )
    if (len(train), len(gold), len(unlabeled)) != (
        EXPECTED_TOTAL,
        EXPECTED_GOLD,
        EXPECTED_UNLABELED,
    ):
        raise RuntimeError(
            f"Unexpected counts train/gold/unlabeled={len(train)}/{len(gold)}/{len(unlabeled)}"
        )
    return train, gold, unlabeled


def resolve_root(
    raw: str, expected_files: Sequence[str], suffix: Optional[str] = None
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
            "folds/fold_1/heldout_gold_stage_b_predictions.csv",
            "folds/fold_5/heldout_gold_stage_b_predictions.csv",
        ],
        suffix="rsna_w2_3",
    )


def load_folds(w23_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_csv(w23_root / "results" / "00_outer_fold_assignments.csv")
    folds[UID] = folds[UID].astype(str)
    folds = folds[[UID, "OuterFold"]].sort_values(UID).reset_index(drop=True)
    if set(folds[UID]) != set(gold[UID]):
        raise RuntimeError("W2.3 fold UID set mismatch")
    digest = fold_sha256(folds)
    if digest != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"Fold hash mismatch: {digest}")
    return folds


def load_w2_gold(w2_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    x = pd.read_csv(w2_root / "results" / "04_gold_structured_report_features.csv")
    x[UID] = x[UID].astype(str)
    needed = {UID, "Label", *W2_PROMPT_COLUMNS}
    missing = needed - set(x.columns)
    if missing:
        raise RuntimeError(f"W2 gold structured file missing: {sorted(missing)}")
    if len(x) != EXPECTED_GOLD * len(LABELS) or x[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Invalid W2 gold structured table")
    if set(x[UID]) != set(gold[UID]):
        raise RuntimeError("W2 gold UID mismatch")
    return x


def load_w23_oof(w23_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for fold in range(1, 6):
        p = w23_root / "folds" / f"fold_{fold}" / "heldout_gold_stage_b_predictions.csv"
        x = pd.read_csv(p)
        x[UID] = x[UID].astype(str)
        req = {UID, "Label", "Gold", "FoldSafeChallengeProbability"}
        if req - set(x.columns):
            raise RuntimeError(f"{p} missing {sorted(req - set(x.columns))}")
        x = x[[UID, "Label", "Gold", "FoldSafeChallengeProbability"]].copy()
        x["OuterFold"] = fold
        pieces.append(x)
    out = pd.concat(pieces, ignore_index=True)
    if (
        len(out) != EXPECTED_GOLD * len(LABELS)
        or out[[UID, "Label"]].duplicated().any()
    ):
        raise RuntimeError("Invalid W2.3 OOF table")
    if set(out[UID]) != set(gold[UID]):
        raise RuntimeError("W2.3 OOF UID mismatch")
    return out


# ============================================================
# 4. W2 FEATURE SUMMARIES FOR PROMPTS / RETRIEVAL
# ============================================================


def clean_scalar(v: Any, digits: int = 3) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


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
        if evidence:
            # dedupe while preserving order
            seen, unique = set(), []
            for e in evidence:
                key = normalize_text(e).casefold()
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            parts.append("evidence=" + " || ".join(unique[:3]))
    return "; ".join(parts)


def build_feature_lookup(w2_gold: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
    return {(str(r[UID]), str(r["Label"])): r for _, r in w2_gold.iterrows()}


# ============================================================
# 5. FOLD-SAFE EXEMPLAR RETRIEVAL
# ============================================================


def retrieval_document(report: str, feature_summary: str, label: str) -> str:
    return (
        f"target {label} target {label} "
        f"{normalize_text(feature_summary)} "
        f"{normalize_text(report)}"
    )


def select_exemplars_for_fold_label(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_lookup: Dict[Tuple[str, str], pd.Series],
    fold: int,
    label: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()
    train_df = gold[[UID, REPORT, label]].copy()
    train_df["OuterFold"] = train_df[UID].map(fold_map).astype(int)
    outer_train = (
        train_df[train_df["OuterFold"] != fold]
        .copy()
        .sort_values(UID)
        .reset_index(drop=True)
    )
    heldout = (
        train_df[train_df["OuterFold"] == fold]
        .copy()
        .sort_values(UID)
        .reset_index(drop=True)
    )

    if outer_train[label].nunique() < 2:
        raise RuntimeError(f"{label}/fold{fold}: outer train has one class")
    if int((outer_train[label] == 1).sum()) < N_POS_EXAMPLES:
        raise RuntimeError(f"{label}/fold{fold}: insufficient positives")
    if int((outer_train[label] == 0).sum()) < N_NEG_EXAMPLES:
        raise RuntimeError(f"{label}/fold{fold}: insufficient negatives")

    train_docs = []
    for _, r in outer_train.iterrows():
        uid = str(r[UID])
        fs = w2_row_summary(w2_lookup[(uid, label)], include_evidence=True)
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
    audit_rows: List[Dict[str, Any]] = []

    for _, q in heldout.iterrows():
        q_uid = str(q[UID])
        q_fs = w2_row_summary(w2_lookup[(q_uid, label)], include_evidence=True)
        q_doc = retrieval_document(str(q[REPORT]), q_fs, label)
        q_vec = vectorizer.transform([q_doc])
        sims = (X_train @ q_vec.T).toarray().ravel()

        tmp = outer_train[[UID, REPORT, label]].copy()
        tmp["Similarity"] = sims
        tmp = tmp.sort_values(["Similarity", UID], ascending=[False, True]).reset_index(
            drop=True
        )

        pos = tmp[tmp[label] == 1].head(N_POS_EXAMPLES)
        neg = tmp[tmp[label] == 0].head(N_NEG_EXAMPLES)
        chosen = pd.concat([pos, neg], ignore_index=True)
        # Stable presentation: highest similarity first, while preserving labels in audit.
        chosen = chosen.sort_values(
            ["Similarity", UID], ascending=[False, True]
        ).reset_index(drop=True)

        exs: List[Dict[str, Any]] = []
        for rank, (_, ex) in enumerate(chosen.iterrows(), start=1):
            ex_uid = str(ex[UID])
            ex_fs = w2_row_summary(w2_lookup[(ex_uid, label)], include_evidence=True)
            item = {
                "StudyInstanceUID": ex_uid,
                "Gold": int(ex[label]),
                "Similarity": float(ex["Similarity"]),
                "Report": str(ex[REPORT]),
                "W2Summary": ex_fs,
            }
            exs.append(item)
            audit_rows.append(
                {
                    "OuterFold": fold,
                    "QueryStudyInstanceUID": q_uid,
                    "Label": label,
                    "ExampleRank": rank,
                    "ExampleStudyInstanceUID": ex_uid,
                    "ExampleGold": int(ex[label]),
                    "Similarity": float(ex["Similarity"]),
                }
            )
        selected[q_uid] = exs

    return selected, pd.DataFrame(audit_rows)


def build_all_exemplars(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_gold: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], pd.DataFrame]:
    lookup = build_feature_lookup(w2_gold)
    all_selected: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    audits = []
    for fold in range(1, 6):
        for label in FS4_LABELS:
            selected, audit = select_exemplars_for_fold_label(
                gold, folds, lookup, fold, label
            )
            for uid, exs in selected.items():
                all_selected[(uid, label)] = exs
            audits.append(audit)
    out = pd.concat(audits, ignore_index=True)
    if len(all_selected) != EXPECTED_GOLD * len(FS4_LABELS):
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD * len(FS4_LABELS)} query/label exemplar sets, got {len(all_selected)}"
        )
    return all_selected, out


# ============================================================
# 6. CHALLENGE-MAPPER PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a competition-label mapper for knee MRI reports.
Your task is NOT merely to decide whether a finding is literally mentioned.
You must infer the binary annotation convention of THIS challenge from the
provided labeled examples, then estimate the probability that the query study
has challenge label 1.

Important:
- Examples come only from the training portion of the current outer fold.
- The query gold label is unavailable to you.
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
    outer_train_prior: float,
) -> str:
    chunks = [
        f"TARGET LABEL: {label}",
        f"TARGET DEFINITION: {TARGET_DEFINITIONS[label]}",
        f"OUTER-TRAIN LABEL-1 FRACTION: {outer_train_prior:.3f}",
        "",
        "FOLD-SAFE LABELED EXAMPLES:",
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
        "--- QUERY ---",
        f"W2ReportFeatures: {query_w2}",
        "Report:",
        compact_report(query_report, MAX_QUERY_REPORT_CHARS),
        "",
        "OUTPUT CONTRACT:",
        'Return exactly one JSON object: {"p": 0.000..1.000, "confidence": 0..100, "state": "P|A|U|N", "reason": "brief"}',
        "p is the probability of THIS CHALLENGE LABEL being 1, after considering the supplied challenge examples.",
        "state is only a compact report-side interpretation: P present, A explicitly absent, U uncertain/indirect, N not addressed.",
        "Use a genuinely graded p; do not restrict yourself to a small fixed set of probability values.",
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

    c_raw = payload.get("confidence", 50)
    try:
        c = int(round(float(c_raw)))
    except Exception:
        c = 50
    c = int(np.clip(c, 0, 100))

    state = str(payload.get("state", "U")).strip().upper()
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

    return {
        "Probability": float(p),
        "Confidence": c,
        "State": state,
        "Reason": str(payload.get("reason", "") or "").strip()[:800],
    }


# ============================================================
# 7. LOCAL QWEN LOADER — PROVEN 2xT4 PATH
# ============================================================


class LocalMapperLLM:
    def __init__(self, model_path: Path):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:
            raise RuntimeError("W2.6 requires torch + transformers") from exc

        self.torch = torch
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        log(f"Loading mapper LLM       : {self.model_path}")
        log(f"CUDA GPUs                : {gpu_count}")
        log(f"4-bit requested          : {USE_4BIT}")

        if torch.cuda.is_available():
            if USE_4BIT:
                try:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    kwargs["device_map"] = {"": GPU_ID}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path), **kwargs
                    )
                    self.load_mode = f"4bit_nf4_single_gpu_{GPU_ID}"
                except Exception as exc:
                    if not ALLOW_FP16_FALLBACK:
                        raise
                    kwargs.pop("quantization_config", None)
                    if gpu_count >= 2 and FP16_SHARD_ALL_GPUS:
                        kwargs["dtype"] = torch.float16
                        kwargs["device_map"] = "balanced"
                        kwargs["max_memory"] = {
                            i: f"{FP16_GPU_MAX_GIB:.1f}GiB" for i in range(gpu_count)
                        }
                        warnings.warn(
                            f"4-bit failed; using sharded FP16. Original: {exc}"
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            str(self.model_path), **kwargs
                        )
                        self.load_mode = f"fp16_balanced_{gpu_count}gpu"
                    else:
                        raise RuntimeError(
                            "4-bit failed and safe sharded FP16 unavailable"
                        ) from exc
            else:
                kwargs["dtype"] = torch.float16
                if gpu_count >= 2 and FP16_SHARD_ALL_GPUS:
                    kwargs["device_map"] = "balanced"
                    kwargs["max_memory"] = {
                        i: f"{FP16_GPU_MAX_GIB:.1f}GiB" for i in range(gpu_count)
                    }
                    self.load_mode = f"fp16_balanced_{gpu_count}gpu"
                else:
                    kwargs["device_map"] = {"": GPU_ID}
                    self.load_mode = f"fp16_single_gpu_{GPU_ID}"
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), **kwargs
                )
        else:
            kwargs["dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), **kwargs
            )
            self.load_mode = "fp32_cpu"

        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        log(f"Model load mode          : {self.load_mode}")
        log(f"Input embedding device   : {self.input_device}")
        if hasattr(self.model, "hf_device_map"):
            counts: Dict[str, int] = {}
            for dev in self.model.hf_device_map.values():
                counts[str(dev)] = counts.get(str(dev), 0) + 1
            log(f"HF device-map modules    : {counts}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(i)
                log(
                    f"GPU {i} free after load   : {free_b/(1024**3):.2f} / {total_b/(1024**3):.2f} GiB"
                )

    def _chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nJSON:"

    def generate_text(self, user_prompt: str) -> str:
        torch = self.torch
        prompt = self._chat_prompt(user_prompt)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}
        input_len = int(encoded["input_ids"].shape[1])
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
        return self.tokenizer.decode(
            generated[0, input_len:], skip_special_tokens=True
        ).strip()

    def map_probability(self, user_prompt: str) -> Tuple[Dict[str, Any], str, str]:
        raw = ""
        try:
            raw = self.generate_text(user_prompt)
            parsed = parse_mapper_payload(extract_json_blob(raw))
            return parsed, raw, "ok"
        except self.torch.cuda.OutOfMemoryError as exc:
            self.torch.cuda.empty_cache()
            raise RuntimeError("CUDA OOM during W2.6 mapping") from exc
        except Exception as first_exc:
            # One syntax/contract repair, then nonfatal 0.5 fallback.
            repair = (
                user_prompt
                + "\n\nYOUR PREVIOUS RESPONSE WAS INVALID. "
                + f"Error: {first_exc}. Return only valid JSON with numeric p in [0,1]. "
                + "Previous response:\n<<<\n"
                + raw[:1800]
                + "\n>>>"
            )
            try:
                raw2 = self.generate_text(repair)
                parsed = parse_mapper_payload(extract_json_blob(raw2))
                return parsed, raw2, "repaired"
            except Exception as second_exc:
                log(f"  mapper parse fallback to 0.5: {second_exc}")
                return (
                    {
                        "Probability": 0.5,
                        "Confidence": 0,
                        "State": "U",
                        "Reason": f"parse_fallback: {second_exc}",
                    },
                    raw,
                    "fallback_0.5",
                )


# ============================================================
# 8. RESUMABLE CACHE
# ============================================================


def cache_path() -> Path:
    return CACHE_ROOT / "w26_fs4_gold_oof_v1.jsonl"


def cache_key(uid: str, label: str) -> str:
    return f"{uid}|||{label}"


def load_cache() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    p = cache_path()
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                out[cache_key(str(row[UID]), str(row["Label"]))] = row
            except Exception as exc:
                raise RuntimeError(f"Corrupt cache line {line_no}: {exc}") from exc
    return out


def append_cache(row: Mapping[str, Any]) -> None:
    ensure_dirs()
    with cache_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# 9. GOLD GATE
# ============================================================


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

    for uid in gold[UID].astype(str):
        fold = int(fold_map[uid])
        for label in FS4_LABELS:
            outer_train_uids = [
                u for u in gold[UID].astype(str) if int(fold_map[u]) != fold
            ]
            prior = float(gold_lookup.loc[outer_train_uids, label].astype(float).mean())
            q_w2 = w2_row_summary(w2_lookup[(uid, label)], include_evidence=True)
            prompt = build_mapper_prompt(
                label=label,
                query_report=str(gold_lookup.at[uid, REPORT]),
                query_w2=q_w2,
                examples=exemplars[(uid, label)],
                outer_train_prior=prior,
            )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            queries.append(
                {
                    UID: uid,
                    "Label": label,
                    "Gold": int(gold_lookup.at[uid, label]),
                    "OuterFold": fold,
                    "OuterTrainPrior": prior,
                    "Prompt": prompt,
                    "PromptSHA256": prompt_hash,
                }
            )
    return queries


def run_gold_mapping(
    queries: Sequence[Mapping[str, Any]],
    model_path: Path,
) -> pd.DataFrame:
    ensure_dirs()
    cache = load_cache()

    needed = []
    stale = 0
    for q in queries:
        key = cache_key(str(q[UID]), str(q["Label"]))
        if key not in cache:
            needed.append(q)
        elif str(cache[key].get("PromptSHA256", "")) != str(q["PromptSHA256"]):
            # Stale cache is never silently reused.
            stale += 1
            needed.append(q)

    if stale:
        raise RuntimeError(
            f"Found {stale} stale cache entries with different prompt hashes. "
            f"Delete {cache_path()} before rerunning this script version."
        )

    log(f"Gold FS4 query-label cells : {len(queries)}")
    log(f"Already cached             : {len(queries) - len(needed)}")
    log(f"Need mapper calls          : {len(needed)}")

    llm = LocalMapperLLM(model_path) if needed else None
    start = time.time()

    for i, q in enumerate(needed, start=1):
        t0 = time.time()
        parsed, raw, parse_status = llm.map_probability(str(q["Prompt"]))
        row = {
            UID: str(q[UID]),
            "Label": str(q["Label"]),
            "Gold": int(q["Gold"]),
            "OuterFold": int(q["OuterFold"]),
            "OuterTrainPrior": float(q["OuterTrainPrior"]),
            "Probability": float(parsed["Probability"]),
            "Confidence": int(parsed["Confidence"]),
            "State": str(parsed["State"]),
            "Reason": str(parsed["Reason"]),
            "ParseStatus": parse_status,
            "PromptSHA256": str(q["PromptSHA256"]),
            "RawOutput": raw,
            "ModelPath": str(model_path),
            "PromptVersion": "w26_fs4_fold_safe_fewshot_v1",
        }
        append_cache(row)
        cache[cache_key(row[UID], row["Label"])] = row

        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-9)
        eta = (len(needed) - i) / max(rate, 1e-9)
        log(
            f"  mapped {i:>3}/{len(needed)} fold={int(q['OuterFold'])} "
            f"target={str(q['Label']):<10s} study={str(q[UID])[-10:]} "
            f"p={float(parsed['Probability']):.3f} call={time.time()-t0:4.1f}s ETA={eta/60:4.1f}m"
        )

    rows = [cache[cache_key(str(q[UID]), str(q["Label"]))] for q in queries]
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_GOLD * len(FS4_LABELS):
        raise RuntimeError(
            f"W2.6 mapper rows={len(out)}, expected={EXPECTED_GOLD * len(FS4_LABELS)}"
        )
    if out[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate W2.6 UID/Label")
    return out


def build_variant_oof(
    gold: pd.DataFrame,
    w23_oof: pd.DataFrame,
    fs4_oof: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    base = w23_oof[
        [UID, "Label", "Gold", "OuterFold", "FoldSafeChallengeProbability"]
    ].copy()
    base = base.rename(columns={"FoldSafeChallengeProbability": "W23Probability"})
    fs = fs4_oof[[UID, "Label", "Probability"]].rename(
        columns={"Probability": "FS4Probability"}
    )
    x = base.merge(fs, on=[UID, "Label"], how="left", validate="one_to_one")

    if variant == "w23":
        x["Probability"] = x["W23Probability"].astype(float)
    elif variant == "fs4_replace":
        x["Probability"] = x["W23Probability"].astype(float)
        mask = x["Label"].isin(FS4_LABELS)
        x.loc[mask, "Probability"] = x.loc[mask, "FS4Probability"].astype(float)
    elif variant == "fs4_fixed50":
        x["Probability"] = x["W23Probability"].astype(float)
        mask = x["Label"].isin(FS4_LABELS)
        a = FIXED_BLEND_ALPHA
        x.loc[mask, "Probability"] = (1.0 - a) * x.loc[mask, "W23Probability"].astype(
            float
        ) + a * x.loc[mask, "FS4Probability"].astype(float)
    else:
        raise ValueError(variant)

    x["Variant"] = variant
    return x[[UID, "Label", "Gold", "OuterFold", "Variant", "Probability"]]


def metrics_for_variant(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        x = oof[oof["Label"] == label]
        y = x["Gold"].to_numpy(dtype=int)
        p = x["Probability"].to_numpy(dtype=float)
        prior = float(y.mean())
        rows.append(
            {
                "Label": label,
                "N": len(x),
                "Positive": int(y.sum()),
                "Negative": int(len(y) - y.sum()),
                "AUROC": safe_auc(y, p),
                "AP": safe_ap(y, p),
                "Brier": float(brier_score_loss(y, np.clip(p, 1e-5, 1 - 1e-5))),
                "PriorBrier": float(brier_score_loss(y, np.full(len(y), prior))),
            }
        )
    return pd.DataFrame(rows)


def matrix_from_oof(oof: pd.DataFrame, gold: pd.DataFrame) -> np.ndarray:
    wide = oof.pivot(index=UID, columns="Label", values="Probability")
    wide = wide.reindex(index=gold[UID].astype(str), columns=LABELS)
    return wide.to_numpy(dtype=float)


def evaluate_all(
    gold: pd.DataFrame,
    variants: Sequence[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metric_parts = []
    matrices: Dict[str, np.ndarray] = {}
    y = gold[LABELS].to_numpy(dtype=int)

    for oof in variants:
        name = str(oof["Variant"].iloc[0])
        m = metrics_for_variant(oof)
        m.insert(0, "Variant", name)
        metric_parts.append(m)
        matrices[name] = matrix_from_oof(oof, gold)

    metrics = pd.concat(metric_parts, ignore_index=True)
    summary: Dict[str, Any] = {"variants": {}}
    for name in matrices:
        m = metrics[metrics["Variant"] == name]
        summary["variants"][name] = {
            "macro_AUROC": float(np.nanmean(m["AUROC"])),
            "macro_AP": float(np.nanmean(m["AP"])),
            "macro_Brier": float(np.nanmean(m["Brier"])),
        }

    for name in ["fs4_replace", "fs4_fixed50"]:
        summary[f"{name}_minus_w23_bootstrap"] = bootstrap_delta(
            y, matrices[name], matrices["w23"]
        )

    # Gate is based on the predeclared fixed-50 variant, not whichever score wins.
    ref = summary["variants"]["w23"]["macro_AUROC"]
    cand = summary["variants"]["fs4_fixed50"]["macro_AUROC"]
    delta = cand - ref
    summary["primary_variant"] = "fs4_fixed50"
    summary["primary_delta_vs_w23"] = float(delta)
    if cand >= 0.82 and delta >= 0.03:
        verdict = "STRONG_PROCEED_TO_PRODUCTION_FS4"
    elif delta >= 0.015:
        verdict = "PROMISING_ONE_MORE_REVIEW"
    else:
        verdict = "STOP_FS4_AND_MOVE_ON"
    summary["gate_verdict"] = verdict
    return metrics, summary


def gold_w26() -> Dict[str, Any]:
    ensure_dirs()
    train, gold, _ = load_train()
    w2_root = get_w2_root()
    w23_root = get_w23_root()
    folds = load_folds(w23_root, gold)
    w2_gold = load_w2_gold(w2_root, gold)
    w23_oof = load_w23_oof(w23_root, gold)

    model_path = Path(MODEL_PATH_ENV)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    log("=" * 96)
    log("RSNA W2.6 FS4 — FOLD-SAFE FEW-SHOT CHALLENGE MAPPER")
    log("=" * 96)
    log(f"Train/gold               : {len(train)} / {len(gold)}")
    log(f"Weak labels              : {FS4_LABELS}")
    log(f"Examples per query       : +{N_POS_EXAMPLES} / -{N_NEG_EXAMPLES}")
    log(f"Fold SHA256              : {fold_sha256(folds)}")
    log(f"Pilkwang labels used     : NO")

    exemplars, exemplar_audit = build_all_exemplars(gold, folds, w2_gold)
    exemplar_audit.to_csv(RESULT_ROOT / "01_exemplar_audit.csv", index=False)
    folds.to_csv(RESULT_ROOT / "00_outer_fold_assignments.csv", index=False)

    queries = build_gold_queries(gold, folds, w2_gold, exemplars)
    # Persist prompt metadata but not full reports duplicated in a giant CSV.
    pd.DataFrame(
        [
            {
                UID: q[UID],
                "Label": q["Label"],
                "Gold": q["Gold"],
                "OuterFold": q["OuterFold"],
                "OuterTrainPrior": q["OuterTrainPrior"],
                "PromptSHA256": q["PromptSHA256"],
                "PromptChars": len(q["Prompt"]),
            }
            for q in queries
        ]
    ).to_csv(RESULT_ROOT / "02_query_prompt_audit.csv", index=False)

    fs4 = run_gold_mapping(queries, model_path)
    fs4.to_csv(
        RESULT_ROOT / "03_fs4_mapper_oof_long.csv", index=False, encoding="utf-8-sig"
    )

    variants = [
        build_variant_oof(gold, w23_oof, fs4, "w23"),
        build_variant_oof(gold, w23_oof, fs4, "fs4_replace"),
        build_variant_oof(gold, w23_oof, fs4, "fs4_fixed50"),
    ]
    all_oof = pd.concat(variants, ignore_index=True)
    all_oof.to_csv(RESULT_ROOT / "04_all_variants_oof_long.csv", index=False)

    metrics, summary = evaluate_all(gold, variants)
    metrics.to_csv(RESULT_ROOT / "05_metrics_per_label_variant.csv", index=False)

    # Side-by-side weak-label attribution.
    weak_rows = []
    piv = metrics.pivot(index="Label", columns="Variant", values="AUROC")
    for label in FS4_LABELS:
        weak_rows.append(
            {
                "Label": label,
                "W23_AUROC": float(piv.at[label, "w23"]),
                "FS4_Replace_AUROC": float(piv.at[label, "fs4_replace"]),
                "FS4_Fixed50_AUROC": float(piv.at[label, "fs4_fixed50"]),
                "Replace_Delta": float(
                    piv.at[label, "fs4_replace"] - piv.at[label, "w23"]
                ),
                "Fixed50_Delta": float(
                    piv.at[label, "fs4_fixed50"] - piv.at[label, "w23"]
                ),
            }
        )
    pd.DataFrame(weak_rows).to_csv(
        RESULT_ROOT / "06_fs4_per_label_attribution.csv", index=False
    )

    write_json(RESULT_ROOT / "07_w26_gold_summary.json", summary)

    manifest = {
        "experiment": "RSNA W2.6 FS4 fold-safe few-shot challenge mapper",
        "version": "w26_fs4_fold_safe_fewshot_v1",
        "uses_pilkwang_labels": False,
        "fs4_labels": FS4_LABELS,
        "kept_w23_labels": KEEP_W23_LABELS,
        "n_pos_examples": N_POS_EXAMPLES,
        "n_neg_examples": N_NEG_EXAMPLES,
        "retrieval": "outer-train-only char_wb TF-IDF 3-5 grams + W2 report features",
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "fold_sha256": fold_sha256(folds),
        "fold_sha256_match": fold_sha256(folds) == EXPECTED_FOLD_SHA256,
        "model_path": str(model_path),
        "model_config_sha256": (
            sha256_file(model_path / "config.json")
            if (model_path / "config.json").exists()
            else None
        ),
        "summary": summary,
    }
    write_json(RESULT_ROOT / "w2_6_manifest.json", manifest)

    log("\n" + "=" * 96)
    log("W2.6 GOLD GATE COMPLETE")
    log("=" * 96)
    for name, x in summary["variants"].items():
        log(
            f"{name:16s} AUROC={x['macro_AUROC']:.6f} AP={x['macro_AP']:.6f} Brier={x['macro_Brier']:.6f}"
        )
    log(f"Primary delta vs W2.3    : {summary['primary_delta_vs_w23']:+.6f}")
    log(f"GATE VERDICT             : {summary['gate_verdict']}")
    log(f"Results                  : {RESULT_ROOT}")
    return summary


# ============================================================
# 10. STATUS / VALIDATION
# ============================================================


def status_w26() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "experiment": "RSNA W2.6 FS4 fold-safe few-shot challenge mapper",
        "uses_pilkwang_labels": False,
        "train_csv": str(TRAIN_CSV),
        "train_exists": TRAIN_CSV.exists(),
        "w2_root_env": W2_ROOT_ENV,
        "w23_root_env": W23_ROOT_ENV,
        "model_path_env": MODEL_PATH_ENV,
        "model_exists": Path(MODEL_PATH_ENV).exists(),
        "fs4_labels": FS4_LABELS,
        "kept_w23_labels": KEEP_W23_LABELS,
        "gold_mapper_calls": EXPECTED_GOLD * len(FS4_LABELS),
        "examples_per_query": {"positive": N_POS_EXAMPLES, "negative": N_NEG_EXAMPLES},
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "use_4bit": USE_4BIT,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
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
        # Build exemplar audit without model loading — catches fold/retrieval issues early.
        w2_gold = load_w2_gold(w2_root, gold)
        ex, audit = build_all_exemplars(gold, folds, w2_gold)
        payload["exemplar_query_label_sets"] = len(ex)
        payload["exemplar_rows"] = len(audit)
        payload["exemplar_leakage_count"] = int(
            sum(
                1
                for _, r in audit.iterrows()
                if int(
                    folds.set_index(UID).at[
                        str(r["ExampleStudyInstanceUID"]), "OuterFold"
                    ]
                )
                == int(r["OuterFold"])
            )
        )
    except Exception as exc:
        payload["input_error"] = repr(exc)

    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu_count"] = int(torch.cuda.device_count())
        payload["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        payload["torch_error"] = repr(exc)

    cache = load_cache() if CACHE_ROOT.exists() else {}
    payload["cached_mapper_cells"] = len(cache)
    log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def validate_w26() -> Dict[str, Any]:
    ensure_dirs()
    _, gold, _ = load_train()
    w23_root = get_w23_root()
    folds = load_folds(w23_root, gold)

    required = [
        RESULT_ROOT / "01_exemplar_audit.csv",
        RESULT_ROOT / "03_fs4_mapper_oof_long.csv",
        RESULT_ROOT / "04_all_variants_oof_long.csv",
        RESULT_ROOT / "05_metrics_per_label_variant.csv",
        RESULT_ROOT / "06_fs4_per_label_attribution.csv",
        RESULT_ROOT / "07_w26_gold_summary.json",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}; run run_w26('gold') first")

    audit = pd.read_csv(RESULT_ROOT / "01_exemplar_audit.csv")
    fs4 = pd.read_csv(RESULT_ROOT / "03_fs4_mapper_oof_long.csv")
    all_oof = pd.read_csv(RESULT_ROOT / "04_all_variants_oof_long.csv")
    metrics = pd.read_csv(RESULT_ROOT / "05_metrics_per_label_variant.csv")
    summary = json.loads(
        (RESULT_ROOT / "07_w26_gold_summary.json").read_text(encoding="utf-8")
    )

    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()
    leakage = 0
    for _, r in audit.iterrows():
        if int(fold_map[str(r["ExampleStudyInstanceUID"])]) == int(r["OuterFold"]):
            leakage += 1

    checks = {
        "fold_hash_match": fold_sha256(folds) == EXPECTED_FOLD_SHA256,
        "fs4_rows_232": len(fs4) == EXPECTED_GOLD * len(FS4_LABELS),
        "fs4_no_duplicates": not fs4[[UID, "Label"]].duplicated().any(),
        "fs4_all_58_uids": fs4[UID].astype(str).nunique() == EXPECTED_GOLD,
        "fs4_exact_labels": set(fs4["Label"].astype(str)) == set(FS4_LABELS),
        "fs4_probabilities_finite": bool(
            np.isfinite(pd.to_numeric(fs4["Probability"], errors="coerce")).all()
        ),
        "fs4_probabilities_in_range": bool(
            ((fs4["Probability"] >= 0) & (fs4["Probability"] <= 1)).all()
        ),
        "exemplar_rows_expected": len(audit)
        == EXPECTED_GOLD * len(FS4_LABELS) * (N_POS_EXAMPLES + N_NEG_EXAMPLES),
        "exemplar_outer_fold_leakage_zero": leakage == 0,
        "all_variants_rows": len(all_oof) == EXPECTED_GOLD * len(LABELS) * 3,
        "metrics_rows": len(metrics) == len(LABELS) * 3,
        "pilkwang_not_used": True,
    }
    checks["overall_pass"] = bool(all(checks.values()))
    payload = {"checks": checks, "summary": summary, "results_root": str(RESULT_ROOT)}
    write_json(RESULT_ROOT / "08_validation_summary.json", payload)
    log(json.dumps(payload, indent=2, allow_nan=True))
    if not checks["overall_pass"]:
        raise RuntimeError("W2.6 validation failed")
    return payload


def run_w26(mode: str = "status"):
    mode = str(mode).strip().lower()
    if mode == "status":
        return status_w26()
    if mode == "gold":
        return gold_w26()
    if mode == "validate":
        return validate_w26()
    raise ValueError("mode must be one of: status, gold, validate")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", choices=["status", "gold", "validate"], default="status")
    # args = parser.parse_args()
    # run_w26(args.mode)
    run_w26("status")
    # run_w26("gold")
    # run_w26("validate")


if __name__ == "__main__":
    main()
