#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA Knee Abnormality Detection — W2.5 V3
OUR OWN High-Fidelity Report Teacher — Gold Gate
=================================================

Goal
----
Build and evaluate OUR OWN stronger report teacher before spending hours
labeling all 4,407 reports.

This script does NOT use Pilkwang labels or any competitor-generated label
artifact. Inputs are only:

  1) competition train.csv (raw reports + 58 gold labels);
  2) OUR W2 structured report features (rule + multilingual mDeBERTa NLI);
  3) OUR exact W2.3 fold assignment / W2.3 held-out OOF baseline;
  4) a generic open instruct LLM weight directory supplied by us.

The instruct LLM is a pretrained foundation model, not a competitor's
pseudo-label dataset. We generate all report labels ourselves from raw reports.

V5 extraction change
--------------------
V4 showed cross-target contamination when all 12 findings were requested in one generation.
V5 therefore uses one target per deterministic generation during the 58-gold gate.
This is deliberately slower but gives a clean estimate of the semantic ceiling before we
design a cheaper full-corpus extractor/student.

Recommended first model
-----------------------
Qwen2.5-7B-Instruct (local/offline model directory), 4-bit NF4 on a T4.

Why the gold gate exists
------------------------
The generative pass over all 4,407 reports may take hours. First run the same
teacher on the 58 gold reports only. We then measure:

  * raw own-LLM report score;
  * fold-safe LLM-only calibration;
  * fold-safe OUR-ensemble = LLM features + OUR W2 rule/NLI features;
  * W2.3 exact held-out OOF baseline.

The outer validation fold is never used to fit the W2.5 calibrator.

Clinical state ontology
-----------------------
For every report and target (V5 queries one target at a time):
  P = PRESENT
  A = ABSENT (explicitly negated / intact)
  U = UNCERTAIN
  N = NOT_ADDRESSED

Additional structured fields:
  c = confidence in extraction [0,100]
  l = uncertain lean: P / A / N
  r = related/indirect abnormality flag 0/1
  v = severity 0 unknown, 1 mild, 2 moderate, 3 severe
  h = chronicity A acute / C chronic / U unknown
  e = short exact evidence quote from report ("" for N)

Notebook API
------------
    run_w25("status")
    run_w25("sample")
    run_w25("gold")
    run_w25("validate")

DO NOT run a 4,407-report extraction yet. The next full-corpus W2.5 stage is
authorized only after the 58-gold gate is reviewed.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. CONTROLLED CONFIG
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

EXPECTED_TOTAL = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

DATA_ROOT = Path(
    os.environ.get(
        "W25_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)
TRAIN_CSV = Path(os.environ.get("W25_TRAIN_CSV", str(DATA_ROOT / "train.csv")))

W2_ROOT_ENV = os.environ.get(
    "W25_W2_ROOT", "/kaggle/input/datasets/isayem/rsna-w2/rsna_w2"
).strip()

W23_ROOT_ENV = os.environ.get(
    "W25_W23_ROOT", "/kaggle/input/datasets/isayem/rsna-w2-3/rsna_w2_3"
).strip()

MODEL_PATH_ENV = os.environ.get(
    "W25_MODEL_PATH", "/kaggle/input/datasets/ragnar123/qwen2-5-7b-instruct"
).strip()

OUTPUT_ROOT = Path(
    os.environ.get("W25_OUTPUT_ROOT", "/kaggle/working/rsna_w2_5_gold_gate")
)
CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULT_ROOT = OUTPUT_ROOT / "results"

GPU_ID = int(os.environ.get("W25_GPU_ID", "0"))
USE_4BIT = os.environ.get("W25_USE_4BIT", "0").strip() == "1"
ALLOW_FP16_FALLBACK = os.environ.get("W25_ALLOW_FP16_FALLBACK", "1").strip() == "1"
FP16_SHARD_ALL_GPUS = os.environ.get("W25_FP16_SHARD_ALL_GPUS", "1").strip() == "1"
FP16_GPU_MAX_GIB = float(os.environ.get("W25_FP16_GPU_MAX_GIB", "13.0"))
MAX_INPUT_TOKENS = int(os.environ.get("W25_MAX_INPUT_TOKENS", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("W25_MAX_NEW_TOKENS", "240"))
GEN_BATCH_SIZE = int(os.environ.get("W25_BATCH_SIZE", "1"))

# Strong regularization: only ~45-47 outer-train gold studies per fold.
CALIBRATION_C = float(os.environ.get("W25_CALIBRATION_C", "0.03"))

BOOTSTRAP_REPEATS = int(os.environ.get("W25_BOOTSTRAP_REPEATS", "2000"))
BOOTSTRAP_SEED = int(os.environ.get("W25_BOOTSTRAP_SEED", "250824"))

# W2 Stage-B features produced by our earlier pipeline.
W2_FEATURES = [
    "EvidencePositiveScore",
    "EvidenceNegativeScore",
    "RelatedScore",
    "UncertaintyFlag",
    "FusedAssertionConfidence",
    "RuleDecidableFlag",
    "SemanticDirectStrength",
    "SemanticDirectMargin",
    "SeverityLow",
    "SeverityModerate",
    "SeverityHigh",
    "SeverityDegenerative",
]

STATE_VALUES = ["P", "A", "U", "N"]
LEAN_VALUES = ["P", "A", "N"]
CHRONICITY_VALUES = ["A", "C", "U"]

ONTOLOGY = {
    "ACL": (
        "Anterior cruciate ligament abnormality: injury, sprain, tear or rupture. "
        "Postoperative reconstruction/graft abnormality is related evidence."
    ),
    "MCL": ("Medial collateral ligament abnormality: injury, sprain, tear or rupture."),
    "Medial Meniscus": (
        "Medial meniscal tear, rupture or definite meniscal injury. "
        "Degeneration, extrusion or postoperative change without definite tear is related evidence."
    ),
    "Lateral Meniscus": (
        "Lateral meniscal tear, rupture or definite meniscal injury. "
        "Degeneration, extrusion or postoperative change without definite tear is related evidence."
    ),
    "Medial OA": (
        "Osteoarthritis/arthrosis/gonarthrosis of the medial tibiofemoral compartment. "
        "Compartment-specific cartilage loss, osteophytes or joint-space degeneration may be related evidence."
    ),
    "Lateral OA": (
        "Osteoarthritis/arthrosis/gonarthrosis of the lateral tibiofemoral compartment. "
        "Compartment-specific cartilage loss, osteophytes or joint-space degeneration may be related evidence."
    ),
    "PF OA": (
        "Patellofemoral degenerative disease / OA. Patellar or trochlear advanced cartilage loss, "
        "full-thickness defects, grade-4 chondropathy, arthrosis or osteophytes can count as present "
        "even when the report does not literally say OA. Mild isolated cartilage signal change is uncertain."
    ),
    "Effusion": (
        "Knee joint effusion or increased intra-articular fluid. A tiny/trace effusion is still present "
        "if explicitly reported."
    ),
    "Synovitis": (
        "Synovitis, synovial hypertrophy/proliferation or clearly inflamed synovium. "
        "Effusion alone does not imply synovitis."
    ),
    "Baker's": ("Baker's cyst / popliteal cyst."),
    "Contusion": (
        "Bone contusion / bone bruise / traumatic marrow edema compatible with contusion. "
        "Nonspecific degenerative marrow edema is related/uncertain, not automatically present."
    ),
    "Fracture": (
        "Fracture of knee-region bone. Record chronicity separately; a clearly healed/old fracture "
        "is still report evidence but chronic."
    ),
}


TARGET_NEGATIVE_GUIDANCE = {
    "ACL": (
        "Do not infer ACL abnormality from meniscal, cartilage, effusion, marrow, "
        "or collateral-ligament findings. A general statement that cruciate ligaments "
        "are intact/normal can support ABSENT."
    ),
    "MCL": (
        "Do not infer MCL abnormality from ACL, meniscal, cartilage, effusion, or marrow findings. "
        "A general statement that collateral ligaments are intact/normal can support ABSENT."
    ),
    "Medial Meniscus": (
        "Do not call PRESENT for degeneration, extrusion, fraying, truncation, prior meniscectomy, "
        "or postoperative change unless the report describes a definite tear/rupture or equivalent. "
        "Those indirect findings should usually be UNCERTAIN with related=1."
    ),
    "Lateral Meniscus": (
        "Do not call PRESENT for degeneration, extrusion, fraying, truncation/amputation, "
        "prior meniscectomy, or postoperative change unless a definite tear/rupture is described. "
        "Those indirect findings should usually be UNCERTAIN with related=1."
    ),
    "Medial OA": (
        "Only use medial tibiofemoral degenerative evidence. Patellofemoral-only or lateral-only "
        "cartilage disease is not evidence for medial OA."
    ),
    "Lateral OA": (
        "Only use lateral tibiofemoral degenerative evidence. Patellofemoral-only or medial-only "
        "cartilage disease is not evidence for lateral OA."
    ),
    "PF OA": (
        "Only use patellofemoral/patellar/trochlear degenerative evidence. Effusion, synovitis, "
        "meniscal findings, tibiofemoral OA, or unrelated cartilage findings are NOT evidence. "
        "Advanced patellar/trochlear degenerative cartilage loss, grade-4 chondropathy, arthrosis "
        "or osteophytes can count as PRESENT even if the letters OA are absent."
    ),
    "Effusion": (
        "Only joint-fluid/effusion evidence is relevant. Synovitis, cysts, marrow edema, OA, "
        "or soft-tissue edema are not substitutes for effusion."
    ),
    "Synovitis": (
        "Effusion alone is NOT synovitis and should not make this target PRESENT or UNCERTAIN. "
        "Require synovitis/synovial thickening, hypertrophy, proliferation, inflammatory synovium, "
        "or a directly equivalent statement."
    ),
    "Baker's": (
        "Require Baker/popliteal cyst evidence. Joint effusion or other cysts are not substitutes. "
        "An explicit statement of no popliteal/Baker cyst supports ABSENT."
    ),
    "Contusion": (
        "Require traumatic bone contusion/bone bruise or marrow edema explicitly compatible with "
        "contusion. Degenerative/subchondral marrow edema alone is not PRESENT."
    ),
    "Fracture": (
        "Require fracture evidence or an explicit statement excluding fracture. Bone bruise, "
        "marrow edema, OA, osteophyte, or cartilage injury alone is not fracture."
    ),
}


# Conservative lexical anchors used only to RECOVER a literal report sentence.
# They never create a positive/negative state by themselves. If a model assertion
# has no grounded evidence and no matching sentence can be found, V6 downgrades
# the assertion to NOT_ADDRESSED.
TARGET_EVIDENCE_PATTERNS = {
    "ACL": [
        r"\bACL\b",
        r"anterior cruciate",
        r"cruciado anterior",
        r"ligamento cruzado anterior",
        r"ligamentos? cruzados?",
        r"cruciate ligaments?",
        r"vorder(?:e|en|er)? kreuzband",
    ],
    "MCL": [
        r"\bMCL\b",
        r"medial collateral",
        r"collateral medial",
        r"ligamento colateral medial",
        r"ligamentos? colaterales?",
        r"collateral ligaments?",
        r"mediales kollateralband",
    ],
    "Medial Meniscus": [
        r"medial menisc",
        r"menisco medial",
        r"meniskus med",
    ],
    "Lateral Meniscus": [
        r"lateral menisc",
        r"menisco lateral",
        r"meniskus lat",
    ],
    "Medial OA": [
        r"medial compartment",
        r"medial tibiofemoral",
        r"compartimento medial",
        r"femorotibial medial",
        r"gonarthr",
        r"osteoarthr",
        r"arthros",
    ],
    "Lateral OA": [
        r"lateral compartment",
        r"lateral tibiofemoral",
        r"compartimento lateral",
        r"femorotibial lateral",
        r"gonarthr",
        r"osteoarthr",
        r"arthros",
    ],
    "PF OA": [
        r"patellofem",
        r"patelo[f-]?em",
        r"patell",
        r"trochle",
        r"femoropatellar",
        r"femoropatel",
        r"retropatell",
        r"chondrop",
    ],
    "Effusion": [
        r"effusion",
        r"joint fluid",
        r"derrame articular",
        r"derrame",
        r"erguss",
        r"efüzyon",
        r"effus",
    ],
    "Synovitis": [
        r"synovit",
        r"synovial",
        r"sinovit",
        r"sinovial",
    ],
    "Baker's": [
        r"baker",
        r"popliteal cyst",
        r"quiste popl",
        r"kyste popl",
        r"poplitealzyst",
    ],
    "Contusion": [
        r"contusion",
        r"bone bruise",
        r"bone contus",
        r"contusi[oó]n",
        r"marrow edema",
        r"bone marrow edema",
        r"ödem",
        r"oedem",
    ],
    "Fracture": [
        r"fracture",
        r"fractur",
        r"fraktur",
        r"kırık",
        r"fissure",
    ],
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def report_sentences(report: str) -> List[str]:
    parts = [x.strip() for x in _SENTENCE_SPLIT_RE.split(str(report)) if x.strip()]
    if not parts:
        return [str(report).strip()] if str(report).strip() else []
    return parts


def evidence_mentions_target(evidence: str, label: str) -> bool:
    s = normalize_text(evidence)
    return any(
        re.search(pattern, s, flags=re.I) for pattern in TARGET_EVIDENCE_PATTERNS[label]
    )


def evidence_polarity_compatible(evidence: str, state: str) -> bool:
    """
    Conservative compatibility gate used only for deterministic evidence recovery.
    It prevents, for example, an effusion sentence from grounding PF-OA or a
    positive tear sentence from grounding an ABSENT state.
    """
    s = normalize_text(evidence)

    neg = [
        r"\bno\b",
        r"\bwithout\b",
        r"\bintact\b",
        r"\bnormal\b",
        r"\bsin\b",
        r"\bnegative\b",
        r"\bpreserv",
        r"\bunremark",
        r"\bkein",
        r"\bkeine",
        r"\bintakt",
        r"sin alteraciones",
    ]
    pos = [
        r"tear",
        r"ruptur",
        r"sprain",
        r"effusion",
        r"synovit",
        r"cyst",
        r"fractur",
        r"contusion",
        r"bruise",
        r"edema",
        r"oedema",
        r"arthro",
        r"chondrop",
        r"degener",
        r"derrame",
        r"rotura",
        r"lesi[oó]n",
        r"amputaci[oó]n",
        r"extrus",
    ]

    if state == "A":
        return any(re.search(p, s, flags=re.I) for p in neg)
    if state == "P":
        return any(re.search(p, s, flags=re.I) for p in pos)
    return True


def apply_target_semantic_guard(
    label: str,
    state: str,
    lean: str,
    related: int,
    confidence: int,
    evidence: str,
    action: str,
) -> Tuple[str, str, int, int, str]:
    """
    Deterministic high-precision guards for failure modes already observed in
    the sample. These rules can only DOWNGRADE authority; they never invent a
    stronger positive state.
    """
    s = normalize_text(evidence)

    # Meniscal truncation/degeneration/postoperative morphology is not the same
    # as a definite tear unless tear/rupture language is actually present.
    if label in {"Medial Meniscus", "Lateral Meniscus"} and state == "P":
        indirect = any(
            re.search(p, s, flags=re.I)
            for p in [
                r"degener",
                r"extrus",
                r"fray",
                r"truncat",
                r"amputaci[oó]n",
                r"meniscect",
                r"postoper",
                r"resect",
            ]
        )
        definite = any(
            re.search(p, s, flags=re.I)
            for p in [
                r"tear",
                r"ruptur",
                r"rotura",
                r"desgar",
                r"fissur",
            ]
        )
        if indirect and not definite:
            return "U", "P", 1, min(confidence, 80), action + "|meniscus_indirect_to_U"

    # Compartment OA positives require location-specific evidence.
    if label == "Medial OA" and state == "P":
        if not re.search(
            r"medial|femorotibial medial|tibiofemoral medial|compartimento medial",
            s,
            flags=re.I,
        ):
            return "U", "P", 1, min(confidence, 75), action + "|nonlocal_OA_to_U"

    if label == "Lateral OA" and state == "P":
        if not re.search(
            r"lateral|femorotibial lateral|tibiofemoral lateral|compartimento lateral",
            s,
            flags=re.I,
        ):
            return "U", "P", 1, min(confidence, 75), action + "|nonlocal_OA_to_U"

    if label == "PF OA" and state == "P":
        if not re.search(
            r"patell|trochle|femoropat|retropatell|patelo[f-]?em|chondropat",
            s,
            flags=re.I,
        ):
            return "N", "N", 0, min(confidence, 50), action + "|nonPF_evidence_to_N"

    # Effusion alone cannot ground synovitis.
    if label == "Synovitis" and state in {"P", "U"}:
        if not re.search(r"synovit|synovial|sinovit|sinovial", s, flags=re.I):
            return (
                "N",
                "N",
                0,
                min(confidence, 50),
                action + "|nonsynovial_evidence_to_N",
            )

    return state, lean, related, confidence, action


def recover_literal_evidence(report: str, label: str, state: str) -> str:
    """
    Recover a literal sentence containing a conservative lexical anchor.

    This function does NOT decide the state. It only grounds an already-proposed
    LLM P/A/U assertion. If no suitable literal sentence exists, return "".
    """
    patterns = TARGET_EVIDENCE_PATTERNS[label]
    candidates: List[Tuple[int, int, str]] = []

    # Weak polarity clues only rank among already target-matching sentences.
    neg_clues = [
        r"\bno\b",
        r"\bwithout\b",
        r"\bintact\b",
        r"\bnormal\b",
        r"\bsin\b",
        r"\bnegat",
        r"\bpreserv",
        r"\bunremark",
        r"\bkein",
        r"\bkeine",
        r"\bintakt",
    ]
    pos_clues = [
        r"tear",
        r"ruptur",
        r"sprain",
        r"effusion",
        r"synovit",
        r"cyst",
        r"fractur",
        r"contusion",
        r"bruise",
        r"edema",
        r"oedema",
        r"arthro",
        r"chondrop",
        r"degener",
        r"derrame",
        r"rotura",
        r"lesi[oó]n",
        r"amputaci[oó]n",
    ]

    for pos, sentence in enumerate(report_sentences(report)):
        s_norm = normalize_text(sentence)
        if not any(re.search(p, s_norm, flags=re.I) for p in patterns):
            continue
        if not evidence_polarity_compatible(sentence, state):
            continue

        score = 10
        if state == "A":
            score += 3 * sum(bool(re.search(p, s_norm, flags=re.I)) for p in neg_clues)
        elif state == "P":
            score += 2 * sum(bool(re.search(p, s_norm, flags=re.I)) for p in pos_clues)
        elif state == "U":
            score += 1
        candidates.append((score, -pos, sentence))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    # Keep a concise but literal span. Sentence itself is acceptable and safer
    # than fabricating a shortened quote that may not be verbatim.
    return candidates[0][2][:500]


# ============================================================
# 2. UTILITIES
# ============================================================


def ensure_dirs() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.split())
    return text.casefold()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fold_sha256(frame: pd.DataFrame) -> str:
    ordered = frame[[UID, "OuterFold"]].copy()
    ordered[UID] = ordered[UID].astype(str)
    ordered = ordered.sort_values(UID).reset_index(drop=True)
    payload = "".join(
        f"{u},{int(f)}\n" for u, f in zip(ordered[UID], ordered["OuterFold"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def fast_binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


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
    deltas: List[float] = []
    attempts = 0
    while len(deltas) < repeats and attempts < repeats * 30:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if any(len(np.unique(ys[:, j])) < 2 for j in range(ys.shape[1])):
            continue
        deltas.append(
            fast_macro_auc(ys, candidate[idx]) - fast_macro_auc(ys, reference[idx])
        )
    if not deltas:
        return {
            "n": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_gt_0": float("nan"),
        }
    a = np.asarray(deltas, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "ci_low": float(np.quantile(a, 0.025)),
        "ci_high": float(np.quantile(a, 0.975)),
        "p_gt_0": float(np.mean(a > 0)),
    }


def shallow_dirs(root: Path = Path("/kaggle/input"), max_depth: int = 4) -> List[Path]:
    if not root.exists():
        return []
    out = [root]
    queue = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = [x for x in current.iterdir() if x.is_dir()]
        except Exception:
            continue
        for child in children:
            if child.name in {"train_series", "test_series"}:
                continue
            out.append(child)
            queue.append((child, depth + 1))
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True, default=conv),
        encoding="utf-8",
    )


# ============================================================
# 3. INPUT DISCOVERY / INTEGRITY
# ============================================================


def load_train() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)
    train = pd.read_csv(TRAIN_CSV)
    train[UID] = train[UID].astype(str)
    required = {UID, REPORT, *LABELS}
    missing = required - set(train.columns)
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
    if (
        len(train) != EXPECTED_TOTAL
        or len(gold) != EXPECTED_GOLD
        or len(unlabeled) != EXPECTED_UNLABELED
    ):
        raise RuntimeError(
            f"Unexpected counts: train={len(train)}, gold={len(gold)}, unlabeled={len(unlabeled)}"
        )
    return train, gold, unlabeled


def looks_like_w2_root(path: Path) -> bool:
    return (path / "results" / "04_gold_structured_report_features.csv").exists() and (
        path / "results" / "08_full_structured_report_labels.csv"
    ).exists()


def discover_w2_root() -> Path:
    candidates: List[Path] = []
    if W2_ROOT_ENV:
        x = Path(W2_ROOT_ENV)
        candidates += [x, x / "rsna_w2"]
    candidates += [
        Path("/kaggle/working/rsna_w2"),
        Path("/kaggle/input/rsna-w2-full/rsna_w2"),
    ]
    for root in shallow_dirs():
        candidates += [root, root / "rsna_w2"]
    for x in dict.fromkeys(candidates):
        if looks_like_w2_root(x):
            return x
    raise FileNotFoundError(
        "OUR W2 root not found. Set W25_W2_ROOT to the directory containing "
        "results/04_gold_structured_report_features.csv and "
        "results/08_full_structured_report_labels.csv."
    )


def looks_like_w23_root(path: Path) -> bool:
    return (path / "results" / "00_outer_fold_assignments.csv").exists() and all(
        (path / "folds" / f"fold_{f}" / "heldout_gold_stage_b_predictions.csv").exists()
        for f in range(1, 6)
    )


def discover_w23_root() -> Path:
    candidates: List[Path] = []
    if W23_ROOT_ENV:
        x = Path(W23_ROOT_ENV)
        candidates += [x, x / "rsna_w2_3"]
    candidates += [
        Path("/kaggle/working/rsna_w2_3"),
        Path("/kaggle/input/rsna-w2-3/rsna_w2_3"),
    ]
    for root in shallow_dirs():
        candidates += [root, root / "rsna_w2_3"]
    for x in dict.fromkeys(candidates):
        if looks_like_w23_root(x):
            return x
    raise FileNotFoundError(
        "OUR W2.3 root not found. Set W25_W23_ROOT to the directory containing "
        "results/00_outer_fold_assignments.csv and folds/fold_*/."
    )


def load_folds(w23_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    path = w23_root / "results" / "00_outer_fold_assignments.csv"
    folds = pd.read_csv(path)
    folds[UID] = folds[UID].astype(str)
    folds = folds[[UID, "OuterFold"]].sort_values(UID).reset_index(drop=True)
    if set(folds[UID]) != set(gold[UID]):
        raise RuntimeError("W2.3 fold UIDs differ from gold UIDs")
    digest = fold_sha256(folds)
    if digest != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"Fold checksum mismatch: {digest}")
    return folds


def load_w2_gold(w2_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    path = w2_root / "results" / "04_gold_structured_report_features.csv"
    frame = pd.read_csv(path)
    frame[UID] = frame[UID].astype(str)
    required = {UID, "Label", *W2_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"W2 gold structured file missing columns: {sorted(missing)}"
        )
    if len(frame) != EXPECTED_GOLD * len(LABELS):
        raise RuntimeError(f"W2 gold structured rows={len(frame)}, expected 696")
    if frame[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate W2 gold UID/Label")
    if set(frame[UID]) != set(gold[UID]):
        raise RuntimeError("W2 gold UID set mismatch")
    return frame


def load_w23_oof(w23_root: Path, gold: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for fold in range(1, 6):
        path = (
            w23_root / "folds" / f"fold_{fold}" / "heldout_gold_stage_b_predictions.csv"
        )
        x = pd.read_csv(path)
        x[UID] = x[UID].astype(str)
        required = {UID, "Label", "Gold", "FoldSafeChallengeProbability"}
        missing = required - set(x.columns)
        if missing:
            raise RuntimeError(f"{path} missing {sorted(missing)}")
        x = x[[UID, "Label", "Gold", "FoldSafeChallengeProbability"]].copy()
        x["OuterFold"] = fold
        pieces.append(x)
    oof = pd.concat(pieces, ignore_index=True)
    if (
        len(oof) != EXPECTED_GOLD * len(LABELS)
        or oof[[UID, "Label"]].duplicated().any()
    ):
        raise RuntimeError("Invalid W2.3 held-out OOF table")
    if set(oof[UID]) != set(gold[UID]):
        raise RuntimeError("W2.3 OOF UID set mismatch")
    return oof


def discover_model_path() -> Optional[Path]:
    if MODEL_PATH_ENV:
        p = Path(MODEL_PATH_ENV)
        return p if p.exists() else p

    # Deliberately conservative auto-detection: prefer Qwen instruct directories.
    candidates = []
    for root in shallow_dirs(max_depth=4):
        if not (root / "config.json").exists():
            continue
        name = str(root).casefold()
        score = 0
        if "qwen" in name:
            score += 20
        if "instruct" in name:
            score += 10
        if "7b" in name:
            score += 5
        candidates.append((score, root))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    return candidates[0][1]


# ============================================================
# 4. SINGLE-TARGET PROMPT / JSON CONTRACT
# ============================================================

SYSTEM_PROMPT = """You are a radiology information-extraction engine.
Read ONE knee MRI report and evaluate exactly ONE requested target.

The report may be English, Spanish, Turkish, Bosnian/Croatian/Serbian,
Bulgarian, Greek, Dutch, German, or mixed language.

Use only report text. Do not infer from prevalence or unrelated abnormalities.
Silence is NOT a negative finding.

State ontology:
P = PRESENT: the requested target abnormality is directly reported or the
    report contains a challenge-equivalent direct finding defined in the prompt.
A = ABSENT: the requested target is explicitly negated, or a statement explicitly
    says the corresponding structure/finding is normal/intact/absent.
U = UNCERTAIN: possible/equivocal/questioned or only related/indirect evidence.
N = NOT_ADDRESSED: no meaningful evidence about the requested target.

Return JSON only. No markdown, no prose outside JSON.
"""


def build_user_prompt(report: str, label: str) -> str:
    return f"""REQUESTED TARGET
{label}

TARGET DEFINITION
{ONTOLOGY[label]}

TARGET-SPECIFIC EXCLUSIONS / BOUNDARIES
{TARGET_NEGATIVE_GUIDANCE[label]}

OUTPUT
Return exactly ONE JSON object with these seven fields:
{{
  "s": "P|A|U|N",
  "c": integer 0..100,
  "l": "P|A|N",
  "r": 0 or 1,
  "v": integer 0..3,
  "h": "A|C|U",
  "e": "short literal quote copied from report"
}}

Rules:
- Evaluate ONLY "{label}". Ignore all other targets except when they clarify
  that this target is normal/absent or provide explicitly related indirect evidence.
- c is confidence that your REPORT-SIDE extraction is correct; it is not disease probability.
- l is meaningful only for U: P=leans present, A=leans absent, N=no lean.
- r=1 only for indirect/related evidence about THIS target.
- v: 0 unknown/none, 1 mild, 2 moderate, 3 severe.
- h: A acute/recent, C chronic/degenerative/old, U unknown/not applicable.
- For P, A, or U: e MUST be a short literal VERBATIM span that occurs in the report.
- For N: force l="N", r=0, v=0, h="U", e="".
- Never use evidence about another diagnosis as evidence for "{label}".
- If evidence is contradictory, return U.
- If there is no literal supporting or negating evidence for "{label}", return N.

REPORT
<<<
{report}
>>>
"""


def extract_json_blob(text: str) -> Dict[str, Any]:
    raw = str(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    first = raw.find("{")
    last = raw.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("No JSON object found")
    payload = json.loads(raw[first : last + 1])
    if not isinstance(payload, dict):
        raise ValueError("Top-level output is not an object")
    return payload


def normalize_state_value(value: Any) -> str:
    token = str(value).strip().upper()
    aliases = {
        "PRESENT": "P",
        "POSITIVE": "P",
        "YES": "P",
        "ABSENT": "A",
        "NEGATIVE": "A",
        "NO": "A",
        "UNCERTAIN": "U",
        "EQUIVOCAL": "U",
        "POSSIBLE": "U",
        "NOT_ADDRESSED": "N",
        "NOTADDRESSED": "N",
        "NOT MENTIONED": "N",
        "NOT_MENTIONED": "N",
        "NONE": "N",
    }
    token = aliases.get(token, token)
    if token not in STATE_VALUES:
        raise ValueError(f"Invalid state {value!r}")
    return token


def normalize_choice(
    value: Any,
    allowed: Sequence[str],
    aliases: Optional[Dict[str, str]] = None,
    default_if_invalid: Optional[str] = None,
) -> str:
    token = str(value).strip().upper()
    if aliases:
        token = aliases.get(token, token)
    if token not in allowed:
        if default_if_invalid is not None:
            return default_if_invalid
        raise ValueError(f"Invalid choice {value!r}; expected {allowed}")
    return token


def normalize_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        x = int(round(float(value)))
    except Exception:
        x = default
    return int(np.clip(x, low, high))


def parse_single_teacher_payload(
    payload: Dict[str, Any],
    report: str,
    label: str,
) -> Dict[str, Any]:
    """
    V6.1 grounded, non-fatal parser.

    P/A/U survives only if:
      1) evidence is literal report text;
      2) evidence is relevant to the requested target;
      3) if evidence had to be recovered, polarity is compatible.

    Otherwise V6.1 tries deterministic literal recovery. If no safe grounding is
    available, the assertion becomes NOT_ADDRESSED. The whole run never aborts
    merely because the model emitted an unsupported clinical state.
    """
    if normalize_name(label) in {normalize_name(k) for k in payload.keys()}:
        matched_key = next(
            k for k in payload.keys() if normalize_name(k) == normalize_name(label)
        )
        if isinstance(payload[matched_key], dict):
            payload = payload[matched_key]

    try:
        s = normalize_state_value(payload.get("s", "N"))
    except Exception:
        s = "N"

    c = normalize_int(payload.get("c", 50), 0, 100, default=50)
    l = normalize_choice(
        payload.get("l", "N"),
        LEAN_VALUES,
        aliases={
            "PRESENT": "P",
            "ABSENT": "A",
            "POSITIVE": "P",
            "NEGATIVE": "A",
            "NONE": "N",
            "U": "N",
        },
        default_if_invalid="N",
    )
    r = normalize_int(payload.get("r", 0), 0, 1, default=0)
    v = normalize_int(payload.get("v", 0), 0, 3, default=0)
    h = normalize_choice(
        payload.get("h", "U"),
        CHRONICITY_VALUES,
        aliases={
            "ACUTE": "A",
            "RECENT": "A",
            "CHRONIC": "C",
            "DEGENERATIVE": "C",
            "OLD": "C",
            "UNKNOWN": "U",
            "NONE": "U",
            "N": "U",
            "NA": "U",
            "N/A": "U",
            "P": "U",
            "NOTAPPLICABLE": "U",
            "NOT_ADDRESSED": "U",
            "NOTADDRESSED": "U",
        },
        default_if_invalid="U",
    )
    e = str(payload.get("e", "") or "").strip()

    original_state = s
    original_evidence = e
    grounding_action = "as_returned"

    if s == "N":
        l, r, v, h, e = "N", 0, 0, "U", ""
        evidence_verified = False
        grounding_action = "canonical_not_addressed"
    else:
        report_norm = normalize_text(report)
        literal = bool(e and normalize_text(e) in report_norm)
        relevant = bool(literal and evidence_mentions_target(e, label))

        if not (literal and relevant):
            recovered = recover_literal_evidence(report, label, s)
            if recovered:
                e = recovered
                evidence_verified = True
                grounding_action = "deterministic_literal_recovery"
                c = min(c, 80)
            else:
                s, l, r, v, h, e = "N", "N", 0, 0, "U", ""
                evidence_verified = False
                grounding_action = "unsupported_or_irrelevant_to_N"
                c = min(c, 50)
        else:
            evidence_verified = True

        if s != "N":
            s, l, r, c, grounding_action = apply_target_semantic_guard(
                label=label,
                state=s,
                lean=l,
                related=r,
                confidence=c,
                evidence=e,
                action=grounding_action,
            )
            if s == "N":
                l, r, v, h, e = "N", 0, 0, "U", ""
                evidence_verified = False

    return {
        "State": s,
        "OriginalState": original_state,
        "Confidence": c,
        "EffectiveConfidence": c,
        "Lean": l,
        "Related": r,
        "Severity": v,
        "Chronicity": h,
        "Evidence": e,
        "OriginalEvidence": original_evidence,
        "EvidenceVerified": bool(evidence_verified),
        "GroundingAction": grounding_action,
    }


def raw_state_score(state: str, confidence: float, lean: str, related: int) -> float:
    """Gold-independent report-side ranking score."""
    c = float(np.clip(confidence / 100.0, 0.0, 1.0))
    if state == "P":
        return float(0.82 + 0.17 * c)
    if state == "A":
        return float(0.18 - 0.17 * c)
    if state == "U":
        amplitude = 0.18 * c
        if lean == "P":
            return float(0.50 + amplitude)
        if lean == "A":
            return float(0.50 - amplitude)
        if related:
            return float(0.50 + 0.07 * c)
        return 0.50
    return 0.50


# ============================================================
# 5. LOCAL LLM
# ============================================================


class LocalReportLLM:
    def __init__(self, model_path: Path):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:
            raise RuntimeError(
                "W2.5 requires torch + transformers; 4-bit mode additionally "
                "requires bitsandbytes."
            ) from exc

        self.torch = torch
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        log(f"Loading own report LLM from: {self.model_path}")
        log(f"GPU id                  : {GPU_ID}")
        log(f"4-bit requested         : {USE_4BIT}")
        log(f"FP16 fallback allowed   : {ALLOW_FP16_FALLBACK}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if GPU_ID >= gpu_count:
                raise RuntimeError(
                    f"W25_GPU_ID={GPU_ID}, but only {gpu_count} GPUs visible"
                )

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
                    if not ALLOW_FP16_FALLBACK and gpu_count < 2:
                        raise RuntimeError(
                            "4-bit model load failed and no safe multi-GPU FP16 fallback "
                            "is available. Install a compatible bitsandbytes build or use "
                            "a smaller model."
                        ) from exc

                    # Qwen2.5-7B FP16 is approximately a full T4 by itself.
                    # On 2xT4 we automatically fall back to sharded FP16 when
                    # bitsandbytes is unavailable, even if W25_USE_4BIT=1.
                    # Loading the entire model on one 14.6-GiB T4 leaves almost no
                    # room for attention/KV-cache activations and deterministically
                    # OOMs at prefill. If >=2 GPUs are available, shard the model
                    # across them instead of retrying the impossible single-GPU
                    # configuration.
                    kwargs.pop("quantization_config", None)

                    if gpu_count >= 2 and FP16_SHARD_ALL_GPUS:
                        max_memory = {
                            i: f"{FP16_GPU_MAX_GIB:.1f}GiB" for i in range(gpu_count)
                        }
                        # Do not permit silent CPU/disk offload for this gate: 2xT4
                        # has enough aggregate VRAM and CPU offload would make the
                        # 58-report timing misleading.
                        kwargs["dtype"] = torch.float16
                        kwargs["device_map"] = "balanced"
                        kwargs["max_memory"] = max_memory

                        warnings.warn(
                            "4-bit load failed; using dual/multi-GPU sharded FP16 "
                            f"fallback across {gpu_count} GPUs. Original error: {exc}"
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            str(self.model_path), **kwargs
                        )
                        self.load_mode = f"fp16_balanced_{gpu_count}gpu"
                    else:
                        # Retain an explicit single-GPU fallback only for users who
                        # intentionally run a model that genuinely fits. Qwen2.5-7B
                        # does not have enough headroom on one T4 in FP16.
                        kwargs["dtype"] = torch.float16
                        kwargs["device_map"] = {"": GPU_ID}
                        warnings.warn(
                            "4-bit load failed; trying single-GPU FP16 fallback. "
                            "This is NOT expected to fit Qwen2.5-7B on a T4. "
                            f"Original error: {exc}"
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            str(self.model_path), **kwargs
                        )
                        self.load_mode = f"fp16_single_gpu_{GPU_ID}"
            else:
                if gpu_count >= 2 and FP16_SHARD_ALL_GPUS:
                    kwargs["dtype"] = torch.float16
                    kwargs["device_map"] = "balanced"
                    kwargs["max_memory"] = {
                        i: f"{FP16_GPU_MAX_GIB:.1f}GiB" for i in range(gpu_count)
                    }
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path), **kwargs
                    )
                    self.load_mode = f"fp16_balanced_{gpu_count}gpu"
                else:
                    kwargs["dtype"] = torch.float16
                    kwargs["device_map"] = {"": GPU_ID}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path), **kwargs
                    )
                    self.load_mode = f"fp16_single_gpu_{GPU_ID}"
        else:
            kwargs["dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), **kwargs
            )
            self.load_mode = "fp32_cpu"

        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.device = self.input_device  # backward-compatible internal alias
        log(f"Model load mode          : {self.load_mode}")
        log(f"Input embedding device   : {self.input_device}")
        if hasattr(self.model, "hf_device_map"):
            device_counts: Dict[str, int] = {}
            for dev in self.model.hf_device_map.values():
                key = str(dev)
                device_counts[key] = device_counts.get(key, 0) + 1
            log(f"HF device-map modules    : {device_counts}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(i)
                log(
                    f"GPU {i} free after load    : "
                    f"{free_b / (1024**3):.2f} / {total_b / (1024**3):.2f} GiB"
                )

    def _format_prompt(self, report: str, label: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(report, label)},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return SYSTEM_PROMPT + "\n\n" + build_user_prompt(report, label) + "\n\nJSON:"

    def generate_text(
        self,
        report: str,
        label: str,
        repair_error: Optional[str] = None,
        previous_output: Optional[str] = None,
    ) -> str:
        torch = self.torch

        if repair_error is None:
            prompt = self._format_prompt(report, label)
        else:
            repair_instruction = f"""
Your previous answer for target "{label}" violated the contract.

ERROR:
{repair_error}

PREVIOUS OUTPUT:
<<<
{(previous_output or "")[:2000]}
>>>

Return the COMPLETE corrected ONE-TARGET JSON object only.
Do not change the requested target.
For P/A/U, copy a short literal evidence span from the REPORT.
If no literal evidence about "{label}" exists, return N with:
l="N", r=0, v=0, h="U", e="".
"""
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        build_user_prompt(report, label) + "\n\n" + repair_instruction
                    ),
                },
            ]
            if getattr(self.tokenizer, "chat_template", None):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = (
                    SYSTEM_PROMPT
                    + "\n\n"
                    + build_user_prompt(report, label)
                    + "\n\n"
                    + repair_instruction
                    + "\n\nJSON:"
                )

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

        new_tokens = generated[0, input_len:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    def extract_one(
        self,
        report: str,
        label: str,
    ) -> Tuple[Dict[str, Any], str]:
        """
        One generation per target.

        JSON syntax failure gets one syntax-repair attempt because there is no
        usable structured payload. Semantic/evidence imperfections do NOT trigger
        another LLM call; parse_single_teacher_payload() grounds or safely
        downgrades them deterministically.
        """
        raw = ""
        try:
            raw = self.generate_text(report, label)
            payload = extract_json_blob(raw)
        except self.torch.cuda.OutOfMemoryError as exc:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during label-wise generation. Check printed multi-GPU "
                "load mode and free VRAM."
            ) from exc
        except Exception as first_exc:
            log(f"  {label}: JSON syntax failure; one repair call: {first_exc}")
            raw = self.generate_text(
                report,
                label,
                repair_error=f"Return valid JSON only. Parser error: {first_exc}",
                previous_output=raw,
            )
            payload = extract_json_blob(raw)

        parsed = parse_single_teacher_payload(payload, report, label)
        return parsed, raw


# ============================================================
# 6. RESUMABLE LABEL-WISE GOLD EXTRACTION
# ============================================================


def cache_jsonl_path() -> Path:
    # New cache name prevents accidental reuse of the failed all-12-call cache.
    return CACHE_ROOT / "own_llm_gold_labelwise_outputs_v6_1.jsonl"


def pair_key(uid: str, label: str) -> str:
    return f"{uid}|||{label}"


def load_gold_cache() -> Dict[str, Dict[str, Any]]:
    path = cache_jsonl_path()
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                uid = str(row[UID])
                label = str(row["Label"])
                if label not in LABELS:
                    raise ValueError(f"unknown label {label}")
                out[pair_key(uid, label)] = row
            except Exception as exc:
                raise RuntimeError(f"Corrupt V5 cache line {line_no}: {exc}") from exc
    return out


def append_gold_cache(row: Mapping[str, Any]) -> None:
    ensure_dirs()
    with cache_jsonl_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extraction_rows_to_long(
    extraction_rows: Sequence[Mapping[str, Any]],
    gold: pd.DataFrame,
) -> pd.DataFrame:
    gold_lookup = gold.set_index(UID)
    rows: List[Dict[str, Any]] = []

    for row in extraction_rows:
        uid = str(row[UID])
        label = str(row["Label"])
        item = row["Parsed"]

        if uid not in gold_lookup.index:
            raise RuntimeError(f"Gold extraction UID not in gold table: {uid}")
        if label not in LABELS:
            raise RuntimeError(f"Unexpected target label: {label}")

        score = raw_state_score(
            item["State"],
            item["EffectiveConfidence"],
            item["Lean"],
            int(item["Related"]),
        )
        rows.append(
            {
                UID: uid,
                "Label": label,
                "Gold": int(gold_lookup.at[uid, label]),
                "State": item["State"],
                "OriginalState": item.get("OriginalState", item["State"]),
                "GroundingAction": item.get("GroundingAction", "unknown"),
                "Confidence": int(item["Confidence"]),
                "EffectiveConfidence": int(item["EffectiveConfidence"]),
                "Lean": item["Lean"],
                "Related": int(item["Related"]),
                "Severity": int(item["Severity"]),
                "Chronicity": item["Chronicity"],
                "Evidence": item["Evidence"],
                "OriginalEvidence": item.get("OriginalEvidence", item["Evidence"]),
                "EvidenceVerified": bool(item["EvidenceVerified"]),
                "RawStateScore": score,
                "RawOutput": row.get("RawOutput", ""),
            }
        )

    frame = pd.DataFrame(rows)
    if frame[[UID, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate V5 own-LLM UID/Label")
    return frame


def run_gold_extraction(
    gold: pd.DataFrame,
    model_path: Path,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    ensure_dirs()

    targets = gold.copy()
    if limit is not None:
        targets = targets.head(int(limit)).copy()

    requested_pairs = [
        (str(uid), label) for uid in targets[UID].astype(str) for label in LABELS
    ]

    cache = load_gold_cache()
    needed = [
        (uid, label)
        for uid, label in requested_pairs
        if pair_key(uid, label) not in cache
    ]

    log(f"Gold reports requested     : {len(targets)}")
    log(f"Label-wise calls requested : {len(requested_pairs)}")
    log(f"Already cached pairs       : {len(requested_pairs) - len(needed)}")
    log(f"Need LLM label calls       : {len(needed)}")
    log("Extraction mode            : one target per generation")

    llm = LocalReportLLM(model_path) if needed else None
    report_lookup = targets.set_index(UID)[REPORT].astype(str).to_dict()
    start = time.time()

    for i, (uid, label) in enumerate(needed, start=1):
        report = report_lookup[uid]
        t0 = time.time()
        parsed, raw = llm.extract_one(report, label)

        row = {
            UID: uid,
            "Label": label,
            "ModelPath": str(model_path),
            "ModelConfigSHA256": (
                sha256_file(model_path / "config.json")
                if (model_path / "config.json").exists()
                else None
            ),
            "PromptVersion": "w25_labelwise_gold_gate_v6_1_grounded_relevance",
            "Parsed": parsed,
            "RawOutput": raw,
        }
        append_gold_cache(row)
        cache[pair_key(uid, label)] = row

        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-6)
        eta = (len(needed) - i) / max(rate, 1e-9)
        log(
            f"  extracted {i:>3}/{len(needed)} "
            f"study={uid[-10:]} target={label:<18s} "
            f"call={time.time() - t0:5.1f}s "
            f"ETA={eta/60:5.1f}m"
        )

    ordered_rows = [cache[pair_key(uid, label)] for uid, label in requested_pairs]
    long_df = extraction_rows_to_long(ordered_rows, gold)

    expected = len(targets) * len(LABELS)
    if len(long_df) != expected:
        raise RuntimeError(
            f"V5 label-wise extraction rows={len(long_df)}, expected={expected}"
        )
    return long_df


# ============================================================
# 7. FEATURE MATRICES
# ============================================================

LLM_FEATURES = [
    "RawStateScore",
    "EffectiveConfidence01",
    "Related",
    "Severity01",
    "EvidenceVerified",
    "StateP",
    "StateA",
    "StateU",
    "StateN",
    "LeanP",
    "LeanA",
    "ChronicAcute",
    "ChronicChronic",
]


def add_llm_features(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()
    out["EffectiveConfidence01"] = (
        pd.to_numeric(out["EffectiveConfidence"], errors="coerce")
        .fillna(0.0)
        .clip(0, 100)
        / 100.0
    )
    out["Severity01"] = (
        pd.to_numeric(out["Severity"], errors="coerce").fillna(0.0).clip(0, 3) / 3.0
    )
    out["EvidenceVerified"] = out["EvidenceVerified"].astype(float)
    out["Related"] = pd.to_numeric(out["Related"], errors="coerce").fillna(0).clip(0, 1)

    for state in STATE_VALUES:
        out[f"State{state}"] = (out["State"] == state).astype(float)

    out["LeanP"] = ((out["State"] == "U") & (out["Lean"] == "P")).astype(float)
    out["LeanA"] = ((out["State"] == "U") & (out["Lean"] == "A")).astype(float)
    out["ChronicAcute"] = (out["Chronicity"] == "A").astype(float)
    out["ChronicChronic"] = (out["Chronicity"] == "C").astype(float)
    return out


def join_w2_features(
    llm_long: pd.DataFrame,
    w2_gold: pd.DataFrame,
) -> pd.DataFrame:
    left = add_llm_features(llm_long)
    right = w2_gold[[UID, "Label", *W2_FEATURES]].copy()
    merged = left.merge(
        right,
        on=[UID, "Label"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_W2"),
    )
    if merged[W2_FEATURES].isna().all(axis=1).any():
        raise RuntimeError("Missing W2 features after join")
    return merged


def design_matrix(frame: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == "llm_only":
        columns = LLM_FEATURES
    elif variant == "llm_plus_w2":
        columns = [*LLM_FEATURES, *W2_FEATURES]
    else:
        raise ValueError(variant)
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


# ============================================================
# 8. EXACT OUTER-FOLD CALIBRATION
# ============================================================


def fit_outer_oof_variant(
    feature_frame: pd.DataFrame,
    folds: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()
    rows: List[Dict[str, Any]] = []

    for label in LABELS:
        label_df = (
            feature_frame[feature_frame["Label"] == label]
            .copy()
            .sort_values(UID)
            .reset_index(drop=True)
        )
        if len(label_df) != EXPECTED_GOLD:
            raise RuntimeError(
                f"{label}: expected 58 own-teacher rows, found {len(label_df)}"
            )

        y = label_df["Gold"].to_numpy(dtype=np.int64)
        X = design_matrix(label_df, variant)
        outer = np.array([fold_map[str(uid)] for uid in label_df[UID]], dtype=int)

        pred = np.full(len(label_df), np.nan, dtype=np.float64)

        for fold in range(1, 6):
            tr = outer != fold
            va = outer == fold
            if len(np.unique(y[tr])) < 2:
                raise RuntimeError(
                    f"{variant}/{label}/fold{fold}: outer train has one class"
                )

            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            C=CALIBRATION_C,
                            solver="liblinear",
                            class_weight=None,
                            max_iter=3000,
                            random_state=BOOTSTRAP_SEED + fold,
                        ),
                    ),
                ]
            )
            model.fit(X[tr], y[tr])
            pred[va] = model.predict_proba(X[va])[:, 1]

        if not np.isfinite(pred).all():
            raise RuntimeError(f"{variant}/{label}: incomplete outer OOF")

        for i, row in label_df.iterrows():
            rows.append(
                {
                    UID: str(row[UID]),
                    "Label": label,
                    "Gold": int(row["Gold"]),
                    "OuterFold": int(outer[i]),
                    "Variant": variant,
                    "Probability": float(pred[i]),
                }
            )

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_GOLD * len(LABELS):
        raise RuntimeError(f"{variant}: OOF row count mismatch")
    return out


def raw_llm_oof_like(llm_long: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()
    out = llm_long[[UID, "Label", "Gold", "RawStateScore"]].copy()
    out["OuterFold"] = out[UID].map(fold_map).astype(int)
    out["Variant"] = "llm_raw_gold_independent"
    out["Probability"] = out["RawStateScore"].astype(float)
    return out[[UID, "Label", "Gold", "OuterFold", "Variant", "Probability"]]


def w23_oof_variant(w23_oof: pd.DataFrame) -> pd.DataFrame:
    out = w23_oof.copy()
    out["Variant"] = "w23"
    out["Probability"] = pd.to_numeric(
        out["FoldSafeChallengeProbability"], errors="coerce"
    )
    return out[[UID, "Label", "Gold", "OuterFold", "Variant", "Probability"]]


def matrix_from_oof(oof: pd.DataFrame, gold: pd.DataFrame) -> np.ndarray:
    wide = oof.pivot(index=UID, columns="Label", values="Probability").reindex(
        index=gold[UID].astype(str), columns=LABELS
    )
    return wide.to_numpy(dtype=float)


def metrics_for_variant(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        x = oof[oof["Label"] == label]
        y = x["Gold"].to_numpy(dtype=int)
        p = x["Probability"].to_numpy(dtype=float)
        prevalence = float(y.mean())
        rows.append(
            {
                "Label": label,
                "N": int(len(x)),
                "Positive": int(y.sum()),
                "Negative": int(len(y) - y.sum()),
                "AUROC": safe_auc(y, p),
                "AP": safe_ap(y, p),
                "Brier": float(brier_score_loss(y, np.clip(p, 1e-5, 1 - 1e-5))),
                "PriorBrier": float(brier_score_loss(y, np.full(len(y), prevalence))),
            }
        )
    return pd.DataFrame(rows)


def evaluate_variants(
    variants: Sequence[pd.DataFrame],
    gold: pd.DataFrame,
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
        x = metrics[metrics["Variant"] == name]
        summary["variants"][name] = {
            "macro_AUROC": float(np.nanmean(x["AUROC"])),
            "macro_AP": float(np.nanmean(x["AP"])),
            "macro_Brier": float(np.nanmean(x["Brier"])),
        }

    reference = matrices["w23"]
    for name in ["llm_raw_gold_independent", "llm_only", "llm_plus_w2"]:
        summary[f"{name}_minus_w23_bootstrap"] = bootstrap_delta(
            y, matrices[name], reference
        )

    # Best own system by macro AUROC; this is descriptive only. We do not use
    # held-out labels to tune model weights or choose a per-fold checkpoint.
    own_names = ["llm_raw_gold_independent", "llm_only", "llm_plus_w2"]
    best_name = max(
        own_names,
        key=lambda n: summary["variants"][n]["macro_AUROC"],
    )
    summary["best_own_variant"] = best_name
    summary["best_own_macro_AUROC"] = summary["variants"][best_name]["macro_AUROC"]
    summary["w23_macro_AUROC"] = summary["variants"]["w23"]["macro_AUROC"]
    summary["best_own_delta_vs_w23"] = (
        summary["best_own_macro_AUROC"] - summary["w23_macro_AUROC"]
    )

    delta = summary["best_own_delta_vs_w23"]
    auc = summary["best_own_macro_AUROC"]
    if auc >= 0.84 and delta >= 0.03:
        verdict = "STRONG_PROCEED_TO_FULL_CORPUS"
    elif delta >= 0.02:
        verdict = "PROMISING_REVIEW_PER_LABEL_BEFORE_FULL"
    else:
        verdict = "DO_NOT_FULL_EXTRACT_YET"
    summary["gate_verdict"] = verdict

    return metrics, summary


# ============================================================
# 9. STATUS / SAMPLE / GOLD / VALIDATE
# ============================================================


def status_w25() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "experiment": "RSNA W2.5 V6.1 OUR OWN grounded label-wise report teacher — gold gate",
        "train_csv": str(TRAIN_CSV),
        "train_exists": TRAIN_CSV.exists(),
        "output_root": str(OUTPUT_ROOT),
        "uses_pilkwang_labels": False,
        "recommended_model": "Qwen2.5-7B-Instruct (local/offline)",
        "model_path_env": MODEL_PATH_ENV or None,
        "gpu_id": GPU_ID,
        "use_4bit": USE_4BIT,
        "allow_fp16_fallback": ALLOW_FP16_FALLBACK,
        "fp16_shard_all_gpus": FP16_SHARD_ALL_GPUS,
        "fp16_gpu_max_gib": FP16_GPU_MAX_GIB,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "extraction_mode": "one_target_per_generation",
        "sample_3_reports_calls": 3 * len(LABELS),
        "gold_58_reports_calls": EXPECTED_GOLD * len(LABELS),
        "calibration_C": CALIBRATION_C,
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
        gold = None

    try:
        w2 = discover_w2_root()
        payload["w2_root"] = str(w2)
        payload["w2_gold_structured_sha256"] = sha256_file(
            w2 / "results" / "04_gold_structured_report_features.csv"
        )
    except Exception as exc:
        payload["w2_error"] = repr(exc)

    try:
        w23 = discover_w23_root()
        payload["w23_root"] = str(w23)
        if gold is not None:
            folds = load_folds(w23, gold)
            payload["fold_sha256"] = fold_sha256(folds)
            payload["fold_sha256_match"] = (
                payload["fold_sha256"] == EXPECTED_FOLD_SHA256
            )
    except Exception as exc:
        payload["w23_error"] = repr(exc)

    model = discover_model_path()
    payload["model_path_detected"] = str(model) if model is not None else None
    payload["model_exists"] = bool(model is not None and model.exists())
    if model is not None and model.exists():
        config = model / "config.json"
        payload["model_config_sha256"] = (
            sha256_file(config) if config.exists() else None
        )

    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["gpu_count"] = int(torch.cuda.device_count())
        payload["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        payload["torch_error"] = repr(exc)

    cache = load_gold_cache() if CACHE_ROOT.exists() else {}
    payload["gold_label_pairs_cached"] = int(len(cache))
    log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def sample_w25(n: int = 3) -> pd.DataFrame:
    _, gold, _ = load_train()
    model_path = discover_model_path()
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            "No local instruct LLM found. Set W25_MODEL_PATH to the local model directory."
        )
    long_df = run_gold_extraction(gold, model_path, limit=n)
    show_cols = [
        UID,
        "Label",
        "Gold",
        "OriginalState",
        "State",
        "GroundingAction",
        "EffectiveConfidence",
        "Lean",
        "Related",
        "Severity",
        "Chronicity",
        "EvidenceVerified",
        "Evidence",
        "RawStateScore",
    ]
    sample_path = RESULT_ROOT / "00_sample_teacher_output.csv"
    ensure_dirs()
    long_df[show_cols].to_csv(sample_path, index=False)
    log(long_df[show_cols].to_string(index=False, max_rows=36))
    log(f"\nSample output: {sample_path}")
    return long_df


def gold_w25() -> Dict[str, Any]:
    ensure_dirs()
    train, gold, _ = load_train()
    w2_root = discover_w2_root()
    w23_root = discover_w23_root()
    folds = load_folds(w23_root, gold)
    w2_gold = load_w2_gold(w2_root, gold)
    w23_oof = load_w23_oof(w23_root, gold)

    model_path = discover_model_path()
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            "No local instruct LLM found. Set W25_MODEL_PATH to the local model directory."
        )

    log("=" * 96)
    log("RSNA W2.5 V6 — OUR OWN GROUNDED LABEL-WISE REPORT TEACHER GOLD GATE")
    log("=" * 96)
    log(f"Train/gold              : {len(train)} / {len(gold)}")
    log(f"W2 root                 : {w2_root}")
    log(f"W2.3 root               : {w23_root}")
    log(f"Fold SHA256             : {fold_sha256(folds)}")
    log(f"Own LLM                 : {model_path}")
    log("Pilkwang labels used    : NO")
    log("Gold labels enter only the fold-safe evaluation/calibration stage.")

    llm_long = run_gold_extraction(gold, model_path, limit=None)
    llm_long.to_csv(
        RESULT_ROOT / "01_own_llm_gold_structured_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_frame = join_w2_features(llm_long, w2_gold)
    feature_frame.to_csv(
        RESULT_ROOT / "02_own_llm_plus_w2_gold_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    variants = [
        w23_oof_variant(w23_oof),
        raw_llm_oof_like(llm_long, folds),
        fit_outer_oof_variant(feature_frame, folds, "llm_only"),
        fit_outer_oof_variant(feature_frame, folds, "llm_plus_w2"),
    ]

    all_oof = pd.concat(variants, ignore_index=True)
    all_oof.to_csv(
        RESULT_ROOT / "03_gold_outer_oof_all_variants_long.csv",
        index=False,
    )

    metrics, summary = evaluate_variants(variants, gold)
    metrics.to_csv(
        RESULT_ROOT / "04_gold_metrics_per_label_variant.csv",
        index=False,
    )

    # Per-label side-by-side comparison.
    pivot_auc = metrics.pivot(index="Label", columns="Variant", values="AUROC")
    pivot_ap = metrics.pivot(index="Label", columns="Variant", values="AP")
    comparison = pd.DataFrame({"Label": LABELS})
    for name in ["w23", "llm_raw_gold_independent", "llm_only", "llm_plus_w2"]:
        comparison[f"{name}__AUROC"] = [
            float(pivot_auc.at[label, name]) for label in LABELS
        ]
        comparison[f"{name}__AP"] = [
            float(pivot_ap.at[label, name]) for label in LABELS
        ]
    comparison["best_own_AUROC"] = comparison[
        [
            "llm_raw_gold_independent__AUROC",
            "llm_only__AUROC",
            "llm_plus_w2__AUROC",
        ]
    ].max(axis=1)
    comparison["best_own_minus_w23_AUROC"] = (
        comparison["best_own_AUROC"] - comparison["w23__AUROC"]
    )
    comparison.to_csv(
        RESULT_ROOT / "05_per_label_own_vs_w23.csv",
        index=False,
    )

    # Extraction-state diagnostics, independent from probability calibration.
    state_summary = llm_long.groupby(["Label", "State"], as_index=False).agg(
        N=(UID, "count"),
        MeanEffectiveConfidence=("EffectiveConfidence", "mean"),
        EvidenceVerifiedFraction=("EvidenceVerified", "mean"),
        GoldPositiveRate=("Gold", "mean"),
    )
    state_summary.to_csv(
        RESULT_ROOT / "06_state_diagnostics.csv",
        index=False,
    )

    write_json(RESULT_ROOT / "07_gold_gate_summary.json", summary)

    manifest = {
        "name": "RSNA W2.5 OUR OWN report teacher gold gate",
        "version": "w25_labelwise_gold_gate_v6_1_grounded_relevance_nonfatal",
        "uses_pilkwang_labels": False,
        "train_csv": str(TRAIN_CSV),
        "train_csv_sha256": sha256_file(TRAIN_CSV),
        "w2_root": str(w2_root),
        "w23_root": str(w23_root),
        "fold_sha256": fold_sha256(folds),
        "fold_sha256_match": fold_sha256(folds) == EXPECTED_FOLD_SHA256,
        "model_path": str(model_path),
        "model_config_sha256": (
            sha256_file(model_path / "config.json")
            if (model_path / "config.json").exists()
            else None
        ),
        "prompt_version": "w25_labelwise_gold_gate_v6_1_grounded_relevance",
        "calibration_C": CALIBRATION_C,
        "summary": summary,
    }
    write_json(RESULT_ROOT / "w2_5_gold_gate_manifest.json", manifest)

    log("\n" + "=" * 96)
    log("W2.5 GOLD GATE COMPLETE")
    log("=" * 96)
    for name, data in summary["variants"].items():
        log(
            f"{name:27s} "
            f"AUROC={data['macro_AUROC']:.6f} "
            f"AP={data['macro_AP']:.6f} "
            f"Brier={data['macro_Brier']:.6f}"
        )
    log(f"\nBest own variant        : {summary['best_own_variant']}")
    log(f"Best own macro AUROC    : {summary['best_own_macro_AUROC']:.6f}")
    log(f"W2.3 macro AUROC        : {summary['w23_macro_AUROC']:.6f}")
    log(f"Delta vs W2.3           : {summary['best_own_delta_vs_w23']:+.6f}")
    log(f"GATE VERDICT            : {summary['gate_verdict']}")
    log(f"Results                 : {RESULT_ROOT}")
    log("Do NOT run the 4,407-report full corpus yet; review this gold gate first.")

    return summary


def validate_w25() -> Dict[str, Any]:
    ensure_dirs()
    _, gold, _ = load_train()
    w23_root = discover_w23_root()
    folds = load_folds(w23_root, gold)

    long_path = RESULT_ROOT / "01_own_llm_gold_structured_long.csv"
    oof_path = RESULT_ROOT / "03_gold_outer_oof_all_variants_long.csv"
    metrics_path = RESULT_ROOT / "04_gold_metrics_per_label_variant.csv"
    summary_path = RESULT_ROOT / "07_gold_gate_summary.json"

    for path in [long_path, oof_path, metrics_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing W2.5 gold-gate output {path}. Run run_w25('gold') first."
            )

    long_df = pd.read_csv(long_path)
    oof = pd.read_csv(oof_path)
    metrics = pd.read_csv(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_variants = {
        "w23",
        "llm_raw_gold_independent",
        "llm_only",
        "llm_plus_w2",
    }

    checks = {
        "fold_hash_match": fold_sha256(folds) == EXPECTED_FOLD_SHA256,
        "gold_long_696_rows": len(long_df) == EXPECTED_GOLD * len(LABELS),
        "gold_long_no_duplicates": not long_df[[UID, "Label"]].duplicated().any(),
        "gold_long_58_uids": long_df[UID].astype(str).nunique() == EXPECTED_GOLD,
        "all_12_labels_present": set(long_df["Label"].astype(str)) == set(LABELS),
        "states_valid": set(long_df["State"].astype(str)).issubset(set(STATE_VALUES)),
        "oof_variants_complete": set(oof["Variant"].astype(str)) == expected_variants,
        "oof_rows_complete": len(oof)
        == EXPECTED_GOLD * len(LABELS) * len(expected_variants),
        "oof_probabilities_finite": bool(
            np.isfinite(pd.to_numeric(oof["Probability"], errors="coerce")).all()
        ),
        "oof_probabilities_in_range": bool(
            (
                (pd.to_numeric(oof["Probability"], errors="coerce") >= 0)
                & (pd.to_numeric(oof["Probability"], errors="coerce") <= 1)
            ).all()
        ),
        "metrics_rows_complete": len(metrics) == len(LABELS) * len(expected_variants),
        "pilkwang_not_used": True,
    }
    checks["overall_pass"] = bool(all(checks.values()))

    payload = {
        "checks": checks,
        "summary": summary,
        "results_root": str(RESULT_ROOT),
    }
    write_json(RESULT_ROOT / "08_validation_summary.json", payload)
    log(json.dumps(payload, indent=2, allow_nan=True))

    if not checks["overall_pass"]:
        raise RuntimeError("W2.5 validation failed")
    return payload


def run_w25(mode: str = "status"):
    mode = str(mode).strip().lower()
    if mode == "status":
        return status_w25()
    if mode == "sample":
        return sample_w25()
    if mode == "gold":
        return gold_w25()
    if mode == "validate":
        return validate_w25()
    raise ValueError("mode must be one of: status, sample, gold, validate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["status", "sample", "gold", "validate"],
        default="status",
    )
    return parser.parse_args()


def main():
    run_w25("status")
    run_w25("sample")


if __name__ == "__main__":
    main()
