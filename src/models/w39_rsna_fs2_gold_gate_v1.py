#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W39 — FS2 Gold-Only Fold-Safe FAST Challenge Mapper v1
======================================================

Targets
-------
    Contusion
    Effusion

Purpose
-------
W38 identified Contusion and Effusion as teacher-probability problems rather
than merely masking/weighting problems.

W39 tests ONE hypothesis only:

    Can the already-proven W2.6-P FAST challenge-mapping method improve
    Contusion + Effusion over their W2.3 fold-safe teacher?

The method is intentionally copied from the successful W2.6-P FAST design:

    report
    + W2 structured features
    + outer-train-only TF-IDF retrieval
    + 2 positive exemplars
    + 2 negative exemplars
        ->
    Qwen2.5-Instruct
        ->
    direct next-token A/B logits
        ->
    FS2 probability

There is NO autoregressive JSON generation.

Controlled variants
-------------------
Only two candidate variants are evaluated:

    fast_replace
    fast_fixed50 = 0.50 * W2.3 + 0.50 * FAST

No alpha search.
No prompt sweep.
No model comparison.
No production generation.

Hard primary gate
-----------------
Primary variant is fixed50.

PASS only when ALL are true:

    1. mean AUROC improvement across Contusion + Effusion >= +0.040
    2. neither individual label regresses by more than 0.005 AUROC
    3. bootstrap P(delta > 0) >= 0.95
    4. mean Brier does not worsen by more than +0.010

Otherwise STOP.

Important
---------
- Gold only.
- Exact locked folds.
- Query gold is never included in prompts.
- Exemplars are outer-train only.
- No Pilkwang.
- No DICOM.
- No Curia.
- No image training.
- No 4,349-study production generation.
- No project .py imports.
- No environment variables are required.

CLI
---
    python w39_rsna_fs2_gold_gate_v1.py status --accelerator localGPU

    python w39_rsna_fs2_gold_gate_v1.py gold --accelerator localGPU

    python w39_rsna_fs2_gold_gate_v1.py validate

Outputs
-------
output/results/rsna_w39_fs2_gold_gate_v1/

    cache/
        w39_fs2_fast_gold_v1.jsonl

    results/
        00_outer_fold_assignments.csv
        01_exemplar_audit.csv
        02_prompt_audit.csv
        03_fs2_fast_oof_long.csv
        04_target_variants_oof_long.csv
        05_target_metrics.csv
        06_fs2_attribution.csv
        07_w39_gold_summary.json
        08_validation_summary.json
"""

from __future__ import annotations

# =============================================================================
# IMPORTS / ALLOCATOR
# =============================================================================

import os

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import argparse
import gc
import hashlib
import inspect
import json
import re
import time
import unicodedata
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# =============================================================================
# VERSION / LABELS
# =============================================================================

SCRIPT_VERSION = "w39_fs2_gold_gate_v1"

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

FS2_LABELS = [
    "Contusion",
    "Effusion",
]

EXPECTED_TRAIN = 4407
EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349

EXPECTED_FS2_GOLD_CELLS = EXPECTED_GOLD * len(FS2_LABELS)

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

# Historical W2.3 values. Used only as an identity guard.
EXPECTED_W23_AUC = {
    "Contusion": 0.703779,
    "Effusion": 0.693168,
}

W23_AUC_TOLERANCE = 0.010


# =============================================================================
# PREDECLARED EXPERIMENT SETTINGS
# =============================================================================

FIXED_BLEND_ALPHA = 0.50

# Same validated W2.6-P FAST scorer temperature.
LOGIT_TEMPERATURE = 2.0

N_POS_EXAMPLES = 2
N_NEG_EXAMPLES = 2

TFIDF_MAX_FEATURES = 30000

COMPACT_EXAMPLE_CHARS = 650
COMPACT_QUERY_CHARS = 1100

MAX_INPUT_TOKENS = 2304
SINGLE_PROMPT_FALLBACK_TOKENS = 1536

MAX_BATCH_SIZE = 24

BOOTSTRAP_REPEATS = 3000
BOOTSTRAP_SEED = 390901

# -------------------------------------------------------------------------
# HARD PASS GATE
# -------------------------------------------------------------------------

GATE_MIN_MEAN_AUC_DELTA = 0.040

# "Neither label materially regresses"
GATE_MIN_PER_LABEL_DELTA = -0.005

GATE_MIN_P_DELTA_GT_ZERO = 0.95

# Candidate mean Brier may be at most 0.01 worse.
GATE_MAX_BRIER_DELTA = 0.010


# =============================================================================
# TARGET DEFINITIONS
# =============================================================================

TARGET_DEFINITIONS = {
    "Contusion": (
        "challenge label for bone contusion / bone bruise / traumatic marrow "
        "injury. Bone-marrow edema associated with acute impact, trabecular "
        "injury, impaction, or bone bruise may support positive. Nonspecific "
        "degenerative or reactive marrow signal should not automatically be "
        "treated as contusion. Infer the exact challenge annotation convention "
        "from the supplied labeled examples."
    ),
    "Effusion": (
        "challenge label for knee joint effusion / abnormal excess "
        "intra-articular fluid. Literal report mention and challenge annotation "
        "can disagree, so do not blindly map every mention to positive and do "
        "not assume silence means negative. Infer the challenge annotation "
        "convention from the supplied labeled examples."
    ),
}


# =============================================================================
# TARGET-FOCUSED SNIPPET PATTERNS
# =============================================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

TARGET_PATTERNS = {
    "Contusion": [
        r"\bcontusion",
        r"bone bruise",
        r"bone bruis",
        r"marrow edema",
        r"marrow oedema",
        r"bone marrow edema",
        r"bone marrow oedema",
        r"trabecular",
        r"impaction",
        r"impact injury",
        r"osseous edema",
        r"osseous oedema",
        r"edema óseo",
        r"edema osseo",
        r"bone edema",
    ],
    "Effusion": [
        r"\beffusion",
        r"joint fluid",
        r"intra[- ]?articular fluid",
        r"suprapatellar fluid",
        r"joint disten",
        r"hydarth",
        r"hydrarth",
        r"hydroarth",
        r"derrame",
        r"líquido articular",
        r"liquido articular",
        r"fluid collection",
    ],
}


# =============================================================================
# W2 FEATURE COLUMNS
# =============================================================================

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


# =============================================================================
# BASIC UTILITIES
# =============================================================================


def log(msg: str = "") -> None:
    print(msg, flush=True)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )

    return " ".join(text.split())


def clean_scalar(
    value: Any,
    digits: int = 3,
) -> str:

    if pd.isna(value):
        return "NA"

    if isinstance(
        value,
        (float, np.floating),
    ):
        return f"{float(value):.{digits}f}"

    return str(value)


def stable_sha256(
    text: str,
) -> str:

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(
    path: Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def compact_report(
    text: str,
    max_chars: int,
) -> str:

    s = normalize_text(text)

    if len(s) <= max_chars:
        return s

    head = max_chars // 2
    tail = max_chars - head

    return s[:head] + " ... " + s[-tail:]


def target_focused_snippet(
    report: str,
    label: str,
    max_chars: int,
) -> str:

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

        hits = sum(
            1
            for pattern in patterns
            if re.search(
                pattern,
                s,
                flags=re.I,
            )
        )

        if hits:
            scored.append(
                (
                    hits,
                    i,
                    s,
                )
            )

    selected: List[str] = []

    if scored:

        scored.sort(
            key=lambda x: (
                -x[0],
                -x[1],
            )
        )

        for _, _, sentence in scored[:4]:

            if sentence not in selected:
                selected.append(sentence)

    # Preserve report conclusion/tail even when keyword retrieval misses
    # multilingual or unusual terminology.
    tail = text[-500:]

    if tail and tail not in selected:
        selected.append(tail)

    if not selected:
        return compact_report(
            text,
            max_chars,
        )

    return compact_report(
        " | ".join(selected),
        max_chars,
    )


def safe_auc(
    y: np.ndarray,
    p: np.ndarray,
) -> float:

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def safe_ap(
    y: np.ndarray,
    p: np.ndarray,
) -> float:

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y,
            p,
        )
    )


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:

    def convert(value):

        if isinstance(
            value,
            np.integer,
        ):
            return int(value)

        if isinstance(
            value,
            np.floating,
        ):
            return float(value)

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            Path,
        ):
            return str(value)

        raise TypeError(type(value).__name__)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# =============================================================================
# PROJECT PATHS
# =============================================================================


def script_dir() -> Path:

    try:
        return Path(__file__).resolve().parent

    except NameError:
        return Path.cwd().resolve()


def first_existing(
    candidates: Iterable[Path],
) -> Optional[Path]:

    for path in candidates:

        if path.exists():
            return path.resolve()

    return None


@dataclass
class ProjectPaths:

    project_root: Path

    train_csv: Path

    w2_root: Path

    w23_root: Path

    model_path: Path

    output_root: Path

    cache_root: Path

    result_root: Path

    @classmethod
    def discover(
        cls,
        args,
    ) -> "ProjectPaths":

        is_kaggle = Path("/kaggle/input").exists()

        if args.project_root:

            project_root = Path(args.project_root).expanduser().resolve()

        elif is_kaggle:

            project_root = Path("/kaggle/working")

        else:

            project_root = (script_dir() / ".." / "..").resolve()

        # -----------------------------------------------------------------
        # train.csv
        # -----------------------------------------------------------------

        if args.train_csv:

            train_csv = Path(args.train_csv).expanduser().resolve()

        elif is_kaggle:

            train_csv = Path(
                "/kaggle/input/competitions/"
                "rsna-knee-abnormality-detection/"
                "train.csv"
            )

        else:

            train_csv = project_root / "input" / "train.csv"

        # -----------------------------------------------------------------
        # W2
        # -----------------------------------------------------------------

        if args.w2_root:

            w2_root = Path(args.w2_root).expanduser().resolve()

        elif is_kaggle:

            w2_root = Path("/kaggle/input/datasets/" "isayem/rsna-w2/rsna_w2")

        else:

            w2_root = project_root / "output" / "results" / "rsna_w2"

        # -----------------------------------------------------------------
        # W2.3
        # -----------------------------------------------------------------

        if args.w23_root:

            w23_root = Path(args.w23_root).expanduser().resolve()

        elif is_kaggle:

            w23_root = Path("/kaggle/input/datasets/" "isayem/rsna-w2-3/rsna_w2_3")

        else:

            w23_root = project_root / "output" / "results" / "rsna_w2_3"

        # -----------------------------------------------------------------
        # Qwen model
        # -----------------------------------------------------------------

        if args.model_path:

            model_path = Path(args.model_path).expanduser().resolve()

        elif is_kaggle:

            model_path = Path("/kaggle/input/datasets/" "ragnar123/qwen2-5-7b-instruct")

        else:

            candidates = [
                project_root / "models" / "qwen2-5-7b-instruct",
                project_root / "models" / "Qwen2.5-7B-Instruct",
                project_root / "input" / "qwen2-5-7b-instruct",
            ]

            model_path = first_existing(candidates) or candidates[0]

        # -----------------------------------------------------------------
        # Output
        # -----------------------------------------------------------------

        if args.output_root:

            output_root = Path(args.output_root).expanduser().resolve()

        elif is_kaggle:

            output_root = Path("/kaggle/working/" "rsna_w39_fs2_gold_gate_v1")

        else:

            output_root = (
                project_root / "output" / "results" / "rsna_w39_fs2_gold_gate_v1"
            )

        cache_root = output_root / "cache"

        result_root = output_root / "results"

        return cls(
            project_root=project_root,
            train_csv=train_csv,
            w2_root=w2_root,
            w23_root=w23_root,
            model_path=model_path,
            output_root=output_root,
            cache_root=cache_root,
            result_root=result_root,
        )


def ensure_dirs(
    paths: ProjectPaths,
) -> None:

    paths.cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.result_root.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# ACCELERATOR
# =============================================================================


def normalize_accelerator(
    value: str,
) -> str:

    x = str(value or "auto").strip().lower().replace("-", "_")

    aliases = {
        "localgpu": "local_gpu",
        "local": "local_gpu",
        "gpu": "local_gpu",
        "cuda": "local_gpu",
        "t4": "kaggle_t4",
        "kaggle": "kaggle_t4",
        "kagglegpu": "kaggle_t4",
        "apple": "apple_mps",
        "mps": "apple_mps",
        "mac": "apple_mps",
        "kaggle_tpu": "tpu",
        "v5e": "tpu",
    }

    return aliases.get(
        x,
        x,
    )


def mps_available() -> bool:

    try:

        import torch

        return bool(
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        )

    except Exception:
        return False


def tpu_detected() -> bool:

    try:

        import torch_xla.core.xla_model  # noqa

        return True

    except Exception:
        return False


def resolve_accelerator(
    requested: str,
) -> str:

    req = normalize_accelerator(requested)

    allowed = {
        "auto",
        "local_gpu",
        "kaggle_t4",
        "apple_mps",
        "cpu",
        "tpu",
    }

    if req not in allowed:

        raise ValueError(
            f"Unknown accelerator {requested!r}. "
            "Use auto, localGPU, kaggle_t4, "
            "apple_mps, cpu, or kaggle_tpu."
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

            if (
                Path("/kaggle/input").exists()
                and len(names) >= 2
                and all("t4" in name for name in names)
            ):
                return "kaggle_t4"

            return "local_gpu"

    except Exception:
        pass

    if mps_available():
        return "apple_mps"

    if tpu_detected():
        return "tpu"

    return "cpu"


def hard_cleanup(
    torch_module,
    accelerator: str,
) -> None:

    gc.collect()

    if accelerator in {
        "local_gpu",
        "kaggle_t4",
    }:

        if torch_module.cuda.is_available():

            torch_module.cuda.empty_cache()

    elif accelerator == "apple_mps":

        try:
            torch_module.mps.empty_cache()

        except Exception:
            pass


# =============================================================================
# DATA / FOLD LOADERS
# =============================================================================


def fold_sha256(
    frame: pd.DataFrame,
) -> str:

    x = frame[
        [
            UID,
            "OuterFold",
        ]
    ].copy()

    x[UID] = x[UID].astype(str)

    x["OuterFold"] = x["OuterFold"].astype(int)

    x = x.sort_values(UID)

    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(
            x[UID],
            x["OuterFold"],
        )
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_train(
    paths: ProjectPaths,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    if not paths.train_csv.exists():

        raise FileNotFoundError(f"train.csv not found: " f"{paths.train_csv}")

    train = pd.read_csv(paths.train_csv)

    train[UID] = train[UID].astype(str)

    required = {
        UID,
        REPORT,
        *LABELS,
    }

    missing = required - set(train.columns)

    if missing:

        raise RuntimeError(f"train.csv missing columns: " f"{sorted(missing)}")

    is_gold = train[LABELS].notna().all(axis=1)

    is_unlabeled = train[LABELS].isna().all(axis=1)

    partial = ~(is_gold | is_unlabeled)

    if partial.any():

        raise RuntimeError(
            "Unexpected partially labeled rows: " f"{int(partial.sum())}"
        )

    gold = train[is_gold].copy().sort_values(UID).reset_index(drop=True)

    unlabeled = train[is_unlabeled].copy().sort_values(UID).reset_index(drop=True)

    observed = (
        len(train),
        len(gold),
        len(unlabeled),
    )

    expected = (
        EXPECTED_TRAIN,
        EXPECTED_GOLD,
        EXPECTED_UNLABELED,
    )

    if observed != expected:

        raise RuntimeError(
            "Unexpected train/gold/unlabeled counts: "
            f"{observed}; expected={expected}"
        )

    return (
        train,
        gold,
        unlabeled,
    )


def validate_source_roots(
    paths: ProjectPaths,
) -> None:

    required_w2 = [
        paths.w2_root / "results" / "04_gold_structured_report_features.csv",
    ]

    required_w23 = [
        paths.w23_root / "results" / "00_outer_fold_assignments.csv",
        paths.w23_root / "folds" / "fold_1" / "heldout_gold_stage_b_predictions.csv",
        paths.w23_root / "folds" / "fold_5" / "heldout_gold_stage_b_predictions.csv",
    ]

    missing = [
        str(path)
        for path in [
            *required_w2,
            *required_w23,
        ]
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required input artifacts:\n" + "\n".join(missing)
        )


def load_folds(
    paths: ProjectPaths,
    gold: pd.DataFrame,
) -> pd.DataFrame:

    path = paths.w23_root / "results" / "00_outer_fold_assignments.csv"

    folds = pd.read_csv(path)

    folds[UID] = folds[UID].astype(str)

    folds = (
        folds[
            [
                UID,
                "OuterFold",
            ]
        ]
        .sort_values(UID)
        .reset_index(drop=True)
    )

    if set(folds[UID]) != set(gold[UID]):

        raise RuntimeError("W2.3 fold UID set mismatch")

    digest = fold_sha256(folds)

    if digest != EXPECTED_FOLD_SHA256:

        raise RuntimeError(f"Fold SHA mismatch: {digest}")

    return folds


def load_w2_gold(
    paths: ProjectPaths,
    gold: pd.DataFrame,
) -> pd.DataFrame:

    path = paths.w2_root / "results" / "04_gold_structured_report_features.csv"

    frame = pd.read_csv(path)

    frame[UID] = frame[UID].astype(str)

    required = {
        UID,
        "Label",
        *W2_REQUIRED_COLUMNS,
    }

    missing = required - set(frame.columns)

    if missing:

        raise RuntimeError("W2 gold structured table missing: " f"{sorted(missing)}")

    if (
        frame[
            [
                UID,
                "Label",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError("W2 gold table has duplicate UID/Label rows")

    expected_rows = EXPECTED_GOLD * len(LABELS)

    if len(frame) != expected_rows:

        raise RuntimeError(f"W2 gold rows={len(frame)}, " f"expected={expected_rows}")

    if set(frame[UID]) != set(gold[UID]):

        raise RuntimeError("W2 gold UID mismatch")

    return frame


def load_w23_oof(
    paths: ProjectPaths,
    gold: pd.DataFrame,
) -> pd.DataFrame:

    pieces = []

    for fold in range(
        1,
        6,
    ):

        path = (
            paths.w23_root
            / "folds"
            / f"fold_{fold}"
            / "heldout_gold_stage_b_predictions.csv"
        )

        frame = pd.read_csv(path)

        frame[UID] = frame[UID].astype(str)

        required = {
            UID,
            "Label",
            "Gold",
            "FoldSafeChallengeProbability",
        }

        missing = required - set(frame.columns)

        if missing:

            raise RuntimeError(f"{path} missing " f"{sorted(missing)}")

        frame = frame[
            [
                UID,
                "Label",
                "Gold",
                "FoldSafeChallengeProbability",
            ]
        ].copy()

        frame["OuterFold"] = fold

        pieces.append(frame)

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    expected_rows = EXPECTED_GOLD * len(LABELS)

    if (
        len(result) != expected_rows
        or result[
            [
                UID,
                "Label",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError("Invalid W2.3 OOF table")

    if set(result[UID]) != set(gold[UID]):

        raise RuntimeError("W2.3 OOF UID mismatch")

    # Identity guard on the two W39 targets.
    for label in FS2_LABELS:

        subset = result[result["Label"] == label]

        observed = safe_auc(
            subset["Gold"].to_numpy(dtype=int),
            subset["FoldSafeChallengeProbability"].to_numpy(dtype=float),
        )

        expected = EXPECTED_W23_AUC[label]

        if abs(observed - expected) > W23_AUC_TOLERANCE:

            raise RuntimeError(
                f"W2.3 identity guard failed "
                f"for {label}: "
                f"observed={observed:.6f}, "
                f"expected≈{expected:.6f}"
            )

    return result


# =============================================================================
# W2 SUMMARIES
# =============================================================================


def build_feature_lookup(
    frame: pd.DataFrame,
) -> Dict[
    Tuple[str, str],
    pd.Series,
]:

    return {
        (
            str(row[UID]),
            str(row["Label"]),
        ): row
        for _, row in frame.iterrows()
    }


def dedup_evidence(
    row: pd.Series,
) -> List[str]:

    values: List[str] = []

    seen = set()

    for column in [
        "FusedEvidence",
        "SemanticPositiveEvidence",
        "SemanticNegativeEvidence",
        "SemanticRelatedEvidence",
    ]:

        value = row.get(column)

        if pd.notna(value) and str(value).strip():

            text = normalize_text(value)

            key = text.casefold()

            if key not in seen:

                seen.add(key)

                values.append(text)

    return values


def w2_retrieval_summary(
    row: pd.Series,
) -> str:

    evidence = dedup_evidence(row)

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

    if evidence:

        parts.append("ev=" + " || ".join(evidence[:3]))

    return "; ".join(parts)


def w2_prompt_summary(
    row: pd.Series,
) -> str:

    evidence = dedup_evidence(row)

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

    if evidence:

        compact_ev = " || ".join(evidence[:2])

        parts.append(
            "ev="
            + compact_report(
                compact_ev,
                240,
            )
        )

    return "; ".join(parts)


# =============================================================================
# RETRIEVAL
# =============================================================================


def retrieval_document(
    report: str,
    feature_summary: str,
    label: str,
) -> str:

    return (
        f"target {label} "
        f"target {label} "
        f"{normalize_text(feature_summary)} "
        f"{normalize_text(report)}"
    )


def select_top_by_class(
    reference: pd.DataFrame,
    similarities: np.ndarray,
    label: str,
) -> List[int]:

    tmp = reference[
        [
            UID,
            label,
        ]
    ].copy()

    tmp["Similarity"] = similarities

    tmp["_idx"] = np.arange(len(tmp))

    tmp = tmp.sort_values(
        [
            "Similarity",
            UID,
        ],
        ascending=[
            False,
            True,
        ],
    )

    positives = tmp[tmp[label] == 1].head(N_POS_EXAMPLES)

    negatives = tmp[tmp[label] == 0].head(N_NEG_EXAMPLES)

    chosen = pd.concat(
        [
            positives,
            negatives,
        ],
        ignore_index=True,
    )

    chosen = chosen.sort_values(
        [
            "Similarity",
            UID,
        ],
        ascending=[
            False,
            True,
        ],
    )

    return chosen["_idx"].astype(int).tolist()


def select_gold_fold_exemplars(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_lookup: Dict[
        Tuple[str, str],
        pd.Series,
    ],
    fold: int,
    label: str,
) -> Tuple[
    Dict[
        str,
        List[Dict[str, Any]],
    ],
    pd.DataFrame,
]:

    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()

    tmp = gold[
        [
            UID,
            REPORT,
            label,
        ]
    ].copy()

    tmp["OuterFold"] = tmp[UID].map(fold_map).astype(int)

    outer_train = (
        tmp[tmp["OuterFold"] != fold].copy().sort_values(UID).reset_index(drop=True)
    )

    heldout = (
        tmp[tmp["OuterFold"] == fold].copy().sort_values(UID).reset_index(drop=True)
    )

    if int((outer_train[label] == 1).sum()) < N_POS_EXAMPLES:

        raise RuntimeError(f"{label}/fold{fold}: " "insufficient positives")

    if int((outer_train[label] == 0).sum()) < N_NEG_EXAMPLES:

        raise RuntimeError(f"{label}/fold{fold}: " "insufficient negatives")

    documents = []

    for _, row in outer_train.iterrows():

        uid = str(row[UID])

        features = w2_retrieval_summary(
            w2_lookup[
                (
                    uid,
                    label,
                )
            ]
        )

        documents.append(
            retrieval_document(
                str(row[REPORT]),
                features,
                label,
            )
        )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(
            3,
            5,
        ),
        min_df=1,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
    )

    x_train = vectorizer.fit_transform(documents)

    selected = {}

    audit_rows = []

    for _, query in heldout.iterrows():

        query_uid = str(query[UID])

        query_features = w2_retrieval_summary(
            w2_lookup[
                (
                    query_uid,
                    label,
                )
            ]
        )

        query_doc = retrieval_document(
            str(query[REPORT]),
            query_features,
            label,
        )

        query_vector = vectorizer.transform([query_doc])

        similarities = (x_train @ query_vector.T).toarray().ravel()

        chosen_indices = select_top_by_class(
            outer_train,
            similarities,
            label,
        )

        examples = []

        for rank, j in enumerate(
            chosen_indices,
            start=1,
        ):

            example = outer_train.iloc[j]

            example_uid = str(example[UID])

            examples.append(
                {
                    UID: example_uid,
                    "Gold": int(example[label]),
                    "Similarity": float(similarities[j]),
                    "Report": str(example[REPORT]),
                    "W2Summary": w2_prompt_summary(
                        w2_lookup[
                            (
                                example_uid,
                                label,
                            )
                        ]
                    ),
                }
            )

            audit_rows.append(
                {
                    "OuterFold": fold,
                    "QueryStudyInstanceUID": query_uid,
                    "Label": label,
                    "ExampleRank": rank,
                    "ExampleStudyInstanceUID": example_uid,
                    "ExampleGold": int(example[label]),
                    "Similarity": float(similarities[j]),
                }
            )

        selected[query_uid] = examples

    return (
        selected,
        pd.DataFrame(audit_rows),
    )


def build_all_exemplars(
    gold: pd.DataFrame,
    folds: pd.DataFrame,
    w2_gold: pd.DataFrame,
) -> Tuple[
    Dict[
        Tuple[str, str],
        List[Dict[str, Any]],
    ],
    pd.DataFrame,
]:

    lookup = build_feature_lookup(w2_gold)

    all_selected = {}

    audits = []

    for fold in range(
        1,
        6,
    ):

        for label in FS2_LABELS:

            selected, audit = select_gold_fold_exemplars(
                gold,
                folds,
                lookup,
                fold,
                label,
            )

            for uid, examples in selected.items():

                all_selected[
                    (
                        uid,
                        label,
                    )
                ] = examples

            audits.append(audit)

    result = pd.concat(
        audits,
        ignore_index=True,
    )

    if len(all_selected) != EXPECTED_FS2_GOLD_CELLS:

        raise RuntimeError("Gold exemplar-set count mismatch: " f"{len(all_selected)}")

    expected_audit_rows = EXPECTED_FS2_GOLD_CELLS * (N_POS_EXAMPLES + N_NEG_EXAMPLES)

    if len(result) != expected_audit_rows:

        raise RuntimeError(
            "Exemplar audit row mismatch: "
            f"{len(result)} vs "
            f"{expected_audit_rows}"
        )

    return (
        all_selected,
        result,
    )


# =============================================================================
# FAST PROMPT
# =============================================================================

SYSTEM_PROMPT = (
    "You map knee MRI reports to THIS competition's binary labels. "
    "Infer the challenge annotation convention from labeled examples. "
    "Do not equate silence with negative. "
    "You will choose A or B only."
)


def prompt_report(
    report: str,
    label: str,
    is_query: bool,
) -> str:

    max_chars = COMPACT_QUERY_CHARS if is_query else COMPACT_EXAMPLE_CHARS

    return target_focused_snippet(
        report,
        label,
        max_chars,
    )


def build_fast_prompt(
    label: str,
    query_report: str,
    query_w2: str,
    examples: Sequence[Mapping[str, Any]],
    prior: float,
) -> str:

    chunks = [
        f"TARGET={label}",
        ("DEFINITION=" + TARGET_DEFINITIONS[label]),
        ("REFERENCE_POSITIVE_RATE=" f"{prior:.3f}"),
        "A = challenge label 0",
        "B = challenge label 1",
        "",
        "FOLD-SAFE EXAMPLES:",
    ]

    for i, example in enumerate(
        examples,
        start=1,
    ):

        answer = "B" if int(example["Gold"]) == 1 else "A"

        chunks += [
            (
                f"E{i}: "
                f"answer={answer}; "
                f"sim="
                f"{float(example['Similarity']):.3f}; "
                f"w2="
                f"{example['W2Summary']}"
            ),
            (
                "report="
                + prompt_report(
                    example["Report"],
                    label,
                    is_query=False,
                )
            ),
        ]

    chunks += [
        "",
        ("QUERY_W2=" + query_w2),
        (
            "QUERY_REPORT="
            + prompt_report(
                query_report,
                label,
                is_query=True,
            )
        ),
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
    exemplars: Dict[
        Tuple[str, str],
        List[Dict[str, Any]],
    ],
) -> List[Dict[str, Any]]:

    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()

    w2_lookup = build_feature_lookup(w2_gold)

    gold_lookup = gold.set_index(UID)

    gold_uids = gold[UID].astype(str).tolist()

    queries = []

    for uid in gold_uids:

        fold = int(fold_map[uid])

        outer_train_uids = [
            other_uid for other_uid in gold_uids if int(fold_map[other_uid]) != fold
        ]

        for label in FS2_LABELS:

            prior = float(
                gold_lookup.loc[
                    outer_train_uids,
                    label,
                ]
                .astype(float)
                .mean()
            )

            prompt = build_fast_prompt(
                label=label,
                query_report=str(
                    gold_lookup.at[
                        uid,
                        REPORT,
                    ]
                ),
                query_w2=w2_prompt_summary(
                    w2_lookup[
                        (
                            uid,
                            label,
                        )
                    ]
                ),
                examples=exemplars[
                    (
                        uid,
                        label,
                    )
                ],
                prior=prior,
            )

            queries.append(
                {
                    UID: uid,
                    "Label": label,
                    "Gold": int(
                        gold_lookup.at[
                            uid,
                            label,
                        ]
                    ),
                    "OuterFold": fold,
                    "Prior": prior,
                    "Prompt": prompt,
                    "PromptSHA256": stable_sha256(prompt),
                    "PromptChars": len(prompt),
                    "ExampleUIDs": "|".join(
                        str(x[UID])
                        for x in exemplars[
                            (
                                uid,
                                label,
                            )
                        ]
                    ),
                }
            )

    if len(queries) != EXPECTED_FS2_GOLD_CELLS:

        raise RuntimeError("Gold query count mismatch: " f"{len(queries)}")

    return queries


# =============================================================================
# FAST LOGIT MAPPER
# =============================================================================


class FastLogitMapper:

    def __init__(
        self,
        model_path: Path,
        output_root: Path,
        accelerator: str,
        precision: str,
    ):

        try:

            import torch

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

        except Exception as exc:

            raise RuntimeError(
                "W39 requires torch, transformers, "
                "accelerate and optionally bitsandbytes."
            ) from exc

        self.torch = torch

        self.model_path = Path(model_path)

        self.output_root = Path(output_root)

        self.accelerator = resolve_accelerator(accelerator)

        self.precision = str(precision or "auto").strip().lower()

        if not self.model_path.exists():

            raise FileNotFoundError(f"Model path not found: " f"{self.model_path}")

        if self.accelerator == "tpu":

            raise RuntimeError(
                "kaggle_tpu/TPU is not implemented or validated "
                "for this Qwen decoder scoring path. "
                "Use localGPU or kaggle_t4."
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

        self.answer_zero, self.answer_one = self.choose_verbalizers()

        common = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        log(f"Loading FAST mapper       : " f"{self.model_path}")

        log(f"Accelerator resolved      : " f"{self.accelerator}")

        log(f"Precision policy          : " f"{self.precision}")

        log(
            f"Answer tokens             : "
            f"{self.answer_zero[0]!r}/"
            f"{self.answer_one[0]!r}"
        )

        # -----------------------------------------------------------------
        # CUDA
        # -----------------------------------------------------------------

        if self.accelerator == "local_gpu":

            if not torch.cuda.is_available():

                raise RuntimeError("localGPU requested but CUDA is unavailable.")

            gpu_count = torch.cuda.device_count()

            if gpu_count < 1:

                raise RuntimeError("No visible CUDA devices.")

            log(f"Visible CUDA GPUs         : " f"{gpu_count}")

            # -------------------------------------------------------------
            # Prefer 4-bit.
            #
            # device_map='auto' means every visible GPU is available to the
            # loader. No W39_GPU_IDS environment variable is required.
            # -------------------------------------------------------------

            loaded = False

            if self.precision in {
                "auto",
                "4bit",
                "nf4",
            }:

                try:

                    compute_dtype = (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )

                    max_memory = {}

                    for i in range(gpu_count):

                        try:

                            _, total_b = torch.cuda.mem_get_info(i)

                            total_gib = total_b / (1024**3)

                        except Exception:

                            total_gib = 16.0

                        cap = max(
                            4.0,
                            total_gib - 2.5,
                        )

                        max_memory[i] = f"{cap:.1f}GiB"

                    max_memory["cpu"] = "36GiB"

                    kwargs = dict(common)

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )

                    kwargs["device_map"] = "auto"

                    kwargs["max_memory"] = max_memory

                    kwargs["attn_implementation"] = "sdpa"

                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        **kwargs,
                    )

                    self.load_mode = f"4bit_nf4_auto_{gpu_count}gpu"

                    loaded = True

                except Exception as exc:

                    if self.precision in {
                        "4bit",
                        "nf4",
                    }:

                        raise RuntimeError(
                            "Explicit 4-bit model loading failed."
                        ) from exc

                    warnings.warn(
                        "4-bit loading unavailable; "
                        "falling back to FP16 + automatic CPU/GPU offload. "
                        f"Original error: {exc}"
                    )

            if not loaded:

                max_memory = {}

                for i in range(gpu_count):

                    try:

                        _, total_b = torch.cuda.mem_get_info(i)

                        total_gib = total_b / (1024**3)

                    except Exception:

                        total_gib = 16.0

                    cap = max(
                        3.5,
                        min(
                            11.5,
                            total_gib - 3.5,
                        ),
                    )

                    max_memory[i] = f"{cap:.1f}GiB"

                max_memory["cpu"] = "36GiB"

                offload_dir = self.output_root / "model_offload"

                offload_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                kwargs = dict(common)

                kwargs["dtype"] = torch.float16

                kwargs["device_map"] = "auto"

                kwargs["max_memory"] = max_memory

                kwargs["offload_folder"] = str(offload_dir)

                kwargs["offload_state_dict"] = True

                kwargs["attn_implementation"] = "sdpa"

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **kwargs,
                )

                self.load_mode = f"fp16_auto_{gpu_count}gpu_cpu_offload"

        # -----------------------------------------------------------------
        # Kaggle T4
        # -----------------------------------------------------------------

        elif self.accelerator == "kaggle_t4":

            if not torch.cuda.is_available():

                raise RuntimeError("kaggle_t4 requested but CUDA is unavailable.")

            gpu_count = torch.cuda.device_count()

            if gpu_count < 1:

                raise RuntimeError("No visible CUDA devices.")

            kwargs = dict(common)

            kwargs["dtype"] = torch.float16

            kwargs["device_map"] = "balanced" if gpu_count > 1 else "auto"

            kwargs["max_memory"] = {i: "11.5GiB" for i in range(gpu_count)}

            kwargs["attn_implementation"] = "sdpa"

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

            self.load_mode = f"fp16_balanced_{gpu_count}gpu"

        # -----------------------------------------------------------------
        # Apple MPS
        # -----------------------------------------------------------------

        elif self.accelerator == "apple_mps":

            if not mps_available():

                raise RuntimeError("apple_mps requested but MPS is unavailable.")

            kwargs = dict(common)

            kwargs["dtype"] = torch.float16

            kwargs["attn_implementation"] = "sdpa"

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

            self.model.to("mps")

            self.load_mode = "fp16_apple_mps"

        # -----------------------------------------------------------------
        # CPU
        # -----------------------------------------------------------------

        elif self.accelerator == "cpu":

            kwargs = dict(common)

            kwargs["dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

            self.load_mode = "fp32_cpu"

        else:

            raise RuntimeError(f"Unhandled accelerator: " f"{self.accelerator}")

        self.model.eval()

        self.input_device = self.model.get_input_embeddings().weight.device

        try:

            parameters = inspect.signature(self.model.forward).parameters

            self.supports_logits_to_keep = "logits_to_keep" in parameters

        except Exception:

            self.supports_logits_to_keep = False

        self.token_budget = self.resolve_token_budget()

        log(f"Model load mode           : " f"{self.load_mode}")

        log(f"Input device              : " f"{self.input_device}")

        log(f"logits_to_keep support    : " f"{self.supports_logits_to_keep}")

        log(f"Token batch budget        : " f"{self.token_budget}")

    def choose_verbalizers(
        self,
    ) -> Tuple[
        Tuple[str, int],
        Tuple[str, int],
    ]:

        candidates = [
            (
                "A",
                "B",
            ),
            (
                "0",
                "1",
            ),
            (
                "N",
                "Y",
            ),
        ]

        for zero, one in candidates:

            zero_ids = self.tokenizer.encode(
                zero,
                add_special_tokens=False,
            )

            one_ids = self.tokenizer.encode(
                one,
                add_special_tokens=False,
            )

            if len(zero_ids) == 1 and len(one_ids) == 1 and zero_ids[0] != one_ids[0]:

                return (
                    (
                        zero,
                        int(zero_ids[0]),
                    ),
                    (
                        one,
                        int(one_ids[0]),
                    ),
                )

        raise RuntimeError("Could not find single-token binary verbalizers.")

    def resolve_token_budget(
        self,
    ) -> int:

        if self.accelerator == "local_gpu":

            if "4bit" in self.load_mode:

                return 18000

            return 9000

        if self.accelerator == "kaggle_t4":

            return 8000

        if self.accelerator == "apple_mps":

            return 3500

        return 4000

    def chat_prompt(
        self,
        prompt: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if getattr(
            self.tokenizer,
            "chat_template",
            None,
        ):

            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return SYSTEM_PROMPT + "\n\n" + prompt

    def prepare_lengths(
        self,
        prompts: Sequence[str],
    ) -> List[int]:

        chat = [self.chat_prompt(prompt) for prompt in prompts]

        encoded = self.tokenizer(
            chat,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=False,
        )

        return [len(tokens) for tokens in encoded["input_ids"]]

    def forward_batch(
        self,
        prompts: Sequence[str],
        token_limit: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        torch = self.torch

        chat = [self.chat_prompt(prompt) for prompt in prompts]

        encoded = self.tokenizer(
            chat,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(token_limit),
        )

        input_lengths = encoded["attention_mask"].sum(dim=1).cpu().numpy().astype(int)

        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}

        kwargs = {
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

                raise RuntimeError("Unexpected logits shape: " f"{tuple(logits.shape)}")

            last = logits[
                :,
                -1,
                :,
            ]

            zero_id = self.answer_zero[1]

            one_id = self.answer_one[1]

            pair = torch.stack(
                [
                    last[
                        :,
                        zero_id,
                    ],
                    last[
                        :,
                        one_id,
                    ],
                ],
                dim=1,
            ).float()

            pair = pair / float(LOGIT_TEMPERATURE)

            probabilities = torch.softmax(
                pair,
                dim=1,
            )[
                :,
                1,
            ]

            margins = (
                pair[
                    :,
                    1,
                ]
                - pair[
                    :,
                    0,
                ]
            )

            return (
                probabilities.detach().cpu().numpy().astype(float),
                margins.detach().cpu().numpy().astype(float),
                input_lengths,
            )

        finally:

            try:
                del encoded
            except Exception:
                pass

            try:
                del outputs
            except Exception:
                pass

    def score_batch_recursive(
        self,
        prompts: Sequence[str],
        token_limit: int = MAX_INPUT_TOKENS,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        try:

            return self.forward_batch(
                prompts,
                token_limit,
            )

        except self.torch.cuda.OutOfMemoryError:

            hard_cleanup(
                self.torch,
                self.accelerator,
            )

            if len(prompts) > 1:

                middle = len(prompts) // 2

                log(
                    f"  OOM: splitting "
                    f"{len(prompts)} "
                    f"-> {middle}+"
                    f"{len(prompts)-middle}"
                )

                left = self.score_batch_recursive(
                    prompts[:middle],
                    token_limit,
                )

                right = self.score_batch_recursive(
                    prompts[middle:],
                    token_limit,
                )

                return (
                    np.concatenate(
                        [
                            left[0],
                            right[0],
                        ]
                    ),
                    np.concatenate(
                        [
                            left[1],
                            right[1],
                        ]
                    ),
                    np.concatenate(
                        [
                            left[2],
                            right[2],
                        ]
                    ),
                )

            if token_limit > SINGLE_PROMPT_FALLBACK_TOKENS:

                return self.score_batch_recursive(
                    prompts,
                    SINGLE_PROMPT_FALLBACK_TOKENS,
                )

            raise

        except RuntimeError as exc:

            message = str(exc).casefold()

            if self.accelerator == "apple_mps" and (
                "out of memory" in message or "mps backend" in message
            ):

                hard_cleanup(
                    self.torch,
                    self.accelerator,
                )

                if len(prompts) > 1:

                    middle = len(prompts) // 2

                    left = self.score_batch_recursive(
                        prompts[:middle],
                        token_limit,
                    )

                    right = self.score_batch_recursive(
                        prompts[middle:],
                        token_limit,
                    )

                    return (
                        np.concatenate(
                            [
                                left[0],
                                right[0],
                            ]
                        ),
                        np.concatenate(
                            [
                                left[1],
                                right[1],
                            ]
                        ),
                        np.concatenate(
                            [
                                left[2],
                                right[2],
                            ]
                        ),
                    )

                if token_limit > SINGLE_PROMPT_FALLBACK_TOKENS:

                    return self.score_batch_recursive(
                        prompts,
                        SINGLE_PROMPT_FALLBACK_TOKENS,
                    )

            raise


# =============================================================================
# LENGTH-AWARE BATCHING
# =============================================================================


def make_length_aware_batches(
    items: Sequence[Mapping[str, Any]],
    lengths: Sequence[int],
    token_budget: int,
) -> List[List[int]]:

    order = sorted(
        range(len(items)),
        key=lambda i: (
            int(lengths[i]),
            str(
                items[i].get(
                    "Label",
                    "",
                )
            ),
        ),
    )

    batches = []

    current = []

    current_max = 0

    for index in order:

        length = int(lengths[index])

        proposed_max = max(
            current_max,
            length,
        )

        proposed_n = len(current) + 1

        padded_tokens = proposed_max * proposed_n

        if current and (proposed_n > MAX_BATCH_SIZE or padded_tokens > token_budget):

            batches.append(current)

            current = []

            current_max = 0

        current.append(index)

        current_max = max(
            current_max,
            length,
        )

    if current:

        batches.append(current)

    return batches


# =============================================================================
# RESUMABLE CACHE
# =============================================================================


def cache_path(
    paths: ProjectPaths,
) -> Path:

    return paths.cache_root / "w39_fs2_fast_gold_v1.jsonl"


def read_cache(
    path: Path,
) -> Dict[
    str,
    Dict[str, Any],
]:

    result = {}

    if not path.exists():
        return result

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            try:

                row = json.loads(line)

                key = f"{row[UID]}" f"|||" f"{row['Label']}"

                result[key] = row

            except Exception as exc:

                raise RuntimeError(f"Corrupt cache line " f"{line_no}: {exc}") from exc

    return result


def append_cache(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                )
                + "\n"
            )


def scorer_signature(
    paths: ProjectPaths,
    accelerator: str,
    precision: str,
) -> str:

    config_path = paths.model_path / "config.json"

    model_sha = sha256_file(config_path) if config_path.exists() else "no-config"

    payload = {
        "script_version": SCRIPT_VERSION,
        "model_config_sha256": model_sha,
        "accelerator": resolve_accelerator(accelerator),
        "precision": precision,
        "logit_temperature": LOGIT_TEMPERATURE,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "fs2_labels": FS2_LABELS,
        "target_definitions": TARGET_DEFINITIONS,
    }

    return stable_sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
        )
    )


def run_fast_scoring(
    paths: ProjectPaths,
    queries: Sequence[Mapping[str, Any]],
    accelerator: str,
    precision: str,
) -> pd.DataFrame:

    ensure_dirs(paths)

    path = cache_path(paths)

    cache = read_cache(path)

    signature = scorer_signature(
        paths,
        accelerator,
        precision,
    )

    valid = {}

    needed = []

    for query in queries:

        key = f"{query[UID]}" f"|||" f"{query['Label']}"

        row = cache.get(key)

        if (
            row is not None
            and str(row.get("PromptSHA256")) == str(query["PromptSHA256"])
            and str(row.get("ScorerSignature")) == signature
        ):

            valid[key] = row

        else:

            needed.append(query)

    log(f"FAST query cells          : " f"{len(queries)}")

    log(f"Valid cached cells        : " f"{len(valid)}")

    log(f"New cells                 : " f"{len(needed)}")

    if needed:

        mapper = FastLogitMapper(
            model_path=paths.model_path,
            output_root=paths.output_root,
            accelerator=accelerator,
            precision=precision,
        )

        lengths = mapper.prepare_lengths([str(query["Prompt"]) for query in needed])

        batches = make_length_aware_batches(
            needed,
            lengths,
            mapper.token_budget,
        )

        log(f"Length-aware batches      : " f"{len(batches)}")

        if lengths:

            array = np.asarray(lengths)

            log(
                "Token lengths "
                "p50/p90/p99/max       : "
                f"{np.quantile(array, .50):.0f}/"
                f"{np.quantile(array, .90):.0f}/"
                f"{np.quantile(array, .99):.0f}/"
                f"{array.max()}"
            )

        started = time.time()

        completed = 0

        for batch_number, indices in enumerate(
            batches,
            start=1,
        ):

            batch_queries = [needed[i] for i in indices]

            prompts = [str(query["Prompt"]) for query in batch_queries]

            t0 = time.time()

            (
                probabilities,
                margins,
                actual_lengths,
            ) = mapper.score_batch_recursive(prompts)

            rows = []

            for (
                query,
                probability,
                margin,
                input_length,
            ) in zip(
                batch_queries,
                probabilities,
                margins,
                actual_lengths,
            ):

                row = {
                    UID: str(query[UID]),
                    "Label": str(query["Label"]),
                    "Gold": int(query["Gold"]),
                    "OuterFold": int(query["OuterFold"]),
                    "Prior": float(query["Prior"]),
                    "FastProbability": float(probability),
                    "FastLogitMargin": float(margin),
                    "InputTokens": int(input_length),
                    "PromptChars": int(query["PromptChars"]),
                    "PromptSHA256": str(query["PromptSHA256"]),
                    "ScorerSignature": signature,
                    "LogitTemperature": LOGIT_TEMPERATURE,
                    "Accelerator": mapper.accelerator,
                    "LoadMode": mapper.load_mode,
                    "Verbalizer0": mapper.answer_zero[0],
                    "Verbalizer1": mapper.answer_one[0],
                    "ScriptVersion": SCRIPT_VERSION,
                }

                rows.append(row)

            append_cache(
                path,
                rows,
            )

            for row in rows:

                key = f"{row[UID]}" f"|||" f"{row['Label']}"

                valid[key] = row

            completed += len(rows)

            elapsed = time.time() - started

            rate = completed / max(
                elapsed,
                1e-9,
            )

            if batch_number <= 5 or batch_number % 10 == 0 or completed == len(needed):

                log(
                    f"  scored "
                    f"{completed:>3}/"
                    f"{len(needed)} "
                    f"batch={len(rows):>2} "
                    f"maxTok="
                    f"{max(actual_lengths):>4} "
                    f"call="
                    f"{time.time()-t0:5.2f}s "
                    f"rate="
                    f"{rate*60:6.1f} cells/min"
                )

        del mapper

        gc.collect()

    result_rows = []

    for query in queries:

        key = f"{query[UID]}" f"|||" f"{query['Label']}"

        if key not in valid:

            raise RuntimeError("FAST cache incomplete for " f"{key}")

        result_rows.append(valid[key])

    result = pd.DataFrame(result_rows)

    if len(result) != EXPECTED_FS2_GOLD_CELLS:

        raise RuntimeError(
            f"FAST result rows={len(result)}, "
            f"expected="
            f"{EXPECTED_FS2_GOLD_CELLS}"
        )

    if (
        result[
            [
                UID,
                "Label",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError("Duplicate FAST UID/Label rows")

    probabilities = pd.to_numeric(
        result["FastProbability"],
        errors="raise",
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):

        raise RuntimeError("Invalid FAST probabilities")

    return result


# =============================================================================
# TARGET VARIANTS
# =============================================================================


def build_variant(
    w23_oof: pd.DataFrame,
    fast_oof: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:

    base = (
        w23_oof[w23_oof["Label"].isin(FS2_LABELS)][
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

    fast = fast_oof[
        [
            UID,
            "Label",
            "FastProbability",
        ]
    ].copy()

    merged = base.merge(
        fast,
        on=[
            UID,
            "Label",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["FastProbability"].isna().any():

        raise RuntimeError("Missing FS2 FAST predictions")

    if variant == "w23":

        merged["Probability"] = merged["W23Probability"].astype(float)

    elif variant == "fast_replace":

        merged["Probability"] = merged["FastProbability"].astype(float)

    elif variant == "fast_fixed50":

        merged["Probability"] = (1.0 - FIXED_BLEND_ALPHA) * merged[
            "W23Probability"
        ].astype(float) + FIXED_BLEND_ALPHA * merged["FastProbability"].astype(float)

    else:

        raise ValueError(variant)

    merged["Variant"] = variant

    return merged[
        [
            UID,
            "Label",
            "Gold",
            "OuterFold",
            "Variant",
            "Probability",
        ]
    ]


def metrics_for_variant(
    variant_oof: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for label in FS2_LABELS:

        subset = variant_oof[variant_oof["Label"] == label]

        y = subset["Gold"].to_numpy(dtype=int)

        p = subset["Probability"].to_numpy(dtype=float)

        rows.append(
            {
                "Label": label,
                "N": len(subset),
                "Positive": int(y.sum()),
                "Negative": int(len(y) - y.sum()),
                "AUROC": safe_auc(
                    y,
                    p,
                ),
                "AP": safe_ap(
                    y,
                    p,
                ),
                "Brier": float(
                    brier_score_loss(
                        y,
                        np.clip(
                            p,
                            1e-5,
                            1 - 1e-5,
                        ),
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def target_matrix(
    oof: pd.DataFrame,
    gold: pd.DataFrame,
) -> np.ndarray:

    wide = oof.pivot(
        index=UID,
        columns="Label",
        values="Probability",
    ).reindex(
        index=gold[UID].astype(str),
        columns=FS2_LABELS,
    )

    if wide.isna().any().any():

        raise RuntimeError("Target probability matrix has NaNs")

    return wide.to_numpy(dtype=float)


def target_truth_matrix(
    gold: pd.DataFrame,
) -> np.ndarray:

    frame = gold.set_index(UID).reindex(gold[UID].astype(str))

    return frame[FS2_LABELS].to_numpy(dtype=int)


def mean_target_auc(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    scores = []

    for j in range(y.shape[1]):

        scores.append(
            safe_auc(
                y[
                    :,
                    j,
                ],
                probabilities[
                    :,
                    j,
                ],
            )
        )

    return float(np.nanmean(scores))


def bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, Any]:

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    n = len(y)

    deltas = []

    for _ in range(BOOTSTRAP_REPEATS):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sampled_y = y[indices]

        candidate_scores = []

        reference_scores = []

        for j in range(sampled_y.shape[1]):

            yy = sampled_y[
                :,
                j,
            ]

            if len(np.unique(yy)) < 2:

                continue

            candidate_scores.append(
                roc_auc_score(
                    yy,
                    candidate[
                        indices,
                        j,
                    ],
                )
            )

            reference_scores.append(
                roc_auc_score(
                    yy,
                    reference[
                        indices,
                        j,
                    ],
                )
            )

        if candidate_scores:

            deltas.append(float(np.mean(candidate_scores) - np.mean(reference_scores)))

    if not deltas:

        return {
            "n": 0,
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_delta_gt_0": np.nan,
        }

    array = np.asarray(
        deltas,
        dtype=float,
    )

    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(
            np.quantile(
                array,
                0.025,
            )
        ),
        "ci95_high": float(
            np.quantile(
                array,
                0.975,
            )
        ),
        "p_delta_gt_0": float(np.mean(array > 0)),
    }


# =============================================================================
# GOLD GATE
# =============================================================================


def evaluate_gate(
    gold: pd.DataFrame,
    variants: Sequence[pd.DataFrame],
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:

    metric_parts = {}

    matrices = {}

    for frame in variants:

        name = str(frame["Variant"].iloc[0])

        metrics = metrics_for_variant(frame)

        metrics.insert(
            0,
            "Variant",
            name,
        )

        metric_parts[name] = metrics

        matrices[name] = target_matrix(
            frame,
            gold,
        )

    metrics = pd.concat(
        metric_parts.values(),
        ignore_index=True,
    )

    y = target_truth_matrix(gold)

    summary = {
        "script_version": SCRIPT_VERSION,
        "primary_variant": "fast_fixed50",
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "logit_temperature": LOGIT_TEMPERATURE,
        "target_labels": FS2_LABELS,
        "variants": {},
    }

    for name, metric in metric_parts.items():

        summary["variants"][name] = {
            "mean_AUROC": float(np.nanmean(metric["AUROC"])),
            "mean_AP": float(np.nanmean(metric["AP"])),
            "mean_Brier": float(np.nanmean(metric["Brier"])),
        }

    reference_auc = summary["variants"]["w23"]["mean_AUROC"]

    candidate_auc = summary["variants"]["fast_fixed50"]["mean_AUROC"]

    mean_auc_delta = candidate_auc - reference_auc

    reference_brier = summary["variants"]["w23"]["mean_Brier"]

    candidate_brier = summary["variants"]["fast_fixed50"]["mean_Brier"]

    brier_delta = candidate_brier - reference_brier

    attribution_rows = []

    per_label_deltas = {}

    for label in FS2_LABELS:

        w23_auc = float(
            metric_parts["w23"]
            .set_index("Label")
            .at[
                label,
                "AUROC",
            ]
        )

        replace_auc = float(
            metric_parts["fast_replace"]
            .set_index("Label")
            .at[
                label,
                "AUROC",
            ]
        )

        fixed_auc = float(
            metric_parts["fast_fixed50"]
            .set_index("Label")
            .at[
                label,
                "AUROC",
            ]
        )

        delta = fixed_auc - w23_auc

        per_label_deltas[label] = delta

        attribution_rows.append(
            {
                "Label": label,
                "W23_AUROC": w23_auc,
                "FAST_Replace_AUROC": replace_auc,
                "FAST_Fixed50_AUROC": fixed_auc,
                "Replace_Delta": replace_auc - w23_auc,
                "Fixed50_Delta": delta,
            }
        )

    bootstrap = bootstrap_delta(
        y,
        matrices["fast_fixed50"],
        matrices["w23"],
    )

    conditions = {
        "mean_auc_delta_ge_0_040": bool(mean_auc_delta >= GATE_MIN_MEAN_AUC_DELTA),
        "no_label_regression_gt_0_005": bool(
            min(per_label_deltas.values()) >= GATE_MIN_PER_LABEL_DELTA
        ),
        "bootstrap_p_delta_gt_0_ge_0_95": bool(
            bootstrap["p_delta_gt_0"] >= GATE_MIN_P_DELTA_GT_ZERO
        ),
        "mean_brier_not_worse_gt_0_010": bool(brier_delta <= GATE_MAX_BRIER_DELTA),
    }

    passed = bool(all(conditions.values()))

    verdict = "PASS_FS2_GOLD_GATE" if passed else "STOP_FS2"

    summary.update(
        {
            "w23_target_mean_AUROC": reference_auc,
            "fixed50_target_mean_AUROC": candidate_auc,
            "primary_mean_AUROC_delta": mean_auc_delta,
            "w23_target_mean_Brier": reference_brier,
            "fixed50_target_mean_Brier": candidate_brier,
            "primary_mean_Brier_delta": brier_delta,
            "per_label_fixed50_AUROC_delta": per_label_deltas,
            "bootstrap": bootstrap,
            "gate_thresholds": {
                "min_mean_AUROC_delta": GATE_MIN_MEAN_AUC_DELTA,
                "min_per_label_AUROC_delta": GATE_MIN_PER_LABEL_DELTA,
                "min_bootstrap_p_delta_gt_0": GATE_MIN_P_DELTA_GT_ZERO,
                "max_mean_Brier_delta": GATE_MAX_BRIER_DELTA,
            },
            "gate_conditions": conditions,
            "gate_verdict": verdict,
            # Deliberately false. Even PASS requires review and a new
            # versioned production script.
            "production_generation_authorized": False,
            "image_training_authorized": False,
        }
    )

    return (
        metrics,
        {
            "summary": summary,
            "attribution": pd.DataFrame(attribution_rows),
        },
    )


# =============================================================================
# STATUS
# =============================================================================


def run_status(
    paths: ProjectPaths,
    accelerator: str,
    precision: str,
) -> Dict[str, Any]:

    ensure_dirs(paths)

    validate_source_roots(paths)

    _, gold, _ = load_train(paths)

    folds = load_folds(
        paths,
        gold,
    )

    w2_gold = load_w2_gold(
        paths,
        gold,
    )

    w23 = load_w23_oof(
        paths,
        gold,
    )

    exemplars, audit = build_all_exemplars(
        gold,
        folds,
        w2_gold,
    )

    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()

    leakage = 0

    for _, row in audit.iterrows():

        example_fold = int(fold_map[str(row["ExampleStudyInstanceUID"])])

        if example_fold == int(row["OuterFold"]):

            leakage += 1

    w23_target_auc = {}

    for label in FS2_LABELS:

        subset = w23[w23["Label"] == label]

        w23_target_auc[label] = safe_auc(
            subset["Gold"].to_numpy(dtype=int),
            subset["FoldSafeChallengeProbability"].to_numpy(dtype=float),
        )

    resolved = resolve_accelerator(accelerator)

    payload = {
        "script_version": SCRIPT_VERSION,
        "targets": FS2_LABELS,
        "gold_mapper_cells": EXPECTED_FS2_GOLD_CELLS,
        "fixed_blend_alpha": FIXED_BLEND_ALPHA,
        "logit_temperature": LOGIT_TEMPERATURE,
        "accelerator_requested": accelerator,
        "accelerator_resolved": resolved,
        "precision": precision,
        "accelerator_env_variables_required": False,
        "paths": {
            "project_root": str(paths.project_root),
            "train_csv": str(paths.train_csv),
            "w2_root": str(paths.w2_root),
            "w23_root": str(paths.w23_root),
            "model_path": str(paths.model_path),
            "model_exists": paths.model_path.exists(),
            "output_root": str(paths.output_root),
        },
        "fold_sha256": fold_sha256(folds),
        "fold_sha256_match": (fold_sha256(folds) == EXPECTED_FOLD_SHA256),
        "exemplar_query_label_sets": len(exemplars),
        "exemplar_rows": len(audit),
        "exemplar_outer_fold_leakage": leakage,
        "w23_target_AUROC": w23_target_auc,
        "cached_cells": len(read_cache(cache_path(paths))),
        "gate": {
            "min_mean_auc_delta": GATE_MIN_MEAN_AUC_DELTA,
            "min_per_label_delta": GATE_MIN_PER_LABEL_DELTA,
            "min_p_delta_gt_zero": GATE_MIN_P_DELTA_GT_ZERO,
            "max_brier_delta": GATE_MAX_BRIER_DELTA,
        },
        "scope": {
            "gold_only": True,
            "production_mode_exists": False,
            "image_training": False,
            "dicom": False,
            "curia": False,
            "pilkwang": False,
        },
    }

    write_json(
        paths.result_root / "status.json",
        payload,
    )

    log(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )

    return payload


# =============================================================================
# GOLD RUN
# =============================================================================


def run_gold(
    paths: ProjectPaths,
    accelerator: str,
    precision: str,
) -> Dict[str, Any]:

    ensure_dirs(paths)

    validate_source_roots(paths)

    train, gold, _ = load_train(paths)

    folds = load_folds(
        paths,
        gold,
    )

    w2_gold = load_w2_gold(
        paths,
        gold,
    )

    w23_oof = load_w23_oof(
        paths,
        gold,
    )

    if not paths.model_path.exists():

        raise FileNotFoundError(f"Qwen model not found: " f"{paths.model_path}")

    log("=" * 100)

    log("W39 FS2 GOLD-ONLY FAST CHALLENGE MAPPER v1")

    log("=" * 100)

    log(f"Targets                   : " f"{FS2_LABELS}")

    log(f"Gold studies              : " f"{len(gold)}")

    log(f"FAST cells                : " f"{EXPECTED_FS2_GOLD_CELLS}")

    log(f"Examples/query            : " f"+{N_POS_EXAMPLES} / " f"-{N_NEG_EXAMPLES}")

    log(f"Fixed blend               : " f"{FIXED_BLEND_ALPHA:.2f}")

    log(f"Logit temperature         : " f"{LOGIT_TEMPERATURE:.2f}")

    log(f"Fold SHA256               : " f"{fold_sha256(folds)}")

    log("Pilkwang used             : NO")

    log("Production generation     : DISABLED")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    exemplars, exemplar_audit = build_all_exemplars(
        gold,
        folds,
        w2_gold,
    )

    folds.to_csv(
        paths.result_root / "00_outer_fold_assignments.csv",
        index=False,
    )

    exemplar_audit.to_csv(
        paths.result_root / "01_exemplar_audit.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    queries = build_gold_queries(
        gold,
        folds,
        w2_gold,
        exemplars,
    )

    pd.DataFrame(
        [
            {
                UID: query[UID],
                "Label": query["Label"],
                "Gold": query["Gold"],
                "OuterFold": query["OuterFold"],
                "Prior": query["Prior"],
                "PromptSHA256": query["PromptSHA256"],
                "PromptChars": query["PromptChars"],
                "ExampleUIDs": query["ExampleUIDs"],
            }
            for query in queries
        ]
    ).to_csv(
        paths.result_root / "02_prompt_audit.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # FAST scoring
    # ------------------------------------------------------------------

    fast = run_fast_scoring(
        paths=paths,
        queries=queries,
        accelerator=accelerator,
        precision=precision,
    )

    fast.to_csv(
        paths.result_root / "03_fs2_fast_oof_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Controlled variants
    # ------------------------------------------------------------------

    variants = [
        build_variant(
            w23_oof,
            fast,
            "w23",
        ),
        build_variant(
            w23_oof,
            fast,
            "fast_replace",
        ),
        build_variant(
            w23_oof,
            fast,
            "fast_fixed50",
        ),
    ]

    all_variants = pd.concat(
        variants,
        ignore_index=True,
    )

    all_variants.to_csv(
        paths.result_root / "04_target_variants_oof_long.csv",
        index=False,
    )

    metrics, evaluation = evaluate_gate(
        gold,
        variants,
    )

    metrics.to_csv(
        paths.result_root / "05_target_metrics.csv",
        index=False,
    )

    attribution = evaluation["attribution"]

    attribution.to_csv(
        paths.result_root / "06_fs2_attribution.csv",
        index=False,
    )

    summary = evaluation["summary"]

    summary.update(
        {
            "fold_sha256": fold_sha256(folds),
            "fold_sha256_match": (fold_sha256(folds) == EXPECTED_FOLD_SHA256),
            "model_path": str(paths.model_path),
            "model_config_sha256": (
                sha256_file(paths.model_path / "config.json")
                if (paths.model_path / "config.json").exists()
                else None
            ),
            "retrieval": ("outer-train-only char_wb " "TF-IDF 3-5 grams + W2 features"),
            "n_pos_examples": N_POS_EXAMPLES,
            "n_neg_examples": N_NEG_EXAMPLES,
            "uses_pilkwang_labels": False,
            "production_generation_authorized": False,
            "image_training_authorized": False,
            "next_step": (
                "If PASS_FS2_GOLD_GATE: review results and "
                "build a NEW versioned FS2 production script. "
                "If STOP_FS2: close Contusion/Effusion rebuild branch."
            ),
        }
    )

    write_json(
        paths.result_root / "07_w39_gold_summary.json",
        summary,
    )

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    log("")

    log("=" * 100)

    log("W39 FS2 GOLD GATE RESULT")

    log("=" * 100)

    for _, row in attribution.iterrows():

        log(
            f"{row['Label']:<10} "
            f"W23={row['W23_AUROC']:.6f} "
            f"FAST={row['FAST_Replace_AUROC']:.6f} "
            f"FIXED50={row['FAST_Fixed50_AUROC']:.6f} "
            f"DELTA={row['Fixed50_Delta']:+.6f}"
        )

    log("")

    log(f"W23 FS2 mean AUROC      : " f"{summary['w23_target_mean_AUROC']:.6f}")

    log(f"Fixed50 FS2 mean AUROC  : " f"{summary['fixed50_target_mean_AUROC']:.6f}")

    log(f"Mean AUROC delta         : " f"{summary['primary_mean_AUROC_delta']:+.6f}")

    log(f"W23 mean Brier          : " f"{summary['w23_target_mean_Brier']:.6f}")

    log(f"Fixed50 mean Brier      : " f"{summary['fixed50_target_mean_Brier']:.6f}")

    log(f"Mean Brier delta         : " f"{summary['primary_mean_Brier_delta']:+.6f}")

    log(f"Bootstrap P(delta>0)     : " f"{summary['bootstrap']['p_delta_gt_0']:.6f}")

    log("")

    for name, passed in summary["gate_conditions"].items():

        log(f"{name:<42}: " f"{'PASS' if passed else 'FAIL'}")

    log("")

    log(f"GATE VERDICT             : " f"{summary['gate_verdict']}")

    log(f"Results                  : " f"{paths.result_root}")

    log("")

    if summary["gate_verdict"] == "PASS_FS2_GOLD_GATE":

        log(
            "FS2 PASSED. Do NOT run image training yet. "
            "Next step is a versioned production-teacher update."
        )

    else:

        log(
            "FS2 FAILED. Close this teacher-rebuild branch. "
            "Do not spend production or image-training time on it."
        )

    return summary


# =============================================================================
# VALIDATION
# =============================================================================


def run_validate(
    paths: ProjectPaths,
) -> Dict[str, Any]:

    required = {
        "folds": paths.result_root / "00_outer_fold_assignments.csv",
        "exemplars": paths.result_root / "01_exemplar_audit.csv",
        "prompts": paths.result_root / "02_prompt_audit.csv",
        "fast": paths.result_root / "03_fs2_fast_oof_long.csv",
        "variants": paths.result_root / "04_target_variants_oof_long.csv",
        "metrics": paths.result_root / "05_target_metrics.csv",
        "attribution": paths.result_root / "06_fs2_attribution.csv",
        "summary": paths.result_root / "07_w39_gold_summary.json",
    }

    exists = {f"{name}_exists": path.exists() for name, path in required.items()}

    if not all(exists.values()):

        payload = {
            "overall_pass": False,
            "checks": exists,
        }

        write_json(
            paths.result_root / "08_validation_summary.json",
            payload,
        )

        log(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return payload

    folds = pd.read_csv(required["folds"])

    exemplars = pd.read_csv(required["exemplars"])

    prompts = pd.read_csv(required["prompts"])

    fast = pd.read_csv(required["fast"])

    variants = pd.read_csv(required["variants"])

    metrics = pd.read_csv(required["metrics"])

    attribution = pd.read_csv(required["attribution"])

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))

    folds[UID] = folds[UID].astype(str)

    fold_map = folds.set_index(UID)["OuterFold"].astype(int).to_dict()

    leakage = 0

    for _, row in exemplars.iterrows():

        example_uid = str(row["ExampleStudyInstanceUID"])

        if int(fold_map[example_uid]) == int(row["OuterFold"]):

            leakage += 1

    expected_exemplar_rows = EXPECTED_FS2_GOLD_CELLS * (N_POS_EXAMPLES + N_NEG_EXAMPLES)

    checks = {
        **exists,
        "fold_hash_match": (fold_sha256(folds) == EXPECTED_FOLD_SHA256),
        "fast_rows_116": (len(fast) == EXPECTED_FS2_GOLD_CELLS),
        "fast_58_uids": (fast[UID].astype(str).nunique() == EXPECTED_GOLD),
        "fast_exact_two_labels": (set(fast["Label"]) == set(FS2_LABELS)),
        "fast_no_duplicates": (
            not fast[
                [
                    UID,
                    "Label",
                ]
            ]
            .duplicated()
            .any()
        ),
        "fast_probabilities_valid": bool(
            np.isfinite(fast["FastProbability"].to_numpy(dtype=float)).all()
            and fast["FastProbability"]
            .between(
                0,
                1,
            )
            .all()
        ),
        "prompt_rows_116": (len(prompts) == EXPECTED_FS2_GOLD_CELLS),
        "exemplar_rows_expected": (len(exemplars) == expected_exemplar_rows),
        "outer_fold_leakage_zero": (leakage == 0),
        "variant_rows": (len(variants) == EXPECTED_FS2_GOLD_CELLS * 3),
        "metric_rows": (len(metrics) == len(FS2_LABELS) * 3),
        "attribution_rows": (len(attribution) == len(FS2_LABELS)),
        "summary_verdict_valid": (
            summary.get("gate_verdict")
            in {
                "PASS_FS2_GOLD_GATE",
                "STOP_FS2",
            }
        ),
        "production_not_authorized": (
            summary.get("production_generation_authorized") is False
        ),
        "image_training_not_authorized": (
            summary.get("image_training_authorized") is False
        ),
        "pilkwang_not_used": True,
    }

    payload = {
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "gate_verdict": summary.get("gate_verdict"),
        "summary": summary,
        "results_root": str(paths.result_root),
    }

    write_json(
        paths.result_root / "08_validation_summary.json",
        payload,
    )

    log(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )

    if not payload["overall_pass"]:

        raise RuntimeError("W39 validation failed")

    return payload


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=("RSNA W39 FS2 gold-only " "FAST challenge mapper")
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "status",
            "gold",
            "validate",
        ],
        default=None,
    )

    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=[
            "status",
            "gold",
            "validate",
        ],
        default=None,
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
            "4bit",
            "nf4",
            "fp16",
        ],
    )

    # Optional path overrides. None are required.
    parser.add_argument(
        "--project-root",
        default=None,
    )

    parser.add_argument(
        "--train-csv",
        default=None,
    )

    parser.add_argument(
        "--w2-root",
        default=None,
    )

    parser.add_argument(
        "--w23-root",
        default=None,
    )

    parser.add_argument(
        "--model-path",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        default=None,
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    mode = args.mode_flag or args.mode or "status"

    paths = ProjectPaths.discover(args)

    if mode == "status":

        run_status(
            paths,
            accelerator=args.accelerator,
            precision=args.precision,
        )

    elif mode == "gold":

        run_gold(
            paths,
            accelerator=args.accelerator,
            precision=args.precision,
        )

    elif mode == "validate":

        run_validate(paths)

    else:

        raise ValueError(mode)


if __name__ == "__main__":
    main()
