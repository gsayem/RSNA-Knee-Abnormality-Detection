#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W40 — FS2 Production Teacher v1
================================

Validated predecessor
---------------------
W39 FS2 gold-only gate:

    Targets:
        Contusion
        Effusion

    W23 target mean AUROC     = 0.698473
    FAST50 target mean AUROC  = 0.748929
    Delta                     = +0.050456
    Bootstrap P(delta > 0)    = 0.968

    Verdict:
        PASS_FS2_GOLD_GATE

Purpose
-------
Generate FINAL production FAST probabilities for:

    Contusion
    Effusion

using all 58 labeled gold reports as production exemplars.

Then apply the LOCKED formula:

    W40 probability
        =
    0.50 * existing W2.6-P teacher probability
        +
    0.50 * W40 FAST probability

Important control
-----------------
ONLY Contusion and Effusion probabilities change.

The following remain exactly unchanged from W2.6-P FAST:

    - all other 10 diagnosis probabilities
    - RecommendedTeacherWeight for all 12 labels
    - RecommendedTeacherMask for all 12 labels

Synovitis is intentionally NOT modified here.

Why?
----
W38 identified Synovitis as a REMASK issue, but no production remask
formula has been independently validated. Mixing that policy change into
W40 would make the experiment non-identifiable.

This script therefore performs one controlled teacher change only.

Production cells
----------------
    4,349 unlabeled studies
    x 2 labels
    = 8,698 FAST scores

No image training.
No Curia.
No DICOM.
No Pilkwang.
No project .py imports.
No external accelerator environment variables required.

CLI
---
    python w40_rsna_fs2_production_teacher_v1.py status \
        --accelerator localGPU

    python w40_rsna_fs2_production_teacher_v1.py production \
        --accelerator localGPU

    python w40_rsna_fs2_production_teacher_v1.py validate

Outputs
-------
output/results/rsna_w40_fs2_production_teacher_v1/

    cache/
        w40_fs2_production_fast_v1.jsonl

    results/
        00_status.json
        10_base_teacher_snapshot.csv
        11_production_exemplar_audit.csv
        12_production_prompt_metadata.csv
        13_fast_fs2_production_long.csv
        14_fast_fs2_probabilities_wide.csv
        15_final_teacher_long.csv
        16_final_probabilities_wide.csv
        17_teacher_weights_wide.csv
        18_teacher_mask_wide.csv
        19_production_summary.json
        20_validation_summary.json
"""

from __future__ import annotations

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

from dataclasses import asdict, dataclass
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

# =============================================================================
# VERSION
# =============================================================================

SCRIPT_VERSION = "w40_fs2_production_teacher_v1"

DISPLAY_VERSION = "W40 FS2 PRODUCTION TEACHER v1"


# =============================================================================
# DATA CONSTANTS
# =============================================================================

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

EXPECTED_ALL_PRODUCTION_CELLS = EXPECTED_UNLABELED * len(LABELS)

EXPECTED_FS2_PRODUCTION_CELLS = EXPECTED_UNLABELED * len(FS2_LABELS)

EXPECTED_W2_FULL_ROWS = EXPECTED_TRAIN * len(LABELS)

EXPECTED_W2_GOLD_ROWS = EXPECTED_GOLD * len(LABELS)

EXPECTED_BASE_SELECTED_CELLS = 32027

EXPECTED_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

EXPECTED_W39_SCRIPT_VERSION = "w39_fs2_gold_gate_v1"

EXPECTED_W39_GATE = "PASS_FS2_GOLD_GATE"

EXPECTED_W39_MIN_DELTA = 0.040

EXPECTED_W39_MODEL_CONFIG_SHA256 = (
    "72032dd2ac37578561aeae9033a843572" "7b3846afd2509847325beb38e7a4de7"
)


# =============================================================================
# LOCKED PRODUCTION SETTINGS
# =============================================================================

FIXED_BLEND_ALPHA = 0.50

LOGIT_TEMPERATURE = 2.0

N_POS_EXAMPLES = 2
N_NEG_EXAMPLES = 2

TFIDF_MAX_FEATURES = 30000

COMPACT_EXAMPLE_CHARS = 650
COMPACT_QUERY_CHARS = 1100

MAX_INPUT_TOKENS = 2304

SINGLE_PROMPT_FALLBACK_TOKENS = 1536

MAX_BATCH_SIZE = 24


# =============================================================================
# W2 REQUIRED FEATURES
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
# TARGET DEFINITIONS — LOCKED TO W39
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
# TARGET SNIPPET PATTERNS — LOCKED TO W39
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
# BASIC HELPERS
# =============================================================================


def log(
    message: str = "",
) -> None:
    print(
        message,
        flush=True,
    )


def normalize_text(
    value: Any,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(value if value is not None else ""),
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
        (
            float,
            np.floating,
        ),
    ):
        return f"{float(value):.{digits}f}"

    return str(value)


def stable_sha256(
    text: str,
) -> str:

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(
    path: Path,
    chunk_size: int = (16 * 1024 * 1024),
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:

    def convert(
        value: Any,
    ):

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


def compact_report(
    text: str,
    max_chars: int,
) -> str:

    text = normalize_text(text)

    if len(text) <= max_chars:
        return text

    head = max_chars // 2

    tail = max_chars - head

    return text[:head] + " ... " + text[-tail:]


def target_focused_snippet(
    report: str,
    label: str,
    max_chars: int,
) -> str:

    full = normalize_text(report)

    if not full:
        return ""

    sentences = [
        sentence.strip()
        for sentence in _SENT_SPLIT_RE.split(str(report))
        if (sentence and sentence.strip())
    ]

    patterns = TARGET_PATTERNS[label]

    scored: List[
        Tuple[
            int,
            int,
            str,
        ]
    ] = []

    for index, sentence in enumerate(sentences):

        text = normalize_text(sentence)

        hits = sum(
            1
            for pattern in patterns
            if re.search(
                pattern,
                text,
                flags=re.I,
            )
        )

        if hits:

            scored.append(
                (
                    hits,
                    index,
                    text,
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

    tail = full[-500:]

    if tail and tail not in selected:
        selected.append(tail)

    if not selected:

        return compact_report(
            full,
            max_chars,
        )

    return compact_report(
        " | ".join(selected),
        max_chars,
    )


def parse_bool_series(
    series: pd.Series,
) -> np.ndarray:

    if pd.api.types.is_bool_dtype(series):

        return series.to_numpy(dtype=bool)

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():

        return numeric.to_numpy(dtype=float) > 0.5

    text = series.astype(str).str.strip().str.lower()

    return text.isin(
        [
            "true",
            "1",
            "yes",
            "y",
            "t",
        ]
    ).to_numpy(dtype=bool)


def first_existing(
    candidates: Iterable[Path],
) -> Optional[Path]:

    for candidate in candidates:

        if candidate.exists():

            return candidate.resolve()

    return None


# =============================================================================
# PATHS
# =============================================================================


def script_directory() -> Path:

    try:

        return Path(__file__).resolve().parent

    except NameError:

        return Path.cwd().resolve()


@dataclass
class ProjectPaths:

    project_root: Path

    train_csv: Path

    w2_root: Path

    w26p_root: Path

    w39_root: Path

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

        # -------------------------------------------------------------
        # Project
        # -------------------------------------------------------------

        if args.project_root:

            project_root = Path(args.project_root).expanduser().resolve()

        elif is_kaggle:

            project_root = Path("/kaggle/working")

        else:

            project_root = (script_directory() / ".." / "..").resolve()

        # -------------------------------------------------------------
        # train.csv
        # -------------------------------------------------------------

        if args.train_csv:

            train_csv = Path(args.train_csv).expanduser().resolve()

        elif is_kaggle:

            train_csv = Path(
                "/kaggle/input/"
                "competitions/"
                "rsna-knee-abnormality-detection/"
                "train.csv"
            )

        else:

            train_csv = project_root / "input" / "train.csv"

        # -------------------------------------------------------------
        # W2
        # -------------------------------------------------------------

        if args.w2_root:

            w2_root = Path(args.w2_root).expanduser().resolve()

        elif is_kaggle:

            w2_root = Path("/kaggle/input/" "datasets/" "isayem/" "rsna-w2/" "rsna_w2")

        else:

            w2_root = project_root / "output" / "results" / "rsna_w2"

        # -------------------------------------------------------------
        # W2.6-P FAST
        # -------------------------------------------------------------

        if args.w26p_root:

            w26p_root = Path(args.w26p_root).expanduser().resolve()

        else:

            w26p_root = project_root / "output" / "results" / "rsna_w2_6p_fast"

        # -------------------------------------------------------------
        # W39
        # -------------------------------------------------------------

        if args.w39_root:

            w39_root = Path(args.w39_root).expanduser().resolve()

        else:

            w39_root = project_root / "output" / "results" / "rsna_w39_fs2_gold_gate_v1"

        # -------------------------------------------------------------
        # Qwen
        # -------------------------------------------------------------

        if args.model_path:

            model_path = Path(args.model_path).expanduser().resolve()

        elif is_kaggle:

            model_path = Path(
                "/kaggle/input/" "datasets/" "ragnar123/" "qwen2-5-7b-instruct"
            )

        else:

            candidates = [
                project_root / "models" / "Qwen2.5-7B-Instruct",
                project_root / "models" / "qwen2-5-7b-instruct",
                project_root / "input" / "qwen2-5-7b-instruct",
            ]

            model_path = first_existing(candidates) or candidates[0]

        # -------------------------------------------------------------
        # Output
        # -------------------------------------------------------------

        if args.output_root:

            output_root = Path(args.output_root).expanduser().resolve()

        elif is_kaggle:

            output_root = Path("/kaggle/working/" "rsna_w40_fs2_production_teacher_v1")

        else:

            output_root = (
                project_root
                / "output"
                / "results"
                / "rsna_w40_fs2_production_teacher_v1"
            )

        cache_root = output_root / "cache"

        result_root = output_root / "results"

        return cls(
            project_root=project_root,
            train_csv=train_csv,
            w2_root=w2_root,
            w26p_root=w26p_root,
            w39_root=w39_root,
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
# INPUT PATHS
# =============================================================================


def w39_summary_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w39_root / "results" / "07_w39_gold_summary.json"


def base_teacher_long_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w26p_root / "results" / "15_final_hybrid_teacher_long.csv"


def base_probabilities_wide_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w26p_root / "results" / "16_final_hybrid_probabilities_wide.csv"


def base_weights_wide_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w26p_root / "results" / "17_recommended_teacher_weights_wide.csv"


def base_mask_wide_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w26p_root / "results" / "18_recommended_teacher_mask_wide.csv"


def w2_gold_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w2_root / "results" / "04_gold_structured_report_features.csv"


def w2_full_path(
    paths: ProjectPaths,
) -> Path:

    return paths.w2_root / "results" / "08_full_structured_report_labels.csv"


def production_cache_path(
    paths: ProjectPaths,
) -> Path:

    return paths.cache_root / "w40_fs2_production_fast_v1.jsonl"


# =============================================================================
# INPUT GUARDS
# =============================================================================


def validate_required_files(
    paths: ProjectPaths,
) -> None:

    required = [
        paths.train_csv,
        w39_summary_path(paths),
        base_teacher_long_path(paths),
        base_probabilities_wide_path(paths),
        base_weights_wide_path(paths),
        base_mask_wide_path(paths),
        w2_gold_path(paths),
        w2_full_path(paths),
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:

        raise FileNotFoundError("Missing required W40 inputs:\n" + "\n".join(missing))


def validate_w39_gate(
    paths: ProjectPaths,
) -> Dict[str, Any]:

    path = w39_summary_path(paths)

    if not path.exists():

        raise FileNotFoundError(f"W39 summary missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("script_version") != EXPECTED_W39_SCRIPT_VERSION:

        raise RuntimeError(
            "Unexpected W39 script version: " f"{payload.get('script_version')}"
        )

    if payload.get("gate_verdict") != EXPECTED_W39_GATE:

        raise RuntimeError(
            "W39 did not pass FS2 gate: " f"{payload.get('gate_verdict')}"
        )

    if not payload.get(
        "fold_sha256_match",
        False,
    ):

        raise RuntimeError("W39 fold checksum did not match.")

    if payload.get("fold_sha256") != EXPECTED_FOLD_SHA256:

        raise RuntimeError("Unexpected W39 fold SHA256.")

    labels = payload.get("target_labels")

    if set(labels or []) != set(FS2_LABELS):

        raise RuntimeError(f"W39 target mismatch: {labels}")

    if abs(float(payload.get("fixed_blend_alpha")) - FIXED_BLEND_ALPHA) > 1e-12:

        raise RuntimeError("W39 blend alpha mismatch.")

    if abs(float(payload.get("logit_temperature")) - LOGIT_TEMPERATURE) > 1e-12:

        raise RuntimeError("W39 logit temperature mismatch.")

    delta = float(
        payload.get(
            "primary_mean_AUROC_delta",
            -999,
        )
    )

    if delta < EXPECTED_W39_MIN_DELTA:

        raise RuntimeError(f"W39 delta no longer satisfies gate: " f"{delta:+.6f}")

    model_config_sha = payload.get("model_config_sha256")

    if model_config_sha and model_config_sha != EXPECTED_W39_MODEL_CONFIG_SHA256:

        raise RuntimeError("Unexpected W39 Qwen config SHA256: " f"{model_config_sha}")

    return payload


# =============================================================================
# DATA LOADERS
# =============================================================================


def load_train(
    paths: ProjectPaths,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

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

        raise RuntimeError("Partially labeled studies found: " f"{int(partial.sum())}")

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
            f"Unexpected train split: " f"{observed}, expected={expected}"
        )

    return (
        train,
        gold,
        unlabeled,
    )


def load_w2_tables(
    paths: ProjectPaths,
    train: pd.DataFrame,
    gold: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    gold_w2 = pd.read_csv(w2_gold_path(paths))

    full_w2 = pd.read_csv(w2_full_path(paths))

    for frame in [
        gold_w2,
        full_w2,
    ]:

        frame[UID] = frame[UID].astype(str)

    required = {
        UID,
        "Label",
        *W2_REQUIRED_COLUMNS,
    }

    for (
        name,
        frame,
    ) in [
        (
            "gold",
            gold_w2,
        ),
        (
            "full",
            full_w2,
        ),
    ]:

        missing = required - set(frame.columns)

        if missing:

            raise RuntimeError(f"W2 {name} missing columns: " f"{sorted(missing)}")

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

            raise RuntimeError(f"W2 {name} has duplicate UID/Label rows.")

    if len(gold_w2) != EXPECTED_W2_GOLD_ROWS:

        raise RuntimeError("Unexpected W2 gold row count.")

    if len(full_w2) != EXPECTED_W2_FULL_ROWS:

        raise RuntimeError("Unexpected W2 full row count.")

    if set(gold_w2[UID]) != set(gold[UID]):

        raise RuntimeError("W2 gold UID mismatch.")

    if set(full_w2[UID]) != set(train[UID]):

        raise RuntimeError("W2 full UID mismatch.")

    return (
        gold_w2,
        full_w2,
    )


def load_base_teacher(
    paths: ProjectPaths,
    unlabeled: pd.DataFrame,
) -> pd.DataFrame:

    path = base_teacher_long_path(paths)

    frame = pd.read_csv(path)

    frame[UID] = frame[UID].astype(str)

    required = {
        UID,
        "Label",
        "TeacherProbability",
        "RecommendedTeacherWeight",
        "RecommendedTeacherMask",
    }

    missing = required - set(frame.columns)

    if missing:

        raise RuntimeError("W2.6-P base teacher missing: " f"{sorted(missing)}")

    if len(frame) != EXPECTED_ALL_PRODUCTION_CELLS:

        raise RuntimeError(
            f"Base teacher rows={len(frame)}, "
            f"expected="
            f"{EXPECTED_ALL_PRODUCTION_CELLS}"
        )

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

        raise RuntimeError("Base teacher contains duplicates.")

    if set(frame["Label"]) != set(LABELS):

        raise RuntimeError("Base teacher label set mismatch.")

    if set(frame[UID]) != set(unlabeled[UID].astype(str)):

        raise RuntimeError("Base teacher UID mismatch.")

    p = pd.to_numeric(
        frame["TeacherProbability"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():

        raise RuntimeError("Invalid base teacher probabilities.")

    weights = pd.to_numeric(
        frame["RecommendedTeacherWeight"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():

        raise RuntimeError("Invalid base teacher weights.")

    masks = parse_bool_series(frame["RecommendedTeacherMask"])

    selected = int(masks.sum())

    if selected != EXPECTED_BASE_SELECTED_CELLS:

        raise RuntimeError(
            f"Base teacher selected cells="
            f"{selected}, expected="
            f"{EXPECTED_BASE_SELECTED_CELLS}"
        )

    # -------------------------------------------------------------
    # Critical identity guard:
    #
    # Existing W2.6-P only changed FS4, not Contusion/Effusion.
    # Therefore TeacherProbability must still equal W23 raw
    # probability for FS2 when that column exists.
    # -------------------------------------------------------------

    if "W23RawProbabilityMean" in frame.columns:

        target = frame[frame["Label"].isin(FS2_LABELS)]

        if not np.allclose(
            target["TeacherProbability"].to_numpy(dtype=float),
            target["W23RawProbabilityMean"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=1e-12,
        ):

            raise RuntimeError(
                "FS2 base teacher is no longer identical "
                "to W2.3 raw probability. Refusing to "
                "apply W39 blend to an unexpected base."
            )

    return frame.sort_values(
        [
            UID,
            "Label",
        ]
    ).reset_index(drop=True)


# =============================================================================
# W2 SUMMARIES
# =============================================================================


def build_feature_lookup(
    frame: pd.DataFrame,
) -> Dict[
    Tuple[
        str,
        str,
    ],
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

        compact = " || ".join(evidence[:2])

        parts.append(
            "ev="
            + compact_report(
                compact,
                240,
            )
        )

    return "; ".join(parts)


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


# =============================================================================
# PRODUCTION RETRIEVAL
# =============================================================================


def select_top_by_class(
    reference: pd.DataFrame,
    similarities: np.ndarray,
    label: str,
) -> List[int]:

    table = reference[
        [
            UID,
            label,
        ]
    ].copy()

    table["Similarity"] = similarities

    table["_idx"] = np.arange(len(table))

    table = table.sort_values(
        [
            "Similarity",
            UID,
        ],
        ascending=[
            False,
            True,
        ],
    )

    positives = table[table[label] == 1].head(N_POS_EXAMPLES)

    negatives = table[table[label] == 0].head(N_NEG_EXAMPLES)

    if len(positives) != N_POS_EXAMPLES or len(negatives) != N_NEG_EXAMPLES:

        raise RuntimeError(f"Insufficient production exemplars " f"for {label}")

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


SYSTEM_PROMPT = (
    "You map knee MRI reports to THIS competition's binary labels. "
    "Infer the challenge annotation convention from labeled examples. "
    "Do not equate silence with negative. "
    "You will choose A or B only."
)


def build_fast_prompt(
    label: str,
    query_report: str,
    query_w2: str,
    examples: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    prior: float,
) -> str:

    chunks = [
        f"TARGET={label}",
        ("DEFINITION=" + TARGET_DEFINITIONS[label]),
        ("REFERENCE_POSITIVE_RATE=" f"{prior:.3f}"),
        "A = challenge label 0",
        "B = challenge label 1",
        "",
        "LABELED PRODUCTION EXAMPLES:",
    ]

    for index, example in enumerate(
        examples,
        start=1,
    ):

        code = "B" if int(example["Gold"]) == 1 else "A"

        chunks += [
            (
                f"E{index}: "
                f"answer={code}; "
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


def build_production_queries(
    gold: pd.DataFrame,
    unlabeled: pd.DataFrame,
    gold_w2: pd.DataFrame,
    full_w2: pd.DataFrame,
) -> Tuple[
    List[
        Dict[
            str,
            Any,
        ]
    ],
    pd.DataFrame,
]:

    gold_lookup = build_feature_lookup(gold_w2)

    full_lookup = build_feature_lookup(full_w2)

    queries: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    audits: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for label in FS2_LABELS:

        log(f"Building production retrieval: " f"{label}")

        # -------------------------------------------------------------
        # All 58 gold reports are eligible production exemplars.
        # -------------------------------------------------------------

        gold_docs = []

        for _, row in gold.iterrows():

            uid = str(row[UID])

            feature_summary = w2_retrieval_summary(
                gold_lookup[
                    (
                        uid,
                        label,
                    )
                ]
            )

            gold_docs.append(
                retrieval_document(
                    str(row[REPORT]),
                    feature_summary,
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

        x_gold = vectorizer.fit_transform(gold_docs)

        query_docs = []

        query_w2 = []

        for _, row in unlabeled.iterrows():

            uid = str(row[UID])

            feature_summary = w2_retrieval_summary(
                full_lookup[
                    (
                        uid,
                        label,
                    )
                ]
            )

            query_docs.append(
                retrieval_document(
                    str(row[REPORT]),
                    feature_summary,
                    label,
                )
            )

            query_w2.append(
                w2_prompt_summary(
                    full_lookup[
                        (
                            uid,
                            label,
                        )
                    ]
                )
            )

        # -------------------------------------------------------------
        # 4349 x 58 similarity matrix: small and deterministic.
        # -------------------------------------------------------------

        x_query = vectorizer.transform(query_docs)

        similarities = (x_query @ x_gold.T).toarray()

        y_gold = gold[label].astype(int).to_numpy()

        prior = float(y_gold.mean())

        for query_index, (
            _,
            query_row,
        ) in enumerate(unlabeled.iterrows()):

            query_uid = str(query_row[UID])

            chosen = select_top_by_class(
                gold,
                similarities[query_index],
                label,
            )

            examples = []

            for rank, reference_index in enumerate(
                chosen,
                start=1,
            ):

                example = gold.iloc[reference_index]

                example_uid = str(example[UID])

                examples.append(
                    {
                        UID: example_uid,
                        "Gold": int(example[label]),
                        "Similarity": float(
                            similarities[
                                query_index,
                                reference_index,
                            ]
                        ),
                        "Report": str(example[REPORT]),
                        "W2Summary": w2_prompt_summary(
                            gold_lookup[
                                (
                                    example_uid,
                                    label,
                                )
                            ]
                        ),
                    }
                )

                audits.append(
                    {
                        "QueryStudyInstanceUID": query_uid,
                        "Label": label,
                        "ExampleRank": rank,
                        "ExampleStudyInstanceUID": example_uid,
                        "ExampleGold": int(example[label]),
                        "Similarity": float(
                            similarities[
                                query_index,
                                reference_index,
                            ]
                        ),
                    }
                )

            prompt = build_fast_prompt(
                label=label,
                query_report=str(query_row[REPORT]),
                query_w2=query_w2[query_index],
                examples=examples,
                prior=prior,
            )

            queries.append(
                {
                    UID: query_uid,
                    "Label": label,
                    "Prompt": prompt,
                    "PromptSHA256": stable_sha256(prompt),
                    "PromptChars": len(prompt),
                    "Prior": prior,
                    "ExampleUIDs": "|".join(str(example[UID]) for example in examples),
                }
            )

    if len(queries) != EXPECTED_FS2_PRODUCTION_CELLS:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_FS2_PRODUCTION_CELLS} "
            f"production queries; "
            f"got {len(queries)}"
        )

    expected_audit = EXPECTED_FS2_PRODUCTION_CELLS * (N_POS_EXAMPLES + N_NEG_EXAMPLES)

    if len(audits) != expected_audit:

        raise RuntimeError(
            f"Expected {expected_audit} " f"exemplar audit rows; " f"got {len(audits)}"
        )

    return (
        queries,
        pd.DataFrame(audits),
    )


# =============================================================================
# ACCELERATOR
# =============================================================================


def normalize_accelerator(
    value: str,
) -> str:

    text = (
        str(value or "auto")
        .strip()
        .lower()
        .replace(
            "-",
            "_",
        )
    )

    aliases = {
        "localgpu": "local_gpu",
        "local": "local_gpu",
        "gpu": "local_gpu",
        "cuda": "local_gpu",
        "t4": "kaggle_t4",
        "kagglegpu": "kaggle_t4",
        "apple": "apple_mps",
        "mps": "apple_mps",
        "mac": "apple_mps",
        "kaggle_tpu": "tpu",
        "v5e": "tpu",
    }

    return aliases.get(
        text,
        text,
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


def resolve_accelerator(
    requested: str,
) -> str:

    requested = normalize_accelerator(requested)

    allowed = {
        "auto",
        "local_gpu",
        "kaggle_t4",
        "apple_mps",
        "cpu",
        "tpu",
    }

    if requested not in allowed:

        raise ValueError(f"Unknown accelerator: " f"{requested}")

    if requested != "auto":

        return requested

    try:

        import torch

        if torch.cuda.is_available():

            names = [
                torch.cuda.get_device_name(i).lower()
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
# FAST LOGIT MODEL
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
                "W40 requires torch, transformers, "
                "accelerate and optionally bitsandbytes."
            ) from exc

        self.torch = torch

        self.model_path = Path(model_path)

        self.output_root = Path(output_root)

        self.accelerator = resolve_accelerator(accelerator)

        self.precision = str(precision or "auto").strip().lower()

        if not self.model_path.exists():

            raise FileNotFoundError(f"Qwen model missing: " f"{self.model_path}")

        if self.accelerator == "tpu":

            raise RuntimeError(
                "TPU is not implemented or validated "
                "for the Qwen decoder scoring path."
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

        (
            self.answer_zero,
            self.answer_one,
        ) = self.choose_verbalizers()

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

        # -------------------------------------------------------------
        # CUDA
        # -------------------------------------------------------------

        if self.accelerator == "local_gpu":

            if not (torch.cuda.is_available()):

                raise RuntimeError("localGPU requested " "but CUDA unavailable.")

            gpu_count = torch.cuda.device_count()

            log(f"Visible CUDA GPUs         : " f"{gpu_count}")

            loaded = False

            if self.precision in {
                "auto",
                "4bit",
                "nf4",
            }:

                try:

                    compute_dtype = (
                        torch.bfloat16
                        if (torch.cuda.is_bf16_supported())
                        else torch.float16
                    )

                    max_memory = {}

                    for gpu_id in range(gpu_count):

                        try:

                            _free, total = torch.cuda.mem_get_info(gpu_id)

                            total_gib = total / (1024**3)

                        except Exception:

                            total_gib = 16.0

                        max_memory[gpu_id] = f"{max(4.0, total_gib - 2.5):.1f}GiB"

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

                    self.load_mode = f"4bit_nf4_auto_" f"{gpu_count}gpu"

                    loaded = True

                except Exception as exc:

                    if self.precision in {
                        "4bit",
                        "nf4",
                    }:

                        raise

                    warnings.warn(
                        "4-bit loading failed; "
                        "falling back to FP16 offload. "
                        f"{exc}"
                    )

            if not loaded:

                max_memory = {}

                for gpu_id in range(gpu_count):

                    try:

                        _free, total = torch.cuda.mem_get_info(gpu_id)

                        total_gib = total / (1024**3)

                    except Exception:

                        total_gib = 16.0

                    max_memory[gpu_id] = (
                        f"{max(3.5, min(11.5, total_gib - 3.5)):.1f}GiB"
                    )

                max_memory["cpu"] = "36GiB"

                offload = self.output_root / "model_offload"

                offload.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                kwargs = dict(common)

                kwargs["dtype"] = torch.float16

                kwargs["device_map"] = "auto"

                kwargs["max_memory"] = max_memory

                kwargs["offload_folder"] = str(offload)

                kwargs["offload_state_dict"] = True

                kwargs["attn_implementation"] = "sdpa"

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **kwargs,
                )

                self.load_mode = f"fp16_auto_" f"{gpu_count}gpu_cpu_offload"

        # -------------------------------------------------------------
        # Kaggle T4
        # -------------------------------------------------------------

        elif self.accelerator == "kaggle_t4":

            if not (torch.cuda.is_available()):

                raise RuntimeError("kaggle_t4 requested " "but CUDA unavailable.")

            gpu_count = torch.cuda.device_count()

            kwargs = dict(common)

            kwargs["dtype"] = torch.float16

            kwargs["device_map"] = "balanced" if gpu_count > 1 else "auto"

            kwargs["max_memory"] = {gpu_id: "11.5GiB" for gpu_id in range(gpu_count)}

            kwargs["attn_implementation"] = "sdpa"

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

            self.load_mode = f"fp16_balanced_" f"{gpu_count}gpu"

        # -------------------------------------------------------------
        # Apple MPS
        # -------------------------------------------------------------

        elif self.accelerator == "apple_mps":

            if not mps_available():

                raise RuntimeError("MPS unavailable.")

            kwargs = dict(common)

            kwargs["dtype"] = torch.float16

            kwargs["attn_implementation"] = "sdpa"

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

            self.model.to("mps")

            self.load_mode = "fp16_apple_mps"

        # -------------------------------------------------------------
        # CPU
        # -------------------------------------------------------------

        elif self.accelerator == "cpu":

            kwargs = dict(common)

            kwargs["dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

            self.load_mode = "fp32_cpu"

        else:

            raise RuntimeError(f"Unhandled accelerator " f"{self.accelerator}")

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
        Tuple[
            str,
            int,
        ],
        Tuple[
            str,
            int,
        ],
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

        for (
            zero,
            one,
        ) in candidates:

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

        raise RuntimeError("Could not find single-token " "binary verbalizers.")

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

        encoded = {
            key: value.to(self.input_device)
            for (
                key,
                value,
            ) in encoded.items()
        }

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

                raise RuntimeError(
                    f"Unexpected logits shape: " f"{tuple(logits.shape)}"
                )

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
        token_limit: int = (MAX_INPUT_TOKENS),
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
                    f"{len(prompts)} -> "
                    f"{middle}+"
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
# BATCHING
# =============================================================================


def make_length_aware_batches(
    queries: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    lengths: Sequence[int],
    token_budget: int,
) -> List[List[int]]:

    order = sorted(
        range(len(queries)),
        key=lambda index: (
            int(lengths[index]),
            str(
                queries[index].get(
                    "Label",
                    "",
                )
            ),
        ),
    )

    batches: List[List[int]] = []

    current: List[int] = []

    current_max = 0

    for index in order:

        length = int(lengths[index])

        proposed_max = max(
            current_max,
            length,
        )

        proposed_count = len(current) + 1

        padded_tokens = proposed_max * proposed_count

        if current and (
            proposed_count > MAX_BATCH_SIZE or padded_tokens > token_budget
        ):

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
# CACHE
# =============================================================================


def read_cache(
    path: Path,
) -> Dict[
    str,
    Dict[
        str,
        Any,
    ],
]:

    result: Dict[
        str,
        Dict[
            str,
            Any,
        ],
    ] = {}

    if not path.exists():

        return result

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
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

                raise RuntimeError(
                    f"Corrupt W40 cache line " f"{line_number}: {exc}"
                ) from exc

    return result


def append_cache(
    path: Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
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

    config = paths.model_path / "config.json"

    config_sha = sha256_file(config) if config.exists() else "missing"

    payload = {
        "script_version": SCRIPT_VERSION,
        "model_config_sha256": config_sha,
        "accelerator": resolve_accelerator(accelerator),
        "precision": precision,
        "logit_temperature": LOGIT_TEMPERATURE,
        "blend_alpha": FIXED_BLEND_ALPHA,
        "targets": FS2_LABELS,
        "target_definitions": TARGET_DEFINITIONS,
        "max_input_tokens": MAX_INPUT_TOKENS,
    }

    return stable_sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
        )
    )


# =============================================================================
# FAST PRODUCTION SCORING
# =============================================================================


def run_fast_scoring(
    paths: ProjectPaths,
    queries: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    accelerator: str,
    precision: str,
) -> pd.DataFrame:

    cache_file = production_cache_path(paths)

    cache = read_cache(cache_file)

    expected_signature = scorer_signature(
        paths,
        accelerator,
        precision,
    )

    valid: Dict[
        str,
        Dict[
            str,
            Any,
        ],
    ] = {}

    needed = []

    for query in queries:

        key = f"{query[UID]}" f"|||" f"{query['Label']}"

        row = cache.get(key)

        if (
            row is not None
            and str(row.get("PromptSHA256")) == str(query["PromptSHA256"])
            and str(row.get("ScorerSignature")) == expected_signature
        ):

            valid[key] = row

        else:

            needed.append(query)

    log(f"FAST production cells     : " f"{len(queries)}")

    log(f"Valid cached cells        : " f"{len(valid)}")

    log(f"New cells                 : " f"{len(needed)}")

    if needed:

        mapper = FastLogitMapper(
            model_path=paths.model_path,
            output_root=paths.output_root,
            accelerator=accelerator,
            precision=precision,
        )

        log("Tokenizing prompt lengths...")

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
                "Input token p50/p90/p99/max: "
                f"{np.quantile(array, .50):.0f}/"
                f"{np.quantile(array, .90):.0f}/"
                f"{np.quantile(array, .99):.0f}/"
                f"{array.max()}"
            )

        started = time.time()

        completed = 0

        for (
            batch_number,
            indices,
        ) in enumerate(
            batches,
            start=1,
        ):

            batch_queries = [needed[index] for index in indices]

            prompts = [str(query["Prompt"]) for query in batch_queries]

            call_start = time.time()

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
                    "FastProbability": float(probability),
                    "FastLogitMargin": float(margin),
                    "InputTokens": int(input_length),
                    "PromptChars": int(query["PromptChars"]),
                    "PromptSHA256": str(query["PromptSHA256"]),
                    "ScorerSignature": expected_signature,
                    "Prior": float(query["Prior"]),
                    "ExampleUIDs": str(query["ExampleUIDs"]),
                    "LogitTemperature": LOGIT_TEMPERATURE,
                    "Accelerator": mapper.accelerator,
                    "LoadMode": mapper.load_mode,
                    "Verbalizer0": mapper.answer_zero[0],
                    "Verbalizer1": mapper.answer_one[0],
                    "ScriptVersion": SCRIPT_VERSION,
                }

                rows.append(row)

            append_cache(
                cache_file,
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

            if batch_number <= 5 or batch_number % 25 == 0 or completed == len(needed):

                log(
                    f"  scored "
                    f"{completed:>5}/"
                    f"{len(needed)} "
                    f"batch="
                    f"{len(rows):>2} "
                    f"maxTok="
                    f"{max(actual_lengths):>4} "
                    f"call="
                    f"{time.time()-call_start:5.2f}s "
                    f"rate="
                    f"{rate*60:7.1f} cells/min"
                )

        del mapper

        gc.collect()

    rows = []

    for query in queries:

        key = f"{query[UID]}" f"|||" f"{query['Label']}"

        if key not in valid:

            raise RuntimeError(f"Missing completed FAST cell: " f"{key}")

        rows.append(valid[key])

    result = pd.DataFrame(rows)

    if len(result) != EXPECTED_FS2_PRODUCTION_CELLS:

        raise RuntimeError(
            f"FAST production incomplete: "
            f"{len(result)}/"
            f"{EXPECTED_FS2_PRODUCTION_CELLS}"
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

        raise RuntimeError("Duplicate FAST production cells.")

    if set(result["Label"]) != set(FS2_LABELS):

        raise RuntimeError("FAST production label mismatch.")

    if result[UID].astype(str).nunique() != EXPECTED_UNLABELED:

        raise RuntimeError("FAST production UID count mismatch.")

    probabilities = pd.to_numeric(
        result["FastProbability"],
        errors="raise",
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):

        raise RuntimeError("Invalid FAST production probabilities.")

    return result.sort_values(
        [
            UID,
            "Label",
        ]
    ).reset_index(drop=True)


# =============================================================================
# TEACHER ASSEMBLY
# =============================================================================


def pivot_value(
    long_frame: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:

    wide = (
        long_frame.pivot(
            index=UID,
            columns="Label",
            values=value_column,
        )
        .reindex(columns=LABELS)
        .reset_index()
    )

    return wide


def assemble_teacher(
    base_teacher: pd.DataFrame,
    fast: pd.DataFrame,
) -> pd.DataFrame:

    base = base_teacher.copy()

    base["BaseTeacherProbability"] = base["TeacherProbability"].astype(float)

    fs2 = (
        fast[
            [
                UID,
                "Label",
                "FastProbability",
                "FastLogitMargin",
                "InputTokens",
                "PromptSHA256",
                "Accelerator",
                "LoadMode",
            ]
        ]
        .copy()
        .rename(
            columns={
                "FastProbability": "W40FastProbability",
                "FastLogitMargin": "W40FastLogitMargin",
                "InputTokens": "W40InputTokens",
                "PromptSHA256": "W40PromptSHA256",
                "Accelerator": "W40Accelerator",
                "LoadMode": "W40LoadMode",
            }
        )
    )

    merged = base.merge(
        fs2,
        on=[
            UID,
            "Label",
        ],
        how="left",
        validate="one_to_one",
    )

    target_mask = merged["Label"].isin(FS2_LABELS)

    if (
        merged.loc[
            target_mask,
            "W40FastProbability",
        ]
        .isna()
        .any()
    ):

        raise RuntimeError("Missing W40 FAST probability " "for one or more FS2 cells.")

    # -------------------------------------------------------------
    # LOCKED W39 formula.
    # -------------------------------------------------------------

    merged.loc[
        target_mask,
        "TeacherProbability",
    ] = (1.0 - FIXED_BLEND_ALPHA) * merged.loc[
        target_mask,
        "BaseTeacherProbability",
    ].astype(
        float
    ) + FIXED_BLEND_ALPHA * merged.loc[
        target_mask,
        "W40FastProbability",
    ].astype(
        float
    )

    # -------------------------------------------------------------
    # Provenance only.
    # -------------------------------------------------------------

    if "TeacherSource" not in merged.columns:

        merged["TeacherSource"] = "W2.6P_base"

    merged.loc[
        target_mask,
        "TeacherSource",
    ] = (
        "W2.6P_base_50pct_plus_" "W39_FS2_FAST_50pct"
    )

    # -------------------------------------------------------------
    # CRITICAL:
    # RecommendedTeacherWeight and RecommendedTeacherMask are never
    # modified in W40.
    # -------------------------------------------------------------

    probabilities = merged["TeacherProbability"].to_numpy(dtype=float)

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):

        raise RuntimeError("Invalid W40 teacher probabilities.")

    if len(merged) != EXPECTED_ALL_PRODUCTION_CELLS:

        raise RuntimeError("W40 final teacher row count mismatch.")

    # -------------------------------------------------------------
    # Other 10 labels MUST remain unchanged.
    # -------------------------------------------------------------

    non_target = ~target_mask

    if not np.allclose(
        merged.loc[
            non_target,
            "TeacherProbability",
        ].to_numpy(dtype=float),
        merged.loc[
            non_target,
            "BaseTeacherProbability",
        ].to_numpy(dtype=float),
        atol=0,
        rtol=0,
    ):

        raise RuntimeError("A non-FS2 label changed unexpectedly.")

    return merged.sort_values(
        [
            UID,
            "Label",
        ]
    ).reset_index(drop=True)


# =============================================================================
# STATUS
# =============================================================================


def run_status(
    paths: ProjectPaths,
    accelerator: str,
    precision: str,
) -> Dict[
    str,
    Any,
]:

    ensure_dirs(paths)

    validate_required_files(paths)

    gate = validate_w39_gate(paths)

    _train, _gold, unlabeled = load_train(paths)

    base = load_base_teacher(
        paths,
        unlabeled,
    )

    selected = int(parse_bool_series(base["RecommendedTeacherMask"]).sum())

    config = paths.model_path / "config.json"

    model_config_sha = sha256_file(config) if config.exists() else None

    payload = {
        "script_version": SCRIPT_VERSION,
        "experiment": DISPLAY_VERSION,
        "targets": FS2_LABELS,
        "expected_fast_cells": EXPECTED_FS2_PRODUCTION_CELLS,
        "blend_alpha": FIXED_BLEND_ALPHA,
        "logit_temperature": LOGIT_TEMPERATURE,
        "accelerator_requested": accelerator,
        "accelerator_resolved": resolve_accelerator(accelerator),
        "precision": precision,
        "accelerator_env_variables_required": False,
        "paths": {
            key: str(value)
            for (
                key,
                value,
            ) in asdict(paths).items()
        },
        "model_exists": paths.model_path.exists(),
        "model_config_sha256": model_config_sha,
        "model_matches_w39": (model_config_sha == EXPECTED_W39_MODEL_CONFIG_SHA256),
        "w39_gate_verdict": gate["gate_verdict"],
        "w39_primary_delta": gate["primary_mean_AUROC_delta"],
        "base_teacher_rows": len(base),
        "base_selected_cells": selected,
        "cached_fast_cells": len(read_cache(production_cache_path(paths))),
        "policy": {
            "change_probabilities": FS2_LABELS,
            "preserve_other_probabilities": True,
            "preserve_all_weights": True,
            "preserve_all_masks": True,
            "synovitis_changed": False,
        },
        "scope": {
            "production_teacher_only": True,
            "image_training": False,
            "curia": False,
            "dicom": False,
            "pilkwang": False,
        },
    }

    if not payload["model_matches_w39"]:

        raise RuntimeError("Qwen config SHA does not match W39.")

    write_json(
        paths.result_root / "00_status.json",
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
# PRODUCTION
# =============================================================================


def run_production(
    paths: ProjectPaths,
    accelerator: str,
    precision: str,
) -> Dict[
    str,
    Any,
]:

    ensure_dirs(paths)

    validate_required_files(paths)

    gate = validate_w39_gate(paths)

    config = paths.model_path / "config.json"

    if not config.exists():

        raise FileNotFoundError(f"Qwen config missing: " f"{config}")

    current_model_sha = sha256_file(config)

    if current_model_sha != EXPECTED_W39_MODEL_CONFIG_SHA256:

        raise RuntimeError("W40 Qwen config does not " "match the W39 validated model.")

    train, gold, unlabeled = load_train(paths)

    gold_w2, full_w2 = load_w2_tables(
        paths,
        train,
        gold,
    )

    base_teacher = load_base_teacher(
        paths,
        unlabeled,
    )

    # Keep a complete snapshot of the production base.
    base_teacher.to_csv(
        paths.result_root / "10_base_teacher_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("=" * 100)

    log(DISPLAY_VERSION)

    log("=" * 100)

    log(f"W39 verdict              : " f"{gate['gate_verdict']}")

    log(f"W39 FS2 AUROC delta      : " f"{gate['primary_mean_AUROC_delta']:+.6f}")

    log(f"Production studies       : " f"{len(unlabeled)}")

    log(f"FAST production cells    : " f"{EXPECTED_FS2_PRODUCTION_CELLS}")

    log(f"Targets                  : " f"{FS2_LABELS}")

    log(f"Blend                    : " f"{FIXED_BLEND_ALPHA:.2f}")

    log("Weights changed          : NO")

    log("Masks changed            : NO")

    log("Synovitis changed        : NO")

    log("Image training           : DISABLED")

    # -------------------------------------------------------------
    # Build deterministic all-58 production retrieval.
    # -------------------------------------------------------------

    queries, exemplar_audit = build_production_queries(
        gold,
        unlabeled,
        gold_w2,
        full_w2,
    )

    exemplar_audit.to_csv(
        paths.result_root / "11_production_exemplar_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prompt_metadata = pd.DataFrame(
        [
            {
                UID: query[UID],
                "Label": query["Label"],
                "PromptSHA256": query["PromptSHA256"],
                "PromptChars": query["PromptChars"],
                "Prior": query["Prior"],
                "ExampleUIDs": query["ExampleUIDs"],
            }
            for query in queries
        ]
    )

    prompt_metadata.to_csv(
        paths.result_root / "12_production_prompt_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # FAST scoring.
    # -------------------------------------------------------------

    fast = run_fast_scoring(
        paths=paths,
        queries=queries,
        accelerator=accelerator,
        precision=precision,
    )

    fast.to_csv(
        paths.result_root / "13_fast_fs2_production_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        fast.pivot(
            index=UID,
            columns="Label",
            values="FastProbability",
        )
        .reindex(columns=FS2_LABELS)
        .reset_index()
        .to_csv(
            paths.result_root / "14_fast_fs2_probabilities_wide.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )

    # -------------------------------------------------------------
    # Controlled assembly.
    # -------------------------------------------------------------

    final_teacher = assemble_teacher(
        base_teacher,
        fast,
    )

    final_teacher.to_csv(
        paths.result_root / "15_final_teacher_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    probability_wide = pivot_value(
        final_teacher,
        "TeacherProbability",
    )

    weight_wide = pivot_value(
        final_teacher,
        "RecommendedTeacherWeight",
    )

    mask_wide = pivot_value(
        final_teacher,
        "RecommendedTeacherMask",
    )

    probability_wide.to_csv(
        paths.result_root / "16_final_probabilities_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weight_wide.to_csv(
        paths.result_root / "17_teacher_weights_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mask_wide.to_csv(
        paths.result_root / "18_teacher_mask_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    target_rows = final_teacher[final_teacher["Label"].isin(FS2_LABELS)]

    mask_bool = parse_bool_series(final_teacher["RecommendedTeacherMask"])

    label_stats = []

    for label in FS2_LABELS:

        label_rows = target_rows[target_rows["Label"] == label]

        label_masks = parse_bool_series(label_rows["RecommendedTeacherMask"])

        label_stats.append(
            {
                "Label": label,
                "N": len(label_rows),
                "BaseProbabilityMean": float(
                    label_rows["BaseTeacherProbability"].mean()
                ),
                "FastProbabilityMean": float(label_rows["W40FastProbability"].mean()),
                "FinalProbabilityMean": float(label_rows["TeacherProbability"].mean()),
                "FinalProbabilityStd": float(label_rows["TeacherProbability"].std()),
                "SelectedCells": int(label_masks.sum()),
                "SelectedFraction": float(label_masks.mean()),
                "MeanSelectedWeight": float(
                    label_rows.loc[
                        label_masks,
                        "RecommendedTeacherWeight",
                    ].mean()
                ),
            }
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "PRODUCTION_COMPLETE",
        "w39_gate": {
            "verdict": gate["gate_verdict"],
            "target_mean_AUROC_delta": gate["primary_mean_AUROC_delta"],
            "bootstrap_p_delta_gt_0": gate["bootstrap"]["p_delta_gt_0"],
        },
        "model_config_sha256": current_model_sha,
        "targets": FS2_LABELS,
        "blend_alpha": FIXED_BLEND_ALPHA,
        "logit_temperature": LOGIT_TEMPERATURE,
        "counts": {
            "unlabeled_studies": int(final_teacher[UID].nunique()),
            "fast_fs2_cells": int(len(fast)),
            "final_teacher_cells": int(len(final_teacher)),
            "selected_cells": int(mask_bool.sum()),
        },
        "target_statistics": label_stats,
        "controlled_changes": {
            "probabilities_changed": FS2_LABELS,
            "other_10_probabilities_preserved": True,
            "all_teacher_weights_preserved": True,
            "all_teacher_masks_preserved": True,
            "synovitis_probability_preserved": True,
            "synovitis_policy_preserved": True,
        },
        "scientific_warning": (
            "Production FAST predictions use all 58 gold reports "
            "as labeled exemplars. They are valid final training "
            "resources but are not pristine OOF evidence."
        ),
        "image_training_authorized": False,
        "next_step": (
            "Run W40 validate. If validation passes, use these "
            "teacher artifacts as the single changed supervision "
            "input for the next W6.0-architecture image run."
        ),
    }

    write_json(
        paths.result_root / "19_production_summary.json",
        summary,
    )

    log("")
    log("=" * 100)
    log("W40 PRODUCTION COMPLETE")
    log("=" * 100)

    log(f"FAST FS2 cells            : " f"{len(fast)}")

    log(f"Final teacher cells       : " f"{len(final_teacher)}")

    log(f"Selected cells preserved  : " f"{int(mask_bool.sum())}")

    for row in label_stats:

        log(
            f"{row['Label']:<10} "
            f"base_mean="
            f"{row['BaseProbabilityMean']:.4f} "
            f"fast_mean="
            f"{row['FastProbabilityMean']:.4f} "
            f"final_mean="
            f"{row['FinalProbabilityMean']:.4f} "
            f"coverage="
            f"{row['SelectedFraction']:.1%}"
        )

    log("")
    log(f"Results                  : " f"{paths.result_root}")

    log("")
    log("NEXT: run validate. " "Do not train the image model before validation passes.")

    return summary


# =============================================================================
# VALIDATION
# =============================================================================


def run_validate(
    paths: ProjectPaths,
) -> Dict[
    str,
    Any,
]:

    ensure_dirs(paths)

    validate_required_files(paths)

    gate = validate_w39_gate(paths)

    required = {
        "base": paths.result_root / "10_base_teacher_snapshot.csv",
        "exemplars": paths.result_root / "11_production_exemplar_audit.csv",
        "prompts": paths.result_root / "12_production_prompt_metadata.csv",
        "fast": paths.result_root / "13_fast_fs2_production_long.csv",
        "fast_wide": paths.result_root / "14_fast_fs2_probabilities_wide.csv",
        "teacher": paths.result_root / "15_final_teacher_long.csv",
        "probabilities": paths.result_root / "16_final_probabilities_wide.csv",
        "weights": paths.result_root / "17_teacher_weights_wide.csv",
        "masks": paths.result_root / "18_teacher_mask_wide.csv",
        "summary": paths.result_root / "19_production_summary.json",
    }

    exists = {
        f"{name}_exists": path.exists()
        for (
            name,
            path,
        ) in required.items()
    }

    if not all(exists.values()):

        payload = {
            "overall_pass": False,
            "checks": exists,
        }

        write_json(
            paths.result_root / "20_validation_summary.json",
            payload,
        )

        log(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return payload

    base = pd.read_csv(required["base"])

    fast = pd.read_csv(required["fast"])

    exemplars = pd.read_csv(required["exemplars"])

    prompts = pd.read_csv(required["prompts"])

    teacher = pd.read_csv(required["teacher"])

    probabilities = pd.read_csv(required["probabilities"])

    weights = pd.read_csv(required["weights"])

    masks = pd.read_csv(required["masks"])

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))

    for frame in [
        base,
        fast,
        exemplars,
        prompts,
        teacher,
        probabilities,
        weights,
        masks,
    ]:

        if UID in frame.columns:

            frame[UID] = frame[UID].astype(str)

    # -------------------------------------------------------------
    # Join final/base.
    # -------------------------------------------------------------

    compare = (
        base[
            [
                UID,
                "Label",
                "TeacherProbability",
                "RecommendedTeacherWeight",
                "RecommendedTeacherMask",
            ]
        ]
        .rename(
            columns={
                "TeacherProbability": "BaseProbability",
                "RecommendedTeacherWeight": "BaseWeight",
                "RecommendedTeacherMask": "BaseMask",
            }
        )
        .merge(
            teacher[
                [
                    UID,
                    "Label",
                    "TeacherProbability",
                    "RecommendedTeacherWeight",
                    "RecommendedTeacherMask",
                    "W40FastProbability",
                ]
            ],
            on=[
                UID,
                "Label",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    target = compare["Label"].isin(FS2_LABELS)

    non_target = ~target

    expected_target_probability = (1.0 - FIXED_BLEND_ALPHA) * compare.loc[
        target,
        "BaseProbability",
    ].astype(float) + FIXED_BLEND_ALPHA * compare.loc[
        target,
        "W40FastProbability",
    ].astype(
        float
    )

    weights_unchanged = np.allclose(
        compare["BaseWeight"].to_numpy(dtype=float),
        compare["RecommendedTeacherWeight"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
    )

    base_mask = parse_bool_series(compare["BaseMask"])

    final_mask = parse_bool_series(compare["RecommendedTeacherMask"])

    masks_unchanged = np.array_equal(
        base_mask,
        final_mask,
    )

    selected_cells = int(final_mask.sum())

    expected_exemplar_rows = EXPECTED_FS2_PRODUCTION_CELLS * (
        N_POS_EXAMPLES + N_NEG_EXAMPLES
    )

    checks = {
        **exists,
        "w39_gate_pass": (gate["gate_verdict"] == EXPECTED_W39_GATE),
        "w39_delta_ge_0_040": (
            float(gate["primary_mean_AUROC_delta"]) >= EXPECTED_W39_MIN_DELTA
        ),
        "base_rows_52188": (len(base) == EXPECTED_ALL_PRODUCTION_CELLS),
        "fast_rows_8698": (len(fast) == EXPECTED_FS2_PRODUCTION_CELLS),
        "fast_uids_4349": (fast[UID].nunique() == EXPECTED_UNLABELED),
        "fast_two_labels": (set(fast["Label"]) == set(FS2_LABELS)),
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
        "prompt_rows_8698": (len(prompts) == EXPECTED_FS2_PRODUCTION_CELLS),
        "exemplar_rows_expected": (len(exemplars) == expected_exemplar_rows),
        "teacher_rows_52188": (len(teacher) == EXPECTED_ALL_PRODUCTION_CELLS),
        "teacher_uids_4349": (teacher[UID].nunique() == EXPECTED_UNLABELED),
        "teacher_12_labels": (set(teacher["Label"]) == set(LABELS)),
        "teacher_no_duplicates": (
            not teacher[
                [
                    UID,
                    "Label",
                ]
            ]
            .duplicated()
            .any()
        ),
        "target_formula_exact": bool(
            np.allclose(
                compare.loc[
                    target,
                    "TeacherProbability",
                ].to_numpy(dtype=float),
                expected_target_probability.to_numpy(dtype=float),
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "other_10_probabilities_unchanged": bool(
            np.allclose(
                compare.loc[
                    non_target,
                    "TeacherProbability",
                ].to_numpy(dtype=float),
                compare.loc[
                    non_target,
                    "BaseProbability",
                ].to_numpy(dtype=float),
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "all_weights_unchanged": bool(weights_unchanged),
        "all_masks_unchanged": bool(masks_unchanged),
        "selected_cells_preserved_32027": (
            selected_cells == EXPECTED_BASE_SELECTED_CELLS
        ),
        "probability_wide_shape": (
            probabilities.shape
            == (
                EXPECTED_UNLABELED,
                13,
            )
        ),
        "weight_wide_shape": (
            weights.shape
            == (
                EXPECTED_UNLABELED,
                13,
            )
        ),
        "mask_wide_shape": (
            masks.shape
            == (
                EXPECTED_UNLABELED,
                13,
            )
        ),
        "summary_complete": (summary.get("status") == "PRODUCTION_COMPLETE"),
        "synovitis_unchanged_declared": bool(
            summary.get(
                "controlled_changes",
                {},
            ).get(
                "synovitis_policy_preserved",
                False,
            )
        ),
        "image_training_not_inside_w40": (
            summary.get("image_training_authorized") is False
        ),
        "pilkwang_not_used": True,
    }

    overall_pass = bool(all(checks.values()))

    payload = {
        "overall_pass": overall_pass,
        "checks": checks,
        "counts": {
            "fast_cells": len(fast),
            "teacher_cells": len(teacher),
            "selected_cells": selected_cells,
        },
        "targets": FS2_LABELS,
        "w39_gate_verdict": gate["gate_verdict"],
        "w39_AUROC_delta": gate["primary_mean_AUROC_delta"],
        "results_root": str(paths.result_root),
        "next_step": (
            "If overall_pass=true, W40 teacher artifacts are "
            "ready for the next image experiment using the "
            "unchanged W6.0 Curia architecture."
        ),
    }

    write_json(
        paths.result_root / "20_validation_summary.json",
        payload,
    )

    log(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )

    if not overall_pass:

        raise RuntimeError("W40 validation FAILED.")

    return payload


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
            "production",
            "validate",
        ],
        default=None,
    )

    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=[
            "status",
            "production",
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
        "--w26p-root",
        default=None,
    )

    parser.add_argument(
        "--w39-root",
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

    elif mode == "production":

        run_production(
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
