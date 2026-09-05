#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W38 Targeted Teacher Policy Audit v1

Target labels only:
    - Contusion
    - Synovitis
    - Effusion

Purpose
-------
Determine whether the remaining teacher weakness is primarily:

    KEEP      : current teacher/policy is adequate
    REWEIGHT  : probabilities have useful ranking but supervision authority is too strong
    REMASK    : specific report-state regions should not be supervised broadly
    REBUILD   : teacher probability ranking itself is too weak for policy-only repair

This is a DIAGNOSTIC ONLY.

It does NOT:
- train an image model
- run Curia
- read DICOM
- use GPU or TPU
- import another project Python script
- generate new pseudo labels
- modify W2.6-P artifacts

Inputs
------
Existing validated W37 v3 outputs:
    output/results/rsna_teacher_error_audit_v3/
        00_summary.json
        01_gold_teacher_metrics.csv
        02_gold_teacher_errors.csv
        04_report_state_vs_gold.csv          optional
        05_production_policy_audit.csv
        06_priority_findings.csv

Existing W2 structured report artifact, when available:
    output/results/rsna_w2/results/
        08_full_structured_report_labels.csv

Existing W2.6-P FAST:
    output/results/rsna_w2_6p_fast/results/
        16_final_hybrid_probabilities_wide.csv
        17_recommended_teacher_weights_wide.csv
        18_recommended_teacher_mask_wide.csv

Outputs
-------
output/results/rsna_targeted_teacher_policy_audit_v1/

    00_summary.json
    01_target_label_summary.csv
    02_gold_state_breakdown.csv
    03_production_state_policy.csv
    04_high_confidence_errors.csv
    05_recommendations.csv
    06_manual_review_cases.csv

Version
-------
w38_rsna_targeted_teacher_policy_audit_v1.py
"""

from __future__ import annotations

import argparse
import json
import re

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# VERSION
# =============================================================================

SCRIPT_VERSION = "rsna_targeted_teacher_policy_audit_v1"
DISPLAY_VERSION = "W38 TARGETED TEACHER POLICY AUDIT v1"

OUTPUT_DIR_NAME = "rsna_targeted_teacher_policy_audit_v1"


# =============================================================================
# PROJECT CONSTANTS
# =============================================================================

UID = "StudyInstanceUID"
REPORT = "Report"

TARGET_LABELS = [
    "Contusion",
    "Synovitis",
    "Effusion",
]

ALL_LABELS = [
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

# W2.6 currently changed only these.
FS4_LABELS = [
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Synovitis",
]

EXPECTED_GOLD_STUDIES = 58
EXPECTED_UNLABELED_STUDIES = 4349
EXPECTED_SELECTED_CELLS = 32027

EXPECTED_W23_MACRO_AUC = 0.771243
EXPECTED_W26_FAST_MACRO_AUC = 0.813203

HIGH_CONFIDENCE_ERROR = 0.80


# =============================================================================
# DECISION THRESHOLDS
# =============================================================================

# Probability quality
WEAK_AUC = 0.72
VERY_WEAK_AUC = 0.70

# Production-policy authority
VERY_BROAD_COVERAGE = 0.95
BROAD_COVERAGE = 0.80

HIGH_MEAN_WEIGHT = 0.80
VERY_HIGH_MEAN_WEIGHT = 0.88

# Report-state diagnostics
MIN_STATE_N = 3
HIGH_NOT_ADDRESSED_GOLD_POSITIVE_RATE = 0.30

# If a teacher already materially improved a label, do not casually rebuild it.
MATERIAL_AUC_GAIN = 0.05


# =============================================================================
# HELPERS
# =============================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def norm(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def safe_json_dump(
    obj: object,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=str,
        )


def first_existing(
    paths: Iterable[Path],
) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path.resolve()

    return None


def find_column(
    columns: Sequence[str],
    choices: Sequence[str],
) -> Optional[str]:
    mapping = {norm(c): c for c in columns}

    for choice in choices:
        key = norm(choice)

        if key in mapping:
            return mapping[key]

    return None


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


def finite_or_nan(
    value: object,
) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return np.nan


# =============================================================================
# PROJECT PATHS
# =============================================================================


def discover_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent

    for candidate in [
        script_dir,
        *script_dir.parents,
    ]:
        if (candidate / "input" / "train.csv").exists():
            return candidate

    # Standard project:
    #
    # PROJECT_ROOT/src/models/script.py
    candidate = (script_dir / ".." / "..").resolve()

    if (candidate / "input" / "train.csv").exists():
        return candidate

    raise FileNotFoundError(
        "Could not discover PROJECT_ROOT " "containing input/train.csv"
    )


@dataclass
class Paths:
    project_root: str
    train_csv: str

    w37_root: str

    w2_root: Optional[str]
    w26p_fast_root: Optional[str]

    output_root: str

    @classmethod
    def discover(cls) -> "Paths":
        root = discover_project_root()

        results = root / "output" / "results"

        train_csv = root / "input" / "train.csv"

        w37 = results / "rsna_teacher_error_audit_v3"

        if not w37.exists():
            raise FileNotFoundError(f"W37 v3 results not found: {w37}")

        w2 = first_existing(
            [
                results / "rsna_w2",
                results / "rsna_w2_full",
            ]
        )

        w26p = first_existing(
            [
                results / "rsna_w2_6p_fast",
            ]
        )

        output = results / OUTPUT_DIR_NAME

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        return cls(
            project_root=str(root),
            train_csv=str(train_csv.resolve()),
            w37_root=str(w37.resolve()),
            w2_root=(str(w2) if w2 else None),
            w26p_fast_root=(str(w26p) if w26p else None),
            output_root=str(output.resolve()),
        )


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_w37_outputs(
    paths: Paths,
) -> Dict[str, Path]:
    root = Path(paths.w37_root)

    required = {
        "summary": root / "00_summary.json",
        "metrics": root / "01_gold_teacher_metrics.csv",
        "errors": root / "02_gold_teacher_errors.csv",
        "policy": root / "05_production_policy_audit.csv",
        "priority": root / "06_priority_findings.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]

    if missing:
        raise FileNotFoundError("Missing required W37 outputs:\n" + "\n".join(missing))

    optional_state = root / "04_report_state_vs_gold.csv"

    if optional_state.exists():
        required["state_summary"] = optional_state

    return required


def validate_w37_benchmark(
    summary_path: Path,
) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    gold = summary.get(
        "gold_validation",
        {},
    )

    w23 = finite_or_nan(gold.get("w23_macro_auroc"))

    w26 = finite_or_nan(gold.get("w26_fixed50_macro_auroc"))

    if not np.isfinite(w23):
        raise RuntimeError("W37 summary does not contain " "valid W2.3 macro AUROC.")

    if not np.isfinite(w26):
        raise RuntimeError("W37 summary does not contain " "valid W2.6 macro AUROC.")

    if abs(w23 - EXPECTED_W23_MACRO_AUC) > 0.002:
        raise RuntimeError("W37 W2.3 benchmark mismatch: " f"{w23:.6f}")

    if abs(w26 - EXPECTED_W26_FAST_MACRO_AUC) > 0.003:
        raise RuntimeError("W37 W2.6 FAST benchmark mismatch: " f"{w26:.6f}")

    return summary


# =============================================================================
# LOAD W37 TABLES
# =============================================================================


def load_target_metrics(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "Model",
        "Label",
        "AUROC",
        "AP",
        "Brier",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"{path}: missing columns " f"{sorted(missing)}")

    df = df[df["Label"].isin(TARGET_LABELS)].copy()

    expected_models = {
        "W2.3",
        "W2.6_fixed50",
    }

    actual_models = set(df["Model"])

    if not expected_models.issubset(actual_models):
        raise RuntimeError(
            f"{path}: missing model rows. " f"Found={sorted(actual_models)}"
        )

    return df


def load_target_errors(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        UID,
        "Label",
        "Gold",
        "ReportState",
        "W23Probability",
        "W26Probability",
        "W23ErrorScore",
        "W26ErrorScore",
        "W26HighConfidenceError",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"{path}: missing columns " f"{sorted(missing)}")

    df = df[df["Label"].isin(TARGET_LABELS)].copy()

    if len(df) != (EXPECTED_GOLD_STUDIES * len(TARGET_LABELS)):
        raise RuntimeError("Unexpected target gold-cell count: " f"{len(df)}")

    df[UID] = df[UID].astype(str)

    df["Gold"] = pd.to_numeric(
        df["Gold"],
        errors="raise",
    ).astype(int)

    for column in [
        "W23Probability",
        "W26Probability",
        "W23ErrorScore",
        "W26ErrorScore",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df["W26HighConfidenceError"] = parse_bool_series(df["W26HighConfidenceError"])

    return df


def load_target_policy(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "Label",
        "Studies",
        "Selected",
        "Coverage",
        "MeanProbability",
        "StdProbability",
        "MeanSelectedWeight",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"{path}: missing columns " f"{sorted(missing)}")

    df = df[df["Label"].isin(TARGET_LABELS)].copy()

    if len(df) != len(TARGET_LABELS):
        raise RuntimeError(
            "Production policy does not contain " "all three target labels."
        )

    return df


def load_priority(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Label" not in df.columns:
        raise RuntimeError(f"{path}: Label column missing")

    return df[df["Label"].isin(TARGET_LABELS)].copy()


# =============================================================================
# GOLD STATE BREAKDOWN
# =============================================================================


def normalize_state(
    value: object,
) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    if text in {
        "P",
        "PRESENT",
        "POSITIVE",
    }:
        return "P"

    if text in {
        "A",
        "ABSENT",
        "NEGATIVE",
    }:
        return "A"

    if text in {
        "U",
        "UNCERTAIN",
        "EQUIVOCAL",
    }:
        return "U"

    if text in {
        "N",
        "NOT_ADDRESSED",
        "NOT ADDRESSED",
        "NOTMENTIONED",
    }:
        return "N"

    return "UNKNOWN"


def build_gold_state_breakdown(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    x = errors.copy()

    x["ReportStateNormalized"] = x["ReportState"].map(normalize_state)

    rows = []

    for (
        label,
        state,
    ), group in x.groupby(
        [
            "Label",
            "ReportStateNormalized",
        ],
        dropna=False,
    ):
        n = len(group)

        positives = int(group["Gold"].sum())

        negative = n - positives

        high_errors = int(group["W26HighConfidenceError"].sum())

        rows.append(
            {
                "Label": label,
                "ReportState": state,
                "N": n,
                "GoldPositive": positives,
                "GoldNegative": negative,
                "GoldPositiveRate": (positives / n if n else np.nan),
                "MeanW23Probability": float(group["W23Probability"].mean()),
                "MeanW26Probability": float(group["W26Probability"].mean()),
                "MeanW23Error": float(group["W23ErrorScore"].mean()),
                "MeanW26Error": float(group["W26ErrorScore"].mean()),
                "HighConfidenceErrors": high_errors,
                "HighConfidenceErrorRate": (high_errors / n if n else np.nan),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Label",
                "ReportState",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# FULL W2 REPORT STATE DISCOVERY
# =============================================================================

LABEL_ALIASES = {norm(label): label for label in ALL_LABELS}

LABEL_ALIASES.update(
    {
        "baker": "Baker's",
        "bakers": "Baker's",
        "bakercyst": "Baker's",
        "bakerscyst": "Baker's",
        "pfoa": "PF OA",
    }
)


def canonical_label(
    value: object,
) -> Optional[str]:
    return LABEL_ALIASES.get(norm(value))


def explicit_state(
    value: object,
) -> Optional[str]:
    if pd.isna(value):
        return None

    x = norm(value)

    if x in {
        "p",
        "present",
        "positive",
        "abnormal",
    }:
        return "P"

    if x in {
        "a",
        "absent",
        "negative",
        "normal",
    }:
        return "A"

    if x in {
        "u",
        "uncertain",
        "equivocal",
        "indeterminate",
    }:
        return "U"

    if x in {
        "n",
        "notaddressed",
        "notmentioned",
        "unaddressed",
    }:
        return "N"

    return None


def discover_full_state_file(
    paths: Paths,
) -> Optional[Path]:
    if paths.w2_root is None:
        return None

    root = Path(paths.w2_root)

    exact = first_existing(
        [
            root / "results" / "08_full_structured_report_labels.csv",
            root / "08_full_structured_report_labels.csv",
        ]
    )

    if exact:
        return exact

    matches = list(root.rglob("08_full_structured_report_labels.csv"))

    if matches:
        return sorted(
            matches,
            key=lambda p: len(str(p)),
        )[0].resolve()

    return None


def load_full_report_states(
    paths: Paths,
) -> Tuple[
    Optional[pd.DataFrame],
    dict,
]:
    path = discover_full_state_file(paths)

    if path is None:
        return (
            None,
            {
                "available": False,
                "reason": "08_full_structured_report_labels.csv not found",
            },
        )

    df = pd.read_csv(path)

    uid_col = find_column(
        df.columns,
        [
            UID,
            "StudyUID",
            "study_uid",
            "UID",
        ],
    )

    label_col = find_column(
        df.columns,
        [
            "Label",
            "Target",
            "Diagnosis",
            "Abnormality",
        ],
    )

    # Structural label-column fallback.
    if label_col is None:
        for column in df.columns:
            values = df[column].dropna().astype(str).map(canonical_label)

            if len(values) >= 100 and values.notna().mean() >= 0.80:
                label_col = column
                break

    if uid_col is None or label_col is None:
        return (
            None,
            {
                "available": False,
                "source": str(path),
                "reason": "Could not structurally identify UID/label columns",
            },
        )

    tmp = df.copy()

    tmp[uid_col] = tmp[uid_col].astype(str)

    tmp["_Label"] = tmp[label_col].map(canonical_label)

    tmp = tmp[tmp["_Label"].isin(TARGET_LABELS)].copy()

    state_candidates = []

    for column in tmp.columns:
        if column in {
            uid_col,
            label_col,
            "_Label",
        }:
            continue

        mapped = tmp[column].map(explicit_state)

        coverage = float(mapped.notna().mean())

        unique = set(mapped.dropna())

        if coverage < 0.65 or len(unique) < 2:
            continue

        score = coverage * 10.0

        name = norm(column)

        if "state" in name:
            score += 5

        if "final" in name:
            score += 4

        if "fused" in name:
            score += 3

        if "assertion" in name:
            score += 2

        state_candidates.append(
            (
                score,
                column,
                mapped,
            )
        )

    if not state_candidates:
        return (
            None,
            {
                "available": False,
                "source": str(path),
                "reason": "No explicit P/A/U/N state column identified",
            },
        )

    state_candidates.sort(key=lambda x: -x[0])

    _, state_col, mapped = state_candidates[0]

    tmp["_State"] = mapped

    tmp = tmp[tmp["_State"].notna()].copy()

    if (
        tmp[
            [
                uid_col,
                "_Label",
            ]
        ]
        .duplicated()
        .any()
    ):
        tmp = tmp.sort_values(
            [
                uid_col,
                "_Label",
            ]
        ).drop_duplicates(
            [
                uid_col,
                "_Label",
            ],
            keep="first",
        )

    result = tmp[
        [
            uid_col,
            "_Label",
            "_State",
        ]
    ].copy()

    result.rename(
        columns={
            uid_col: UID,
            "_Label": "Label",
            "_State": "ReportState",
        },
        inplace=True,
    )

    return (
        result,
        {
            "available": True,
            "source": str(path.resolve()),
            "state_column": state_col,
            "rows": len(result),
        },
    )


# =============================================================================
# LOAD RAW PRODUCTION W2.6-P FILES
# =============================================================================


def discover_named_file(
    root: Path,
    names: Sequence[str],
) -> Optional[Path]:
    for name in names:
        candidates = [
            root / name,
            root / "results" / name,
        ]

        found = first_existing(candidates)

        if found:
            return found

        matches = list(root.rglob(name))

        if matches:
            return sorted(
                matches,
                key=lambda p: len(str(p)),
            )[0].resolve()

    return None


def load_raw_production(
    paths: Paths,
) -> Tuple[
    Optional[pd.DataFrame],
    dict,
]:
    if paths.w26p_fast_root is None:
        return (
            None,
            {
                "available": False,
                "reason": "rsna_w2_6p_fast not found",
            },
        )

    root = Path(paths.w26p_fast_root)

    probability_file = discover_named_file(
        root,
        [
            "16_final_hybrid_probabilities_wide.csv",
            "07_final_hybrid_probabilities_wide.csv",
        ],
    )

    weight_file = discover_named_file(
        root,
        [
            "17_recommended_teacher_weights_wide.csv",
            "08_recommended_teacher_weights_wide.csv",
        ],
    )

    mask_file = discover_named_file(
        root,
        [
            "18_recommended_teacher_mask_wide.csv",
            "09_recommended_teacher_mask_wide.csv",
        ],
    )

    if not all(
        [
            probability_file,
            weight_file,
            mask_file,
        ]
    ):
        return (
            None,
            {
                "available": False,
                "reason": "Production probability/weight/mask files incomplete",
                "probability_file": str(probability_file) if probability_file else None,
                "weight_file": str(weight_file) if weight_file else None,
                "mask_file": str(mask_file) if mask_file else None,
            },
        )

    p = pd.read_csv(probability_file)

    w = pd.read_csv(weight_file)

    m = pd.read_csv(mask_file)

    def uid_column(
        frame: pd.DataFrame,
    ) -> str:
        column = find_column(
            frame.columns,
            [
                UID,
                "StudyUID",
                "study_uid",
            ],
        )

        if column is None:
            raise RuntimeError("Production file missing UID column")

        return column

    puid = uid_column(p)
    wuid = uid_column(w)
    muid = uid_column(m)

    for frame, column in [
        (p, puid),
        (w, wuid),
        (m, muid),
    ]:
        frame[column] = frame[column].astype(str)

    if not (
        len(p) == EXPECTED_UNLABELED_STUDIES
        and len(w) == EXPECTED_UNLABELED_STUDIES
        and len(m) == EXPECTED_UNLABELED_STUDIES
    ):
        raise RuntimeError("Unexpected production row count")

    p = p.set_index(puid)

    w = w.set_index(wuid)

    m = m.set_index(muid)

    common = set(p.index) & set(w.index) & set(m.index)

    if len(common) != (EXPECTED_UNLABELED_STUDIES):
        raise RuntimeError("Production UID sets do not match")

    rows = []

    selected_total = 0

    for uid in sorted(common):
        for label in TARGET_LABELS:
            if (
                label not in p.columns
                or label not in w.columns
                or label not in m.columns
            ):
                raise RuntimeError(f"Production files missing target label {label}")

            probability = float(
                p.loc[
                    uid,
                    label,
                ]
            )

            weight = float(
                w.loc[
                    uid,
                    label,
                ]
            )

            mask = bool(
                parse_bool_series(
                    pd.Series(
                        [
                            m.loc[
                                uid,
                                label,
                            ]
                        ]
                    )
                )[0]
            )

            selected_total += int(mask)

            rows.append(
                {
                    UID: uid,
                    "Label": label,
                    "Probability": probability,
                    "Weight": weight,
                    "Mask": mask,
                }
            )

    return (
        pd.DataFrame(rows),
        {
            "available": True,
            "probability_file": str(probability_file.resolve()),
            "weight_file": str(weight_file.resolve()),
            "mask_file": str(mask_file.resolve()),
            "target_selected_cells": selected_total,
        },
    )


# =============================================================================
# PRODUCTION STATE × POLICY ANALYSIS
# =============================================================================


def build_production_state_policy(
    production: Optional[pd.DataFrame],
    full_states: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if production is None or full_states is None:
        return None

    merged = production.merge(
        full_states,
        on=[
            UID,
            "Label",
        ],
        how="left",
        validate="one_to_one",
    )

    merged["ReportState"] = merged["ReportState"].fillna("UNKNOWN")

    rows = []

    for (
        label,
        state,
    ), group in merged.groupby(
        [
            "Label",
            "ReportState",
        ]
    ):
        n = len(group)

        selected = int(group["Mask"].sum())

        selected_group = group[group["Mask"]]

        rows.append(
            {
                "Label": label,
                "ReportState": state,
                "N": n,
                "ShareOfLabel": n
                / max(
                    1,
                    len(merged[merged["Label"] == label]),
                ),
                "Selected": selected,
                "SelectedRate": selected / n if n else np.nan,
                "MeanProbability": float(group["Probability"].mean()),
                "MeanWeightAll": float(group["Weight"].mean()),
                "MeanSelectedWeight": (
                    float(selected_group["Weight"].mean())
                    if len(selected_group)
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Label",
                "ReportState",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# LABEL SUMMARY
# =============================================================================


def metric_lookup(
    metrics: pd.DataFrame,
    model: str,
    label: str,
    field: str,
) -> float:
    row = metrics[(metrics["Model"] == model) & (metrics["Label"] == label)]

    if len(row) != 1:
        raise RuntimeError(f"Cannot uniquely find " f"{model}/{label}/{field}")

    return float(row.iloc[0][field])


def state_lookup(
    state_breakdown: pd.DataFrame,
    label: str,
    state: str,
    field: str,
) -> float:
    row = state_breakdown[
        (state_breakdown["Label"] == label) & (state_breakdown["ReportState"] == state)
    ]

    if row.empty:
        return np.nan

    return finite_or_nan(row.iloc[0][field])


def build_target_summary(
    metrics: pd.DataFrame,
    errors: pd.DataFrame,
    state_breakdown: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    policy_idx = policy.set_index("Label")

    rows = []

    for label in TARGET_LABELS:
        w23_auc = metric_lookup(
            metrics,
            "W2.3",
            label,
            "AUROC",
        )

        w26_auc = metric_lookup(
            metrics,
            "W2.6_fixed50",
            label,
            "AUROC",
        )

        w23_ap = metric_lookup(
            metrics,
            "W2.3",
            label,
            "AP",
        )

        w26_ap = metric_lookup(
            metrics,
            "W2.6_fixed50",
            label,
            "AP",
        )

        w23_brier = metric_lookup(
            metrics,
            "W2.3",
            label,
            "Brier",
        )

        w26_brier = metric_lookup(
            metrics,
            "W2.6_fixed50",
            label,
            "Brier",
        )

        label_errors = errors[errors["Label"] == label]

        high_errors = int(label_errors["W26HighConfidenceError"].sum())

        p = policy_idx.loc[label]

        n_n = state_lookup(
            state_breakdown,
            label,
            "N",
            "N",
        )

        n_positive_rate = state_lookup(
            state_breakdown,
            label,
            "N",
            "GoldPositiveRate",
        )

        n_error = state_lookup(
            state_breakdown,
            label,
            "N",
            "MeanW26Error",
        )

        present_gold_negative = 0

        p_rows = label_errors[label_errors["ReportState"].map(normalize_state) == "P"]

        if len(p_rows):
            present_gold_negative = int((p_rows["Gold"] == 0).sum())

        absent_gold_positive = 0

        a_rows = label_errors[label_errors["ReportState"].map(normalize_state) == "A"]

        if len(a_rows):
            absent_gold_positive = int((a_rows["Gold"] == 1).sum())

        rows.append(
            {
                "Label": label,
                "ChangedByW26": label in FS4_LABELS,
                "W23_AUROC": w23_auc,
                "W26_AUROC": w26_auc,
                "Delta_AUROC": w26_auc - w23_auc,
                "W23_AP": w23_ap,
                "W26_AP": w26_ap,
                "Delta_AP": w26_ap - w23_ap,
                "W23_Brier": w23_brier,
                "W26_Brier": w26_brier,
                "Delta_Brier": w26_brier - w23_brier,
                "HighConfidenceErrors": high_errors,
                "ReportPresentGoldNegative": present_gold_negative,
                "ReportAbsentGoldPositive": absent_gold_positive,
                "NotAddressedN": n_n,
                "NotAddressedGoldPositiveRate": n_positive_rate,
                "NotAddressedMeanTeacherError": n_error,
                "ProductionSelected": int(p["Selected"]),
                "ProductionCoverage": float(p["Coverage"]),
                "ProductionMeanWeight": float(p["MeanSelectedWeight"]),
                "ProductionMeanProbability": float(p["MeanProbability"]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================


def recommendation_for_label(
    row: pd.Series,
    production_state_policy: Optional[pd.DataFrame],
) -> dict:
    label = str(row["Label"])

    auc = float(row["W26_AUROC"])

    delta_auc = float(row["Delta_AUROC"])

    brier_delta = float(row["Delta_Brier"])

    coverage = float(row["ProductionCoverage"])

    mean_weight = float(row["ProductionMeanWeight"])

    high_errors = int(row["HighConfidenceErrors"])

    changed = bool(row["ChangedByW26"])

    n_count = finite_or_nan(row["NotAddressedN"])

    n_positive_rate = finite_or_nan(row["NotAddressedGoldPositiveRate"])

    reasons: List[str] = []

    # -----------------------------------------------------------------
    # 1. REBUILD
    #
    # Low AUROC means ranking itself is weak.
    # Masking/reweighting cannot fundamentally repair poor ranking.
    # -----------------------------------------------------------------

    rebuild = False

    if auc < VERY_WEAK_AUC:
        rebuild = True

        reasons.append(f"Gold AUROC is very weak ({auc:.4f} < {VERY_WEAK_AUC:.2f}).")

    elif auc < WEAK_AUC and not changed:
        rebuild = True

        reasons.append(
            f"Gold AUROC remains weak ({auc:.4f}) "
            "and this label was never upgraded by W2.6."
        )

    elif auc < WEAK_AUC and high_errors >= 4 and delta_auc < MATERIAL_AUC_GAIN:
        rebuild = True

        reasons.append(
            f"Weak AUROC plus {high_errors} high-confidence errors "
            "suggest probability quality rather than only policy weakness."
        )

    # Protect a teacher that already materially improved the label.
    if changed and delta_auc >= MATERIAL_AUC_GAIN:
        rebuild = False

        reasons.append(
            f"W2.6 materially improved AUROC by {delta_auc:+.4f}; "
            "avoid discarding useful ranking."
        )

    if rebuild:
        return {
            "Recommendation": "REBUILD",
            "Confidence": "HIGH" if auc < VERY_WEAK_AUC else "MEDIUM",
            "Reasons": reasons,
            "PriorityScore": 100 + (WEAK_AUC - auc) * 100 + high_errors,
        }

    # -----------------------------------------------------------------
    # 2. REMASK
    #
    # Prefer REMASK only when we can identify a state-specific unsafe
    # region and production is actually supervising that region.
    # -----------------------------------------------------------------

    if production_state_policy is not None:
        label_policy = production_state_policy[
            production_state_policy["Label"] == label
        ]

        n_policy = label_policy[label_policy["ReportState"] == "N"]

        if (
            not n_policy.empty
            and np.isfinite(n_positive_rate)
            and np.isfinite(n_count)
            and n_count >= MIN_STATE_N
        ):
            production_n_selected_rate = float(n_policy.iloc[0]["SelectedRate"])

            production_n_weight = finite_or_nan(n_policy.iloc[0]["MeanSelectedWeight"])

            # Unsafe state-specific policy:
            #
            # N is often gold-positive, yet production is strongly
            # supervising essentially all N rows.
            if (
                n_positive_rate >= HIGH_NOT_ADDRESSED_GOLD_POSITIVE_RATE
                and production_n_selected_rate >= BROAD_COVERAGE
                and np.isfinite(production_n_weight)
                and production_n_weight >= HIGH_MEAN_WEIGHT
            ):
                reasons.append(
                    "NOT_ADDRESSED is frequently challenge-positive "
                    f"on gold ({n_positive_rate:.1%})."
                )

                reasons.append(
                    "Production selects "
                    f"{production_n_selected_rate:.1%} of N-state rows "
                    f"with mean weight {production_n_weight:.3f}."
                )

                return {
                    "Recommendation": "REMASK",
                    "Confidence": "HIGH",
                    "Reasons": reasons,
                    "PriorityScore": 80 + production_n_selected_rate * 10,
                }

    # -----------------------------------------------------------------
    # 3. REWEIGHT
    #
    # Useful probabilities but too much authority.
    # -----------------------------------------------------------------

    if (
        coverage >= VERY_BROAD_COVERAGE
        and mean_weight >= VERY_HIGH_MEAN_WEIGHT
        and auc < 0.80
    ):
        reasons.append(
            f"Production supervision is nearly universal " f"(coverage={coverage:.1%})."
        )

        reasons.append(
            f"Mean selected authority is very high "
            f"({mean_weight:.3f}) despite gold AUROC {auc:.4f}."
        )

        if delta_auc >= MATERIAL_AUC_GAIN:
            reasons.append(
                "Teacher ranking improved materially, so reducing authority "
                "is safer than rebuilding probabilities."
            )

        return {
            "Recommendation": "REWEIGHT",
            "Confidence": "HIGH",
            "Reasons": reasons,
            "PriorityScore": 65 + coverage * 10,
        }

    if coverage >= BROAD_COVERAGE and mean_weight >= HIGH_MEAN_WEIGHT and auc < 0.76:
        reasons.append(
            "Broad/high-authority production supervision "
            "is stronger than gold validation supports."
        )

        return {
            "Recommendation": "REWEIGHT",
            "Confidence": "MEDIUM",
            "Reasons": reasons,
            "PriorityScore": 55,
        }

    # -----------------------------------------------------------------
    # 4. KEEP
    # -----------------------------------------------------------------

    reasons.append(
        "No strong evidence that probability ranking, mask coverage, "
        "or supervision authority requires intervention."
    )

    if brier_delta < 0:
        reasons.append(f"Calibration improved (Brier Δ={brier_delta:+.4f}).")

    return {
        "Recommendation": "KEEP",
        "Confidence": "MEDIUM",
        "Reasons": reasons,
        "PriorityScore": 0,
    }


def build_recommendations(
    target_summary: pd.DataFrame,
    production_state_policy: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for _, row in target_summary.iterrows():
        decision = recommendation_for_label(
            row,
            production_state_policy,
        )

        rows.append(
            {
                "Label": row["Label"],
                "Recommendation": decision["Recommendation"],
                "Confidence": decision["Confidence"],
                "PriorityScore": decision["PriorityScore"],
                "W26_AUROC": row["W26_AUROC"],
                "Delta_AUROC": row["Delta_AUROC"],
                "ProductionCoverage": row["ProductionCoverage"],
                "ProductionMeanWeight": row["ProductionMeanWeight"],
                "HighConfidenceErrors": row["HighConfidenceErrors"],
                "Reasons": " | ".join(decision["Reasons"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "PriorityScore",
                "W26_AUROC",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# SINGLE NEXT-STEP DECISION
# =============================================================================


def determine_next_step(
    recommendations: pd.DataFrame,
) -> dict:
    by_label = recommendations.set_index("Label")

    rebuild_labels = recommendations[recommendations["Recommendation"] == "REBUILD"][
        "Label"
    ].tolist()

    remask_labels = recommendations[recommendations["Recommendation"] == "REMASK"][
        "Label"
    ].tolist()

    reweight_labels = recommendations[recommendations["Recommendation"] == "REWEIGHT"][
        "Label"
    ].tolist()

    # -----------------------------------------------------------------
    # Highest-value low-experiment path:
    #
    # If both Contusion and Effusion need REBUILD, they share the same
    # defect:
    #     old W2.3 teacher remains weak and was never upgraded by W2.6.
    #
    # That justifies ONE gold-only FS2 challenge-mapper gate.
    #
    # It does NOT authorize all-4349 production generation yet.
    # -----------------------------------------------------------------

    if {
        "Contusion",
        "Effusion",
    }.issubset(set(rebuild_labels)):
        return {
            "Verdict": "GATE_ONE_FS2_REBUILD",
            "NextExperiment": "Gold-only fold-safe FS2 challenge mapper",
            "Labels": [
                "Contusion",
                "Effusion",
            ],
            "ProductionGenerationAuthorized": False,
            "ImageTrainingAuthorized": False,
            "Reason": (
                "Both untouched W2.3 labels show weak probability ranking. "
                "A single shared challenge-mapping methodology can test both "
                "with only 2 x 58 held-out gold predictions before any "
                "4,349-study production generation."
            ),
        }

    if rebuild_labels:
        return {
            "Verdict": "GATE_ONE_TARGET_REBUILD",
            "NextExperiment": "Gold-only fold-safe challenge mapper",
            "Labels": rebuild_labels,
            "ProductionGenerationAuthorized": False,
            "ImageTrainingAuthorized": False,
            "Reason": (
                "Probability quality appears weak. "
                "Run a gold-only mapper gate before production."
            ),
        }

    if remask_labels:
        return {
            "Verdict": "TEST_ONE_POLICY_REMASK",
            "NextExperiment": "Gold-only policy remask simulation",
            "Labels": remask_labels,
            "ProductionGenerationAuthorized": False,
            "ImageTrainingAuthorized": False,
            "Reason": (
                "Teacher ranking is usable but specific report states "
                "appear unsafe for broad supervision."
            ),
        }

    if reweight_labels:
        return {
            "Verdict": "TEST_ONE_POLICY_REWEIGHT",
            "NextExperiment": "Gold-only supervision-weight simulation",
            "Labels": reweight_labels,
            "ProductionGenerationAuthorized": False,
            "ImageTrainingAuthorized": False,
            "Reason": (
                "Teacher probabilities appear useful, but authority is "
                "stronger than gold evidence supports."
            ),
        }

    return {
        "Verdict": "STOP_TEACHER_BRANCH",
        "NextExperiment": None,
        "Labels": [],
        "ProductionGenerationAuthorized": False,
        "ImageTrainingAuthorized": False,
        "Reason": (
            "No targeted teacher-policy defect strong enough "
            "to justify another experiment."
        ),
    }


# =============================================================================
# MANUAL REVIEW CASES
# =============================================================================


def build_manual_review_cases(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for label in TARGET_LABELS:
        x = errors[errors["Label"] == label].copy()

        # Strongest errors first.
        x = x.sort_values(
            "W26ErrorScore",
            ascending=False,
        ).head(12)

        rows.append(x)

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    preferred = [
        UID,
        "Label",
        "Gold",
        "ReportState",
        "ReportGoldCategory",
        "W23Probability",
        "W26Probability",
        "W23ErrorScore",
        "W26ErrorScore",
        "ErrorImprovement",
        "W26HighConfidenceError",
        REPORT,
    ]

    existing = [column for column in preferred if column in result.columns]

    return result[existing]


# =============================================================================
# STATUS
# =============================================================================


def status(
    accelerator: str,
) -> dict:
    paths = Paths.discover()

    files = validate_w37_outputs(paths)

    summary = validate_w37_benchmark(files["summary"])

    result = {
        "script_version": SCRIPT_VERSION,
        "display_version": DISPLAY_VERSION,
        "accelerator_requested": accelerator,
        "effective_backend": "cpu",
        "accelerator_env_variables_required": False,
        "paths": asdict(paths),
        "targets": TARGET_LABELS,
        "w37_validation": {
            "w23_macro_auroc": summary["gold_validation"]["w23_macro_auroc"],
            "w26_macro_auroc": summary["gold_validation"]["w26_fixed50_macro_auroc"],
        },
        "scope": {
            "gpu": False,
            "tpu": False,
            "dicom": False,
            "curia": False,
            "training": False,
            "production_generation": False,
        },
    }

    log(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


# =============================================================================
# AUDIT
# =============================================================================


def audit(
    accelerator: str,
) -> dict:
    paths = Paths.discover()

    output = Path(paths.output_root)

    files = validate_w37_outputs(paths)

    w37_summary = validate_w37_benchmark(files["summary"])

    log("=" * 100)
    log(DISPLAY_VERSION)
    log("=" * 100)

    log("")
    log("[1/6] Loading validated W37 outputs...")

    metrics = load_target_metrics(files["metrics"])

    errors = load_target_errors(files["errors"])

    policy = load_target_policy(files["policy"])

    priority = load_priority(files["priority"])

    log("W37 benchmark guard: PASS")

    log(
        f"  W2.3 macro AUROC = "
        f"{w37_summary['gold_validation']['w23_macro_auroc']:.6f}"
    )

    log(
        f"  W2.6 macro AUROC = "
        f"{w37_summary['gold_validation']['w26_fixed50_macro_auroc']:.6f}"
    )

    # -----------------------------------------------------------------
    # GOLD STATE BREAKDOWN
    # -----------------------------------------------------------------

    log("")
    log("[2/6] Building gold P/A/U/N state diagnostics...")

    gold_state = build_gold_state_breakdown(errors)

    gold_state.to_csv(
        output / "02_gold_state_breakdown.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # FULL PRODUCTION STATE POLICY
    # -----------------------------------------------------------------

    log("")
    log("[3/6] Linking full-report states to production policy...")

    full_states, full_state_meta = load_full_report_states(paths)

    production, production_meta = load_raw_production(paths)

    production_state_policy = build_production_state_policy(
        production,
        full_states,
    )

    if production_state_policy is not None:
        production_state_policy.to_csv(
            output / "03_production_state_policy.csv",
            index=False,
        )

        log("Full production state-policy join: PASS")

        log(f"  state artifact = " f"{full_state_meta.get('source')}")

    else:
        log("Full production state-policy join unavailable.")

        log(f"  state reason = " f"{full_state_meta.get('reason')}")

        log(f"  production reason = " f"{production_meta.get('reason')}")

    # -----------------------------------------------------------------
    # TARGET SUMMARY
    # -----------------------------------------------------------------

    log("")
    log("[4/6] Building targeted label evidence table...")

    target_summary = build_target_summary(
        metrics,
        errors,
        gold_state,
        policy,
    )

    target_summary.to_csv(
        output / "01_target_label_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # HIGH-CONFIDENCE / MANUAL CASES
    # -----------------------------------------------------------------

    high_errors = (
        errors[errors["W26HighConfidenceError"]]
        .sort_values(
            [
                "Label",
                "W26ErrorScore",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    high_errors.to_csv(
        output / "04_high_confidence_errors.csv",
        index=False,
    )

    manual_review = build_manual_review_cases(errors)

    manual_review.to_csv(
        output / "06_manual_review_cases.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------------------

    log("")
    log("[5/6] Determining KEEP / REWEIGHT / REMASK / REBUILD...")

    recommendations = build_recommendations(
        target_summary,
        production_state_policy,
    )

    recommendations.to_csv(
        output / "05_recommendations.csv",
        index=False,
    )

    next_step = determine_next_step(recommendations)

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    result_rows = []

    for _, row in recommendations.iterrows():
        result_rows.append(
            {
                "Label": row["Label"],
                "Recommendation": row["Recommendation"],
                "Confidence": row["Confidence"],
                "W26_AUROC": float(row["W26_AUROC"]),
                "Delta_AUROC": float(row["Delta_AUROC"]),
                "ProductionCoverage": float(row["ProductionCoverage"]),
                "ProductionMeanWeight": float(row["ProductionMeanWeight"]),
                "HighConfidenceErrors": int(row["HighConfidenceErrors"]),
                "Reasons": row["Reasons"],
            }
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "AUDIT_COMPLETE",
        "effective_backend": "cpu",
        "targets": TARGET_LABELS,
        "w37_reference": {
            "w23_macro_auroc": w37_summary["gold_validation"]["w23_macro_auroc"],
            "w26_macro_auroc": w37_summary["gold_validation"][
                "w26_fixed50_macro_auroc"
            ],
        },
        "full_report_state": full_state_meta,
        "raw_production": production_meta,
        "recommendations": result_rows,
        "next_step": next_step,
        "guardrails": {
            "image_training_authorized": False,
            "production_teacher_generation_authorized": False,
            "kaggle_required": False,
            "gpu_required": False,
            "tpu_required": False,
            "important": (
                "A REBUILD recommendation authorizes only a small "
                "fold-safe GOLD gate, not 4,349-study production."
            ),
        },
    }

    safe_json_dump(
        summary,
        output / "00_summary.json",
    )

    # -----------------------------------------------------------------
    # CONSOLE SUMMARY
    # -----------------------------------------------------------------

    log("")
    log("[6/6] Complete.")

    log("")
    log("=" * 100)
    log("TARGETED TEACHER DECISION")
    log("=" * 100)

    for _, row in recommendations.iterrows():
        log(
            f"{row['Label']:<12} "
            f"{row['Recommendation']:<9} "
            f"confidence={row['Confidence']:<6} "
            f"AUC={row['W26_AUROC']:.4f} "
            f"coverage={row['ProductionCoverage']:.1%} "
            f"weight={row['ProductionMeanWeight']:.3f}"
        )

    log("")
    log(f"NEXT VERDICT : " f"{next_step['Verdict']}")

    log(f"NEXT ACTION  : " f"{next_step['NextExperiment']}")

    log(f"LABELS       : " f"{next_step['Labels']}")

    log("")
    log("Image training authorized      : NO")

    log("Production generation authorized: NO")

    log("")
    log(f"Results: {output}")

    log("")
    log("STOP after this audit and review the verdict.")

    return summary


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("RSNA W38 targeted teacher policy audit v1")
    )

    parser.add_argument(
        "mode",
        choices=[
            "status",
            "audit",
        ],
    )

    parser.add_argument(
        "--accelerator",
        default="auto",
        help=("Accepted only for interface consistency. " "W38 always runs on CPU."),
    )

    args = parser.parse_args()

    if args.mode == "status":
        status(args.accelerator)

    elif args.mode == "audit":
        audit(args.accelerator)


if __name__ == "__main__":
    main()
