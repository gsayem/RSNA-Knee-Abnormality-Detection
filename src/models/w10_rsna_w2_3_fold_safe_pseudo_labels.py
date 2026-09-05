#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA W2.3 — Fold-Safe Weak Label Generation
============================================

Purpose
-------
Create fold-specific report-derived soft labels for the 4,349 unlabeled
RSNA knee MRI studies without allowing the held-out gold labels of an image
OOF fold to influence that fold's Stage-B report->challenge calibration.

This script DOES NOT:
- run mDeBERTa / NLI again;
- change W1.2/W2 Stage-A extraction;
- train the MRI image model;
- use the all-58 Stage-B calibrators from W2 full mode;
- use W1.3 benchmark PPV/NPV as pseudo-label weights.

Inputs
------
1) competition train.csv
2) W2 `04_gold_structured_report_features.csv`
3) W2 `08_full_structured_report_labels.csv`

Only Stage-A columns/features are consumed from the W2 files. Existing W2
Stage-B probability/gate columns are deliberately ignored.

Fold-safety protocol
--------------------
The exact V1.1/V4 5-fold image split is reproduced:
- gold studies sorted by StudyInstanceUID;
- greedy multilabel fold assignment;
- 5 folds;
- seed 42.

For each outer image fold:
1) hold out that fold's gold studies completely;
2) on the remaining ~45-47 gold studies, perform repeated stratified inner
   OOF Stage-B calibration separately for each of the 12 labels;
3) decide the per-label pseudo-label eligibility gate ONLY from that
   outer-training cohort's inner OOF metrics;
4) fit the final per-label Stage-B mapper on all outer-training gold studies;
5) predict all 4,349 unlabeled reports;
6) write fold-specific soft labels/masks for image training;
7) predict the held-out gold fold only for leakage-free Stage-B diagnostics.

Important methodological scope
------------------------------
This makes Stage-B challenge-ontology calibration fold-safe with respect to
the image OOF folds. It does NOT turn the whole W1/W1.2/W1.3/W2 development
history into a fully nested external validation: Stage-A rules and design were
investigated using the 58-study research cohort. The script states this
explicitly in its report.

Default Kaggle paths can be overridden with environment variables:
- W23_TRAIN_CSV
- W23_W2_RESULTS_DIR
- W23_GOLD_STRUCTURED_CSV
- W23_FULL_STRUCTURED_CSV
- W23_OUTPUT_DIR
- W23_V4_OOF_CSV (optional independent split verification)

Typical Kaggle use
------------------
    !python rsna_w2_3_fold_safe_pseudo_labels.py

Outputs are written to:
    /kaggle/working/rsna_w2_3
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

UID_COLUMN = "StudyInstanceUID"

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

STAGE_B_FEATURES = [
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

# Small audit fields retained in fold-specific long outputs.
STAGE_A_AUDIT_COLUMNS = [
    UID_COLUMN,
    "Label",
    "Language",
    "RuleAssertion",
    "SemanticAssertion",
    "FusedAssertion",
    "FusedAssertionConfidence",
    "FusionSource",
    "EvidenceAvailable",
]

EXPECTED_TOTAL_STUDIES = 4407
EXPECTED_GOLD_STUDIES = 58
EXPECTED_UNLABELED_STUDIES = 4349
EXPECTED_LABELS = 12

# Exact V1.1/V4 outer split.
NUM_OUTER_FOLDS = 5
OUTER_RANDOM_SEED = 42

# SHA256 of sorted "StudyInstanceUID,Fold\n" rows from the controlled
# V4 split. This gives us a hard audit that W2.3 reproduced the same split.
EXPECTED_V4_FOLD_SHA256 = (
    "1d9959b027c055974325f4de59e26974" "b036ae8b2c1b63aa417d3eef7aaf9f4a"
)

# W2.2 Stage-B mapper settings.
CALIBRATION_C = 0.10
INNER_CV_SPLITS = 5
INNER_CV_REPEATS = 5
INNER_CV_RANDOM_STATE = 42042

# W2.2 conservative eligibility gate, now evaluated independently inside
# each outer training fold.
PSEUDO_GATE_MIN_EVIDENCE_N = 15
PSEUDO_GATE_MIN_EVIDENCE_AUROC = 0.55
PSEUDO_GATE_MIN_BRIER_IMPROVEMENT = 0.0

# Diagnostic/triage threshold only. This is not a loss weight and is not
# required for SoftLabelAvailable=True.
SELECTION_SCORE_THRESHOLD = float(
    os.environ.get("W23_SELECTION_SCORE_THRESHOLD", "0.5")
)

DATA_ROOT = Path(
    os.environ.get(
        "W23_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)

TRAIN_CSV = Path(
    os.environ.get(
        "W23_TRAIN_CSV",
        str(DATA_ROOT / "train.csv"),
    )
)

W2_RESULTS_DIR = Path(
    os.environ.get(
        "W23_W2_RESULTS_DIR",
        "/kaggle/working/rsna_w2/results",
    )
)

GOLD_STRUCTURED_CSV = Path(
    os.environ.get(
        "W23_GOLD_STRUCTURED_CSV",
        str(W2_RESULTS_DIR / "04_gold_structured_report_features.csv"),
    )
)

FULL_STRUCTURED_CSV = Path(
    os.environ.get(
        "W23_FULL_STRUCTURED_CSV",
        str(W2_RESULTS_DIR / "08_full_structured_report_labels.csv"),
    )
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "W23_OUTPUT_DIR",
        "/kaggle/working/rsna_w2_3",
    )
)

RESULT_ROOT = OUTPUT_ROOT / "results"
MODEL_ROOT = OUTPUT_ROOT / "models"
FOLD_ROOT = OUTPUT_ROOT / "folds"

OPTIONAL_V4_OOF_CSV = os.environ.get("W23_V4_OOF_CSV", "").strip()

# Default behavior is strict: the generated split must have the known V4
# checksum. Only use this escape hatch for deliberate experimental work.
ALLOW_FOLD_MISMATCH = os.environ.get("W23_ALLOW_FOLD_MISMATCH", "0").strip() == "1"


# ============================================================
# 2. GENERIC UTILITIES
# ============================================================


def safe_json_value(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return [safe_json_value(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}

    try:
        if np.isscalar(value) or (hasattr(value, "ndim") and value.ndim == 0):
            if pd.isna(value):
                return None
    except (ValueError, TypeError):
        pass

    return value


def ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, RESULT_ROOT, MODEL_ROOT, FOLD_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Set the corresponding W23_* environment variable if the "
            "file is stored elsewhere."
        )


def require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    source_name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{source_name} is missing required columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    """Robustly parse bool-like CSV values."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()

    result = pd.Series(False, index=series.index, dtype=bool)
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype(float) != 0.0

    text_mask = ~numeric_mask & series.notna()
    if text_mask.any():
        text = series.loc[text_mask].astype(str).str.strip().str.lower()
        true_values = {"true", "t", "yes", "y", "1"}
        false_values = {"false", "f", "no", "n", "0", "", "nan", "none"}
        unknown = sorted(set(text.unique()) - true_values - false_values)
        if unknown:
            raise RuntimeError("Unexpected boolean values: " + ", ".join(unknown[:20]))
        result.loc[text_mask] = text.isin(true_values).values

    return result


def nanmean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())


def format_float(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


# ============================================================
# 3. EXACT V1.1/V4 OUTER-FOLD ASSIGNMENT
# ============================================================


def greedy_multilabel_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """
    Exact controlled V1.1/V4 greedy multilabel fold algorithm.

    Do not change this function without intentionally changing the outer
    image OOF protocol.
    """

    rng = np.random.default_rng(seed)

    n_samples, n_labels = y.shape
    fold_assignments = -np.ones(n_samples, dtype=int)

    label_frequency = y.sum(axis=0) + 1e-8
    sample_rarity = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        positive_labels = np.where(y[i] > 0)[0]

        if len(positive_labels) == 0:
            sample_rarity[i] = 0.0
        else:
            sample_rarity[i] = float(np.sum(1.0 / label_frequency[positive_labels]))

    tie_noise = rng.random(n_samples) * 1e-6
    order = np.argsort(-sample_rarity - tie_noise)

    fold_label_counts = np.zeros(
        (n_splits, n_labels),
        dtype=np.float64,
    )
    fold_sizes = np.zeros(n_splits, dtype=int)
    desired_fold_label_counts = label_frequency / n_splits

    for sample_idx in order:
        sample = y[sample_idx]
        positive_labels = np.where(sample > 0)[0]
        scores: List[float] = []

        for fold in range(n_splits):
            label_score = 0.0

            if len(positive_labels) > 0:
                ratios = fold_label_counts[fold, positive_labels] / (
                    desired_fold_label_counts[positive_labels] + 1e-8
                )
                label_score = float(np.mean(ratios))

            size_score = fold_sizes[fold] / max(
                1,
                math.ceil(n_samples / n_splits),
            )

            scores.append(label_score + 0.05 * size_score)

        best_fold = int(np.argmin(scores))
        fold_assignments[sample_idx] = best_fold
        fold_sizes[best_fold] += 1
        fold_label_counts[best_fold] += sample

    return fold_assignments


def fold_assignment_sha256(assignments: pd.DataFrame) -> str:
    ordered = assignments.sort_values(UID_COLUMN).reset_index(drop=True)
    payload = "".join(
        f"{uid},{int(fold)}\n"
        for uid, fold in zip(
            ordered[UID_COLUMN].astype(str),
            ordered["OuterFold"].astype(int),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optional_v4_oof_path() -> Optional[Path]:
    candidates: List[Path] = []

    if OPTIONAL_V4_OOF_CSV:
        candidates.append(Path(OPTIONAL_V4_OOF_CSV))

    candidates.extend(
        [
            Path("/kaggle/working/rsna_v4/oof_predictions.csv"),
            Path("/kaggle/working/v4/oof_predictions.csv"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def verify_v4_oof_if_available(
    assignments: pd.DataFrame,
) -> Tuple[bool, str]:
    path = optional_v4_oof_path()

    if path is None:
        return False, "No V4 OOF file found; checksum verification used."

    v4 = pd.read_csv(path)
    require_columns(v4, [UID_COLUMN, "Fold"], str(path))

    v4[UID_COLUMN] = v4[UID_COLUMN].astype(str)
    v4 = v4[[UID_COLUMN, "Fold"]].drop_duplicates()

    merged = assignments.merge(
        v4,
        on=UID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if merged["Fold"].isna().any():
        missing = merged.loc[merged["Fold"].isna(), UID_COLUMN].tolist()
        raise RuntimeError(f"V4 OOF verification missing {len(missing)} gold UIDs.")

    matches = (
        merged["OuterFold"].astype(int).values == merged["Fold"].astype(int).values
    )

    if not bool(matches.all()):
        bad = merged.loc[~matches, [UID_COLUMN, "OuterFold", "Fold"]]
        raise RuntimeError(
            "Generated W2.3 folds do not match the supplied V4 OOF file.\n"
            + bad.to_string(index=False)
        )

    return True, f"Matched V4 OOF assignments exactly: {path}"


# ============================================================
# 4. INPUT LOADING / INTEGRITY
# ============================================================


def load_inputs() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, str],
]:
    """
    Returns:
      train_df,
      gold_df (58, sorted exactly like V4),
      gold_structured (696 with trusted Gold reattached from train.csv),
      unlabeled_structured (4349*12, Stage-A fields only),
      source_hashes.
    """

    require_file(TRAIN_CSV, "competition train.csv")
    require_file(GOLD_STRUCTURED_CSV, "W2 gold structured features")
    require_file(FULL_STRUCTURED_CSV, "W2 full structured features")

    train_df = pd.read_csv(TRAIN_CSV)
    require_columns(
        train_df,
        [UID_COLUMN, "Report", *LABEL_COLUMNS],
        str(TRAIN_CSV),
    )
    train_df[UID_COLUMN] = train_df[UID_COLUMN].astype(str)

    if len(train_df) != EXPECTED_TOTAL_STUDIES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_STUDIES} studies, found {len(train_df)}."
        )

    label_non_null_count = train_df[LABEL_COLUMNS].notna().sum(axis=1)

    partial_mask = (label_non_null_count > 0) & (label_non_null_count < EXPECTED_LABELS)
    if partial_mask.any():
        raise RuntimeError(
            f"Found {int(partial_mask.sum())} partially labeled studies; "
            "the controlled dataset is expected to have none."
        )

    gold_mask = label_non_null_count == EXPECTED_LABELS
    unlabeled_mask = label_non_null_count == 0

    gold_df = (
        train_df.loc[gold_mask].copy().sort_values(UID_COLUMN).reset_index(drop=True)
    )
    unlabeled_df = train_df.loc[unlabeled_mask].copy()

    if len(gold_df) != EXPECTED_GOLD_STUDIES:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_STUDIES} gold studies, " f"found {len(gold_df)}."
        )

    if len(unlabeled_df) != EXPECTED_UNLABELED_STUDIES:
        raise RuntimeError(
            f"Expected {EXPECTED_UNLABELED_STUDIES} unlabeled studies, "
            f"found {len(unlabeled_df)}."
        )

    # ------------------------------
    # Gold Stage-A feature table
    # ------------------------------
    gold_structured = pd.read_csv(GOLD_STRUCTURED_CSV)
    require_columns(
        gold_structured,
        [
            UID_COLUMN,
            "Label",
            "EvidenceAvailable",
            *STAGE_B_FEATURES,
        ],
        str(GOLD_STRUCTURED_CSV),
    )
    gold_structured[UID_COLUMN] = gold_structured[UID_COLUMN].astype(str)

    if len(gold_structured) != EXPECTED_GOLD_STUDIES * EXPECTED_LABELS:
        raise RuntimeError(
            "W2 gold structured feature table must contain exactly "
            f"{EXPECTED_GOLD_STUDIES * EXPECTED_LABELS} rows; "
            f"found {len(gold_structured)}."
        )

    if gold_structured[[UID_COLUMN, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate UID/Label rows in W2 gold structured file.")

    expected_gold_uids = set(gold_df[UID_COLUMN])
    actual_gold_uids = set(gold_structured[UID_COLUMN])
    if actual_gold_uids != expected_gold_uids:
        raise RuntimeError(
            "W2 gold structured UIDs do not exactly match train.csv gold UIDs."
        )

    per_label_gold_count = gold_structured["Label"].value_counts()
    for label in LABEL_COLUMNS:
        if int(per_label_gold_count.get(label, 0)) != EXPECTED_GOLD_STUDIES:
            raise RuntimeError(
                f"W2 gold structured table has wrong count for {label}: "
                f"{int(per_label_gold_count.get(label, 0))}."
            )

    # Reattach challenge gold directly from train.csv. Never trust a stale
    # intermediate Gold column for W2.3.
    trusted_gold_long = gold_df[[UID_COLUMN, *LABEL_COLUMNS]].melt(
        id_vars=[UID_COLUMN],
        value_vars=LABEL_COLUMNS,
        var_name="Label",
        value_name="TrustedGold",
    )

    if "Gold" in gold_structured.columns:
        old_gold = pd.to_numeric(gold_structured["Gold"], errors="coerce")
        gold_structured = gold_structured.drop(columns=["Gold"])
    else:
        old_gold = None

    gold_structured = gold_structured.merge(
        trusted_gold_long,
        on=[UID_COLUMN, "Label"],
        how="left",
        validate="one_to_one",
    )
    gold_structured = gold_structured.rename(columns={"TrustedGold": "Gold"})
    gold_structured["Gold"] = gold_structured["Gold"].astype(int)

    if old_gold is not None and old_gold.notna().any():
        # Align by original order using the known UID/Label key.
        old_check = pd.read_csv(
            GOLD_STRUCTURED_CSV, usecols=[UID_COLUMN, "Label", "Gold"]
        )
        old_check[UID_COLUMN] = old_check[UID_COLUMN].astype(str)
        old_check["Gold"] = pd.to_numeric(old_check["Gold"], errors="coerce")
        old_check = old_check.merge(
            trusted_gold_long,
            on=[UID_COLUMN, "Label"],
            how="left",
            validate="one_to_one",
        )
        comparable = old_check["Gold"].notna()
        mismatch = (
            old_check.loc[comparable, "Gold"].astype(int).values
            != old_check.loc[comparable, "TrustedGold"].astype(int).values
        )
        if mismatch.any():
            raise RuntimeError(
                "Gold values embedded in W2 structured features conflict "
                "with train.csv."
            )

    gold_structured["EvidenceAvailable"] = bool_series(
        gold_structured["EvidenceAvailable"]
    )

    # ------------------------------
    # Full Stage-A table -> unlabeled only
    # ------------------------------
    full_structured = pd.read_csv(FULL_STRUCTURED_CSV)
    require_columns(
        full_structured,
        [
            UID_COLUMN,
            "Label",
            "EvidenceAvailable",
            *STAGE_B_FEATURES,
        ],
        str(FULL_STRUCTURED_CSV),
    )
    full_structured[UID_COLUMN] = full_structured[UID_COLUMN].astype(str)

    expected_full_rows = EXPECTED_TOTAL_STUDIES * EXPECTED_LABELS
    if len(full_structured) != expected_full_rows:
        raise RuntimeError(
            f"Expected {expected_full_rows} full structured rows, "
            f"found {len(full_structured)}."
        )

    if full_structured[[UID_COLUMN, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate UID/Label rows in W2 full structured file.")

    if full_structured[UID_COLUMN].nunique() != EXPECTED_TOTAL_STUDIES:
        raise RuntimeError(
            "W2 full structured table does not contain exactly 4,407 studies."
        )

    for label in LABEL_COLUMNS:
        n = int((full_structured["Label"] == label).sum())
        if n != EXPECTED_TOTAL_STUDIES:
            raise RuntimeError(
                f"W2 full structured table has wrong count for {label}: {n}."
            )

    unlabeled_uid_set = set(unlabeled_df[UID_COLUMN])
    unlabeled_structured = full_structured[
        full_structured[UID_COLUMN].isin(unlabeled_uid_set)
    ].copy()

    expected_unlabeled_rows = EXPECTED_UNLABELED_STUDIES * EXPECTED_LABELS
    if len(unlabeled_structured) != expected_unlabeled_rows:
        raise RuntimeError(
            f"Expected {expected_unlabeled_rows} unlabeled structured rows, "
            f"found {len(unlabeled_structured)}."
        )

    unlabeled_structured["EvidenceAvailable"] = bool_series(
        unlabeled_structured["EvidenceAvailable"]
    )

    # Deliberately select only Stage-A information. This is the core guard
    # against accidentally reusing all-58 Stage-B results from W2 full mode.
    keep_columns: List[str] = []
    for column in [*STAGE_A_AUDIT_COLUMNS, *STAGE_B_FEATURES]:
        if column in unlabeled_structured.columns and column not in keep_columns:
            keep_columns.append(column)

    unlabeled_structured = unlabeled_structured[keep_columns].copy()

    # Numeric coercion mirrors W2's fillna(0.0) behavior at matrix creation.
    for column in STAGE_B_FEATURES:
        gold_structured[column] = pd.to_numeric(
            gold_structured[column], errors="coerce"
        )
        unlabeled_structured[column] = pd.to_numeric(
            unlabeled_structured[column], errors="coerce"
        )

    source_hashes = {
        "train_csv_sha256": sha256_file(TRAIN_CSV),
        "gold_structured_sha256": sha256_file(GOLD_STRUCTURED_CSV),
        "full_structured_sha256": sha256_file(FULL_STRUCTURED_CSV),
    }

    return (
        train_df,
        gold_df,
        gold_structured,
        unlabeled_structured,
        source_hashes,
    )


# ============================================================
# 5. STAGE-B MODEL / INNER OOF GATE
# ============================================================


def import_sklearn():
    try:
        from joblib import dump
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("W2.3 requires scikit-learn and joblib.") from exc

    return {
        "dump": dump,
        "LogisticRegression": LogisticRegression,
        "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
        "roc_auc_score": roc_auc_score,
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


SK = import_sklearn()


def make_stage_b_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[STAGE_B_FEATURES].fillna(0.0).astype(np.float32).values


def build_pipeline():
    return SK["Pipeline"](
        [
            ("scale", SK["StandardScaler"]()),
            (
                "logreg",
                SK["LogisticRegression"](
                    C=CALIBRATION_C,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    class_weight=None,
                    random_state=INNER_CV_RANDOM_STATE,
                ),
            ),
        ]
    )


def predict_calibrator(calibrator: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    kind = calibrator["kind"]

    if kind == "pipeline":
        return calibrator["model"].predict_proba(X)[:, 1].astype(np.float64)

    if kind == "constant":
        return np.full(
            len(X),
            float(calibrator["probability"]),
            dtype=np.float64,
        )

    raise RuntimeError(f"Unknown calibrator kind: {kind}")


def gate_reason_from_metrics(metrics: Dict[str, Any]) -> str:
    reasons: List[str] = []

    if int(metrics["EvidenceAvailableN"]) < PSEUDO_GATE_MIN_EVIDENCE_N:
        reasons.append("insufficient_evidence_n")

    evidence_auc = metrics["EvidenceOnly_AUROC"]
    if (
        evidence_auc is None
        or not np.isfinite(float(evidence_auc))
        or float(evidence_auc) < PSEUDO_GATE_MIN_EVIDENCE_AUROC
    ):
        reasons.append("evidence_auc_below_gate")

    brier_improvement = metrics["EvidenceOnly_BrierImprovementVsPrior"]
    if (
        brier_improvement is None
        or not np.isfinite(float(brier_improvement))
        or float(brier_improvement) <= PSEUDO_GATE_MIN_BRIER_IMPROVEMENT
    ):
        reasons.append("evidence_brier_not_better_than_prior")

    return "pass" if not reasons else "|".join(reasons)


def fit_inner_oof_and_final(
    label_train_df: pd.DataFrame,
    outer_fold: int,
    label: str,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    pd.DataFrame,
    List[Dict[str, Any]],
]:
    """
    Fit one label using only the current outer-training gold studies.

    Returns:
      calibrator,
      inner metrics/gate,
      inner OOF rows,
      coefficient rows.
    """

    y = label_train_df["Gold"].astype(int).values
    X = make_stage_b_matrix(label_train_df)
    evidence_mask = label_train_df["EvidenceAvailable"].astype(bool).values

    class_counts = np.bincount(y, minlength=2)
    positive_n = int(class_counts[1])
    negative_n = int(class_counts[0])

    inner_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []

    # Robust fallback; not expected on this dataset but prevents accidental
    # fitting crashes if a future outer split loses one class.
    if positive_n == 0 or negative_n == 0:
        constant_probability = float(y.mean())
        calibrator = {
            "kind": "constant",
            "probability": constant_probability,
        }

        for _, source_row in label_train_df.iterrows():
            inner_rows.append(
                {
                    "OuterFold": outer_fold,
                    UID_COLUMN: str(source_row[UID_COLUMN]),
                    "Label": label,
                    "Gold": int(source_row["Gold"]),
                    "EvidenceAvailable": bool(source_row["EvidenceAvailable"]),
                    "InnerOOFProbability": constant_probability,
                    "InnerOOFRepeatedPredictions": 0,
                }
            )

        metrics: Dict[str, Any] = {
            "OuterFold": outer_fold,
            "Label": label,
            "OuterTrainN": int(len(y)),
            "GoldPositive": positive_n,
            "GoldNegative": negative_n,
            "InnerCVSplits": 0,
            "InnerCVRepeats": 0,
            "InnerOOF_AUROC": np.nan,
            "InnerOOF_AP": np.nan,
            "InnerOOF_Brier": np.nan,
            "InnerPrior_Brier": np.nan,
            "InnerBrierImprovementVsPrior": np.nan,
            "InnerOOF_LogLoss": np.nan,
            "EvidenceAvailableN": int(evidence_mask.sum()),
            "EvidenceCoverage": float(evidence_mask.mean()),
            "EvidenceOnly_AUROC": np.nan,
            "EvidenceOnly_AP": np.nan,
            "EvidenceOnly_Brier": np.nan,
            "EvidenceOnly_Prior_Brier": np.nan,
            "EvidenceOnly_BrierImprovementVsPrior": np.nan,
            "OuterTrainPriorProbability": constant_probability,
            "OuterTrainEvidencePriorProbability": (
                float(y[evidence_mask].mean()) if evidence_mask.any() else np.nan
            ),
            "PseudoLabelGateReason": "single_class_outer_train",
            "PseudoLabelGatePass": False,
        }
        return calibrator, metrics, pd.DataFrame(inner_rows), coefficient_rows

    min_class_n = int(class_counts.min())
    n_splits = min(INNER_CV_SPLITS, min_class_n)

    if n_splits < 2:
        raise RuntimeError(
            f"Outer fold {outer_fold} {label}: not enough examples for inner CV."
        )

    cv = SK["RepeatedStratifiedKFold"](
        n_splits=n_splits,
        n_repeats=INNER_CV_REPEATS,
        random_state=INNER_CV_RANDOM_STATE,
    )

    probability_sum = np.zeros(len(y), dtype=np.float64)
    probability_count = np.zeros(len(y), dtype=np.int32)

    for train_index, val_index in cv.split(X, y):
        pipeline = build_pipeline()
        pipeline.fit(X[train_index], y[train_index])
        probability = pipeline.predict_proba(X[val_index])[:, 1]
        probability_sum[val_index] += probability
        probability_count[val_index] += 1

    if (probability_count == 0).any():
        raise RuntimeError(
            f"Outer fold {outer_fold} {label}: incomplete inner OOF predictions."
        )

    inner_oof_probability = probability_sum / probability_count
    prevalence = float(y.mean())
    prior_probability = np.full(len(y), prevalence, dtype=np.float64)

    auc = float(SK["roc_auc_score"](y, inner_oof_probability))
    ap = float(SK["average_precision_score"](y, inner_oof_probability))
    brier = float(SK["brier_score_loss"](y, inner_oof_probability))
    prior_brier = float(SK["brier_score_loss"](y, prior_probability))
    ll = float(
        SK["log_loss"](
            y,
            inner_oof_probability,
            labels=[0, 1],
        )
    )

    evidence_n = int(evidence_mask.sum())
    evidence_auc = np.nan
    evidence_ap = np.nan
    evidence_brier = np.nan
    evidence_prior_brier = np.nan
    evidence_brier_improvement = np.nan
    evidence_prior_probability_value = np.nan

    if evidence_n > 0:
        evidence_y = y[evidence_mask]
        evidence_probability = inner_oof_probability[evidence_mask]
        evidence_prevalence = float(evidence_y.mean())
        evidence_prior_probability_value = evidence_prevalence
        evidence_prior_probability = np.full(
            len(evidence_y),
            evidence_prevalence,
            dtype=np.float64,
        )

        evidence_brier = float(SK["brier_score_loss"](evidence_y, evidence_probability))
        evidence_prior_brier = float(
            SK["brier_score_loss"](evidence_y, evidence_prior_probability)
        )
        evidence_brier_improvement = float(evidence_prior_brier - evidence_brier)

        if evidence_n > 1 and len(np.unique(evidence_y)) == 2:
            evidence_auc = float(SK["roc_auc_score"](evidence_y, evidence_probability))
            evidence_ap = float(
                SK["average_precision_score"](
                    evidence_y,
                    evidence_probability,
                )
            )

    metrics = {
        "OuterFold": outer_fold,
        "Label": label,
        "OuterTrainN": int(len(y)),
        "GoldPositive": positive_n,
        "GoldNegative": negative_n,
        "InnerCVSplits": int(n_splits),
        "InnerCVRepeats": int(INNER_CV_REPEATS),
        "InnerOOF_AUROC": auc,
        "InnerOOF_AP": ap,
        "InnerOOF_Brier": brier,
        "InnerPrior_Brier": prior_brier,
        "InnerBrierImprovementVsPrior": float(prior_brier - brier),
        "InnerOOF_LogLoss": ll,
        "EvidenceAvailableN": evidence_n,
        "EvidenceCoverage": float(evidence_n / len(y)),
        "EvidenceOnly_AUROC": evidence_auc,
        "EvidenceOnly_AP": evidence_ap,
        "EvidenceOnly_Brier": evidence_brier,
        "EvidenceOnly_Prior_Brier": evidence_prior_brier,
        "EvidenceOnly_BrierImprovementVsPrior": evidence_brier_improvement,
        "OuterTrainPriorProbability": prevalence,
        "OuterTrainEvidencePriorProbability": evidence_prior_probability_value,
    }

    gate_reason = gate_reason_from_metrics(metrics)
    metrics["PseudoLabelGateReason"] = gate_reason
    metrics["PseudoLabelGatePass"] = gate_reason == "pass"

    for source_row, probability, count in zip(
        label_train_df.to_dict(orient="records"),
        inner_oof_probability,
        probability_count,
    ):
        inner_rows.append(
            {
                "OuterFold": outer_fold,
                UID_COLUMN: str(source_row[UID_COLUMN]),
                "Label": label,
                "Gold": int(source_row["Gold"]),
                "EvidenceAvailable": bool(source_row["EvidenceAvailable"]),
                "FusedAssertion": source_row.get("FusedAssertion", ""),
                "InnerOOFProbability": float(probability),
                "InnerOOFRepeatedPredictions": int(count),
            }
        )

    # Final fold-specific mapper: fit ONLY on this outer-training cohort.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    calibrator = {
        "kind": "pipeline",
        "model": final_pipeline,
    }

    coefficients = final_pipeline.named_steps["logreg"].coef_[0]
    for feature, coefficient in zip(STAGE_B_FEATURES, coefficients):
        coefficient_rows.append(
            {
                "OuterFold": outer_fold,
                "Label": label,
                "Feature": feature,
                "StandardizedCoefficient": float(coefficient),
            }
        )

    coefficient_rows.append(
        {
            "OuterFold": outer_fold,
            "Label": label,
            "Feature": "Intercept",
            "StandardizedCoefficient": float(
                final_pipeline.named_steps["logreg"].intercept_[0]
            ),
        }
    )

    return calibrator, metrics, pd.DataFrame(inner_rows), coefficient_rows


# ============================================================
# 6. FOLD-SPECIFIC SCORING
# ============================================================


def score_unlabeled_for_fold(
    unlabeled_structured: pd.DataFrame,
    calibrators: Dict[str, Dict[str, Any]],
    gate_metrics: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    output_parts: List[pd.DataFrame] = []

    gate_map = gate_metrics.set_index("Label")["PseudoLabelGatePass"].to_dict()
    reason_map = gate_metrics.set_index("Label")["PseudoLabelGateReason"].to_dict()

    for label in LABEL_COLUMNS:
        label_df = (
            unlabeled_structured[unlabeled_structured["Label"] == label]
            .copy()
            .reset_index(drop=True)
        )

        if len(label_df) != EXPECTED_UNLABELED_STUDIES:
            raise RuntimeError(
                f"Outer fold {outer_fold} {label}: expected "
                f"{EXPECTED_UNLABELED_STUDIES} unlabeled rows, "
                f"found {len(label_df)}."
            )

        X = make_stage_b_matrix(label_df)
        probability = predict_calibrator(calibrators[label], X)

        gate_pass = bool(gate_map[label])
        gate_reason = str(reason_map[label])

        label_df["OuterFold"] = outer_fold
        label_df["ChallengeProbabilityRaw"] = probability
        label_df["LabelCalibrationEligible"] = gate_pass
        label_df["LabelCalibrationGateReason"] = gate_reason

        label_df["SoftLabelAvailable"] = (
            label_df["EvidenceAvailable"].astype(bool) & gate_pass
        )

        label_df["ChallengeSoftLabel"] = np.where(
            label_df["SoftLabelAvailable"],
            label_df["ChallengeProbabilityRaw"],
            np.nan,
        )

        label_df["CandidateSelectionScore"] = np.where(
            label_df["SoftLabelAvailable"],
            label_df["FusedAssertionConfidence"].fillna(0.0).astype(float)
            * (2.0 * np.abs(label_df["ChallengeProbabilityRaw"].astype(float) - 0.5)),
            0.0,
        )

        label_df["InitialHighSelectionCandidate"] = label_df["SoftLabelAvailable"] & (
            label_df["CandidateSelectionScore"] >= SELECTION_SCORE_THRESHOLD
        )

        output_parts.append(label_df)

    output = pd.concat(output_parts, ignore_index=True)

    expected_rows = EXPECTED_UNLABELED_STUDIES * EXPECTED_LABELS
    if len(output) != expected_rows:
        raise RuntimeError(
            f"Outer fold {outer_fold}: expected {expected_rows} scored rows, "
            f"found {len(output)}."
        )

    return output


def score_outer_validation_gold(
    validation_gold: pd.DataFrame,
    calibrators: Dict[str, Dict[str, Any]],
    gate_metrics: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    gate_lookup = gate_metrics.set_index("Label").to_dict(orient="index")

    for label in LABEL_COLUMNS:
        label_df = (
            validation_gold[validation_gold["Label"] == label]
            .copy()
            .reset_index(drop=True)
        )

        X = make_stage_b_matrix(label_df)
        probability = predict_calibrator(calibrators[label], X)
        gate_info = gate_lookup[label]
        gate_pass = bool(gate_info["PseudoLabelGatePass"])

        for source_row, prob in zip(
            label_df.to_dict(orient="records"),
            probability,
        ):
            evidence_available = bool(source_row["EvidenceAvailable"])
            rows.append(
                {
                    "OuterFold": outer_fold,
                    UID_COLUMN: str(source_row[UID_COLUMN]),
                    "Label": label,
                    "Gold": int(source_row["Gold"]),
                    "EvidenceAvailable": evidence_available,
                    "FusedAssertion": source_row.get("FusedAssertion", ""),
                    "FusedAssertionConfidence": float(
                        source_row.get("FusedAssertionConfidence", 0.0) or 0.0
                    ),
                    "FusionSource": source_row.get("FusionSource", ""),
                    "FoldSafeChallengeProbability": float(prob),
                    "OuterTrainPriorProbability": float(
                        gate_info["OuterTrainPriorProbability"]
                    ),
                    "OuterTrainEvidencePriorProbability": (
                        float(gate_info["OuterTrainEvidencePriorProbability"])
                        if pd.notna(gate_info["OuterTrainEvidencePriorProbability"])
                        else np.nan
                    ),
                    "LabelCalibrationEligible": gate_pass,
                    "LabelCalibrationGateReason": str(
                        gate_info["PseudoLabelGateReason"]
                    ),
                    "EligibleEvidenceForPseudoSupervision": (
                        evidence_available & gate_pass
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 7. OUTPUT HELPERS
# ============================================================


def write_fold_wide_outputs(
    fold_scored: pd.DataFrame,
    fold_dir: Path,
) -> None:
    def pivot(value_column: str) -> pd.DataFrame:
        return (
            fold_scored.pivot(
                index=UID_COLUMN,
                columns="Label",
                values=value_column,
            )
            .reindex(columns=LABEL_COLUMNS)
            .reset_index()
        )

    pivot("ChallengeSoftLabel").to_csv(
        fold_dir / "soft_probabilities_wide.csv",
        index=False,
    )

    pivot("SoftLabelAvailable").to_csv(
        fold_dir / "soft_label_availability_wide.csv",
        index=False,
    )

    pivot("CandidateSelectionScore").to_csv(
        fold_dir / "candidate_selection_scores_wide.csv",
        index=False,
    )

    pivot("InitialHighSelectionCandidate").to_csv(
        fold_dir / "high_selection_candidate_mask_wide.csv",
        index=False,
    )

    # Raw probabilities are useful for diagnostics, including labels that
    # were held by the gate. They must not be mistaken for training labels.
    pivot("ChallengeProbabilityRaw").to_csv(
        fold_dir / "raw_probabilities_diagnostic_wide.csv",
        index=False,
    )


def summarize_fold_coverage(
    fold_scored: pd.DataFrame,
    outer_fold: int,
    gate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    gate_lookup = gate_metrics.set_index("Label").to_dict(orient="index")
    rows: List[Dict[str, Any]] = []

    for label in LABEL_COLUMNS:
        group = fold_scored[fold_scored["Label"] == label]
        available = group[group["SoftLabelAvailable"]]
        high = group[group["InitialHighSelectionCandidate"]]

        fusion_text = group["FusionSource"].fillna("").astype(str)
        rule_explicit = fusion_text.str.startswith("rule_")
        semantic_recovery = fusion_text.str.startswith("semantic_recovery")

        rows.append(
            {
                "OuterFold": outer_fold,
                "Label": label,
                "UnlabeledStudies": EXPECTED_UNLABELED_STUDIES,
                "PseudoLabelGatePass": bool(gate_lookup[label]["PseudoLabelGatePass"]),
                "PseudoLabelGateReason": str(
                    gate_lookup[label]["PseudoLabelGateReason"]
                ),
                "EvidenceAvailableN": int(group["EvidenceAvailable"].sum()),
                "SoftLabelAvailableN": int(len(available)),
                "SoftLabelCoverage": float(len(available) / EXPECTED_UNLABELED_STUDIES),
                "MeanProbabilityAvailable": (
                    float(available["ChallengeSoftLabel"].mean())
                    if len(available) > 0
                    else np.nan
                ),
                "MedianProbabilityAvailable": (
                    float(available["ChallengeSoftLabel"].median())
                    if len(available) > 0
                    else np.nan
                ),
                "StdProbabilityAvailable": (
                    float(available["ChallengeSoftLabel"].std(ddof=0))
                    if len(available) > 0
                    else np.nan
                ),
                f"SelectionScoreN_ge_{SELECTION_SCORE_THRESHOLD:g}": int(len(high)),
                "RuleStructuredN": int(rule_explicit.sum()),
                "SemanticRecoveredN": int(semantic_recovery.sum()),
            }
        )

    return pd.DataFrame(rows)


def summarize_study_coverage(
    fold_scored: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    summary = fold_scored.groupby(UID_COLUMN, as_index=False).agg(
        AvailableLabelCount=("SoftLabelAvailable", "sum"),
        HighSelectionLabelCount=("InitialHighSelectionCandidate", "sum"),
        EvidenceLabelCount=("EvidenceAvailable", "sum"),
    )
    summary.insert(0, "OuterFold", outer_fold)
    summary["HasAnySoftLabel"] = summary["AvailableLabelCount"] > 0
    summary["HasAnyHighSelectionLabel"] = summary["HighSelectionLabelCount"] > 0
    return summary


def compute_outer_oof_metrics(
    outer_oof: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for label in LABEL_COLUMNS:
        df = outer_oof[outer_oof["Label"] == label].copy()
        y = df["Gold"].astype(int).values
        p = df["FoldSafeChallengeProbability"].astype(float).values
        prior = df["OuterTrainPriorProbability"].astype(float).values

        auc = float(SK["roc_auc_score"](y, p)) if len(np.unique(y)) == 2 else np.nan
        ap = (
            float(SK["average_precision_score"](y, p))
            if len(np.unique(y)) == 2
            else np.nan
        )
        brier = float(SK["brier_score_loss"](y, p))
        prior_brier = float(SK["brier_score_loss"](y, prior))
        ll = float(SK["log_loss"](y, p, labels=[0, 1]))

        eligible = df["EligibleEvidenceForPseudoSupervision"].astype(bool).values
        eligible_n = int(eligible.sum())

        eligible_auc = np.nan
        eligible_ap = np.nan
        eligible_brier = np.nan
        eligible_prior_brier = np.nan
        eligible_brier_improvement = np.nan

        if eligible_n > 0:
            ey = y[eligible]
            ep = p[eligible]
            eprior = (
                df.loc[
                    eligible,
                    "OuterTrainEvidencePriorProbability",
                ]
                .astype(float)
                .values
            )

            eligible_brier = float(SK["brier_score_loss"](ey, ep))

            if np.isfinite(eprior).all():
                eligible_prior_brier = float(SK["brier_score_loss"](ey, eprior))
                eligible_brier_improvement = float(
                    eligible_prior_brier - eligible_brier
                )

            if eligible_n > 1 and len(np.unique(ey)) == 2:
                eligible_auc = float(SK["roc_auc_score"](ey, ep))
                eligible_ap = float(SK["average_precision_score"](ey, ep))

        rows.append(
            {
                "Label": label,
                "GoldPositive": int(y.sum()),
                "GoldNegative": int(len(y) - y.sum()),
                "FoldSafeOOF_AUROC": auc,
                "FoldSafeOOF_AP": ap,
                "FoldSafeOOF_Brier": brier,
                "FoldSafePrior_Brier": prior_brier,
                "FoldSafeBrierImprovementVsPrior": float(prior_brier - brier),
                "FoldSafeOOF_LogLoss": ll,
                "HeldOutEvidenceAvailableN": int(
                    df["EvidenceAvailable"].astype(bool).sum()
                ),
                "GateEligibleHeldOutEvidenceN": eligible_n,
                "GateEligibleHeldOutEvidence_AUROC": eligible_auc,
                "GateEligibleHeldOutEvidence_AP": eligible_ap,
                "GateEligibleHeldOutEvidence_Brier": eligible_brier,
                "GateEligibleHeldOutEvidence_Prior_Brier": eligible_prior_brier,
                "GateEligibleHeldOutEvidence_BrierImprovementVsPrior": (
                    eligible_brier_improvement
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    macro = {
        "Label": "MACRO_MEAN",
        "GoldPositive": int(metrics["GoldPositive"].sum()),
        "GoldNegative": int(metrics["GoldNegative"].sum()),
    }

    numeric_mean_columns = [
        "FoldSafeOOF_AUROC",
        "FoldSafeOOF_AP",
        "FoldSafeOOF_Brier",
        "FoldSafePrior_Brier",
        "FoldSafeBrierImprovementVsPrior",
        "FoldSafeOOF_LogLoss",
        "HeldOutEvidenceAvailableN",
        "GateEligibleHeldOutEvidenceN",
        "GateEligibleHeldOutEvidence_AUROC",
        "GateEligibleHeldOutEvidence_AP",
        "GateEligibleHeldOutEvidence_Brier",
        "GateEligibleHeldOutEvidence_Prior_Brier",
        "GateEligibleHeldOutEvidence_BrierImprovementVsPrior",
    ]

    for column in numeric_mean_columns:
        macro[column] = float(metrics[column].mean(skipna=True))

    return pd.concat(
        [metrics, pd.DataFrame([macro])],
        ignore_index=True,
    )


def build_gate_stability_summary(
    all_gate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for label in LABEL_COLUMNS:
        group = all_gate_metrics[all_gate_metrics["Label"] == label].copy()
        passed = group[group["PseudoLabelGatePass"]]

        rows.append(
            {
                "Label": label,
                "PassFolds": int(group["PseudoLabelGatePass"].sum()),
                "PassFraction": float(group["PseudoLabelGatePass"].mean()),
                "PassedFoldIDs": ",".join(
                    str(int(value)) for value in passed["OuterFold"].tolist()
                ),
                "MinOuterTrainEvidenceN": int(group["EvidenceAvailableN"].min()),
                "MaxOuterTrainEvidenceN": int(group["EvidenceAvailableN"].max()),
                "MeanInnerEvidenceAUROC": float(
                    group["EvidenceOnly_AUROC"].mean(skipna=True)
                ),
                "MinInnerEvidenceAUROC": (
                    float(group["EvidenceOnly_AUROC"].min(skipna=True))
                    if group["EvidenceOnly_AUROC"].notna().any()
                    else np.nan
                ),
                "MeanInnerEvidenceBrierImprovement": float(
                    group["EvidenceOnly_BrierImprovementVsPrior"].mean(skipna=True)
                ),
                "MinInnerEvidenceBrierImprovement": (
                    float(
                        group["EvidenceOnly_BrierImprovementVsPrior"].min(skipna=True)
                    )
                    if group["EvidenceOnly_BrierImprovementVsPrior"].notna().any()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_probability_stability_diagnostics(
    all_unlabeled_scored: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-fold diagnostic only.

    DO NOT use this table to select pseudo labels for a particular image OOF
    fold: other calibrators may have seen that fold's held-out gold labels.
    """

    grouped = all_unlabeled_scored.groupby([UID_COLUMN, "Label"], sort=False)

    result = grouped.agg(
        RawProbabilityMean=("ChallengeProbabilityRaw", "mean"),
        RawProbabilityStd=("ChallengeProbabilityRaw", "std"),
        RawProbabilityMin=("ChallengeProbabilityRaw", "min"),
        RawProbabilityMax=("ChallengeProbabilityRaw", "max"),
        GatePassFolds=("LabelCalibrationEligible", "sum"),
        SoftLabelAvailableFolds=("SoftLabelAvailable", "sum"),
        HighSelectionFolds=("InitialHighSelectionCandidate", "sum"),
    ).reset_index()

    result["RawProbabilityRange"] = (
        result["RawProbabilityMax"] - result["RawProbabilityMin"]
    )

    # Mean over available fold-specific labels only; still diagnostic.
    available = all_unlabeled_scored[all_unlabeled_scored["SoftLabelAvailable"]].copy()
    available_mean = (
        available.groupby([UID_COLUMN, "Label"])["ChallengeSoftLabel"]
        .mean()
        .rename("AvailableSoftLabelMean_DiagnosticOnly")
        .reset_index()
    )

    result = result.merge(
        available_mean,
        on=[UID_COLUMN, "Label"],
        how="left",
        validate="one_to_one",
    )

    return result


# ============================================================
# 8. REPORT / MANIFEST
# ============================================================


def write_report(
    fold_assignments: pd.DataFrame,
    fold_hash: str,
    v4_verified: bool,
    v4_message: str,
    gate_stability: pd.DataFrame,
    outer_oof_metrics: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    study_coverage: pd.DataFrame,
    source_hashes: Dict[str, str],
) -> None:
    lines: List[str] = []

    lines.extend(
        [
            "# RSNA W2.3 — Fold-Safe Weak Label Generation",
            "",
            "## Scope",
            "",
            (
                "W2.3 regenerates Stage-B report→challenge calibration "
                "inside each of the five controlled V1.1/V4 outer image "
                "folds. Held-out gold labels are never used to fit the "
                "calibrator or decide the pseudo-label gate for that fold."
            ),
            "",
            (
                "Stage-A report features are frozen from W2; no NLI model "
                "is run in W2.3. Existing all-58 W2 Stage-B probabilities "
                "and gate decisions are ignored."
            ),
            "",
            "## Outer-fold integrity",
            "",
            f"- Generated fold SHA256: `{fold_hash}`",
            f"- Expected controlled V4 SHA256: `{EXPECTED_V4_FOLD_SHA256}`",
            f"- SHA256 match: `{fold_hash == EXPECTED_V4_FOLD_SHA256}`",
            f"- Optional V4 OOF verification: `{v4_verified}` — {v4_message}",
            "",
            "Fold sizes:",
            "",
            "```text",
            fold_assignments["OuterFold"]
            .value_counts()
            .sort_index()
            .rename("ValidationStudies")
            .to_string(),
            "```",
            "",
            "## Nested pseudo-label gate",
            "",
            f"- Minimum outer-training evidence N: {PSEUDO_GATE_MIN_EVIDENCE_N}",
            f"- Minimum inner evidence-only AUROC: {PSEUDO_GATE_MIN_EVIDENCE_AUROC}",
            (
                "- Inner evidence-only OOF Brier must be strictly better "
                "than the outer-training evidence-subset prevalence prior"
            ),
            "",
            "Gate stability across the five image folds:",
            "",
            "```text",
            gate_stability.to_string(index=False),
            "```",
            "",
            "## Leakage-free outer Stage-B OOF diagnostics",
            "",
            "```text",
            outer_oof_metrics.to_string(index=False),
            "```",
            "",
            "## Fold-specific unlabeled coverage",
            "",
            "```text",
            coverage_summary.to_string(index=False),
            "```",
            "",
            "## Study-level weak-label coverage",
            "",
        ]
    )

    study_fold_summary = study_coverage.groupby("OuterFold", as_index=False).agg(
        Studies=(UID_COLUMN, "count"),
        StudiesWithAnySoftLabel=("HasAnySoftLabel", "sum"),
        StudiesWithAnyHighSelection=("HasAnyHighSelectionLabel", "sum"),
        MeanAvailableLabels=("AvailableLabelCount", "mean"),
        MedianAvailableLabels=("AvailableLabelCount", "median"),
    )

    lines.extend(
        [
            "```text",
            study_fold_summary.to_string(index=False),
            "```",
            "",
            "## Important methodological boundary",
            "",
            (
                "This is fold-safe for Stage-B challenge-ontology mapping. "
                "It is not a fully nested external validation of the entire "
                "W1/W1.2/W1.3/W2 research process, because Stage-A rule and "
                "semantic design was developed using the same 58-study "
                "research cohort."
            ),
            "",
            (
                "The cross-fold probability-stability file is diagnostic "
                "only. Do not use information from other outer-fold "
                "calibrators to select samples inside a specific image OOF "
                "fold, because those other calibrators may have used that "
                "fold's validation gold labels."
            ),
            "",
            "## Source hashes",
            "",
            "```text",
            json.dumps(source_hashes, indent=2),
            "```",
            "",
        ]
    )

    (RESULT_ROOT / "W2_3_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_manifest(
    fold_hash: str,
    v4_verified: bool,
    v4_message: str,
    source_hashes: Dict[str, str],
) -> None:
    manifest = {
        "name": "RSNA W2.3 Fold-Safe Weak Label Generation",
        "train_csv": str(TRAIN_CSV),
        "gold_structured_csv": str(GOLD_STRUCTURED_CSV),
        "full_structured_csv": str(FULL_STRUCTURED_CSV),
        "output_root": str(OUTPUT_ROOT),
        "counts": {
            "total_studies": EXPECTED_TOTAL_STUDIES,
            "gold_studies": EXPECTED_GOLD_STUDIES,
            "unlabeled_studies": EXPECTED_UNLABELED_STUDIES,
            "labels": EXPECTED_LABELS,
        },
        "outer_folds": {
            "n_splits": NUM_OUTER_FOLDS,
            "seed": OUTER_RANDOM_SEED,
            "assignment_sha256": fold_hash,
            "expected_v4_sha256": EXPECTED_V4_FOLD_SHA256,
            "sha256_match": fold_hash == EXPECTED_V4_FOLD_SHA256,
            "v4_oof_verified": v4_verified,
            "v4_oof_message": v4_message,
        },
        "stage_b": {
            "features": STAGE_B_FEATURES,
            "calibration_C": CALIBRATION_C,
            "inner_cv_splits_max": INNER_CV_SPLITS,
            "inner_cv_repeats": INNER_CV_REPEATS,
            "inner_cv_random_state": INNER_CV_RANDOM_STATE,
            "pseudo_label_gate": {
                "min_evidence_n": PSEUDO_GATE_MIN_EVIDENCE_N,
                "min_evidence_auc": PSEUDO_GATE_MIN_EVIDENCE_AUROC,
                "min_evidence_brier_improvement": (PSEUDO_GATE_MIN_BRIER_IMPROVEMENT),
            },
            "selection_score_threshold_diagnostic": SELECTION_SCORE_THRESHOLD,
        },
        "fold_safety": {
            "held_out_gold_used_for_calibrator_fit": False,
            "held_out_gold_used_for_gate": False,
            "all_58_w2_calibrators_reused": False,
            "existing_w2_stage_b_probabilities_reused": False,
            "stage_a_features_reused": True,
        },
        "source_hashes": source_hashes,
    }

    with (RESULT_ROOT / "w2_3_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(safe_json_value(manifest), handle, indent=2, ensure_ascii=False)


# ============================================================
# 9. MAIN
# ============================================================


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
    )

    ensure_dirs()

    print("=" * 88)
    print("RSNA W2.3 - FOLD-SAFE WEAK LABEL GENERATION")
    print("=" * 88)
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Gold Stage-A features: {GOLD_STRUCTURED_CSV}")
    print(f"Full Stage-A features: {FULL_STRUCTURED_CSV}")
    print(f"Output: {OUTPUT_ROOT}")
    print()
    print("No NLI model will be loaded.")
    print("No MRI model will be trained.")
    print("All-58 W2 Stage-B probabilities/calibrators are ignored.")

    (
        train_df,
        gold_df,
        gold_structured,
        unlabeled_structured,
        source_hashes,
    ) = load_inputs()

    # --------------------------------------------------------
    # Exact outer image fold reproduction.
    # --------------------------------------------------------
    y_all = gold_df[LABEL_COLUMNS].values.astype(np.int64)
    fold_ids_zero_based = greedy_multilabel_folds(
        y_all,
        n_splits=NUM_OUTER_FOLDS,
        seed=OUTER_RANDOM_SEED,
    )

    fold_assignments = gold_df[[UID_COLUMN, *LABEL_COLUMNS]].copy()
    fold_assignments["OuterFold"] = fold_ids_zero_based + 1

    # Put fold near UID for readability.
    ordered_columns = [UID_COLUMN, "OuterFold", *LABEL_COLUMNS]
    fold_assignments = fold_assignments[ordered_columns]

    fold_hash = fold_assignment_sha256(fold_assignments[[UID_COLUMN, "OuterFold"]])

    print()
    print("Outer fold SHA256:", fold_hash)
    print("Expected V4 SHA256:", EXPECTED_V4_FOLD_SHA256)

    if fold_hash != EXPECTED_V4_FOLD_SHA256:
        message = (
            "Generated outer fold assignment does not match the controlled "
            "V1.1/V4 fold checksum."
        )
        if ALLOW_FOLD_MISMATCH:
            print("WARNING:", message)
            print("W23_ALLOW_FOLD_MISMATCH=1, so execution will continue.")
        else:
            raise RuntimeError(
                message + "\nRefusing to generate pseudo labels because they would "
                "not align with the controlled image OOF protocol."
            )

    v4_verified, v4_message = verify_v4_oof_if_available(
        fold_assignments[[UID_COLUMN, "OuterFold"]]
    )
    print(v4_message)

    fold_assignments.to_csv(
        RESULT_ROOT / "00_outer_fold_assignments.csv",
        index=False,
    )

    # Fold label distribution audit.
    distribution_rows: List[Dict[str, Any]] = []
    for outer_fold in range(1, NUM_OUTER_FOLDS + 1):
        val = fold_assignments[fold_assignments["OuterFold"] == outer_fold]
        trn = fold_assignments[fold_assignments["OuterFold"] != outer_fold]

        for label in LABEL_COLUMNS:
            distribution_rows.append(
                {
                    "OuterFold": outer_fold,
                    "Label": label,
                    "TrainN": int(len(trn)),
                    "TrainPositive": int(trn[label].sum()),
                    "TrainNegative": int(len(trn) - trn[label].sum()),
                    "ValidationN": int(len(val)),
                    "ValidationPositive": int(val[label].sum()),
                    "ValidationNegative": int(len(val) - val[label].sum()),
                }
            )

    fold_distribution = pd.DataFrame(distribution_rows)
    fold_distribution.to_csv(
        RESULT_ROOT / "01_outer_fold_label_distribution.csv",
        index=False,
    )

    # Attach outer fold to gold structured table.
    uid_to_fold = dict(
        zip(
            fold_assignments[UID_COLUMN].astype(str),
            fold_assignments["OuterFold"].astype(int),
        )
    )
    gold_structured = gold_structured.copy()
    gold_structured["OuterFold"] = gold_structured[UID_COLUMN].map(uid_to_fold)

    if gold_structured["OuterFold"].isna().any():
        raise RuntimeError("Could not attach outer folds to all gold structured rows.")

    all_gate_metrics_parts: List[pd.DataFrame] = []
    all_inner_oof_parts: List[pd.DataFrame] = []
    all_outer_oof_parts: List[pd.DataFrame] = []
    all_unlabeled_parts: List[pd.DataFrame] = []
    all_coverage_parts: List[pd.DataFrame] = []
    all_study_coverage_parts: List[pd.DataFrame] = []
    all_coefficients: List[Dict[str, Any]] = []

    print()
    print("=" * 88)
    print("GENERATING 5 FOLD-SAFE STAGE-B CALIBRATIONS")
    print("=" * 88)

    for outer_fold in range(1, NUM_OUTER_FOLDS + 1):
        print()
        print("-" * 88)
        print(f"OUTER IMAGE FOLD {outer_fold}/{NUM_OUTER_FOLDS}")
        print("-" * 88)

        fold_dir = FOLD_ROOT / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_gold = gold_structured[gold_structured["OuterFold"] != outer_fold].copy()
        validation_gold = gold_structured[
            gold_structured["OuterFold"] == outer_fold
        ].copy()

        train_uid_n = train_gold[UID_COLUMN].nunique()
        validation_uid_n = validation_gold[UID_COLUMN].nunique()

        expected_train_n = EXPECTED_GOLD_STUDIES - validation_uid_n
        if train_uid_n != expected_train_n:
            raise RuntimeError(
                f"Fold {outer_fold}: unexpected train UID count {train_uid_n}."
            )

        print(f"Gold train/validation studies: {train_uid_n} / {validation_uid_n}")

        calibrators: Dict[str, Dict[str, Any]] = {}
        fold_metric_rows: List[Dict[str, Any]] = []
        fold_inner_parts: List[pd.DataFrame] = []
        fold_coefficients: List[Dict[str, Any]] = []

        for label in LABEL_COLUMNS:
            label_train = (
                train_gold[train_gold["Label"] == label].copy().reset_index(drop=True)
            )

            calibrator, metrics, inner_oof, coefficient_rows = fit_inner_oof_and_final(
                label_train,
                outer_fold=outer_fold,
                label=label,
            )

            calibrators[label] = calibrator
            fold_metric_rows.append(metrics)
            fold_inner_parts.append(inner_oof)
            fold_coefficients.extend(coefficient_rows)

            print(
                f"{label:18s} "
                f"gate={'PASS' if metrics['PseudoLabelGatePass'] else 'HOLD':4s} "
                f"evidence_n={int(metrics['EvidenceAvailableN']):2d} "
                f"evidence_auc={format_float(metrics['EvidenceOnly_AUROC'], 3)} "
                f"evidence_brier_delta="
                f"{format_float(metrics['EvidenceOnly_BrierImprovementVsPrior'], 3)}"
            )

        fold_gate_metrics = pd.DataFrame(fold_metric_rows)
        fold_inner_oof = pd.concat(fold_inner_parts, ignore_index=True)
        fold_coefficient_df = pd.DataFrame(fold_coefficients)

        fold_gate_metrics.to_csv(
            fold_dir / "stage_b_inner_gate_metrics.csv",
            index=False,
        )
        fold_inner_oof.to_csv(
            fold_dir / "stage_b_inner_gold_oof_predictions.csv",
            index=False,
        )
        fold_coefficient_df.to_csv(
            fold_dir / "stage_b_coefficients.csv",
            index=False,
        )

        # Persist fold-specific models and the exact gold UID partition.
        SK["dump"](
            {
                "outer_fold": outer_fold,
                "train_gold_uids": sorted(
                    train_gold[UID_COLUMN].astype(str).unique().tolist()
                ),
                "validation_gold_uids": sorted(
                    validation_gold[UID_COLUMN].astype(str).unique().tolist()
                ),
                "calibrators": calibrators,
                "features": STAGE_B_FEATURES,
                "calibration_C": CALIBRATION_C,
                "inner_cv": {
                    "max_splits": INNER_CV_SPLITS,
                    "repeats": INNER_CV_REPEATS,
                    "random_state": INNER_CV_RANDOM_STATE,
                },
                "gate_metrics": fold_gate_metrics,
                "source_hashes": source_hashes,
                "outer_fold_assignment_sha256": fold_hash,
            },
            MODEL_ROOT / f"fold_{outer_fold}_stage_b_calibrators.joblib",
        )

        # Held-out gold diagnostics: these probabilities are never used to
        # generate pseudo labels or determine the gate.
        outer_validation_predictions = score_outer_validation_gold(
            validation_gold,
            calibrators,
            fold_gate_metrics,
            outer_fold,
        )
        outer_validation_predictions.to_csv(
            fold_dir / "heldout_gold_stage_b_predictions.csv",
            index=False,
        )

        # Fold-specific unlabeled pseudo labels.
        fold_scored = score_unlabeled_for_fold(
            unlabeled_structured,
            calibrators,
            fold_gate_metrics,
            outer_fold,
        )

        compact_columns = [
            "OuterFold",
            UID_COLUMN,
            "Label",
            "Language",
            "RuleAssertion",
            "SemanticAssertion",
            "FusedAssertion",
            "FusedAssertionConfidence",
            "FusionSource",
            "EvidenceAvailable",
            "ChallengeProbabilityRaw",
            "LabelCalibrationEligible",
            "LabelCalibrationGateReason",
            "SoftLabelAvailable",
            "ChallengeSoftLabel",
            "CandidateSelectionScore",
            "InitialHighSelectionCandidate",
        ]
        compact_columns = [
            column for column in compact_columns if column in fold_scored.columns
        ]

        fold_scored[compact_columns].to_csv(
            fold_dir / "soft_labels_long.csv",
            index=False,
            encoding="utf-8-sig",
        )

        write_fold_wide_outputs(fold_scored, fold_dir)

        coverage = summarize_fold_coverage(
            fold_scored,
            outer_fold,
            fold_gate_metrics,
        )
        coverage.to_csv(
            fold_dir / "unlabeled_label_coverage_summary.csv",
            index=False,
        )

        study_coverage = summarize_study_coverage(
            fold_scored,
            outer_fold,
        )
        study_coverage.to_csv(
            fold_dir / "unlabeled_study_coverage.csv",
            index=False,
        )

        all_gate_metrics_parts.append(fold_gate_metrics)
        all_inner_oof_parts.append(fold_inner_oof)
        all_outer_oof_parts.append(outer_validation_predictions)
        all_unlabeled_parts.append(fold_scored[compact_columns].copy())
        all_coverage_parts.append(coverage)
        all_study_coverage_parts.append(study_coverage)
        all_coefficients.extend(fold_coefficients)

    # --------------------------------------------------------
    # Aggregate outputs.
    # --------------------------------------------------------
    all_gate_metrics = pd.concat(all_gate_metrics_parts, ignore_index=True)
    all_inner_oof = pd.concat(all_inner_oof_parts, ignore_index=True)
    outer_oof = pd.concat(all_outer_oof_parts, ignore_index=True)
    all_unlabeled = pd.concat(all_unlabeled_parts, ignore_index=True)
    all_coverage = pd.concat(all_coverage_parts, ignore_index=True)
    all_study_coverage = pd.concat(
        all_study_coverage_parts,
        ignore_index=True,
    )
    all_coefficient_df = pd.DataFrame(all_coefficients)

    # Strong integrity: each of 58 gold studies must appear exactly once per
    # label in held-out outer OOF predictions.
    expected_outer_oof_rows = EXPECTED_GOLD_STUDIES * EXPECTED_LABELS
    if len(outer_oof) != expected_outer_oof_rows:
        raise RuntimeError(
            f"Expected {expected_outer_oof_rows} outer OOF rows, "
            f"found {len(outer_oof)}."
        )

    if outer_oof[[UID_COLUMN, "Label"]].duplicated().any():
        raise RuntimeError("Duplicate gold UID/Label in outer OOF predictions.")

    expected_combined_unlabeled_rows = (
        NUM_OUTER_FOLDS * EXPECTED_UNLABELED_STUDIES * EXPECTED_LABELS
    )
    if len(all_unlabeled) != expected_combined_unlabeled_rows:
        raise RuntimeError(
            f"Expected {expected_combined_unlabeled_rows} combined pseudo rows, "
            f"found {len(all_unlabeled)}."
        )

    all_gate_metrics.to_csv(
        RESULT_ROOT / "02_all_fold_inner_gate_metrics.csv",
        index=False,
    )
    all_inner_oof.to_csv(
        RESULT_ROOT / "03_all_fold_inner_gold_oof_predictions.csv",
        index=False,
    )
    outer_oof.to_csv(
        RESULT_ROOT / "04_fold_safe_gold_outer_oof_predictions.csv",
        index=False,
    )

    outer_oof_metrics = compute_outer_oof_metrics(outer_oof)
    outer_oof_metrics.to_csv(
        RESULT_ROOT / "05_fold_safe_gold_outer_oof_metrics.csv",
        index=False,
    )

    all_unlabeled.to_csv(
        RESULT_ROOT / "06_fold_safe_unlabeled_soft_labels_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_coverage.to_csv(
        RESULT_ROOT / "07_fold_unlabeled_label_coverage_summary.csv",
        index=False,
    )
    all_study_coverage.to_csv(
        RESULT_ROOT / "08_fold_unlabeled_study_coverage.csv",
        index=False,
    )

    gate_stability = build_gate_stability_summary(all_gate_metrics)
    gate_stability.to_csv(
        RESULT_ROOT / "09_label_gate_stability_summary.csv",
        index=False,
    )

    probability_stability = build_probability_stability_diagnostics(all_unlabeled)
    probability_stability.to_csv(
        RESULT_ROOT / "10_cross_fold_probability_stability_DIAGNOSTIC_ONLY.csv",
        index=False,
    )

    all_coefficient_df.to_csv(
        RESULT_ROOT / "11_all_fold_stage_b_coefficients.csv",
        index=False,
    )

    write_manifest(
        fold_hash,
        v4_verified,
        v4_message,
        source_hashes,
    )
    write_report(
        fold_assignments,
        fold_hash,
        v4_verified,
        v4_message,
        gate_stability,
        outer_oof_metrics,
        all_coverage,
        all_study_coverage,
        source_hashes,
    )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------
    print()
    print("=" * 88)
    print("W2.3 FOLD-SAFE GENERATION COMPLETE")
    print("=" * 88)
    print()
    print("Gate stability across outer image folds:")
    print(
        gate_stability[
            [
                "Label",
                "PassFolds",
                "PassFraction",
                "PassedFoldIDs",
                "MinOuterTrainEvidenceN",
                "MeanInnerEvidenceAUROC",
                "MeanInnerEvidenceBrierImprovement",
            ]
        ].to_string(index=False)
    )

    print()
    print("Fold-safe Stage-B outer OOF metrics:")
    print(
        outer_oof_metrics[
            [
                "Label",
                "FoldSafeOOF_AUROC",
                "FoldSafeOOF_AP",
                "FoldSafeOOF_Brier",
                "FoldSafePrior_Brier",
                "FoldSafeBrierImprovementVsPrior",
                "GateEligibleHeldOutEvidenceN",
            ]
        ].to_string(index=False)
    )

    print()
    print("Fold-specific pseudo-label coverage:")
    print(
        all_coverage[
            [
                "OuterFold",
                "Label",
                "PseudoLabelGatePass",
                "SoftLabelAvailableN",
                "SoftLabelCoverage",
                f"SelectionScoreN_ge_{SELECTION_SCORE_THRESHOLD:g}",
            ]
        ].to_string(index=False)
    )

    print()
    print("Most important image-training inputs:")
    for outer_fold in range(1, NUM_OUTER_FOLDS + 1):
        print(
            f"  Fold {outer_fold}: "
            f"{FOLD_ROOT / f'fold_{outer_fold}' / 'soft_probabilities_wide.csv'}"
        )
        print(
            f"          "
            f"{FOLD_ROOT / f'fold_{outer_fold}' / 'soft_label_availability_wide.csv'}"
        )
        print(
            f"          "
            f"{FOLD_ROOT / f'fold_{outer_fold}' / 'candidate_selection_scores_wide.csv'}"
        )

    print()
    print("Global audit/report files:")
    print(f"  {RESULT_ROOT / '00_outer_fold_assignments.csv'}")
    print(f"  {RESULT_ROOT / '05_fold_safe_gold_outer_oof_metrics.csv'}")
    print(f"  {RESULT_ROOT / '09_label_gate_stability_summary.csv'}")
    print(f"  {RESULT_ROOT / 'W2_3_REPORT.md'}")
    print()
    print("No MRI image model was trained.")


if __name__ == "__main__":
    main()
