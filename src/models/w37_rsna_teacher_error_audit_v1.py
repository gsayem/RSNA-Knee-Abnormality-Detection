#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W37 / Teacher Error Audit v1

ZERO-GPU diagnostic only.

Goals
-----
1. Reconstruct W2.3 fold-safe 58-gold OOF predictions.
2. Reconstruct W2.6 FS4/fixed50 fold-safe 58-gold predictions.
3. Independently recompute AUROC / AP / Brier.
4. Audit study/label error cases against raw reports.
5. Use report P/A/U/N state data when an explicit state artifact exists.
   Never invent/derive states heuristically.
6. Separately audit the W2.6-P production probabilities, masks and weights
   for the 4,349 unlabeled studies.
7. Produce a small priority report before any W8 experiment.

IMPORTANT
---------
- This script does NOT train anything.
- This script does NOT load Curia.
- This script does NOT read DICOMs.
- This script does NOT use Pilkwang as production truth.
- Production W2.6-P uses all 58 gold reports and is NOT pristine OOF evidence.
- Gold evaluation comes only from fold-safe held-out prediction artifacts.
- No project .py script is imported or executed.

Expected project layout
-----------------------
PROJECT_ROOT/
├── input/
│   └── train.csv
└── output/results/
    ├── rsna_w2/
    ├── rsna_w2_3/
    ├── rsna_w2_6_v2/
    └── rsna_w2_6p_fast/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# =============================================================================
# VERSION / CONSTANTS
# =============================================================================

SCRIPT_VERSION = "rsna_teacher_error_audit_v1"

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

FS4 = [
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Synovitis",
]

EXPECTED_GOLD = 58
EXPECTED_UNLABELED = 4349
EXPECTED_SELECTED_CELLS = 32027

EXPECTED_W23_AUC = 0.771243
EXPECTED_W26_ORIGINAL_FIXED50_AUC = 0.80929
EXPECTED_W26_FAST_FIXED50_AUC = 0.81320

W23_TOLERANCE = 0.008
W26_TOLERANCE = 0.020

HIGH_CONFIDENCE_ERROR = 0.80


# =============================================================================
# BASIC HELPERS
# =============================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def norm(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


LABEL_BY_NORM = {norm(label): label for label in LABELS}

LABEL_ALIASES = {
    "baker": "Baker's",
    "bakers": "Baker's",
    "bakercyst": "Baker's",
    "bakerscyst": "Baker's",
    "medialmeniscus": "Medial Meniscus",
    "lateralmeniscus": "Lateral Meniscus",
    "medialoa": "Medial OA",
    "lateraloa": "Lateral OA",
    "pfoa": "PF OA",
}


def canonical_label(value: object) -> Optional[str]:
    key = norm(value)

    if key in LABEL_BY_NORM:
        return LABEL_BY_NORM[key]

    return LABEL_ALIASES.get(key)


def safe_json_dump(obj: object, path: Path) -> None:
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


def find_column(
    columns: Sequence[str],
    possibilities: Sequence[str],
) -> Optional[str]:
    by_norm = {norm(c): c for c in columns}

    for candidate in possibilities:
        key = norm(candidate)

        if key in by_norm:
            return by_norm[key]

    return None


def find_uid_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df.columns,
        [
            UID,
            "StudyUID",
            "study_uid",
            "UID",
        ],
    )


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    explicit = find_column(
        df.columns,
        [
            "Label",
            "Target",
            "Abnormality",
            "Diagnosis",
        ],
    )

    if explicit is not None:
        values = df[explicit].dropna().astype(str).map(canonical_label)

        if values.notna().mean() >= 0.60:
            return explicit

    # Structural fallback: find a column mostly containing known labels.
    for c in df.columns:
        if c == find_uid_column(df):
            continue

        if not (
            pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
        ):
            continue

        values = df[c].dropna().astype(str).map(canonical_label)

        if len(values) >= 20 and values.notna().mean() >= 0.80:
            return c

    return None


def numeric_probability_columns(
    df: pd.DataFrame,
    exclude: Sequence[str] = (),
) -> List[Tuple[int, str]]:
    exclude_set = set(exclude)

    out: List[Tuple[int, str]] = []

    for c in df.columns:
        if c in exclude_set:
            continue

        values = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        valid = values.notna()

        if valid.sum() < max(
            10,
            int(0.50 * len(df)),
        ):
            continue

        vv = values[valid].to_numpy(dtype=float)

        if not np.isfinite(vv).all():
            continue

        if vv.min() < -1e-7 or vv.max() > 1.0000001:
            continue

        name = norm(c)

        score = 0

        if "probability" in name:
            score += 10
        elif "prob" in name:
            score += 8

        if "prediction" in name:
            score += 7
        elif "pred" in name:
            score += 5

        if "qwen" in name:
            score += 6

        if "fs4" in name:
            score += 6

        if "challenge" in name:
            score += 5

        if "fast" in name:
            score += 4

        # Things we do NOT want as a prediction column.
        for bad in [
            "true",
            "truth",
            "gold",
            "target",
            "label",
            "weight",
            "mask",
            "std",
            "variance",
            "stability",
            "confidence",
        ]:
            if bad in name:
                score -= 20

        out.append((score, c))

    return sorted(
        out,
        key=lambda x: (-x[0], x[1]),
    )


def parse_bool_series(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)

    numeric = pd.to_numeric(
        s,
        errors="coerce",
    )

    if numeric.notna().all():
        return numeric.to_numpy(dtype=float) > 0.5

    text = s.astype(str).str.strip().str.lower()

    return text.isin(
        [
            "true",
            "1",
            "yes",
            "y",
            "t",
        ]
    ).to_numpy(dtype=bool)


# =============================================================================
# PROJECT DISCOVERY
# =============================================================================


def discover_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent

    for p in [
        script_dir,
        *script_dir.parents,
    ]:
        if (p / "input" / "train.csv").exists():
            return p

    # User's standard project structure:
    # PROJECT_ROOT/src/models/script.py
    candidate = (script_dir / ".." / "..").resolve()

    if (candidate / "input" / "train.csv").exists():
        return candidate

    raise FileNotFoundError(
        "Could not discover PROJECT_ROOT containing input/train.csv"
    )


def first_existing(
    candidates: Iterable[Path],
) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p.resolve()

    return None


@dataclass
class Paths:
    project_root: str
    train_csv: str
    output_root: str

    w2_root: Optional[str]
    w23_root: Optional[str]
    w26_root: Optional[str]
    w26p_fast_root: Optional[str]

    @classmethod
    def discover(cls) -> "Paths":
        root = discover_project_root()

        results = root / "output" / "results"

        train_csv = root / "input" / "train.csv"

        output_root = results / "rsna_teacher_error_audit_v1"

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        def find_stage(
            exact_names: Sequence[str],
        ) -> Optional[Path]:
            candidates = [results / name for name in exact_names]

            found = first_existing(candidates)

            if found:
                return found

            if results.exists():
                lowered = [x.lower() for x in exact_names]

                matches = []

                for child in results.iterdir():
                    if not child.is_dir():
                        continue

                    c = child.name.lower()

                    if any(x in c or c in x for x in lowered):
                        matches.append(child)

                if matches:
                    return sorted(
                        matches,
                        key=lambda p: len(str(p)),
                    )[0].resolve()

            return None

        w2 = find_stage(
            [
                "rsna_w2",
                "rsna_w2_full",
            ]
        )

        w23 = find_stage(
            [
                "rsna_w2_3",
            ]
        )

        w26 = find_stage(
            [
                "rsna_w2_6_v2",
            ]
        )

        w26p_fast = find_stage(
            [
                "rsna_w2_6p_fast",
            ]
        )

        return cls(
            project_root=str(root),
            train_csv=str(train_csv.resolve()),
            output_root=str(output_root.resolve()),
            w2_root=(str(w2) if w2 else None),
            w23_root=(str(w23) if w23 else None),
            w26_root=(str(w26) if w26 else None),
            w26p_fast_root=(str(w26p_fast) if w26p_fast else None),
        )


# =============================================================================
# GOLD DATA
# =============================================================================


def load_gold(
    paths: Paths,
) -> pd.DataFrame:
    df = pd.read_csv(paths.train_csv)

    required = [
        UID,
        REPORT,
        *LABELS,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"train.csv missing columns: {missing}")

    df[UID] = df[UID].astype(str)

    gold = df[df[LABELS].notna().all(axis=1)].copy()

    if len(gold) != EXPECTED_GOLD:
        raise RuntimeError(
            f"Expected {EXPECTED_GOLD} gold studies, " f"found {len(gold)}"
        )

    for label in LABELS:
        gold[label] = pd.to_numeric(
            gold[label],
            errors="raise",
        ).astype(int)

    return gold.reset_index(drop=True)


# =============================================================================
# METRICS
# =============================================================================


def per_label_metrics(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    g = gold.set_index(UID).loc[
        predictions.index,
        LABELS,
    ]

    rows = []

    for label in LABELS:
        y = g[label].to_numpy(dtype=float)

        p = predictions[label].to_numpy(dtype=float)

        if not np.isfinite(p).all():
            raise RuntimeError(f"{model_name}/{label}: non-finite probabilities")

        if (p < 0).any() or (p > 1).any():
            raise RuntimeError(f"{model_name}/{label}: probability outside [0,1]")

        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

        ap = (
            float(
                average_precision_score(
                    y,
                    p,
                )
            )
            if y.sum() > 0
            else np.nan
        )

        brier = float(np.mean((p - y) ** 2))

        rows.append(
            {
                "Model": model_name,
                "Label": label,
                "N": len(y),
                "Positives": int(y.sum()),
                "AUROC": auc,
                "AP": ap,
                "Brier": brier,
            }
        )

    return pd.DataFrame(rows)


def macro_auc(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
) -> float:
    metrics = per_label_metrics(
        gold,
        predictions,
        "_tmp",
    )

    return float(metrics["AUROC"].mean())


# =============================================================================
# GENERIC PREDICTION TABLE EXTRACTION
# =============================================================================


def require_gold_uid_coverage(
    wide: pd.DataFrame,
    gold: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    wide = wide.copy()

    wide.index = wide.index.astype(str)

    expected = set(gold[UID].astype(str))

    actual = set(wide.index.astype(str))

    missing = expected - actual
    extras = actual - expected

    if missing:
        raise RuntimeError(f"{source}: missing {len(missing)} gold UIDs")

    # Extras are acceptable only if the table also contains unlabeled
    # rows. Restrict deterministically to the gold UIDs.
    ordered_uids = list(gold[UID].astype(str))

    wide = wide.loc[ordered_uids]

    if wide.index.duplicated().any():
        raise RuntimeError(f"{source}: duplicate StudyInstanceUID")

    return wide


def extract_long_prediction(
    df: pd.DataFrame,
    required_labels: Sequence[str],
    source_name: str,
    preferred_tokens: Sequence[str] = (),
) -> Tuple[pd.DataFrame, dict]:
    uid_col = find_uid_column(df)
    label_col = find_label_column(df)

    if uid_col is None or label_col is None:
        raise ValueError("not a recognizable long prediction table")

    tmp = df.copy()

    tmp["_CanonicalLabel"] = tmp[label_col].map(canonical_label)

    tmp = tmp[tmp["_CanonicalLabel"].isin(required_labels)].copy()

    if tmp.empty:
        raise ValueError("no required labels")

    candidates = numeric_probability_columns(
        tmp,
        exclude=[
            uid_col,
            label_col,
        ],
    )

    rescored = []

    for score, col in candidates:
        name = norm(col)

        for token in preferred_tokens:
            if norm(token) in name:
                score += 6

        rescored.append((score, col))

    rescored = sorted(
        rescored,
        key=lambda x: (-x[0], x[1]),
    )

    if not rescored:
        raise ValueError(f"{source_name}: no probability column")

    best_score = rescored[0][0]

    tied = [x for x in rescored if x[0] == best_score]

    if len(tied) > 1 and best_score <= 0:
        raise ValueError(
            f"{source_name}: ambiguous probability columns " f"{[x[1] for x in tied]}"
        )

    probability_col = rescored[0][1]

    tmp[uid_col] = tmp[uid_col].astype(str)

    tmp[probability_col] = pd.to_numeric(
        tmp[probability_col],
        errors="coerce",
    )

    if tmp[[uid_col, "_CanonicalLabel"]].duplicated().any():
        raise ValueError(f"{source_name}: duplicate UID/Label rows")

    wide = tmp.pivot(
        index=uid_col,
        columns="_CanonicalLabel",
        values=probability_col,
    )

    missing_labels = [label for label in required_labels if label not in wide.columns]

    if missing_labels:
        raise ValueError(f"{source_name}: missing labels {missing_labels}")

    wide = wide[list(required_labels)]

    return wide, {
        "format": "long",
        "uid_column": uid_col,
        "label_column": label_col,
        "probability_column": probability_col,
    }


def wide_label_column_candidates(
    df: pd.DataFrame,
    label: str,
) -> List[Tuple[int, str]]:
    target = norm(label)

    candidates = []

    for c in df.columns:
        n = norm(c)

        if target not in n:
            continue

        values = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        if values.notna().mean() < 0.90:
            continue

        vv = values.dropna().to_numpy(dtype=float)

        if len(vv) == 0 or vv.min() < -1e-7 or vv.max() > 1.0000001:
            continue

        score = 0

        if n == target:
            score += 2

        if "probability" in n:
            score += 10
        elif "prob" in n:
            score += 8

        if "prediction" in n:
            score += 7
        elif "pred" in n:
            score += 5

        if "gold" in n or "true" in n:
            score -= 20

        candidates.append((score, c))

    return sorted(
        candidates,
        key=lambda x: (-x[0], x[1]),
    )


def extract_wide_prediction(
    df: pd.DataFrame,
    required_labels: Sequence[str],
    source_name: str,
) -> Tuple[pd.DataFrame, dict]:
    uid_col = find_uid_column(df)

    if uid_col is None:
        raise ValueError(f"{source_name}: no UID column")

    selected = {}

    for label in required_labels:
        candidates = wide_label_column_candidates(
            df,
            label,
        )

        if not candidates:
            raise ValueError(
                f"{source_name}: cannot find wide probability " f"column for {label}"
            )

        selected[label] = candidates[0][1]

    wide = pd.DataFrame(
        {
            label: pd.to_numeric(
                df[col],
                errors="coerce",
            )
            for label, col in selected.items()
        },
        index=df[uid_col].astype(str),
    )

    return wide, {
        "format": "wide",
        "uid_column": uid_col,
        "columns": selected,
    }


def extract_prediction_table(
    df: pd.DataFrame,
    required_labels: Sequence[str],
    source_name: str,
    preferred_tokens: Sequence[str] = (),
) -> Tuple[pd.DataFrame, dict]:
    errors = []

    try:
        return extract_long_prediction(
            df,
            required_labels,
            source_name,
            preferred_tokens,
        )

    except Exception as exc:
        errors.append(f"long: {exc}")

    try:
        return extract_wide_prediction(
            df,
            required_labels,
            source_name,
        )

    except Exception as exc:
        errors.append(f"wide: {exc}")

    raise ValueError(
        f"{source_name}: unsupported prediction table. " + " | ".join(errors)
    )


# =============================================================================
# W2.3 GOLD OOF
# =============================================================================


def find_file_recursive(
    root: Path,
    filename: str,
) -> Optional[Path]:
    direct = [
        root / filename,
        root / "results" / filename,
    ]

    found = first_existing(direct)

    if found:
        return found

    matches = list(root.rglob(filename))

    if matches:
        return sorted(
            matches,
            key=lambda p: len(str(p)),
        )[0]

    return None


def load_w23_gold_predictions(
    paths: Paths,
    gold: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    if paths.w23_root is None:
        raise FileNotFoundError("rsna_w2_3 root not found")

    root = Path(paths.w23_root)

    source = find_file_recursive(
        root,
        "heldout_gold_stage_b_predictions.csv",
    )

    if source is None:
        raise FileNotFoundError(
            "Could not locate " "heldout_gold_stage_b_predictions.csv"
        )

    df = pd.read_csv(source)

    wide, meta = extract_prediction_table(
        df,
        LABELS,
        str(source),
        preferred_tokens=[
            "prob",
            "stage_b",
            "heldout",
        ],
    )

    wide = require_gold_uid_coverage(
        wide,
        gold,
        str(source),
    )

    observed_auc = macro_auc(
        gold,
        wide,
    )

    if abs(observed_auc - EXPECTED_W23_AUC) > W23_TOLERANCE:
        raise RuntimeError(
            "W2.3 benchmark guard FAILED. "
            f"Recomputed={observed_auc:.6f}, "
            f"expected≈{EXPECTED_W23_AUC:.6f}. "
            f"Selected source={source}, meta={meta}"
        )

    meta.update(
        {
            "source": str(source.resolve()),
            "macro_auroc": observed_auc,
        }
    )

    return wide, meta


# =============================================================================
# W2.6 / CURRENT FAST FS4 GOLD DISCOVERY
# =============================================================================


def score_fs4_candidate_name(
    path: Path,
    probability_col: Optional[str],
) -> int:
    text = norm(str(path) + " " + str(probability_col or ""))

    score = 0

    for good in [
        "fs4",
        "challenge",
        "qwen",
        "fast",
        "gold",
        "oof",
        "heldout",
        "validation",
    ]:
        if good in text:
            score += 2

    for bad in [
        "fixed50",
        "hybrid",
        "finalhybrid",
        "productionwide",
        "mask",
        "weight",
    ]:
        if bad in text:
            score -= 4

    return score


def candidate_expected_fixed_auc(
    path: Path,
) -> float:
    text = str(path).lower()

    if "w2_6p_fast" in text or "fast" in text:
        return EXPECTED_W26_FAST_FIXED50_AUC

    return EXPECTED_W26_ORIGINAL_FIXED50_AUC


def discover_fs4_gold_predictions(
    paths: Paths,
    gold: pd.DataFrame,
    w23: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    roots = []

    if paths.w26p_fast_root:
        roots.append(Path(paths.w26p_fast_root))

    if paths.w26_root:
        roots.append(Path(paths.w26_root))

    if not roots:
        raise FileNotFoundError("Neither rsna_w2_6p_fast nor rsna_w2_6_v2 was found")

    candidates = []

    for root in roots:
        for path in root.rglob("*.csv"):
            try:
                # Skip huge production files unless they clearly
                # look like validation/gold/OFF artifacts.
                lower = path.name.lower()

                likely_gold = any(
                    token in lower
                    for token in [
                        "gold",
                        "oof",
                        "heldout",
                        "validation",
                        "fs4",
                    ]
                )

                if path.stat().st_size > 20 * 1024 * 1024 and not likely_gold:
                    continue

                df = pd.read_csv(path)

                wide4, meta = extract_prediction_table(
                    df,
                    FS4,
                    str(path),
                    preferred_tokens=[
                        "fs4",
                        "qwen",
                        "challenge",
                        "fast",
                    ],
                )

                wide4 = require_gold_uid_coverage(
                    wide4,
                    gold,
                    str(path),
                )

                # Build controlled variants from exact W2.3.
                replace = w23.copy()
                fixed50 = w23.copy()

                for label in FS4:
                    replace[label] = wide4[label]

                    fixed50[label] = 0.50 * w23[label] + 0.50 * wide4[label]

                replace_auc = macro_auc(
                    gold,
                    replace,
                )

                fixed_auc = macro_auc(
                    gold,
                    fixed50,
                )

                expected = candidate_expected_fixed_auc(path)

                distance = abs(fixed_auc - expected)

                probability_col = meta.get("probability_column")

                structural = score_fs4_candidate_name(
                    path,
                    probability_col,
                )

                candidates.append(
                    {
                        "path": path,
                        "wide4": wide4,
                        "replace": replace,
                        "fixed50": fixed50,
                        "meta": meta,
                        "replace_auc": replace_auc,
                        "fixed_auc": fixed_auc,
                        "expected": expected,
                        "distance": distance,
                        "structural_score": structural,
                    }
                )

            except Exception:
                continue

    if not candidates:
        raise RuntimeError(
            "Could not identify any 58-gold FS4 prediction artifact "
            "under W2.6/W2.6-P FAST."
        )

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["distance"],
            -x["structural_score"],
            len(str(x["path"])),
        ),
    )

    best = candidates[0]

    if best["distance"] > W26_TOLERANCE:
        preview = [
            {
                "path": str(x["path"]),
                "fixed_auc": x["fixed_auc"],
                "replace_auc": x["replace_auc"],
                "expected": x["expected"],
                "distance": x["distance"],
                "meta": x["meta"],
            }
            for x in candidates[:10]
        ]

        raise RuntimeError(
            "W2.6 benchmark guard FAILED. "
            "No FS4 candidate reproduced the known fold-safe "
            f"benchmark within ±{W26_TOLERANCE}. "
            f"Top candidates:\n"
            + json.dumps(
                preview,
                indent=2,
                default=str,
            )
        )

    meta = dict(best["meta"])

    meta.update(
        {
            "source": str(best["path"].resolve()),
            "replace_macro_auroc": best["replace_auc"],
            "fixed50_macro_auroc": best["fixed_auc"],
            "expected_fixed50_macro_auroc": best["expected"],
            "benchmark_distance": best["distance"],
            "candidate_count": len(candidates),
        }
    )

    return (
        best["wide4"],
        best["fixed50"],
        meta,
    )


# =============================================================================
# REPORT STATE DISCOVERY — EXPLICIT STATES ONLY
# =============================================================================


def normalize_state(
    value: object,
) -> Optional[str]:
    if pd.isna(value):
        return None

    x = norm(value)

    present = {
        "p",
        "present",
        "positive",
        "abnormal",
    }

    absent = {
        "a",
        "absent",
        "negative",
        "normal",
    }

    uncertain = {
        "u",
        "uncertain",
        "indeterminate",
        "equivocal",
    }

    not_addressed = {
        "n",
        "notaddressed",
        "notmentioned",
        "unaddressed",
    }

    if x in present:
        return "P"

    if x in absent:
        return "A"

    if x in uncertain:
        return "U"

    if x in not_addressed:
        return "N"

    return None


def try_extract_state_table(
    path: Path,
    gold: pd.DataFrame,
) -> Optional[Tuple[pd.DataFrame, dict]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    uid_col = find_uid_column(df)
    label_col = find_label_column(df)

    if uid_col is None or label_col is None:
        return None

    tmp = df.copy()

    tmp["_Label"] = tmp[label_col].map(canonical_label)

    tmp = tmp[tmp["_Label"].isin(LABELS)].copy()

    if len(tmp) < 100:
        return None

    state_candidates = []

    for c in tmp.columns:
        if c in {
            uid_col,
            label_col,
            "_Label",
        }:
            continue

        mapped = tmp[c].map(normalize_state)

        coverage = float(mapped.notna().mean())

        unique = set(mapped.dropna())

        if coverage < 0.70 or len(unique) < 2:
            continue

        name = norm(c)

        score = int(coverage * 10)

        if "final" in name:
            score += 5

        if "fused" in name:
            score += 5

        if "state" in name:
            score += 4

        if "assertion" in name:
            score += 3

        state_candidates.append(
            (
                score,
                c,
                mapped,
            )
        )

    if not state_candidates:
        return None

    state_candidates.sort(key=lambda x: -x[0])

    _, state_col, mapped = state_candidates[0]

    tmp["_State"] = mapped

    tmp[uid_col] = tmp[uid_col].astype(str)

    tmp = tmp[tmp["_State"].notna()]

    wide = tmp.pivot_table(
        index=uid_col,
        columns="_Label",
        values="_State",
        aggfunc="first",
    )

    expected_uids = set(gold[UID].astype(str))

    overlap = len(expected_uids & set(wide.index.astype(str))) / len(expected_uids)

    if overlap < 0.80:
        return None

    wide = wide.reindex(gold[UID].astype(str))

    return wide, {
        "source": str(path.resolve()),
        "state_column": state_col,
        "coverage": float(wide.notna().sum().sum() / (len(gold) * len(LABELS))),
    }


def discover_report_states(
    paths: Paths,
    gold: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], dict]:
    candidates: List[Path] = []

    if paths.w2_root:
        root = Path(paths.w2_root)

        preferred = find_file_recursive(
            root,
            "04_gold_structured_report_features.csv",
        )

        if preferred:
            candidates.append(preferred)

        candidates.extend(root.rglob("*.csv"))

    results_root = Path(paths.project_root) / "output" / "results"

    if results_root.exists():
        for child in results_root.iterdir():
            if child.is_dir() and "w2_5" in child.name.lower():
                candidates.extend(child.rglob("*.csv"))

    seen = set()

    ordered = []

    for p in candidates:
        rp = str(p.resolve())

        if rp not in seen:
            seen.add(rp)
            ordered.append(p)

    for path in ordered:
        result = try_extract_state_table(
            path,
            gold,
        )

        if result is not None:
            return result

    return None, {
        "available": False,
        "reason": (
            "No explicit P/A/U/N-style state artifact "
            "was structurally identified. "
            "States were NOT inferred from raw report text."
        ),
    }


# =============================================================================
# W2.6-P PRODUCTION POLICY
# =============================================================================


def find_any_named(
    root: Path,
    filenames: Sequence[str],
) -> Optional[Path]:
    for filename in filenames:
        p = find_file_recursive(
            root,
            filename,
        )

        if p is not None:
            return p

    return None


def load_exact_wide(
    path: Path,
    expected_rows: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(path)

    uid_col = find_uid_column(df)

    if uid_col is None:
        raise RuntimeError(f"{path}: UID column missing")

    missing = [label for label in LABELS if label not in df.columns]

    if missing:
        raise RuntimeError(f"{path}: missing label columns {missing}")

    if expected_rows is not None and len(df) != expected_rows:
        raise RuntimeError(
            f"{path}: expected {expected_rows} rows, " f"found {len(df)}"
        )

    out = df[[uid_col, *LABELS]].copy()

    out.rename(
        columns={uid_col: UID},
        inplace=True,
    )

    out[UID] = out[UID].astype(str)

    if out[UID].duplicated().any():
        raise RuntimeError(f"{path}: duplicate UIDs")

    return out.set_index(UID)


def load_production_policy(
    paths: Paths,
) -> Tuple[
    Optional[pd.DataFrame],
    dict,
]:
    if paths.w26p_fast_root is None:
        return None, {
            "available": False,
            "reason": "rsna_w2_6p_fast root not found",
        }

    root = Path(paths.w26p_fast_root)

    probability_file = find_any_named(
        root,
        [
            "16_final_hybrid_probabilities_wide.csv",
            "07_final_hybrid_probabilities_wide.csv",
        ],
    )

    weight_file = find_any_named(
        root,
        [
            "17_recommended_teacher_weights_wide.csv",
            "08_recommended_teacher_weights_wide.csv",
        ],
    )

    mask_file = find_any_named(
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
        return None, {
            "available": False,
            "reason": (
                "Could not locate final probability/weight/mask "
                "wide files in W2.6-P FAST."
            ),
            "probability_file": str(probability_file) if probability_file else None,
            "weight_file": str(weight_file) if weight_file else None,
            "mask_file": str(mask_file) if mask_file else None,
        }

    probs = load_exact_wide(
        probability_file,
        EXPECTED_UNLABELED,
    )

    weights = load_exact_wide(
        weight_file,
        EXPECTED_UNLABELED,
    )

    masks_raw = load_exact_wide(
        mask_file,
        EXPECTED_UNLABELED,
    )

    if not (set(probs.index) == set(weights.index) == set(masks_raw.index)):
        raise RuntimeError("Production probability/weight/mask UID sets differ.")

    masks = masks_raw.copy()

    for label in LABELS:
        masks[label] = parse_bool_series(masks_raw[label])

    weights = weights.reindex(probs.index)

    masks = masks.reindex(probs.index)

    rows = []

    selected_total = 0

    for label in LABELS:
        p = pd.to_numeric(
            probs[label],
            errors="raise",
        ).to_numpy(dtype=float)

        w = pd.to_numeric(
            weights[label],
            errors="raise",
        ).to_numpy(dtype=float)

        m = masks[label].to_numpy(dtype=bool)

        selected_total += int(m.sum())

        selected_weights = (
            w[m]
            if m.any()
            else np.asarray(
                [],
                dtype=float,
            )
        )

        rows.append(
            {
                "Label": label,
                "Studies": len(p),
                "Selected": int(m.sum()),
                "Coverage": float(m.mean()),
                "MeanProbability": float(p.mean()),
                "StdProbability": float(p.std()),
                "P_lt_0.10": float((p < 0.10).mean()),
                "P_gt_0.90": float((p > 0.90).mean()),
                "MeanSelectedWeight": (
                    float(selected_weights.mean()) if len(selected_weights) else np.nan
                ),
                "MinSelectedWeight": (
                    float(selected_weights.min()) if len(selected_weights) else np.nan
                ),
            }
        )

    policy = pd.DataFrame(rows)

    meta = {
        "available": True,
        "probability_file": str(probability_file.resolve()),
        "weight_file": str(weight_file.resolve()),
        "mask_file": str(mask_file.resolve()),
        "selected_cells": selected_total,
        "expected_selected_cells": EXPECTED_SELECTED_CELLS,
    }

    if selected_total != EXPECTED_SELECTED_CELLS:
        raise RuntimeError(
            "Production mask selected-cell count mismatch: "
            f"{selected_total} vs expected "
            f"{EXPECTED_SELECTED_CELLS}"
        )

    return policy, meta


# =============================================================================
# GOLD ERROR TABLE
# =============================================================================


def classification_name(
    y: int,
    p: float,
) -> str:
    pred = int(p >= 0.5)

    if y == 1 and pred == 1:
        return "TP"

    if y == 0 and pred == 0:
        return "TN"

    if y == 0 and pred == 1:
        return "FP"

    return "FN"


def error_score(
    y: int,
    p: float,
) -> float:
    return 1.0 - p if y == 1 else p


def mismatch_category(
    state: Optional[str],
    gold: int,
) -> Optional[str]:
    if state is None:
        return None

    if state == "P":
        return "REPORT_POS_GOLD_POS" if gold == 1 else "REPORT_POS_GOLD_NEG"

    if state == "A":
        return "REPORT_NEG_GOLD_POS" if gold == 1 else "REPORT_NEG_GOLD_NEG"

    if state == "U":
        return "UNCERTAIN_GOLD_POS" if gold == 1 else "UNCERTAIN_GOLD_NEG"

    if state == "N":
        return "NOT_ADDRESSED_GOLD_POS" if gold == 1 else "NOT_ADDRESSED_GOLD_NEG"

    return None


def build_error_table(
    gold: pd.DataFrame,
    w23: pd.DataFrame,
    w26: pd.DataFrame,
    states: Optional[pd.DataFrame],
) -> pd.DataFrame:
    gold_by_uid = gold.set_index(UID)

    rows = []

    for uid in gold[UID].astype(str):
        report = str(
            gold_by_uid.loc[
                uid,
                REPORT,
            ]
        )

        for label in LABELS:
            y = int(
                gold_by_uid.loc[
                    uid,
                    label,
                ]
            )

            p23 = float(
                w23.loc[
                    uid,
                    label,
                ]
            )

            p26 = float(
                w26.loc[
                    uid,
                    label,
                ]
            )

            e23 = error_score(
                y,
                p23,
            )

            e26 = error_score(
                y,
                p26,
            )

            state = None

            if states is not None and uid in states.index and label in states.columns:
                value = states.loc[
                    uid,
                    label,
                ]

                if not pd.isna(value):
                    state = str(value)

            rows.append(
                {
                    UID: uid,
                    "Label": label,
                    "Gold": y,
                    "ReportState": state,
                    "ReportGoldCategory": mismatch_category(
                        state,
                        y,
                    ),
                    "W23Probability": p23,
                    "W26Probability": p26,
                    "W23Class": classification_name(
                        y,
                        p23,
                    ),
                    "W26Class": classification_name(
                        y,
                        p26,
                    ),
                    "W23ErrorScore": e23,
                    "W26ErrorScore": e26,
                    "W26MinusW23Probability": p26 - p23,
                    "ErrorImprovement": e23 - e26,
                    "W26HighConfidenceError": bool(e26 >= HIGH_CONFIDENCE_ERROR),
                    REPORT: report,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# REPORT-STATE SUMMARY
# =============================================================================


def build_state_summary(
    errors: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    x = errors[errors["ReportState"].notna()].copy()

    if x.empty:
        return None

    rows = []

    for (
        label,
        state,
    ), group in x.groupby(
        [
            "Label",
            "ReportState",
        ],
        dropna=False,
    ):
        rows.append(
            {
                "Label": label,
                "ReportState": state,
                "N": len(group),
                "GoldPositiveRate": float(group["Gold"].mean()),
                "MeanW23Probability": float(group["W23Probability"].mean()),
                "MeanW26Probability": float(group["W26Probability"].mean()),
                "W26HighConfidenceErrors": int(group["W26HighConfidenceError"].sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PRIORITY FINDINGS
# =============================================================================


def build_priority_findings(
    metrics: pd.DataFrame,
    errors: pd.DataFrame,
    state_summary: Optional[pd.DataFrame],
    policy: Optional[pd.DataFrame],
) -> pd.DataFrame:
    m23 = metrics[metrics["Model"] == "W2.3"].set_index("Label")

    m26 = metrics[metrics["Model"] == "W2.6_fixed50"].set_index("Label")

    policy_by_label = policy.set_index("Label") if policy is not None else None

    rows = []

    for label in LABELS:
        auc23 = float(
            m23.loc[
                label,
                "AUROC",
            ]
        )

        auc26 = float(
            m26.loc[
                label,
                "AUROC",
            ]
        )

        ap23 = float(
            m23.loc[
                label,
                "AP",
            ]
        )

        ap26 = float(
            m26.loc[
                label,
                "AP",
            ]
        )

        b23 = float(
            m23.loc[
                label,
                "Brier",
            ]
        )

        b26 = float(
            m26.loc[
                label,
                "Brier",
            ]
        )

        label_errors = errors[errors["Label"] == label]

        high_errors = int(label_errors["W26HighConfidenceError"].sum())

        not_addressed_n = np.nan
        not_addressed_positive_rate = np.nan

        if state_summary is not None:
            tmp = state_summary[
                (state_summary["Label"] == label)
                & (state_summary["ReportState"] == "N")
            ]

            if not tmp.empty:
                not_addressed_n = int(tmp.iloc[0]["N"])

                not_addressed_positive_rate = float(tmp.iloc[0]["GoldPositiveRate"])

        coverage = np.nan
        mean_weight = np.nan

        if policy_by_label is not None and label in policy_by_label.index:
            coverage = float(
                policy_by_label.loc[
                    label,
                    "Coverage",
                ]
            )

            mean_weight = float(
                policy_by_label.loc[
                    label,
                    "MeanSelectedWeight",
                ]
            )

        flags = []

        if auc26 < 0.72:
            flags.append("LOW_GOLD_AUC")

        if not label in FS4 and auc23 < 0.72:
            flags.append("UNIMPROVED_WEAK_BASE_LABEL")

        if auc26 < auc23 - 0.01:
            flags.append("W26_AUC_REGRESSION")

        if b26 > b23 + 0.01:
            flags.append("CALIBRATION_REGRESSION")

        if high_errors >= 2:
            flags.append("MULTIPLE_HIGH_CONF_ERRORS")

        if np.isfinite(coverage) and coverage >= 0.95 and auc26 < 0.78:
            flags.append("BROAD_PRODUCTION_MASK_WITH_MODEST_GOLD_AUC")

        if (
            np.isfinite(not_addressed_positive_rate)
            and not_addressed_n >= 3
            and not_addressed_positive_rate >= 0.30
        ):
            flags.append("NOT_ADDRESSED_OFTEN_GOLD_POSITIVE")

        priority_score = 0

        priority_score += int(auc26 < 0.72) * 4

        priority_score += int(not label in FS4 and auc23 < 0.72) * 3

        priority_score += int(auc26 < auc23 - 0.01) * 3

        priority_score += min(
            high_errors,
            4,
        )

        priority_score += (
            int(np.isfinite(coverage) and coverage >= 0.95 and auc26 < 0.78) * 2
        )

        if priority_score >= 5:
            action = "REVIEW_FIRST"

        elif flags:
            action = "REVIEW"

        else:
            action = "KEEP"

        rows.append(
            {
                "Label": label,
                "ChangedByW26": label in FS4,
                "W23_AUROC": auc23,
                "W26_AUROC": auc26,
                "Delta_AUROC": auc26 - auc23,
                "W23_AP": ap23,
                "W26_AP": ap26,
                "Delta_AP": ap26 - ap23,
                "W23_Brier": b23,
                "W26_Brier": b26,
                "Delta_Brier": b26 - b23,
                "HighConfidenceErrors": high_errors,
                "NotAddressedN": not_addressed_n,
                "NotAddressedGoldPositiveRate": not_addressed_positive_rate,
                "ProductionCoverage": coverage,
                "MeanSelectedWeight": mean_weight,
                "PriorityScore": priority_score,
                "Action": action,
                "Flags": ";".join(flags),
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
# STATUS
# =============================================================================


def status(
    accelerator: str,
) -> dict:
    paths = Paths.discover()

    gold = load_gold(paths)

    result = {
        "script_version": SCRIPT_VERSION,
        "accelerator_requested": accelerator,
        "effective_backend": "cpu",
        "reason": "Teacher audit is CPU/tabular only.",
        "accelerator_env_variables_required": False,
        "paths": asdict(paths),
        "gold_studies": len(gold),
        "gold_positive_counts": {label: int(gold[label].sum()) for label in LABELS},
        "expected_benchmarks": {
            "w23_macro_auroc": EXPECTED_W23_AUC,
            "w26_original_fixed50_macro_auroc": EXPECTED_W26_ORIGINAL_FIXED50_AUC,
            "w26_fast_fixed50_macro_auroc": EXPECTED_W26_FAST_FIXED50_AUC,
        },
        "scope": {
            "gpu_used": False,
            "tpu_used": False,
            "dicom_used": False,
            "curia_used": False,
            "training_performed": False,
            "pilkwang_used_for_production": False,
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
# FULL AUDIT
# =============================================================================


def audit(
    accelerator: str,
) -> dict:
    paths = Paths.discover()

    output = Path(paths.output_root)

    gold = load_gold(paths)

    # -----------------------------------------------------------------
    # 1. Fold-safe gold teacher reconstruction
    # -----------------------------------------------------------------

    log("=" * 96)
    log("W37 TEACHER ERROR AUDIT v1")
    log("=" * 96)

    log("")
    log("[1/6] Loading exact W2.3 fold-safe gold predictions...")

    w23, w23_meta = load_w23_gold_predictions(
        paths,
        gold,
    )

    log(f"W2.3 reconstructed macro AUROC = " f"{w23_meta['macro_auroc']:.6f}")

    log("")
    log("[2/6] Discovering fold-safe W2.6 FS4 gold predictions...")

    (
        fs4,
        w26,
        w26_meta,
    ) = discover_fs4_gold_predictions(
        paths,
        gold,
        w23,
    )

    log(
        f"W2.6 fixed50 reconstructed macro AUROC = "
        f"{w26_meta['fixed50_macro_auroc']:.6f}"
    )

    log(f"W2.6 source = {w26_meta['source']}")

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    metrics = pd.concat(
        [
            per_label_metrics(
                gold,
                w23,
                "W2.3",
            ),
            per_label_metrics(
                gold,
                w26,
                "W2.6_fixed50",
            ),
        ],
        ignore_index=True,
    )

    metrics.to_csv(
        output / "01_gold_teacher_metrics.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 2. Explicit report states
    # -----------------------------------------------------------------

    log("")
    log("[3/6] Looking for explicit report-state artifact...")

    states, state_meta = discover_report_states(
        paths,
        gold,
    )

    if states is None:
        log("No explicit P/A/U/N artifact found; " "state analysis will be skipped.")

    else:
        log(
            f"Explicit report states found: "
            f"{state_meta['source']} "
            f"(coverage={state_meta['coverage']:.3f})"
        )

    # -----------------------------------------------------------------
    # 3. Error table
    # -----------------------------------------------------------------

    errors = build_error_table(
        gold,
        w23,
        w26,
        states,
    )

    errors.to_csv(
        output / "02_gold_teacher_errors.csv",
        index=False,
    )

    top_errors = (
        errors.sort_values(
            [
                "W26ErrorScore",
                "Label",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .groupby(
            "Label",
            sort=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    top_errors.to_csv(
        output / "03_top_error_cases.csv",
        index=False,
    )

    state_summary = build_state_summary(errors)

    if state_summary is not None:
        state_summary.to_csv(
            output / "04_report_state_vs_gold.csv",
            index=False,
        )

    # -----------------------------------------------------------------
    # 4. Production mask / weights
    # -----------------------------------------------------------------

    log("")
    log("[4/6] Auditing W2.6-P FAST production mask/weights...")

    policy, policy_meta = load_production_policy(paths)

    if policy is not None:
        policy.to_csv(
            output / "05_production_policy_audit.csv",
            index=False,
        )

        log(f"Production selected cells = " f"{policy_meta['selected_cells']}")

    else:
        log(
            "Production policy artifacts unavailable; " "gold audit can still continue."
        )

    # -----------------------------------------------------------------
    # 5. Priority findings
    # -----------------------------------------------------------------

    log("")
    log("[5/6] Building label-priority findings...")

    priorities = build_priority_findings(
        metrics,
        errors,
        state_summary,
        policy,
    )

    priorities.to_csv(
        output / "06_priority_findings.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------

    macro23 = float(metrics[metrics["Model"] == "W2.3"]["AUROC"].mean())

    macro26 = float(metrics[metrics["Model"] == "W2.6_fixed50"]["AUROC"].mean())

    brier23 = float(metrics[metrics["Model"] == "W2.3"]["Brier"].mean())

    brier26 = float(metrics[metrics["Model"] == "W2.6_fixed50"]["Brier"].mean())

    high_conf_errors = errors[errors["W26HighConfidenceError"]]

    review_first = priorities[priorities["Action"] == "REVIEW_FIRST"]["Label"].tolist()

    review = priorities[priorities["Action"] == "REVIEW"]["Label"].tolist()

    summary = {
        "script_version": SCRIPT_VERSION,
        "status": "AUDIT_COMPLETE",
        "accelerator_requested": accelerator,
        "effective_backend": "cpu",
        "gold_validation": {
            "w23_macro_auroc": macro23,
            "w26_fixed50_macro_auroc": macro26,
            "delta_macro_auroc": macro26 - macro23,
            "w23_macro_brier": brier23,
            "w26_fixed50_macro_brier": brier26,
            "delta_macro_brier": brier26 - brier23,
            "w23_source": w23_meta,
            "w26_source": w26_meta,
            "high_confidence_error_cells": len(high_conf_errors),
        },
        "report_state_audit": state_meta,
        "production_policy_audit": policy_meta,
        "priority": {
            "review_first": review_first,
            "review": review,
        },
        "important_interpretation": [
            (
                "Gold metrics are reconstructed only from "
                "fold-safe held-out prediction artifacts."
            ),
            (
                "W2.6-P all-58 production pseudo labels are "
                "not treated as pristine gold validation."
            ),
            (
                "P/A/U/N states are used only when an explicit "
                "state artifact is structurally identified."
            ),
            (
                "0.5 threshold error classes are diagnostic only; "
                "competition evaluation is AUROC-based."
            ),
            (
                "This audit does not authorize W8 automatically. "
                "Review priority findings first."
            ),
        ],
        "next_step": (
            "Review 01_gold_teacher_metrics.csv, "
            "03_top_error_cases.csv, "
            "05_production_policy_audit.csv (if available), "
            "and 06_priority_findings.csv before deciding "
            "whether a single controlled W8 teacher change "
            "is justified."
        ),
    }

    safe_json_dump(
        summary,
        output / "00_summary.json",
    )

    log("")
    log("[6/6] Complete.")

    log("")
    log("=" * 96)
    log("AUDIT SUMMARY")
    log("=" * 96)

    log(f"W2.3 macro AUROC     : {macro23:.6f}")

    log(f"W2.6 fixed50 AUROC   : {macro26:.6f}")

    log(f"Delta                : " f"{macro26 - macro23:+.6f}")

    log(f"W2.3 macro Brier     : {brier23:.6f}")

    log(f"W2.6 macro Brier     : {brier26:.6f}")

    log(f"High-confidence errors: " f"{len(high_conf_errors)}")

    log(f"REVIEW_FIRST labels  : {review_first}")

    log(f"REVIEW labels        : {review}")

    log("")
    log(f"Results: {output}")

    log("")
    log("STOP HERE. Do not train W8 yet.")

    return summary


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("RSNA teacher error audit v1 — CPU only")
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
        help=(
            "Accepted for interface consistency. "
            "This audit always executes on CPU and does "
            "not use GPU/TPU."
        ),
    )

    args = parser.parse_args()

    if args.mode == "status":
        status(args.accelerator)

    elif args.mode == "audit":
        audit(args.accelerator)


if __name__ == "__main__":
    main()
