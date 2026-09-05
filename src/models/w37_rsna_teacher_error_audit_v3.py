#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSNA Knee Abnormality Detection
W37 Teacher Error Audit v3

CPU-only standalone diagnostic.

Purpose
-------
1. Reconstruct W2.3 fold-safe 58-gold OOF predictions.
2. Reconstruct W2.6 FS4/fixed50 fold-safe 58-gold predictions.
3. Independently recompute AUROC / AP / Brier.
4. Audit high-confidence teacher errors against raw reports.
5. Use explicit P/A/U/N report states only when a real state artifact exists.
6. Audit W2.6-P FAST production probabilities / masks / weights separately.
7. Produce label-priority findings before any W8 experiment.

Important
---------
- No GPU/TPU is used.
- No DICOM is read.
- No Curia model is loaded.
- No project Python script is imported or executed.
- W2.6-P all-58 production outputs are NOT treated as pristine OOF evidence.
- Gold evaluation uses fold-safe held-out predictions only.

Version history
---------------
v1:
    Initial audit.

v2:
    Fixed W2.3/W2.6 fold-wise held-out reconstruction.

v3:
    Fixed W2.6 provenance logging: sources vs source.
    Fixed stale v1 banner/version/output directory.
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

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# =============================================================================
# VERSION
# =============================================================================

SCRIPT_VERSION = "rsna_teacher_error_audit_v3"
DISPLAY_VERSION = "W37 TEACHER ERROR AUDIT v3"
OUTPUT_DIR_NAME = "rsna_teacher_error_audit_v3"


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

# Current optimized W2.6-P FAST validation
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


def canonical_label(
    value: object,
) -> Optional[str]:

    key = norm(value)

    if key in LABEL_BY_NORM:
        return LABEL_BY_NORM[key]

    return LABEL_ALIASES.get(key)


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


def find_uid_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return find_column(
        df.columns,
        [
            UID,
            "StudyUID",
            "study_uid",
            "UID",
        ],
    )


def find_label_column(
    df: pd.DataFrame,
) -> Optional[str]:

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

        if len(values) and values.notna().mean() >= 0.60:
            return explicit

    uid_col = find_uid_column(df)

    for c in df.columns:

        if c == uid_col:
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

    result: List[Tuple[int, str]] = []

    for column in df.columns:

        if column in exclude_set:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = values.notna()

        if valid.sum() < max(
            10,
            int(0.50 * len(df)),
        ):
            continue

        vv = values[valid].to_numpy(
            dtype=float,
        )

        if not np.isfinite(vv).all():
            continue

        if vv.min() < -1e-7 or vv.max() > 1.0000001:
            continue

        name = norm(column)

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

        result.append(
            (
                score,
                column,
            )
        )

    return sorted(
        result,
        key=lambda x: (
            -x[0],
            x[1],
        ),
    )


def parse_bool_series(
    series: pd.Series,
) -> np.ndarray:

    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(
            dtype=bool,
        )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():

        return (
            numeric.to_numpy(
                dtype=float,
            )
            > 0.5
        )

    text = series.astype(str).str.strip().str.lower()

    return text.isin(
        [
            "true",
            "1",
            "yes",
            "y",
            "t",
        ]
    ).to_numpy(
        dtype=bool,
    )


# =============================================================================
# PATH DISCOVERY
# =============================================================================


def discover_project_root() -> Path:

    script_dir = Path(__file__).resolve().parent

    for path in [
        script_dir,
        *script_dir.parents,
    ]:

        if (path / "input" / "train.csv").exists():

            return path

    candidate = (script_dir / ".." / "..").resolve()

    if (candidate / "input" / "train.csv").exists():

        return candidate

    raise FileNotFoundError(
        "Could not discover PROJECT_ROOT " "containing input/train.csv"
    )


def first_existing(
    candidates: Iterable[Path],
) -> Optional[Path]:

    for path in candidates:

        if path.exists():
            return path.resolve()

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
    def discover(
        cls,
    ) -> "Paths":

        root = discover_project_root()

        results = root / "output" / "results"

        train_csv = root / "input" / "train.csv"

        output_root = results / OUTPUT_DIR_NAME

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        def find_stage(
            exact_names: Sequence[str],
        ) -> Optional[Path]:

            found = first_existing([results / name for name in exact_names])

            if found:
                return found

            if not results.exists():
                return None

            lower_names = [x.lower() for x in exact_names]

            matches = []

            for child in results.iterdir():

                if not child.is_dir():
                    continue

                name = child.name.lower()

                if any(
                    expected in name or name in expected for expected in lower_names
                ):

                    matches.append(child)

            if not matches:
                return None

            return sorted(
                matches,
                key=lambda p: len(str(p)),
            )[0].resolve()

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
            w2_root=str(w2) if w2 else None,
            w23_root=str(w23) if w23 else None,
            w26_root=str(w26) if w26 else None,
            w26p_fast_root=str(w26p_fast) if w26p_fast else None,
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

    missing = [column for column in required if column not in df.columns]

    if missing:

        raise RuntimeError(f"train.csv missing columns: " f"{missing}")

    df[UID] = df[UID].astype(str)

    gold = df[df[LABELS].notna().all(axis=1)].copy()

    if len(gold) != EXPECTED_GOLD:

        raise RuntimeError(
            f"Expected {EXPECTED_GOLD} " f"gold studies, " f"found {len(gold)}"
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

    truth = gold.set_index(UID).loc[
        predictions.index,
        LABELS,
    ]

    rows = []

    for label in LABELS:

        y = truth[label].to_numpy(
            dtype=float,
        )

        p = predictions[label].to_numpy(
            dtype=float,
        )

        if not np.isfinite(p).all():

            raise RuntimeError(f"{model_name}/{label}: " f"non-finite probabilities")

        if (p < 0).any() or (p > 1).any():

            raise RuntimeError(f"{model_name}/{label}: " f"probability outside [0,1]")

        auc = (
            float(
                roc_auc_score(
                    y,
                    p,
                )
            )
            if len(np.unique(y)) == 2
            else np.nan
        )

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

    table = per_label_metrics(
        gold,
        predictions,
        "_temporary_",
    )

    return float(table["AUROC"].mean())


# =============================================================================
# PREDICTION TABLE EXTRACTION
# =============================================================================


def extract_long_prediction(
    df: pd.DataFrame,
    required_labels: Sequence[str],
    source_name: str,
    preferred_tokens: Sequence[str] = (),
) -> Tuple[
    pd.DataFrame,
    dict,
]:

    uid_col = find_uid_column(df)

    label_col = find_label_column(df)

    if uid_col is None or label_col is None:

        raise ValueError("not a recognizable " "long prediction table")

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

    for score, column in candidates:

        name = norm(column)

        for token in preferred_tokens:

            if norm(token) in name:
                score += 6

        rescored.append(
            (
                score,
                column,
            )
        )

    rescored.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    if not rescored:

        raise ValueError(f"{source_name}: " f"no probability column")

    best_score = rescored[0][0]

    tied = [x for x in rescored if x[0] == best_score]

    if len(tied) > 1 and best_score <= 0:

        raise ValueError(
            f"{source_name}: "
            f"ambiguous probability columns "
            f"{[x[1] for x in tied]}"
        )

    probability_col = rescored[0][1]

    tmp[uid_col] = tmp[uid_col].astype(str)

    tmp[probability_col] = pd.to_numeric(
        tmp[probability_col],
        errors="coerce",
    )

    if (
        tmp[
            [
                uid_col,
                "_CanonicalLabel",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(f"{source_name}: " f"duplicate UID/Label rows")

    wide = tmp.pivot(
        index=uid_col,
        columns="_CanonicalLabel",
        values=probability_col,
    )

    missing = [label for label in required_labels if label not in wide.columns]

    if missing:

        raise ValueError(f"{source_name}: " f"missing labels {missing}")

    return (
        wide[list(required_labels)],
        {
            "format": "long",
            "uid_column": uid_col,
            "label_column": label_col,
            "probability_column": probability_col,
        },
    )


def wide_label_column_candidates(
    df: pd.DataFrame,
    label: str,
) -> List[Tuple[int, str]]:

    target = norm(label)

    result = []

    for column in df.columns:

        name = norm(column)

        if target not in name:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.notna().mean() < 0.90:
            continue

        vv = values.dropna().to_numpy(dtype=float)

        if not len(vv):
            continue

        if vv.min() < -1e-7 or vv.max() > 1.0000001:
            continue

        score = 0

        if name == target:
            score += 2

        if "probability" in name:
            score += 10
        elif "prob" in name:
            score += 8

        if "prediction" in name:
            score += 7
        elif "pred" in name:
            score += 5

        if "gold" in name or "true" in name:
            score -= 20

        result.append(
            (
                score,
                column,
            )
        )

    return sorted(
        result,
        key=lambda x: (
            -x[0],
            x[1],
        ),
    )


def extract_wide_prediction(
    df: pd.DataFrame,
    required_labels: Sequence[str],
    source_name: str,
) -> Tuple[
    pd.DataFrame,
    dict,
]:

    uid_col = find_uid_column(df)

    if uid_col is None:

        raise ValueError(f"{source_name}: " f"no UID column")

    selected = {}

    for label in required_labels:

        candidates = wide_label_column_candidates(
            df,
            label,
        )

        if not candidates:

            raise ValueError(
                f"{source_name}: "
                f"cannot find wide "
                f"probability column "
                f"for {label}"
            )

        selected[label] = candidates[0][1]

    wide = pd.DataFrame(
        {
            label: pd.to_numeric(
                df[column],
                errors="coerce",
            ).to_numpy()
            for label, column in selected.items()
        },
        index=df[uid_col].astype(str).to_numpy(),
    )

    return (
        wide,
        {
            "format": "wide",
            "uid_column": uid_col,
            "columns": selected,
        },
    )


def extract_prediction_table(
    df: pd.DataFrame,
    required_labels: Sequence[str],
    source_name: str,
    preferred_tokens: Sequence[str] = (),
) -> Tuple[
    pd.DataFrame,
    dict,
]:

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
        f"{source_name}: " f"unsupported prediction table. " + " | ".join(errors)
    )


# =============================================================================
# FOLD FILE HELPERS
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


def find_files_recursive(
    root: Path,
    filename: str,
) -> List[Path]:

    found: Dict[
        str,
        Path,
    ] = {}

    for candidate in [
        root / filename,
        root / "results" / filename,
    ]:

        if candidate.exists():

            found[str(candidate.resolve())] = candidate.resolve()

    for candidate in root.rglob(filename):

        found[str(candidate.resolve())] = candidate.resolve()

    return sorted(
        found.values(),
        key=lambda p: str(p),
    )


def restrict_prediction_to_gold_overlap(
    wide: pd.DataFrame,
    gold: pd.DataFrame,
) -> pd.DataFrame:

    wide = wide.copy()

    wide.index = wide.index.astype(str)

    gold_order = list(gold[UID].astype(str))

    overlap = [uid for uid in gold_order if uid in wide.index]

    if not overlap:

        return wide.iloc[0:0].copy()

    result = wide.loc[overlap].copy()

    if result.index.duplicated().any():

        raise RuntimeError("Duplicate StudyInstanceUID " "inside prediction artifact.")

    return result


def combine_fold_prediction_tables(
    pieces: Sequence[
        Tuple[
            Path,
            pd.DataFrame,
        ]
    ],
    gold: pd.DataFrame,
    source_name: str,
) -> Tuple[
    pd.DataFrame,
    List[str],
]:

    if not pieces:

        raise RuntimeError(f"{source_name}: " f"no prediction pieces " f"supplied.")

    gold_order = list(gold[UID].astype(str))

    frames = []

    sources = []

    for path, frame in pieces:

        if frame.empty:
            continue

        x = frame.copy()

        x.index = x.index.astype(str)

        frames.append(x)

        sources.append(str(path.resolve()))

    if not frames:

        raise RuntimeError(f"{source_name}: " f"no gold prediction rows found.")

    combined = pd.concat(
        frames,
        axis=0,
    )

    if combined.index.duplicated().any():

        deduped = []

        for uid, group in combined.groupby(
            level=0,
            sort=False,
        ):

            if len(group) == 1:

                deduped.append(group)

                continue

            values = group.to_numpy(dtype=float)

            reference = values[0]

            if not np.allclose(
                values,
                reference[None, :],
                atol=1e-8,
                rtol=1e-7,
                equal_nan=True,
            ):

                raise RuntimeError(
                    f"{source_name}: "
                    f"conflicting duplicate "
                    f"predictions for UID {uid}"
                )

            deduped.append(group.iloc[[0]])

        combined = pd.concat(
            deduped,
            axis=0,
        )

    expected = set(gold_order)

    actual = set(combined.index.astype(str))

    missing = expected - actual

    if missing:

        raise RuntimeError(
            f"{source_name}: "
            f"after combining fold artifacts, "
            f"still missing {len(missing)} "
            f"gold UIDs. "
            f"Example={sorted(missing)[:5]}"
        )

    combined = combined.loc[gold_order]

    if len(combined) != EXPECTED_GOLD:

        raise RuntimeError(
            f"{source_name}: "
            f"expected {EXPECTED_GOLD} "
            f"combined OOF rows, "
            f"got {len(combined)}"
        )

    return (
        combined,
        sources,
    )


# =============================================================================
# W2.3 GOLD OOF
# =============================================================================


def load_w23_gold_predictions(
    paths: Paths,
    gold: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    dict,
]:

    if paths.w23_root is None:

        raise FileNotFoundError("rsna_w2_3 root not found")

    root = Path(paths.w23_root)

    sources = find_files_recursive(
        root,
        "heldout_gold_stage_b_predictions.csv",
    )

    if not sources:

        raise FileNotFoundError(
            "Could not locate any " "heldout_gold_stage_b_predictions.csv"
        )

    log(f"W2.3 held-out files discovered: " f"{len(sources)}")

    pieces = []

    extraction_meta = []

    for source in sources:

        try:

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

            partial = restrict_prediction_to_gold_overlap(
                wide,
                gold,
            )

            if partial.empty:
                continue

            try:

                shown = source.relative_to(root)

            except Exception:

                shown = source

            log(f"  {shown} " f"-> {len(partial)} gold rows")

            pieces.append(
                (
                    source,
                    partial,
                )
            )

            extraction_meta.append(
                {
                    "source": str(source.resolve()),
                    "rows": len(partial),
                    "extraction": meta,
                }
            )

        except Exception as exc:

            log(f"  skipped " f"{source}: {exc}")

    if not pieces:

        raise RuntimeError("No usable W2.3 " "held-out prediction files.")

    full_candidates = [
        (
            path,
            frame,
        )
        for path, frame in pieces
        if len(frame) == EXPECTED_GOLD
    ]

    if full_candidates:

        source, wide = sorted(
            full_candidates,
            key=lambda x: len(str(x[0])),
        )[0]

        used_sources = [str(source.resolve())]

    else:

        wide, used_sources = combine_fold_prediction_tables(
            pieces,
            gold,
            "W2.3 fold-safe OOF",
        )

    observed_auc = macro_auc(
        gold,
        wide,
    )

    if abs(observed_auc - EXPECTED_W23_AUC) > W23_TOLERANCE:

        raise RuntimeError(
            "W2.3 benchmark guard FAILED. "
            f"Recomputed={observed_auc:.6f}, "
            f"expected≈"
            f"{EXPECTED_W23_AUC:.6f}. "
            f"Sources={used_sources}"
        )

    return (
        wide,
        {
            "format": "concatenated_fold_safe_oof",
            "sources": used_sources,
            "files_used": len(used_sources),
            "rows": len(wide),
            "macro_auroc": observed_auc,
            "all_extraction_attempts": extraction_meta,
        },
    )


# =============================================================================
# W2.6 GOLD FS4 DISCOVERY
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


def discover_fs4_gold_predictions(
    paths: Paths,
    gold: pd.DataFrame,
    w23: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    roots = []

    if paths.w26p_fast_root:

        roots.append(Path(paths.w26p_fast_root))

    if paths.w26_root:

        roots.append(Path(paths.w26_root))

    if not roots:

        raise FileNotFoundError("Neither rsna_w2_6p_fast " "nor rsna_w2_6_v2 was found")

    extracted = []

    gold_uids = set(gold[UID].astype(str))

    for root in roots:

        for path in root.rglob("*.csv"):

            try:

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

                partial = restrict_prediction_to_gold_overlap(
                    wide4,
                    gold,
                )

                if partial.empty:
                    continue

                overlap = set(partial.index.astype(str))

                if not overlap.issubset(gold_uids):
                    continue

                if meta.get("format") == "long":

                    signature = (
                        "long",
                        path.name,
                        meta.get("label_column"),
                        meta.get("probability_column"),
                    )

                else:

                    columns = meta.get(
                        "columns",
                        {},
                    )

                    signature = (
                        "wide",
                        path.name,
                        tuple(
                            sorted(
                                (
                                    str(k),
                                    str(v),
                                )
                                for k, v in columns.items()
                            )
                        ),
                    )

                extracted.append(
                    {
                        "path": path,
                        "wide4": partial,
                        "meta": meta,
                        "signature": signature,
                    }
                )

            except Exception:
                continue

    if not extracted:

        raise RuntimeError(
            "Could not identify any " "gold-overlapping FS4 " "prediction artifacts."
        )

    candidate_sets = []

    # Complete standalone 58-row candidates.
    for item in extracted:

        if len(item["wide4"]) == EXPECTED_GOLD:

            candidate_sets.append(
                {
                    "wide4": item["wide4"],
                    "paths": [item["path"]],
                    "meta": item["meta"],
                }
            )

    # Fold-wise candidates.
    groups: Dict[
        object,
        list,
    ] = {}

    for item in extracted:

        groups.setdefault(
            item["signature"],
            [],
        ).append(item)

    for signature, items in groups.items():

        if len(items) < 2:
            continue

        pieces = [
            (
                item["path"],
                item["wide4"],
            )
            for item in items
        ]

        try:

            combined, used_sources = combine_fold_prediction_tables(
                pieces,
                gold,
                f"W2.6 FS4 candidate " f"{signature}",
            )

        except Exception:
            continue

        candidate_sets.append(
            {
                "wide4": combined,
                "paths": [Path(x) for x in used_sources],
                "meta": {
                    "format": "concatenated_fold_safe_oof",
                    "signature": repr(signature),
                },
            }
        )

    if not candidate_sets:

        raise RuntimeError(
            "FS4 files were found, "
            "but no standalone or "
            "combined candidate covered "
            "all 58 gold studies."
        )

    candidates = []

    for candidate in candidate_sets:

        wide4 = candidate["wide4"]

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

        paths_used = candidate["paths"]

        if any("fast" in str(path).lower() for path in paths_used):

            expected = EXPECTED_W26_FAST_FIXED50_AUC

        else:

            expected = EXPECTED_W26_ORIGINAL_FIXED50_AUC

        distance = abs(fixed_auc - expected)

        structural_score = sum(
            score_fs4_candidate_name(
                Path(path),
                candidate["meta"].get("probability_column"),
            )
            for path in paths_used
        )

        candidates.append(
            {
                "wide4": wide4,
                "replace": replace,
                "fixed50": fixed50,
                "paths": paths_used,
                "meta": candidate["meta"],
                "replace_auc": replace_auc,
                "fixed_auc": fixed_auc,
                "expected": expected,
                "distance": distance,
                "structural_score": structural_score,
            }
        )

    candidates.sort(
        key=lambda x: (
            x["distance"],
            -x["structural_score"],
            len(x["paths"]),
        )
    )

    best = candidates[0]

    if best["distance"] > W26_TOLERANCE:

        preview = []

        for candidate in candidates[:10]:

            preview.append(
                {
                    "paths": [str(path) for path in candidate["paths"]],
                    "fixed_auc": candidate["fixed_auc"],
                    "replace_auc": candidate["replace_auc"],
                    "expected": candidate["expected"],
                    "distance": candidate["distance"],
                    "meta": candidate["meta"],
                }
            )

        raise RuntimeError(
            "W2.6 benchmark guard FAILED. "
            "No candidate reproduced "
            "the known fold-safe benchmark "
            f"within ±{W26_TOLERANCE}.\n"
            + json.dumps(
                preview,
                indent=2,
                default=str,
            )
        )

    meta = dict(best["meta"])

    meta.update(
        {
            "sources": [str(Path(path).resolve()) for path in best["paths"]],
            "files_used": len(best["paths"]),
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
# EXPLICIT REPORT STATES
# =============================================================================


def normalize_state(
    value: object,
) -> Optional[str]:

    if pd.isna(value):
        return None

    value = norm(value)

    if value in {
        "p",
        "present",
        "positive",
        "abnormal",
    }:
        return "P"

    if value in {
        "a",
        "absent",
        "negative",
        "normal",
    }:
        return "A"

    if value in {
        "u",
        "uncertain",
        "indeterminate",
        "equivocal",
    }:
        return "U"

    if value in {
        "n",
        "notaddressed",
        "notmentioned",
        "unaddressed",
    }:
        return "N"

    return None


def try_extract_state_table(
    path: Path,
    gold: pd.DataFrame,
) -> Optional[
    Tuple[
        pd.DataFrame,
        dict,
    ]
]:

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

    for column in tmp.columns:

        if column in {
            uid_col,
            label_col,
            "_Label",
        }:
            continue

        mapped = tmp[column].map(normalize_state)

        coverage = float(mapped.notna().mean())

        unique = set(mapped.dropna())

        if coverage < 0.70 or len(unique) < 2:
            continue

        name = norm(column)

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
                column,
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

    return (
        wide,
        {
            "available": True,
            "source": str(path.resolve()),
            "state_column": state_col,
            "coverage": float(wide.notna().sum().sum() / (len(gold) * len(LABELS))),
        },
    )


def discover_report_states(
    paths: Paths,
    gold: pd.DataFrame,
) -> Tuple[
    Optional[pd.DataFrame],
    dict,
]:

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

    for path in candidates:

        resolved = str(path.resolve())

        if resolved not in seen:

            seen.add(resolved)

            ordered.append(path)

    for path in ordered:

        result = try_extract_state_table(
            path,
            gold,
        )

        if result is not None:
            return result

    return (
        None,
        {
            "available": False,
            "reason": (
                "No explicit P/A/U/N-style state artifact "
                "was structurally identified. "
                "States were NOT inferred "
                "from raw report text."
            ),
        },
    )


# =============================================================================
# PRODUCTION W2.6-P POLICY
# =============================================================================


def find_any_named(
    root: Path,
    filenames: Sequence[str],
) -> Optional[Path]:

    for filename in filenames:

        path = find_file_recursive(
            root,
            filename,
        )

        if path is not None:
            return path

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

        raise RuntimeError(f"{path}: " f"missing label columns {missing}")

    if expected_rows is not None and len(df) != expected_rows:

        raise RuntimeError(
            f"{path}: expected " f"{expected_rows} rows, " f"found {len(df)}"
        )

    out = df[
        [
            uid_col,
            *LABELS,
        ]
    ].copy()

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

        return (
            None,
            {
                "available": False,
                "reason": "rsna_w2_6p_fast root not found",
            },
        )

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

        return (
            None,
            {
                "available": False,
                "reason": (
                    "Could not locate final "
                    "probability/weight/mask "
                    "wide files in W2.6-P FAST."
                ),
                "probability_file": str(probability_file) if probability_file else None,
                "weight_file": str(weight_file) if weight_file else None,
                "mask_file": str(mask_file) if mask_file else None,
            },
        )

    probabilities = load_exact_wide(
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

    if not (set(probabilities.index) == set(weights.index) == set(masks_raw.index)):

        raise RuntimeError("Production probability/weight/mask " "UID sets differ.")

    masks = masks_raw.copy()

    for label in LABELS:

        masks[label] = parse_bool_series(masks_raw[label])

    weights = weights.reindex(probabilities.index)

    masks = masks.reindex(probabilities.index)

    rows = []

    selected_total = 0

    for label in LABELS:

        p = pd.to_numeric(
            probabilities[label],
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
            "Production mask selected-cell "
            "count mismatch: "
            f"{selected_total} "
            f"vs expected "
            f"{EXPECTED_SELECTED_CELLS}"
        )

    return (
        policy,
        meta,
    )


# =============================================================================
# ERROR ANALYSIS
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

    if y == 1:
        return 1.0 - p

    return p


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

            error23 = error_score(
                y,
                p23,
            )

            error26 = error_score(
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
                    "W23ErrorScore": error23,
                    "W26ErrorScore": error26,
                    "W26MinusW23Probability": p26 - p23,
                    "ErrorImprovement": error23 - error26,
                    "W26HighConfidenceError": bool(error26 >= HIGH_CONFIDENCE_ERROR),
                    REPORT: report,
                }
            )

    return pd.DataFrame(rows)


def build_state_summary(
    errors: pd.DataFrame,
) -> Optional[pd.DataFrame]:

    filtered = errors[errors["ReportState"].notna()].copy()

    if filtered.empty:
        return None

    rows = []

    for (
        label,
        state,
    ), group in filtered.groupby(
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

    w23 = metrics[metrics["Model"] == "W2.3"].set_index("Label")

    w26 = metrics[metrics["Model"] == "W2.6_fixed50"].set_index("Label")

    policy_by_label = policy.set_index("Label") if policy is not None else None

    rows = []

    for label in LABELS:

        auc23 = float(
            w23.loc[
                label,
                "AUROC",
            ]
        )

        auc26 = float(
            w26.loc[
                label,
                "AUROC",
            ]
        )

        ap23 = float(
            w23.loc[
                label,
                "AP",
            ]
        )

        ap26 = float(
            w26.loc[
                label,
                "AP",
            ]
        )

        brier23 = float(
            w23.loc[
                label,
                "Brier",
            ]
        )

        brier26 = float(
            w26.loc[
                label,
                "Brier",
            ]
        )

        label_errors = errors[errors["Label"] == label]

        high_errors = int(label_errors["W26HighConfidenceError"].sum())

        not_addressed_n = np.nan

        not_addressed_positive_rate = np.nan

        if state_summary is not None:

            state_rows = state_summary[
                (state_summary["Label"] == label)
                & (state_summary["ReportState"] == "N")
            ]

            if not state_rows.empty:

                row = state_rows.iloc[0]

                not_addressed_n = int(row["N"])

                not_addressed_positive_rate = float(row["GoldPositiveRate"])

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

        if label not in FS4 and auc23 < 0.72:

            flags.append("UNIMPROVED_WEAK_BASE_LABEL")

        if auc26 < auc23 - 0.01:

            flags.append("W26_AUC_REGRESSION")

        if brier26 > brier23 + 0.01:

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

        priority_score += int(label not in FS4 and auc23 < 0.72) * 3

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
                "W23_Brier": brier23,
                "W26_Brier": brier26,
                "Delta_Brier": brier26 - brier23,
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
        "display_version": DISPLAY_VERSION,
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
# AUDIT
# =============================================================================


def audit(
    accelerator: str,
) -> dict:

    paths = Paths.discover()

    output = Path(paths.output_root)

    gold = load_gold(paths)

    log("=" * 96)

    log(DISPLAY_VERSION)

    log("=" * 96)

    # -----------------------------------------------------------------
    # W2.3
    # -----------------------------------------------------------------

    log("")

    log("[1/6] Loading exact " "W2.3 fold-safe gold predictions...")

    w23, w23_meta = load_w23_gold_predictions(
        paths,
        gold,
    )

    log(f"W2.3 reconstructed " f"macro AUROC = " f"{w23_meta['macro_auroc']:.6f}")

    # -----------------------------------------------------------------
    # W2.6
    # -----------------------------------------------------------------

    log("")

    log("[2/6] Discovering " "fold-safe W2.6 FS4 " "gold predictions...")

    (
        _fs4,
        w26,
        w26_meta,
    ) = discover_fs4_gold_predictions(
        paths,
        gold,
        w23,
    )

    log(
        f"W2.6 fixed50 reconstructed "
        f"macro AUROC = "
        f"{w26_meta['fixed50_macro_auroc']:.6f}"
    )

    # -------------------------------------------------------------
    # v3 FIX:
    # W2.6 can be reconstructed from multiple fold sources.
    # Never assume singular metadata key "source".
    # -------------------------------------------------------------

    w26_sources = w26_meta.get("sources")

    if not w26_sources:

        source = w26_meta.get("source")

        w26_sources = [source] if source else []

    if w26_sources:

        log(f"W2.6 source files " f"({len(w26_sources)}):")

        for source in w26_sources:

            log(f"  {source}")

    else:

        log("W2.6 source provenance " "not available in metadata.")

    # -----------------------------------------------------------------
    # Gold metrics
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
    # Report state
    # -----------------------------------------------------------------

    log("")

    log("[3/6] Looking for " "explicit report-state artifact...")

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
            f"(coverage="
            f"{state_meta['coverage']:.3f})"
        )

    # -----------------------------------------------------------------
    # Errors
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
    # Production policy
    # -----------------------------------------------------------------

    log("")

    log("[4/6] Auditing " "W2.6-P FAST production " "mask/weights...")

    policy, policy_meta = load_production_policy(paths)

    if policy is not None:

        policy.to_csv(
            output / "05_production_policy_audit.csv",
            index=False,
        )

        log(f"Production selected cells = " f"{policy_meta['selected_cells']}")

    else:

        log(
            "Production policy artifacts "
            "unavailable; gold audit "
            "can still continue."
        )

    # -----------------------------------------------------------------
    # Priorities
    # -----------------------------------------------------------------

    log("")

    log("[5/6] Building " "label-priority findings...")

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
    # Summary
    # -----------------------------------------------------------------

    macro23 = float(metrics[metrics["Model"] == "W2.3"]["AUROC"].mean())

    macro26 = float(metrics[metrics["Model"] == "W2.6_fixed50"]["AUROC"].mean())

    brier23 = float(metrics[metrics["Model"] == "W2.3"]["Brier"].mean())

    brier26 = float(metrics[metrics["Model"] == "W2.6_fixed50"]["Brier"].mean())

    high_confidence_errors = errors[errors["W26HighConfidenceError"]]

    review_first = priorities[priorities["Action"] == "REVIEW_FIRST"]["Label"].tolist()

    review = priorities[priorities["Action"] == "REVIEW"]["Label"].tolist()

    summary = {
        "script_version": SCRIPT_VERSION,
        "display_version": DISPLAY_VERSION,
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
            "high_confidence_error_cells": len(high_confidence_errors),
        },
        "report_state_audit": state_meta,
        "production_policy_audit": policy_meta,
        "priority": {
            "review_first": review_first,
            "review": review,
        },
        "important_interpretation": [
            (
                "Gold metrics are reconstructed "
                "only from fold-safe held-out "
                "prediction artifacts."
            ),
            (
                "W2.6-P all-58 production "
                "pseudo labels are not treated "
                "as pristine gold validation."
            ),
            (
                "P/A/U/N states are used only "
                "when an explicit state artifact "
                "is structurally identified."
            ),
            (
                "0.5 threshold error classes "
                "are diagnostic only; "
                "competition evaluation is AUROC-based."
            ),
            (
                "This audit does not authorize W8 "
                "automatically. Review priority "
                "findings first."
            ),
        ],
        "next_step": (
            "Review 01_gold_teacher_metrics.csv, "
            "03_top_error_cases.csv, "
            "05_production_policy_audit.csv, "
            "and 06_priority_findings.csv before "
            "deciding whether a single controlled "
            "W8 teacher change is justified."
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

    log(f"W2.3 macro AUROC      : " f"{macro23:.6f}")

    log(f"W2.6 fixed50 AUROC    : " f"{macro26:.6f}")

    log(f"Delta                 : " f"{macro26 - macro23:+.6f}")

    log(f"W2.3 macro Brier      : " f"{brier23:.6f}")

    log(f"W2.6 macro Brier      : " f"{brier26:.6f}")

    log(f"High-confidence errors: " f"{len(high_confidence_errors)}")

    log(f"REVIEW_FIRST labels   : " f"{review_first}")

    log(f"REVIEW labels         : " f"{review}")

    log("")

    log(f"Results: {output}")

    log("")

    log("STOP HERE. " "Do not train W8 yet.")

    return summary


# =============================================================================
# CLI
# =============================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("RSNA teacher error " "audit v3 — CPU only")
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
        help=("Accepted for interface consistency. " "This audit always runs on CPU."),
    )

    args = parser.parse_args()

    if args.mode == "status":

        status(args.accelerator)

    elif args.mode == "audit":

        audit(args.accelerator)


if __name__ == "__main__":
    main()
