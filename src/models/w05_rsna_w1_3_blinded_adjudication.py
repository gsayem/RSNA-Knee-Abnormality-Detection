# ============================================================
# RSNA KNEE ABNORMALITY DETECTION - W1.3
# Blinded Gold-Report Adjudication Benchmark
# ============================================================
#
# W1.3 converts the W1.2 120-pair benchmark into a blinded
# human-review benchmark.
#
# PREPARE mode:
#   Reviewer sees only:
#     CaseID, TargetLabel, Language, Report, blank review fields
#
#   Reviewer does NOT see:
#     StudyInstanceUID, Gold, parser assertion/evidence,
#     alignment category, or W1.2 selection reason.
#
# FINALIZE mode:
#   After blinded review is complete, merge with the private
#   answer key and calculate:
#     - reviewer vs W1.2 parser agreement
#     - report assertion -> challenge gold mapping
#     - label-specific positive PPV / negative NPV
#     - severity -> gold mapping
#     - language-specific parser behavior
#
# W1.3 trains NO NLP model and creates NO pseudo-labels.
#
# Environment variables:
#   W13_MODE               prepare | finalize
#   W13_W1_2_SOURCE        W1.2 results directory or zip
#   W13_WORK_ROOT          output root
#   W13_COMPLETED_REVIEW   completed blinded CSV
#
# Default output:
#   /kaggle/working/rsna_w1_3/
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIG
# ============================================================

EXPECTED_BENCHMARK_PAIRS = 120
BLINDING_SEED = 20260817

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

ASSERTION_VALUES = [
    "positive",
    "negative",
    "uncertain",
    "mixed",
    "related_abnormality",
    "mentioned_neutral",
    "not_mentioned",
]

REVIEW_STATUS_VALUES = [
    "complete",
    "needs_translation",
    "unable_to_adjudicate",
]

CONFIDENCE_VALUES = [
    "high",
    "medium",
    "low",
]

PHENOTYPE_VALUES = [
    "direct_target",
    "mild_or_partial_target",
    "moderate_target",
    "severe_or_complete_target",
    "degenerative_target",
    "related_finding_not_target",
    "historical_or_postoperative",
    "negated_target",
    "uncertain_target",
    "mixed_target",
    "neutral_anatomy_mention",
    "not_mentioned",
    "other",
]

SEVERITY_VALUES = [
    "not_stated",
    "minimal",
    "mild",
    "grade_1",
    "grade_2",
    "grade_3",
    "grade_4",
    "partial",
    "moderate",
    "severe",
    "complete",
    "degenerative",
    "chronic",
    "acute",
    "other",
]

WORK_ROOT = Path(
    os.environ.get(
        "W13_WORK_ROOT",
        "/kaggle/working/rsna_w1_3",
    )
)

REVIEW_ROOT = WORK_ROOT / "review"
PRIVATE_ROOT = WORK_ROOT / "private"
RESULT_ROOT = WORK_ROOT / "results"

for directory in [
    WORK_ROOT,
    REVIEW_ROOT,
    PRIVATE_ROOT,
    RESULT_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

COMPLETED_REVIEW_PATH = Path(
    os.environ.get(
        "W13_COMPLETED_REVIEW",
        str(REVIEW_ROOT / "01_W1_3_BLINDED_REVIEW_COMPLETED.csv"),
    )
)

BENCHMARK_FILENAME = "12_adjudication_benchmark_seed.csv"
CONFIG_FILENAME = "w1_2_config.json"


# ============================================================
# 2. SAFE SERIALIZATION
# ============================================================


def safe_json_value(value: Any) -> Any:

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


def save_json(path: Path, payload: Dict[str, Any]) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            safe_json_value(payload),
            f,
            indent=4,
            ensure_ascii=False,
        )


def sha256_text(value: str) -> str:

    return hashlib.sha256(
        value.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 3. LOAD W1.2 BENCHMARK
# ============================================================


def candidate_sources() -> List[Path]:

    sources: List[Path] = []

    explicit = os.environ.get(
        "W13_W1_2_SOURCE",
        "",
    ).strip()

    if explicit:
        sources.append(Path(explicit))

    sources.extend(
        [
            Path("/kaggle/working/rsna_w1_2/results"),
            Path("/kaggle/working/rsna_w1_2"),
            Path("/kaggle/working/rsna_w1_2.zip"),
            Path("/mnt/data/rsna_w1_2.zip"),
        ]
    )

    kaggle_input = Path("/kaggle/input")

    if kaggle_input.exists():
        try:
            sources.extend(sorted(kaggle_input.glob("**/rsna_w1_2.zip")))
        except Exception:
            pass

    result: List[Path] = []
    seen = set()

    for source in sources:
        key = str(source)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)

    return result


def locate_seed_in_directory(
    source: Path,
) -> Optional[Path]:

    if not source.exists() or not source.is_dir():
        return None

    candidates = [
        source / BENCHMARK_FILENAME,
        source / "results" / BENCHMARK_FILENAME,
        source / "rsna_w1_2" / "results" / BENCHMARK_FILENAME,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    try:
        matches = list(source.glob(f"**/{BENCHMARK_FILENAME}"))
    except Exception:
        matches = []

    return matches[0] if matches else None


def read_w12_source() -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
    str,
]:

    attempted: List[str] = []

    for source in candidate_sources():

        attempted.append(str(source))

        seed_path = locate_seed_in_directory(source)

        if seed_path is not None:

            df = pd.read_csv(seed_path)

            config: Dict[str, Any] = {}

            config_path = seed_path.parent / CONFIG_FILENAME

            if config_path.exists():
                try:
                    with open(
                        config_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        config = json.load(f)
                except Exception:
                    config = {}

            return (
                df,
                config,
                str(seed_path),
            )

        if source.exists() and source.is_file() and source.suffix.casefold() == ".zip":

            try:
                with zipfile.ZipFile(
                    source,
                    "r",
                ) as z:

                    seed_names = [
                        name
                        for name in z.namelist()
                        if name.endswith(BENCHMARK_FILENAME)
                    ]

                    if not seed_names:
                        continue

                    seed_name = seed_names[0]

                    df = pd.read_csv(io.BytesIO(z.read(seed_name)))

                    config: Dict[str, Any] = {}

                    config_names = [
                        name for name in z.namelist() if name.endswith(CONFIG_FILENAME)
                    ]

                    if config_names:
                        try:
                            config = json.loads(z.read(config_names[0]).decode("utf-8"))
                        except Exception:
                            config = {}

                    return (
                        df,
                        config,
                        f"{source}::{seed_name}",
                    )

            except zipfile.BadZipFile:
                continue

    raise FileNotFoundError(
        "Could not locate W1.2 benchmark. "
        "Set W13_W1_2_SOURCE to the W1.2 "
        "results directory or rsna_w1_2.zip.\n"
        "Attempted:\n  - " + "\n  - ".join(attempted)
    )


REQUIRED_W12_COLUMNS = [
    "StudyInstanceUID",
    "Language",
    "Label",
    "Gold",
    "ReportAssertion",
    "AlignmentCategory",
    "RepresentativeEvidence",
    "Severity",
    "ManualReviewReason",
    "Report",
    "PairKey",
    "ReportFingerprint",
    "ParserChangedFromW11",
    "AdjudicationReason",
    "AdjudicationPriority",
]


def validate_w12(df: pd.DataFrame) -> None:

    missing = [column for column in REQUIRED_W12_COLUMNS if column not in df.columns]

    if missing:
        raise RuntimeError("W1.2 seed missing required columns: " f"{missing}")

    if len(df) != EXPECTED_BENCHMARK_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_BENCHMARK_PAIRS} rows, " f"found {len(df)}."
        )

    if df["PairKey"].duplicated().any():
        raise RuntimeError("Duplicate PairKey values found.")

    invalid_labels = sorted(set(df["Label"].astype(str)) - set(LABEL_COLUMNS))

    if invalid_labels:
        raise RuntimeError("Unexpected labels: " f"{invalid_labels}")

    if df["Report"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Every benchmark case must have report text.")


# ============================================================
# 4. BLINDING
# ============================================================

BLINDED_COLUMNS = [
    "CaseID",
    "TargetLabel",
    "Language",
    "Report",
    "ReportFingerprint",
    "ReviewerStatus",
    "ReviewerReportAssertion",
    "ReviewerEvidenceSpan",
    "ReviewerSeverity",
    "ReviewerPhenotypeCategory",
    "ReviewerConfidence",
    "ReviewerID",
    "ReviewerTranslationUsed",
    "ReviewerNotes",
]

FORBIDDEN_BLINDED_COLUMNS = [
    "StudyInstanceUID",
    "Gold",
    "ReportAssertion",
    "ReportBinaryValue",
    "Decidable",
    "AlignmentCategory",
    "RepresentativeSection",
    "RepresentativeEvidence",
    "ManualReviewPriority",
    "ManualReviewReason",
    "ParserChangedFromW11",
    "AdjudicationReason",
    "AdjudicationPriority",
]


def create_blinded_order(
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:

    df = benchmark_df.copy()

    df["_BlindSort"] = (
        df["PairKey"]
        .astype(str)
        .map(lambda pair_key: sha256_text(f"{BLINDING_SEED}||{pair_key}"))
    )

    df = df.sort_values("_BlindSort").reset_index(drop=True)

    df["CaseID"] = [
        f"W13-{index:03d}"
        for index in range(
            1,
            len(df) + 1,
        )
    ]

    return df


def build_blinded_sheet(
    ordered_df: pd.DataFrame,
) -> pd.DataFrame:

    blinded = pd.DataFrame(
        {
            "CaseID": ordered_df["CaseID"].astype(str),
            "TargetLabel": ordered_df["Label"].astype(str),
            "Language": ordered_df["Language"].astype(str),
            "Report": ordered_df["Report"].astype(str),
            "ReportFingerprint": ordered_df["ReportFingerprint"].astype(str),
            "ReviewerStatus": "",
            "ReviewerReportAssertion": "",
            "ReviewerEvidenceSpan": "",
            "ReviewerSeverity": "",
            "ReviewerPhenotypeCategory": "",
            "ReviewerConfidence": "",
            "ReviewerID": "",
            "ReviewerTranslationUsed": "",
            "ReviewerNotes": "",
        }
    )

    return blinded[BLINDED_COLUMNS]


def build_private_answer_key(
    ordered_df: pd.DataFrame,
) -> pd.DataFrame:

    private = ordered_df.drop(
        columns=["_BlindSort"],
        errors="ignore",
    ).copy()

    # W1.2's seed contains blank Reviewer* columns.
    # They are not part of the private answer key and would
    # collide with the actual blinded-review fields at FINALIZE.
    private = private.drop(
        columns=[
            column for column in private.columns if str(column).startswith("Reviewer")
        ],
        errors="ignore",
    )

    first = [
        "CaseID",
        "PairKey",
        "StudyInstanceUID",
        "Label",
        "Language",
        "Gold",
        "ReportAssertion",
        "ReportBinaryValue",
        "Decidable",
        "AlignmentCategory",
        "RepresentativeSection",
        "RepresentativeEvidence",
        "Severity",
        "RepresentativeNegativeCues",
        "RepresentativeUncertaintyCues",
        "RepresentativeHistoryCues",
        "ParserChangedFromW11",
        "AdjudicationReason",
        "AdjudicationPriority",
        "ManualReviewPriority",
        "ManualReviewReason",
        "ReportFingerprint",
        "Report",
    ]

    existing_first = [column for column in first if column in private.columns]

    remaining = [column for column in private.columns if column not in existing_first]

    return private[existing_first + remaining]


# ============================================================
# 5. BLINDED REVIEW GUIDE
# ============================================================


def write_review_guide(path: Path) -> None:

    guide = """# W1.3 Blinded Report Adjudication Guide

## Blinding rule

During adjudication, open only:

- `01_W1_3_BLINDED_REVIEW.csv`
- this guide

Do not open the `private/` directory until the review is
finished. The private file contains the challenge gold labels
and the W1.2 parser outputs.

The task is to annotate what the **report itself asserts**.
Do not try to infer or reproduce the challenge gold label.

---

## ReviewerStatus

Use exactly one:

- `complete`
- `needs_translation`
- `unable_to_adjudicate`

If the report cannot be interpreted reliably, use
`needs_translation`. Do not guess.

---

## ReviewerReportAssertion

Use exactly one:

- `positive`
- `negative`
- `uncertain`
- `mixed`
- `related_abnormality`
- `mentioned_neutral`
- `not_mentioned`

### positive
The report explicitly asserts the target pathology/finding.

### negative
The report explicitly says the target is absent, intact,
normal, or otherwise negative.

### uncertain
The report says possible, suspicious, cannot exclude, etc.

### mixed
Materially conflicting positive and negative statements are
present and cannot be resolved from the report.

### related_abnormality
A related finding is present, but the report does not directly
assert the target.

Examples:
- meniscal degeneration without explicit tear
- bone marrow edema without explicit contusion/bruise
- synovial thickening without explicit synovitis
- compartmental chondral/cartilage abnormality without
  explicit OA/osteoarthritis/arthrosis

### mentioned_neutral
The target anatomy is mentioned without a clear positive,
negative, uncertain, or related finding.

### not_mentioned
No relevant target statement appears.

`not_mentioned` must never be converted to `negative`.

---

## Conservative target interpretation

### ACL
Direct ACL injury/tear/rupture/sprain.
Record grade/partial/complete separately.

### MCL
Direct MCL injury/tear/rupture/sprain.

### Medial Meniscus / Lateral Meniscus
Direct tear/rupture of the specified meniscus.
Degenerative signal without tear -> `related_abnormality`.

### Medial OA / Lateral OA / PF OA
For blinded adjudication, use a conservative definition:
explicit OA / osteoarthritis / arthrosis / gonarthrosis in the
target compartment -> `positive`.

Isolated chondromalacia, cartilage loss, chondral erosion,
osteophytes, or similar degeneration without explicit
OA/arthrosis wording -> `related_abnormality`.

This preserves terminology/ontology uncertainty for later
comparison with challenge gold.

### Effusion
Direct joint effusion or explicit increased intra-articular
fluid/equivalent fluid collection.

### Synovitis
Explicit synovitis -> `positive`.
Synovial hypertrophy/thickening alone -> `related_abnormality`.

### Baker's
Explicit Baker/popliteal cyst.

### Contusion
Explicit bone contusion / bone bruise -> `positive`.
Marrow edema alone -> `related_abnormality`.

### Fracture
Any explicit fracture statement is report-positive regardless
of subtype.

---

## ReviewerEvidenceSpan

Copy the shortest phrase that supports the assertion.

Leave blank only for `not_mentioned`.

---

## ReviewerSeverity

Use one or more values separated by `|`:

- `not_stated`
- `minimal`
- `mild`
- `grade_1`
- `grade_2`
- `grade_3`
- `grade_4`
- `partial`
- `moderate`
- `severe`
- `complete`
- `degenerative`
- `chronic`
- `acute`
- `other`

Record only explicitly stated severity.

---

## ReviewerPhenotypeCategory

Use one:

- `direct_target`
- `mild_or_partial_target`
- `moderate_target`
- `severe_or_complete_target`
- `degenerative_target`
- `related_finding_not_target`
- `historical_or_postoperative`
- `negated_target`
- `uncertain_target`
- `mixed_target`
- `neutral_anatomy_mention`
- `not_mentioned`
- `other`

This describes the report phenotype only. It must not refer to
whether challenge gold is 0 or 1.

---

## ReviewerConfidence

Use:
- `high`
- `medium`
- `low`

This is confidence in the report-side interpretation.

---

## ReviewerTranslationUsed

Optional:
- `yes`
- `no`

If translation or language assistance was used, mention the
method briefly in `ReviewerNotes`.

---

## Methodological rule

Do not repair a report interpretation to make it agree with the
challenge gold label. Disagreement is a primary W1.3 outcome.
"""

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(guide)


# ============================================================
# 6. PREPARE
# ============================================================


def prepare_mode() -> None:

    benchmark_df, w12_config, source_description = read_w12_source()

    validate_w12(benchmark_df)

    ordered_df = create_blinded_order(benchmark_df)

    blinded_df = build_blinded_sheet(ordered_df)

    private_df = build_private_answer_key(ordered_df)

    blinded_path = REVIEW_ROOT / "01_W1_3_BLINDED_REVIEW.csv"

    private_path = PRIVATE_ROOT / "02_W1_3_PRIVATE_ANSWER_KEY.csv"

    guide_path = REVIEW_ROOT / "REVIEW_GUIDE_BLINDED.md"

    blinded_df.to_csv(
        blinded_path,
        index=False,
        encoding="utf-8-sig",
    )

    private_df.to_csv(
        private_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_review_guide(guide_path)

    forbidden_present = [
        column for column in FORBIDDEN_BLINDED_COLUMNS if column in blinded_df.columns
    ]

    if forbidden_present:
        raise RuntimeError(
            "Blinded sheet contains hidden fields: " f"{forbidden_present}"
        )

    if len(blinded_df) != EXPECTED_BENCHMARK_PAIRS:
        raise RuntimeError("Blinded row count mismatch.")

    if blinded_df["CaseID"].duplicated().any():
        raise RuntimeError("Duplicate CaseID generated.")

    if set(blinded_df["CaseID"]) != set(private_df["CaseID"]):
        raise RuntimeError("Blinded/private CaseID mismatch.")

    composition_rows = []

    for dimension in [
        "TargetLabel",
        "Language",
    ]:
        counts = blinded_df[dimension].value_counts(dropna=False)

        for value, count in counts.items():
            composition_rows.append(
                {
                    "Dimension": dimension,
                    "Value": value,
                    "Count": int(count),
                    "Percent": float(100.0 * count / len(blinded_df)),
                }
            )

    pd.DataFrame(composition_rows).to_csv(
        RESULT_ROOT / "03_blinded_benchmark_composition.csv",
        index=False,
    )

    manifest = {
        "phase": "W1.3 blinded gold-report adjudication",
        "mode": "prepare",
        "source": source_description,
        "w1_2_audit_rule_version": w12_config.get("audit_rule_version"),
        "benchmark_pairs": int(len(blinded_df)),
        "blinding_seed": BLINDING_SEED,
        "blinded_columns": BLINDED_COLUMNS,
        "hidden_from_reviewer": FORBIDDEN_BLINDED_COLUMNS,
        "blinded_file": str(blinded_path),
        "private_answer_key": str(private_path),
        "blinded_file_sha256": sha256_file(blinded_path),
        "private_file_sha256": sha256_file(private_path),
        "nlp_model_trained": False,
        "pseudo_labels_generated": False,
    }

    save_json(
        RESULT_ROOT / "04_blinding_manifest.json",
        manifest,
    )

    readme = (
        "W1.3 BLINDED ADJUDICATION\n\n"
        "REVIEW ONLY:\n"
        f"  {blinded_path}\n"
        f"  {guide_path}\n\n"
        "DO NOT OPEN UNTIL REVIEW IS COMPLETE:\n"
        f"  {private_path}\n\n"
        f"Rows: {len(blinded_df)}\n\n"
        "After review, save the completed file as:\n"
        f"  {COMPLETED_REVIEW_PATH}\n\n"
        "Then rerun with:\n"
        "  W13_MODE=finalize\n"
    )

    with open(
        REVIEW_ROOT / "00_READ_ME_FIRST.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(readme)

    print()
    print("=" * 80)
    print("W1.3 PREPARE COMPLETE")
    print("=" * 80)
    print()
    print(f"Benchmark pairs: {len(blinded_df)}")
    print()
    print("REVIEW THESE FILES ONLY:")
    print(f"  {blinded_path}")
    print(f"  {guide_path}")
    print()
    print("KEEP PRIVATE UNTIL REVIEW IS COMPLETE:")
    print(f"  {private_path}")
    print()
    print("NO NLP model trained. " "NO pseudo-labels generated.")


# ============================================================
# 7. FINALIZE VALIDATION
# ============================================================


def clean_string(value: Any) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def split_values(value: Any) -> List[str]:

    value = clean_string(value)

    if not value:
        return []

    return [
        part.strip()
        for part in re.split(
            r"[|,;]+",
            value,
        )
        if part.strip()
    ]


def validate_completed_review(
    review_df: pd.DataFrame,
    answer_df: pd.DataFrame,
) -> pd.DataFrame:

    missing = [column for column in BLINDED_COLUMNS if column not in review_df.columns]

    if missing:
        raise RuntimeError("Completed review missing columns: " f"{missing}")

    hidden_present = [
        column for column in FORBIDDEN_BLINDED_COLUMNS if column in review_df.columns
    ]

    if hidden_present:
        raise RuntimeError(
            "Completed review contains hidden answer-key " f"columns: {hidden_present}"
        )

    if len(review_df) != EXPECTED_BENCHMARK_PAIRS:
        raise RuntimeError(
            "Completed review must contain exactly " f"{EXPECTED_BENCHMARK_PAIRS} rows."
        )

    if review_df["CaseID"].duplicated().any():
        raise RuntimeError("Duplicate CaseID in completed review.")

    if set(review_df["CaseID"].astype(str)) != set(answer_df["CaseID"].astype(str)):
        raise RuntimeError(
            "Completed-review CaseID set does not match " "the private answer key."
        )

    df = review_df.copy()

    review_text_columns = [
        "ReviewerStatus",
        "ReviewerReportAssertion",
        "ReviewerEvidenceSpan",
        "ReviewerSeverity",
        "ReviewerPhenotypeCategory",
        "ReviewerConfidence",
        "ReviewerID",
        "ReviewerTranslationUsed",
        "ReviewerNotes",
    ]

    for column in review_text_columns:
        df[column] = df[column].map(clean_string)

    # Convenience: if assertion is filled but status is blank,
    # infer that the row is complete.
    infer_complete = df["ReviewerStatus"].eq("") & df["ReviewerReportAssertion"].ne("")

    df.loc[
        infer_complete,
        "ReviewerStatus",
    ] = "complete"

    invalid_status = sorted(
        set(df["ReviewerStatus"]) - set(REVIEW_STATUS_VALUES) - {""}
    )

    if invalid_status:
        raise RuntimeError("Invalid ReviewerStatus: " f"{invalid_status}")

    complete_mask = df["ReviewerStatus"] == "complete"

    invalid_assertions = sorted(
        set(
            df.loc[
                complete_mask,
                "ReviewerReportAssertion",
            ]
        )
        - set(ASSERTION_VALUES)
    )

    if invalid_assertions:
        raise RuntimeError("Invalid ReviewerReportAssertion: " f"{invalid_assertions}")

    invalid_confidence = sorted(
        set(
            df.loc[
                complete_mask,
                "ReviewerConfidence",
            ]
        )
        - set(CONFIDENCE_VALUES)
    )

    if invalid_confidence:
        raise RuntimeError("Invalid ReviewerConfidence: " f"{invalid_confidence}")

    invalid_phenotype = sorted(
        set(
            df.loc[
                complete_mask,
                "ReviewerPhenotypeCategory",
            ]
        )
        - set(PHENOTYPE_VALUES)
    )

    if invalid_phenotype:
        raise RuntimeError("Invalid ReviewerPhenotypeCategory: " f"{invalid_phenotype}")

    invalid_severity = set()

    for value in df.loc[
        complete_mask,
        "ReviewerSeverity",
    ]:
        for token in split_values(value):
            if token not in SEVERITY_VALUES:
                invalid_severity.add(token)

    if invalid_severity:
        raise RuntimeError(
            "Invalid ReviewerSeverity token(s): " f"{sorted(invalid_severity)}"
        )

    evidence_required = {
        "positive",
        "negative",
        "uncertain",
        "mixed",
        "related_abnormality",
        "mentioned_neutral",
    }

    missing_evidence = (
        complete_mask
        & df["ReviewerReportAssertion"].isin(evidence_required)
        & df["ReviewerEvidenceSpan"].eq("")
    )

    if missing_evidence.any():
        cases = df.loc[
            missing_evidence,
            "CaseID",
        ].tolist()

        raise RuntimeError(
            "ReviewerEvidenceSpan is required for "
            "explicit/related/neutral cases. "
            f"Missing: {cases[:20]}"
        )

    # Normalize blank completed severity.
    df.loc[
        complete_mask & df["ReviewerSeverity"].eq(""),
        "ReviewerSeverity",
    ] = "not_stated"

    return df


# ============================================================
# 8. STATISTICS
# ============================================================


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> Tuple[
    Optional[float],
    Optional[float],
]:

    if total <= 0:
        return (None, None)

    p = successes / total

    denominator = 1.0 + z * z / total

    center = (p + z * z / (2.0 * total)) / denominator

    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def gold_mapping_category(
    assertion: str,
    gold: int,
) -> str:

    prefix = {
        "positive": "report_positive",
        "negative": "report_negative",
        "uncertain": "report_uncertain",
        "mixed": "report_mixed",
        "related_abnormality": "report_related",
        "mentioned_neutral": "report_neutral",
        "not_mentioned": "report_not_mentioned",
    }[assertion]

    suffix = "gold_positive" if gold == 1 else "gold_negative"

    return f"{prefix}_{suffix}"


def reviewer_parser_summary(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    complete = merged[merged["ReviewerStatus"] == "complete"].copy()

    rows = []

    groups = [("ALL", complete)]

    for label in LABEL_COLUMNS:
        groups.append(
            (
                label,
                complete[complete["Label"] == label],
            )
        )

    for label_name, group in groups:

        if len(group) == 0:
            continue

        exact = group["ReviewerReportAssertion"] == group["ReportAssertion"].astype(str)

        binary_mask = group["ReviewerReportAssertion"].isin(
            ["positive", "negative"]
        ) & group["ReportAssertion"].isin(["positive", "negative"])

        binary = group[binary_mask]

        binary_agreement = (
            float(
                (binary["ReviewerReportAssertion"] == binary["ReportAssertion"]).mean()
            )
            if len(binary) > 0
            else np.nan
        )

        parser_fp = int(
            (
                (binary["ReviewerReportAssertion"] == "negative")
                & (binary["ReportAssertion"] == "positive")
            ).sum()
        )

        parser_fn = int(
            (
                (binary["ReviewerReportAssertion"] == "positive")
                & (binary["ReportAssertion"] == "negative")
            ).sum()
        )

        rows.append(
            {
                "Label": label_name,
                "CompletedN": int(len(group)),
                "ExactAssertionAgreement": float(exact.mean()),
                "BinaryComparableN": int(len(binary)),
                "BinaryAgreement": binary_agreement,
                "ParserFalsePositiveVsReviewer": parser_fp,
                "ParserFalseNegativeVsReviewer": parser_fn,
            }
        )

    return pd.DataFrame(rows)


def assertion_to_gold_mapping(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    complete = merged[merged["ReviewerStatus"] == "complete"].copy()

    rows = []

    for (
        label,
        assertion,
    ), group in complete.groupby(
        [
            "Label",
            "ReviewerReportAssertion",
        ],
        dropna=False,
    ):

        n = int(len(group))

        gold_positive = int((group["Gold"] == 1).sum())

        low, high = wilson_interval(
            gold_positive,
            n,
        )

        rows.append(
            {
                "Label": label,
                "ReviewerReportAssertion": assertion,
                "N": n,
                "GoldPositive": gold_positive,
                "GoldNegative": int(n - gold_positive),
                "P_GoldPositive": float(gold_positive / n),
                "P_GoldPositive_WilsonLow": low,
                "P_GoldPositive_WilsonHigh": high,
                "P_GoldNegative": float((n - gold_positive) / n),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "Label",
            "ReviewerReportAssertion",
        ]
    )


def label_trust_summary(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    complete = merged[merged["ReviewerStatus"] == "complete"].copy()

    rows = []

    for label in LABEL_COLUMNS:

        group = complete[complete["Label"] == label]

        positive = group[group["ReviewerReportAssertion"] == "positive"]

        negative = group[group["ReviewerReportAssertion"] == "negative"]

        pos_n = int(len(positive))
        neg_n = int(len(negative))

        pos_gold1 = int((positive["Gold"] == 1).sum())

        neg_gold0 = int((negative["Gold"] == 0).sum())

        pos_low, pos_high = wilson_interval(
            pos_gold1,
            pos_n,
        )

        neg_low, neg_high = wilson_interval(
            neg_gold0,
            neg_n,
        )

        rows.append(
            {
                "Label": label,
                "CompletedN": int(len(group)),
                "ReviewerPositiveN": pos_n,
                "GoldPositiveAmongReviewerPositive": pos_gold1,
                "ReportPositivePPV": (pos_gold1 / pos_n if pos_n > 0 else np.nan),
                "ReportPositivePPV_WilsonLow": pos_low,
                "ReportPositivePPV_WilsonHigh": pos_high,
                "ReviewerNegativeN": neg_n,
                "GoldNegativeAmongReviewerNegative": neg_gold0,
                "ReportNegativeNPV": (neg_gold0 / neg_n if neg_n > 0 else np.nan),
                "ReportNegativeNPV_WilsonLow": neg_low,
                "ReportNegativeNPV_WilsonHigh": neg_high,
                "ReviewerUncertainN": int(
                    (group["ReviewerReportAssertion"] == "uncertain").sum()
                ),
                "ReviewerMixedN": int(
                    (group["ReviewerReportAssertion"] == "mixed").sum()
                ),
                "ReviewerRelatedN": int(
                    (group["ReviewerReportAssertion"] == "related_abnormality").sum()
                ),
                "ReviewerNotMentionedN": int(
                    (group["ReviewerReportAssertion"] == "not_mentioned").sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def severity_to_gold_mapping(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    complete = merged[merged["ReviewerStatus"] == "complete"].copy()

    exploded = []

    for _, row in complete.iterrows():

        severity_values = split_values(row["ReviewerSeverity"])

        if not severity_values:
            severity_values = ["not_stated"]

        for severity in severity_values:
            exploded.append(
                {
                    "CaseID": row["CaseID"],
                    "Label": row["Label"],
                    "ReviewerReportAssertion": row["ReviewerReportAssertion"],
                    "ReviewerSeverity": severity,
                    "Gold": int(row["Gold"]),
                }
            )

    if not exploded:
        return pd.DataFrame(
            columns=[
                "Label",
                "ReviewerReportAssertion",
                "ReviewerSeverity",
                "N",
                "GoldPositive",
                "GoldNegative",
                "P_GoldPositive",
                "P_GoldPositive_WilsonLow",
                "P_GoldPositive_WilsonHigh",
            ]
        )

    severity_df = pd.DataFrame(exploded)

    rows = []

    for (
        label,
        assertion,
        severity,
    ), group in severity_df.groupby(
        [
            "Label",
            "ReviewerReportAssertion",
            "ReviewerSeverity",
        ],
        dropna=False,
    ):

        n = int(len(group))

        gold_positive = int((group["Gold"] == 1).sum())

        low, high = wilson_interval(
            gold_positive,
            n,
        )

        rows.append(
            {
                "Label": label,
                "ReviewerReportAssertion": assertion,
                "ReviewerSeverity": severity,
                "N": n,
                "GoldPositive": gold_positive,
                "GoldNegative": int(n - gold_positive),
                "P_GoldPositive": float(gold_positive / n),
                "P_GoldPositive_WilsonLow": low,
                "P_GoldPositive_WilsonHigh": high,
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "Label",
            "ReviewerReportAssertion",
            "ReviewerSeverity",
        ]
    )


def language_summary(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for language, group in merged.groupby(
        "Language",
        dropna=False,
    ):

        complete = group[group["ReviewerStatus"] == "complete"]

        parser_exact = (
            float(
                (
                    complete["ReviewerReportAssertion"] == complete["ReportAssertion"]
                ).mean()
            )
            if len(complete) > 0
            else np.nan
        )

        rows.append(
            {
                "Language": language,
                "BenchmarkN": int(len(group)),
                "CompletedN": int(len(complete)),
                "NeedsTranslationN": int(
                    (group["ReviewerStatus"] == "needs_translation").sum()
                ),
                "UnableToAdjudicateN": int(
                    (group["ReviewerStatus"] == "unable_to_adjudicate").sum()
                ),
                "ParserExactAssertionAgreement": parser_exact,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "BenchmarkN",
        ascending=False,
    )


# ============================================================
# 9. FINALIZE
# ============================================================


def finalize_mode() -> None:

    private_path = PRIVATE_ROOT / "02_W1_3_PRIVATE_ANSWER_KEY.csv"

    if not private_path.exists():
        raise FileNotFoundError(
            "Private answer key missing. "
            "Run PREPARE first.\n"
            f"Expected: {private_path}"
        )

    if not COMPLETED_REVIEW_PATH.exists():
        raise FileNotFoundError(
            "Completed blinded review not found.\n"
            f"Expected: {COMPLETED_REVIEW_PATH}\n"
            "Or set W13_COMPLETED_REVIEW."
        )

    answer_df = pd.read_csv(private_path)

    # Defensive compatibility with an older/private file that
    # might still contain W1.2's blank Reviewer* columns.
    answer_df = answer_df.drop(
        columns=[
            column for column in answer_df.columns if str(column).startswith("Reviewer")
        ],
        errors="ignore",
    )

    review_df = pd.read_csv(COMPLETED_REVIEW_PATH)

    review_df = validate_completed_review(
        review_df,
        answer_df,
    )

    reviewer_columns = [
        "CaseID",
        "ReviewerStatus",
        "ReviewerReportAssertion",
        "ReviewerEvidenceSpan",
        "ReviewerSeverity",
        "ReviewerPhenotypeCategory",
        "ReviewerConfidence",
        "ReviewerID",
        "ReviewerTranslationUsed",
        "ReviewerNotes",
    ]

    merged = answer_df.merge(
        review_df[reviewer_columns],
        on="CaseID",
        how="left",
        validate="one_to_one",
    )

    merged["ReviewerGoldMappingCategory"] = merged.apply(
        lambda row: (
            gold_mapping_category(
                row["ReviewerReportAssertion"],
                int(row["Gold"]),
            )
            if row["ReviewerStatus"] == "complete"
            else ""
        ),
        axis=1,
    )

    merged["ReviewerMatchesParserAssertion"] = np.where(
        merged["ReviewerStatus"] == "complete",
        (merged["ReviewerReportAssertion"] == merged["ReportAssertion"].astype(str)),
        np.nan,
    )

    merged.to_csv(
        RESULT_ROOT / "10_W1_3_ADJUDICATED_MERGED.csv",
        index=False,
        encoding="utf-8-sig",
    )

    parser_df = reviewer_parser_summary(merged)

    parser_df.to_csv(
        RESULT_ROOT / "11_reviewer_vs_parser_summary.csv",
        index=False,
    )

    mapping_df = assertion_to_gold_mapping(merged)

    mapping_df.to_csv(
        RESULT_ROOT / "12_report_assertion_to_gold_mapping.csv",
        index=False,
    )

    trust_df = label_trust_summary(merged)

    trust_df.to_csv(
        RESULT_ROOT / "13_label_report_trust_summary.csv",
        index=False,
    )

    severity_df = severity_to_gold_mapping(merged)

    severity_df.to_csv(
        RESULT_ROOT / "14_severity_to_gold_mapping.csv",
        index=False,
    )

    language_df = language_summary(merged)

    language_df.to_csv(
        RESULT_ROOT / "15_language_review_summary.csv",
        index=False,
    )

    completed_n = int((merged["ReviewerStatus"] == "complete").sum())

    completion_df = pd.DataFrame(
        [
            {
                "Metric": "BenchmarkPairs",
                "Count": int(len(merged)),
            },
            {
                "Metric": "Completed",
                "Count": completed_n,
            },
            {
                "Metric": "NeedsTranslation",
                "Count": int((merged["ReviewerStatus"] == "needs_translation").sum()),
            },
            {
                "Metric": "UnableToAdjudicate",
                "Count": int(
                    (merged["ReviewerStatus"] == "unable_to_adjudicate").sum()
                ),
            },
            {
                "Metric": "Unreviewed",
                "Count": int(merged["ReviewerStatus"].fillna("").eq("").sum()),
            },
        ]
    )

    completion_df.to_csv(
        RESULT_ROOT / "16_review_completion_summary.csv",
        index=False,
    )

    category_df = (
        merged[merged["ReviewerStatus"] == "complete"]["ReviewerGoldMappingCategory"]
        .value_counts(dropna=False)
        .rename_axis("ReviewerGoldMappingCategory")
        .reset_index(name="Count")
    )

    if len(category_df) > 0:
        category_df["Percent"] = (
            100.0 * category_df["Count"] / category_df["Count"].sum()
        )

    category_df.to_csv(
        RESULT_ROOT / "17_gold_mapping_case_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Human-readable final report
    # --------------------------------------------------------

    parser_all = parser_df[parser_df["Label"] == "ALL"]

    parser_exact = (
        float(parser_all.iloc[0]["ExactAssertionAgreement"])
        if len(parser_all) > 0
        else np.nan
    )

    lines = [
        "# RSNA W1.3 Final Adjudication Report",
        "",
        "## Completion",
        "",
        f"- Benchmark pairs: {len(merged)}",
        ("- Completed blinded adjudications: " f"{completed_n}"),
        (
            "- Needs translation: "
            f"{int((merged['ReviewerStatus'] == 'needs_translation').sum())}"
        ),
        (
            "- Unable to adjudicate: "
            f"{int((merged['ReviewerStatus'] == 'unable_to_adjudicate').sum())}"
        ),
        "",
        "## W1.2 parser benchmark",
        "",
        (
            "- Exact reviewer ↔ parser assertion agreement: "
            + (f"{100.0 * parser_exact:.1f}%" if not np.isnan(parser_exact) else "NA")
        ),
        "",
        (
            "The reviewer assertion is the blinded "
            "report-side reference. Challenge gold was "
            "revealed only after report interpretation."
        ),
        "",
        "## Label-specific report ↔ challenge mapping",
        "",
        (
            "| Label | Report-positive N | Positive PPV | "
            "Report-negative N | Negative NPV |"
        ),
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in trust_df.iterrows():

        ppv = (
            "NA"
            if pd.isna(row["ReportPositivePPV"])
            else (f"{100.0 * row['ReportPositivePPV']:.1f}%")
        )

        npv = (
            "NA"
            if pd.isna(row["ReportNegativeNPV"])
            else (f"{100.0 * row['ReportNegativeNPV']:.1f}%")
        )

        lines.append(
            (
                f"| {row['Label']} "
                f"| {int(row['ReviewerPositiveN'])} "
                f"| {ppv} "
                f"| {int(row['ReviewerNegativeN'])} "
                f"| {npv} |"
            )
        )

    lines.extend(
        [
            "",
            "## W2 gate",
            "",
            (
                "W1.3 still generates no pseudo-labels. "
                "Use the adjudicated label-specific PPV/NPV, "
                "severity mapping, and parser benchmark to "
                "choose and calibrate W2."
            ),
            "",
            (
                "The 120-pair benchmark is intentionally "
                "high-information and not prevalence-"
                "representative. Raw overall accuracy must not "
                "be interpreted as expected full-corpus "
                "performance."
            ),
            "",
        ]
    )

    with open(
        RESULT_ROOT / "W1_3_FINAL_REPORT.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))

    save_json(
        RESULT_ROOT / "w1_3_final_config.json",
        {
            "phase": "W1.3 blinded gold-report adjudication",
            "mode": "finalize",
            "benchmark_pairs": int(len(merged)),
            "completed_pairs": completed_n,
            "completed_review_path": str(COMPLETED_REVIEW_PATH),
            "private_answer_key_path": str(private_path),
            "nlp_model_trained": False,
            "pseudo_labels_generated": False,
            "outputs": [
                "10_W1_3_ADJUDICATED_MERGED.csv",
                "11_reviewer_vs_parser_summary.csv",
                "12_report_assertion_to_gold_mapping.csv",
                "13_label_report_trust_summary.csv",
                "14_severity_to_gold_mapping.csv",
                "15_language_review_summary.csv",
                "16_review_completion_summary.csv",
                "17_gold_mapping_case_counts.csv",
                "W1_3_FINAL_REPORT.md",
            ],
        },
    )

    print()
    print("=" * 80)
    print("W1.3 FINALIZE COMPLETE")
    print("=" * 80)
    print()
    print(f"Completed adjudications: " f"{completed_n}/{len(merged)}")
    print()
    print(f"Results directory:\n" f"{RESULT_ROOT}")
    print()
    print("Most important outputs:")
    for filename in [
        "W1_3_FINAL_REPORT.md",
        "13_label_report_trust_summary.csv",
        "12_report_assertion_to_gold_mapping.csv",
        "14_severity_to_gold_mapping.csv",
        "11_reviewer_vs_parser_summary.csv",
        "10_W1_3_ADJUDICATED_MERGED.csv",
    ]:
        print(f"  - {filename}")
    print()
    print("NO NLP model trained. " "NO pseudo-labels generated.")


# ============================================================
# 10. ENTRY POINT
# ============================================================


def resolve_mode() -> str:

    env_mode = (
        os.environ.get(
            "W13_MODE",
            "",
        )
        .strip()
        .casefold()
    )

    default_mode = (
        env_mode
        if env_mode
        in {
            "prepare",
            "finalize",
        }
        else "prepare"
    )

    parser = argparse.ArgumentParser(description=("W1.3 blinded report adjudication"))

    parser.add_argument(
        "--mode",
        choices=[
            "prepare",
            "finalize",
        ],
        default=default_mode,
    )

    # Robust inside Kaggle/Jupyter, where the kernel may add
    # unrelated command-line arguments.
    args, _ = parser.parse_known_args()

    return args.mode


def main() -> None:

    mode = resolve_mode()

    print("=" * 80)
    print("RSNA W1.3 - BLINDED GOLD-REPORT ADJUDICATION")
    print("=" * 80)
    print(f"Mode: {mode}")

    if mode == "prepare":
        prepare_mode()
    else:
        finalize_mode()


if __name__ == "__main__":
    main()
