# ============================================================
# RSNA KNEE ABNORMALITY DETECTION - W1
# Report Investigation / Weak-Supervision Preparation
# ============================================================
#
# Goal:
#   Investigate ALL training reports before building any NLP
#   pseudo-labeling model.
#
# This script does NOT train a report classifier.
#
# It answers:
#   1. How many reports are usable?
#   2. What scripts/languages are present?
#   3. Are gold (58) reports representative of the 4,349
#      unlabeled reports?
#   4. How long/structured are reports?
#   5. Are there duplicate / template-like reports?
#   6. What quality issues exist?
#   7. Which terms/ngrams are associated with each of the
#      12 gold labels?
#   8. Which report sentences are potentially useful for
#      weak-label rule development?
#
# Kaggle input:
#   /kaggle/input/competitions/rsna-knee-abnormality-detection
#
# Output:
#   /kaggle/working/rsna_w1/
#
# No internet is required.
# Optional language detectors are used only if already installed.
# ============================================================

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIG
# ============================================================

DATA_ROOT = Path("/kaggle/input/competitions/" "rsna-knee-abnormality-detection")

TRAIN_CSV = DATA_ROOT / "train.csv"

WORK_ROOT = Path("/kaggle/working/rsna_w1")

RESULT_ROOT = WORK_ROOT / "results"

WORK_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_TOTAL_STUDIES = 4407
EXPECTED_GOLD_STUDIES = 58
EXPECTED_UNLABELED_STUDIES = 4349


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


REPORT_COLUMN = "Report"
UID_COLUMN = "StudyInstanceUID"
SEX_COLUMN = "PatientSex"


# Descriptive term mining only.
TOP_TERMS_PER_LABEL = 40
TOP_CORPUS_TERMS = 100
TOP_LANGUAGE_EXAMPLES = 8

NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.96
NEAR_DUPLICATE_NEIGHBORS = 3

RANDOM_SEED = 42


# ============================================================
# 2. BASIC UTILITIES
# ============================================================


def safe_json_value(
    value: Any,
) -> Any:

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (np.floating,)):

        if np.isnan(value):

            return None

        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):

        return [safe_json_value(item) for item in value.tolist()]

    if pd.isna(value):

        return None

    return value


def save_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:

    clean = {key: safe_json_value(value) for key, value in payload.items()}

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean,
            f,
            indent=4,
            ensure_ascii=False,
        )


def normalize_whitespace(
    text: str,
) -> str:

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_report(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = normalize_whitespace(text)

    return text


def normalized_duplicate_key(
    text: str,
) -> str:

    text = normalize_report(text).casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def template_normalized_key(
    text: str,
) -> str:
    """
    A looser normalization used to detect templated reports.

    Numbers are replaced, repeated whitespace collapsed, and
    punctuation spacing normalized.

    We do NOT remove words or medical content.
    """

    text = normalized_duplicate_key(text)

    text = re.sub(
        r"\d+(?:[.,]\d+)?",
        "<num>",
        text,
    )

    text = re.sub(
        r"\s*([,:;.!?()\[\]])\s*",
        r"\1",
        text,
    )

    return text


def md5_text(
    text: str,
) -> str:

    return hashlib.md5(
        text.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


def split_sentences(
    text: str,
) -> List[str]:
    """
    Lightweight multilingual-ish sentence splitter.

    It intentionally avoids language-specific NLP packages.
    """

    text = normalize_report(text)

    if not text:

        return []

    parts = re.split(
        r"(?<=[.!?。！？])\s+|\n+",
        text,
    )

    parts = [part.strip() for part in parts if part.strip()]

    return parts


def unicode_word_tokens(
    text: str,
) -> List[str]:

    text = normalize_report(text)

    # \w is Unicode-aware in Python 3.
    tokens = re.findall(
        r"\b[^\W\d_][\w'-]*\b",
        text,
        flags=re.UNICODE,
    )

    return tokens


# ============================================================
# 3. SCRIPT / LANGUAGE INVESTIGATION
# ============================================================

SCRIPT_RANGES = {
    "Latin": [
        (0x0041, 0x024F),
        (0x1E00, 0x1EFF),
    ],
    "Cyrillic": [
        (0x0400, 0x052F),
        (0x2DE0, 0x2DFF),
        (0xA640, 0xA69F),
    ],
    "Greek": [
        (0x0370, 0x03FF),
        (0x1F00, 0x1FFF),
    ],
    "Arabic": [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
    ],
    "Hebrew": [
        (0x0590, 0x05FF),
    ],
    "Devanagari": [
        (0x0900, 0x097F),
    ],
    "Bengali": [
        (0x0980, 0x09FF),
    ],
    "Thai": [
        (0x0E00, 0x0E7F),
    ],
    "Hangul": [
        (0xAC00, 0xD7AF),
        (0x1100, 0x11FF),
    ],
    "Hiragana": [
        (0x3040, 0x309F),
    ],
    "Katakana": [
        (0x30A0, 0x30FF),
    ],
    "CJK": [
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
    ],
}


def char_script(
    ch: str,
) -> Optional[str]:

    code = ord(ch)

    for script, ranges in SCRIPT_RANGES.items():

        for start, end in ranges:

            if start <= code <= end:

                return script

    return None


def script_profile(
    text: str,
) -> Dict[str, Any]:

    counts = Counter()

    alpha_count = 0

    for ch in text:

        if not ch.isalpha():

            continue

        alpha_count += 1

        script = char_script(ch)

        if script is None:

            script = "Other"

        counts[script] += 1

    if alpha_count == 0:

        return {
            "dominant_script": "NoAlphabeticText",
            "script_confidence": 0.0,
            "script_counts": {},
        }

    dominant_script, dominant_count = counts.most_common(1)[0]

    confidence = dominant_count / alpha_count

    return {
        "dominant_script": dominant_script,
        "script_confidence": float(confidence),
        "script_counts": dict(counts),
    }


# ------------------------------------------------------------
# Optional installed language detectors
# ------------------------------------------------------------

LANGUAGE_BACKEND = None
LANGUAGE_DETECTOR = None


def initialize_language_detector() -> None:

    global LANGUAGE_BACKEND
    global LANGUAGE_DETECTOR

    # 1. lingua
    try:

        from lingua import (
            LanguageDetectorBuilder,
        )

        detector = LanguageDetectorBuilder.from_all_languages().build()

        LANGUAGE_BACKEND = "lingua"
        LANGUAGE_DETECTOR = detector

        return

    except Exception:

        pass

    # 2. langid
    try:

        import langid

        LANGUAGE_BACKEND = "langid"
        LANGUAGE_DETECTOR = langid

        return

    except Exception:

        pass

    # 3. langdetect
    try:

        from langdetect import (
            detect_langs,
            DetectorFactory,
        )

        DetectorFactory.seed = RANDOM_SEED

        LANGUAGE_BACKEND = "langdetect"
        LANGUAGE_DETECTOR = detect_langs

        return

    except Exception:

        pass

    LANGUAGE_BACKEND = "script_only_fallback"

    LANGUAGE_DETECTOR = None


def detect_language(
    text: str,
    dominant_script: str,
) -> Tuple[str, float]:

    text = normalize_report(text)

    if len(unicode_word_tokens(text)) < 3:

        return (
            "too_short",
            0.0,
        )

    if LANGUAGE_BACKEND == "lingua":

        try:

            language = LANGUAGE_DETECTOR.detect_language_of(text)

            if language is None:

                return (
                    "unknown",
                    0.0,
                )

            confidence_values = LANGUAGE_DETECTOR.compute_language_confidence_values(
                text
            )

            confidence = 0.0

            if confidence_values:

                confidence = float(confidence_values[0].value)

            return (
                language.iso_code_639_1.name.lower(),
                confidence,
            )

        except Exception:

            pass

    if LANGUAGE_BACKEND == "langid":

        try:

            code, score = LANGUAGE_DETECTOR.classify(text)

            # langid's raw score is not a calibrated
            # probability, so preserve it separately-ish.
            return (
                str(code),
                float(score),
            )

        except Exception:

            pass

    if LANGUAGE_BACKEND == "langdetect":

        try:

            candidates = LANGUAGE_DETECTOR(text)

            if not candidates:

                return (
                    "unknown",
                    0.0,
                )

            top = candidates[0]

            return (
                str(top.lang),
                float(top.prob),
            )

        except Exception:

            pass

    # Deterministic fallback: script family only.
    return (
        f"script_{dominant_script}",
        0.0,
    )


# ============================================================
# 4. REPORT QUALITY FLAGS
# ============================================================


def report_quality_flags(
    text: str,
) -> List[str]:

    flags = []

    normalized = normalize_report(text)

    char_count = len(normalized)

    word_count = len(unicode_word_tokens(normalized))

    if char_count == 0:

        flags.append("empty")

        return flags

    if word_count < 5:

        flags.append("very_short")

    if char_count > 10000:

        flags.append("very_long")

    if "\ufffd" in normalized:

        flags.append("unicode_replacement_character")

    control_count = sum(
        1
        for ch in normalized
        if (unicodedata.category(ch) == "Cc" and ch not in "\n\t")
    )

    if control_count > 0:

        flags.append("control_characters")

    alpha_count = sum(ch.isalpha() for ch in normalized)

    digit_count = sum(ch.isdigit() for ch in normalized)

    if char_count > 0 and digit_count / char_count > 0.40:

        flags.append("digit_heavy")

    if alpha_count == 0:

        flags.append("no_alphabetic_text")

    # Repeated identical line pattern.
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]

    if len(lines) >= 4:

        unique_ratio = len(set(lines)) / len(lines)

        if unique_ratio < 0.50:

            flags.append("repeated_lines")

    return flags


# ============================================================
# 5. GOLD LABEL STATUS
# ============================================================


def classify_label_status(
    row: pd.Series,
) -> str:

    values = row[LABEL_COLUMNS]

    non_missing = values.notna().sum()

    if non_missing == 0:

        return "unlabeled"

    if non_missing == len(LABEL_COLUMNS):

        return "gold"

    return "partial"


# ============================================================
# 6. LOAD DATA
# ============================================================

print("=" * 80)

print("RSNA W1 - REPORT INVESTIGATION")

print("=" * 80)


if not TRAIN_CSV.exists():

    raise FileNotFoundError(f"Could not find train.csv: " f"{TRAIN_CSV}")


train_df = pd.read_csv(TRAIN_CSV)


required_columns = [
    UID_COLUMN,
    REPORT_COLUMN,
] + LABEL_COLUMNS


missing_columns = [
    column for column in required_columns if column not in train_df.columns
]


if missing_columns:

    raise RuntimeError("train.csv is missing required " f"columns: {missing_columns}")


print(f"Rows: {len(train_df):,}")


if len(train_df) != EXPECTED_TOTAL_STUDIES:

    print(
        "WARNING: expected "
        f"{EXPECTED_TOTAL_STUDIES:,} studies "
        f"but found {len(train_df):,}."
    )


train_df["LabelStatus"] = train_df.apply(
    classify_label_status,
    axis=1,
)


status_counts = train_df["LabelStatus"].value_counts(dropna=False).to_dict()


print(f"Label status: {status_counts}")


# Strong assertion because prior dataset investigation
# established no partially labeled studies.
partial_count = int((train_df["LabelStatus"] == "partial").sum())


if partial_count > 0:

    raise RuntimeError(
        "W1 expected no partially labeled " f"studies, but found {partial_count}."
    )


# ============================================================
# 7. INITIALIZE OPTIONAL LANGUAGE DETECTOR
# ============================================================

initialize_language_detector()


print(f"Language detector backend: " f"{LANGUAGE_BACKEND}")


# GPU visibility is reported for planning only.
# W1 is intentionally CPU-heavy descriptive analysis.
try:

    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

except Exception:

    gpu_count = 0


print(f"Visible CUDA GPUs: {gpu_count}")

print("W1 does not require GPU inference.")


# ============================================================
# 8. REPORT-LEVEL INVENTORY
# ============================================================

inventory_records = []


for row_idx, row in train_df.iterrows():

    uid = str(row[UID_COLUMN])

    raw_report = "" if pd.isna(row[REPORT_COLUMN]) else str(row[REPORT_COLUMN])

    normalized = normalize_report(raw_report)

    duplicate_normalized = normalized_duplicate_key(normalized)

    template_normalized = template_normalized_key(normalized)

    tokens = unicode_word_tokens(normalized)

    sentences = split_sentences(normalized)

    profile = script_profile(normalized)

    language, language_confidence = detect_language(
        normalized,
        profile["dominant_script"],
    )

    flags = report_quality_flags(normalized)

    inventory_records.append(
        {
            UID_COLUMN: uid,
            "LabelStatus": row["LabelStatus"],
            SEX_COLUMN: (row[SEX_COLUMN] if SEX_COLUMN in train_df.columns else np.nan),
            "ReportPresent": bool(normalized),
            "CharacterCount": len(normalized),
            "WordCount": len(tokens),
            "SentenceCount": len(sentences),
            "LineCount": (len(normalized.split("\n")) if normalized else 0),
            "DominantScript": profile["dominant_script"],
            "ScriptConfidence": profile["script_confidence"],
            "Language": language,
            "LanguageConfidence": language_confidence,
            "LanguageBackend": LANGUAGE_BACKEND,
            "QualityFlagCount": len(flags),
            "QualityFlags": "|".join(flags),
            "NormalizedReportMD5": md5_text(duplicate_normalized),
            "TemplateReportMD5": md5_text(template_normalized),
            "NormalizedReport": normalized,
        }
    )


inventory_df = pd.DataFrame(inventory_records)


inventory_path = RESULT_ROOT / "01_report_inventory.csv"


inventory_df.to_csv(
    inventory_path,
    index=False,
)


# ============================================================
# 9. CORPUS SUMMARY
# ============================================================

report_present_count = int(inventory_df["ReportPresent"].sum())


missing_report_count = len(inventory_df) - report_present_count


summary = {
    "total_studies": int(len(train_df)),
    "gold_studies": int((train_df["LabelStatus"] == "gold").sum()),
    "unlabeled_studies": int((train_df["LabelStatus"] == "unlabeled").sum()),
    "partial_studies": partial_count,
    "reports_present": report_present_count,
    "reports_missing": int(missing_report_count),
    "language_backend": LANGUAGE_BACKEND,
    "visible_gpu_count": gpu_count,
    "median_report_characters": float(inventory_df["CharacterCount"].median()),
    "median_report_words": float(inventory_df["WordCount"].median()),
    "median_report_sentences": float(inventory_df["SentenceCount"].median()),
}


save_json(
    RESULT_ROOT / "02_report_summary.json",
    summary,
)


# ============================================================
# 10. SCRIPT / LANGUAGE DISTRIBUTIONS
# ============================================================


def distribution_table(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    rows = []

    total = max(
        1,
        len(df),
    )

    for value, count in df[column].fillna("<NA>").value_counts(dropna=False).items():

        rows.append(
            {
                column: value,
                "Count": int(count),
                "Percent": float(100.0 * count / total),
            }
        )

    return pd.DataFrame(rows)


script_distribution_df = distribution_table(
    inventory_df,
    "DominantScript",
)


script_distribution_df.to_csv(
    RESULT_ROOT / "03_script_distribution.csv",
    index=False,
)


language_distribution_df = distribution_table(
    inventory_df,
    "Language",
)


language_distribution_df.to_csv(
    RESULT_ROOT / "04_language_distribution.csv",
    index=False,
)


# Gold vs unlabeled language/script representation.
language_status_df = (
    inventory_df.groupby(
        [
            "LabelStatus",
            "Language",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="Count")
)


language_status_df.to_csv(
    RESULT_ROOT / "05_language_by_label_status.csv",
    index=False,
)


script_status_df = (
    inventory_df.groupby(
        [
            "LabelStatus",
            "DominantScript",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="Count")
)


script_status_df.to_csv(
    RESULT_ROOT / "06_script_by_label_status.csv",
    index=False,
)


# ============================================================
# 11. REPORT LENGTH STATISTICS
# ============================================================

length_rows = []


for status, group in inventory_df.groupby("LabelStatus"):

    for metric in [
        "CharacterCount",
        "WordCount",
        "SentenceCount",
        "LineCount",
    ]:

        values = group[metric].astype(float).values

        length_rows.append(
            {
                "LabelStatus": status,
                "Metric": metric,
                "Count": int(len(values)),
                "Mean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "P05": float(
                    np.percentile(
                        values,
                        5,
                    )
                ),
                "P25": float(
                    np.percentile(
                        values,
                        25,
                    )
                ),
                "P75": float(
                    np.percentile(
                        values,
                        75,
                    )
                ),
                "P95": float(
                    np.percentile(
                        values,
                        95,
                    )
                ),
                "Min": float(np.min(values)),
                "Max": float(np.max(values)),
            }
        )


length_stats_df = pd.DataFrame(length_rows)


length_stats_df.to_csv(
    RESULT_ROOT / "07_report_length_stats.csv",
    index=False,
)


# ============================================================
# 12. QUALITY FLAGS
# ============================================================

quality_rows = []


for _, row in inventory_df.iterrows():

    flag_text = str(row["QualityFlags"] or "")

    flags = [flag for flag in flag_text.split("|") if flag]

    if not flags:

        continue

    for flag in flags:

        quality_rows.append(
            {
                UID_COLUMN: row[UID_COLUMN],
                "LabelStatus": row["LabelStatus"],
                "Language": row["Language"],
                "DominantScript": row["DominantScript"],
                "QualityFlag": flag,
                "CharacterCount": row["CharacterCount"],
                "WordCount": row["WordCount"],
                "ReportPreview": str(row["NormalizedReport"])[:500],
            }
        )


quality_df = pd.DataFrame(quality_rows)


quality_df.to_csv(
    RESULT_ROOT / "08_report_quality_flags.csv",
    index=False,
)


# ============================================================
# 13. EXACT / NORMALIZED / TEMPLATE DUPLICATES
# ============================================================

duplicate_rows = []


for duplicate_kind, hash_column in [
    (
        "normalized_exact",
        "NormalizedReportMD5",
    ),
    (
        "template_normalized",
        "TemplateReportMD5",
    ),
]:

    grouped = inventory_df.groupby(hash_column)

    for hash_value, group in grouped:

        if len(group) <= 1:

            continue

        reports = group["NormalizedReport"].astype(str).tolist()

        duplicate_rows.append(
            {
                "DuplicateKind": duplicate_kind,
                "Hash": hash_value,
                "StudyCount": int(len(group)),
                "GoldCount": int((group["LabelStatus"] == "gold").sum()),
                "UnlabeledCount": int((group["LabelStatus"] == "unlabeled").sum()),
                "Languages": "|".join(sorted(set(group["Language"].astype(str)))),
                "StudyInstanceUIDs": "|".join(group[UID_COLUMN].astype(str).tolist()),
                "ExampleReport": reports[0][:1200],
            }
        )


duplicate_df = pd.DataFrame(duplicate_rows)


duplicate_df.to_csv(
    RESULT_ROOT / "09_duplicate_report_groups.csv",
    index=False,
)


# ============================================================
# 14. OPTIONAL NEAR-DUPLICATE SEARCH
# ============================================================

near_duplicate_rows = []


usable_for_similarity = inventory_df[inventory_df["WordCount"] >= 5].reset_index(
    drop=True
)


try:

    from sklearn.feature_extraction.text import (
        TfidfVectorizer,
    )

    from sklearn.neighbors import (
        NearestNeighbors,
    )

    similarity_texts = (
        usable_for_similarity["NormalizedReport"].fillna("").astype(str).tolist()
    )

    if len(similarity_texts) >= 2:

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(
                3,
                5,
            ),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(similarity_texts)

        neighbors = NearestNeighbors(
            n_neighbors=min(
                NEAR_DUPLICATE_NEIGHBORS,
                len(similarity_texts),
            ),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        neighbors.fit(matrix)

        distances, indices = neighbors.kneighbors(matrix)

        seen_pairs = set()

        for row_idx in range(len(similarity_texts)):

            uid_a = usable_for_similarity.iloc[row_idx][UID_COLUMN]

            for neighbor_position in range(
                1,
                indices.shape[1],
            ):

                other_idx = int(indices[row_idx, neighbor_position])

                uid_b = usable_for_similarity.iloc[other_idx][UID_COLUMN]

                pair = tuple(
                    sorted(
                        [
                            str(uid_a),
                            str(uid_b),
                        ]
                    )
                )

                if pair in seen_pairs:

                    continue

                seen_pairs.add(pair)

                similarity = 1.0 - float(distances[row_idx, neighbor_position])

                if similarity < NEAR_DUPLICATE_SIMILARITY_THRESHOLD:

                    continue

                report_a = str(usable_for_similarity.iloc[row_idx]["NormalizedReport"])

                report_b = str(
                    usable_for_similarity.iloc[other_idx]["NormalizedReport"]
                )

                near_duplicate_rows.append(
                    {
                        "StudyInstanceUID_A": uid_a,
                        "StudyInstanceUID_B": uid_b,
                        "Similarity": similarity,
                        "Status_A": usable_for_similarity.iloc[row_idx]["LabelStatus"],
                        "Status_B": usable_for_similarity.iloc[other_idx][
                            "LabelStatus"
                        ],
                        "Language_A": usable_for_similarity.iloc[row_idx]["Language"],
                        "Language_B": usable_for_similarity.iloc[other_idx]["Language"],
                        "Report_A": report_a[:1000],
                        "Report_B": report_b[:1000],
                    }
                )

except Exception as exc:

    print("Near-duplicate analysis skipped: " f"{exc}")


near_duplicate_df = pd.DataFrame(near_duplicate_rows)


near_duplicate_df.to_csv(
    RESULT_ROOT / "10_near_duplicate_reports.csv",
    index=False,
)


# ============================================================
# 15. LANGUAGE / SCRIPT EXAMPLES
# ============================================================

example_rows = []


rng = np.random.default_rng(RANDOM_SEED)


for language, group in inventory_df.groupby(
    "Language",
    dropna=False,
):

    candidate_indices = np.arange(len(group))

    if len(candidate_indices) > TOP_LANGUAGE_EXAMPLES:

        chosen = rng.choice(
            candidate_indices,
            size=TOP_LANGUAGE_EXAMPLES,
            replace=False,
        )

    else:

        chosen = candidate_indices

    sample = group.iloc[chosen]

    for _, row in sample.iterrows():

        example_rows.append(
            {
                "Language": language,
                "DominantScript": row["DominantScript"],
                "LabelStatus": row["LabelStatus"],
                UID_COLUMN: row[UID_COLUMN],
                "WordCount": row["WordCount"],
                "ReportPreview": str(row["NormalizedReport"])[:1500],
            }
        )


examples_df = pd.DataFrame(example_rows)


examples_df.to_csv(
    RESULT_ROOT / "11_language_report_examples.csv",
    index=False,
)


# ============================================================
# 16. GOLD COHORT REPORT + LABEL TABLE
# ============================================================

gold_source_df = train_df[train_df["LabelStatus"] == "gold"].copy()


gold_inventory_df = inventory_df[inventory_df["LabelStatus"] == "gold"][
    [
        UID_COLUMN,
        "Language",
        "DominantScript",
        "CharacterCount",
        "WordCount",
        "SentenceCount",
        "NormalizedReport",
    ]
]


gold_report_df = gold_source_df.merge(
    gold_inventory_df,
    on=UID_COLUMN,
    how="left",
)


gold_report_df.to_csv(
    RESULT_ROOT / "12_gold_reports_with_labels.csv",
    index=False,
)


# ============================================================
# 17. GOLD-LABEL TERM DISCOVERY
# ============================================================
#
# Descriptive only.
#
# We use TF-IDF on the 58 gold reports and compute:
#
#   mean TFIDF among positives
#       -
#   mean TFIDF among negatives
#
# for each label.
#
# This is NOT a classifier and these terms should not be
# treated as causal or clinically validated.
# ============================================================

term_rows = []


try:

    from sklearn.feature_extraction.text import (
        TfidfVectorizer,
    )

    gold_texts = gold_report_df["NormalizedReport"].fillna("").astype(str).tolist()

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=(r"(?u)\b[^\W\d_]" r"[\w'-]{1,}\b"),
        lowercase=True,
        ngram_range=(
            1,
            3,
        ),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )

    gold_word_matrix = word_vectorizer.fit_transform(gold_texts)

    vocabulary = np.asarray(word_vectorizer.get_feature_names_out())

    for label in LABEL_COLUMNS:

        targets = gold_report_df[label].astype(int).values

        positive_mask = targets == 1

        negative_mask = targets == 0

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:

            continue

        positive_mean = np.asarray(gold_word_matrix[positive_mask].mean(axis=0)).ravel()

        negative_mean = np.asarray(gold_word_matrix[negative_mask].mean(axis=0)).ravel()

        delta = positive_mean - negative_mean

        positive_order = np.argsort(-delta)

        negative_order = np.argsort(delta)

        for rank, feature_idx in enumerate(
            positive_order[:TOP_TERMS_PER_LABEL],
            start=1,
        ):

            term_rows.append(
                {
                    "Label": label,
                    "Direction": "positive_association",
                    "Rank": rank,
                    "Term": vocabulary[feature_idx],
                    "PositiveMeanTFIDF": float(positive_mean[feature_idx]),
                    "NegativeMeanTFIDF": float(negative_mean[feature_idx]),
                    "Delta": float(delta[feature_idx]),
                }
            )

        for rank, feature_idx in enumerate(
            negative_order[:TOP_TERMS_PER_LABEL],
            start=1,
        ):

            term_rows.append(
                {
                    "Label": label,
                    "Direction": "negative_association",
                    "Rank": rank,
                    "Term": vocabulary[feature_idx],
                    "PositiveMeanTFIDF": float(positive_mean[feature_idx]),
                    "NegativeMeanTFIDF": float(negative_mean[feature_idx]),
                    "Delta": float(delta[feature_idx]),
                }
            )

except Exception as exc:

    print("Gold term discovery skipped: " f"{exc}")


term_df = pd.DataFrame(term_rows)


term_df.to_csv(
    RESULT_ROOT / "13_gold_label_term_associations.csv",
    index=False,
)


# ============================================================
# 18. CONCORDANCE SENTENCES FOR TOP GOLD TERMS
# ============================================================

concordance_rows = []


if len(term_df) > 0:

    top_positive_terms = term_df[
        (term_df["Direction"] == "positive_association") & (term_df["Rank"] <= 15)
    ]

    for _, term_row in top_positive_terms.iterrows():

        label = term_row["Label"]

        term = str(term_row["Term"]).casefold()

        positive_gold = gold_report_df[gold_report_df[label].astype(int) == 1]

        matches = 0

        for _, report_row in positive_gold.iterrows():

            sentences = split_sentences(str(report_row["NormalizedReport"]))

            for sentence in sentences:

                if term not in (sentence.casefold()):

                    continue

                concordance_rows.append(
                    {
                        "Label": label,
                        "Term": term_row["Term"],
                        "TermRank": int(term_row["Rank"]),
                        UID_COLUMN: report_row[UID_COLUMN],
                        "Language": report_row["Language"],
                        "Sentence": sentence[:1800],
                    }
                )

                matches += 1

                if matches >= 5:

                    break

            if matches >= 5:

                break


concordance_df = pd.DataFrame(concordance_rows)


concordance_df.to_csv(
    RESULT_ROOT / "14_gold_term_sentence_concordance.csv",
    index=False,
)


# ============================================================
# 19. CORPUS-WIDE FREQUENT TERMS BY LANGUAGE
# ============================================================

corpus_term_rows = []


try:

    from sklearn.feature_extraction.text import (
        CountVectorizer,
    )

    for language, group in inventory_df.groupby(
        "Language",
        dropna=False,
    ):

        texts = group["NormalizedReport"].fillna("").astype(str).tolist()

        if len(texts) < 3:

            continue

        vectorizer = CountVectorizer(
            analyzer="word",
            token_pattern=(r"(?u)\b[^\W\d_]" r"[\w'-]{1,}\b"),
            lowercase=True,
            ngram_range=(
                1,
                2,
            ),
            min_df=2,
            max_features=10000,
        )

        try:

            matrix = vectorizer.fit_transform(texts)

        except Exception:

            continue

        terms = np.asarray(vectorizer.get_feature_names_out())

        counts = np.asarray(matrix.sum(axis=0)).ravel()

        order = np.argsort(-counts)

        for rank, idx in enumerate(
            order[:TOP_CORPUS_TERMS],
            start=1,
        ):

            corpus_term_rows.append(
                {
                    "Language": language,
                    "Rank": rank,
                    "Term": terms[idx],
                    "CorpusCount": int(counts[idx]),
                    "ReportCountInLanguage": int(len(group)),
                }
            )

except Exception as exc:

    print("Corpus term analysis skipped: " f"{exc}")


corpus_terms_df = pd.DataFrame(corpus_term_rows)


corpus_terms_df.to_csv(
    RESULT_ROOT / "15_frequent_terms_by_language.csv",
    index=False,
)


# ============================================================
# 20. SIMPLE NEGATION / UNCERTAINTY SURVEY
# ============================================================
#
# This is intentionally only a corpus survey, NOT the final
# negation engine.
#
# It helps reveal which common cue words actually appear.
# ============================================================

CUE_TERMS = {
    "english_negation": [
        "no",
        "not",
        "without",
        "negative for",
        "absent",
        "none",
        "denies",
    ],
    "english_uncertainty": [
        "possible",
        "possibly",
        "probable",
        "probably",
        "may represent",
        "may be",
        "cannot exclude",
        "can't exclude",
        "suspicious for",
        "suggestive of",
        "likely",
        "unlikely",
    ],
    "english_history": [
        "history of",
        "previous",
        "prior",
        "status post",
        "postoperative",
        "post-operative",
    ],
}


cue_rows = []


for cue_group, cues in CUE_TERMS.items():

    for cue in cues:

        count = 0
        gold_count = 0
        unlabeled_count = 0
        examples = []

        for _, row in inventory_df.iterrows():

            report = str(row["NormalizedReport"])

            if cue.casefold() not in (report.casefold()):

                continue

            count += 1

            if row["LabelStatus"] == "gold":

                gold_count += 1

            else:

                unlabeled_count += 1

            if len(examples) < 5:

                examples.append(
                    {
                        UID_COLUMN: row[UID_COLUMN],
                        "Language": row["Language"],
                        "Preview": report[:1000],
                    }
                )

        cue_rows.append(
            {
                "CueGroup": cue_group,
                "Cue": cue,
                "TotalReportCount": count,
                "GoldReportCount": gold_count,
                "UnlabeledReportCount": unlabeled_count,
                "ExampleJSON": json.dumps(
                    examples,
                    ensure_ascii=False,
                ),
            }
        )


cue_df = pd.DataFrame(cue_rows)


cue_df.to_csv(
    RESULT_ROOT / "16_negation_uncertainty_cue_survey.csv",
    index=False,
)


# ============================================================
# 21. GOLD LABEL COUNTS + REPORT REPRESENTATIVENESS
# ============================================================

label_rows = []


for label in LABEL_COLUMNS:

    positive_count = int(gold_report_df[label].astype(int).sum())

    negative_count = int(len(gold_report_df) - positive_count)

    positive_reports = gold_report_df[gold_report_df[label].astype(int) == 1]

    negative_reports = gold_report_df[gold_report_df[label].astype(int) == 0]

    label_rows.append(
        {
            "Label": label,
            "PositiveCount": positive_count,
            "NegativeCount": negative_count,
            "PositiveMedianWords": float(positive_reports["WordCount"].median()),
            "NegativeMedianWords": float(negative_reports["WordCount"].median()),
            "PositiveLanguages": "|".join(
                sorted(set(positive_reports["Language"].astype(str)))
            ),
            "NegativeLanguages": "|".join(
                sorted(set(negative_reports["Language"].astype(str)))
            ),
        }
    )


gold_label_summary_df = pd.DataFrame(label_rows)


gold_label_summary_df.to_csv(
    RESULT_ROOT / "17_gold_label_report_summary.csv",
    index=False,
)


# ============================================================
# 22. HUMAN-READABLE MARKDOWN SUMMARY
# ============================================================

language_top = language_distribution_df.head(15)


script_top = script_distribution_df.head(15)


duplicate_group_count = int(len(duplicate_df))


near_duplicate_pair_count = int(len(near_duplicate_df))


quality_flagged_reports = int((inventory_df["QualityFlagCount"] > 0).sum())


markdown_lines = [
    "# RSNA W1 Report Investigation",
    "",
    "## Dataset inventory",
    "",
    f"- Total studies: {len(train_df):,}",
    ("- Gold-labeled studies: " f"{summary['gold_studies']:,}"),
    ("- Fully unlabeled studies: " f"{summary['unlabeled_studies']:,}"),
    ("- Reports present: " f"{report_present_count:,}"),
    ("- Reports missing: " f"{missing_report_count:,}"),
    ("- Language backend: " f"`{LANGUAGE_BACKEND}`"),
    "",
    "## Report size",
    "",
    ("- Median characters: " f"{summary['median_report_characters']:.1f}"),
    ("- Median words: " f"{summary['median_report_words']:.1f}"),
    ("- Median sentences: " f"{summary['median_report_sentences']:.1f}"),
    "",
    "## Dominant scripts",
    "",
]


for _, row in script_top.iterrows():

    markdown_lines.append(
        (
            f"- {row['DominantScript']}: "
            f"{int(row['Count']):,} "
            f"({row['Percent']:.2f}%)"
        )
    )


markdown_lines.extend(
    [
        "",
        "## Detected languages / language families",
        "",
    ]
)


for _, row in language_top.iterrows():

    markdown_lines.append(
        (f"- {row['Language']}: " f"{int(row['Count']):,} " f"({row['Percent']:.2f}%)")
    )


markdown_lines.extend(
    [
        "",
        "## Quality / duplication",
        "",
        ("- Reports with one or more quality flags: " f"{quality_flagged_reports:,}"),
        ("- Exact/template duplicate groups: " f"{duplicate_group_count:,}"),
        (
            "- Near-duplicate pairs above threshold "
            f"{NEAR_DUPLICATE_SIMILARITY_THRESHOLD:.2f}: "
            f"{near_duplicate_pair_count:,}"
        ),
        "",
        "## W1 interpretation checklist",
        "",
        ("1. Compare gold vs unlabeled language/script " "distribution."),
        ("2. Inspect duplicate/template reports before " "pseudo-labeling."),
        (
            "3. Inspect `13_gold_label_term_associations.csv` "
            "and concordance sentences."
        ),
        (
            "4. Determine which languages need dedicated "
            "negation/uncertainty handling."
        ),
        (
            "5. Do not build W2 until report terminology and "
            "gold representativeness are understood."
        ),
        "",
        "## Important limitation",
        "",
        (
            "W1 term associations are descriptive because the "
            "gold cohort contains only 58 studies. They are not "
            "a report classifier and must not be treated as "
            "validated pseudo-labeling rules."
        ),
        "",
    ]
)


markdown_path = RESULT_ROOT / "W1_REPORT.md"


with open(
    markdown_path,
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(markdown_lines))


# ============================================================
# 23. SAVE W1 CONFIG
# ============================================================

config = {
    "data_root": str(DATA_ROOT),
    "train_csv": str(TRAIN_CSV),
    "total_studies": int(len(train_df)),
    "expected_total_studies": EXPECTED_TOTAL_STUDIES,
    "gold_studies": int(summary["gold_studies"]),
    "unlabeled_studies": int(summary["unlabeled_studies"]),
    "report_investigation_only": True,
    "nlp_model_trained": False,
    "language_backend": LANGUAGE_BACKEND,
    "visible_gpu_count": gpu_count,
    "top_terms_per_label": TOP_TERMS_PER_LABEL,
    "near_duplicate_similarity_threshold": NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
    "labels": LABEL_COLUMNS,
    "outputs": [
        "01_report_inventory.csv",
        "02_report_summary.json",
        "03_script_distribution.csv",
        "04_language_distribution.csv",
        "05_language_by_label_status.csv",
        "06_script_by_label_status.csv",
        "07_report_length_stats.csv",
        "08_report_quality_flags.csv",
        "09_duplicate_report_groups.csv",
        "10_near_duplicate_reports.csv",
        "11_language_report_examples.csv",
        "12_gold_reports_with_labels.csv",
        "13_gold_label_term_associations.csv",
        "14_gold_term_sentence_concordance.csv",
        "15_frequent_terms_by_language.csv",
        "16_negation_uncertainty_cue_survey.csv",
        "17_gold_label_report_summary.csv",
        "W1_REPORT.md",
    ],
}


save_json(
    RESULT_ROOT / "w1_config.json",
    config,
)


# ============================================================
# 24. DONE
# ============================================================

print()
print("=" * 80)
print("W1 REPORT INVESTIGATION COMPLETE")
print("=" * 80)
print()

print(f"Results directory:\n" f"{RESULT_ROOT}")

print()

print("Most important files to review:")

for filename in [
    "W1_REPORT.md",
    "02_report_summary.json",
    "04_language_distribution.csv",
    "05_language_by_label_status.csv",
    "09_duplicate_report_groups.csv",
    "10_near_duplicate_reports.csv",
    "13_gold_label_term_associations.csv",
    "14_gold_term_sentence_concordance.csv",
    "16_negation_uncertainty_cue_survey.csv",
    "17_gold_label_report_summary.csv",
]:

    print(f"  - {filename}")

print()
print("W1 intentionally trained NO NLP model.")
