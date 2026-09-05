# ============================================================
# RSNA KNEE ABNORMALITY DETECTION - W2
# Multilingual Structured Report Labeler
# ============================================================
#
# W1-W1.3 established:
#   * all 4,407 training studies have reports
#   * 58 studies have complete 12-label gold annotations
#   * report assertions are NOT equivalent to challenge labels
#   * severity/type matters for several targets
#   * not_mentioned must remain UNKNOWN, never negative
#   * W1.2 rules have useful polarity but insufficient recall
#
# W2 therefore separates:
#
#   Stage A: REPORT UNDERSTANDING
#       report -> rule evidence + multilingual NLI evidence
#              -> structured assertion / severity / confidence
#
#   Stage B: CHALLENGE ONTOLOGY MAPPING
#       structured evidence -> P(challenge gold = 1)
#
# Stage B is fitted only on the 58 gold studies, with strongly
# regularized per-label logistic models. Cross-fitted OOF
# predictions are written before any full-data fit is used.
#
# IMPORTANT:
#   - W1.3's 120-case stress-test PPV/NPV values are NOT used
#     as direct pseudo-label weights.
#   - not_mentioned remains unavailable for pseudo-supervision.
#   - W2 creates candidate soft labels only in FULL mode.
#   - W2 does NOT train the MRI image model.
#
# Modes
# -----
# benchmark:
#   Evaluate Stage A on the 120 W1.3 adjudicated cases.
#
# validate (DEFAULT):
#   1) benchmark Stage A on W1.3
#   2) extract all 58 x 12 gold report features
#   3) cross-fit Stage B challenge mapping
#   4) save validation metrics + fitted calibrators
#
# full:
#   1) extract gold features and fit Stage B
#   2) extract all 4,407 x 12 report-label pairs
#   3) create structured report evidence
#   4) create candidate calibrated soft labels
#   5) keep not_mentioned as NaN / unavailable
#
# Kaggle environment overrides
# ----------------------------
# W2_MODE                 benchmark | validate | full
# RSNA_DATA_ROOT          competition root
# W2_W1_SOURCE            W1 results dir or rsna_w1.zip
# W2_W13_SOURCE           W1.3 dir or rsna_w1_3_zip_file.zip
# W2_WORK_ROOT            output root
# W2_MODEL_PATH           local HF model directory
# W2_MODEL_ID             HF model id fallback
# W2_LOCAL_FILES_ONLY     1 = never download
# W2_USE_MULTI_GPU        1 = DataParallel when >=2 GPUs
# W2_BATCH_SIZE           default 32
# W2_MAX_LENGTH           default 192
# W2_PAIR_CHUNK_SIZE      default 256
# W2_MAX_UNITS            default 40
#
# Default semantic model:
#   MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
#
# The model is intentionally run in FP32 because the model
# card warns that mDeBERTa FP16 may be unsupported/problematic.
# Two T4s are used via DataParallel when available.
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1. W2 CONFIG
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "RSNA_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)

TRAIN_CSV = DATA_ROOT / "train.csv"

WORK_ROOT = Path(
    os.environ.get(
        "W2_WORK_ROOT",
        "/kaggle/working/rsna_w2",
    )
)

RESULT_ROOT = WORK_ROOT / "results"
CACHE_ROOT = WORK_ROOT / "cache"
MODEL_ROOT = WORK_ROOT / "models"

for directory in [
    WORK_ROOT,
    RESULT_ROOT,
    CACHE_ROOT,
    MODEL_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

UID_COLUMN = "StudyInstanceUID"
REPORT_COLUMN = "Report"

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

EXPECTED_STUDIES = 4407
EXPECTED_GOLD_STUDIES = 58
EXPECTED_UNLABELED_STUDIES = 4349
EXPECTED_W13_CASES = 120

CONTEXT_WINDOW_CHARS = 100

MODEL_ID = os.environ.get(
    "W2_MODEL_ID",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
)

MODEL_PATH = os.environ.get(
    "W2_MODEL_PATH",
    "",
).strip()

LOCAL_FILES_ONLY = (
    os.environ.get(
        "W2_LOCAL_FILES_ONLY",
        "0",
    )
    == "1"
)

USE_MULTI_GPU = (
    os.environ.get(
        "W2_USE_MULTI_GPU",
        "1",
    )
    == "1"
)

NLI_BATCH_SIZE = int(
    os.environ.get(
        "W2_BATCH_SIZE",
        "32",
    )
)

NLI_MAX_LENGTH = int(
    os.environ.get(
        "W2_MAX_LENGTH",
        "192",
    )
)

PAIR_CHUNK_SIZE = int(
    os.environ.get(
        "W2_PAIR_CHUNK_SIZE",
        "256",
    )
)

MAX_UNITS_PER_REPORT = int(
    os.environ.get(
        "W2_MAX_UNITS",
        "40",
    )
)

# Conservative fixed semantic thresholds.
# These are not fitted to the 120-case W1.3 benchmark.
SEMANTIC_STRONG_THRESHOLD = 0.72
SEMANTIC_RELATED_THRESHOLD = 0.70
SEMANTIC_WEAK_THRESHOLD = 0.48
SEMANTIC_MARGIN = 0.18
SEMANTIC_CONFLICT_THRESHOLD = 0.72

# Semantic NLI is a recall-recovery channel, not a replacement
# for W1.2's explicit rule decisions. A high threshold is used
# before semantic evidence may recover a rule "not_mentioned".
SEMANTIC_RECOVERY_MIN_CONFIDENCE = 0.90

# Stage B mapping.
CALIBRATION_C = 0.10
CV_SPLITS = 5
CV_REPEATS = 5
CV_RANDOM_STATE = 42042

# Conservative per-label pseudo-label eligibility gate.
# FULL mode only consumes evidence-available rows, so each label
# must demonstrate useful calibration in that exact subset before
# its soft labels can enter the candidate pool.
PSEUDO_GATE_MIN_EVIDENCE_N = 15
PSEUDO_GATE_MIN_EVIDENCE_AUROC = 0.55
PSEUDO_GATE_MIN_BRIER_IMPROVEMENT = 0.0

# Do not use a challenge probability if Stage A finds no report
# evidence. This keeps not_mentioned = UNKNOWN.
UNKNOWN_ASSERTION = "not_mentioned"

ASSERTION_VALUES = [
    "positive",
    "negative",
    "uncertain",
    "mixed",
    "related_abnormality",
    "mentioned_neutral",
    "not_mentioned",
]

# W1.2 rule confidence means confidence that the report-side
# assertion was correctly recognized; it is NOT challenge-label
# probability.
RULE_ASSERTION_CONFIDENCE = {
    "positive": 0.95,
    "negative": 0.95,
    "uncertain": 0.75,
    "mixed": 0.70,
    "related_abnormality": 0.85,
    "mentioned_neutral": 0.55,
    "not_mentioned": 0.00,
}


# ============================================================
# 1A. ZERO-SHOT NLI HYPOTHESES
# ============================================================

DIRECT_HYPOTHESES = {
    "ACL": (
        "The anterior cruciate ligament (ACL) has an injury, "
        "sprain, tear, or rupture."
    ),
    "MCL": (
        "The medial collateral ligament (MCL) has an injury, "
        "sprain, tear, or rupture."
    ),
    "Medial Meniscus": ("The medial meniscus has a tear or rupture."),
    "Lateral Meniscus": ("The lateral meniscus has a tear or rupture."),
    "Medial OA": (
        "There is osteoarthritis, arthrosis, or gonarthrosis "
        "in the medial knee compartment."
    ),
    "Lateral OA": (
        "There is osteoarthritis, arthrosis, or gonarthrosis "
        "in the lateral knee compartment."
    ),
    "PF OA": (
        "There is osteoarthritis or arthrosis in the " "patellofemoral compartment."
    ),
    "Effusion": ("There is knee joint effusion or increased " "intra-articular fluid."),
    "Synovitis": ("There is synovitis in the knee."),
    "Baker's": ("There is a Baker's cyst or popliteal cyst."),
    "Contusion": ("There is a bone contusion or bone bruise in the knee."),
    "Fracture": ("There is a fracture in the knee."),
}


NEGATIVE_HYPOTHESES = {
    "ACL": (
        "The anterior cruciate ligament (ACL) is intact and "
        "there is no ACL injury, sprain, tear, or rupture."
    ),
    "MCL": (
        "The medial collateral ligament (MCL) is intact and "
        "there is no MCL injury, sprain, tear, or rupture."
    ),
    "Medial Meniscus": ("The medial meniscus is intact and has no tear or " "rupture."),
    "Lateral Meniscus": (
        "The lateral meniscus is intact and has no tear or " "rupture."
    ),
    "Medial OA": (
        "There is no osteoarthritis, arthrosis, or gonarthrosis "
        "in the medial knee compartment."
    ),
    "Lateral OA": (
        "There is no osteoarthritis, arthrosis, or gonarthrosis "
        "in the lateral knee compartment."
    ),
    "PF OA": (
        "There is no osteoarthritis or arthrosis in the " "patellofemoral compartment."
    ),
    "Effusion": (
        "There is no knee joint effusion and no increased " "intra-articular fluid."
    ),
    "Synovitis": ("There is no synovitis in the knee."),
    "Baker's": ("There is no Baker's cyst or popliteal cyst."),
    "Contusion": ("There is no bone contusion or bone bruise in the knee."),
    "Fracture": ("There is no fracture in the knee."),
}


RELATED_HYPOTHESES = {
    "ACL": (
        "The report describes abnormal or degenerative signal "
        "of the ACL without a definite ACL tear or rupture."
    ),
    "MCL": (
        "The report describes abnormality around the MCL "
        "without a definite MCL tear, rupture, or sprain."
    ),
    "Medial Meniscus": (
        "The report describes degeneration or intrasubstance "
        "signal of the medial meniscus without a definite tear."
    ),
    "Lateral Meniscus": (
        "The report describes degeneration or intrasubstance "
        "signal of the lateral meniscus without a definite tear."
    ),
    "Medial OA": (
        "The report describes cartilage loss, chondromalacia, "
        "chondral degeneration, or osteophytes in the medial "
        "knee compartment."
    ),
    "Lateral OA": (
        "The report describes cartilage loss, chondromalacia, "
        "chondral degeneration, or osteophytes in the lateral "
        "knee compartment."
    ),
    "PF OA": (
        "The report describes cartilage loss, chondromalacia, "
        "chondral degeneration, or osteophytes in the "
        "patellofemoral compartment."
    ),
    "Synovitis": (
        "The report describes synovial thickening, hypertrophy, "
        "or another synovial abnormality without explicitly "
        "calling it synovitis."
    ),
    "Contusion": (
        "The report describes bone marrow edema or osseous "
        "edema without explicitly calling it a bone contusion "
        "or bone bruise."
    ),
    "Fracture": (
        "The report describes an osseous injury, impaction, "
        "or insufficiency-type bone injury related to fracture."
    ),
}


# ============================================================
# W1.2 RULE ENGINE (copied forward unchanged in principle)
# ============================================================
#
# The following normalization, language, lexicon, segmentation,
# and concept-local assertion functions are carried forward from
# W1.2 so W2 remains standalone.
#
# ============================================================
# 2. USER-FIXED JSON SERIALIZATION
# ============================================================


def safe_json_value(
    value: Any,
) -> Any:
    """
    Robust JSON conversion.

    Lists/tuples/arrays must be handled before pd.isna(),
    because pd.isna(list_like) returns an array and cannot be
    evaluated directly in an if-statement.
    """

    # 1. Handle NumPy integers
    if isinstance(
        value,
        (np.integer,),
    ):

        return int(value)

    # 2. Handle NumPy floating-point numbers
    if isinstance(
        value,
        (np.floating,),
    ):

        if np.isnan(value):

            return None

        return float(value)

    # 3. Handle arrays / list-like values before pd.isna
    if isinstance(
        value,
        np.ndarray,
    ):

        return [safe_json_value(item) for item in value.tolist()]

    if isinstance(
        value,
        (list, tuple),
    ):

        return [safe_json_value(item) for item in value]

    if isinstance(
        value,
        dict,
    ):

        return {str(key): safe_json_value(item) for key, item in value.items()}

    # 4. Handle pandas/scalar missing values safely
    try:

        if np.isscalar(value) or (
            hasattr(
                value,
                "ndim",
            )
            and value.ndim == 0
        ):

            if pd.isna(value):

                return None

    except (
        ValueError,
        TypeError,
    ):

        pass

    return value


def save_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:

    clean = safe_json_value(payload)

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


# ============================================================
# 3. TEXT NORMALIZATION
# ============================================================


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

    text = text.replace(
        "\u00a0",
        " ",
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


def repair_mojibake(
    text: str,
) -> Tuple[str, bool]:
    """
    Use ftfy if already installed. No package is downloaded.

    If ftfy is absent, return Unicode-normalized text unchanged.
    """

    original = str(text)

    try:

        from ftfy import fix_text

        repaired = fix_text(original)

        return (
            repaired,
            repaired != original,
        )

    except Exception:

        return (
            original,
            False,
        )


def normalize_report(
    text: str,
) -> Tuple[str, bool]:

    text = "" if text is None else str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text, repaired = repair_mojibake(text)

    text = normalize_whitespace(text)

    return (
        text,
        repaired,
    )


def fold_for_match(
    text: str,
) -> str:
    """
    Case-fold and remove combining marks.

    This preserves Greek/Cyrillic characters while making
    Latin/Greek accent differences less brittle.
    """

    text = unicodedata.normalize(
        "NFKD",
        str(text),
    )

    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    text = text.casefold()

    text = (
        text.replace(
            "’",
            "'",
        )
        .replace(
            "‘",
            "'",
        )
        .replace(
            "–",
            "-",
        )
        .replace(
            "—",
            "-",
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def split_sentences(
    text: str,
) -> List[str]:
    """
    W1.2 clause-aware splitter.

    W1.1 exposed a common export pattern where report clauses
    are concatenated without whitespace:

        "... yoktur.Suprapatellar ..."

    Splitting only on punctuation followed by whitespace lets a
    negation cue leak into the next finding. W1.2 therefore
    splits after sentence punctuation when the next character
    is alphabetic, even if no space is present.

    Decimal numbers are preserved because a digit after "."
    does not satisfy the alphabetic look-ahead.
    """

    text = normalize_whitespace(text)

    if not text:

        return []

    # Normalize common visual separators.
    text = re.sub(
        r"\s*\|\s*",
        "\n",
        text,
    )

    # Common inline bullet/export separators.
    text = re.sub(
        r"\s+--+\s+",
        "\n",
        text,
    )

    text = re.sub(
        r"\s+-\s+(?=[^\W\d_])",
        "\n",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"(?:^|\n)\s*[•●▪◦]\s*",
        "\n",
        text,
    )

    # Force section headings onto their own unit even when the
    # exported report has no space after the colon.
    heading_pattern = re.compile(
        (
            r"(?i)\b("
            r"impression|conclusions?|findings?|"
            r"hallazgos|impresion|"
            r"sonuc|sonuç|bulgular|"
            r"zakljucak|zaključak|misljenje|mišljenje|nalaz|"
            r"συμπερασμα|συμπέρασμα|ευρηματα|ευρήματα|"
            r"beurteilung|befund(?:e)?|"
            r"заключение|извод|находка|"
            r"besluit|conclusie|bevindingen|"
            r"resultats|résultats|constatations"
            r")\s*:"
        )
    )

    text = heading_pattern.sub(
        lambda match: "\n" + match.group(0) + "\n",
        text,
    )

    # Split:
    #   - line breaks
    #   - ; always
    #   - .!? when followed by a Unicode alphabetic character,
    #     with or without whitespace
    #
    # [^\W\d_] == Unicode alphabetic character in Python re.
    parts = re.split(
        (r"\n+" r"|(?<=[;])\s*" r"|(?<=[.!?。！？])(?=\s*[^\W\d_])"),
        text,
        flags=re.UNICODE,
    )

    cleaned = []

    for part in parts:

        part = part.strip()

        if not part:

            continue

        # Split repeated inline bullets while preserving ordinary
        # hyphens inside medical terminology.
        subparts = re.split(
            r"\s+[•●▪◦]\s*",
            part,
        )

        for subpart in subparts:

            subpart = subpart.strip()

            if subpart:

                cleaned.append(subpart)

    return cleaned


# ============================================================
# 4. LANGUAGE CANONICALIZATION
# ============================================================

LANGUAGE_CANONICALIZATION = {
    "en": "en",
    "la": "en",  # W1's single "la" sample was English-like
    "es": "es",
    "tr": "tr",
    "hr": "bcs",
    "bs": "bcs",
    "sr": "bcs",
    "el": "el",
    "de": "de",
    "bg": "bg",
    "nl": "nl",
    "fr": "fr",
}


def canonicalize_language(
    language: Any,
) -> str:

    language = str(language if language is not None else "").strip().casefold()

    return LANGUAGE_CANONICALIZATION.get(
        language,
        language if language else "unknown",
    )


def detect_language_fallback(
    text: str,
) -> str:
    """
    Used only when W1 language output is unavailable.
    """

    try:

        import langid

        code, _ = langid.classify(text)

        return canonicalize_language(code)

    except Exception:

        pass

    # Script-level fallback.
    if re.search(
        r"[\u0370-\u03ff]",
        text,
    ):

        return "el"

    if re.search(
        r"[\u0400-\u052f]",
        text,
    ):

        # Gold Cyrillic corpus is Bulgarian in W1.
        return "bg"

    return "unknown"


# ============================================================
# 5. SECTION HEADING DETECTION
# ============================================================

SECTION_PATTERNS = {
    "impression": [
        r"\bimpression\b",
        r"\bconclusion\b",
        r"\bconclusions\b",
        r"\bconclusion(?:es)?\b",
        r"\bimpresion\b",
        r"\bsonuc\b",
        r"\bzakljucak\b",
        r"\bmisljenje\b",
        r"\bσυμπερασμα\b",
        r"\bbeurteilung\b",
        r"\bzusammenfassung\b",
        r"\bзаключение\b",
        r"\bизвод\b",
        r"\bbesluit\b",
        r"\bconclusie\b",
        r"\bconclusion\b",
    ],
    "findings": [
        r"\bfindings\b",
        r"\bfindings?:\b",
        r"\bhallazgos\b",
        r"\bbulgular\b",
        r"\bnalaz\b",
        r"\bευρηματα\b",
        r"\bbefund(?:e)?\b",
        r"\bмр находка\b",
        r"\bнаходка\b",
        r"\bbevindingen\b",
        r"\bresultats\b",
        r"\bconstatations\b",
    ],
}


def detect_section(
    unit: str,
    current_section: str,
) -> str:

    folded = fold_for_match(unit)

    for section_name, patterns in SECTION_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                folded,
                flags=re.IGNORECASE,
            ):

                return section_name

    return current_section


def report_units(
    report: str,
) -> List[Tuple[str, str]]:
    """
    Return (section, sentence/line) units.
    """

    units = []

    current_section = "body"

    for unit in split_sentences(report):

        current_section = detect_section(
            unit,
            current_section,
        )

        units.append(
            (
                current_section,
                unit,
            )
        )

    return units


# ============================================================
# 6. MULTILINGUAL CUE LEXICONS
# ============================================================
#
# These are conservative audit lexicons, not final W2 rules.
#
# Patterns are applied to fold_for_match(text).
# ============================================================

NEGATION_PATTERNS = {
    "universal": [
        r"\bno\b",
        r"\bnot\b",
        r"\bwithout\b",
        r"\babsent\b",
        r"\bnone\b",
        r"\bintact\b",
        r"\bnormal\b",
        r"\bunremarkable\b",
        r"\bnegative for\b",
        r"\bno evidence of\b",
    ],
    "es": [
        r"\bsin\b",
        r"\bno\b",
        r"\bnormal(?:es)?\b",
        r"\bintegro(?:s|a|as)?\b",
        r"\bconservad[oa]s?\b",
        r"\bsin evidencia de\b",
        r"\bno se observa\b",
    ],
    "tr": [
        r"\byok(?:tur)?\b",
        r"\bizlenmem(?:is|istir|ektedir)\b",
        r"\bizlenmedi\b",
        r"\bnormal(?:dir)?\b",
        r"\bkorunmus\b",
        r"\bintakt\b",
        r"\bbutunlugu korunmus\b",
        r"\beslik etmiyor\b",
    ],
    "bcs": [
        r"\bbez\b",
        r"\bnema\b",
        r"\buredan\b",
        r"\burednog\b",
        r"\bodrzan(?:og|a|i)?\b",
        r"\bintaktan\b",
        r"\bbez znakova\b",
        r"\bne vidi\b",
    ],
    "el": [
        r"\bδεν\b",
        r"\bχωρις\b",
        r"\bφυσιολογικ\w*\b",
        r"\bακεραι\w*\b",
        r"\bδεν παρατηρ\w*\b",
        r"\bδεν σημειων\w*\b",
    ],
    "de": [
        r"\bkein(?:e|en|er|es)?\b",
        r"\bohne\b",
        r"\bintakt\b",
        r"\bunauffallig\b",
        r"\bphysiologisch\b",
        r"\bregelrecht\b",
    ],
    "bg": [
        r"\bбез\b",
        r"\bняма\b",
        r"\bнормал\w*\b",
        r"\bзапазен\w*\b",
        r"\bбез особености\b",
        r"\bсъхранен\w*\b",
    ],
    "nl": [
        r"\bgeen\b",
        r"\bzonder\b",
        r"\bnormaal\b",
        r"\bintact\b",
        r"\bbehouden\b",
        r"\bgeen tekenen van\b",
    ],
    "fr": [
        r"\bpas de\b",
        r"\bsans\b",
        r"\baucun(?:e)?\b",
        r"\bnormal(?:e|es|aux)?\b",
        r"\bintact(?:e|es)?\b",
        r"\babsence de\b",
    ],
}


# Negative state words that may occur AFTER the target concept,
# e.g. "Baker cyst: None" or "ACL normal".
#
# Directional handling avoids a major scope error: a later
# phrase such as "without fracture" must not negate an earlier
# "bone marrow edema" or ligament mention.
POST_STATE_NEGATION_PATTERNS = {
    "universal": [
        r"^\s*\)?\s*(?:\([a-z]{2,6}\))?\s*[:\-;,]?\s*(?:is\s+)?(?:normal|intact|absent|none|unremarkable)\b",
        r"^\s*\)?\s*(?:\([a-z]{2,6}\))?\s*[:\-;,]?\s*(?:are\s+)?(?:normal|intact|absent|none|unremarkable)\b",
        r"^\s*\)?\s*(?:\([a-z]{2,6}\))?\s*[:\-;,]?\s*(?:no|without|negative for)\s+(?:evidence of\s+)?(?:tear\w*|ruptur\w*|injury|sprain\w*)\b",
        r"^\s*[:\-;,]?\s*degenerative\s+signal.*\bwithout\b.*\btear\w*\b",
    ],
    "es": [
        r"^\s*[:\-;,]?\s*(?:normal(?:es)?|integro(?:s|a|as)?|ausente(?:s)?)\b",
        r"^\s*[:\-;,]?\s*(?:sin|no)\s+(?:evidencia de\s+)?(?:rotur\w*|ruptur\w*|lesion\w*)\b",
    ],
    "tr": [
        r"^\s*[:\-;,]?\s*(?:normal(?:dir)?|intakt|korunmus|korunmuş|yok(?:tur)?)\b",
        r"^\s*[:\-;,]?\s*(?:yirtik|yırtık|ruptur|rüptür)\s+(?:yok|izlenmedi|izlenmemis)\b",
    ],
    "bcs": [
        r"^\s*[:\-;,]?\s*(?:uredan|uredni|odrzan\w*|održan\w*|intaktan)\b",
        r"^\s*[:\-;,]?\s*bez\s+(?:znakova\s+)?(?:ruptur\w*|lezij\w*)\b",
    ],
    "el": [
        r"^\s*[:\-;,]?\s*(?:φυσιολογικ\w*|ακεραι\w*)\b",
    ],
    "de": [
        r"^\s*[:\-;,]?\s*(?:intakt|unauffallig|physiologisch|regelrecht)\b",
        r"^\s*[:\-;,]?\s*(?:kein\w*|ohne)\s+(?:riss\w*|ruptur\w*|lasion\w*)\b",
    ],
    "bg": [
        r"^\s*[:\-;,]?\s*(?:нормал\w*|запазен\w*|съхранен\w*)\b",
    ],
    "nl": [
        r"^\s*[:\-;,]?\s*(?:normaal|intact|behouden|geen)\b",
        r"^\s*[:\-;,]?\s*(?:geen|zonder)\s+(?:scheur\w*|ruptur\w*|lesie\w*)\b",
    ],
    "fr": [
        r"^\s*[:\-;,]?\s*(?:normal\w*|intact\w*|absent\w*)\b",
        r"^\s*[:\-;,]?\s*(?:sans|pas de)\s+(?:ruptur\w*|dechir\w*|lesion\w*)\b",
    ],
}


POST_FINDING_NEGATION_PATTERNS = {
    "universal": [
        r"^\s*(?:is\s+)?(?:absent|not seen|not identified|not visualized|not demonstrated)\b",
        r"^\s*(?:is\s+)?(?:excluded|unlikely)\b",
    ],
    "es": [
        r"^\s*(?:no se observa|no se identifica|ausente)\b",
    ],
    "tr": [
        r"^\s*(?:yok(?:tur)?|izlenmedi|izlenmemis|izlenmemiş|saptanmadi|saptanmadı)\b",
    ],
    "bcs": [
        r"^\s*(?:nema|nije vidljiv\w*|ne vidi se|bez znakova)\b",
    ],
    "el": [
        r"^\s*(?:δεν παρατηρ\w*|δεν απεικον\w*|απουσια)\b",
    ],
    "de": [
        r"^\s*(?:nicht nachweisbar|nicht darstellbar|kein\w*)\b",
    ],
    "bg": [
        r"^\s*(?:не се установ\w*|не се визуализ\w*|липсва)\b",
    ],
    "nl": [
        r"^\s*(?:niet zichtbaar|niet aantoonbaar|geen)\b",
    ],
    "fr": [
        r"^\s*(?:non visualis\w*|non identifi\w*|absent\w*)\b",
    ],
}


UNCERTAINTY_PATTERNS = {
    "universal": [
        r"\bpossible\b",
        r"\bpossibly\b",
        r"\bprobable\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bunlikely\b",
        r"\bsuspect(?:ed)?\b",
        r"\bsuspicious\b",
        r"\bmay\b",
        r"\bmight\b",
        r"\bcannot exclude\b",
        r"\bcan't exclude\b",
        r"\bquestion of\b",
        r"\bversus\b",
        r"\bddx?\b",
        r"\br/?o\b",
    ],
    "es": [
        r"\bposible\b",
        r"\bprobable\b",
        r"\bsugestiv\w*\b",
        r"\bsospech\w*\b",
        r"\bno se puede excluir\b",
    ],
    "tr": [
        r"\bolasi\b",
        r"\bolasilikla\b",
        r"\bsupheli\b",
        r"\bdusunul\w*\b",
        r"\bile uyumlu olabilir\b",
    ],
    "bcs": [
        r"\bmoguc\w*\b",
        r"\bvjerojat\w*\b",
        r"\bsumnj\w*\b",
        r"\bmoze odgovarati\b",
    ],
    "el": [
        r"\bπιθαν\w*\b",
        r"\bενδεχομεν\w*\b",
        r"\bθα μπορουσε\b",
    ],
    "de": [
        r"\bmoglich\w*\b",
        r"\bwahrscheinlich\w*\b",
        r"\bverdacht\b",
        r"\bdd\b",
    ],
    "bg": [
        r"\bвероят\w*\b",
        r"\bвъзмож\w*\b",
        r"\bсъмн\w*\b",
    ],
    "nl": [
        r"\bmogelijk\w*\b",
        r"\bwaarschijnlijk\w*\b",
        r"\bverdacht\b",
    ],
    "fr": [
        r"\bpossible\b",
        r"\bprobable\b",
        r"\bsuspect\w*\b",
        r"\bne peut exclure\b",
    ],
}


HISTORY_PATTERNS = {
    "universal": [
        r"\bhistory of\b",
        r"\bprior\b",
        r"\bprevious\b",
        r"\bstatus post\b",
        r"\bpostoperative\b",
        r"\bpost-operative\b",
        r"\bknown\b",
        r"\bchronic\b",
    ],
    "es": [
        r"\bantecedente\w*\b",
        r"\bprevio\w*\b",
        r"\bpostoperator\w*\b",
        r"\bcronico\w*\b",
    ],
    "tr": [
        r"\bgecmis\b",
        r"\bonceki\b",
        r"\bpostoperatif\b",
        r"\bkronik\b",
    ],
    "bcs": [
        r"\branij\w*\b",
        r"\bprethod\w*\b",
        r"\bpostoperativ\w*\b",
        r"\bkronic\w*\b",
    ],
    "el": [
        r"\bιστορικ\w*\b",
        r"\bπροηγ\w*\b",
        r"\bμετεγχειρητικ\w*\b",
        r"\bχρονι\w*\b",
    ],
    "de": [
        r"\banamnestisch\b",
        r"\bvorbekannt\w*\b",
        r"\bpostoperativ\w*\b",
        r"\bchronisch\w*\b",
    ],
    "bg": [
        r"\bанамнез\w*\b",
        r"\bпредход\w*\b",
        r"\bследоператив\w*\b",
        r"\bхронич\w*\b",
    ],
    "nl": [
        r"\bvoorgeschiedenis\b",
        r"\bstatus na\b",
        r"\bpostoperatief\b",
        r"\bchronisch\w*\b",
    ],
    "fr": [
        r"\bantecedent\w*\b",
        r"\bancien\w*\b",
        r"\bpostoperatoire\b",
        r"\bchronique\b",
    ],
}


SEVERITY_PATTERNS = {
    "complete": [
        r"\bcomplete\b",
        r"\bfull[- ]?thickness\b",
        r"\bfull thickness\b",
        r"\bcomplet[oa]\b",
        r"\bcompleta\b",
        r"\bkomplet\b",
        r"\btam kat\b",
        r"\btam kata yakin\b",
        r"\bkompletn\w*\b",
        r"\bpotpun\w*\b",
        r"\bπληρ\w*\b",
        r"\bkomplett\w*\b",
        r"\bvollstandig\w*\b",
        r"\bпълн\w*\b",
        r"\bvolledig\w*\b",
        r"\bcomplet(?:e|es)?\b",
    ],
    "partial": [
        r"\bpartial\b",
        r"\binterstitial\b",
        r"\bintrasubstance\b",
        r"\bparcial\b",
        r"\bintrasustancia\b",
        r"\bparsiyel\b",
        r"\bkismi\b",
        r"\bkısmi\b",
        r"\bparcijal\w*\b",
        r"\bdjelomic\w*\b",
        r"\bμερικ\w*\b",
        r"\bpartiell\w*\b",
        r"\bчастич\w*\b",
        r"\bpartieel\w*\b",
        r"\bpartiele\w*\b",
        r"\bpartiel(?:le|les)?\b",
    ],
    "mild": [
        r"\bmild\b",
        r"\bminimal\b",
        r"\bsmall\b",
        r"\bgrade\s*[i1]\b",
        r"\bleve\b",
        r"\bligero\w*\b",
        r"\bhafif\b",
        r"\bgrade\s*[i1]\b",
        r"\bblag\w*\b",
        r"\bηπι\w*\b",
        r"\bgering\w*\b",
        r"\bleicht\w*\b",
        r"\bминимал\w*\b",
        r"\bлек\w*\b",
        r"\blicht\b",
        r"\bgering\b",
        r"\bleger\w*\b",
    ],
    "moderate": [
        r"\bmoderate\b",
        r"\bgrade\s*(?:ii|2)\b",
        r"\bmoderad\w*\b",
        r"\bgrado\s*(?:ii|2)\b",
        r"\borta\b",
        r"\bgrade\s*(?:ii|2)\b",
        r"\bumjeren\w*\b",
        r"\bμετρι\w*\b",
        r"\bmassig\w*\b",
        r"\bgrad\s*(?:ii|2)\b",
        r"\bумерен\w*\b",
        r"\bmatig\b",
        r"\bmodere\w*\b",
    ],
    "severe": [
        r"\bsevere\b",
        r"\bmarked\b",
        r"\bgrade\s*(?:iii|iv|3|4)\b",
        r"\bsever\w*\b",
        r"\bgrave\b",
        r"\bileri\b",
        r"\bbelirgin\b",
        r"\bgrade\s*(?:iii|iv|3|4)\b",
        r"\btesk\w*\b",
        r"\buznapredoval\w*\b",
        r"\bσοβαρ\w*\b",
        r"\bschwer\w*\b",
        r"\bausgepragt\w*\b",
        r"\bзначим\w*\b",
        r"\bтеж\w*\b",
        r"\bernstig\w*\b",
        r"\bgevorderd\w*\b",
        r"\bsevere\w*\b",
    ],
    "degenerative": [
        r"\bdegenerative\b",
        r"\bdegeneration\b",
        r"\bdegenerativ\w*\b",
        r"\bdejener\w*\b",
        r"\bdegenerativ\w*\b",
        r"\bεκφυλισ\w*\b",
        r"\bdegenerativ\w*\b",
        r"\bдегенератив\w*\b",
        r"\bdegeneratief\w*\b",
        r"\bdegenerati\w*\b",
    ],
}


# ------------------------------------------------------------
# Anatomy / target patterns
# ------------------------------------------------------------

STRUCTURE_PATTERNS = {
    "ACL": {
        "universal": [
            r"\bacl\b",
            r"\banterior cruciate ligament\b",
        ],
        "es": [
            r"\bligamento cruzado anterior\b",
            r"\blca\b",
        ],
        "tr": [
            r"\bon capraz bag\b",
            r"\bön çapraz bağ\b",
            r"\banterior capraz bag\b",
        ],
        "bcs": [
            r"\bprednji krizni ligament\b",
            r"\bprednji križni ligament\b",
        ],
        "el": [
            r"\bπροσθι\w* χιαστ\w* συνδεσμ\w*\b",
        ],
        "de": [
            r"\bvkb\b",
            r"\bvorder(?:e|en|er)? kreuzband\b",
        ],
        "bg": [
            r"\bпредн\w* кръстн\w* връзк\w*\b",
        ],
        "nl": [
            r"\bvoorste kruisband\b",
            r"\bvkb\b",
        ],
        "fr": [
            r"\bligament croise anterieur\b",
            r"\blca\b",
        ],
    },
    "MCL": {
        "universal": [
            r"\bmcl\b",
            r"\bmedial collateral ligament\b",
        ],
        "es": [
            r"\bligamento colateral medial\b",
            r"\blcm\b",
        ],
        "tr": [
            r"\bmedial kollateral ligam\w*\b",
            r"\bmedyal kollateral ligam\w*\b",
            r"\bic yan bag\b",
        ],
        "bcs": [
            r"\bmedijaln\w* kolateraln\w* ligament\w*\b",
        ],
        "el": [
            r"\bεσω πλαγι\w* συνδεσμ\w*\b",
        ],
        "de": [
            r"\bmedial(?:e|en|er)? kollateralband\b",
            r"\binnenband\b",
        ],
        "bg": [
            r"\bмедиалн\w* колатералн\w* връзк\w*\b",
        ],
        "nl": [
            r"\bmediale collaterale ligament\b",
            r"\bmediale collaterale band\b",
        ],
        "fr": [
            r"\bligament collateral medial\b",
            r"\blcm\b",
        ],
    },
    "Medial Meniscus": {
        "universal": [
            r"\bmedial menisc\w*\b",
        ],
        "es": [
            r"\bmenisco medial\b",
        ],
        "tr": [
            r"\bmedyal menisk\w*\b",
            r"\bmedial menisk\w*\b",
        ],
        "bcs": [
            r"\bmedijaln\w* menisk\w*\b",
        ],
        "el": [
            r"\bεσω μηνισκ\w*\b",
        ],
        "de": [
            r"\binnenmenisk\w*\b",
        ],
        "bg": [
            r"\bмедиалн\w* мениск\w*\b",
        ],
        "nl": [
            r"\bmediale menisc\w*\b",
        ],
        "fr": [
            r"\bmenisque medial\b",
        ],
    },
    "Lateral Meniscus": {
        "universal": [
            r"\blateral menisc\w*\b",
        ],
        "es": [
            r"\bmenisco lateral\b",
        ],
        "tr": [
            r"\blateral menisk\w*\b",
        ],
        "bcs": [
            r"\blateraln\w* menisk\w*\b",
        ],
        "el": [
            r"\bεξω μηνισκ\w*\b",
        ],
        "de": [
            r"\baussenmenisk\w*\b",
        ],
        "bg": [
            r"\bлатералн\w* мениск\w*\b",
        ],
        "nl": [
            r"\blaterale menisc\w*\b",
        ],
        "fr": [
            r"\bmenisque lateral\b",
        ],
    },
}


TEAR_INJURY_PATTERNS = {
    "universal": [
        r"\btear\w*\b",
        r"\bruptur\w*\b",
        r"\bsprain\w*\b",
        r"\binjury\b",
        r"\bdiscontinu\w*\b",
        r"\bavulsion\b",
        r"\bavulsed\b",
        r"\blaxity\b",
    ],
    "es": [
        r"\brotur\w*\b",
        r"\bruptur\w*\b",
        r"\blesion\w*\b",
        r"\besguince\b",
        r"\bdiscontinuidad\b",
    ],
    "tr": [
        r"\byirtik\w*\b",
        r"\byırtık\w*\b",
        r"\bruptur\w*\b",
        r"\brüptür\w*\b",
        r"\bsprain\b",
        r"\bzorlanma\b",
        r"\bbütünlük kaybı\b",
        r"\bbutunluk kaybi\b",
        r"\blaksite\b",
    ],
    "bcs": [
        r"\bruptur\w*\b",
        r"\blezij\w*\b",
        r"\bdistenzij\w*\b",
        r"\bprekid\w*\b",
        r"\bdiskontinuit\w*\b",
    ],
    "el": [
        r"\bρηξ\w*\b",
        r"\bκακωσ\w*\b",
        r"\bασυνεχ\w*\b",
    ],
    "de": [
        r"\briss\w*\b",
        r"\bruptur\w*\b",
        r"\blasion\w*\b",
        r"\bdiskontinuit\w*\b",
        r"\bausriss\w*\b",
    ],
    "bg": [
        r"\bруптур\w*\b",
        r"\bскъс\w*\b",
        r"\bлезия\b",
        r"\bнарушен\w* цялост\b",
    ],
    "nl": [
        r"\bscheur\w*\b",
        r"\bruptur\w*\b",
        r"\blesie\w*\b",
        r"\bdiscontinu\w*\b",
        r"\bdistors\w*\b",
    ],
    "fr": [
        r"\bruptur\w*\b",
        r"\bdechir\w*\b",
        r"\blesion\w*\b",
        r"\bentorse\b",
        r"\bdiscontinu\w*\b",
    ],
}


DEGENERATION_PATTERNS = {
    "universal": [
        r"\bdegenerat\w*\b",
        r"\bmucoid\b",
        r"\bdegenerative signal\b",
    ],
    "es": [
        r"\bdegener\w*\b",
    ],
    "tr": [
        r"\bdejener\w*\b",
        r"\bmeniskopati\b",
    ],
    "bcs": [
        r"\bdegenerativ\w*\b",
    ],
    "el": [
        r"\bεκφυλισ\w*\b",
    ],
    "de": [
        r"\bdegenerativ\w*\b",
        r"\bmukoid\w*\b",
    ],
    "bg": [
        r"\bдегенератив\w*\b",
    ],
    "nl": [
        r"\bdegeneratief\w*\b",
        r"\bmucoide\b",
    ],
    "fr": [
        r"\bdegenerati\w*\b",
    ],
}


OA_PATTERNS = {
    "universal": [
        r"\bosteoarthr\w*\b",
        r"\bosteoarthrit\w*\b",
        r"\barthros\w*\b",
        r"\bgonarthr\w*\b",
        r"\bjoint space narrowing\b",
        r"\bfull thickness cartilage loss\b",
        r"\bfull-thickness cartilage loss\b",
        r"\bosteophyt\w*\b",
        r"\bchondromalaci\w*\b",
        r"\bchondropath\w*\b",
    ],
    "es": [
        r"\bartrosis\b",
        r"\bosteoartrosis\b",
        r"\bosteoartritis\b",
        r"\bpinzamiento\b",
        r"\bosteofit\w*\b",
        r"\bcondropat\w*\b",
        r"\bcondromalaci\w*\b",
    ],
    "tr": [
        r"\bosteoartr\w*\b",
        r"\bartroz\b",
        r"\bgonartroz\b",
        r"\beklem araligi daral\w*\b",
        r"\beklem aralığı daral\w*\b",
        r"\bosteofit\w*\b",
        r"\bkondromalaz\w*\b",
    ],
    "bcs": [
        r"\boa promjen\w*\b",
        r"\bartroz\w*\b",
        r"\bgonartroz\w*\b",
        r"\bosteofit\w*\b",
        r"\bhondromalacij\w*\b",
        r"\bhondropat\w*\b",
        r"\breduciran\w* zglobn\w* prostor\b",
    ],
    "el": [
        r"\bοστεοαρθρ\w*\b",
        r"\bοστεοφυτ\w*\b",
        r"\bχονδρομαλακ\w*\b",
        r"\bχονδροπαθ\w*\b",
        r"\bεξαλειψη του αρθρικου χονδρου\b",
        r"\bδιαβρωση του αρθρικου χονδρου\b",
    ],
    "de": [
        r"\barthros\w*\b",
        r"\bgonarthros\w*\b",
        r"\bosteophyt\w*\b",
        r"\bchondropath\w*\b",
        r"\bknorpelverlust\b",
    ],
    "bg": [
        r"\bартроз\w*\b",
        r"\bостеоарт\w*\b",
        r"\bостеофит\w*\b",
        r"\bхондромалац\w*\b",
        r"\bизтъняване на .*хрущял\b",
    ],
    "nl": [
        r"\bartrose\b",
        r"\bgonartrose\b",
        r"\bosteofyt\w*\b",
        r"\bkraakbeenverlies\b",
        r"\bkraakbeenlijden\b",
        r"\bchondropath\w*\b",
    ],
    "fr": [
        r"\barthrose\b",
        r"\bgonarthrose\b",
        r"\bosteoarthr\w*\b",
        r"\bosteophyt\w*\b",
        r"\bchondropath\w*\b",
        r"\bperte cartilag\w*\b",
    ],
}


COMPARTMENT_PATTERNS = {
    "medial": {
        "universal": [
            r"\bmedial compartment\b",
            r"\bmedial femorotibial\b",
            r"\bmedial femoral condyl\w*\b",
            r"\bmedial tibial plateau\b",
        ],
        "es": [
            r"\bcompartimento medial\b",
            r"\bcondilo femoral medial\b",
            r"\bplatillo tibial medial\b",
        ],
        "tr": [
            r"\bmedial kompart\w*\b",
            r"\bmedyal kompart\w*\b",
            r"\bmedial femoral kondil\w*\b",
            r"\bmedyal femoral kondil\w*\b",
            r"\bmedial tibial plato\b",
            r"\bmedyal tibial plato\b",
        ],
        "bcs": [
            r"\bmedijaln\w* kompartment\w*\b",
            r"\bmedijaln\w* kondil\w* femur\w*\b",
            r"\bmedijaln\w* plato\w* tibij\w*\b",
        ],
        "el": [
            r"\bεσω διαμερισμ\w*\b",
            r"\bεσω μηριαι\w* κονδυλ\w*\b",
            r"\bεσω κνημιαι\w* κονδυλ\w*\b",
        ],
        "de": [
            r"\bmedial\w* kompartiment\b",
            r"\bmedial\w* femorotibial\w*\b",
            r"\bmedial\w* femurkondyl\w*\b",
            r"\bmedial\w* tibiaplateau\b",
        ],
        "bg": [
            r"\bмедиалн\w* компартимент\w*\b",
            r"\bмедиалн\w* феморалн\w* кондил\w*\b",
            r"\bмедиалн\w* тибиалн\w* плато\b",
        ],
        "nl": [
            r"\bmediaal femorotibiaal\b",
            r"\bmediale femorale condyl\b",
            r"\bmediale tibiaplateau\b",
        ],
        "fr": [
            r"\bcompartiment medial\b",
            r"\bfemorotibial medial\b",
            r"\bcondyle femoral medial\b",
            r"\bplateau tibial medial\b",
        ],
    },
    "lateral": {
        "universal": [
            r"\blateral compartment\b",
            r"\blateral femorotibial\b",
            r"\blateral femoral condyl\w*\b",
            r"\blateral tibial plateau\b",
        ],
        "es": [
            r"\bcompartimento lateral\b",
            r"\bcondilo femoral lateral\b",
            r"\bplatillo tibial lateral\b",
        ],
        "tr": [
            r"\blateral kompart\w*\b",
            r"\blateral femoral kondil\w*\b",
            r"\blateral tibial plato\b",
        ],
        "bcs": [
            r"\blateraln\w* kompartment\w*\b",
            r"\blateraln\w* kondil\w* femur\w*\b",
            r"\blateraln\w* plato\w* tibij\w*\b",
        ],
        "el": [
            r"\bεξω διαμερισμ\w*\b",
            r"\bεξω μηριαι\w* κονδυλ\w*\b",
            r"\bεξω κνημιαι\w* κονδυλ\w*\b",
        ],
        "de": [
            r"\blateral\w* kompartiment\b",
            r"\blateral\w* femorotibial\w*\b",
            r"\blateral\w* femurkondyl\w*\b",
            r"\blateral\w* tibiaplateau\b",
        ],
        "bg": [
            r"\bлатералн\w* компартимент\w*\b",
            r"\bлатералн\w* феморалн\w* кондил\w*\b",
            r"\bлатералн\w* тибиалн\w* плато\b",
        ],
        "nl": [
            r"\blateraal femorotibiaal\b",
            r"\blaterale femorale condyl\b",
            r"\blaterale tibiaplateau\b",
        ],
        "fr": [
            r"\bcompartiment lateral\b",
            r"\bfemorotibial lateral\b",
            r"\bcondyle femoral lateral\b",
            r"\bplateau tibial lateral\b",
        ],
    },
    "pf": {
        "universal": [
            r"\bpatellofemoral\b",
            r"\bpatello[- ]?femoral\b",
            r"\bpatellar facet\b",
            r"\btrochle\w*\b",
            r"\bretropatellar\b",
        ],
        "es": [
            r"\bpatelofemoral\b",
            r"\bfemoropatelar\b",
            r"\brotul\w*\b",
            r"\btrocle\w*\b",
        ],
        "tr": [
            r"\bpatellofemoral\b",
            r"\bpatellofemoral\b",
            r"\bpatellar\b",
            r"\btroklear\b",
        ],
        "bcs": [
            r"\bpf\b",
            r"\bpatelofemoral\w*\b",
            r"\bpatelarn\w*\b",
            r"\btrohlear\w*\b",
        ],
        "el": [
            r"\bεπιγονατιδ\w*\b",
            r"\bμηροεπιγονατιδ\w*\b",
            r"\bτροχιλ\w*\b",
        ],
        "de": [
            r"\bpatellofemoral\w*\b",
            r"\bretropatellar\w*\b",
            r"\btrochle\w*\b",
        ],
        "bg": [
            r"\bпателофеморал\w*\b",
            r"\bпател\w*\b",
            r"\bтрохле\w*\b",
        ],
        "nl": [
            r"\bfemoropatellair\w*\b",
            r"\bpatellofemoraal\w*\b",
            r"\bpatellair\w*\b",
        ],
        "fr": [
            r"\bfemoropatellaire\b",
            r"\bpatellofemoral\w*\b",
            r"\brotul\w*\b",
            r"\btrochle\w*\b",
        ],
    },
}


DIRECT_CONDITION_PATTERNS = {
    "Effusion": {
        "universal": [
            r"\bjoint effusion\b",
            r"\beffusion\b",
            r"\bhemarthros\w*\b",
        ],
        "es": [
            r"\bderrame articular\b",
            r"\bderrame\b",
            r"\befusion articular\b",
            r"\bhemartros\w*\b",
        ],
        "tr": [
            r"\beklem .*sivi\w*\b",
            r"\beklem .*sıvı\w*\b",
            r"\bsuprapatellar bursa\w*.{0,60}sivi artisi\b",
            r"\bsuprapatellar bursa\w*.{0,60}sıvı artışı\b",
            r"\bdiz eklem\w*.{0,60}sivi artisi\b",
            r"\bdiz eklem\w*.{0,60}sıvı artışı\b",
            r"\befuzyon\b",
            r"\befüzyon\b",
            r"\bhemorajik izliv\b",
        ],
        "bcs": [
            r"\bzglobn\w* izljev\w*\b",
            r"\bizljev\b",
            r"\befuzij\w*\b",
            r"\bhidrops\b",
        ],
        "el": [
            r"\bενδαρθρικ\w* συλλογ\w* υγρου\b",
            r"\bσυλλογ\w* υγρου\b",
            r"\bαρθρικ\w* υγρ\w*\b",
            r"\bυδραρθρ\w*\b",
        ],
        "de": [
            r"\bgelenkerguss\b",
            r"\berguss\b",
            r"\bhamarthros\b",
        ],
        "bg": [
            r"\bставен излив\b",
            r"\bизлив\b",
            r"\bхеморагичен излив\b",
        ],
        "nl": [
            r"\bhydrops\b",
            r"\beffusie\b",
            r"\bgewrichts?vocht\b",
            r"\bvocht in het gewricht\b",
        ],
        "fr": [
            r"\bepanchement articulaire\b",
            r"\bhydarthrose\b",
            r"\beffusion\b",
        ],
    },
    "Synovitis": {
        "universal": [
            r"\bsynovitis\b",
            r"\bsynovial thickening\b",
            r"\bthickening of the synovium\b",
        ],
        "es": [
            r"\bsinovitis\b",
            r"\bengrosamiento sinovial\b",
        ],
        "tr": [
            r"\bsinovit\b",
            r"\bsinovyal kalinlas\w*\b",
            r"\bsinovyal kalınlaş\w*\b",
        ],
        "bcs": [
            r"\bsinovitis\b",
            r"\bsinovijal\w* zadebljan\w*\b",
        ],
        "el": [
            r"\bυμενιτιδ\w*\b",
            r"\bσυνοβιτιδ\w*\b",
            r"\bπαχυνση του υμενα\b",
        ],
        "de": [
            r"\bsynovitis\b",
            r"\bsynovialitis\b",
            r"\bsynovial\w* verdick\w*\b",
        ],
        "bg": [
            r"\bсиновит\b",
            r"\bудебел\w* синов\w*\b",
        ],
        "nl": [
            r"\bsynovitis\b",
            r"\bverdik\w* synovium\b",
        ],
        "fr": [
            r"\bsynovite\b",
            r"\bepaississement synovial\b",
        ],
    },
    "Baker's": {
        "universal": [
            r"\bbaker'?s? cyst\b",
            r"\bpopliteal cyst\b",
        ],
        "es": [
            r"\bquiste de baker\b",
            r"\bquiste popliteo\b",
        ],
        "tr": [
            r"\bbaker kist\w*\b",
            r"\bpopliteal kist\w*\b",
        ],
        "bcs": [
            r"\bbaker\w* cist\w*\b",
            r"\bpopliteal\w* cist\w*\b",
        ],
        "el": [
            r"\bκυστη baker\b",
            r"\bιγνυακ\w* κυστη\b",
        ],
        "de": [
            r"\bbaker[- ]?zyste\b",
            r"\bpoplitealzyste\b",
        ],
        "bg": [
            r"\bкиста на baker\b",
            r"\bбейкърова киста\b",
            r"\bпоплитеал\w* киста\b",
        ],
        "nl": [
            r"\bbaker[- ]?cyste\b",
            r"\bpopliteale cyste\b",
        ],
        "fr": [
            r"\bkyste de baker\b",
            r"\bkyste poplite\b",
        ],
    },
    "Contusion": {
        "universal": [
            r"\bbone bruise\b",
            r"\bbone contusion\b",
            r"\bcontusion\b",
            r"\bcontusional\b",
        ],
        "es": [
            r"\bcontusion osea\b",
            r"\bcontusion\w*\b",
            r"\bedema oseo contus\w*\b",
        ],
        "tr": [
            r"\bkemik kontuzyon\w*\b",
            r"\bkontuzyon\w*\b",
            r"\bkissing kontuzyon\b",
        ],
        "bcs": [
            r"\bkostan\w* kontuz\w*\b",
            r"\bkontuz\w*\b",
        ],
        "el": [
            r"\bοστικ\w* μωλωπ\w*\b",
            r"\bοστικ\w* θλασ\w*\b",
        ],
        "de": [
            r"\bbone bruise\b",
            r"\bknochenkontusion\b",
            r"\bkontusion\b",
        ],
        "bg": [
            r"\bконтузион\w* костномозъч\w* едем\b",
            r"\bконтузион\w*\b",
        ],
        "nl": [
            r"\bbotcontusie\b",
            r"\bcontusie\b",
            r"\bbone bruise\b",
        ],
        "fr": [
            r"\bcontusion osseuse\b",
            r"\bcontusion\w*\b",
            r"\bbone bruise\b",
        ],
    },
    "Fracture": {
        "universal": [
            r"\bfracture\b",
            r"\bfractur\w*\b",
            r"\binsufficiency fracture\b",
            r"\bimpaction fracture\b",
            r"\bosteochondral fracture\b",
        ],
        "es": [
            r"\bfractura\b",
            r"\bfractur\w*\b",
        ],
        "tr": [
            r"\bfraktur\w*\b",
            r"\bfraktür\w*\b",
            r"\bkirik\w*\b",
            r"\bkırık\w*\b",
        ],
        "bcs": [
            r"\bfraktur\w*\b",
        ],
        "el": [
            r"\bκαταγμα\w*\b",
            r"\bκαταγ\w*\b",
        ],
        "de": [
            r"\bfraktur\w*\b",
            r"\bimpressionsfraktur\b",
        ],
        "bg": [
            r"\bфрактур\w*\b",
            r"\bсчупван\w*\b",
        ],
        "nl": [
            r"\bfractuur\b",
            r"\bfractur\w*\b",
            r"\bimpactiefractuur\b",
        ],
        "fr": [
            r"\bfracture\b",
            r"\bfractur\w*\b",
        ],
    },
}


RELATED_PATTERNS = {
    "Contusion": {
        "universal": [
            r"\bbone marrow edema\b",
            r"\bmarrow edema\b",
            r"\bsubchondral edema\b",
        ],
        "es": [
            r"\bedema oseo\b",
            r"\bedema de medula osea\b",
        ],
        "tr": [
            r"\bkemik iligi odem\w*\b",
            r"\bkemik iliği ödem\w*\b",
        ],
        "bcs": [
            r"\bkostan\w* edem\b",
            r"\bkostan\w* edem\w*\b",
        ],
        "el": [
            r"\bοστεομυελικ\w* οιδημα\b",
            r"\bοιδημα του οστικου μυελου\b",
        ],
        "de": [
            r"\bknochenodem\b",
            r"\bknochenmarkodem\b",
        ],
        "bg": [
            r"\bкостно[- ]?мозъч\w* едем\b",
            r"\bкостномозъч\w* едем\b",
        ],
        "nl": [
            r"\bbotoedeem\b",
            r"\bbeenmergoedeem\b",
        ],
        "fr": [
            r"\boedeme osseux\b",
            r"\boedeme medullaire\b",
        ],
    },
}


# ============================================================
# 7. REGEX HELPERS
# ============================================================


def patterns_for_language(
    mapping: Dict[str, List[str]],
    language: str,
) -> List[str]:

    patterns = []

    patterns.extend(
        mapping.get(
            "universal",
            [],
        )
    )

    patterns.extend(
        mapping.get(
            language,
            [],
        )
    )

    return patterns


def first_pattern_match(
    text: str,
    patterns: Sequence[str],
) -> Tuple[Optional[str], Optional[re.Match]]:

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match is not None:

            return (
                pattern,
                match,
            )

    return (
        None,
        None,
    )


def all_pattern_strings(
    text: str,
    patterns: Sequence[str],
) -> List[str]:

    matches = []

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            matches.append(pattern)

    return matches


def context_around_match(
    folded_text: str,
    match: Optional[re.Match],
    radius: int = CONTEXT_WINDOW_CHARS,
) -> str:

    if match is None:

        return folded_text

    start = max(
        0,
        match.start() - radius,
    )

    end = min(
        len(folded_text),
        match.end() + radius,
    )

    return folded_text[start:end]


def evidence_section_weight(
    section: str,
) -> int:

    if section == "impression":

        return 3

    if section == "findings":

        return 2

    return 1


def extract_severity(
    folded_text: str,
) -> List[str]:

    categories = []

    for category, patterns in SEVERITY_PATTERNS.items():

        if any(
            re.search(
                pattern,
                folded_text,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):

            categories.append(category)

    # Preserve explicit grade expressions.
    grade_matches = re.findall(
        (r"\b(?:grade|grad|grado|graad)" r"\s*[:\-]?\s*" r"(?:i{1,4}|[1-4])\b"),
        folded_text,
        flags=re.IGNORECASE,
    )

    categories.extend([f"grade_{grade.casefold()}" for grade in grade_matches])

    return sorted(set(categories))


# ============================================================
# 8. REPORT-SIDE ASSERTION CLASSIFICATION
# ============================================================


def pattern_matches(
    text: str,
    patterns: Sequence[str],
) -> List[Tuple[str, re.Match]]:

    found = []

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            found.append(
                (
                    pattern,
                    match,
                )
            )

    return found


def finding_negation_state(
    folded_text: str,
    finding_match: re.Match,
    language: str,
) -> Tuple[
    bool,
    List[str],
]:
    """
    Decide whether one finding mention is locally negated.

    Scope is anchored to the finding itself rather than the
    whole sentence/report. This is the main W1.2 correction.
    """

    left = folded_text[
        max(
            0,
            finding_match.start() - 65,
        ) : finding_match.start()
    ]

    right = folded_text[
        finding_match.end() : min(
            len(folded_text),
            finding_match.end() + 55,
        )
    ]

    pre_hits = all_pattern_strings(
        left,
        patterns_for_language(
            NEGATION_PATTERNS,
            language,
        ),
    )

    post_hits = all_pattern_strings(
        right,
        patterns_for_language(
            POST_FINDING_NEGATION_PATTERNS,
            language,
        ),
    )

    hits = sorted(set(pre_hits + post_hits))

    return (
        len(hits) > 0,
        hits,
    )


def finding_uncertainty_state(
    folded_text: str,
    finding_match: re.Match,
    language: str,
) -> Tuple[
    bool,
    List[str],
]:

    context = folded_text[
        max(
            0,
            finding_match.start() - 65,
        ) : min(
            len(folded_text),
            finding_match.end() + 65,
        )
    ]

    hits = all_pattern_strings(
        context,
        patterns_for_language(
            UNCERTAINTY_PATTERNS,
            language,
        ),
    )

    return (
        len(hits) > 0,
        hits,
    )


def concept_state_negation(
    folded_text: str,
    concept_match: re.Match,
    language: str,
) -> List[str]:
    """
    Target state expressions such as:
      ACL normal
      Baker cyst: none
      MCL intact
    """

    right = folded_text[
        concept_match.end() : min(
            len(folded_text),
            concept_match.end() + 120,
        )
    ]

    post_hits = all_pattern_strings(
        right,
        patterns_for_language(
            POST_STATE_NEGATION_PATTERNS,
            language,
        ),
    )

    left = folded_text[
        max(
            0,
            concept_match.start() - 70,
        ) : concept_match.start()
    ]

    pre_hits = all_pattern_strings(
        left,
        patterns_for_language(
            NEGATION_PATTERNS,
            language,
        ),
    )

    return sorted(set(pre_hits + post_hits))


def generic_context_cues(
    folded_unit: str,
    language: str,
    concept_match: Optional[re.Match],
) -> Dict[str, List[str]]:
    """
    Concept-local state cues.

    W1.2 keeps this helper for direct conditions and OA, while
    structure injuries receive finding-local handling below.
    """

    if concept_match is None:

        return {
            "negation": [],
            "uncertainty": [],
            "history": [],
        }

    context = folded_unit[
        max(
            0,
            concept_match.start() - 80,
        ) : min(
            len(folded_unit),
            concept_match.end() + 100,
        )
    ]

    negation = concept_state_negation(
        folded_unit,
        concept_match,
        language,
    )

    uncertainty = all_pattern_strings(
        context,
        patterns_for_language(
            UNCERTAINTY_PATTERNS,
            language,
        ),
    )

    history = all_pattern_strings(
        context,
        patterns_for_language(
            HISTORY_PATTERNS,
            language,
        ),
    )

    return {
        "negation": negation,
        "uncertainty": sorted(set(uncertainty)),
        "history": sorted(set(history)),
    }


def classify_structure_unit(
    label: str,
    unit: str,
    language: str,
    section: str,
) -> Optional[Dict[str, Any]]:
    """
    ACL, MCL and side-specific meniscus targets.

    W1.2 evaluates injury assertion around the injury term
    itself. Example:

        "lateral meniscus shows no obvious tear"

    The word "tear" is present, but it is locally negated and
    therefore must not become a positive report assertion.
    """

    folded = fold_for_match(unit)

    concept_pattern, concept_match = first_pattern_match(
        folded,
        patterns_for_language(
            STRUCTURE_PATTERNS[label],
            language,
        ),
    )

    if concept_match is None:

        return None

    concept_start = max(
        0,
        concept_match.start() - 150,
    )

    concept_end = min(
        len(folded),
        concept_match.end() + 150,
    )

    structure_context = folded[concept_start:concept_end]

    injury_patterns = patterns_for_language(
        TEAR_INJURY_PATTERNS,
        language,
    )

    injury_matches = pattern_matches(
        structure_context,
        injury_patterns,
    )

    definite_positive_patterns = []
    uncertain_positive_patterns = []
    negative_injury_patterns = []
    negative_cue_patterns = []
    uncertainty_cue_patterns = []

    for pattern, injury_match in injury_matches:

        is_negated, neg_hits = finding_negation_state(
            structure_context,
            injury_match,
            language,
        )

        is_uncertain, uncertain_hits = finding_uncertainty_state(
            structure_context,
            injury_match,
            language,
        )

        if is_negated:

            negative_injury_patterns.append(pattern)

            negative_cue_patterns.extend(neg_hits)

        elif is_uncertain:

            uncertain_positive_patterns.append(pattern)

            uncertainty_cue_patterns.extend(uncertain_hits)

        else:

            definite_positive_patterns.append(pattern)

    concept_negative_hits = concept_state_negation(
        folded,
        concept_match,
        language,
    )

    negative_cue_patterns.extend(concept_negative_hits)

    degeneration_hits = all_pattern_strings(
        structure_context,
        patterns_for_language(
            DEGENERATION_PATTERNS,
            language,
        ),
    )

    history_hits = all_pattern_strings(
        structure_context,
        patterns_for_language(
            HISTORY_PATTERNS,
            language,
        ),
    )

    severity = extract_severity(structure_context)

    has_positive = len(definite_positive_patterns) > 0

    has_uncertain = len(uncertain_positive_patterns) > 0

    has_negative = (len(negative_injury_patterns) > 0) or (
        len(concept_negative_hits) > 0 and not has_positive
    )

    if has_positive and has_negative:

        assertion = "mixed"

    elif has_positive:

        assertion = "positive"

    elif has_uncertain and has_negative:

        assertion = "mixed"

    elif has_uncertain:

        assertion = "uncertain"

    elif has_negative:

        assertion = "negative"

    elif (
        label
        in [
            "Medial Meniscus",
            "Lateral Meniscus",
        ]
        and degeneration_hits
    ):

        assertion = "related_abnormality"

    else:

        assertion = "mentioned_neutral"

    positive_patterns = sorted(
        set(definite_positive_patterns + uncertain_positive_patterns)
    )

    return {
        "Label": label,
        "Section": section,
        "Assertion": assertion,
        "EvidenceText": unit,
        "MatchedConceptPattern": concept_pattern,
        "PositiveCuePatterns": "|".join(positive_patterns),
        "NegativeCuePatterns": "|".join(sorted(set(negative_cue_patterns))),
        "UncertaintyCuePatterns": "|".join(sorted(set(uncertainty_cue_patterns))),
        "HistoryCuePatterns": "|".join(sorted(set(history_hits))),
        "Severity": "|".join(severity),
        "SectionWeight": evidence_section_weight(section),
    }


def classify_oa_unit(
    label: str,
    unit: str,
    language: str,
    section: str,
) -> Optional[Dict[str, Any]]:
    """
    Side-specific OA evidence.

    W1.2 requires the compartment mention and OA/cartilage
    abnormality to be locally close. W1.1 could connect a
    medial OA phrase to a later lateral-compartment sentence
    when a long exported unit contained multiple compartments.
    """

    folded = fold_for_match(unit)

    side = {
        "Medial OA": "medial",
        "Lateral OA": "lateral",
        "PF OA": "pf",
    }[label]

    compartment_matches = pattern_matches(
        folded,
        patterns_for_language(
            COMPARTMENT_PATTERNS[side],
            language,
        ),
    )

    oa_matches = pattern_matches(
        folded,
        patterns_for_language(
            OA_PATTERNS,
            language,
        ),
    )

    if not compartment_matches or not oa_matches:

        return None

    pair_candidates = []

    for compartment_pattern, compartment_match in compartment_matches:

        compartment_center = (compartment_match.start() + compartment_match.end()) / 2.0

        for oa_pattern, oa_match in oa_matches:

            oa_center = (oa_match.start() + oa_match.end()) / 2.0

            distance = abs(compartment_center - oa_center)

            pair_candidates.append(
                (
                    distance,
                    compartment_pattern,
                    compartment_match,
                    oa_pattern,
                    oa_match,
                )
            )

    pair_candidates.sort(key=lambda item: item[0])

    (
        distance,
        compartment_pattern,
        compartment_match,
        oa_pattern,
        oa_match,
    ) = pair_candidates[0]

    # Conservative local relationship threshold.
    if distance > 150:

        return None

    left = max(
        0,
        min(
            compartment_match.start(),
            oa_match.start(),
        )
        - 55,
    )

    right = min(
        len(folded),
        max(
            compartment_match.end(),
            oa_match.end(),
        )
        + 70,
    )

    local_context = folded[left:right]

    # Re-locate OA in the local context for cue scoping.
    local_oa_pattern, local_oa_match = first_pattern_match(
        local_context,
        patterns_for_language(
            OA_PATTERNS,
            language,
        ),
    )

    if local_oa_match is None:

        return None

    finding_negated, negation_hits = finding_negation_state(
        local_context,
        local_oa_match,
        language,
    )

    finding_uncertain, uncertainty_hits = finding_uncertainty_state(
        local_context,
        local_oa_match,
        language,
    )

    history_hits = all_pattern_strings(
        local_context,
        patterns_for_language(
            HISTORY_PATTERNS,
            language,
        ),
    )

    severity = extract_severity(local_context)

    if finding_negated:

        assertion = "negative"

    elif finding_uncertain:

        assertion = "uncertain"

    else:

        assertion = "positive"

    return {
        "Label": label,
        "Section": section,
        "Assertion": assertion,
        "EvidenceText": unit,
        "MatchedConceptPattern": (f"{compartment_pattern}" " && " f"{oa_pattern}"),
        "PositiveCuePatterns": oa_pattern,
        "NegativeCuePatterns": "|".join(sorted(set(negation_hits))),
        "UncertaintyCuePatterns": "|".join(sorted(set(uncertainty_hits))),
        "HistoryCuePatterns": "|".join(sorted(set(history_hits))),
        "Severity": "|".join(severity),
        "SectionWeight": evidence_section_weight(section),
    }


def classify_direct_condition_unit(
    label: str,
    unit: str,
    language: str,
    section: str,
) -> Optional[Dict[str, Any]]:

    folded = fold_for_match(unit)

    concept_pattern, concept_match = first_pattern_match(
        folded,
        patterns_for_language(
            DIRECT_CONDITION_PATTERNS[label],
            language,
        ),
    )

    related_pattern = None
    related_match = None

    if concept_match is None and label in RELATED_PATTERNS:

        related_pattern, related_match = first_pattern_match(
            folded,
            patterns_for_language(
                RELATED_PATTERNS[label],
                language,
            ),
        )

    active_match = concept_match if concept_match is not None else related_match

    if active_match is None:

        return None

    finding_negated, negation_hits = finding_negation_state(
        folded,
        active_match,
        language,
    )

    # Also support target-state expressions after the concept,
    # e.g. "Baker cyst: None".
    state_negation_hits = concept_state_negation(
        folded,
        active_match,
        language,
    )

    negation_hits = sorted(set(negation_hits + state_negation_hits))

    finding_uncertain, uncertainty_hits = finding_uncertainty_state(
        folded,
        active_match,
        language,
    )

    local_context = context_around_match(
        folded,
        active_match,
        radius=85,
    )

    history_hits = all_pattern_strings(
        local_context,
        patterns_for_language(
            HISTORY_PATTERNS,
            language,
        ),
    )

    severity = extract_severity(local_context)

    is_negated = finding_negated or len(state_negation_hits) > 0

    if related_match is not None:

        if is_negated:

            assertion = "negative"

        elif finding_uncertain:

            assertion = "uncertain"

        else:

            assertion = "related_abnormality"

    else:

        if is_negated:

            assertion = "negative"

        elif finding_uncertain:

            assertion = "uncertain"

        else:

            assertion = "positive"

    return {
        "Label": label,
        "Section": section,
        "Assertion": assertion,
        "EvidenceText": unit,
        "MatchedConceptPattern": (
            concept_pattern if concept_match is not None else related_pattern
        ),
        "PositiveCuePatterns": (concept_pattern if concept_match is not None else ""),
        "NegativeCuePatterns": "|".join(negation_hits),
        "UncertaintyCuePatterns": "|".join(sorted(set(uncertainty_hits))),
        "HistoryCuePatterns": "|".join(sorted(set(history_hits))),
        "Severity": "|".join(severity),
        "SectionWeight": evidence_section_weight(section),
    }


def classify_report_label_evidence(
    report: str,
    language: str,
    label: str,
) -> List[Dict[str, Any]]:

    records = []

    for section, unit in report_units(report):

        if label in [
            "ACL",
            "MCL",
            "Medial Meniscus",
            "Lateral Meniscus",
        ]:

            record = classify_structure_unit(
                label,
                unit,
                language,
                section,
            )

        elif label in [
            "Medial OA",
            "Lateral OA",
            "PF OA",
        ]:

            record = classify_oa_unit(
                label,
                unit,
                language,
                section,
            )

        else:

            record = classify_direct_condition_unit(
                label,
                unit,
                language,
                section,
            )

        if record is not None:

            records.append(record)

    return records


# ============================================================
# 9. AGGREGATE MULTIPLE EVIDENCE UNITS
# ============================================================

ASSERTION_PRIORITY = {
    "positive": 6,
    "negative": 5,
    "uncertain": 4,
    "related_abnormality": 3,
    "mentioned_neutral": 2,
}


def aggregate_assertion(
    records: List[Dict[str, Any]],
) -> Tuple[
    str,
    Optional[Dict[str, Any]],
]:
    """
    Preserve conflict rather than hiding it.
    """

    if not records:

        return (
            "not_mentioned",
            None,
        )

    assertion_set = {record["Assertion"] for record in records}

    has_positive = "positive" in assertion_set

    has_negative = "negative" in assertion_set

    if has_positive and has_negative:

        aggregate = "mixed"

    elif has_positive:

        aggregate = "positive"

    elif has_negative:

        aggregate = "negative"

    elif "uncertain" in assertion_set:

        aggregate = "uncertain"

    elif "related_abnormality" in assertion_set:

        aggregate = "related_abnormality"

    else:

        aggregate = "mentioned_neutral"

    # Pick a representative evidence unit:
    # assertion priority, then impression/findings/body.
    representative = sorted(
        records,
        key=lambda row: (
            ASSERTION_PRIORITY.get(
                row["Assertion"],
                0,
            ),
            row["SectionWeight"],
        ),
        reverse=True,
    )[0]

    return (
        aggregate,
        representative,
    )


# ============================================================
# 10. W2 SOURCE HELPERS
# ============================================================


def read_csv_from_dir_or_zip(
    source: Path,
    filename_suffix: str,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[str],
]:

    if not source.exists():
        return (
            None,
            None,
        )

    if source.is_dir():

        direct_candidates = [
            source / filename_suffix,
            source / "results" / filename_suffix,
            source / "review" / filename_suffix,
            source / "private" / filename_suffix,
        ]

        for candidate in direct_candidates:
            if candidate.exists():
                return (
                    pd.read_csv(candidate),
                    str(candidate),
                )

        try:
            matches = list(source.glob(f"**/{filename_suffix}"))
        except Exception:
            matches = []

        if matches:
            return (
                pd.read_csv(matches[0]),
                str(matches[0]),
            )

        return (
            None,
            None,
        )

    if source.is_file() and source.suffix.casefold() == ".zip":

        try:
            with zipfile.ZipFile(
                source,
                "r",
            ) as z:

                matches = [
                    name for name in z.namelist() if name.endswith(filename_suffix)
                ]

                if not matches:
                    return (
                        None,
                        None,
                    )

                name = matches[0]

                return (
                    pd.read_csv(io.BytesIO(z.read(name))),
                    f"{source}::{name}",
                )

        except zipfile.BadZipFile:

            return (
                None,
                None,
            )

    return (
        None,
        None,
    )


def candidate_w1_sources() -> List[Path]:

    sources = []

    explicit = os.environ.get(
        "W2_W1_SOURCE",
        "",
    ).strip()

    if explicit:
        sources.append(Path(explicit))

    sources.extend(
        [
            Path("/kaggle/working/rsna_w1/results"),
            Path("/kaggle/working/rsna_w1"),
            Path("/kaggle/working/rsna_w1.zip"),
            Path("/mnt/data/rsna_w1.zip"),
        ]
    )

    return sources


def candidate_w13_sources() -> List[Path]:

    sources = []

    explicit = os.environ.get(
        "W2_W13_SOURCE",
        "",
    ).strip()

    if explicit:
        sources.append(Path(explicit))

    sources.extend(
        [
            Path("/kaggle/working/rsna_w1_3"),
            Path("/kaggle/working/" "rsna_w1_3_zip_file.zip"),
            Path("/mnt/data/" "rsna_w1_3_zip_file.zip"),
        ]
    )

    # Common Kaggle input fallback.
    kaggle_input = Path("/kaggle/input")

    if kaggle_input.exists():

        try:
            sources.extend(sorted(kaggle_input.glob("**/rsna_w1_3_zip_file.zip")))
        except Exception:
            pass

    # De-duplicate.
    seen = set()
    result = []

    for source in sources:

        key = str(source)

        if key in seen:
            continue

        seen.add(key)
        result.append(source)

    return result


def load_w13_merged() -> Tuple[
    pd.DataFrame,
    str,
]:

    for source in candidate_w13_sources():

        df, description = read_csv_from_dir_or_zip(
            source,
            "10_W1_3_ADJUDICATED_MERGED.csv",
        )

        if df is not None:

            if len(df) != EXPECTED_W13_CASES:

                raise RuntimeError(
                    "W1.3 merged file must contain "
                    f"{EXPECTED_W13_CASES} rows; "
                    f"found {len(df)}."
                )

            required = [
                "CaseID",
                "Label",
                "Language",
                "Report",
                "ReviewerStatus",
                "ReviewerReportAssertion",
                "ReportAssertion",
                "Gold",
            ]

            missing = [column for column in required if column not in df.columns]

            if missing:

                raise RuntimeError("W1.3 merged file missing columns: " f"{missing}")

            return (
                df,
                description if description is not None else str(source),
            )

    raise FileNotFoundError(
        "Could not locate W1.3 adjudicated merged output. "
        "Set W2_W13_SOURCE to the W1.3 results directory "
        "or rsna_w1_3_zip_file.zip."
    )


def load_w1_language_map() -> Tuple[
    Dict[str, str],
    Optional[str],
]:

    for source in candidate_w1_sources():

        df, description = read_csv_from_dir_or_zip(
            source,
            "01_report_inventory.csv",
        )

        if df is None:
            continue

        if UID_COLUMN not in df.columns or "Language" not in df.columns:
            continue

        mapping = {}

        for _, row in df.iterrows():

            mapping[str(row[UID_COLUMN])] = canonicalize_language(row["Language"])

        return (
            mapping,
            description,
        )

    return (
        {},
        None,
    )


# ============================================================
# 11. TRAIN DATA
# ============================================================


def load_train_data() -> pd.DataFrame:

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(f"train.csv not found: {TRAIN_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)

    required = [
        UID_COLUMN,
        REPORT_COLUMN,
    ] + LABEL_COLUMNS

    missing = [column for column in required if column not in train_df.columns]

    if missing:

        raise RuntimeError("train.csv missing required columns: " f"{missing}")

    if train_df[REPORT_COLUMN].fillna("").astype(str).str.strip().eq("").any():

        raise RuntimeError(
            "W2 requires every training study to have " "a non-empty Report."
        )

    label_missing = train_df[LABEL_COLUMNS].isna().sum(axis=1)

    train_df["W2_LabelStatus"] = np.where(
        label_missing == 0,
        "gold",
        np.where(
            label_missing == len(LABEL_COLUMNS),
            "unlabeled",
            "partial",
        ),
    )

    partial_n = int((train_df["W2_LabelStatus"] == "partial").sum())

    if partial_n > 0:

        raise RuntimeError(
            "W2 expected no partially labeled studies, " f"but found {partial_n}."
        )

    gold_n = int((train_df["W2_LabelStatus"] == "gold").sum())

    unlabeled_n = int((train_df["W2_LabelStatus"] == "unlabeled").sum())

    if gold_n != EXPECTED_GOLD_STUDIES:

        raise RuntimeError(
            f"Expected {EXPECTED_GOLD_STUDIES} gold " f"studies; found {gold_n}."
        )

    if unlabeled_n != EXPECTED_UNLABELED_STUDIES:

        warnings.warn(
            "Unlabeled count differs from the established "
            f"{EXPECTED_UNLABELED_STUDIES}: {unlabeled_n}"
        )

    language_map, language_source = load_w1_language_map()

    normalized_reports = []
    canonical_languages = []

    for _, row in train_df.iterrows():

        normalized, _ = normalize_report(row[REPORT_COLUMN])

        normalized_reports.append(normalized)

        uid = str(row[UID_COLUMN])

        if uid in language_map:

            language = language_map[uid]

        else:

            language = detect_language_fallback(normalized)

        canonical_languages.append(language)

    train_df["W2_NormalizedReport"] = normalized_reports

    train_df["W2_Language"] = canonical_languages

    print(
        "Train studies:",
        len(train_df),
    )

    print(
        "Gold / unlabeled:",
        gold_n,
        "/",
        unlabeled_n,
    )

    if language_source:

        print(
            "Language assignments reused from:",
            language_source,
        )

    else:

        print(
            "W1 language inventory not found; " "using W2 fallback language detection."
        )

    return train_df


def build_pair_table(
    study_df: pd.DataFrame,
    include_gold: bool,
) -> pd.DataFrame:

    rows = []

    for _, row in study_df.iterrows():

        uid = str(row[UID_COLUMN])

        report = str(row["W2_NormalizedReport"])

        language = str(row["W2_Language"])

        for label in LABEL_COLUMNS:

            item = {
                "PairID": (f"{uid}" f"||{label}"),
                UID_COLUMN: uid,
                "Label": label,
                "Language": language,
                "Report": report,
            }

            if include_gold:

                item["Gold"] = int(row[label])

            rows.append(item)

    return pd.DataFrame(rows)


# ============================================================
# 12. NLI MODEL
# ============================================================


def resolve_model_source() -> str:

    if MODEL_PATH:

        model_path = Path(MODEL_PATH)

        if not model_path.exists():

            raise FileNotFoundError("W2_MODEL_PATH does not exist: " f"{model_path}")

        return str(model_path)

    known_local_paths = [
        Path("/kaggle/input/" "mdeberta-v3-base-mnli-xnli"),
        Path(
            "/kaggle/input/"
            "mdeberta-v3-base-mnli-xnli/"
            "MoritzLaurer/"
            "mDeBERTa-v3-base-mnli-xnli"
        ),
    ]

    for candidate in known_local_paths:

        if candidate.exists() and (candidate / "config.json").exists():

            return str(candidate)

    return MODEL_ID


class NLILogitWrapper:
    """
    Namespace placeholder replaced after torch import.
    """


class NLIScorer:

    def __init__(
        self,
        model_source: str,
    ) -> None:

        try:
            import torch
            import torch.nn as nn
            import transformers
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

        except Exception as exc:

            raise RuntimeError(
                "W2 semantic inference requires torch and " "transformers."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.transformers = transformers

        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        print(
            "Transformers version:",
            transformers.__version__,
        )

        print(
            "CUDA available:",
            torch.cuda.is_available(),
        )

        print(
            "GPU count:",
            torch.cuda.device_count(),
        )

        print(
            "NLI model source:",
            model_source,
        )

        try:

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                local_files_only=LOCAL_FILES_ONLY,
            )

            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                local_files_only=LOCAL_FILES_ONLY,
                dtype=torch.float32,
            )

        except Exception as exc:

            raise RuntimeError(
                "Could not load the multilingual NLI model. "
                "If Kaggle internet is disabled, add the model "
                "as a Kaggle dataset and set W2_MODEL_PATH to "
                "the local model directory. "
                f"Requested source: {model_source}"
            ) from exc

        # Robustly discover NLI label ids.
        label_map = {}

        for idx, name in base_model.config.id2label.items():

            normalized_name = str(name).casefold().strip()

            label_map[normalized_name] = int(idx)

        def find_label_id(
            desired: str,
        ) -> Optional[int]:

            for name, idx in label_map.items():

                if desired in name:

                    return idx

            return None

        self.entailment_id = find_label_id("entail")

        self.neutral_id = find_label_id("neutral")

        self.contradiction_id = find_label_id("contrad")

        if (
            self.entailment_id is None
            or self.neutral_id is None
            or self.contradiction_id is None
        ):

            # Known 3-way model fallback.
            if (
                getattr(
                    base_model.config,
                    "num_labels",
                    None,
                )
                == 3
            ):

                print(
                    "WARNING: NLI labels were not named "
                    "clearly. Falling back to the model-card "
                    "3-way order: entailment, neutral, "
                    "contradiction."
                )

                self.entailment_id = 0
                self.neutral_id = 1
                self.contradiction_id = 2

            else:

                raise RuntimeError(
                    "W2 requires a 3-way NLI model with "
                    "entailment / neutral / contradiction."
                )

        print(
            "NLI label ids:",
            f"entailment={self.entailment_id},",
            f"neutral={self.neutral_id},",
            f"contradiction={self.contradiction_id}",
        )

        class _Wrapper(nn.Module):

            def __init__(
                self,
                model,
            ):

                super().__init__()

                self.model = model

            def forward(
                self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
            ):

                kwargs = {
                    "input_ids": input_ids,
                }

                if attention_mask is not None:

                    kwargs["attention_mask"] = attention_mask

                if token_type_ids is not None:

                    kwargs["token_type_ids"] = token_type_ids

                return self.model(**kwargs).logits

        wrapper = _Wrapper(base_model)

        wrapper = wrapper.to(self.device)

        if USE_MULTI_GPU and torch.cuda.device_count() >= 2:

            wrapper = nn.DataParallel(
                wrapper,
                device_ids=list(range(torch.cuda.device_count())),
            )

            print(
                "W2 semantic model: DataParallel "
                f"across {torch.cuda.device_count()} GPUs."
            )

        else:

            print("W2 semantic model: single device.")

        self.model = wrapper
        self.model.eval()

    def score(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> np.ndarray:
        """
        Returns N x 3 array:
          [:,0] entailment
          [:,1] neutral
          [:,2] contradiction
        """

        if len(premises) != len(hypotheses):

            raise ValueError("premises and hypotheses length mismatch.")

        if len(premises) == 0:

            return np.zeros(
                (
                    0,
                    3,
                ),
                dtype=np.float32,
            )

        torch = self.torch

        output_batches = []

        for start_index in range(
            0,
            len(premises),
            NLI_BATCH_SIZE,
        ):

            end_index = min(
                len(premises),
                start_index + NLI_BATCH_SIZE,
            )

            premise_batch = list(premises[start_index:end_index])

            hypothesis_batch = list(hypotheses[start_index:end_index])

            encoded = self.tokenizer(
                premise_batch,
                hypothesis_batch,
                padding=True,
                truncation=True,
                max_length=NLI_MAX_LENGTH,
                return_tensors="pt",
            )

            model_inputs = {}

            for key in [
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ]:

                if key in encoded:

                    model_inputs[key] = encoded[key].to(self.device)

            with torch.inference_mode():

                logits = self.model(**model_inputs)

                probs = torch.softmax(
                    logits.float(),
                    dim=-1,
                )

            entailment = (
                probs[
                    :,
                    self.entailment_id,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            neutral = (
                probs[
                    :,
                    self.neutral_id,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            contradiction = (
                probs[
                    :,
                    self.contradiction_id,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            output_batches.append(
                np.stack(
                    [
                        entailment,
                        neutral,
                        contradiction,
                    ],
                    axis=1,
                )
            )

        return np.concatenate(
            output_batches,
            axis=0,
        )

    def sanity_check(self) -> None:
        """
        Minimal diagnostic only. It does not tune thresholds.

        We test entailment of explicit positive/negative English
        fracture statements against positive and negative
        hypotheses. This catches gross model/input/label problems.
        """

        premises = [
            "There is an acute fracture of the knee.",
            "No fracture is present in the knee.",
        ]

        positive_hypotheses = [
            DIRECT_HYPOTHESES["Fracture"],
            DIRECT_HYPOTHESES["Fracture"],
        ]

        negative_hypotheses = [
            NEGATIVE_HYPOTHESES["Fracture"],
            NEGATIVE_HYPOTHESES["Fracture"],
        ]

        positive_scores = self.score(
            premises,
            positive_hypotheses,
        )

        negative_scores = self.score(
            premises,
            negative_hypotheses,
        )

        print("NLI sanity check " "(entailment to positive / negative hypothesis):")

        for premise, pos, neg in zip(
            premises,
            positive_scores[:, 0],
            negative_scores[:, 0],
        ):
            print(
                "  ",
                repr(premise),
                "->",
                f"positive={float(pos):.4f}",
                f"negative={float(neg):.4f}",
            )


# ============================================================
# 12A. SEMANTIC TARGET-ANCHOR GATING
# ============================================================
#
# W2 v2 benchmark showed that dual positive/negative entailment
# fixed the catastrophic "everything becomes negative" problem,
# but NLI could still attach a high-confidence decision to the
# WRONG anatomy/target (for example lateral-meniscus text being
# used as evidence for medial OA).
#
# Therefore semantic inference is now separated into:
#
#   entity/target grounding  -> deterministic broad anchor gate
#   assertion semantics      -> multilingual NLI
#
# The anchor gate is intentionally broader than the W1.2 rule
# engine. It does NOT decide positive/negative. It only asks:
#
#   "Is this clause actually about this target?"
#
# NLI is then allowed to determine assertion polarity/state.
#
# This is a general anti-cross-anatomy safeguard rather than a
# challenge-label rule.
# ============================================================

COLLECTIVE_MENISCUS_ANCHORS = {
    "universal": [
        r"\bboth menisc\w*\b",
        r"\bbilateral menisc\w*\b",
    ],
    "es": [
        r"\bambos menisc\w*\b",
        r"\blos dos menisc\w*\b",
    ],
    "tr": [
        r"\bher iki menisk\w*\b",
    ],
    "bcs": [
        r"\boba menisk\w*\b",
        r"\bobostrano.*menisk\w*\b",
    ],
    "el": [
        r"\bκαι των δυο μηνισκ\w*\b",
        r"\bαμφοτερ\w* μηνισκ\w*\b",
    ],
    "de": [
        r"\bbeide menisk\w*\b",
    ],
    "bg": [
        r"\bдвата мениск\w*\b",
        r"\bдвата менискус\w*\b",
    ],
    "nl": [
        r"\bbeide menisc\w*\b",
    ],
    "fr": [
        r"\bles deux menisqu\w*\b",
    ],
}


GLOBAL_OA_ANCHORS = {
    "universal": [
        r"\btricompartmental\b",
        r"\btri-compartmental\b",
        r"\ball three compartments\b",
        r"\bboth compartments\b",
        r"\bbilateral compartment\w*\b",
    ],
    "es": [
        r"\btricompartimental\w*\b",
        r"\bambos compartiment\w*\b",
    ],
    "tr": [
        r"\btrikompartm\w*\b",
        r"\bher iki kompartm\w*\b",
    ],
    "bcs": [
        r"\btrikompartment\w*\b",
        r"\boba kompartment\w*\b",
        r"\bobostran\w*\b",
        r"\boa promjen\w* obostran\w*\b",
    ],
    "el": [
        r"\bτριδιαμερισματικ\w*\b",
        r"\bκαι στα δυο διαμερισμ\w*\b",
    ],
    "de": [
        r"\btrikompartiment\w*\b",
        r"\bbeide kompartiment\w*\b",
    ],
    "bg": [
        r"\bтрикомпартимент\w*\b",
        r"\bдвата компартимент\w*\b",
    ],
    "nl": [
        r"\btricompartment\w*\b",
        r"\bbeide compartiment\w*\b",
    ],
    "fr": [
        r"\btricompartimental\w*\b",
        r"\bles deux compartiment\w*\b",
    ],
}


FRACTURE_DIRECT_ANCHOR_EXTRA = {
    "universal": [
        r"\bbony avulsion\b",
        r"\bosseous avulsion\b",
        r"\bbone avulsion\b",
    ],
    "de": [
        r"\bknochern\w*\s+ausriss\w*\b",
        r"\bknöchern\w*\s+ausriss\w*\b",
    ],
}


FRACTURE_RELATED_ANCHORS = {
    "universal": [
        r"\bbone bruise\b",
        r"\bbone marrow edema\b",
        r"\bmarrow edema\b",
        r"\bosseous edema\b",
        r"\bsubchondral edema\b",
        r"\bimpaction\b",
    ],
    "es": [
        r"\bedema oseo\b",
        r"\bedema de medula osea\b",
    ],
    "tr": [
        r"\bkemik iligi odem\w*\b",
        r"\bkemik iliği ödem\w*\b",
    ],
    "bcs": [
        r"\bkostan\w* edem\w*\b",
    ],
    "el": [
        r"\bοστεομυελικ\w* οιδημα\b",
        r"\bοιδημα του οστικου μυελου\b",
        r"\bοστικ\w* μωλωπ\w*\b",
    ],
    "de": [
        r"\bknochenmarkodem\b",
        r"\bknochenodem\b",
        r"\bbone bruise\b",
    ],
    "bg": [
        r"\bкостно[- ]?мозъч\w* едем\b",
        r"\bкостномозъч\w* едем\b",
    ],
    "nl": [
        r"\bbeenmergoedeem\b",
        r"\bbotoedeem\b",
    ],
    "fr": [
        r"\boedeme osseux\b",
        r"\boedeme medullaire\b",
    ],
}


PF_BROAD_ANCHORS = {
    "universal": [
        r"\bpatell\w*\b",
        r"\btrochle\w*\b",
        r"\bretropatellar\w*\b",
    ],
    "es": [
        r"\brotul\w*\b",
        r"\btrocle\w*\b",
    ],
    "tr": [
        r"\bpatell\w*\b",
        r"\btrokle\w*\b",
    ],
    "bcs": [
        r"\bpatel\w*\b",
        r"\btrohl\w*\b",
    ],
    "el": [
        r"\bεπιγονατιδ\w*\b",
        r"\bτροχιλ\w*\b",
    ],
    "de": [
        r"\bpatell\w*\b",
        r"\btrochle\w*\b",
    ],
    "bg": [
        r"\bпател\w*\b",
        r"\bтрохле\w*\b",
    ],
    "nl": [
        r"\bpatell\w*\b",
        r"\btrochle\w*\b",
    ],
    "fr": [
        r"\brotul\w*\b",
        r"\btrochle\w*\b",
        r"\bpatell\w*\b",
    ],
}


def _matches_any_pattern(
    folded_text: str,
    patterns: Sequence[str],
) -> bool:

    return any(
        re.search(
            pattern,
            folded_text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


def semantic_target_anchor_allowed(
    label: str,
    text: str,
    language: str,
    evidence_kind: str = "direct",
) -> bool:
    """
    Broad target grounding for semantic NLI.

    This function does NOT decide pathology presence.
    It only blocks cross-target evidence leakage.
    """

    folded = fold_for_match(text)

    # --------------------------------------------------------
    # Ligaments
    # --------------------------------------------------------
    if label in [
        "ACL",
        "MCL",
    ]:

        return _matches_any_pattern(
            folded,
            patterns_for_language(
                STRUCTURE_PATTERNS[label],
                language,
            ),
        )

    # --------------------------------------------------------
    # Menisci: side-specific anatomy OR explicit "both menisci"
    # --------------------------------------------------------
    if label in [
        "Medial Meniscus",
        "Lateral Meniscus",
    ]:

        side_specific = _matches_any_pattern(
            folded,
            patterns_for_language(
                STRUCTURE_PATTERNS[label],
                language,
            ),
        )

        collective = _matches_any_pattern(
            folded,
            patterns_for_language(
                COLLECTIVE_MENISCUS_ANCHORS,
                language,
            ),
        )

        return side_specific or collective

    # --------------------------------------------------------
    # OA: require target compartment or a clear global /
    # tricompartmental/both-compartment statement.
    # --------------------------------------------------------
    if label in [
        "Medial OA",
        "Lateral OA",
        "PF OA",
    ]:

        side = {
            "Medial OA": "medial",
            "Lateral OA": "lateral",
            "PF OA": "pf",
        }[label]

        compartment_anchor = _matches_any_pattern(
            folded,
            patterns_for_language(
                COMPARTMENT_PATTERNS[side],
                language,
            ),
        )

        global_anchor = _matches_any_pattern(
            folded,
            patterns_for_language(
                GLOBAL_OA_ANCHORS,
                language,
            ),
        )

        if label == "PF OA":

            pf_broad = _matches_any_pattern(
                folded,
                patterns_for_language(
                    PF_BROAD_ANCHORS,
                    language,
                ),
            )

            return compartment_anchor or global_anchor or pf_broad

        return compartment_anchor or global_anchor

    # --------------------------------------------------------
    # Direct conditions
    # --------------------------------------------------------
    if label in DIRECT_CONDITION_PATTERNS:

        direct_anchor = _matches_any_pattern(
            folded,
            patterns_for_language(
                DIRECT_CONDITION_PATTERNS[label],
                language,
            ),
        )

        if direct_anchor:

            return True

    # --------------------------------------------------------
    # Fracture special cases
    # --------------------------------------------------------
    if label == "Fracture":

        if evidence_kind == "direct":

            return _matches_any_pattern(
                folded,
                patterns_for_language(
                    FRACTURE_DIRECT_ANCHOR_EXTRA,
                    language,
                ),
            )

        return _matches_any_pattern(
            folded,
            patterns_for_language(
                FRACTURE_RELATED_ANCHORS,
                language,
            ),
        )

    # --------------------------------------------------------
    # Related-abnormality channel can use the W1.2 related
    # lexicon where it exists.
    # --------------------------------------------------------
    if evidence_kind == "related" and label in RELATED_PATTERNS:

        if _matches_any_pattern(
            folded,
            patterns_for_language(
                RELATED_PATTERNS[label],
                language,
            ),
        ):

            return True

    # For related OA, target compartment anchoring above is
    # already sufficient. For related ligament/meniscus states,
    # anatomy anchoring above is sufficient.

    return False


# ============================================================
# 13. STAGE A: HYBRID STRUCTURED REPORT EXTRACTION
# ============================================================


def capped_report_units(
    report: str,
) -> List[Tuple[str, str]]:

    units = report_units(report)

    if len(units) <= MAX_UNITS_PER_REPORT:

        return units

    ranked = []

    for index, (
        section,
        text,
    ) in enumerate(units):

        section_priority = {
            "impression": 3,
            "findings": 2,
            "body": 1,
        }.get(
            section,
            1,
        )

        ranked.append(
            (
                -section_priority,
                index,
                section,
                text,
            )
        )

    selected = sorted(ranked)[:MAX_UNITS_PER_REPORT]

    selected = sorted(
        selected,
        key=lambda item: item[1],
    )

    return [
        (
            section,
            text,
        )
        for _, _, section, text in selected
    ]


def unit_has_uncertainty(
    unit: str,
    language: str,
) -> bool:

    folded = fold_for_match(unit)

    patterns = patterns_for_language(
        UNCERTAINTY_PATTERNS,
        language,
    )

    return any(
        re.search(
            pattern,
            folded,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


def collect_severity_text(
    rule_records: List[Dict[str, Any]],
    selected_evidence: str,
) -> str:

    tokens = set()

    for record in rule_records:

        for token in str(
            record.get(
                "Severity",
                "",
            )
        ).split("|"):

            token = token.strip()

            if token:

                tokens.add(token)

    if selected_evidence:

        folded = fold_for_match(selected_evidence)

        for token in extract_severity(folded):

            if token:

                tokens.add(token)

    return "|".join(sorted(tokens))


def severity_flags(
    severity_text: str,
    evidence_text: str,
) -> Dict[str, float]:

    combined = (f"{severity_text} " f"{fold_for_match(evidence_text)}").casefold()

    low_patterns = [
        r"\bminimal\b",
        r"\bmild\b",
        r"\bgrade[_ ]?(?:grade )?(?:1|i)\b",
        r"\bpartial\b",
        r"\binterstitial\b",
        r"\blow[- ]grade\b",
    ]

    moderate_patterns = [
        r"\bmoderate\b",
        r"\bgrade[_ ]?(?:grade )?(?:2|ii)\b",
    ]

    high_patterns = [
        r"\bsevere\b",
        r"\bcomplete\b",
        r"\bfull[- ]?thickness\b",
        r"\bgrade[_ ]?(?:grade )?(?:3|iii|4|iv)\b",
    ]

    degenerative_patterns = [
        r"\bdegenerat",
        r"\bdegenerative\b",
        r"\bmucoid\b",
    ]

    def has_any(
        patterns: Sequence[str],
    ) -> float:

        return float(
            any(
                re.search(
                    pattern,
                    combined,
                    flags=re.IGNORECASE,
                )
                is not None
                for pattern in patterns
            )
        )

    return {
        "SeverityLow": has_any(low_patterns),
        "SeverityModerate": has_any(moderate_patterns),
        "SeverityHigh": has_any(high_patterns),
        "SeverityDegenerative": has_any(degenerative_patterns),
    }


def semantic_assertion(
    positive_score: float,
    negative_score: float,
    related_score: float,
    positive_evidence: str,
    language: str,
) -> Tuple[
    str,
    float,
]:

    has_uncertainty = bool(positive_evidence) and unit_has_uncertainty(
        positive_evidence,
        language,
    )

    if (
        positive_score >= SEMANTIC_CONFLICT_THRESHOLD
        and negative_score >= SEMANTIC_CONFLICT_THRESHOLD
    ):

        return (
            "mixed",
            float(
                min(
                    positive_score,
                    negative_score,
                )
            ),
        )

    if (
        positive_score >= SEMANTIC_STRONG_THRESHOLD
        and (positive_score - negative_score) >= SEMANTIC_MARGIN
    ):

        if has_uncertainty:

            return (
                "uncertain",
                float(positive_score),
            )

        margin_factor = min(
            1.0,
            max(
                0.0,
                (positive_score - negative_score) / 0.50,
            ),
        )

        confidence = positive_score * (0.5 + 0.5 * margin_factor)

        return (
            "positive",
            float(confidence),
        )

    if (
        negative_score >= SEMANTIC_STRONG_THRESHOLD
        and (negative_score - positive_score) >= SEMANTIC_MARGIN
    ):

        margin_factor = min(
            1.0,
            max(
                0.0,
                (negative_score - positive_score) / 0.50,
            ),
        )

        confidence = negative_score * (0.5 + 0.5 * margin_factor)

        return (
            "negative",
            float(confidence),
        )

    if has_uncertainty and positive_score >= SEMANTIC_WEAK_THRESHOLD:

        return (
            "uncertain",
            float(positive_score),
        )

    if related_score >= SEMANTIC_RELATED_THRESHOLD:

        return (
            "related_abnormality",
            float(related_score),
        )

    return (
        "not_mentioned",
        float(
            max(
                positive_score,
                negative_score,
                related_score,
            )
        ),
    )


def fuse_assertions(
    rule_assertion: str,
    semantic_state: str,
    semantic_confidence: float,
    semantic_positive: float,
    semantic_negative: float,
) -> Tuple[
    str,
    float,
    str,
]:

    rule_confidence = RULE_ASSERTION_CONFIDENCE.get(
        rule_assertion,
        0.0,
    )

    # W1.3 showed that when W1.2 made an explicit binary
    # positive/negative decision, reviewer binary polarity
    # agreement was extremely strong. Therefore semantic NLI is
    # NOT allowed to flip or convert an explicit rule decision.
    if rule_assertion in [
        "positive",
        "negative",
    ]:

        return (
            rule_assertion,
            float(rule_confidence),
            "rule_explicit_protected",
        )

    # Preserve other non-missing W1.2 structured states during
    # the semantic validation phase. W2 semantic inference is
    # currently a recall-recovery channel, not a replacement.
    if rule_assertion in [
        "mixed",
        "uncertain",
        "related_abnormality",
        "mentioned_neutral",
    ]:

        return (
            rule_assertion,
            float(rule_confidence),
            "rule_structured_protected",
        )

    # rule == not_mentioned:
    # Only high-confidence semantic evidence may recover it.
    if (
        semantic_state
        in [
            "positive",
            "negative",
            "uncertain",
            "related_abnormality",
        ]
        and semantic_confidence >= SEMANTIC_RECOVERY_MIN_CONFIDENCE
    ):

        return (
            semantic_state,
            float(semantic_confidence),
            "semantic_recovery_high_confidence",
        )

    return (
        "not_mentioned",
        0.0,
        "no_evidence",
    )


def initialize_pair_state(
    row: pd.Series,
) -> Dict[str, Any]:

    report = str(row["Report"])

    language = canonicalize_language(row["Language"])

    label = str(row["Label"])

    units = capped_report_units(report)

    rule_records = classify_report_label_evidence(
        report,
        language,
        label,
    )

    (
        rule_assertion,
        rule_representative,
    ) = aggregate_assertion(rule_records)

    return {
        "PairID": str(row["PairID"]),
        UID_COLUMN: str(
            row.get(
                UID_COLUMN,
                "",
            )
        ),
        "CaseID": str(
            row.get(
                "CaseID",
                "",
            )
        ),
        "Label": label,
        "Language": language,
        "Report": report,
        "Gold": (
            int(row["Gold"])
            if ("Gold" in row.index and pd.notna(row["Gold"]))
            else np.nan
        ),
        "Units": units,
        "RuleRecords": rule_records,
        "RuleAssertion": rule_assertion,
        "RuleRepresentative": rule_representative,
        "SemanticPositive": 0.0,
        "SemanticNegative": 0.0,
        "SemanticNeutralMax": 0.0,
        "SemanticRelated": 0.0,
        "SemanticPositiveEvidence": "",
        "SemanticNegativeEvidence": "",
        "SemanticRelatedEvidence": "",
    }


def extract_structured_pairs(
    pair_df: pd.DataFrame,
    scorer: NLIScorer,
    cohort_name: str,
) -> pd.DataFrame:

    required = [
        "PairID",
        "Label",
        "Language",
        "Report",
    ]

    missing = [column for column in required if column not in pair_df.columns]

    if missing:

        raise RuntimeError("Pair table missing columns: " f"{missing}")

    cache_key = sha256_text(
        json.dumps(
            {
                "cohort": cohort_name,
                "pair_ids": pair_df["PairID"].astype(str).tolist(),
                "model": MODEL_PATH if MODEL_PATH else MODEL_ID,
                "thresholds": [
                    SEMANTIC_STRONG_THRESHOLD,
                    SEMANTIC_RELATED_THRESHOLD,
                    SEMANTIC_WEAK_THRESHOLD,
                    SEMANTIC_MARGIN,
                ],
                "max_length": NLI_MAX_LENGTH,
                "max_units": MAX_UNITS_PER_REPORT,
                "w12_rule_engine": "W1.2_core",
                "semantic_engine": "dual_entailment_target_grounded_recovery_v2_1",
            },
            sort_keys=True,
        )
    )[:16]

    cache_path = CACHE_ROOT / (f"{cohort_name}_" f"{cache_key}.pkl")

    if cache_path.exists():

        print(
            "Loading Stage A cache:",
            cache_path,
        )

        return pd.read_pickle(cache_path)

    output_rows = []

    total_pairs = len(pair_df)

    print(f"Stage A extracting {total_pairs} " f"report-label pairs: {cohort_name}")

    for chunk_start in range(
        0,
        total_pairs,
        PAIR_CHUNK_SIZE,
    ):

        chunk_end = min(
            total_pairs,
            chunk_start + PAIR_CHUNK_SIZE,
        )

        chunk_df = pair_df.iloc[chunk_start:chunk_end].reset_index(drop=True)

        states = [initialize_pair_state(row) for _, row in chunk_df.iterrows()]

        # ----------------------------------------------------
        # Direct target NLI
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We score TWO explicit hypotheses:
        #
        #   positive proposition
        #   negative proposition
        #
        # and use ENTAILMENT for each.
        #
        # We do NOT treat contradiction of a positive hypothesis
        # as an explicit report-negative. On long radiology reports,
        # taking max contradiction over many unrelated clauses caused
        # a severe extreme-value failure in W2 benchmark v1.

        premises = []
        positive_hypotheses = []
        negative_hypotheses = []
        task_meta = []

        for pair_index, state in enumerate(states):

            positive_hypothesis = DIRECT_HYPOTHESES[state["Label"]]

            negative_hypothesis = NEGATIVE_HYPOTHESES[state["Label"]]

            for unit_index, (
                section,
                unit,
            ) in enumerate(state["Units"]):

                premises.append(unit)

                positive_hypotheses.append(positive_hypothesis)

                negative_hypotheses.append(negative_hypothesis)

                task_meta.append(
                    (
                        pair_index,
                        unit_index,
                        section,
                        unit,
                    )
                )

        positive_scores = scorer.score(
            premises,
            positive_hypotheses,
        )

        negative_scores = scorer.score(
            premises,
            negative_hypotheses,
        )

        for (
            (
                pair_index,
                _,
                _,
                unit,
            ),
            positive_score_row,
            negative_score_row,
        ) in zip(
            task_meta,
            positive_scores,
            negative_scores,
        ):

            positive_entailment = float(positive_score_row[0])

            negative_entailment = float(negative_score_row[0])

            neutral = float(
                max(
                    positive_score_row[1],
                    negative_score_row[1],
                )
            )

            state = states[pair_index]

            if not semantic_target_anchor_allowed(
                label=state["Label"],
                text=unit,
                language=state["Language"],
                evidence_kind="direct",
            ):

                continue

            if positive_entailment > state["SemanticPositive"]:

                state["SemanticPositive"] = positive_entailment

                state["SemanticPositiveEvidence"] = unit

            if negative_entailment > state["SemanticNegative"]:

                state["SemanticNegative"] = negative_entailment

                state["SemanticNegativeEvidence"] = unit

            state["SemanticNeutralMax"] = max(
                state["SemanticNeutralMax"],
                neutral,
            )

        # ----------------------------------------------------
        # Selective related-abnormality NLI
        # ----------------------------------------------------

        related_premises = []
        related_hypotheses = []
        related_meta = []

        for pair_index, state in enumerate(states):

            label = state["Label"]

            if label not in RELATED_HYPOTHESES:

                continue

            direct_strength = max(
                state["SemanticPositive"],
                state["SemanticNegative"],
            )

            # Do not spend extra compute on already strong
            # direct explicit evidence unless the deterministic
            # rule engine itself says "related".
            should_check_related = direct_strength < 0.85 or state["RuleAssertion"] in [
                "related_abnormality",
                "mentioned_neutral",
                "not_mentioned",
            ]

            if not should_check_related:

                continue

            hypothesis = RELATED_HYPOTHESES[label]

            for unit_index, (
                section,
                unit,
            ) in enumerate(state["Units"]):

                related_premises.append(unit)

                related_hypotheses.append(hypothesis)

                related_meta.append(
                    (
                        pair_index,
                        unit_index,
                        section,
                        unit,
                    )
                )

        if related_premises:

            related_scores = scorer.score(
                related_premises,
                related_hypotheses,
            )

            for (
                pair_index,
                _,
                _,
                unit,
            ), score in zip(
                related_meta,
                related_scores,
            ):

                entailment = float(score[0])

                state = states[pair_index]

                if not semantic_target_anchor_allowed(
                    label=state["Label"],
                    text=unit,
                    language=state["Language"],
                    evidence_kind="related",
                ):

                    continue

                if entailment > state["SemanticRelated"]:

                    state["SemanticRelated"] = entailment

                    state["SemanticRelatedEvidence"] = unit

        # ----------------------------------------------------
        # Final semantic + rule fusion
        # ----------------------------------------------------

        for state in states:

            (
                semantic_state,
                semantic_confidence,
            ) = semantic_assertion(
                positive_score=state["SemanticPositive"],
                negative_score=state["SemanticNegative"],
                related_score=state["SemanticRelated"],
                positive_evidence=state["SemanticPositiveEvidence"],
                language=state["Language"],
            )

            (
                fused_state,
                fused_confidence,
                fusion_source,
            ) = fuse_assertions(
                rule_assertion=state["RuleAssertion"],
                semantic_state=semantic_state,
                semantic_confidence=semantic_confidence,
                semantic_positive=state["SemanticPositive"],
                semantic_negative=state["SemanticNegative"],
            )

            rule_rep = state["RuleRepresentative"]

            rule_evidence = (
                ""
                if rule_rep is None
                else str(
                    rule_rep.get(
                        "EvidenceText",
                        "",
                    )
                )
            )

            if fusion_source.startswith("rule") and rule_evidence:

                fused_evidence = rule_evidence

            elif fused_state == "negative":

                fused_evidence = state["SemanticNegativeEvidence"]

            elif fused_state == "related_abnormality":

                fused_evidence = state["SemanticRelatedEvidence"] or rule_evidence

            else:

                fused_evidence = state["SemanticPositiveEvidence"] or rule_evidence

            severity_text = collect_severity_text(
                state["RuleRecords"],
                fused_evidence,
            )

            severity_feature_map = severity_flags(
                severity_text,
                fused_evidence,
            )

            direct_strength = float(
                max(
                    state["SemanticPositive"],
                    state["SemanticNegative"],
                )
            )

            direct_margin = float(state["SemanticPositive"] - state["SemanticNegative"])

            evidence_positive_score = max(
                (
                    RULE_ASSERTION_CONFIDENCE["positive"]
                    if state["RuleAssertion"] == "positive"
                    else 0.0
                ),
                state["SemanticPositive"],
            )

            evidence_negative_score = max(
                (
                    RULE_ASSERTION_CONFIDENCE["negative"]
                    if state["RuleAssertion"] == "negative"
                    else 0.0
                ),
                state["SemanticNegative"],
            )

            uncertainty_flag = float(
                (state["RuleAssertion"] == "uncertain")
                or (semantic_state == "uncertain")
            )

            rule_decidable_flag = float(
                state["RuleAssertion"]
                in [
                    "positive",
                    "negative",
                ]
            )

            evidence_available = fused_state not in [
                "not_mentioned",
                "mentioned_neutral",
            ]

            output = {
                "PairID": state["PairID"],
                UID_COLUMN: state[UID_COLUMN],
                "CaseID": state["CaseID"],
                "Label": state["Label"],
                "Language": state["Language"],
                "Gold": state["Gold"],
                "RuleAssertion": state["RuleAssertion"],
                "RuleEvidenceCount": int(len(state["RuleRecords"])),
                "RuleEvidence": rule_evidence,
                "SemanticAssertion": semantic_state,
                "SemanticAssertionConfidence": float(semantic_confidence),
                "SemanticPositiveScore": float(state["SemanticPositive"]),
                "SemanticNegativeScore": float(state["SemanticNegative"]),
                "SemanticRelatedScore": float(state["SemanticRelated"]),
                "SemanticDirectStrength": direct_strength,
                "SemanticDirectMargin": direct_margin,
                "SemanticPositiveEvidence": state["SemanticPositiveEvidence"],
                "SemanticNegativeEvidence": state["SemanticNegativeEvidence"],
                "SemanticRelatedEvidence": state["SemanticRelatedEvidence"],
                "FusedAssertion": fused_state,
                "FusedAssertionConfidence": float(fused_confidence),
                "FusionSource": fusion_source,
                "FusedEvidence": fused_evidence,
                "SeverityTokens": severity_text,
                "EvidenceAvailable": bool(evidence_available),
                # Compact Stage B features:
                "EvidencePositiveScore": float(evidence_positive_score),
                "EvidenceNegativeScore": float(evidence_negative_score),
                "RelatedScore": float(
                    max(
                        (
                            RULE_ASSERTION_CONFIDENCE["related_abnormality"]
                            if state["RuleAssertion"] == "related_abnormality"
                            else 0.0
                        ),
                        state["SemanticRelated"],
                    )
                ),
                "UncertaintyFlag": uncertainty_flag,
                "RuleDecidableFlag": rule_decidable_flag,
                **severity_feature_map,
            }

            output_rows.append(output)

        print(
            f"  {chunk_end:>6}/{total_pairs} pairs "
            f"({100.0 * chunk_end / total_pairs:5.1f}%)"
        )

    result_df = pd.DataFrame(output_rows)

    result_df.to_pickle(cache_path)

    print(
        "Saved Stage A cache:",
        cache_path,
    )

    return result_df


# ============================================================
# 14. W1.3 STAGE-A BENCHMARK
# ============================================================


def safe_accuracy(
    truth: pd.Series,
    pred: pd.Series,
) -> float:

    if len(truth) == 0:

        return np.nan

    return float((truth.astype(str) == pred.astype(str)).mean())


def benchmark_stage_a(
    scorer: NLIScorer,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    w13_df, source_description = load_w13_merged()

    benchmark_pairs = pd.DataFrame(
        {
            "PairID": w13_df["CaseID"].astype(str),
            "CaseID": w13_df["CaseID"].astype(str),
            UID_COLUMN: (
                w13_df[UID_COLUMN].astype(str) if UID_COLUMN in w13_df.columns else ""
            ),
            "Label": w13_df["Label"].astype(str),
            "Language": w13_df["Language"].astype(str),
            "Report": w13_df["Report"].astype(str),
            "Gold": w13_df["Gold"].astype(int),
        }
    )

    stage_a = extract_structured_pairs(
        benchmark_pairs,
        scorer,
        cohort_name="w13_benchmark",
    )

    eval_df = w13_df[
        [
            "CaseID",
            "ReviewerReportAssertion",
            "ReviewerConfidence",
            "ReportAssertion",
            "Gold",
        ]
    ].merge(
        stage_a,
        on="CaseID",
        how="left",
        suffixes=(
            "_W13",
            "_W2",
        ),
        validate="one_to_one",
    )

    eval_df["RuleMatchesStoredW12"] = (
        eval_df["RuleAssertion"] == eval_df["ReportAssertion"]
    )

    metrics_rows = []

    systems = [
        (
            "W1.2_rule",
            "RuleAssertion",
        ),
        (
            "W2_semantic_only",
            "SemanticAssertion",
        ),
        (
            "W2_hybrid",
            "FusedAssertion",
        ),
    ]

    reviewer = eval_df["ReviewerReportAssertion"]

    for system_name, column in systems:

        prediction = eval_df[column]

        exact = safe_accuracy(
            reviewer,
            prediction,
        )

        reviewer_binary = reviewer.isin(
            [
                "positive",
                "negative",
            ]
        )

        pred_binary = prediction.isin(
            [
                "positive",
                "negative",
            ]
        )

        binary_comparable = reviewer_binary & pred_binary

        binary_accuracy = (
            safe_accuracy(
                reviewer[binary_comparable],
                prediction[binary_comparable],
            )
            if binary_comparable.any()
            else np.nan
        )

        polarity_reversals = int(
            ((reviewer == "positive") & (prediction == "negative")).sum()
            + ((reviewer == "negative") & (prediction == "positive")).sum()
        )

        reviewer_positive = reviewer == "positive"

        positive_recall = (
            float((prediction[reviewer_positive] == "positive").mean())
            if reviewer_positive.any()
            else np.nan
        )

        reviewer_negative = reviewer == "negative"

        negative_recall = (
            float((prediction[reviewer_negative] == "negative").mean())
            if reviewer_negative.any()
            else np.nan
        )

        metrics_rows.append(
            {
                "System": system_name,
                "N": int(len(eval_df)),
                "Exact7StateAgreement": exact,
                "BinaryComparableN": int(binary_comparable.sum()),
                "BinaryPolarityAccuracy": binary_accuracy,
                "PolarityReversals": polarity_reversals,
                "ReviewerPositiveN": int(reviewer_positive.sum()),
                "PositiveRecall": positive_recall,
                "ReviewerNegativeN": int(reviewer_negative.sum()),
                "NegativeRecall": negative_recall,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Recovery analysis for the exact W1.2 weakness identified
    # in W1.3: parser not_mentioned but reviewer found evidence.
    rule_not_mentioned = eval_df["ReportAssertion"] == "not_mentioned"

    reviewer_has_evidence = eval_df["ReviewerReportAssertion"] != "not_mentioned"

    recovery_pool = rule_not_mentioned & reviewer_has_evidence

    eval_df["W2RecoveredRuleMiss"] = recovery_pool & (
        eval_df["FusedAssertion"] != "not_mentioned"
    )

    eval_df["W2RecoveredRuleMissExactly"] = recovery_pool & (
        eval_df["FusedAssertion"] == eval_df["ReviewerReportAssertion"]
    )

    metrics_df["RuleNotMentionedEvidenceMisses"] = int(recovery_pool.sum())

    metrics_df["HybridRecoveredAnyEvidence"] = int(eval_df["W2RecoveredRuleMiss"].sum())

    metrics_df["HybridRecoveredExactAssertion"] = int(
        eval_df["W2RecoveredRuleMissExactly"].sum()
    )

    # Per-label/language hybrid diagnostics.
    detail_rows = []

    for dimension in [
        "Label",
        "Language",
    ]:

        for value, group in eval_df.groupby(dimension):

            detail_rows.append(
                {
                    "Dimension": dimension,
                    "Value": value,
                    "N": int(len(group)),
                    "RuleExact": safe_accuracy(
                        group["ReviewerReportAssertion"],
                        group["RuleAssertion"],
                    ),
                    "SemanticExact": safe_accuracy(
                        group["ReviewerReportAssertion"],
                        group["SemanticAssertion"],
                    ),
                    "HybridExact": safe_accuracy(
                        group["ReviewerReportAssertion"],
                        group["FusedAssertion"],
                    ),
                }
            )

    detail_df = pd.DataFrame(detail_rows)

    eval_df.to_csv(
        RESULT_ROOT / "01_w13_stage_a_case_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        RESULT_ROOT / "02_w13_stage_a_summary.csv",
        index=False,
    )

    detail_df.to_csv(
        RESULT_ROOT / "03_w13_stage_a_by_label_language.csv",
        index=False,
    )

    print()
    print(
        "W1.3 Stage A benchmark source:",
        source_description,
    )

    print(metrics_df.to_string(index=False))

    integrity = float(eval_df["RuleMatchesStoredW12"].mean())

    print(
        "W1.2 rule-engine reproduction:",
        f"{100.0 * integrity:.1f}%",
    )

    return (
        eval_df,
        metrics_df,
    )


# ============================================================
# 15. STAGE B: CHALLENGE ONTOLOGY MAPPING
# ============================================================

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


def make_stage_b_matrix(
    structured_df: pd.DataFrame,
) -> np.ndarray:

    return structured_df[STAGE_B_FEATURES].fillna(0.0).astype(np.float32).values


def fit_stage_b_oof(
    gold_structured: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    pd.DataFrame,
]:

    try:
        from joblib import dump
        from sklearn.calibration import (
            calibration_curve,
        )
        from sklearn.linear_model import (
            LogisticRegression,
        )
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import (
            RepeatedStratifiedKFold,
        )
        from sklearn.pipeline import (
            Pipeline,
        )
        from sklearn.preprocessing import (
            StandardScaler,
        )

    except Exception as exc:

        raise RuntimeError("Stage B requires scikit-learn and joblib.") from exc

    oof_rows = []
    metric_rows = []
    calibrators = {}
    coefficient_rows = []

    for label in LABEL_COLUMNS:

        label_df = (
            gold_structured[gold_structured["Label"] == label]
            .copy()
            .reset_index(drop=True)
        )

        if len(label_df) != EXPECTED_GOLD_STUDIES:

            raise RuntimeError(
                f"{label}: expected "
                f"{EXPECTED_GOLD_STUDIES} gold rows, "
                f"found {len(label_df)}."
            )

        y = label_df["Gold"].astype(int).values

        X = make_stage_b_matrix(label_df)

        min_class_n = int(min(np.bincount(y)))

        n_splits = min(
            CV_SPLITS,
            min_class_n,
        )

        if n_splits < 2:

            raise RuntimeError(f"{label}: not enough examples for " "cross-validation.")

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=CV_REPEATS,
            random_state=CV_RANDOM_STATE,
        )

        probability_sum = np.zeros(
            len(label_df),
            dtype=np.float64,
        )

        probability_count = np.zeros(
            len(label_df),
            dtype=np.int32,
        )

        for train_index, val_index in cv.split(
            X,
            y,
        ):

            pipeline = Pipeline(
                [
                    (
                        "scale",
                        StandardScaler(),
                    ),
                    (
                        "logreg",
                        LogisticRegression(
                            C=CALIBRATION_C,
                            penalty="l2",
                            solver="liblinear",
                            max_iter=2000,
                            class_weight=None,
                            random_state=CV_RANDOM_STATE,
                        ),
                    ),
                ]
            )

            pipeline.fit(
                X[train_index],
                y[train_index],
            )

            probability = pipeline.predict_proba(X[val_index])[:, 1]

            probability_sum[val_index] += probability

            probability_count[val_index] += 1

        if (probability_count == 0).any():

            raise RuntimeError(f"{label}: incomplete OOF predictions.")

        oof_probability = probability_sum / probability_count

        prevalence = float(y.mean())

        prior_probability = np.full(
            len(y),
            prevalence,
            dtype=np.float64,
        )

        auc = float(
            roc_auc_score(
                y,
                oof_probability,
            )
        )

        ap = float(
            average_precision_score(
                y,
                oof_probability,
            )
        )

        brier = float(
            brier_score_loss(
                y,
                oof_probability,
            )
        )

        ll = float(
            log_loss(
                y,
                oof_probability,
                labels=[
                    0,
                    1,
                ],
            )
        )

        prior_brier = float(
            brier_score_loss(
                y,
                prior_probability,
            )
        )

        evidence_mask = label_df["EvidenceAvailable"].astype(bool).values

        evidence_auc = np.nan
        evidence_ap = np.nan
        evidence_brier = np.nan
        evidence_prior_brier = np.nan
        evidence_brier_improvement = np.nan

        evidence_n = int(evidence_mask.sum())

        if evidence_n > 0:

            evidence_y = y[evidence_mask]

            evidence_probability = oof_probability[evidence_mask]

            evidence_brier = float(
                brier_score_loss(
                    evidence_y,
                    evidence_probability,
                )
            )

            evidence_prevalence = float(evidence_y.mean())

            evidence_prior_probability = np.full(
                len(evidence_y),
                evidence_prevalence,
                dtype=np.float64,
            )

            evidence_prior_brier = float(
                brier_score_loss(
                    evidence_y,
                    evidence_prior_probability,
                )
            )

            evidence_brier_improvement = float(evidence_prior_brier - evidence_brier)

            if evidence_n > 1 and len(np.unique(evidence_y)) == 2:

                evidence_auc = float(
                    roc_auc_score(
                        evidence_y,
                        evidence_probability,
                    )
                )

                evidence_ap = float(
                    average_precision_score(
                        evidence_y,
                        evidence_probability,
                    )
                )

        metric_rows.append(
            {
                "Label": label,
                "GoldPositive": int(y.sum()),
                "GoldNegative": int(len(y) - y.sum()),
                "OOF_AUROC": auc,
                "OOF_AP": ap,
                "OOF_Brier": brier,
                "Prior_Brier": prior_brier,
                "BrierImprovementVsPrior": (prior_brier - brier),
                "OOF_LogLoss": ll,
                "EvidenceAvailableN": evidence_n,
                "EvidenceCoverage": float(evidence_n / len(y)),
                "EvidenceOnly_AUROC": evidence_auc,
                "EvidenceOnly_AP": evidence_ap,
                "EvidenceOnly_Brier": evidence_brier,
                "EvidenceOnly_Prior_Brier": evidence_prior_brier,
                "EvidenceOnly_BrierImprovementVsPrior": evidence_brier_improvement,
            }
        )

        for row_index, (
            probability,
            fold_count,
        ) in enumerate(
            zip(
                oof_probability,
                probability_count,
            )
        ):

            source_row = label_df.iloc[row_index]

            oof_rows.append(
                {
                    "PairID": source_row["PairID"],
                    UID_COLUMN: source_row[UID_COLUMN],
                    "Label": label,
                    "Gold": int(source_row["Gold"]),
                    "EvidenceAvailable": bool(source_row["EvidenceAvailable"]),
                    "FusedAssertion": source_row["FusedAssertion"],
                    "OOF_ChallengeProbability": float(probability),
                    "OOF_RepeatedPredictions": int(fold_count),
                }
            )

        # Fit final mapper on all 58 after OOF evaluation.
        final_pipeline = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "logreg",
                    LogisticRegression(
                        C=CALIBRATION_C,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=2000,
                        class_weight=None,
                        random_state=CV_RANDOM_STATE,
                    ),
                ),
            ]
        )

        final_pipeline.fit(
            X,
            y,
        )

        calibrators[label] = final_pipeline

        coefficients = final_pipeline.named_steps["logreg"].coef_[0]

        for feature, coefficient in zip(
            STAGE_B_FEATURES,
            coefficients,
        ):

            coefficient_rows.append(
                {
                    "Label": label,
                    "Feature": feature,
                    "StandardizedCoefficient": float(coefficient),
                }
            )

        coefficient_rows.append(
            {
                "Label": label,
                "Feature": "Intercept",
                "StandardizedCoefficient": float(
                    final_pipeline.named_steps["logreg"].intercept_[0]
                ),
            }
        )

    oof_df = pd.DataFrame(oof_rows)

    metrics_df = pd.DataFrame(metric_rows)

    def _pseudo_gate_reason(
        row: pd.Series,
    ) -> str:

        reasons = []

        if int(row["EvidenceAvailableN"]) < PSEUDO_GATE_MIN_EVIDENCE_N:

            reasons.append("insufficient_evidence_n")

        evidence_auc_value = row["EvidenceOnly_AUROC"]

        if (
            pd.isna(evidence_auc_value)
            or float(evidence_auc_value) < PSEUDO_GATE_MIN_EVIDENCE_AUROC
        ):

            reasons.append("evidence_auc_below_gate")

        brier_improvement_value = row["EvidenceOnly_BrierImprovementVsPrior"]

        if (
            pd.isna(brier_improvement_value)
            or float(brier_improvement_value) <= PSEUDO_GATE_MIN_BRIER_IMPROVEMENT
        ):

            reasons.append("evidence_brier_not_better_than_prior")

        return "pass" if not reasons else "|".join(reasons)

    metrics_df["PseudoLabelGateReason"] = metrics_df.apply(
        _pseudo_gate_reason,
        axis=1,
    )

    metrics_df["PseudoLabelGatePass"] = metrics_df["PseudoLabelGateReason"] == "pass"

    coefficient_df = pd.DataFrame(coefficient_rows)

    dump(
        {
            "calibrators": calibrators,
            "features": STAGE_B_FEATURES,
            "calibration_C": CALIBRATION_C,
            "model_id": MODEL_ID,
            "semantic_thresholds": {
                "strong": SEMANTIC_STRONG_THRESHOLD,
                "related": SEMANTIC_RELATED_THRESHOLD,
                "weak": SEMANTIC_WEAK_THRESHOLD,
                "margin": SEMANTIC_MARGIN,
            },
            "pseudo_label_gate": {
                "min_evidence_n": PSEUDO_GATE_MIN_EVIDENCE_N,
                "min_evidence_auc": PSEUDO_GATE_MIN_EVIDENCE_AUROC,
                "min_evidence_brier_improvement": PSEUDO_GATE_MIN_BRIER_IMPROVEMENT,
                "eligible_labels": metrics_df.loc[
                    metrics_df["PseudoLabelGatePass"],
                    "Label",
                ]
                .astype(str)
                .tolist(),
            },
        },
        MODEL_ROOT / "w2_stage_b_calibrators.joblib",
    )

    # Aggregate macro metrics across labels.
    macro_row = {
        "Label": "MACRO_MEAN",
        "GoldPositive": int(metrics_df["GoldPositive"].sum()),
        "GoldNegative": int(metrics_df["GoldNegative"].sum()),
        "OOF_AUROC": float(metrics_df["OOF_AUROC"].mean()),
        "OOF_AP": float(metrics_df["OOF_AP"].mean()),
        "OOF_Brier": float(metrics_df["OOF_Brier"].mean()),
        "Prior_Brier": float(metrics_df["Prior_Brier"].mean()),
        "BrierImprovementVsPrior": float(metrics_df["BrierImprovementVsPrior"].mean()),
        "OOF_LogLoss": float(metrics_df["OOF_LogLoss"].mean()),
        "EvidenceAvailableN": int(metrics_df["EvidenceAvailableN"].sum()),
        "EvidenceCoverage": float(metrics_df["EvidenceCoverage"].mean()),
        "EvidenceOnly_AUROC": float(metrics_df["EvidenceOnly_AUROC"].mean(skipna=True)),
        "EvidenceOnly_AP": float(metrics_df["EvidenceOnly_AP"].mean(skipna=True)),
        "EvidenceOnly_Brier": float(metrics_df["EvidenceOnly_Brier"].mean(skipna=True)),
        "EvidenceOnly_Prior_Brier": float(
            metrics_df["EvidenceOnly_Prior_Brier"].mean(skipna=True)
        ),
        "EvidenceOnly_BrierImprovementVsPrior": float(
            metrics_df["EvidenceOnly_BrierImprovementVsPrior"].mean(skipna=True)
        ),
        "PseudoLabelGateReason": "macro_not_applicable",
        "PseudoLabelGatePass": bool(metrics_df["PseudoLabelGatePass"].all()),
    }

    metrics_with_macro = pd.concat(
        [
            metrics_df,
            pd.DataFrame([macro_row]),
        ],
        ignore_index=True,
    )

    return (
        oof_df,
        metrics_with_macro,
        calibrators,
        coefficient_df,
    )


def run_gold_validation(
    train_df: pd.DataFrame,
    scorer: NLIScorer,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:

    gold_studies = (
        train_df[train_df["W2_LabelStatus"] == "gold"].copy().reset_index(drop=True)
    )

    gold_pairs = build_pair_table(
        gold_studies,
        include_gold=True,
    )

    gold_structured = extract_structured_pairs(
        gold_pairs,
        scorer,
        cohort_name="gold_58x12",
    )

    gold_structured.to_csv(
        RESULT_ROOT / "04_gold_structured_report_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        oof_df,
        metrics_df,
        calibrators,
        coefficient_df,
    ) = fit_stage_b_oof(gold_structured)

    oof_df.to_csv(
        RESULT_ROOT / "05_stage_b_gold_oof_predictions.csv",
        index=False,
    )

    metrics_df.to_csv(
        RESULT_ROOT / "06_stage_b_gold_oof_metrics.csv",
        index=False,
    )

    coefficient_df.to_csv(
        RESULT_ROOT / "07_stage_b_full_fit_coefficients.csv",
        index=False,
    )

    print()
    print("Stage B gold OOF metrics:")

    print(metrics_df.to_string(index=False))

    return (
        gold_structured,
        metrics_df,
        calibrators,
    )


# ============================================================
# 16. FULL-CORPUS SOFT LABEL CANDIDATES
# ============================================================


def apply_stage_b_calibrators(
    structured_df: pd.DataFrame,
    calibrators: Dict[str, Any],
    stage_b_metrics: pd.DataFrame,
) -> pd.DataFrame:

    output = structured_df.copy()

    output["ChallengeProbabilityRaw"] = np.nan

    for label in LABEL_COLUMNS:

        mask = output["Label"] == label

        label_df = output[mask]

        X = make_stage_b_matrix(label_df)

        probability = calibrators[label].predict_proba(X)[:, 1]

        output.loc[
            mask,
            "ChallengeProbabilityRaw",
        ] = probability

    eligible_label_map = (
        stage_b_metrics[stage_b_metrics["Label"] != "MACRO_MEAN"]
        .set_index("Label")["PseudoLabelGatePass"]
        .astype(bool)
        .to_dict()
    )

    gate_reason_map = (
        stage_b_metrics[stage_b_metrics["Label"] != "MACRO_MEAN"]
        .set_index("Label")["PseudoLabelGateReason"]
        .astype(str)
        .to_dict()
    )

    output["LabelCalibrationEligible"] = (
        output["Label"].map(eligible_label_map).fillna(False).astype(bool)
    )

    output["LabelCalibrationGateReason"] = (
        output["Label"].map(gate_reason_map).fillna("missing_gate")
    )

    # Two hard methodological gates:
    # 1) no report evidence -> UNKNOWN, not a soft negative;
    # 2) label calibration must pass evidence-subset validation.
    output["SoftLabelAvailable"] = output["EvidenceAvailable"].astype(bool) & output[
        "LabelCalibrationEligible"
    ].astype(bool)

    output["ChallengeSoftLabel"] = np.where(
        output["SoftLabelAvailable"],
        output["ChallengeProbabilityRaw"],
        np.nan,
    )

    # CandidateSelectionScore is ONLY a triage/ranking score.
    # It is deliberately not named a training weight.
    output["CandidateSelectionScore"] = np.where(
        output["SoftLabelAvailable"],
        (
            output["FusedAssertionConfidence"]
            * (2.0 * np.abs(output["ChallengeProbabilityRaw"] - 0.5))
        ),
        0.0,
    )

    return output


def run_full_corpus(
    train_df: pd.DataFrame,
    scorer: NLIScorer,
    calibrators: Dict[str, Any],
    stage_b_metrics: pd.DataFrame,
) -> None:

    all_pairs = build_pair_table(
        train_df,
        include_gold=False,
    )

    full_structured = extract_structured_pairs(
        all_pairs,
        scorer,
        cohort_name="full_4407x12",
    )

    full_scored = apply_stage_b_calibrators(
        full_structured,
        calibrators,
        stage_b_metrics,
    )

    full_scored.to_csv(
        RESULT_ROOT / "08_full_structured_report_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    status_map = train_df[
        [
            UID_COLUMN,
            "W2_LabelStatus",
        ]
    ].copy()

    full_scored = full_scored.merge(
        status_map,
        on=UID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    unlabeled = full_scored[full_scored["W2_LabelStatus"] == "unlabeled"].copy()

    unlabeled.to_csv(
        RESULT_ROOT / "09_unlabeled_soft_labels_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    probability_wide = (
        unlabeled.pivot(
            index=UID_COLUMN,
            columns="Label",
            values="ChallengeSoftLabel",
        )
        .reindex(columns=LABEL_COLUMNS)
        .reset_index()
    )

    probability_wide.to_csv(
        RESULT_ROOT / "10_unlabeled_soft_probabilities_wide.csv",
        index=False,
    )

    availability_wide = (
        unlabeled.pivot(
            index=UID_COLUMN,
            columns="Label",
            values="SoftLabelAvailable",
        )
        .reindex(columns=LABEL_COLUMNS)
        .reset_index()
    )

    availability_wide.to_csv(
        RESULT_ROOT / "11_unlabeled_soft_label_availability_wide.csv",
        index=False,
    )

    selection_wide = (
        unlabeled.pivot(
            index=UID_COLUMN,
            columns="Label",
            values="CandidateSelectionScore",
        )
        .reindex(columns=LABEL_COLUMNS)
        .reset_index()
    )

    selection_wide.to_csv(
        RESULT_ROOT / "12_unlabeled_candidate_selection_scores_wide.csv",
        index=False,
    )

    coverage_rows = []

    for label in LABEL_COLUMNS:

        group = unlabeled[unlabeled["Label"] == label]

        available = group["SoftLabelAvailable"].astype(bool)

        coverage_rows.append(
            {
                "Label": label,
                "UnlabeledStudies": int(len(group)),
                "SoftLabelAvailableN": int(available.sum()),
                "SoftLabelCoverage": float(available.mean()),
                "MeanProbabilityAvailable": (
                    float(
                        group.loc[
                            available,
                            "ChallengeSoftLabel",
                        ].mean()
                    )
                    if available.any()
                    else np.nan
                ),
                "HighSelectionScoreN_ge_0_5": int(
                    (group["CandidateSelectionScore"] >= 0.5).sum()
                ),
                "RuleExplicitN": int(
                    group["RuleAssertion"]
                    .isin(
                        [
                            "positive",
                            "negative",
                        ]
                    )
                    .sum()
                ),
                "SemanticRecoveredN": int(
                    group["FusionSource"]
                    .fillna("")
                    .astype(str)
                    .str.startswith("semantic_recovery")
                    .sum()
                ),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)

    coverage_df.to_csv(
        RESULT_ROOT / "13_unlabeled_label_coverage_summary.csv",
        index=False,
    )

    print()
    print("Full-corpus W2 candidate outputs written.")

    print(coverage_df.to_string(index=False))


# ============================================================
# 17. REPORT + CONFIG
# ============================================================


def write_w2_report(
    mode: str,
    benchmark_metrics: Optional[pd.DataFrame],
    stage_b_metrics: Optional[pd.DataFrame],
) -> None:

    lines = [
        "# RSNA W2 — Multilingual Structured Report Labeler",
        "",
        f"- Mode: `{mode}`",
        f"- Semantic model: `{MODEL_PATH if MODEL_PATH else MODEL_ID}`",
        (
            "- Stage A: W1.2 deterministic concept-local rules "
            "+ multilingual 3-way NLI"
        ),
        (
            "- Stage B: per-label strongly regularized logistic "
            "challenge-ontology mapping"
        ),
        (
            "- Semantic recovery is target-grounded before NLI "
            "may change a W1.2 `not_mentioned` state"
        ),
        (
            "- `not_mentioned` and `mentioned_neutral` are kept "
            "unavailable for pseudo-supervision in FULL mode"
        ),
        ("- W1.3 stress-test PPV/NPV values are not used as " "direct weights"),
        (
            "- FULL-mode soft labels are additionally gated by "
            "evidence-subset Stage-B validation"
        ),
        "",
    ]

    if benchmark_metrics is not None:

        lines.extend(
            [
                "## W1.3 Stage A benchmark",
                "",
                "```text",
                benchmark_metrics.to_string(index=False),
                "```",
                "",
            ]
        )

    if stage_b_metrics is not None:

        lines.extend(
            [
                "## Stage B gold cross-fit",
                "",
                "```text",
                stage_b_metrics.to_string(index=False),
                "```",
                "",
                (
                    "These are cross-fitted results on only 58 "
                    "gold studies. The W1/W1.1/W1.2 development "
                    "history means they are not an independent "
                    "external validation set."
                ),
                "",
            ]
        )

    if mode == "full":

        lines.extend(
            [
                "## Candidate soft labels",
                "",
                (
                    "FULL mode writes candidate report-derived "
                    "soft labels for the 4,349 unlabeled studies. "
                    "They are not automatically consumed by the "
                    "MRI model."
                ),
                "",
                (
                    "`CandidateSelectionScore` is a ranking/triage "
                    "score, not a validated loss weight."
                ),
                "",
            ]
        )

    with open(
        RESULT_ROOT / "W2_REPORT.md",
        "w",
        encoding="utf-8",
    ) as f:

        f.write("\n".join(lines))


def save_w2_config(
    mode: str,
) -> None:

    payload = {
        "phase": "W2 multilingual structured report labeler",
        "mode": mode,
        "train_csv": str(TRAIN_CSV),
        "semantic_model": MODEL_PATH if MODEL_PATH else MODEL_ID,
        "local_files_only": LOCAL_FILES_ONLY,
        "use_multi_gpu": USE_MULTI_GPU,
        "nli_batch_size": NLI_BATCH_SIZE,
        "nli_max_length": NLI_MAX_LENGTH,
        "max_units_per_report": MAX_UNITS_PER_REPORT,
        "semantic_thresholds": {
            "strong": SEMANTIC_STRONG_THRESHOLD,
            "related": SEMANTIC_RELATED_THRESHOLD,
            "weak": SEMANTIC_WEAK_THRESHOLD,
            "margin": SEMANTIC_MARGIN,
            "conflict": SEMANTIC_CONFLICT_THRESHOLD,
        },
        "stage_b": {
            "features": STAGE_B_FEATURES,
            "logistic_C": CALIBRATION_C,
            "cv_splits": CV_SPLITS,
            "cv_repeats": CV_REPEATS,
            "random_state": CV_RANDOM_STATE,
            "pseudo_label_gate": {
                "min_evidence_n": PSEUDO_GATE_MIN_EVIDENCE_N,
                "min_evidence_auc": PSEUDO_GATE_MIN_EVIDENCE_AUROC,
                "min_evidence_brier_improvement": PSEUDO_GATE_MIN_BRIER_IMPROVEMENT,
            },
        },
        "hard_constraints": {
            "not_mentioned_is_negative": False,
            "not_mentioned_soft_label_available": False,
            "mentioned_neutral_soft_label_available": False,
            "w13_ppv_used_as_direct_weight": False,
            "image_model_trained": False,
        },
    }

    save_json(
        RESULT_ROOT / "w2_config.json",
        payload,
    )


# ============================================================
# 18. ENTRY POINT
# ============================================================


def resolve_mode() -> str:

    env_mode = (
        os.environ.get(
            "W2_MODE",
            "",
        )
        .strip()
        .casefold()
    )

    default_mode = (
        env_mode
        if env_mode
        in {
            "benchmark",
            "validate",
            "full",
        }
        else "validate"
    )

    parser = argparse.ArgumentParser(
        description=("RSNA W2 multilingual structured report labeler")
    )

    parser.add_argument(
        "--mode",
        choices=[
            "benchmark",
            "validate",
            "full",
        ],
        default=default_mode,
    )

    args, _ = parser.parse_known_args()

    return args.mode


def main() -> None:

    mode = resolve_mode()

    print("=" * 80)
    print("RSNA W2 - MULTILINGUAL STRUCTURED REPORT LABELER")
    print("=" * 80)
    print(
        "Mode:",
        mode,
    )

    model_source = resolve_model_source()

    scorer = NLIScorer(model_source)

    scorer.sanity_check()

    benchmark_metrics = None
    stage_b_metrics = None
    calibrators = None

    if mode in [
        "benchmark",
        "validate",
    ]:

        (
            benchmark_case_results,
            benchmark_metrics,
        ) = benchmark_stage_a(scorer)

    if mode in [
        "validate",
        "full",
    ]:

        train_df = load_train_data()

        (
            gold_structured_features,
            stage_b_metrics,
            calibrators,
        ) = run_gold_validation(
            train_df,
            scorer,
        )

    if mode == "full":

        if calibrators is None:

            raise RuntimeError("Stage B calibrators were not fitted.")

        run_full_corpus(
            train_df,
            scorer,
            calibrators,
            stage_b_metrics,
        )

    write_w2_report(
        mode,
        benchmark_metrics,
        stage_b_metrics,
    )

    save_w2_config(mode)

    print()
    print("=" * 80)
    print("W2 COMPLETE")
    print("=" * 80)
    print()
    print(f"Results directory:\n{RESULT_ROOT}")

    print()
    print("W2 does NOT train the MRI image model.")

    if mode != "full":

        print("No 4,349-study pseudo-label pool was " "generated in this mode.")


if __name__ == "__main__":
    main()
