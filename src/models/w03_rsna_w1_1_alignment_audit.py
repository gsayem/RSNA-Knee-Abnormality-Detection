# ============================================================
# RSNA KNEE ABNORMALITY DETECTION - W1.1
# Report <-> Gold Label Alignment Audit
# ============================================================
#
# Purpose
# -------
# Audit how the 58 gold radiology reports align with the 12
# expert binary labels before building W2 pseudo-labeling.
#
# This script does NOT train an NLP model and does NOT create
# pseudo-labels for the 4,349 unlabeled studies.
#
# It creates one audit row for every:
#
#       58 gold studies x 12 labels = 696 pairs
#
# For each pair W1.1 attempts to extract conservative textual
# evidence and classifies the report-side assertion as:
#
#   positive
#   negative
#   uncertain
#   mixed
#   related_abnormality
#   mentioned_neutral
#   not_mentioned
#
# The automatic lexicon pass is intentionally conservative.
# Its purpose is to:
#
#   1. quantify report/gold disagreements,
#   2. reveal ontology/severity issues,
#   3. prioritize cases for human inspection,
#   4. measure which labels/languages are safely extractable,
#   5. prepare the design of W2.
#
# IMPORTANT
# ---------
# "not_mentioned" is NOT treated as a negative report label.
#
# Input preference:
#   A. W1 output: 12_gold_reports_with_labels.csv
#   B. otherwise derive the 58 gold rows from train.csv
#
# Default Kaggle locations:
#   Competition:
#     /kaggle/input/competitions/rsna-knee-abnormality-detection
#
#   W1:
#     /kaggle/working/rsna_w1/results
#
#   Output:
#     /kaggle/working/rsna_w1_1/results
#
# Environment overrides are supported:
#   RSNA_DATA_ROOT
#   RSNA_W1_RESULT_ROOT
#   RSNA_W1_1_WORK_ROOT
#
# No internet is required.
# ============================================================

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIG
# ============================================================

DATA_ROOT = Path(
    os.environ.get(
        "RSNA_DATA_ROOT",
        ("/kaggle/input/competitions/" "rsna-knee-abnormality-detection"),
    )
)

TRAIN_CSV = DATA_ROOT / "train.csv"

W1_RESULT_ROOT = Path(
    os.environ.get(
        "RSNA_W1_RESULT_ROOT",
        "/kaggle/working/rsna_w1/results",
    )
)

WORK_ROOT = Path(
    os.environ.get(
        "RSNA_W1_1_WORK_ROOT",
        "/kaggle/working/rsna_w1_1",
    )
)

RESULT_ROOT = WORK_ROOT / "results"

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_GOLD_STUDIES = 58
EXPECTED_ALIGNMENT_ROWS = 58 * 12


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


# Keep automatic evidence extraction conservative.
CONTEXT_WINDOW_CHARS = 100

# Manual review samples per label for "apparently concordant"
# cases. All discordant/uncertain/uncaptured cases are included.
CONCORDANT_REVIEW_SAMPLE_PER_LABEL = 3

RANDOM_SEED = 42


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

    text = normalize_whitespace(text)

    if not text:

        return []

    # "|" occurs in some exported report representations.
    text = text.replace(
        " | ",
        "\n",
    )

    parts = re.split(
        (r"(?<=[.!?。！？;])\s+" r"|\n+"),
        text,
    )

    return [part.strip() for part in parts if part.strip()]


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
            r"\befusion articular\b",
            r"\bhemartros\w*\b",
        ],
        "tr": [
            r"\beklem .*sivi\w*\b",
            r"\beklem .*sıvı\w*\b",
            r"\bsivi artisi\b",
            r"\bsıvı artışı\b",
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


def generic_context_cues(
    folded_unit: str,
    language: str,
    concept_match: Optional[re.Match],
) -> Dict[str, List[str]]:
    """
    Direction-aware cue handling.

    General negation such as "no", "without", "sin", "geen",
    etc. is trusted primarily BEFORE the concept.

    Only state-like expressions such as "normal", "intact",
    "none" are allowed immediately AFTER the concept.

    This prevents unrelated later clauses from accidentally
    negating the target.
    """

    if concept_match is None:

        left_context = folded_unit
        right_context = ""
        bidirectional_context = folded_unit

    else:

        left_context = folded_unit[
            max(
                0,
                concept_match.start() - 80,
            ) : concept_match.start()
        ]

        right_context = folded_unit[
            concept_match.end() : min(
                len(folded_unit),
                concept_match.end() + 55,
            )
        ]

        bidirectional_context = folded_unit[
            max(
                0,
                concept_match.start() - 80,
            ) : min(
                len(folded_unit),
                concept_match.end() + 80,
            )
        ]

    negation = all_pattern_strings(
        left_context,
        patterns_for_language(
            NEGATION_PATTERNS,
            language,
        ),
    )

    # Allow explicit state negation immediately after concept.
    post_state_hits = all_pattern_strings(
        right_context,
        patterns_for_language(
            POST_STATE_NEGATION_PATTERNS,
            language,
        ),
    )

    negation.extend(post_state_hits)

    negation = sorted(set(negation))

    uncertainty = all_pattern_strings(
        bidirectional_context,
        patterns_for_language(
            UNCERTAINTY_PATTERNS,
            language,
        ),
    )

    history = all_pattern_strings(
        bidirectional_context,
        patterns_for_language(
            HISTORY_PATTERNS,
            language,
        ),
    )

    return {
        "negation": negation,
        "uncertainty": uncertainty,
        "history": history,
    }


def classify_structure_unit(
    label: str,
    unit: str,
    language: str,
    section: str,
) -> Optional[Dict[str, Any]]:
    """
    ACL, MCL and side-specific meniscus targets.
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

    cues = generic_context_cues(
        folded,
        language,
        concept_match,
    )

    injury_patterns = patterns_for_language(
        TEAR_INJURY_PATTERNS,
        language,
    )

    structure_context = context_around_match(
        folded,
        concept_match,
        radius=90,
    )

    injury_hits = all_pattern_strings(
        structure_context,
        injury_patterns,
    )

    degeneration_hits = all_pattern_strings(
        structure_context,
        patterns_for_language(
            DEGENERATION_PATTERNS,
            language,
        ),
    )

    severity = extract_severity(folded)

    # Scope note:
    # For meniscus targets, degeneration alone is not treated
    # as a positive tear label.
    if cues["uncertainty"] and injury_hits:

        assertion = "uncertain"

    elif cues["negation"]:

        assertion = "negative"

    elif injury_hits:

        assertion = "positive"

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

    return {
        "Label": label,
        "Section": section,
        "Assertion": assertion,
        "EvidenceText": unit,
        "MatchedConceptPattern": concept_pattern,
        "PositiveCuePatterns": "|".join(injury_hits),
        "NegativeCuePatterns": "|".join(cues["negation"]),
        "UncertaintyCuePatterns": "|".join(cues["uncertainty"]),
        "HistoryCuePatterns": "|".join(cues["history"]),
        "Severity": "|".join(severity),
        "SectionWeight": evidence_section_weight(section),
    }


def classify_oa_unit(
    label: str,
    unit: str,
    language: str,
    section: str,
) -> Optional[Dict[str, Any]]:

    folded = fold_for_match(unit)

    side = {
        "Medial OA": "medial",
        "Lateral OA": "lateral",
        "PF OA": "pf",
    }[label]

    compartment_pattern, compartment_match = first_pattern_match(
        folded,
        patterns_for_language(
            COMPARTMENT_PATTERNS[side],
            language,
        ),
    )

    oa_pattern, oa_match = first_pattern_match(
        folded,
        patterns_for_language(
            OA_PATTERNS,
            language,
        ),
    )

    # Require compartment + OA evidence in the same unit.
    if compartment_match is None or oa_match is None:

        return None

    cues = generic_context_cues(
        folded,
        language,
        oa_match,
    )

    severity = extract_severity(folded)

    if cues["uncertainty"]:

        assertion = "uncertain"

    elif cues["negation"]:

        assertion = "negative"

    else:

        assertion = "positive"

    return {
        "Label": label,
        "Section": section,
        "Assertion": assertion,
        "EvidenceText": unit,
        "MatchedConceptPattern": (f"{compartment_pattern}" " && " f"{oa_pattern}"),
        "PositiveCuePatterns": oa_pattern,
        "NegativeCuePatterns": "|".join(cues["negation"]),
        "UncertaintyCuePatterns": "|".join(cues["uncertainty"]),
        "HistoryCuePatterns": "|".join(cues["history"]),
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

    cues = generic_context_cues(
        folded,
        language,
        active_match,
    )

    severity = extract_severity(folded)

    if related_match is not None:

        if cues["negation"]:

            assertion = "negative"

        elif cues["uncertainty"]:

            assertion = "uncertain"

        else:

            assertion = "related_abnormality"

    else:

        if cues["uncertainty"]:

            assertion = "uncertain"

        elif cues["negation"]:

            assertion = "negative"

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
        "NegativeCuePatterns": "|".join(cues["negation"]),
        "UncertaintyCuePatterns": "|".join(cues["uncertainty"]),
        "HistoryCuePatterns": "|".join(cues["history"]),
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
# 10. GOLD-REPORT ALIGNMENT CATEGORIES
# ============================================================


def report_binary_value(
    assertion: str,
) -> Optional[int]:

    if assertion == "positive":

        return 1

    if assertion == "negative":

        return 0

    return None


def alignment_category(
    gold: int,
    assertion: str,
) -> str:

    if assertion == "positive":

        if gold == 1:

            return "concordant_positive"

        return "discordant_report_positive_" "gold_negative"

    if assertion == "negative":

        if gold == 0:

            return "concordant_negative_explicit"

        return "discordant_report_negative_" "gold_positive"

    if assertion == "mixed":

        return "indeterminate_mixed"

    if assertion == "uncertain":

        return "indeterminate_uncertain"

    if assertion == "related_abnormality":

        if gold == 1:

            return "gold_positive_related_only"

        return "gold_negative_related_abnormality"

    if assertion == "mentioned_neutral":

        if gold == 1:

            return "gold_positive_mentioned_" "but_not_classified"

        return "gold_negative_mentioned_neutral"

    # not_mentioned
    if gold == 1:

        return "gold_positive_not_captured"

    return "gold_negative_not_mentioned"


def manual_review_priority(
    gold: int,
    assertion: str,
    category: str,
) -> Tuple[int, str]:

    if category.startswith("discordant_"):

        return (
            1,
            "explicit_report_gold_disagreement",
        )

    if gold == 1 and assertion in [
        "not_mentioned",
        "mentioned_neutral",
        "related_abnormality",
        "uncertain",
        "mixed",
    ]:

        return (
            2,
            "gold_positive_not_cleanly_extracted",
        )

    if assertion in [
        "mixed",
        "uncertain",
    ]:

        return (
            3,
            "ambiguous_report_assertion",
        )

    if assertion == "related_abnormality":

        return (
            4,
            "related_but_not_direct_target_finding",
        )

    if category.startswith("concordant_"):

        return (
            5,
            "concordant_sampling_candidate",
        )

    return (
        6,
        "low_priority_unmentioned_or_neutral",
    )


# ============================================================
# 11. WILSON INTERVAL
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

        return (
            None,
            None,
        )

    p = successes / total

    denominator = 1.0 + z * z / total

    center = (p + z * z / (2.0 * total)) / denominator

    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )

    return (
        max(
            0.0,
            center - margin,
        ),
        min(
            1.0,
            center + margin,
        ),
    )


# ============================================================
# 12. LOAD GOLD DATA
# ============================================================

print("=" * 80)

print("RSNA W1.1 - REPORT <-> GOLD LABEL ALIGNMENT AUDIT")

print("=" * 80)


w1_gold_path = W1_RESULT_ROOT / "12_gold_reports_with_labels.csv"


source_name = None


if w1_gold_path.exists():

    gold_df = pd.read_csv(w1_gold_path)

    source_name = "W1 12_gold_reports_with_labels.csv"


else:

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            "Could not find either:\n" f"  {w1_gold_path}\n" "or:\n" f"  {TRAIN_CSV}"
        )

    train_df = pd.read_csv(TRAIN_CSV)

    missing_label_count = train_df[LABEL_COLUMNS].isna().sum(axis=1)

    gold_mask = missing_label_count == 0

    gold_df = train_df[gold_mask].copy()

    source_name = "train.csv derived gold cohort"

    # If W1 inventory exists, reuse its language assignments.
    inventory_path = W1_RESULT_ROOT / "01_report_inventory.csv"

    if inventory_path.exists():

        inventory_df = pd.read_csv(inventory_path)

        keep_columns = [
            UID_COLUMN,
            "Language",
            "DominantScript",
        ]

        available = [
            column for column in keep_columns if column in inventory_df.columns
        ]

        gold_df = gold_df.merge(
            inventory_df[available],
            on=UID_COLUMN,
            how="left",
        )


# ------------------------------------------------------------
# Hard validation
# ------------------------------------------------------------

if len(gold_df) != EXPECTED_GOLD_STUDIES:

    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_GOLD_STUDIES} gold studies, "
        f"found {len(gold_df)}."
    )


missing_required = [
    column
    for column in (
        [
            UID_COLUMN,
        ]
        + LABEL_COLUMNS
    )
    if column not in gold_df.columns
]


if missing_required:

    raise RuntimeError("Gold data is missing required columns: " f"{missing_required}")


report_source_column = (
    "NormalizedReport" if "NormalizedReport" in gold_df.columns else REPORT_COLUMN
)


if report_source_column not in gold_df.columns:

    raise RuntimeError("No report text column found.")


missing_reports = gold_df[report_source_column].isna() | gold_df[
    report_source_column
].astype(str).str.strip().eq("")


if missing_reports.any():

    raise RuntimeError(
        "Every gold study must have a report. " f"Missing: {int(missing_reports.sum())}"
    )


print(f"Gold source: {source_name}")

print(f"Gold studies: {len(gold_df)}")

print(f"Expected report-label pairs: " f"{EXPECTED_ALIGNMENT_ROWS}")


# ============================================================
# 13. NORMALIZE REPORTS + LANGUAGE
# ============================================================

normalization_rows = []


normalized_reports = []
repaired_flags = []
canonical_languages = []


for _, row in gold_df.iterrows():

    normalized, repaired = normalize_report(row[report_source_column])

    normalized_reports.append(normalized)

    repaired_flags.append(repaired)

    if "Language" in gold_df.columns and pd.notna(
        row.get(
            "Language",
            np.nan,
        )
    ):

        language = canonicalize_language(row["Language"])

    else:

        language = detect_language_fallback(normalized)

    canonical_languages.append(language)

    if repaired:

        normalization_rows.append(
            {
                UID_COLUMN: row[UID_COLUMN],
                "Language": language,
                "OriginalPreview": str(row[report_source_column])[:1200],
                "RepairedPreview": normalized[:1200],
            }
        )


gold_df["W11_NormalizedReport"] = normalized_reports

gold_df["W11_MojibakeRepaired"] = repaired_flags

gold_df["W11_Language"] = canonical_languages


normalization_df = pd.DataFrame(normalization_rows)


normalization_df.to_csv(
    RESULT_ROOT / "01_text_normalization_repairs.csv",
    index=False,
)


language_counts = gold_df["W11_Language"].value_counts(dropna=False)


print("\nGold language distribution:")

print(language_counts.to_string())


# ============================================================
# 14. RUN 696-PAIR ALIGNMENT AUDIT
# ============================================================

alignment_rows = []
evidence_rows = []


for _, row in gold_df.iterrows():

    uid = str(row[UID_COLUMN])

    report = row["W11_NormalizedReport"]

    language = row["W11_Language"]

    for label in LABEL_COLUMNS:

        gold_value = int(row[label])

        evidence = classify_report_label_evidence(
            report,
            language,
            label,
        )

        for evidence_index, record in enumerate(
            evidence,
            start=1,
        ):

            evidence_rows.append(
                {
                    UID_COLUMN: uid,
                    "Language": language,
                    "Label": label,
                    "Gold": gold_value,
                    "EvidenceIndex": evidence_index,
                    **record,
                }
            )

        assertion, representative = aggregate_assertion(evidence)

        binary_value = report_binary_value(assertion)

        category = alignment_category(
            gold_value,
            assertion,
        )

        priority, priority_reason = manual_review_priority(
            gold_value,
            assertion,
            category,
        )

        if representative is None:

            representative_text = ""
            representative_section = ""
            representative_severity = ""
            negative_cues = ""
            uncertainty_cues = ""
            history_cues = ""

        else:

            representative_text = representative["EvidenceText"]

            representative_section = representative["Section"]

            representative_severity = representative["Severity"]

            negative_cues = representative["NegativeCuePatterns"]

            uncertainty_cues = representative["UncertaintyCuePatterns"]

            history_cues = representative["HistoryCuePatterns"]

        all_severity = sorted(
            set(
                severity
                for evidence_row in evidence
                for severity in str(evidence_row["Severity"]).split("|")
                if severity
            )
        )

        alignment_rows.append(
            {
                UID_COLUMN: uid,
                "Language": language,
                "Label": label,
                "Gold": gold_value,
                "ReportAssertion": assertion,
                "ReportBinaryValue": (
                    binary_value if binary_value is not None else np.nan
                ),
                "Decidable": (binary_value is not None),
                "AlignmentCategory": category,
                "EvidenceCount": len(evidence),
                "RepresentativeSection": representative_section,
                "RepresentativeEvidence": representative_text,
                "Severity": "|".join(all_severity),
                "RepresentativeNegativeCues": negative_cues,
                "RepresentativeUncertaintyCues": uncertainty_cues,
                "RepresentativeHistoryCues": history_cues,
                "ManualReviewPriority": priority,
                "ManualReviewReason": priority_reason,
                "Report": report,
            }
        )


alignment_df = pd.DataFrame(alignment_rows)


if len(alignment_df) != EXPECTED_ALIGNMENT_ROWS:

    raise RuntimeError(
        "Alignment row count mismatch: "
        f"expected {EXPECTED_ALIGNMENT_ROWS}, "
        f"found {len(alignment_df)}."
    )


alignment_df.to_csv(
    RESULT_ROOT / "02_gold_report_label_alignment_696.csv",
    index=False,
)


evidence_df = pd.DataFrame(evidence_rows)


evidence_df.to_csv(
    RESULT_ROOT / "03_evidence_mentions.csv",
    index=False,
)


# ============================================================
# 15. PER-LABEL ALIGNMENT SUMMARY
# ============================================================

label_summary_rows = []


for label in LABEL_COLUMNS:

    group = alignment_df[alignment_df["Label"] == label].copy()

    gold_positive = int((group["Gold"] == 1).sum())

    gold_negative = int((group["Gold"] == 0).sum())

    report_positive = int((group["ReportAssertion"] == "positive").sum())

    report_negative = int((group["ReportAssertion"] == "negative").sum())

    decidable = group["Decidable"].astype(bool)

    decidable_n = int(decidable.sum())

    if decidable_n > 0:

        accuracy_decidable = float(
            (
                group.loc[
                    decidable,
                    "ReportBinaryValue",
                ]
                .astype(int)
                .values
                == group.loc[
                    decidable,
                    "Gold",
                ]
                .astype(int)
                .values
            ).mean()
        )

    else:

        accuracy_decidable = np.nan

    positive_concordant = int(
        (group["AlignmentCategory"] == "concordant_positive").sum()
    )

    negative_concordant = int(
        (group["AlignmentCategory"] == "concordant_negative_explicit").sum()
    )

    positive_ppv = (
        positive_concordant / report_positive if report_positive > 0 else np.nan
    )

    negative_npv = (
        negative_concordant / report_negative if report_negative > 0 else np.nan
    )

    positive_capture = (
        positive_concordant / gold_positive if gold_positive > 0 else np.nan
    )

    explicit_negative_capture = (
        negative_concordant / gold_negative if gold_negative > 0 else np.nan
    )

    ppv_low, ppv_high = wilson_interval(
        positive_concordant,
        report_positive,
    )

    npv_low, npv_high = wilson_interval(
        negative_concordant,
        report_negative,
    )

    label_summary_rows.append(
        {
            "Label": label,
            "GoldPositive": gold_positive,
            "GoldNegative": gold_negative,
            "ReportPositive": report_positive,
            "ReportNegativeExplicit": report_negative,
            "DecidableN": decidable_n,
            "DecidableCoverage": (decidable_n / len(group)),
            "AccuracyAmongDecidable": accuracy_decidable,
            "ReportPositivePPV": positive_ppv,
            "ReportPositivePPV_WilsonLow": ppv_low,
            "ReportPositivePPV_WilsonHigh": ppv_high,
            "ReportNegativeNPV": negative_npv,
            "ReportNegativeNPV_WilsonLow": npv_low,
            "ReportNegativeNPV_WilsonHigh": npv_high,
            "GoldPositiveExplicitCaptureRate": positive_capture,
            "GoldNegativeExplicitCaptureRate": explicit_negative_capture,
            "DiscordantReportPositiveGoldNegative": int(
                (
                    group["AlignmentCategory"]
                    == ("discordant_report_" "positive_gold_negative")
                ).sum()
            ),
            "DiscordantReportNegativeGoldPositive": int(
                (
                    group["AlignmentCategory"]
                    == ("discordant_report_" "negative_gold_positive")
                ).sum()
            ),
            "GoldPositiveNotCaptured": int(
                (group["AlignmentCategory"] == "gold_positive_not_captured").sum()
            ),
            "Mixed": int((group["ReportAssertion"] == "mixed").sum()),
            "Uncertain": int((group["ReportAssertion"] == "uncertain").sum()),
            "RelatedAbnormality": int(
                (group["ReportAssertion"] == "related_abnormality").sum()
            ),
            "NotMentioned": int((group["ReportAssertion"] == "not_mentioned").sum()),
        }
    )


label_summary_df = pd.DataFrame(label_summary_rows)


label_summary_df.to_csv(
    RESULT_ROOT / "04_per_label_alignment_summary.csv",
    index=False,
)


# ============================================================
# 16. LANGUAGE x LABEL COVERAGE
# ============================================================

language_label_rows = []


for (
    language,
    label,
), group in alignment_df.groupby(
    [
        "Language",
        "Label",
    ]
):

    language_label_rows.append(
        {
            "Language": language,
            "Label": label,
            "StudyCount": int(len(group)),
            "GoldPositive": int((group["Gold"] == 1).sum()),
            "AnyMentionCoverage": float(
                (group["ReportAssertion"] != "not_mentioned").mean()
            ),
            "DecidableCoverage": float(group["Decidable"].astype(bool).mean()),
            "ReportPositive": int((group["ReportAssertion"] == "positive").sum()),
            "ReportNegative": int((group["ReportAssertion"] == "negative").sum()),
            "UncertainOrMixed": int(
                group["ReportAssertion"]
                .isin(
                    [
                        "uncertain",
                        "mixed",
                    ]
                )
                .sum()
            ),
        }
    )


language_label_df = pd.DataFrame(language_label_rows)


language_label_df.to_csv(
    RESULT_ROOT / "05_language_label_coverage.csv",
    index=False,
)


# ============================================================
# 17. DISAGREEMENTS
# ============================================================

disagreement_df = (
    alignment_df[alignment_df["AlignmentCategory"].str.startswith("discordant_")]
    .copy()
    .sort_values(
        [
            "Label",
            "Language",
            UID_COLUMN,
        ]
    )
)


disagreement_df.to_csv(
    RESULT_ROOT / "06_explicit_report_gold_disagreements.csv",
    index=False,
)


# Gold-positive cases that were not cleanly extracted.
gold_positive_review_df = (
    alignment_df[
        (alignment_df["Gold"] == 1) & (alignment_df["ReportAssertion"] != "positive")
    ]
    .copy()
    .sort_values(
        [
            "ManualReviewPriority",
            "Label",
            UID_COLUMN,
        ]
    )
)


gold_positive_review_df.to_csv(
    RESULT_ROOT / "07_gold_positive_not_cleanly_extracted.csv",
    index=False,
)


# ============================================================
# 18. SEVERITY / ONTOLOGY AUDIT
# ============================================================

severity_rows = []


for _, row in alignment_df[
    alignment_df["Severity"].fillna("").astype(str).str.len() > 0
].iterrows():

    for severity in str(row["Severity"]).split("|"):

        if not severity:

            continue

        severity_rows.append(
            {
                UID_COLUMN: row[UID_COLUMN],
                "Language": row["Language"],
                "Label": row["Label"],
                "Gold": row["Gold"],
                "ReportAssertion": row["ReportAssertion"],
                "AlignmentCategory": row["AlignmentCategory"],
                "Severity": severity,
                "Evidence": row["RepresentativeEvidence"],
            }
        )


severity_df = pd.DataFrame(severity_rows)


severity_df.to_csv(
    RESULT_ROOT / "08_severity_ontology_cases.csv",
    index=False,
)


severity_summary_df = (
    severity_df.groupby(
        [
            "Label",
            "Severity",
            "Gold",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="Count")
    if len(severity_df) > 0
    else pd.DataFrame(
        columns=[
            "Label",
            "Severity",
            "Gold",
            "Count",
        ]
    )
)


severity_summary_df.to_csv(
    RESULT_ROOT / "09_severity_by_label_and_gold.csv",
    index=False,
)


# ============================================================
# 19. MANUAL REVIEW QUEUE
# ============================================================

review_parts = []


# All priority 1-4 cases.
high_priority = alignment_df[alignment_df["ManualReviewPriority"] <= 4].copy()


review_parts.append(high_priority)


# Also sample a few concordant positive/negative cases per label
# so the extraction logic is audited for false confidence.
rng = np.random.default_rng(RANDOM_SEED)


for label in LABEL_COLUMNS:

    concordant = alignment_df[
        (alignment_df["Label"] == label)
        & (
            alignment_df["AlignmentCategory"].isin(
                [
                    "concordant_positive",
                    "concordant_negative_explicit",
                ]
            )
        )
    ].copy()

    if len(concordant) == 0:

        continue

    sample_n = min(
        CONCORDANT_REVIEW_SAMPLE_PER_LABEL,
        len(concordant),
    )

    chosen = rng.choice(
        np.arange(len(concordant)),
        size=sample_n,
        replace=False,
    )

    sampled = concordant.iloc[chosen].copy()

    sampled["ManualReviewReason"] = "concordant_quality_control_sample"

    review_parts.append(sampled)


review_df = (
    pd.concat(
        review_parts,
        ignore_index=True,
    )
    .drop_duplicates(
        subset=[
            UID_COLUMN,
            "Label",
        ]
    )
    .sort_values(
        [
            "ManualReviewPriority",
            "Label",
            UID_COLUMN,
        ]
    )
)


# Blank columns for actual human adjudication.
review_df["ReviewerReportAssertion"] = ""

review_df["ReviewerSeverity"] = ""

review_df["ReviewerAgreementWithGold"] = ""

review_df["ReviewerOntologyComment"] = ""

review_df["ReviewerNotes"] = ""


review_df.to_csv(
    RESULT_ROOT / "10_manual_review_queue.csv",
    index=False,
)


# ============================================================
# 20. REPORT-LEVEL OVERVIEW
# ============================================================

report_overview_rows = []


for uid, group in alignment_df.groupby(UID_COLUMN):

    first = group.iloc[0]

    report_overview_rows.append(
        {
            UID_COLUMN: uid,
            "Language": first["Language"],
            "GoldPositiveLabelCount": int(group["Gold"].sum()),
            "ReportPositiveAssertionCount": int(
                (group["ReportAssertion"] == "positive").sum()
            ),
            "ExplicitDisagreementCount": int(
                group["AlignmentCategory"].str.startswith("discordant_").sum()
            ),
            "IndeterminateCount": int(
                group["ReportAssertion"]
                .isin(
                    [
                        "uncertain",
                        "mixed",
                        "related_abnormality",
                        "mentioned_neutral",
                    ]
                )
                .sum()
            ),
            "NotMentionedCount": int(
                (group["ReportAssertion"] == "not_mentioned").sum()
            ),
            "MojibakeRepaired": bool(
                gold_df.loc[
                    gold_df[UID_COLUMN].astype(str) == str(uid),
                    "W11_MojibakeRepaired",
                ].iloc[0]
            ),
            "Report": first["Report"],
        }
    )


report_overview_df = pd.DataFrame(report_overview_rows)


report_overview_df.to_csv(
    RESULT_ROOT / "11_gold_report_overview.csv",
    index=False,
)


# ============================================================
# 21. LEXICON COVERAGE DIAGNOSTICS
# ============================================================

coverage_rows = []


for label in LABEL_COLUMNS:

    group = alignment_df[alignment_df["Label"] == label]

    coverage_rows.append(
        {
            "Label": label,
            "TotalGoldStudies": int(len(group)),
            "AnyEvidenceCount": int((group["EvidenceCount"] > 0).sum()),
            "AnyEvidenceCoverage": float((group["EvidenceCount"] > 0).mean()),
            "GoldPositiveWithoutAnyEvidence": int(
                ((group["Gold"] == 1) & (group["EvidenceCount"] == 0)).sum()
            ),
            "GoldNegativeWithoutAnyEvidence": int(
                ((group["Gold"] == 0) & (group["EvidenceCount"] == 0)).sum()
            ),
        }
    )


coverage_df = pd.DataFrame(coverage_rows)


coverage_df.to_csv(
    RESULT_ROOT / "12_lexicon_coverage_diagnostics.csv",
    index=False,
)


# ============================================================
# 22. MARKDOWN REPORT
# ============================================================

total_decidable = int(alignment_df["Decidable"].astype(bool).sum())


total_explicit_disagreements = int(
    alignment_df["AlignmentCategory"].str.startswith("discordant_").sum()
)


total_gold_positive_uncaptured = int(
    (
        (alignment_df["Gold"] == 1) & (alignment_df["ReportAssertion"] != "positive")
    ).sum()
)


markdown_lines = [
    "# RSNA W1.1 — Report ↔ Gold Label Alignment Audit",
    "",
    "## Scope",
    "",
    (f"- Gold studies: {EXPECTED_GOLD_STUDIES}"),
    (f"- Labels: {len(LABEL_COLUMNS)}"),
    ("- Report/label pairs audited: " f"{len(alignment_df)}"),
    ("- NLP model trained: **No**"),
    "",
    "## Automatic conservative extraction",
    "",
    (
        "- Explicitly decidable report/label pairs: "
        f"{total_decidable} / {len(alignment_df)} "
        f"({100.0 * total_decidable / len(alignment_df):.1f}%)"
    ),
    (
        "- Explicit report ↔ gold disagreements detected: "
        f"{total_explicit_disagreements}"
    ),
    (
        "- Gold-positive pairs not cleanly classified "
        "as report-positive: "
        f"{total_gold_positive_uncaptured}"
    ),
    "",
    (
        "Important: these are lexicon-audit statistics, "
        "not final clinical NLP performance estimates."
    ),
    "",
    "## Per-label summary",
    "",
    (
        "| Label | Gold + | Decidable coverage | "
        "Positive PPV | Negative NPV | "
        "Explicit +/− disagreements |"
    ),
    ("|---|---:|---:|---:|---:|---:|"),
]


for _, row in label_summary_df.iterrows():

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

    disagreement_count = int(
        row["DiscordantReportPositiveGoldNegative"]
        + row["DiscordantReportNegativeGoldPositive"]
    )

    markdown_lines.append(
        (
            f"| {row['Label']} "
            f"| {int(row['GoldPositive'])} "
            f"| {100.0 * row['DecidableCoverage']:.1f}% "
            f"| {ppv} "
            f"| {npv} "
            f"| {disagreement_count} |"
        )
    )


markdown_lines.extend(
    [
        "",
        "## Interpretation rules",
        "",
        ("- `not_mentioned` is **not** treated as negative."),
        (
            "- `related_abnormality` is preserved separately "
            "(for example bone-marrow edema without an explicit "
            "contusion term, or meniscal degeneration without "
            "an explicit tear)."
        ),
        (
            "- `mixed` preserves contradictory statements "
            "instead of forcing a binary label."
        ),
        (
            "- Severity terms are exported separately to help "
            "identify expert-label ontology thresholds."
        ),
        "",
        "## Highest-priority files",
        "",
        ("1. `06_explicit_report_gold_disagreements.csv`"),
        ("2. `07_gold_positive_not_cleanly_extracted.csv`"),
        ("3. `08_severity_ontology_cases.csv`"),
        ("4. `10_manual_review_queue.csv`"),
        ("5. `04_per_label_alignment_summary.csv`"),
        "",
        "## W2 gate",
        "",
        (
            "Do not assign pseudo-label trust weights until "
            "the manual-review queue has been inspected. "
            "W1.1 is designed to distinguish ontology/severity "
            "differences from extraction failures."
        ),
        "",
    ]
)


with open(
    RESULT_ROOT / "W1_1_REPORT.md",
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(markdown_lines))


# ============================================================
# 23. SAVE CONFIG
# ============================================================

config = {
    "phase": "W1.1 report-gold alignment audit",
    "audit_rule_version": "1.1.1_directional_negation_scope",
    "gold_source": source_name,
    "data_root": str(DATA_ROOT),
    "w1_result_root": str(W1_RESULT_ROOT),
    "output_root": str(RESULT_ROOT),
    "gold_studies": int(len(gold_df)),
    "labels": LABEL_COLUMNS,
    "expected_alignment_rows": EXPECTED_ALIGNMENT_ROWS,
    "actual_alignment_rows": int(len(alignment_df)),
    "nlp_model_trained": False,
    "pseudo_labels_generated": False,
    "not_mentioned_is_negative": False,
    "language_canonicalization": LANGUAGE_CANONICALIZATION,
    "automatic_assertion_classes": [
        "positive",
        "negative",
        "uncertain",
        "mixed",
        "related_abnormality",
        "mentioned_neutral",
        "not_mentioned",
    ],
    "context_window_chars": CONTEXT_WINDOW_CHARS,
    "mojibake_repair_backend": ("ftfy_if_installed_else_unicode_only"),
    "outputs": [
        "01_text_normalization_repairs.csv",
        "02_gold_report_label_alignment_696.csv",
        "03_evidence_mentions.csv",
        "04_per_label_alignment_summary.csv",
        "05_language_label_coverage.csv",
        "06_explicit_report_gold_disagreements.csv",
        "07_gold_positive_not_cleanly_extracted.csv",
        "08_severity_ontology_cases.csv",
        "09_severity_by_label_and_gold.csv",
        "10_manual_review_queue.csv",
        "11_gold_report_overview.csv",
        "12_lexicon_coverage_diagnostics.csv",
        "W1_1_REPORT.md",
    ],
}


save_json(
    RESULT_ROOT / "w1_1_config.json",
    config,
)


# ============================================================
# 24. DONE
# ============================================================

print()
print("=" * 80)

print("W1.1 ALIGNMENT AUDIT COMPLETE")

print("=" * 80)

print()

print(f"Output directory:\n" f"{RESULT_ROOT}")

print()

print("Most important outputs:")

for filename in [
    "W1_1_REPORT.md",
    "04_per_label_alignment_summary.csv",
    "06_explicit_report_gold_disagreements.csv",
    "07_gold_positive_not_cleanly_extracted.csv",
    "08_severity_ontology_cases.csv",
    "10_manual_review_queue.csv",
    "02_gold_report_label_alignment_696.csv",
]:

    print(f"  - {filename}")

print()

print("W1.1 trained NO NLP model and generated NO pseudo-labels.")
