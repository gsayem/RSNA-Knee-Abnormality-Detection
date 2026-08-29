# ============================================================
# 07_DICOM_protocol_archetype_analysis.py
# RSNA KNEE ABNORMALITY DETECTION
#
# FINAL METADATA-LEVEL INVESTIGATION
# PROTOCOL ARCHETYPE + SCANNER CONFOUNDING ANALYSIS
#
# Scope:
#   58 fully labeled studies only
#
# Inputs:
#   train.csv
#   train_series.csv
#   01_labeled_series_joined.csv
#   03_study_sequence_matrix.csv
#
# No image decoding
# No model training
#
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = "/kaggle/input/competitions/" "rsna-knee-abnormality-detection"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")

TRAIN_SERIES_CSV = os.path.join(DATA_ROOT, "train_series.csv")

# Previous integrated analysis
MASTER_CSV = (
    "/kaggle/working/" "rsna_58_integrated_analysis/" "01_labeled_series_joined.csv"
)

# Previous acquisition analysis
STUDY_MATRIX_CSV = (
    "/kaggle/working/"
    "rsna_acquisition_label_analysis/"
    "01_study_acquisition_label_matrix.csv"
)

OUTPUT_DIR = "/kaggle/working/" "rsna_protocol_archetype_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ============================================================
# 2. LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_CSV)

train_series = pd.read_csv(TRAIN_SERIES_CSV)

master = pd.read_csv(MASTER_CSV)

study_matrix = pd.read_csv(STUDY_MATRIX_CSV)

print("=" * 90)
print("INPUT DATA LOADED")
print("=" * 90)

print("=" * 90)
print("FINAL METADATA-LEVEL INVESTIGATION")
print("=" * 90)

print(f"\ntrain.csv studies       : {len(train):,}")

print(f"train_series rows       : {len(train_series):,}")

print(f"master series           : {len(master):,}")

print(f"study matrix rows       : {len(study_matrix):,}")


# ============================================================
# 3. IDENTIFY FULLY LABELED STUDIES
# ============================================================

fully_labeled = train[train[LABEL_COLUMNS].notna().all(axis=1)].copy()

fully_labeled["StudyInstanceUID"] = fully_labeled["StudyInstanceUID"].astype(str)


assert len(fully_labeled) == 58, (
    "Expected 58 fully labeled studies, " f"found {len(fully_labeled)}"
)


# ============================================================
# 4. NORMALIZE IDS
# ============================================================

study_matrix["StudyInstanceUID"] = study_matrix["StudyInstanceUID"].astype(str)

master["StudyInstanceUID"] = master["StudyInstanceUID"].astype(str)


# ============================================================
# 5. DEFINE THE SIX ACQUISITION CATEGORIES
# ============================================================

CORE_CATEGORIES = [
    "Has_Axial_Fluid_FS",
    "Has_Sagittal_NonFluid_NonFS",
    "Has_Coronal_Fluid_FS",
    "Has_Sagittal_Fluid_FS",
]

OPTIONAL_CATEGORIES = ["Has_Coronal_NonFluid_NonFS", "Has_Axial_NonFluid_NonFS"]

ALL_ACQUISITION_FEATURES = CORE_CATEGORIES + OPTIONAL_CATEGORIES


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

missing_columns = [c for c in ALL_ACQUISITION_FEATURES if c not in study_matrix.columns]

if missing_columns:

    raise ValueError("Missing acquisition columns:\n" + "\n".join(missing_columns))


# Convert to integer
for col in ALL_ACQUISITION_FEATURES:

    study_matrix[col] = (
        pd.to_numeric(study_matrix[col], errors="coerce").fillna(0).astype(int)
    )


# ============================================================
# 6. DEFINE PROTOCOL ARCHETYPE
# ============================================================


def classify_protocol_archetype(row):
    """
    Classify study according to availability of
    the four core categories plus two optional categories.

    Important:
        This is an analytical grouping, not a clinical
        protocol definition.
    """

    core_count = sum(int(row[c]) for c in CORE_CATEGORIES)

    optional_cor = int(row["Has_Coronal_NonFluid_NonFS"])

    optional_ax = int(row["Has_Axial_NonFluid_NonFS"])

    # --------------------------------------------------------
    # Fully complete core
    # --------------------------------------------------------

    if core_count == 4:

        if optional_cor == 0 and optional_ax == 0:

            return "CORE_4_ONLY"

        if optional_cor == 1 and optional_ax == 0:

            return "CORE_4_PLUS_CORONAL_NONFLUID"

        if optional_cor == 0 and optional_ax == 1:

            return "CORE_4_PLUS_AXIAL_NONFLUID"

        if optional_cor == 1 and optional_ax == 1:

            return "CORE_4_PLUS_BOTH_OPTIONAL"

    # --------------------------------------------------------
    # Any missing core category
    # --------------------------------------------------------

    if core_count == 3:

        return "INCOMPLETE_CORE_3"

    if core_count == 2:

        return "INCOMPLETE_CORE_2"

    if core_count <= 1:

        return "INCOMPLETE_CORE_0_1"

    return "OTHER"


study_matrix["ProtocolArchetype"] = study_matrix.apply(
    classify_protocol_archetype, axis=1
)


# ============================================================
# 7. HUMAN-READABLE ARCHETYPE DESCRIPTION
# ============================================================

ARCHETYPE_DESCRIPTION = {
    "CORE_4_ONLY": ("Core 4 acquisitions only"),
    "CORE_4_PLUS_CORONAL_NONFLUID": ("Core 4 + Coronal non-fluid/non-FS"),
    "CORE_4_PLUS_AXIAL_NONFLUID": ("Core 4 + Axial non-fluid/non-FS"),
    "CORE_4_PLUS_BOTH_OPTIONAL": ("Core 4 + both optional acquisitions"),
    "INCOMPLETE_CORE_3": ("Missing 1 core acquisition"),
    "INCOMPLETE_CORE_2": ("Missing 2 core acquisitions"),
    "INCOMPLETE_CORE_0_1": ("Missing 3+ core acquisitions"),
    "OTHER": "Other",
}


study_matrix["ProtocolArchetypeDescription"] = study_matrix["ProtocolArchetype"].map(
    ARCHETYPE_DESCRIPTION
)


# ============================================================
# 8. CHECK ACQUISITION PATTERNS
# ============================================================

study_matrix["AcquisitionPattern"] = (
    study_matrix[ALL_ACQUISITION_FEATURES].astype(str).agg("-".join, axis=1)
)


# ============================================================
# 9. MERGE GROUND-TRUTH LABELS
# ============================================================

labels = fully_labeled[["StudyInstanceUID"] + LABEL_COLUMNS].copy()

labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)


# ------------------------------------------------------------
# IMPORTANT:
# study_matrix may already contain labels because it was
# loaded from the output of the previous acquisition-analysis
# script.
#
# Remove those existing copies before merging the authoritative
# labels from train.csv.
# ------------------------------------------------------------

existing_label_columns = [c for c in LABEL_COLUMNS if c in study_matrix.columns]

if existing_label_columns:

    print("\nRemoving existing label columns " "from study_matrix before merge:")

    print(existing_label_columns)

    study_matrix = study_matrix.drop(columns=existing_label_columns)


study_matrix = study_matrix.merge(
    labels, on="StudyInstanceUID", how="left", validate="one_to_one"
)


# ------------------------------------------------------------
# Verify that labels now exist exactly once
# ------------------------------------------------------------

missing_labels = [c for c in LABEL_COLUMNS if c not in study_matrix.columns]

if missing_labels:

    raise RuntimeError(
        "Missing label columns after merge:\n" + "\n".join(missing_labels)
    )


# Make sure labels are numeric
for label in LABEL_COLUMNS:

    study_matrix[label] = pd.to_numeric(study_matrix[label], errors="raise").astype(int)


print("\nLabel columns successfully merged:")

print([c for c in study_matrix.columns if c in LABEL_COLUMNS])


# ============================================================
# 10. PREPARE SCANNER DATA FROM SERIES
# ============================================================

# ------------------------------------------------------------
# Normalize manufacturer names into broader families.
#
# This is a heuristic analytical normalization only.
# Raw manufacturer remains available separately.
# ------------------------------------------------------------


def normalize_manufacturer(value):

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().upper()

    if "PHILIPS" in text:
        return "Philips"

    if "SIEMENS" in text:
        return "Siemens"

    if "GE MEDICAL" in text or text.startswith("GE "):
        return "GE"

    if "TOSHIBA" in text:
        return "Toshiba"

    if "CANON" in text:
        return "Canon"

    return str(value)


master["ManufacturerFamily"] = master["Manufacturer"].apply(normalize_manufacturer)


# ============================================================
# 11. REDUCE SERIES-LEVEL SCANNER INFO TO STUDY LEVEL
# ============================================================

scanner_by_study = (
    master.groupby("StudyInstanceUID")
    .agg(
        ManufacturerFamilies=(
            "ManufacturerFamily",
            lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        ),
        RawManufacturers=(
            "Manufacturer",
            lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        ),
        ScannerModels=(
            "ManufacturerModelName",
            lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        ),
        FieldStrengths=(
            "MagneticFieldStrength",
            lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Determine predominant/primary values
# ------------------------------------------------------------


def first_pipe_value(value):

    if pd.isna(value):

        return "Unknown"

    text = str(value)

    if text.strip() == "":

        return "Unknown"

    return text.split("|")[0]


scanner_by_study["PrimaryManufacturerFamily"] = scanner_by_study[
    "ManufacturerFamilies"
].apply(first_pipe_value)


scanner_by_study["PrimaryFieldStrength"] = scanner_by_study["FieldStrengths"].apply(
    first_pipe_value
)


# Count distinct scanner properties
scanner_counts = (
    master.groupby("StudyInstanceUID")
    .agg(
        ManufacturerFamilyCount=("ManufacturerFamily", "nunique"),
        ScannerModelCount=("ManufacturerModelName", "nunique"),
        FieldStrengthCount=("MagneticFieldStrength", "nunique"),
    )
    .reset_index()
)


scanner_by_study = scanner_by_study.merge(
    scanner_counts, on="StudyInstanceUID", how="left"
)


# ============================================================
# 12. MERGE SCANNER INFORMATION
# ============================================================

scanner_columns_to_remove = [
    "ManufacturerFamilies",
    "RawManufacturers",
    "ScannerModels",
    "FieldStrengths",
    "PrimaryManufacturerFamily",
    "PrimaryFieldStrength",
    "ManufacturerFamilyCount",
    "ScannerModelCount",
    "FieldStrengthCount",
]

existing_scanner_columns = [
    c for c in scanner_columns_to_remove if c in study_matrix.columns
]

if existing_scanner_columns:

    print("\nRemoving existing scanner columns " "before scanner merge:")

    print(existing_scanner_columns)

    study_matrix = study_matrix.drop(columns=existing_scanner_columns)


study_matrix = study_matrix.merge(
    scanner_by_study, on="StudyInstanceUID", how="left", validate="one_to_one"
)


# ============================================================
# 13. SAVE PRIMARY STUDY-LEVEL TABLE
# ============================================================

primary_columns = [
    "StudyInstanceUID",
    "ProtocolArchetype",
    "ProtocolArchetypeDescription",
    "AcquisitionPattern",
    "TotalSeries",
    "TotalSlices",
    # Acquisition features
    *ALL_ACQUISITION_FEATURES,
    # Scanner
    "PrimaryManufacturerFamily",
    "ManufacturerFamilies",
    "PrimaryFieldStrength",
    "FieldStrengths",
    "ScannerModels",
    "ManufacturerFamilyCount",
    "ScannerModelCount",
    "FieldStrengthCount",
    # Labels
    *LABEL_COLUMNS,
]


primary_columns = [c for c in primary_columns if c in study_matrix.columns]


study_matrix[primary_columns].to_csv(
    os.path.join(OUTPUT_DIR, "01_protocol_archetypes.csv"), index=False
)


# ============================================================
# 14. PROTOCOL ARCHETYPE SUMMARY
# ============================================================

archetype_summary = (
    study_matrix.groupby(
        ["ProtocolArchetype", "ProtocolArchetypeDescription"], dropna=False
    )
    .agg(
        StudyCount=("StudyInstanceUID", "nunique"),
        MeanSeries=("TotalSeries", "mean"),
        MedianSeries=("TotalSeries", "median"),
        MeanSlices=("TotalSlices", "mean"),
        MedianSlices=("TotalSlices", "median"),
        Manufacturers=(
            "PrimaryManufacturerFamily",
            lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        ),
        FieldStrengths=(
            "PrimaryFieldStrength",
            lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        ),
    )
    .reset_index()
)


archetype_summary["StudyPercentage"] = (
    archetype_summary["StudyCount"] / len(study_matrix) * 100
)


archetype_summary = archetype_summary.sort_values("StudyCount", ascending=False)


archetype_summary.to_csv(
    os.path.join(OUTPUT_DIR, "02_protocol_archetype_summary.csv"), index=False
)

# ============================================================
# 14.5. VALIDATE LABEL SCHEMA
# ============================================================

print("\nFinal label schema:")

print([c for c in study_matrix.columns if c in LABEL_COLUMNS])


# Check for accidental duplicate suffixes
suffix_label_columns = [
    c
    for c in study_matrix.columns
    if (any(c.startswith(label + "_") for label in LABEL_COLUMNS))
]

if suffix_label_columns:

    raise RuntimeError(
        "Unexpected duplicate/suffixed label columns found:\n"
        + "\n".join(suffix_label_columns)
    )


missing_labels = [label for label in LABEL_COLUMNS if label not in study_matrix.columns]

if missing_labels:

    raise RuntimeError("Required labels missing:\n" + "\n".join(missing_labels))

print("Label schema validated successfully.")

# ============================================================
# 15. LABEL PREVALENCE BY PROTOCOL ARCHETYPE
# ============================================================

archetype_label_rows = []

for archetype, group in study_matrix.groupby("ProtocolArchetype"):

    for label in LABEL_COLUMNS:

        n = len(group)

        positive = int(group[label].sum())

        prevalence = positive / n if n > 0 else np.nan

        archetype_label_rows.append(
            {
                "ProtocolArchetype": archetype,
                "ProtocolArchetypeDescription": ARCHETYPE_DESCRIPTION.get(
                    archetype, archetype
                ),
                "StudyCount": n,
                "Label": label,
                "PositiveCount": positive,
                "Prevalence": prevalence,
            }
        )


archetype_label_df = pd.DataFrame(archetype_label_rows)


archetype_label_df.to_csv(
    os.path.join(OUTPUT_DIR, "03_protocol_archetype_labels.csv"), index=False
)


# ============================================================
# 16. SCANNER × PROTOCOL CROSS-TABS
# ============================================================

manufacturer_protocol = pd.crosstab(
    study_matrix["PrimaryManufacturerFamily"], study_matrix["ProtocolArchetype"]
)


manufacturer_protocol.to_csv(
    os.path.join(OUTPUT_DIR, "04_manufacturer_protocol_crosstab.csv")
)


fieldstrength_protocol = pd.crosstab(
    study_matrix["PrimaryFieldStrength"], study_matrix["ProtocolArchetype"]
)


fieldstrength_protocol.to_csv(
    os.path.join(OUTPUT_DIR, "05_field_strength_protocol_crosstab.csv")
)


scanner_model_protocol = pd.crosstab(
    study_matrix["ScannerModels"], study_matrix["ProtocolArchetype"]
)


scanner_model_protocol.to_csv(
    os.path.join(OUTPUT_DIR, "06_scanner_model_protocol_crosstab.csv")
)


# ============================================================
# 17. OPTIONAL ACQUISITION × SCANNER ANALYSIS
# ============================================================

OPTIONAL_FEATURES = ["Has_Coronal_NonFluid_NonFS", "Has_Axial_NonFluid_NonFS"]


optional_scanner_rows = []


for feature in OPTIONAL_FEATURES:

    for manufacturer in sorted(
        study_matrix["PrimaryManufacturerFamily"].dropna().unique()
    ):

        subset = study_matrix[study_matrix["PrimaryManufacturerFamily"] == manufacturer]

        n = len(subset)

        present = int(subset[feature].sum())

        optional_scanner_rows.append(
            {
                "AcquisitionFeature": feature,
                "ManufacturerFamily": manufacturer,
                "Studies": n,
                "Present": present,
                "PercentagePresent": (present / n * 100 if n > 0 else np.nan),
            }
        )


optional_scanner_df = pd.DataFrame(optional_scanner_rows)


optional_scanner_df.to_csv(
    os.path.join(OUTPUT_DIR, "07_optional_acquisition_scanner_analysis.csv"),
    index=False,
)


# ============================================================
# 18. CATEGORICAL ASSOCIATION METRIC
# ============================================================


def cramers_v(table):
    """
    Cramer's V for categorical association.

    Important:
        This is descriptive association only.
        With n=58, small contingency cells can make
        this unstable.
    """

    if table.size == 0:

        return np.nan

    try:

        chi2 = chi2_contingency(table, correction=False)[0]

    except Exception:

        return np.nan

    n = table.to_numpy().sum()

    if n == 0:

        return np.nan

    phi2 = chi2 / n

    rows, cols = table.shape

    # Bias correction (Bergsma/Wicher-style)
    phi2corr = max(0, phi2 - ((cols - 1) * (rows - 1) / (n - 1)))

    rcorr = rows - ((rows - 1) ** 2 / (n - 1))

    kcorr = cols - ((cols - 1) ** 2 / (n - 1))

    denominator = min(kcorr - 1, rcorr - 1)

    if denominator <= 0:

        return np.nan

    return float(np.sqrt(phi2corr / denominator))


# ------------------------------------------------------------
# Archetype vs scanner manufacturer
# ------------------------------------------------------------

scanner_assoc_rows = []


association_pairs = [
    (
        "ProtocolArchetype",
        "PrimaryManufacturerFamily",
        "Protocol_vs_ManufacturerFamily",
    ),
    ("ProtocolArchetype", "PrimaryFieldStrength", "Protocol_vs_FieldStrength"),
    ("ProtocolArchetype", "ScannerModels", "Protocol_vs_ScannerModel"),
]


for col_a, col_b, name in association_pairs:

    table = pd.crosstab(study_matrix[col_a], study_matrix[col_b])

    v = cramers_v(table)

    # Raw chi-square p-value is retained only as
    # an exploratory statistic.
    try:

        chi2, p_value, dof, expected = chi2_contingency(table, correction=False)

        expected_less_5_pct = float((expected < 5).mean() * 100)

    except Exception:

        chi2 = np.nan
        p_value = np.nan
        dof = np.nan
        expected_less_5_pct = np.nan

    scanner_assoc_rows.append(
        {
            "Analysis": name,
            "Rows": int(table.shape[0]),
            "Columns": int(table.shape[1]),
            "CramersV": v,
            "ChiSquare": chi2,
            "ChiSquarePValue": p_value,
            "ExpectedCellsBelow5Pct": expected_less_5_pct,
        }
    )


scanner_association_df = pd.DataFrame(scanner_assoc_rows)


scanner_association_df.to_csv(
    os.path.join(OUTPUT_DIR, "08_protocol_scanner_association.csv"), index=False
)


# ============================================================
# 19. OPTIONAL ACQUISITION × SCANNER FAMILY ASSOCIATION
# ============================================================

optional_assoc_rows = []


for feature in OPTIONAL_FEATURES:

    table = pd.crosstab(
        study_matrix[feature], study_matrix["PrimaryManufacturerFamily"]
    )

    v = cramers_v(table)

    try:

        chi2, p_value, dof, expected = chi2_contingency(table, correction=False)

        expected_less_5_pct = float((expected < 5).mean() * 100)

    except Exception:

        chi2 = np.nan
        p_value = np.nan
        dof = np.nan
        expected_less_5_pct = np.nan

    optional_assoc_rows.append(
        {
            "AcquisitionFeature": feature,
            "CramersV": v,
            "ChiSquare": chi2,
            "ChiSquarePValue": p_value,
            "ExpectedCellsBelow5Pct": expected_less_5_pct,
        }
    )


optional_assoc_df = pd.DataFrame(optional_assoc_rows)


optional_assoc_df.to_csv(
    os.path.join(OUTPUT_DIR, "09_optional_scanner_association.csv"), index=False
)


# ============================================================
# 20. LABEL PREVALENCE BY SCANNER FAMILY
# ============================================================

scanner_label_rows = []


for manufacturer, group in study_matrix.groupby("PrimaryManufacturerFamily"):

    for label in LABEL_COLUMNS:

        n = len(group)

        positive = int(group[label].sum())

        scanner_label_rows.append(
            {
                "ManufacturerFamily": manufacturer,
                "StudyCount": n,
                "Label": label,
                "PositiveCount": positive,
                "Prevalence": (positive / n if n > 0 else np.nan),
            }
        )


scanner_label_df = pd.DataFrame(scanner_label_rows)


scanner_label_df.to_csv(
    os.path.join(OUTPUT_DIR, "10_scanner_label_prevalence.csv"), index=False
)


# ============================================================
# 21. FIELD STRENGTH × LABEL PREVALENCE
# ============================================================

field_label_rows = []


for field_strength, group in study_matrix.groupby("PrimaryFieldStrength"):

    for label in LABEL_COLUMNS:

        n = len(group)

        positive = int(group[label].sum())

        field_label_rows.append(
            {
                "FieldStrength": field_strength,
                "StudyCount": n,
                "Label": label,
                "PositiveCount": positive,
                "Prevalence": (positive / n if n > 0 else np.nan),
            }
        )


field_label_df = pd.DataFrame(field_label_rows)


field_label_df.to_csv(
    os.path.join(OUTPUT_DIR, "11_field_strength_label_prevalence.csv"), index=False
)


# ============================================================
# 22. PROTOCOL PATTERN FREQUENCY
# ============================================================

pattern_summary = (
    study_matrix.groupby(["AcquisitionPattern"])
    .agg(StudyCount=("StudyInstanceUID", "nunique"))
    .reset_index()
    .sort_values("StudyCount", ascending=False)
)


pattern_summary["StudyPercentage"] = (
    pattern_summary["StudyCount"] / len(study_matrix) * 100
)


pattern_summary.to_csv(
    os.path.join(OUTPUT_DIR, "12_acquisition_pattern_frequency.csv"), index=False
)


# ============================================================
# 23. INTEGRATED JSON SUMMARY
# ============================================================

summary = {
    "dataset": {
        "train_studies": int(len(train)),
        "gold_labeled_studies": int(len(study_matrix)),
        "series": int(len(master)),
    },
    "protocol_archetypes": archetype_summary.to_dict(orient="records"),
    "acquisition_patterns": pattern_summary.to_dict(orient="records"),
    "protocol_scanner_associations": scanner_association_df.to_dict(orient="records"),
    "optional_scanner_associations": optional_assoc_df.to_dict(orient="records"),
    "manufacturer_distribution": study_matrix["PrimaryManufacturerFamily"]
    .value_counts()
    .to_dict(),
    "field_strength_distribution": study_matrix["PrimaryFieldStrength"]
    .value_counts()
    .to_dict(),
    "label_prevalence": {
        label: int(study_matrix[label].sum()) for label in LABEL_COLUMNS
    },
}


with open(os.path.join(OUTPUT_DIR, "13_protocol_archetype_summary.json"), "w") as f:

    json.dump(summary, f, indent=4, default=str)


# ============================================================
# 24. VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# A. Protocol archetype distribution
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))


plt.bar(archetype_summary["ProtocolArchetype"], archetype_summary["StudyCount"])


plt.xticks(rotation=65, ha="right")

plt.ylabel("Number of studies")

plt.title("Protocol Archetypes Across 58 Gold-Labeled Studies")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "01_protocol_archetype_distribution.png"), dpi=200)

plt.show()


# ------------------------------------------------------------
# B. Manufacturer × archetype heatmap
# ------------------------------------------------------------

manufacturer_heatmap = manufacturer_protocol


plt.figure(figsize=(12, 7))

plt.imshow(manufacturer_heatmap.values, aspect="auto", interpolation="nearest")

plt.colorbar(label="Study count")

plt.xticks(
    range(len(manufacturer_heatmap.columns)),
    manufacturer_heatmap.columns,
    rotation=65,
    ha="right",
)

plt.yticks(range(len(manufacturer_heatmap.index)), manufacturer_heatmap.index)

plt.xlabel("Protocol archetype")

plt.ylabel("Manufacturer family")

plt.title("Manufacturer Family × Protocol Archetype")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "02_manufacturer_protocol_heatmap.png"), dpi=200)

plt.show()


# ------------------------------------------------------------
# C. Field strength × archetype
# ------------------------------------------------------------

field_heatmap = fieldstrength_protocol


plt.figure(figsize=(12, 5))

plt.imshow(field_heatmap.values, aspect="auto", interpolation="nearest")

plt.colorbar(label="Study count")

plt.xticks(
    range(len(field_heatmap.columns)), field_heatmap.columns, rotation=65, ha="right"
)

plt.yticks(range(len(field_heatmap.index)), field_heatmap.index)

plt.xlabel("Protocol archetype")

plt.ylabel("Field strength")

plt.title("Field Strength × Protocol Archetype")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "03_fieldstrength_protocol_heatmap.png"), dpi=200)

plt.show()


# ------------------------------------------------------------
# D. Optional acquisition by manufacturer
# ------------------------------------------------------------

optional_pivot = optional_scanner_df.pivot(
    index="ManufacturerFamily", columns="AcquisitionFeature", values="PercentagePresent"
)


plt.figure(figsize=(10, 6))

plt.imshow(optional_pivot.values, aspect="auto", interpolation="nearest")

plt.colorbar(label="Percentage of studies")

plt.xticks(
    range(len(optional_pivot.columns)), optional_pivot.columns, rotation=45, ha="right"
)

plt.yticks(range(len(optional_pivot.index)), optional_pivot.index)

plt.xlabel("Optional acquisition")

plt.ylabel("Manufacturer family")

plt.title("Optional Acquisition Availability by Manufacturer Family")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_optional_acquisition_by_manufacturer.png"), dpi=200
)

plt.show()


# ============================================================
# 25. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FINAL PROTOCOL ARCHETYPE ANALYSIS")
print("=" * 90)


print("\nProtocol archetypes:")

print(
    archetype_summary[
        [
            "ProtocolArchetype",
            "StudyCount",
            "StudyPercentage",
            "MeanSeries",
            "MedianSeries",
        ]
    ].to_string(index=False)
)


print("\nExact acquisition patterns:")

print(pattern_summary.head(20).to_string(index=False))


print("\nProtocol × scanner association:")

print(scanner_association_df.to_string(index=False))


print("\nOptional acquisition × manufacturer:")

print(optional_assoc_df.to_string(index=False))


print("\nManufacturer families:")

print(study_matrix["PrimaryManufacturerFamily"].value_counts().to_string())


print("\nField strengths:")

print(study_matrix["PrimaryFieldStrength"].value_counts().to_string())


print("\nOutput directory:")

print(os.path.abspath(OUTPUT_DIR))


print("\nDONE.")
