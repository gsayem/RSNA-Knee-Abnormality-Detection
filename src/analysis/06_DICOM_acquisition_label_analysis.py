# ============================================================
# 06_DICOM_acquisition_label_analysis.py
# RSNA KNEE ABNORMALITY DETECTION
#
# GOLD-LABELED STUDY × ACQUISITION COVERAGE ANALYSIS
#
# Purpose:
#   Investigate what MRI acquisition categories are available
#   across the 58 fully labeled studies and how those categories
#   relate to the 12 abnormality labels.
#
# NO MODEL TRAINING
# NO PIXEL PROCESSING
# NO IMAGE DECODING
#
# Primary source:
#   train.csv
#   train_series.csv
#   01_labeled_series_joined.csv
#
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = "/kaggle/input/competitions/" "rsna-knee-abnormality-detection"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")

TRAIN_SERIES_CSV = os.path.join(DATA_ROOT, "train_series.csv")

# This is the output from our previous integrated
# DICOM + train.csv + train_series.csv analysis.

MASTER_CSV = (
    "/kaggle/working/" "rsna_58_integrated_analysis/" "01_labeled_series_joined.csv"
)

OUTPUT_DIR = "/kaggle/working/" "rsna_acquisition_label_analysis"

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

print("=" * 90)
print("LOADING DATA")
print("=" * 90)

train = pd.read_csv(TRAIN_CSV)

train_series = pd.read_csv(TRAIN_SERIES_CSV)

master = pd.read_csv(MASTER_CSV)


print(f"train.csv studies      : {len(train):,}")

print(f"train_series.csv rows  : {len(train_series):,}")

print(f"master joined rows     : {len(master):,}")


# ============================================================
# 3. IDENTIFY THE GOLD-LABELED COHORT
# ============================================================

fully_labeled_mask = train[LABEL_COLUMNS].notna().all(axis=1)

labeled_studies = train.loc[fully_labeled_mask].copy()

labeled_study_uids = labeled_studies["StudyInstanceUID"].astype(str).tolist()

print(f"\nFully labeled studies : " f"{len(labeled_studies)}")


# Safety check
assert len(labeled_studies) == 58, (
    "Expected 58 fully labeled studies " f"but found {len(labeled_studies)}."
)


# ============================================================
# 4. FILTER MASTER DATA
# ============================================================

master = master[master["StudyInstanceUID"].astype(str).isin(labeled_study_uids)].copy()


# Normalize identifiers
master["StudyInstanceUID"] = master["StudyInstanceUID"].astype(str)


master["SeriesInstanceUID"] = master["SeriesInstanceUID"].astype(str)


# ============================================================
# 5. CREATE PRIMARY ACQUISITION CATEGORIES
# ============================================================

# The competition explicitly defines:
#
#   Fluid_Sensitive
#   Fat_Suppression
#   Anatomical_Plane
#
# We use these as the primary categorical representation.


def acquisition_category(row):

    plane = str(row["TrainSeries_AnatomicalPlane"])

    fluid = int(row["Fluid_Sensitive"])

    fat = int(row["Fat_Suppression"])

    fluid_name = "Fluid" if fluid == 1 else "NonFluid"

    fat_name = "FS" if fat == 1 else "NonFS"

    return f"{plane}_{fluid_name}_{fat_name}"


master["AcquisitionCategory"] = master.apply(acquisition_category, axis=1)


# ============================================================
# 6. CREATE SIMPLER CATEGORY FLAGS
# ============================================================

master["IsSagittal"] = (master["TrainSeries_AnatomicalPlane"] == "Sagittal").astype(int)


master["IsCoronal"] = (master["TrainSeries_AnatomicalPlane"] == "Coronal").astype(int)


master["IsAxial"] = (master["TrainSeries_AnatomicalPlane"] == "Axial").astype(int)


master["IsFluidSensitive"] = (master["Fluid_Sensitive"] == 1).astype(int)


master["IsFatSuppressed"] = (master["Fat_Suppression"] == 1).astype(int)


# ============================================================
# 7. DESCRIPTION / DICOM SUPPORTING FLAGS
# ============================================================

master["IsDummyDescription"] = (master["DescriptionType"] == "Dummy").astype(int)


master["IsBlankDescription"] = (master["DescriptionType"] == "Blank").astype(int)


master["HasNamedDescription"] = (master["DescriptionType"] == "Named").astype(int)


# Existing heuristic flags from the previous analysis
# are retained for exploratory analysis only.

for col in ["Is_PD", "Is_T1", "Is_T2", "Is_Fat_Suppressed"]:

    if col in master.columns:

        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0).astype(int)


# ============================================================
# 8. STUDY-LEVEL ACQUISITION MATRIX
# ============================================================

print("\n")
print("=" * 90)
print("BUILDING STUDY-LEVEL ACQUISITION MATRIX")
print("=" * 90)


study_records = []


for study_uid, group in master.groupby("StudyInstanceUID"):

    record = {
        "StudyInstanceUID": study_uid,
        "TotalSeries": int(len(group)),
        "TotalSlices": int(group["NumberOfSlices"].sum()),
    }

    # --------------------------------------------------------
    # Plane presence/count
    # --------------------------------------------------------

    for plane in ["Sagittal", "Coronal", "Axial"]:

        plane_mask = group["TrainSeries_AnatomicalPlane"] == plane

        record[f"{plane}_SeriesCount"] = int(plane_mask.sum())

        record[f"Has_{plane}"] = int(plane_mask.any())

    # --------------------------------------------------------
    # Fluid / FS
    # --------------------------------------------------------

    record["FluidSensitiveSeriesCount"] = int((group["Fluid_Sensitive"] == 1).sum())

    record["FatSuppressedSeriesCount"] = int((group["Fat_Suppression"] == 1).sum())

    # --------------------------------------------------------
    # Primary acquisition categories
    # --------------------------------------------------------

    category_counts = group["AcquisitionCategory"].value_counts().to_dict()

    all_categories = sorted(master["AcquisitionCategory"].dropna().unique())

    for category in all_categories:

        safe_name = category.replace(" ", "_").replace("/", "_")

        count_value = int(category_counts.get(category, 0))

        record[f"Series_{safe_name}"] = count_value

        record[f"Has_{safe_name}"] = int(count_value > 0)

    # --------------------------------------------------------
    # Description quality
    # --------------------------------------------------------

    record["DummySeriesCount"] = int(group["IsDummyDescription"].sum())

    record["BlankDescriptionCount"] = int(group["IsBlankDescription"].sum())

    record["NamedDescriptionCount"] = int(group["HasNamedDescription"].sum())

    # --------------------------------------------------------
    # Heuristic sequence types
    # --------------------------------------------------------

    if "Is_PD" in group.columns:

        record["PDLikeSeriesCount"] = int(group["Is_PD"].sum())

    if "Is_T1" in group.columns:

        record["T1LikeSeriesCount"] = int(group["Is_T1"].sum())

    if "Is_T2" in group.columns:

        record["T2LikeSeriesCount"] = int(group["Is_T2"].sum())

    # --------------------------------------------------------
    # Scanner
    # --------------------------------------------------------

    record["ManufacturerCount"] = int(
        group["Manufacturer"].replace("", np.nan).dropna().nunique()
    )

    record["ScannerModelCount"] = int(
        group["ManufacturerModelName"].replace("", np.nan).dropna().nunique()
    )

    record["FieldStrengthCount"] = int(
        group["MagneticFieldStrength"].replace("", np.nan).dropna().nunique()
    )

    study_records.append(record)


study_df = pd.DataFrame(study_records)


# ============================================================
# 9. ADD THE 12 LABELS
# ============================================================

label_lookup = labeled_studies[["StudyInstanceUID"] + LABEL_COLUMNS].copy()

label_lookup["StudyInstanceUID"] = label_lookup["StudyInstanceUID"].astype(str)


study_df = study_df.merge(
    label_lookup, on="StudyInstanceUID", how="left", validate="one_to_one"
)


# ============================================================
# 10. SAVE STUDY MATRIX
# ============================================================

study_matrix_path = os.path.join(OUTPUT_DIR, "01_study_acquisition_label_matrix.csv")

study_df.to_csv(study_matrix_path, index=False)

print("Saved:", study_matrix_path)


# ============================================================
# 11. CATEGORY COVERAGE SUMMARY
# ============================================================

category_summary = []


for category in sorted(master["AcquisitionCategory"].dropna().unique()):

    series_mask = master["AcquisitionCategory"] == category

    studies_with_category = master.loc[series_mask, "StudyInstanceUID"].nunique()

    category_summary.append(
        {
            "AcquisitionCategory": category,
            "SeriesCount": int(series_mask.sum()),
            "StudyCount": int(studies_with_category),
            "StudyCoveragePct": (studies_with_category / len(study_df) * 100),
        }
    )


category_summary_df = pd.DataFrame(category_summary)


category_summary_df = category_summary_df.sort_values(
    ["StudyCount", "SeriesCount"], ascending=False
)


category_summary_path = os.path.join(OUTPUT_DIR, "02_acquisition_category_summary.csv")


category_summary_df.to_csv(category_summary_path, index=False)


print("Saved:", category_summary_path)


# ============================================================
# 12. LABEL VS ACQUISITION AVAILABILITY
# ============================================================

print("\n")
print("=" * 90)
print("LABEL × ACQUISITION AVAILABILITY")
print("=" * 90)


# We'll use study-level binary availability.
#
# Example:
#
# Has_Sagittal_Fluid_FS = 1
#
# then compare:
#
# ACL positive vs ACL negative
#
# This is descriptive, NOT causal.


availability_columns = [col for col in study_df.columns if col.startswith("Has_")]


association_records = []


for acquisition in availability_columns:

    for label in LABEL_COLUMNS:

        x = study_df[acquisition].astype(int)

        y = study_df[label].astype(int)

        # ----------------------------------------------------
        # 2x2
        # ----------------------------------------------------

        a = int(((x == 1) & (y == 1)).sum())

        b = int(((x == 1) & (y == 0)).sum())

        c = int(((x == 0) & (y == 1)).sum())

        d = int(((x == 0) & (y == 0)).sum())

        n_x1 = a + b
        n_x0 = c + d

        n_y1 = a + c
        n_y0 = b + d

        prevalence_with = a / n_x1 if n_x1 > 0 else np.nan

        prevalence_without = c / n_x0 if n_x0 > 0 else np.nan

        # Risk ratio-like descriptive ratio.
        #
        # This is NOT inferential and becomes unstable
        # for small denominators.

        if prevalence_without > 0:

            prevalence_ratio = prevalence_with / prevalence_without

        else:

            prevalence_ratio = np.nan

        association_records.append(
            {
                "AcquisitionFeature": acquisition,
                "Label": label,
                "StudiesWithFeature": n_x1,
                "StudiesWithoutFeature": n_x0,
                "PositiveWithFeature": a,
                "NegativeWithFeature": b,
                "PositiveWithoutFeature": c,
                "NegativeWithoutFeature": d,
                "PrevalenceWhenPresent": prevalence_with,
                "PrevalenceWhenAbsent": prevalence_without,
                "DescriptivePrevalenceRatio": prevalence_ratio,
            }
        )


association_df = pd.DataFrame(association_records)


association_path = os.path.join(OUTPUT_DIR, "03_acquisition_label_associations.csv")


association_df.to_csv(association_path, index=False)


print("Saved:", association_path)


# ============================================================
# 13. LABEL × PLANE AVAILABILITY
# ============================================================

plane_features = ["Has_Sagittal", "Has_Coronal", "Has_Axial"]


plane_records = []


for plane_feature in plane_features:

    for label in LABEL_COLUMNS:

        subset = study_df[study_df[plane_feature] == 1]

        positive = int(subset[label].sum())

        count = len(subset)

        prevalence = positive / count if count > 0 else np.nan

        plane_records.append(
            {
                "PlaneFeature": plane_feature,
                "Label": label,
                "Studies": count,
                "Positive": positive,
                "Prevalence": prevalence,
            }
        )


plane_df = pd.DataFrame(plane_records)


plane_df.to_csv(os.path.join(OUTPUT_DIR, "04_plane_label_summary.csv"), index=False)


# ============================================================
# 14. CO-OCCURRENCE OF ACQUISITION CATEGORIES
# ============================================================

# Binary study-level category presence matrix.

category_features = [
    col
    for col in study_df.columns
    if (
        col.startswith("Has_")
        and col not in ["Has_Sagittal", "Has_Coronal", "Has_Axial"]
    )
]


category_binary = study_df[category_features].astype(int)


category_cooccurrence = category_binary.T.dot(category_binary)


category_cooccurrence.to_csv(
    os.path.join(OUTPUT_DIR, "05_acquisition_category_cooccurrence.csv")
)


# ============================================================
# 15. LABEL CO-OCCURRENCE
# ============================================================

label_binary = study_df[LABEL_COLUMNS].astype(int)


label_cooccurrence = label_binary.T.dot(label_binary)


label_cooccurrence.to_csv(os.path.join(OUTPUT_DIR, "06_label_cooccurrence.csv"))


# ============================================================
# 16. UNUSUAL STUDY COVERAGE
# ============================================================

# We identify studies that have:
#
#   - unusually many series
#   - Dummy descriptions
#   - blank descriptions
#   - repeated categories
#
# This is descriptive only.


study_df["NonStandardDescriptionCount"] = (
    study_df["DummySeriesCount"] + study_df["BlankDescriptionCount"]
)


study_df["MaxSeriesInOneCategory"] = study_df[
    [col for col in study_df.columns if col.startswith("Series_")]
].max(axis=1)


unusual_studies = study_df[
    (study_df["DummySeriesCount"] > 0)
    | (study_df["BlankDescriptionCount"] > 0)
    | (study_df["TotalSeries"] >= 8)
    | (study_df["MaxSeriesInOneCategory"] >= 3)
].copy()


unusual_path = os.path.join(OUTPUT_DIR, "07_unusual_study_coverage.csv")


unusual_studies.to_csv(unusual_path, index=False)


# ============================================================
# 17. LABEL PREVALENCE BY ACQUISITION CATEGORY
# ============================================================

# This is a more direct descriptive table:
#
# For each acquisition category:
#
#   how many studies have it?
#   how many of those studies are positive for each label?
#
# Again: descriptive only.


category_label_records = []


for category in category_summary_df["AcquisitionCategory"]:

    has_category = study_df["Has_" + category.replace(" ", "_").replace("/", "_")]

    for label in LABEL_COLUMNS:

        subset = study_df[has_category == 1]

        n = len(subset)

        positive = int(subset[label].sum())

        prevalence = positive / n if n > 0 else np.nan

        category_label_records.append(
            {
                "AcquisitionCategory": category,
                "Label": label,
                "StudiesWithCategory": n,
                "PositiveStudies": positive,
                "Prevalence": prevalence,
            }
        )


category_label_df = pd.DataFrame(category_label_records)


category_label_df.to_csv(
    os.path.join(OUTPUT_DIR, "08_category_label_prevalence.csv"), index=False
)


# ============================================================
# 18. SCANNER / FIELD-STRENGTH SUMMARY
# ============================================================

scanner_summary = (
    master.groupby(
        ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"], dropna=False
    )
    .agg(
        SeriesCount=("SeriesInstanceUID", "count"),
        StudyCount=("StudyInstanceUID", "nunique"),
    )
    .reset_index()
    .sort_values(["StudyCount", "SeriesCount"], ascending=False)
)


scanner_summary.to_csv(os.path.join(OUTPUT_DIR, "09_scanner_summary.csv"), index=False)


# ============================================================
# 19. PLOTS
# ============================================================

print("\n")
print("=" * 90)
print("GENERATING PLOTS")
print("=" * 90)


# ------------------------------------------------------------
# Plot 1: acquisition category study coverage
# ------------------------------------------------------------

plot_df = category_summary_df.sort_values("StudyCount", ascending=False).head(20)


plt.figure(figsize=(12, 7))

plt.bar(plot_df["AcquisitionCategory"], plot_df["StudyCount"])

plt.xticks(rotation=70, ha="right")

plt.ylabel("Number of studies")

plt.title("Acquisition Category Coverage Across 58 Labeled Studies")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "01_acquisition_category_coverage.png"), dpi=200)

plt.show()


# ------------------------------------------------------------
# Plot 2: series count per study
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    study_df["TotalSeries"],
    bins=np.arange(
        study_df["TotalSeries"].min() - 0.5, study_df["TotalSeries"].max() + 1.5, 1
    ),
)

plt.xlabel("Number of series")

plt.ylabel("Number of studies")

plt.title("Number of MRI Series per Labeled Study")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "02_series_count.png"), dpi=200)

plt.show()


# ------------------------------------------------------------
# Plot 3: label prevalence
# ------------------------------------------------------------

label_counts = study_df[LABEL_COLUMNS].sum().sort_values(ascending=False)


plt.figure(figsize=(12, 6))

plt.bar(label_counts.index, label_counts.values)

plt.xticks(rotation=60, ha="right")

plt.ylabel("Positive studies")

plt.title("Ground-Truth Label Prevalence")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "03_label_prevalence.png"), dpi=200)

plt.show()


# ============================================================
# 20. HEATMAP: ACQUISITION FEATURE × LABEL
# ============================================================

heatmap_features = [col for col in availability_columns if (study_df[col].sum() >= 5)]


heatmap_data = []


for feature in heatmap_features:

    row_values = []

    for label in LABEL_COLUMNS:

        mask = study_df[feature] == 1

        n = int(mask.sum())

        if n == 0:

            value = np.nan

        else:

            value = float(study_df.loc[mask, label].mean())

        row_values.append(value)

    heatmap_data.append(row_values)


heatmap_df = pd.DataFrame(heatmap_data, index=heatmap_features, columns=LABEL_COLUMNS)


plt.figure(figsize=(14, 10))

plt.imshow(heatmap_df.values, aspect="auto", interpolation="nearest")

plt.colorbar(label="Positive prevalence")

plt.xticks(range(len(LABEL_COLUMNS)), LABEL_COLUMNS, rotation=60, ha="right")

plt.yticks(range(len(heatmap_features)), heatmap_features)

plt.title("Descriptive Label Prevalence by Acquisition Availability")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "04_acquisition_label_heatmap.png"), dpi=200)

plt.show()


# ============================================================
# 21. GENERATE JSON SUMMARY
# ============================================================

summary = {
    "dataset": {
        "train_studies": int(len(train)),
        "gold_labeled_studies": int(len(labeled_studies)),
        "master_series": int(len(master)),
    },
    "study_structure": {
        "series_mean": float(study_df["TotalSeries"].mean()),
        "series_median": float(study_df["TotalSeries"].median()),
        "series_min": int(study_df["TotalSeries"].min()),
        "series_max": int(study_df["TotalSeries"].max()),
        "slices_mean": float(study_df["TotalSlices"].mean()),
        "slices_median": float(study_df["TotalSlices"].median()),
    },
    "plane_coverage": {
        "sagittal": int(study_df["Has_Sagittal"].sum()),
        "coronal": int(study_df["Has_Coronal"].sum()),
        "axial": int(study_df["Has_Axial"].sum()),
    },
    "acquisition_categories": category_summary_df.to_dict(orient="records"),
    "label_prevalence": {label: int(study_df[label].sum()) for label in LABEL_COLUMNS},
    "description_quality": {
        "dummy_studies": int((study_df["DummySeriesCount"] > 0).sum()),
        "blank_studies": int((study_df["BlankDescriptionCount"] > 0).sum()),
    },
    "unusual_studies": int(len(unusual_studies)),
}


with open(os.path.join(OUTPUT_DIR, "10_acquisition_label_analysis.json"), "w") as f:

    json.dump(summary, f, indent=4, default=str)


# ============================================================
# 22. PRINT KEY RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("KEY RESULTS")
print("=" * 90)


print("\nAcquisition categories:")

print(
    category_summary_df[
        ["AcquisitionCategory", "SeriesCount", "StudyCount", "StudyCoveragePct"]
    ]
    .head(30)
    .to_string(index=False)
)


print("\nUnusual studies:")

print(
    unusual_studies[
        [
            "StudyInstanceUID",
            "TotalSeries",
            "TotalSlices",
            "DummySeriesCount",
            "BlankDescriptionCount",
            "MaxSeriesInOneCategory",
        ]
    ].to_string(index=False)
)


print("\nLabel prevalence:")

for label in LABEL_COLUMNS:

    positive = int(study_df[label].sum())

    print(f"  {label:20s} " f"{positive:2d}/58 " f"({positive / 58:.1%})")


print("\nOutput directory:")

print(os.path.abspath(OUTPUT_DIR))


print("\nDONE.")
