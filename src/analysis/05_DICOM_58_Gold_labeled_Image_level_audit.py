# ============================================================
# 05_DICOM_58_Gold_labeled_Image_level_audit.py
# RSNA KNEE MRI
# IMAGE-LEVEL AUDIT OF:
#
#   1. Strict repeated acquisition groups
#   2. DummySeriesDesc! studies
#   3. Blank SeriesDescription studies
#
# ============================================================

import os
import glob
import json
import math
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

# try:
# import pydicom
# except ImportError:
#     !pip install -q pydicom
#     import pydicom


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

DATA_ROOT = "/kaggle/input/competitions/" "rsna-knee-abnormality-detection"

TRAIN_SERIES_ROOT = os.path.join(DATA_ROOT, "train_series")

MASTER_CSV = (
    "/kaggle/working/" "rsna_58_integrated_analysis/" "01_labeled_series_joined.csv"
)

# If the CSV isn't in /kaggle/working, change this:
#
# MASTER_CSV = "/kaggle/input/..."
#
# or upload/copy your 01_labeled_series_joined.csv there.


OUTPUT_DIR = "/kaggle/working/" "rsna_image_level_audit"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD MASTER METADATA
# ============================================================

master = pd.read_csv(MASTER_CSV)

print(f"Master records: {len(master)}")


# ============================================================
# 3. IDENTIFY STRICT REPEATED GROUPS
# ============================================================

STRICT_COLUMNS = [
    "TrainSeries_AnatomicalPlane",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "EchoTime",
    "RepetitionTime",
    "Manufacturer",
    "ManufacturerModelName",
    "SequenceName",
    "SeriesDescription",
]


def normalize_signature_value(value):
    if pd.isna(value):
        return "NA"

    return str(value).strip()


signature_df = master.copy()

for col in STRICT_COLUMNS:
    signature_df[col] = signature_df[col].apply(normalize_signature_value)


group_cols = ["StudyInstanceUID"] + STRICT_COLUMNS


signature_counts = (
    signature_df.groupby(group_cols, dropna=False)
    .size()
    .reset_index(name="RepeatedCount")
)


strict_repeats = signature_counts[signature_counts["RepeatedCount"] > 1].copy()


print("\nStrict repeated groups:")
# display(strict_repeats)


# ============================================================
# 4. SELECT EXACT REPEATED SERIES
# ============================================================

selected_pairs = []

for _, repeated_group in strict_repeats.iterrows():

    study_uid = repeated_group["StudyInstanceUID"]

    candidates = master[(master["StudyInstanceUID"] == study_uid)].copy()

    # Match all signature fields
    mask = np.ones(len(candidates), dtype=bool)

    for col in STRICT_COLUMNS:

        target = normalize_signature_value(repeated_group[col])

        mask &= candidates[col].apply(normalize_signature_value).values == target

    matching = candidates[mask]

    # Keep every matching series
    series_uids = matching["SeriesInstanceUID"].astype(str).tolist()

    selected_pairs.append(
        {"StudyInstanceUID": study_uid, "SeriesInstanceUIDs": series_uids}
    )


# ============================================================
# 5. DICOM HELPERS
# ============================================================


def read_series(study_uid, series_uid):
    """
    Read one series and return:
        volume
        datasets
        positions
    """

    series_dir = os.path.join(TRAIN_SERIES_ROOT, str(study_uid), str(series_uid))

    paths = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))

    if not paths:

        raise FileNotFoundError(f"No DICOM files:\n{series_dir}")

    datasets = []

    for path in paths:

        ds = pydicom.dcmread(path, force=True)

        datasets.append({"path": path, "ds": ds})

    # --------------------------------------------------------
    # Spatial position
    # --------------------------------------------------------

    def scalar_position(ds):

        orientation = getattr(ds, "ImageOrientationPatient", None)

        position = getattr(ds, "ImagePositionPatient", None)

        if orientation is None or position is None:
            return None

        try:

            row = np.asarray(orientation[:3], dtype=np.float64)

            col = np.asarray(orientation[3:], dtype=np.float64)

            normal = np.cross(row, col)

            pos = np.asarray(position, dtype=np.float64)

            return float(np.dot(pos, normal))

        except Exception:

            return None

    for item in datasets:

        item["position_scalar"] = scalar_position(item["ds"])

    if all(item["position_scalar"] is not None for item in datasets):

        datasets.sort(key=lambda x: x["position_scalar"])

    else:

        datasets.sort(key=lambda x: int(getattr(x["ds"], "InstanceNumber", 0)))

    # --------------------------------------------------------
    # Pixel arrays
    # --------------------------------------------------------

    slices = []

    positions = []

    for item in datasets:

        ds = item["ds"]

        image = ds.pixel_array.astype(np.float32)

        # Apply DICOM rescale
        slope = float(getattr(ds, "RescaleSlope", 1))

        intercept = float(getattr(ds, "RescaleIntercept", 0))

        image = image * slope + intercept

        slices.append(image)

        positions.append(item["position_scalar"])

    volume = np.stack(slices, axis=0)

    return (volume, datasets, positions)


# ============================================================
# 6. NORMALIZATION FOR PIXEL COMPARISON
# ============================================================


def robust_normalize(volume):
    """
    Normalize each volume independently
    using the 1st and 99th percentiles.
    """

    volume = volume.astype(np.float32)

    low = np.percentile(volume, 1)

    high = np.percentile(volume, 99)

    if high <= low:

        return np.zeros_like(volume, dtype=np.float32)

    normalized = (volume - low) / (high - low)

    return np.clip(normalized, 0, 1)


# ============================================================
# 7. COMPARE TWO SERIES
# ============================================================


def compare_series(study_uid, series_uid_a, series_uid_b):

    print("\n" + "=" * 90)

    print("COMPARING:")

    print(f"Study: {study_uid}")

    print(f"A: {series_uid_a}")

    print(f"B: {series_uid_b}")

    volume_a, datasets_a, positions_a = read_series(study_uid, series_uid_a)

    volume_b, datasets_b, positions_b = read_series(study_uid, series_uid_b)

    result = {
        "StudyInstanceUID": study_uid,
        "SeriesA": series_uid_a,
        "SeriesB": series_uid_b,
        "SlicesA": int(volume_a.shape[0]),
        "SlicesB": int(volume_b.shape[0]),
        "ShapeA": str(volume_a.shape),
        "ShapeB": str(volume_b.shape),
    }

    # --------------------------------------------------------
    # Compare shapes
    # --------------------------------------------------------

    result["ExactShapeMatch"] = bool(volume_a.shape == volume_b.shape)

    # --------------------------------------------------------
    # Spatial position comparison
    # --------------------------------------------------------

    valid_a = [x for x in positions_a if x is not None]

    valid_b = [x for x in positions_b if x is not None]

    result["HasSpatialPositionsA"] = bool(len(valid_a) == len(positions_a))

    result["HasSpatialPositionsB"] = bool(len(valid_b) == len(positions_b))

    if len(valid_a) > 0 and len(valid_b) > 0:

        result["PositionMinA"] = float(min(valid_a))

        result["PositionMaxA"] = float(max(valid_a))

        result["PositionMinB"] = float(min(valid_b))

        result["PositionMaxB"] = float(max(valid_b))

        result["SamePositionCount"] = int(
            sum(any(abs(x - y) < 0.05 for y in valid_b) for x in valid_a)
        )

    else:

        result["SamePositionCount"] = None

    # --------------------------------------------------------
    # Pixel comparison
    # --------------------------------------------------------

    if volume_a.shape == volume_b.shape:

        norm_a = robust_normalize(volume_a)

        norm_b = robust_normalize(volume_b)

        absolute_diff = np.abs(norm_a - norm_b)

        mse = float(np.mean((norm_a - norm_b) ** 2))

        mae = float(np.mean(absolute_diff))

        result["PixelMSE"] = mse

        result["PixelMAE"] = mae

        result["PixelMaxAbsDifference"] = float(np.max(absolute_diff))

        # Percentage of nearly identical pixels
        result["PixelFractionAbsDiffBelow001"] = float(np.mean(absolute_diff < 0.01))

        # Correlation
        a_flat = norm_a.reshape(-1)

        b_flat = norm_b.reshape(-1)

        if np.std(a_flat) > 0 and np.std(b_flat) > 0:

            correlation = np.corrcoef(a_flat, b_flat)[0, 1]

            result["PixelCorrelation"] = float(correlation)

        else:

            result["PixelCorrelation"] = None

    else:

        result["PixelMSE"] = None

        result["PixelMAE"] = None

        result["PixelMaxAbsDifference"] = None

        result["PixelFractionAbsDiffBelow001"] = None

        result["PixelCorrelation"] = None

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if (
        result["ExactShapeMatch"]
        and result["PixelCorrelation"] is not None
        and result["PixelCorrelation"] > 0.995
        and result["PixelMAE"] < 0.01
    ):

        result["Classification"] = "LIKELY_EXACT_OR_NEAR_EXACT_DUPLICATE"

    elif (
        result["ExactShapeMatch"]
        and result["PixelCorrelation"] is not None
        and result["PixelCorrelation"] > 0.90
    ):

        result["Classification"] = "SAME_GEOMETRY_HIGHLY_SIMILAR_IMAGES"

    elif result["ExactShapeMatch"]:

        result["Classification"] = "SAME_SHAPE_BUT_DIFFERENT_PIXELS"

    else:

        result["Classification"] = "DIFFERENT_GEOMETRY"

    # --------------------------------------------------------
    # Contact sheets
    # --------------------------------------------------------

    def make_contact_sheet(volume, filename, title):

        n = volume.shape[0]

        n_cols = 5

        n_rows = math.ceil(n / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

        axes = np.asarray(axes).reshape(n_rows, n_cols)

        normalized = robust_normalize(volume)

        for i in range(n_rows * n_cols):

            ax = axes[i // n_cols, i % n_cols]

            if i < n:

                ax.imshow(normalized[i], cmap="gray", vmin=0, vmax=1)

                ax.set_title(f"Slice {i:02d}")

            ax.axis("off")

        plt.suptitle(title, fontsize=15)

        plt.tight_layout()

        plt.savefig(filename, dpi=150, bbox_inches="tight")

        plt.close()

    base = f"{study_uid[-8:]}_" f"{series_uid_a[-6:]}_" f"{series_uid_b[-6:]}"

    make_contact_sheet(
        volume_a, os.path.join(OUTPUT_DIR, f"{base}_A.png"), f"Series A\n{series_uid_a}"
    )

    make_contact_sheet(
        volume_b, os.path.join(OUTPUT_DIR, f"{base}_B.png"), f"Series B\n{series_uid_b}"
    )

    return result


# ============================================================
# 8. ANALYZE THE 5 REPEATED GROUPS
# ============================================================

comparison_results = []


print("\n" + "=" * 90)

print("IMAGE-LEVEL REPEATED-SERIES ANALYSIS")

print("=" * 90)


for pair_info in selected_pairs:

    study_uid = pair_info["StudyInstanceUID"]

    series_uids = pair_info["SeriesInstanceUIDs"]

    # Compare every pair if >2
    for a, b in itertools.combinations(series_uids, 2):

        try:

            result = compare_series(study_uid, a, b)

            comparison_results.append(result)

            print("\nRESULT:")

            print(json.dumps(result, indent=2, default=str))

        except Exception as e:

            comparison_results.append(
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesA": a,
                    "SeriesB": b,
                    "Classification": "ERROR",
                    "Error": str(e),
                }
            )


comparison_df = pd.DataFrame(comparison_results)


comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "repeated_series_pixel_comparison.csv"), index=False
)


# ============================================================
# 9. DUMMY AND BLANK STUDY SELECTION
# ============================================================

dummy_studies = (
    master[master["DescriptionType"] == "Dummy"]["StudyInstanceUID"]
    .drop_duplicates()
    .astype(str)
    .tolist()
)


blank_studies = (
    master[master["DescriptionType"] == "Blank"]["StudyInstanceUID"]
    .drop_duplicates()
    .astype(str)
    .tolist()
)


print("\nDummy studies:")

for uid in dummy_studies:
    print(" ", uid)


print("\nBlank-description studies:")

for uid in blank_studies:
    print(" ", uid)


# ============================================================
# 10. REPRESENTATIVE DUMMY + BLANK CONTACT SHEETS
# ============================================================


def make_study_contact_sheets(study_uid, prefix):

    study_rows = master[master["StudyInstanceUID"] == study_uid].copy()

    results = []

    for _, row in study_rows.iterrows():

        series_uid = str(row["SeriesInstanceUID"])

        try:

            volume, datasets, positions = read_series(study_uid, series_uid)

            # ------------------------------------------------
            # Select up to 9 representative slices
            # ------------------------------------------------

            n = volume.shape[0]

            if n <= 9:

                indices = list(range(n))

            else:

                indices = np.linspace(0, n - 1, 9).astype(int)

            normalized = robust_normalize(volume)

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))

            axes = axes.flatten()

            for i, ax in enumerate(axes):

                if i < len(indices):

                    slice_idx = indices[i]

                    ax.imshow(normalized[slice_idx], cmap="gray", vmin=0, vmax=1)

                    ax.set_title(f"Slice {slice_idx}")

                ax.axis("off")

            desc = row["SeriesDescription"]

            if pd.isna(desc):
                desc = "<BLANK>"

            title = (
                f"{prefix}\n"
                f"Plane={row['DICOM_AnatomicalPlane']} | "
                f"Fluid={row['Fluid_Sensitive']} | "
                f"FatSat={row['Fat_Suppression']}\n"
                f"Description={desc}"
            )

            plt.suptitle(title, fontsize=14)

            plt.tight_layout()

            filename = os.path.join(OUTPUT_DIR, f"{prefix}_" f"{series_uid[-8:]}.png")

            plt.savefig(filename, dpi=150, bbox_inches="tight")

            plt.close()

            results.append(
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Plane": row["DICOM_AnatomicalPlane"],
                    "FluidSensitive": row["Fluid_Sensitive"],
                    "FatSuppression": row["Fat_Suppression"],
                    "DescriptionType": row["DescriptionType"],
                    "SeriesDescription": desc,
                    "Manufacturer": row["Manufacturer"],
                    "Scanner": row["ManufacturerModelName"],
                    "Slices": int(volume.shape[0]),
                    "VolumeShape": str(volume.shape),
                    "OutputImage": filename,
                }
            )

        except Exception as e:

            results.append(
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Error": str(e),
                }
            )

    return results


# ------------------------------------------------------------
# Select first Dummy and first Blank study
# ------------------------------------------------------------

representative_results = []


if dummy_studies:

    representative_results.extend(
        make_study_contact_sheets(dummy_studies[0], "DUMMY_STUDY")
    )


if blank_studies:

    representative_results.extend(
        make_study_contact_sheets(blank_studies[0], "BLANK_STUDY")
    )


representative_df = pd.DataFrame(representative_results)


representative_df.to_csv(
    os.path.join(OUTPUT_DIR, "dummy_blank_representative_series.csv"), index=False
)


# ============================================================
# 11. FINAL JSON
# ============================================================

audit_summary = {
    "strict_repeated_groups": int(len(strict_repeats)),
    "repeated_series_comparisons": comparison_results,
    "dummy_studies": dummy_studies,
    "blank_studies": blank_studies,
    "comparison_classifications": (
        comparison_df["Classification"].value_counts().to_dict()
        if len(comparison_df) > 0
        else {}
    ),
}


with open(os.path.join(OUTPUT_DIR, "image_level_audit_summary.json"), "w") as f:

    json.dump(audit_summary, f, indent=4, default=str)


# ============================================================
# 12. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)

print("FINAL IMAGE-LEVEL AUDIT")

print("=" * 90)


print(f"\nRepeated groups tested: " f"{len(selected_pairs)}")


if len(comparison_df) > 0:

    print("\nClassification:")

    print(comparison_df["Classification"].value_counts().to_string())

    print("\nDetailed comparison:")

    # display(comparison_df)


print("\nRepresentative Dummy study:")

if dummy_studies:

    print(dummy_studies[0])


print("\nRepresentative Blank study:")

if blank_studies:

    print(blank_studies[0])


print("\nOutputs written to:")

print(OUTPUT_DIR)

print("\nDONE.")
