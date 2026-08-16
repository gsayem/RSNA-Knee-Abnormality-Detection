# ============================================================
# 08_DICOM_image_representation_analysis.py
# RSNA KNEE ABNORMALITY DETECTION
#
# IMAGE-LEVEL CHARACTERIZATION
#
# Scope:
#   58 gold-labeled studies
#   336 MRI series
#
# Investigates:
#   - Physical geometry
#   - Spatial coverage
#   - Intensity distributions
#   - Acquisition category
#   - Protocol archetype
#   - Scanner variation
#
# ============================================================

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = "/kaggle/input/competitions/" "rsna-knee-abnormality-detection"

MASTER_CSV = (
    "/kaggle/working/" "rsna_58_integrated_analysis/" "01_labeled_series_joined.csv"
)

PROTOCOL_CSV = (
    "/kaggle/working/" "rsna_protocol_archetype_analysis/" "01_protocol_archetypes.csv"
)

TRAIN_SERIES_ROOT = os.path.join(DATA_ROOT, "train_series")

OUTPUT_DIR = "/kaggle/working/" "rsna_image_representation_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Number of slices used for intensity analysis
INTENSITY_SAMPLE_SLICES = 7


# ============================================================
# 2. IMPORT DICOM LIBRARY
# ============================================================

# try:

#     import pydicom

# except ImportError:

#     !pip install -q pydicom

#     import pydicom


# ============================================================
# 3. LOAD EXISTING METADATA
# ============================================================

print("\n")
print("=" * 90)
print("LOADING METADATA")
print("=" * 90)


master = pd.read_csv(MASTER_CSV)

protocol = pd.read_csv(PROTOCOL_CSV)


if len(master) == 0:

    raise RuntimeError(
        "The master metadata CSV contains zero rows.\n" f"File: {MASTER_CSV}"
    )


print(f"Master series records: {len(master):,}")

print(f"Master studies: " f"{master['StudyInstanceUID'].nunique():,}")


# ------------------------------------------------------------
# Normalize IDs
# ------------------------------------------------------------

master["StudyInstanceUID"] = master["StudyInstanceUID"].astype(str)

master["SeriesInstanceUID"] = master["SeriesInstanceUID"].astype(str)


protocol["StudyInstanceUID"] = protocol["StudyInstanceUID"].astype(str)


# ============================================================
# 4. ADD PROTOCOL ARCHETYPE INFORMATION
# ============================================================

protocol_columns = [
    "StudyInstanceUID",
    "ProtocolArchetype",
    "ProtocolArchetypeDescription",
    "PrimaryManufacturerFamily",
    "PrimaryFieldStrength",
]


protocol_columns = [c for c in protocol_columns if c in protocol.columns]


# Avoid _x / _y columns
for col in protocol_columns:

    if col != "StudyInstanceUID" and col in master.columns:

        master = master.drop(columns=[col])


master = master.merge(
    protocol[protocol_columns],
    on="StudyInstanceUID",
    how="left",
    validate="many_to_one",
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================


def safe_float(value):

    try:

        if value is None:
            return np.nan

        return float(value)

    except Exception:

        return np.nan


def parse_pixel_spacing(value):

    try:

        if value is None:
            return np.nan, np.nan

        return (float(value[0]), float(value[1]))

    except Exception:

        return np.nan, np.nan


def get_slice_position(ds):

    orientation = getattr(ds, "ImageOrientationPatient", None)

    position = getattr(ds, "ImagePositionPatient", None)

    if orientation is None or position is None:

        return np.nan

    try:

        row_cosines = np.asarray(orientation[:3], dtype=np.float64)

        column_cosines = np.asarray(orientation[3:], dtype=np.float64)

        normal = np.cross(row_cosines, column_cosines)

        position = np.asarray(position, dtype=np.float64)

        return float(np.dot(position, normal))

    except Exception:

        return np.nan


def robust_statistics(values):

    values = np.asarray(values, dtype=np.float32)

    values = values[np.isfinite(values)]

    if len(values) == 0:

        return {
            "Min": np.nan,
            "Max": np.nan,
            "Mean": np.nan,
            "Std": np.nan,
            "Median": np.nan,
            "P01": np.nan,
            "P05": np.nan,
            "P25": np.nan,
            "P75": np.nan,
            "P95": np.nan,
            "P99": np.nan,
        }

    return {
        "Min": float(np.min(values)),
        "Max": float(np.max(values)),
        "Mean": float(np.mean(values)),
        "Std": float(np.std(values)),
        "Median": float(np.median(values)),
        "P01": float(np.percentile(values, 1)),
        "P05": float(np.percentile(values, 5)),
        "P25": float(np.percentile(values, 25)),
        "P75": float(np.percentile(values, 75)),
        "P95": float(np.percentile(values, 95)),
        "P99": float(np.percentile(values, 99)),
    }


# ============================================================
# 6. ANALYZE ONE SERIES
# ============================================================


def analyze_series(study_uid, series_uid):

    series_dir = os.path.join(TRAIN_SERIES_ROOT, study_uid, series_uid)

    # --------------------------------------------------------
    # Find DICOM files
    # --------------------------------------------------------

    dicom_paths = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))

    if len(dicom_paths) == 0:

        raise FileNotFoundError("No DICOM files found:\n" f"{series_dir}")

    # --------------------------------------------------------
    # Read headers
    # --------------------------------------------------------

    headers = []

    for path in dicom_paths:

        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

        headers.append(
            {
                "Path": path,
                "Position": get_slice_position(ds),
                "InstanceNumber": getattr(ds, "InstanceNumber", np.nan),
            }
        )

    # --------------------------------------------------------
    # Spatial ordering
    # --------------------------------------------------------

    spatial_order_available = all(np.isfinite(item["Position"]) for item in headers)

    if spatial_order_available:

        headers.sort(key=lambda x: x["Position"])

        ordering_method = "ImagePositionPatient"

    else:

        headers.sort(
            key=lambda x: (
                safe_float(x["InstanceNumber"])
                if np.isfinite(safe_float(x["InstanceNumber"]))
                else 0
            )
        )

        ordering_method = "InstanceNumber"

    # --------------------------------------------------------
    # First slice metadata
    # --------------------------------------------------------

    first_ds = pydicom.dcmread(headers[0]["Path"], stop_before_pixels=True, force=True)

    rows = int(getattr(first_ds, "Rows", 0))

    columns = int(getattr(first_ds, "Columns", 0))

    pixel_spacing_y, pixel_spacing_x = parse_pixel_spacing(
        getattr(first_ds, "PixelSpacing", None)
    )

    slice_thickness = safe_float(getattr(first_ds, "SliceThickness", np.nan))

    dicom_spacing = safe_float(getattr(first_ds, "SpacingBetweenSlices", np.nan))

    # --------------------------------------------------------
    # Spatial extent
    # --------------------------------------------------------

    positions = np.asarray([item["Position"] for item in headers], dtype=np.float64)

    valid_positions = positions[np.isfinite(positions)]

    if len(valid_positions) >= 2:

        ordered_positions = np.sort(valid_positions)

        spacing_diffs = np.diff(ordered_positions)

        spacing_diffs = np.abs(spacing_diffs)

        computed_spacing = float(np.median(spacing_diffs))

        min_spacing = float(np.min(spacing_diffs))

        max_spacing = float(np.max(spacing_diffs))

        spatial_extent_z = float(
            (np.max(valid_positions) - np.min(valid_positions))
            + (slice_thickness if np.isfinite(slice_thickness) else 0)
        )

    else:

        computed_spacing = np.nan
        min_spacing = np.nan
        max_spacing = np.nan
        spatial_extent_z = np.nan

    # --------------------------------------------------------
    # Physical FOV
    # --------------------------------------------------------

    fov_y = rows * pixel_spacing_y if np.isfinite(pixel_spacing_y) else np.nan

    fov_x = columns * pixel_spacing_x if np.isfinite(pixel_spacing_x) else np.nan

    # --------------------------------------------------------
    # Physical volume approximation
    # --------------------------------------------------------

    if np.isfinite(fov_x) and np.isfinite(fov_y) and np.isfinite(spatial_extent_z):

        physical_volume = fov_x * fov_y * spatial_extent_z

    else:

        physical_volume = np.nan

    # ========================================================
    # INTENSITY ANALYSIS
    # ========================================================

    n_slices = len(headers)

    sample_count = min(INTENSITY_SAMPLE_SLICES, n_slices)

    sample_indices = np.linspace(0, n_slices - 1, sample_count).astype(int)

    sampled_pixels = []

    intensity_error = None

    try:

        for index in sample_indices:

            ds = pydicom.dcmread(headers[index]["Path"], force=True)

            image = ds.pixel_array.astype(np.float32)

            slope = safe_float(getattr(ds, "RescaleSlope", 1))

            intercept = safe_float(getattr(ds, "RescaleIntercept", 0))

            if not np.isfinite(slope):

                slope = 1.0

            if not np.isfinite(intercept):

                intercept = 0.0

            image = image * slope + intercept

            image = image[np.isfinite(image)]

            if len(image) > 0:

                sampled_pixels.append(image)

    except Exception as e:

        intensity_error = repr(e)

    # --------------------------------------------------------
    # Intensity statistics
    # --------------------------------------------------------

    if sampled_pixels:

        all_pixels = np.concatenate(sampled_pixels)

        stats = robust_statistics(all_pixels)

        # ----------------------------------------------------
        # Crude foreground estimate
        # ----------------------------------------------------

        foreground_values = []

        for image in sampled_pixels:

            threshold = np.percentile(image, 1)

            foreground = image[image > threshold]

            if len(foreground) > 0:

                foreground_values.append(foreground)

        if foreground_values:

            foreground = np.concatenate(foreground_values)

            foreground_stats = robust_statistics(foreground)

            foreground_fraction = len(foreground) / len(all_pixels)

        else:

            foreground_stats = robust_statistics([])

            foreground_fraction = np.nan

        if np.isfinite(stats["P95"]) and np.isfinite(stats["P05"]):

            robust_range = stats["P95"] - stats["P05"]

        else:

            robust_range = np.nan

        if np.isfinite(stats["Median"]) and stats["Median"] != 0:

            robust_cv = robust_range / abs(stats["Median"])

        else:

            robust_cv = np.nan

    else:

        stats = robust_statistics([])

        foreground_stats = robust_statistics([])

        foreground_fraction = np.nan
        robust_range = np.nan
        robust_cv = np.nan

        if intensity_error is None:

            intensity_error = "No pixel samples decoded"

    # ========================================================
    # RETURN ALL MEASUREMENTS IN ONE DICTIONARY
    # ========================================================

    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "NumberOfSlices": n_slices,
        "Rows": rows,
        "Columns": columns,
        "PixelSpacingY_mm": pixel_spacing_y,
        "PixelSpacingX_mm": pixel_spacing_x,
        "SliceThickness_mm": slice_thickness,
        "SpacingBetweenSlices_DICOM_mm": dicom_spacing,
        "ComputedSliceSpacing_mm": computed_spacing,
        "MinSliceSpacing_mm": min_spacing,
        "MaxSliceSpacing_mm": max_spacing,
        "PhysicalFOV_Y_mm": fov_y,
        "PhysicalFOV_X_mm": fov_x,
        "SpatialExtent_Z_mm": spatial_extent_z,
        "PhysicalVolume_mm3": physical_volume,
        "SliceOrderingMethod": ordering_method,
        "SampledSliceCount": len(sampled_pixels),
        "SampledSliceIndices": "|".join(str(x) for x in sample_indices),
        "IntensityDecodeError": intensity_error,
        "IntensityMin": stats["Min"],
        "IntensityMax": stats["Max"],
        "IntensityMean": stats["Mean"],
        "IntensityStd": stats["Std"],
        "IntensityMedian": stats["Median"],
        "IntensityP01": stats["P01"],
        "IntensityP05": stats["P05"],
        "IntensityP25": stats["P25"],
        "IntensityP75": stats["P75"],
        "IntensityP95": stats["P95"],
        "IntensityP99": stats["P99"],
        "ForegroundFraction": foreground_fraction,
        "ForegroundMedian": foreground_stats["Median"],
        "ForegroundP05": foreground_stats["P05"],
        "ForegroundP95": foreground_stats["P95"],
        "RobustIntensityRange": robust_range,
        "RobustCV": robust_cv,
    }


# ============================================================
# 7. PROCESS ALL SERIES
# ============================================================

print("\n")
print("=" * 90)
print("PROCESSING 336 SERIES")
print("=" * 90)


records = []
failures = []


total = len(master)


for counter, (_, row) in enumerate(master.iterrows(), start=1):

    study_uid = str(row["StudyInstanceUID"])

    series_uid = str(row["SeriesInstanceUID"])

    try:

        record = analyze_series(study_uid, series_uid)

        # ----------------------------------------------------
        # Add authoritative metadata from master
        # ----------------------------------------------------

        metadata_columns = [
            "TrainSeries_AnatomicalPlane",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "DescriptionType",
            "SeriesDescription",
            "Manufacturer",
            "ManufacturerModelName",
            "MagneticFieldStrength",
            "ProtocolArchetype",
            "ProtocolArchetypeDescription",
            "PrimaryManufacturerFamily",
            "PrimaryFieldStrength",
        ]

        for column in metadata_columns:

            if column in row.index:

                record[column] = row[column]

        # ----------------------------------------------------
        # Acquisition category
        # ----------------------------------------------------

        plane = str(row["TrainSeries_AnatomicalPlane"])

        fluid = int(row["Fluid_Sensitive"])

        fat = int(row["Fat_Suppression"])

        record["AcquisitionCategory"] = (
            f"{plane}_"
            f"{'Fluid' if fluid == 1 else 'NonFluid'}_"
            f"{'FS' if fat == 1 else 'NonFS'}"
        )

        records.append(record)

    except Exception as e:

        failures.append(
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Error": repr(e),
            }
        )

    if counter % 25 == 0 or counter == total:

        print(
            f"{counter:3d}/{total} | "
            f"success={len(records):3d} | "
            f"failed={len(failures):3d}"
        )


# ============================================================
# 8. BUILD ONE SINGLE DATAFRAME
# ============================================================

print("\n")
print("=" * 90)
print("BUILDING FINAL IMAGE PROFILE")
print("=" * 90)


if len(records) == 0:

    print("\nNO SERIES WERE SUCCESSFULLY PROCESSED.")

    print("\nFirst failures:")

    for item in failures[:10]:

        print("\nStudy:", item["StudyInstanceUID"])

        print("Series:", item["SeriesInstanceUID"])

        print("Error:", item["Error"])

    raise RuntimeError("All series failed. " "See the failure messages above.")


profile_df = pd.DataFrame(records)


print(f"Successfully analyzed: " f"{len(profile_df):,}")

print(f"Failed completely: " f"{len(failures):,}")


# ------------------------------------------------------------
# Save failed-series report
# ------------------------------------------------------------

failed_df = pd.DataFrame(failures)

failed_df.to_csv(os.path.join(OUTPUT_DIR, "failed_series.csv"), index=False)


# ------------------------------------------------------------
# Save complete series profile
# ------------------------------------------------------------

profile_df.to_csv(os.path.join(OUTPUT_DIR, "03_series_image_profile.csv"), index=False)


# ============================================================
# 9. GEOMETRY / INTENSITY SUMMARIES
# ============================================================

print("\n")
print("=" * 90)
print("GENERATING SUMMARY TABLES")
print("=" * 90)


numeric_columns = [
    "NumberOfSlices",
    "PixelSpacingX_mm",
    "PixelSpacingY_mm",
    "SliceThickness_mm",
    "SpacingBetweenSlices_DICOM_mm",
    "ComputedSliceSpacing_mm",
    "PhysicalFOV_X_mm",
    "PhysicalFOV_Y_mm",
    "SpatialExtent_Z_mm",
    "PhysicalVolume_mm3",
    "IntensityMedian",
    "IntensityP05",
    "IntensityP95",
    "IntensityP99",
    "ForegroundFraction",
    "RobustIntensityRange",
    "RobustCV",
]


numeric_columns = [c for c in numeric_columns if c in profile_df.columns]


# ------------------------------------------------------------
# By acquisition category
# ------------------------------------------------------------

by_acquisition = profile_df.groupby("AcquisitionCategory")[numeric_columns].agg(
    ["count", "mean", "median", "std", "min", "max"]
)


by_acquisition.columns = [
    "_".join(str(x) for x in column) for column in by_acquisition.columns
]


by_acquisition = by_acquisition.reset_index()


by_acquisition.to_csv(
    os.path.join(OUTPUT_DIR, "04_by_acquisition_category.csv"), index=False
)


# ------------------------------------------------------------
# By scanner family
# ------------------------------------------------------------

by_scanner = profile_df.groupby("PrimaryManufacturerFamily")[numeric_columns].agg(
    ["count", "mean", "median", "std", "min", "max"]
)


by_scanner.columns = ["_".join(str(x) for x in column) for column in by_scanner.columns]


by_scanner = by_scanner.reset_index()


by_scanner.to_csv(os.path.join(OUTPUT_DIR, "05_by_scanner_family.csv"), index=False)


# ------------------------------------------------------------
# By protocol archetype
# ------------------------------------------------------------

by_protocol = profile_df.groupby("ProtocolArchetype")[numeric_columns].agg(
    ["count", "mean", "median", "std", "min", "max"]
)


by_protocol.columns = [
    "_".join(str(x) for x in column) for column in by_protocol.columns
]


by_protocol = by_protocol.reset_index()


by_protocol.to_csv(
    os.path.join(OUTPUT_DIR, "06_by_protocol_archetype.csv"), index=False
)


# ============================================================
# 10. DATASET SUMMARY JSON
# ============================================================

summary = {
    "series": {
        "master_records": int(len(master)),
        "successfully_analyzed": int(len(profile_df)),
        "failed": int(len(failures)),
    },
    "studies": int(profile_df["StudyInstanceUID"].nunique()),
    "geometry": {
        "PixelSpacingX_mm": {
            "median": float(profile_df["PixelSpacingX_mm"].median()),
            "min": float(profile_df["PixelSpacingX_mm"].min()),
            "max": float(profile_df["PixelSpacingX_mm"].max()),
        },
        "PixelSpacingY_mm": {
            "median": float(profile_df["PixelSpacingY_mm"].median()),
            "min": float(profile_df["PixelSpacingY_mm"].min()),
            "max": float(profile_df["PixelSpacingY_mm"].max()),
        },
        "ComputedSliceSpacing_mm": {
            "median": float(profile_df["ComputedSliceSpacing_mm"].median()),
            "min": float(profile_df["ComputedSliceSpacing_mm"].min()),
            "max": float(profile_df["ComputedSliceSpacing_mm"].max()),
        },
    },
    "intensity": {
        "IntensityMedian": {
            "median": float(profile_df["IntensityMedian"].median()),
            "min": float(profile_df["IntensityMedian"].min()),
            "max": float(profile_df["IntensityMedian"].max()),
        },
        "RobustCV": {
            "median": float(profile_df["RobustCV"].median()),
            "min": float(profile_df["RobustCV"].min()),
            "max": float(profile_df["RobustCV"].max()),
        },
    },
    "acquisition_categories": profile_df["AcquisitionCategory"]
    .value_counts()
    .to_dict(),
    "protocol_archetypes": profile_df["ProtocolArchetype"].value_counts().to_dict(),
    "intensity_decode_errors": int(profile_df["IntensityDecodeError"].notna().sum()),
}


with open(os.path.join(OUTPUT_DIR, "07_image_representation_summary.json"), "w") as f:

    json.dump(summary, f, indent=4, default=str)


# ============================================================
# 11. REPRESENTATIVE / EXTREME SERIES
# ============================================================

# ------------------------------------------------------------
# Highest robust intensity variation
# ------------------------------------------------------------

extreme_intensity = profile_df.sort_values("RobustCV", ascending=False).head(20)


extreme_intensity.to_csv(
    os.path.join(OUTPUT_DIR, "08_extreme_intensity_series.csv"), index=False
)


# ------------------------------------------------------------
# Largest / smallest slice spacing
# ------------------------------------------------------------

extreme_geometry = profile_df.sort_values("ComputedSliceSpacing_mm")


extreme_geometry[
    [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "AcquisitionCategory",
        "ProtocolArchetype",
        "NumberOfSlices",
        "PixelSpacingX_mm",
        "PixelSpacingY_mm",
        "ComputedSliceSpacing_mm",
        "PhysicalFOV_X_mm",
        "PhysicalFOV_Y_mm",
        "SpatialExtent_Z_mm",
    ]
].head(20).to_csv(
    os.path.join(OUTPUT_DIR, "09_extreme_geometry_series.csv"), index=False
)


# ============================================================
# 12. PLOTS
# ============================================================

print("\n")
print("=" * 90)
print("GENERATING PLOTS")
print("=" * 90)


# ------------------------------------------------------------
# Geometry plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))


for category in sorted(profile_df["AcquisitionCategory"].dropna().unique()):

    subset = profile_df[profile_df["AcquisitionCategory"] == category]

    plt.scatter(
        subset["PixelSpacingX_mm"],
        subset["ComputedSliceSpacing_mm"],
        alpha=0.7,
        label=category,
    )


plt.xlabel("In-plane pixel spacing (mm)")

plt.ylabel("Computed slice spacing (mm)")

plt.title("MRI Physical Geometry by Acquisition Category")

plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "10_geometry_summary.png"), dpi=200, bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# Intensity plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))


for category in sorted(profile_df["AcquisitionCategory"].dropna().unique()):

    subset = profile_df[profile_df["AcquisitionCategory"] == category]

    plt.scatter(
        subset["IntensityP05"], subset["IntensityP95"], alpha=0.7, label=category
    )


plt.xlabel("Intensity P05")

plt.ylabel("Intensity P95")

plt.title("MRI Intensity Distribution by Acquisition Category")

plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "11_intensity_summary.png"), dpi=200, bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 90)
print("IMAGE REPRESENTATION ANALYSIS COMPLETE")
print("=" * 90)


print(f"\nMaster records           : {len(master):,}")

print(f"Successfully analyzed   : {len(profile_df):,}")

print(f"Completely failed        : {len(failures):,}")

print(f"Studies represented      : " f"{profile_df['StudyInstanceUID'].nunique():,}")


print("\nMedian geometry:")

print(f"Pixel spacing X : " f"{profile_df['PixelSpacingX_mm'].median():.4f} mm")

print(f"Pixel spacing Y : " f"{profile_df['PixelSpacingY_mm'].median():.4f} mm")

print(f"Slice spacing   : " f"{profile_df['ComputedSliceSpacing_mm'].median():.4f} mm")


print("\nMedian intensity:")

print(f"Median intensity : " f"{profile_df['IntensityMedian'].median():.4f}")

print(f"Median robust CV : " f"{profile_df['RobustCV'].median():.4f}")


print("\nOutput directory:")

print(os.path.abspath(OUTPUT_DIR))


print("\nDONE.")
