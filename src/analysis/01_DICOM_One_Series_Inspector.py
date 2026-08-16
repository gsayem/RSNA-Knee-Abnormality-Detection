# ============================================================
# 01_DICOM_One_Series_Inspector.py
# RSNA Knee MRI - DICOM Series Inspector
# ============================================================
#
# Purpose:
#   Inspect one DICOM series from the RSNA Knee MRI dataset.
#
# Outputs:
#   dicom_metadata.json
#   dicom_metadata.csv
#   dicom_contact_sheet.png
#   dicom_volume.npy
#
# Tested conceptually for Kaggle environments.
# ============================================================

import os
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

# try:
#     import pydicom
# except ImportError:
#     !pip install -q pydicom
#     import pydicom


# ============================================================
# 1. CONFIGURATION
# ============================================================

# CHANGE THIS PATH if necessary.
#
# Example Kaggle competition path:
# /kaggle/input/rsna-knee-abnormality-detection/train_series
#
# If your series is somewhere else, change this variable.

SERIES_ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series/1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260/1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772"
# SERIES_ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series/1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945"


# ------------------------------------------------------------
# OPTION 1:
# Automatically inspect the FIRST series containing DICOM files
# ------------------------------------------------------------

dicom_files = sorted(
    glob.glob(os.path.join(SERIES_ROOT, "**", "*.dcm"), recursive=True)
)

if len(dicom_files) == 0:
    raise FileNotFoundError(f"No DICOM files found under:\n{SERIES_ROOT}")

print(f"Total DICOM files discovered: {len(dicom_files):,}")


# ------------------------------------------------------------
# Select ONE series
# ------------------------------------------------------------


def get_series_key(path):
    """
    Extract:
        StudyInstanceUID
        SeriesInstanceUID

    from the directory structure:

    train_series/
        StudyUID/
            SeriesUID/
                image.dcm
    """

    series_dir = os.path.dirname(path)
    study_dir = os.path.dirname(series_dir)

    return (os.path.basename(study_dir), os.path.basename(series_dir))


series_groups = {}

for path in dicom_files:
    key = get_series_key(path)

    if key not in series_groups:
        series_groups[key] = []

    series_groups[key].append(path)


print(f"Total series discovered: {len(series_groups):,}")


# Show first few series
print("\nFirst 10 series:")
for i, (key, files) in enumerate(series_groups.items()):
    print(
        f"{i:02d} | " f"Study={key[0]} | " f"Series={key[1]} | " f"Slices={len(files)}"
    )

    if i >= 9:
        break


# ============================================================
# 2. SELECT THE SERIES TO INSPECT
# ============================================================

# By default, inspect the FIRST discovered series.
#
# If you want a specific series, replace this with:
#
# TARGET_STUDY_UID = "..."
# TARGET_SERIES_UID = "..."
#
# and uncomment the corresponding section below.

TARGET_STUDY_UID = None
TARGET_SERIES_UID = None


if TARGET_STUDY_UID is None or TARGET_SERIES_UID is None:

    selected_key = list(series_groups.keys())[0]

else:

    selected_key = (TARGET_STUDY_UID, TARGET_SERIES_UID)

    if selected_key not in series_groups:
        raise ValueError(f"Requested series not found:\n{selected_key}")


selected_files = series_groups[selected_key]

print("\n" + "=" * 80)
print("SELECTED SERIES")
print("=" * 80)

print("StudyInstanceUID :", selected_key[0])
print("SeriesInstanceUID:", selected_key[1])
print("DICOM slices     :", len(selected_files))


# ============================================================
# 3. READ DICOM FILES
# ============================================================

datasets = []

for path in selected_files:

    try:

        ds = pydicom.dcmread(path, force=True)

        datasets.append({"path": path, "dataset": ds})

    except Exception as e:

        print(f"WARNING: Could not read:\n{path}\n{e}")


print(f"\nSuccessfully read: {len(datasets)} DICOM files")


if len(datasets) == 0:
    raise RuntimeError("No DICOM files could be read.")


# ============================================================
# 4. DICOM VALUE HELPER
# ============================================================


def get_value(ds, attribute, default=None):

    value = getattr(ds, attribute, default)

    if value is None:
        return default

    try:
        if isinstance(value, pydicom.multival.MultiValue):
            return [float(x) if isinstance(x, (int, float)) else str(x) for x in value]

        if isinstance(value, (list, tuple)):
            return list(value)

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        return str(value)

    except Exception:
        return str(value)


# ============================================================
# 5. EXTRACT IMPORTANT METADATA
# ============================================================

records = []

for item in datasets:

    ds = item["dataset"]
    path = item["path"]

    image_position = getattr(ds, "ImagePositionPatient", None)

    image_orientation = getattr(ds, "ImageOrientationPatient", None)

    pixel_spacing = getattr(ds, "PixelSpacing", None)

    record = {
        "file": os.path.basename(path),
        # Identifiers
        "StudyInstanceUID": get_value(ds, "StudyInstanceUID"),
        "SeriesInstanceUID": get_value(ds, "SeriesInstanceUID"),
        "SOPInstanceUID": get_value(ds, "SOPInstanceUID"),
        # Instance ordering
        "InstanceNumber": get_value(ds, "InstanceNumber"),
        # Geometry
        "Rows": get_value(ds, "Rows"),
        "Columns": get_value(ds, "Columns"),
        "PixelSpacing": get_value(ds, "PixelSpacing"),
        "SliceThickness": get_value(ds, "SliceThickness"),
        "SpacingBetweenSlices": get_value(ds, "SpacingBetweenSlices"),
        "ImagePositionPatient": get_value(ds, "ImagePositionPatient"),
        "ImageOrientationPatient": get_value(ds, "ImageOrientationPatient"),
        # Acquisition
        "SeriesDescription": get_value(ds, "SeriesDescription"),
        "ProtocolName": get_value(ds, "ProtocolName"),
        "SequenceName": get_value(ds, "SequenceName"),
        "ScanningSequence": get_value(ds, "ScanningSequence"),
        "SequenceVariant": get_value(ds, "SequenceVariant"),
        "ScanOptions": get_value(ds, "ScanOptions"),
        # Scanner
        "Manufacturer": get_value(ds, "Manufacturer"),
        "ManufacturerModelName": get_value(ds, "ManufacturerModelName"),
        "MagneticFieldStrength": get_value(ds, "MagneticFieldStrength"),
        # Image encoding
        "PhotometricInterpretation": get_value(ds, "PhotometricInterpretation"),
        "SamplesPerPixel": get_value(ds, "SamplesPerPixel"),
        "BitsAllocated": get_value(ds, "BitsAllocated"),
        "BitsStored": get_value(ds, "BitsStored"),
        "HighBit": get_value(ds, "HighBit"),
        "PixelRepresentation": get_value(ds, "PixelRepresentation"),
        # Intensity transformation
        "RescaleIntercept": get_value(ds, "RescaleIntercept"),
        "RescaleSlope": get_value(ds, "RescaleSlope"),
        # Timing
        "EchoTime": get_value(ds, "EchoTime"),
        "RepetitionTime": get_value(ds, "RepetitionTime"),
        "FlipAngle": get_value(ds, "FlipAngle"),
    }

    records.append(record)


metadata_df = pd.DataFrame(records)


# ============================================================
# 6. DETERMINE SLICE ORDER
# ============================================================

print("\n" + "=" * 80)
print("SLICE ORDERING")
print("=" * 80)


def position_scalar(ds):
    """
    Calculate a scalar position along the slice direction.

    Uses ImageOrientationPatient to calculate the slice normal.
    """

    orientation = getattr(ds, "ImageOrientationPatient", None)

    position = getattr(ds, "ImagePositionPatient", None)

    if orientation is None or position is None:
        return None

    try:

        row = np.array(orientation[:3], dtype=float)

        col = np.array(orientation[3:], dtype=float)

        normal = np.cross(row, col)

        position = np.array(position, dtype=float)

        return float(np.dot(position, normal))

    except Exception:
        return None


positions = []

for item in datasets:

    scalar = position_scalar(item["dataset"])

    positions.append(scalar)


if all(x is not None for x in positions):

    ordered_indices = np.argsort(positions)

    datasets = [datasets[i] for i in ordered_indices]

    positions = [positions[i] for i in ordered_indices]

    print("Slice ordering: ImagePositionPatient + " "ImageOrientationPatient")

else:

    # Fall back to InstanceNumber
    print("WARNING: Spatial position unavailable.")

    print("Falling back to InstanceNumber.")

    datasets.sort(key=lambda x: int(getattr(x["dataset"], "InstanceNumber", 0)))

    positions = [position_scalar(x["dataset"]) for x in datasets]


# ============================================================
# 7. SLICE SPACING
# ============================================================

valid_positions = [x for x in positions if x is not None]

if len(valid_positions) > 1:

    position_diffs = np.diff(valid_positions)

    print("Slice position differences:")

    print(position_diffs)

    print("\nMedian slice spacing:", np.median(np.abs(position_diffs)))

else:

    position_diffs = []

    print("Cannot calculate spatial slice spacing.")


# ============================================================
# 8. DECODE PIXEL DATA
# ============================================================

print("\n" + "=" * 80)
print("PIXEL DATA")
print("=" * 80)


slices = []

for i, item in enumerate(datasets):

    ds = item["dataset"]

    try:

        pixel = ds.pixel_array.astype(np.float32)

        # Apply DICOM rescale if available
        slope = float(getattr(ds, "RescaleSlope", 1))

        intercept = float(getattr(ds, "RescaleIntercept", 0))

        pixel = pixel * slope + intercept

        slices.append(pixel)

    except Exception as e:

        print(f"Could not decode slice {i}: {e}")


if len(slices) == 0:

    raise RuntimeError("No pixel arrays could be decoded.")


# Verify dimensions
shapes = [x.shape for x in slices]

print("Unique slice dimensions:", set(shapes))


if len(set(shapes)) != 1:

    raise RuntimeError("Slices have different dimensions.")


volume = np.stack(slices, axis=0)


print("Volume shape:", volume.shape)

print("Volume dtype:", volume.dtype)

print("Minimum intensity:", float(np.min(volume)))

print("Maximum intensity:", float(np.max(volume)))

print("Mean intensity:", float(np.mean(volume)))

print("Median intensity:", float(np.median(volume)))

print("Standard deviation:", float(np.std(volume)))


# ============================================================
# 9. SAVE VOLUME
# ============================================================

np.save("dicom_volume.npy", volume)

print("\nSaved: dicom_volume.npy")


# ============================================================
# 10. CREATE METADATA CSV
# ============================================================

# Add calculated spatial position
metadata_df["SpatialPosition"] = positions


metadata_df.to_csv("dicom_metadata.csv", index=False)

print("Saved: dicom_metadata.csv")


# ============================================================
# 11. CREATE JSON SUMMARY
# ============================================================

first_ds = datasets[0]["dataset"]


summary = {
    "study_instance_uid": get_value(first_ds, "StudyInstanceUID"),
    "series_instance_uid": get_value(first_ds, "SeriesInstanceUID"),
    "number_of_slices": len(datasets),
    "volume_shape": list(volume.shape),
    "image_rows": int(volume.shape[1]),
    "image_columns": int(volume.shape[2]),
    "pixel_spacing": get_value(first_ds, "PixelSpacing"),
    "slice_thickness": get_value(first_ds, "SliceThickness"),
    "spacing_between_slices": get_value(first_ds, "SpacingBetweenSlices"),
    "median_computed_slice_spacing": (
        float(np.median(np.abs(position_diffs))) if len(position_diffs) > 0 else None
    ),
    "image_orientation": get_value(first_ds, "ImageOrientationPatient"),
    "series_description": get_value(first_ds, "SeriesDescription"),
    "protocol_name": get_value(first_ds, "ProtocolName"),
    "sequence_name": get_value(first_ds, "SequenceName"),
    "scanning_sequence": get_value(first_ds, "ScanningSequence"),
    "sequence_variant": get_value(first_ds, "SequenceVariant"),
    "manufacturer": get_value(first_ds, "Manufacturer"),
    "manufacturer_model": get_value(first_ds, "ManufacturerModelName"),
    "magnetic_field_strength": get_value(first_ds, "MagneticFieldStrength"),
    "photometric_interpretation": get_value(first_ds, "PhotometricInterpretation"),
    "bits_allocated": get_value(first_ds, "BitsAllocated"),
    "bits_stored": get_value(first_ds, "BitsStored"),
    "rescale_slope": get_value(first_ds, "RescaleSlope"),
    "rescale_intercept": get_value(first_ds, "RescaleIntercept"),
    "echo_time": get_value(first_ds, "EchoTime"),
    "repetition_time": get_value(first_ds, "RepetitionTime"),
    "flip_angle": get_value(first_ds, "FlipAngle"),
    "intensity_statistics": {
        "min": float(np.min(volume)),
        "max": float(np.max(volume)),
        "mean": float(np.mean(volume)),
        "median": float(np.median(volume)),
        "std": float(np.std(volume)),
        "p01": float(np.percentile(volume, 1)),
        "p05": float(np.percentile(volume, 5)),
        "p25": float(np.percentile(volume, 25)),
        "p75": float(np.percentile(volume, 75)),
        "p95": float(np.percentile(volume, 95)),
        "p99": float(np.percentile(volume, 99)),
    },
}


with open("dicom_metadata.json", "w") as f:

    json.dump(summary, f, indent=4, default=str)


print("Saved: dicom_metadata.json")


# ============================================================
# 12. CONTACT SHEET
# ============================================================

print("\nCreating contact sheet...")


n_slices = volume.shape[0]

# 5 columns
n_cols = 5

n_rows = math.ceil(n_slices / n_cols)


fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))


# Make axes always iterable
axes = np.array(axes).reshape(n_rows, n_cols)


for i in range(n_rows * n_cols):

    row = i // n_cols
    col = i % n_cols

    ax = axes[row, col]

    if i < n_slices:

        image = volume[i]

        # Robust display window
        low = np.percentile(image, 1)

        high = np.percentile(image, 99)

        ax.imshow(image, cmap="gray", vmin=low, vmax=high)

        ax.set_title(f"Slice {i:02d}")

    ax.axis("off")


plt.suptitle("Knee MRI DICOM Series - Contact Sheet", fontsize=16)

plt.tight_layout()

plt.savefig("dicom_contact_sheet.png", dpi=200, bbox_inches="tight")

plt.show()


print("Saved: dicom_contact_sheet.png")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"StudyInstanceUID : " f"{summary['study_instance_uid']}")

print(f"SeriesInstanceUID: " f"{summary['series_instance_uid']}")

print(f"Number of slices : " f"{summary['number_of_slices']}")

print(f"Volume shape     : " f"{summary['volume_shape']}")

print(f"Pixel spacing    : " f"{summary['pixel_spacing']}")

print(f"Slice thickness  : " f"{summary['slice_thickness']}")

print(f"Series description: " f"{summary['series_description']}")

print(f"Sequence name    : " f"{summary['sequence_name']}")

print(f"Manufacturer     : " f"{summary['manufacturer']}")

print(f"Magnetic field   : " f"{summary['magnetic_field_strength']}")

print("\nOutput files:")

print("  dicom_metadata.json")

print("  dicom_metadata.csv")

print("  dicom_contact_sheet.png")

print("  dicom_volume.npy")

print("\nDONE.")
