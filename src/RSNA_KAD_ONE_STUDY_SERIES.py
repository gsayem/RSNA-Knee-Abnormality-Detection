# ============================================================
# RSNA Knee Abnormality Detection
# COMPLETE STUDY INSPECTOR
#
# One Study -> Multiple Series -> DICOM slices
#
# Outputs:
#
#   study_metadata.json
#   study_series_metadata.csv
#
#   series_01_contact_sheet.png
#   series_02_contact_sheet.png
#   series_03_contact_sheet.png
#   series_04_contact_sheet.png
#   series_05_contact_sheet.png
#
# ============================================================

import os
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pydicom
except ImportError:
    !pip install -q pydicom
    import pydicom


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

TRAIN_CSV = os.path.join(
    DATA_ROOT,
    "train.csv"
)

TRAIN_SERIES_ROOT = os.path.join(
    DATA_ROOT,
    "train_series"
)

# ------------------------------------------------------------
# The specific study we selected
# ------------------------------------------------------------

STUDY_UID = (
    "1.2.826.0.1.3680043.8.498."
    "32321830776739689645700555055955725945"
)


# ============================================================
# 2. FIND THE STUDY
# ============================================================

STUDY_DIR = os.path.join(
    TRAIN_SERIES_ROOT,
    STUDY_UID
)

if not os.path.isdir(STUDY_DIR):

    raise FileNotFoundError(
        f"Study directory not found:\n{STUDY_DIR}"
    )


print("=" * 80)
print("RSNA KNEE MRI - COMPLETE STUDY INSPECTION")
print("=" * 80)

print(
    f"\nStudyInstanceUID:\n{STUDY_UID}"
)

print(
    f"\nStudy directory:\n{STUDY_DIR}"
)


# ============================================================
# 3. LOAD TRAIN LABELS
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

study_rows = train_df[
    train_df["StudyInstanceUID"].astype(str)
    == STUDY_UID
]

if len(study_rows) == 0:

    raise ValueError(
        "Study was not found in train.csv"
    )

if len(study_rows) > 1:

    raise ValueError(
        "Multiple rows found for this StudyInstanceUID"
    )


study_row = study_rows.iloc[0]


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
    "Fracture"
]


labels = {}

for label in LABEL_COLUMNS:

    value = study_row[label]

    if pd.isna(value):
        labels[label] = None
    else:
        labels[label] = int(value)


print("\n" + "=" * 80)
print("GROUND-TRUTH LABELS")
print("=" * 80)

for label, value in labels.items():

    print(
        f"{label:20s}: {value}"
    )


# ============================================================
# 4. REPORT
# ============================================================

report = study_row.get(
    "Report",
    None
)

if pd.isna(report):

    report = None

else:

    report = str(report)


# ============================================================
# 5. FIND ALL SERIES
# ============================================================

series_dirs = [
    path
    for path in glob.glob(
        os.path.join(
            STUDY_DIR,
            "*"
        )
    )
    if os.path.isdir(path)
]


series_dirs = sorted(
    series_dirs
)


print("\n" + "=" * 80)
print("SERIES")
print("=" * 80)

print(
    f"Number of series: {len(series_dirs)}"
)


if len(series_dirs) == 0:

    raise RuntimeError(
        "No series directories found."
    )


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def safe_value(ds, attribute, default=None):

    value = getattr(
        ds,
        attribute,
        default
    )

    if value is None:
        return default

    try:

        if isinstance(
            value,
            pydicom.multival.MultiValue
        ):

            result = []

            for x in value:

                try:
                    result.append(float(x))

                except Exception:
                    result.append(str(x))

            return result


        if isinstance(
            value,
            (list, tuple)
        ):

            result = []

            for x in value:

                try:
                    result.append(float(x))

                except Exception:
                    result.append(str(x))

            return result


        if isinstance(
            value,
            np.integer
        ):

            return int(value)


        if isinstance(
            value,
            np.floating
        ):

            return float(value)


        return str(value)

    except Exception:

        return str(value)


def get_slice_position(ds):

    """
    Calculate the scalar position of a slice.

    Uses:

        ImageOrientationPatient
        ImagePositionPatient

    to calculate the position along
    the slice normal.
    """

    orientation = getattr(
        ds,
        "ImageOrientationPatient",
        None
    )

    position = getattr(
        ds,
        "ImagePositionPatient",
        None
    )

    if orientation is None:
        return None

    if position is None:
        return None

    try:

        row_cosines = np.array(
            orientation[:3],
            dtype=float
        )

        column_cosines = np.array(
            orientation[3:],
            dtype=float
        )

        normal = np.cross(
            row_cosines,
            column_cosines
        )

        position = np.array(
            position,
            dtype=float
        )

        return float(
            np.dot(
                position,
                normal
            )
        )

    except Exception:

        return None


def read_series(series_dir):

    """
    Read one complete DICOM series.
    """

    dicom_paths = sorted(
        glob.glob(
            os.path.join(
                series_dir,
                "*.dcm"
            )
        )
    )

    if len(dicom_paths) == 0:

        raise RuntimeError(
            f"No DICOM files found:\n{series_dir}"
        )


    datasets = []

    for path in dicom_paths:

        try:

            ds = pydicom.dcmread(
                path,
                force=True
            )

            datasets.append(
                {
                    "path": path,
                    "dataset": ds
                }
            )

        except Exception as e:

            print(
                f"WARNING: Failed to read {path}"
            )

            print(e)


    if len(datasets) == 0:

        raise RuntimeError(
            f"No DICOM files could be read:\n"
            f"{series_dir}"
        )


    # --------------------------------------------------------
    # Calculate spatial positions
    # --------------------------------------------------------

    for item in datasets:

        item["spatial_position"] = (
            get_slice_position(
                item["dataset"]
            )
        )


    # --------------------------------------------------------
    # Sort spatially when possible
    # --------------------------------------------------------

    if all(
        item["spatial_position"] is not None
        for item in datasets
    ):

        datasets.sort(
            key=lambda x:
            x["spatial_position"]
        )

        ordering_method = (
            "ImagePositionPatient + "
            "ImageOrientationPatient"
        )

    else:

        # Fallback
        datasets.sort(
            key=lambda x:
            int(
                getattr(
                    x["dataset"],
                    "InstanceNumber",
                    0
                )
            )
        )

        ordering_method = (
            "InstanceNumber fallback"
        )


    # --------------------------------------------------------
    # Decode pixel data
    # --------------------------------------------------------

    pixels = []

    for item in datasets:

        ds = item["dataset"]

        try:

            image = (
                ds.pixel_array
                .astype(np.float32)
            )

            slope = float(
                getattr(
                    ds,
                    "RescaleSlope",
                    1
                )
            )

            intercept = float(
                getattr(
                    ds,
                    "RescaleIntercept",
                    0
                )
            )

            image = (
                image * slope
                + intercept
            )

            pixels.append(
                image
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not decode "
                f"{item['path']}\n{e}"
            )


    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    shapes = [
        image.shape
        for image in pixels
    ]

    if len(set(shapes)) != 1:

        raise RuntimeError(
            "Inconsistent slice dimensions:\n"
            f"{set(shapes)}"
        )


    volume = np.stack(
        pixels,
        axis=0
    )


    # --------------------------------------------------------
    # Calculate slice spacing
    # --------------------------------------------------------

    positions = [
        item["spatial_position"]
        for item in datasets
    ]

    valid_positions = [
        x for x in positions
        if x is not None
    ]


    if len(valid_positions) > 1:

        position_diffs = np.diff(
            valid_positions
        )

        median_spacing = float(
            np.median(
                np.abs(
                    position_diffs
                )
            )
        )

        min_spacing = float(
            np.min(
                np.abs(
                    position_diffs
                )
            )
        )

        max_spacing = float(
            np.max(
                np.abs(
                    position_diffs
                )
            )
        )

    else:

        median_spacing = None
        min_spacing = None
        max_spacing = None


    # --------------------------------------------------------
    # Metadata from first slice
    # --------------------------------------------------------

    first_ds = datasets[0]["dataset"]


    metadata = {

        "SeriesInstanceUID":
            safe_value(
                first_ds,
                "SeriesInstanceUID"
            ),

        "StudyInstanceUID":
            safe_value(
                first_ds,
                "StudyInstanceUID"
            ),

        "NumberOfSlices":
            len(datasets),

        "VolumeShape":
            list(volume.shape),

        "Rows":
            safe_value(
                first_ds,
                "Rows"
            ),

        "Columns":
            safe_value(
                first_ds,
                "Columns"
            ),

        "PixelSpacing":
            safe_value(
                first_ds,
                "PixelSpacing"
            ),

        "SliceThickness":
            safe_value(
                first_ds,
                "SliceThickness"
            ),

        "SpacingBetweenSlices":
            safe_value(
                first_ds,
                "SpacingBetweenSlices"
            ),

        "ComputedMedianSliceSpacing":
            median_spacing,

        "ComputedMinSliceSpacing":
            min_spacing,

        "ComputedMaxSliceSpacing":
            max_spacing,

        "ImageOrientationPatient":
            safe_value(
                first_ds,
                "ImageOrientationPatient"
            ),

        "SeriesDescription":
            safe_value(
                first_ds,
                "SeriesDescription"
            ),

        "ProtocolName":
            safe_value(
                first_ds,
                "ProtocolName"
            ),

        "SequenceName":
            safe_value(
                first_ds,
                "SequenceName"
            ),

        "ScanningSequence":
            safe_value(
                first_ds,
                "ScanningSequence"
            ),

        "SequenceVariant":
            safe_value(
                first_ds,
                "SequenceVariant"
            ),

        "Manufacturer":
            safe_value(
                first_ds,
                "Manufacturer"
            ),

        "ManufacturerModelName":
            safe_value(
                first_ds,
                "ManufacturerModelName"
            ),

        "MagneticFieldStrength":
            safe_value(
                first_ds,
                "MagneticFieldStrength"
            ),

        "PhotometricInterpretation":
            safe_value(
                first_ds,
                "PhotometricInterpretation"
            ),

        "BitsAllocated":
            safe_value(
                first_ds,
                "BitsAllocated"
            ),

        "BitsStored":
            safe_value(
                first_ds,
                "BitsStored"
            ),

        "RescaleSlope":
            safe_value(
                first_ds,
                "RescaleSlope"
            ),

        "RescaleIntercept":
            safe_value(
                first_ds,
                "RescaleIntercept"
            ),

        "EchoTime":
            safe_value(
                first_ds,
                "EchoTime"
            ),

        "RepetitionTime":
            safe_value(
                first_ds,
                "RepetitionTime"
            ),

        "FlipAngle":
            safe_value(
                first_ds,
                "FlipAngle"
            ),

        "SliceOrderingMethod":
            ordering_method,

        "IntensityStatistics": {

            "min":
                float(np.min(volume)),

            "max":
                float(np.max(volume)),

            "mean":
                float(np.mean(volume)),

            "median":
                float(np.median(volume)),

            "std":
                float(np.std(volume)),

            "p01":
                float(
                    np.percentile(
                        volume,
                        1
                    )
                ),

            "p05":
                float(
                    np.percentile(
                        volume,
                        5
                    )
                ),

            "p25":
                float(
                    np.percentile(
                        volume,
                        25
                    )
                ),

            "p75":
                float(
                    np.percentile(
                        volume,
                        75
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        volume,
                        95
                    )
                ),

            "p99":
                float(
                    np.percentile(
                        volume,
                        99
                    )
                )
        }
    }


    return (
        metadata,
        datasets,
        volume
    )


# ============================================================
# 7. PROCESS ALL SERIES
# ============================================================

all_series_metadata = []
all_series_details = []


for series_index, series_dir in enumerate(
    series_dirs,
    start=1
):

    series_uid = os.path.basename(
        series_dir
    )

    print("\n" + "=" * 80)

    print(
        f"SERIES {series_index}"
    )

    print(
        f"SeriesInstanceUID: {series_uid}"
    )


    metadata, datasets, volume = (
        read_series(
            series_dir
        )
    )


    print(
        f"Slices: "
        f"{metadata['NumberOfSlices']}"
    )

    print(
        f"Volume: "
        f"{metadata['VolumeShape']}"
    )

    print(
        f"Plane metadata: "
        f"{metadata['ImageOrientationPatient']}"
    )

    print(
        f"Description: "
        f"{metadata['SeriesDescription']}"
    )

    print(
        f"Sequence: "
        f"{metadata['SequenceName']}"
    )

    print(
        f"Slice ordering: "
        f"{metadata['SliceOrderingMethod']}"
    )


    # --------------------------------------------------------
    # Create contact sheet
    # --------------------------------------------------------

    n_slices = volume.shape[0]

    n_cols = 5

    n_rows = math.ceil(
        n_slices / n_cols
    )


    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            15,
            3 * n_rows
        )
    )


    axes = np.array(
        axes
    ).reshape(
        n_rows,
        n_cols
    )


    for i in range(
        n_rows * n_cols
    ):

        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]


        if i < n_slices:

            image = volume[i]


            low = np.percentile(
                image,
                1
            )

            high = np.percentile(
                image,
                99
            )


            ax.imshow(
                image,
                cmap="gray",
                vmin=low,
                vmax=high
            )


            ax.set_title(
                f"Slice {i:02d}"
            )


        ax.axis("off")


    plt.suptitle(
        (
            f"Series {series_index} - "
            f"{metadata['SeriesDescription']}"
        ),
        fontsize=16
    )


    plt.tight_layout()


    contact_filename = (
        f"series_{series_index:02d}_"
        f"contact_sheet.png"
    )


    plt.savefig(
        contact_filename,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved: {contact_filename}"
    )


    # --------------------------------------------------------
    # Flatten metadata for CSV
    # --------------------------------------------------------

    csv_record = {

        "StudyInstanceUID":
            STUDY_UID,

        "SeriesIndex":
            series_index,

        "SeriesInstanceUID":
            metadata[
                "SeriesInstanceUID"
            ],

        "NumberOfSlices":
            metadata[
                "NumberOfSlices"
            ],

        "Rows":
            metadata["Rows"],

        "Columns":
            metadata["Columns"],

        "PixelSpacing":
            str(
                metadata[
                    "PixelSpacing"
                ]
            ),

        "SliceThickness":
            metadata[
                "SliceThickness"
            ],

        "SpacingBetweenSlices":
            metadata[
                "SpacingBetweenSlices"
            ],

        "ComputedMedianSliceSpacing":
            metadata[
                "ComputedMedianSliceSpacing"
            ],

        "SeriesDescription":
            metadata[
                "SeriesDescription"
            ],

        "ProtocolName":
            metadata[
                "ProtocolName"
            ],

        "SequenceName":
            metadata[
                "SequenceName"
            ],

        "ScanningSequence":
            metadata[
                "ScanningSequence"
            ],

        "SequenceVariant":
            metadata[
                "SequenceVariant"
            ],

        "Manufacturer":
            metadata[
                "Manufacturer"
            ],

        "ManufacturerModelName":
            metadata[
                "ManufacturerModelName"
            ],

        "MagneticFieldStrength":
            metadata[
                "MagneticFieldStrength"
            ],

        "EchoTime":
            metadata[
                "EchoTime"
            ],

        "RepetitionTime":
            metadata[
                "RepetitionTime"
            ],

        "FlipAngle":
            metadata[
                "FlipAngle"
            ],

        "SliceOrderingMethod":
            metadata[
                "SliceOrderingMethod"
            ],

        "IntensityMin":
            metadata[
                "IntensityStatistics"
            ]["min"],

        "IntensityMax":
            metadata[
                "IntensityStatistics"
            ]["max"],

        "IntensityMean":
            metadata[
                "IntensityStatistics"
            ]["mean"],

        "IntensityMedian":
            metadata[
                "IntensityStatistics"
            ]["median"],

        "IntensityStd":
            metadata[
                "IntensityStatistics"
            ]["std"],

        "IntensityP01":
            metadata[
                "IntensityStatistics"
            ]["p01"],

        "IntensityP05":
            metadata[
                "IntensityStatistics"
            ]["p05"],

        "IntensityP95":
            metadata[
                "IntensityStatistics"
            ]["p95"],

        "IntensityP99":
            metadata[
                "IntensityStatistics"
            ]["p99"]
    }


    all_series_metadata.append(
        csv_record
    )


    all_series_details.append(
        {
            "SeriesIndex":
                series_index,

            "SeriesInstanceUID":
                metadata[
                    "SeriesInstanceUID"
                ],

            "Metadata":
                metadata
        }
    )


# ============================================================
# 8. SAVE SERIES CSV
# ============================================================

series_metadata_df = pd.DataFrame(
    all_series_metadata
)


series_metadata_df.to_csv(
    "study_series_metadata.csv",
    index=False
)


print("\n" + "=" * 80)

print(
    "Saved: study_series_metadata.csv"
)


# ============================================================
# 9. BUILD STUDY METADATA JSON
# ============================================================

study_metadata = {

    "StudyInstanceUID":
        STUDY_UID,

    "NumberOfSeries":
        len(series_dirs),

    "TotalSlices":
        int(
            sum(
                x["NumberOfSlices"]
                for x in all_series_metadata
            )
        ),

    "GroundTruthLabels":
        labels,

    # We keep the report because this is the
    # study-level training information.
    #
    # If you don't want it stored in the output,
    # set this to None.
    "Report":
        report,

    "Series": [
        x["Metadata"]
        for x in all_series_details
    ]
}


with open(
    "study_metadata.json",
    "w"
) as f:

    json.dump(
        study_metadata,
        f,
        indent=4,
        default=str
    )


print(
    "Saved: study_metadata.json"
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL STUDY SUMMARY")
print("=" * 80)

print(f"Study UID    : {STUDY_UID}"           )

print(
    f"Series count : "
    f"{len(series_dirs)}"
)

print(
    f"Total slices : "
    f"{study_metadata['TotalSlices']}"
)


print("\nGround truth:")

for label, value in labels.items():

    print(
        f"  {label:20s}: {value}"
    )


print("\nSeries:")

for record in all_series_metadata:

    print(
        f"  Series {record['SeriesIndex']:02d} | "
        f"{record['NumberOfSlices']:3d} slices | "
        f"{record['SeriesDescription']}"
    )


print("\nOutput files:")
print("  study_metadata.json")
print("  study_series_metadata.csv")

for i in range(
    1,
    len(series_dirs) + 1
):

    print(
        f"  series_{i:02d}_contact_sheet.png"
    )


print("\nDONE.")