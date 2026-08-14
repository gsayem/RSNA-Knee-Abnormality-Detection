# ============================================================
# RSNA KNEE ABNORMALITY DETECTION
# GOLD-LABELED COHORT:
# train.csv + train_series.csv + DICOM HEADER ANALYSIS
#
# PURPOSE
# -------
# Investigate the 58 fully labeled studies without reading
# pixel arrays.
#
# Main questions:
#
# 1. Do train_series.csv and DICOM metadata agree?
# 2. What exactly is DummySeriesDesc! ?
# 3. Are there blank SeriesDescription values?
# 4. What acquisition types exist within each study?
# 5. Are there repeated / near-duplicate acquisitions?
# 6. How heterogeneous are scanners, planes and sequences?
#
# OUTPUTS
# -------
# 01_labeled_series_joined.csv
# 02_dummy_series_analysis.csv
# 03_study_sequence_matrix.csv
# 04_metadata_consistency_report.csv
# 05_labeled_dataset_analysis.json
#
# ============================================================

import os
import re
import json
import glob
import warnings
from collections import Counter

import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError:
    !pip install -q pydicom
    import pydicom


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = (
    "/kaggle/input/competitions/"
    "rsna-knee-abnormality-detection"
)

TRAIN_CSV = os.path.join(
    DATA_ROOT,
    "train.csv"
)

TRAIN_SERIES_CSV = os.path.join(
    DATA_ROOT,
    "train_series.csv"
)

TRAIN_SERIES_ROOT = os.path.join(
    DATA_ROOT,
    "train_series"
)

OUTPUT_DIR = "./rsna_58_integrated_analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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


# ============================================================
# 2. LOAD CSV FILES
# ============================================================

print("=" * 90)
print("LOADING DATA")
print("=" * 90)

train = pd.read_csv(
    TRAIN_CSV
)

train_series = pd.read_csv(
    TRAIN_SERIES_CSV
)


# ------------------------------------------------------------
# Identify fully labeled studies
# ------------------------------------------------------------

fully_labeled_mask = (
    train[LABEL_COLUMNS]
    .notna()
    .all(axis=1)
)

labeled_studies = (
    train.loc[
        fully_labeled_mask
    ]
    .copy()
)

labeled_study_uids = set(
    labeled_studies[
        "StudyInstanceUID"
    ].astype(str)
)


print(
    f"Total train studies        : "
    f"{len(train):,}"
)

print(
    f"Fully labeled studies     : "
    f"{len(labeled_studies):,}"
)

print(
    f"Total train series         : "
    f"{len(train_series):,}"
)


# ============================================================
# 3. LIMIT TRAIN_SERIES TO THE 58 LABELED STUDIES
# ============================================================

train_series_labeled = (
    train_series[
        train_series[
            "StudyInstanceUID"
        ]
        .astype(str)
        .isin(
            labeled_study_uids
        )
    ]
    .copy()
)


print(
    f"Series belonging to labeled "
    f"studies                 : "
    f"{len(train_series_labeled):,}"
)


# ============================================================
# 4. BASIC DICOM HELPERS
# ============================================================

def normalize_string(value):
    """
    Convert a DICOM value into a clean string.
    """

    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


def safe_value(
    ds,
    attribute,
    default=None
):
    """
    Safely extract a DICOM attribute.
    """

    value = getattr(
        ds,
        attribute,
        default
    )

    if value is None:
        return default

    try:

        # MultiValue
        if isinstance(
            value,
            pydicom.multival.MultiValue
        ):

            result = []

            for x in value:

                try:
                    result.append(
                        float(x)
                    )
                except Exception:
                    result.append(
                        str(x)
                    )

            return result


        # List / tuple
        if isinstance(
            value,
            (list, tuple)
        ):

            result = []

            for x in value:

                try:
                    result.append(
                        float(x)
                    )
                except Exception:
                    result.append(
                        str(x)
                    )

            return result


        # Numeric
        if isinstance(
            value,
            (np.integer,)
        ):

            return int(value)


        if isinstance(
            value,
            (np.floating,)
        ):

            return float(value)


        return str(value)

    except Exception:

        return str(value)


# ============================================================
# 5. ANATOMICAL PLANE FROM DICOM ORIENTATION
# ============================================================

def infer_plane_from_orientation(
    ds
):
    """
    Infer anatomical plane from
    ImageOrientationPatient.

    This is compared against
    train_series.csv.
    """

    orientation = getattr(
        ds,
        "ImageOrientationPatient",
        None
    )

    if orientation is None:
        return "Unknown"

    try:

        row = np.asarray(
            orientation[:3],
            dtype=float
        )

        col = np.asarray(
            orientation[3:],
            dtype=float
        )

        normal = np.abs(
            np.cross(
                row,
                col
            )
        )

        dominant_axis = int(
            np.argmax(normal)
        )

        if dominant_axis == 0:
            return "Sagittal"

        if dominant_axis == 1:
            return "Coronal"

        if dominant_axis == 2:
            return "Axial"

        return "Unknown"

    except Exception:

        return "Unknown"


# ============================================================
# 6. SERIES DESCRIPTION CLASSIFICATION
# ============================================================

def classify_description(
    description
):
    """
    Categorize SeriesDescription.

    This is intentionally conservative.
    """

    if description is None:

        return "Blank"

    text = str(
        description
    ).strip()

    if text == "":

        return "Blank"

    if text.lower() == (
        "dummyseriesdesc!"
    ).lower():

        return "Dummy"

    return "Named"


# ============================================================
# 7. ACQUISITION SIGNATURE
# ============================================================

def build_acquisition_signature(
    record
):
    """
    Create a coarse signature representing
    the acquisition type.

    This is NOT intended to identify exact
    clinical equivalence.

    It is only used to find potentially
    repeated / similar acquisitions.
    """

    def normalize_num(value):

        try:

            if pd.isna(value):
                return "NA"

        except Exception:
            pass

        try:

            return str(
                round(
                    float(value),
                    4
                )
            )

        except Exception:

            return str(value)


    signature = "|".join(
        [
            str(
                record.get(
                    "AnatomicalPlane",
                    "NA"
                )
            ),

            str(
                record.get(
                    "Fluid_Sensitive",
                    "NA"
                )
            ),

            str(
                record.get(
                    "Fat_Suppression",
                    "NA"
                )
            ),

            normalize_num(
                record.get(
                    "Rows"
                )
            ),

            normalize_num(
                record.get(
                    "Columns"
                )
            ),

            str(
                record.get(
                    "PixelSpacing",
                    "NA"
                )
            ),

            normalize_num(
                record.get(
                    "SliceThickness"
                )
            ),

            normalize_num(
                record.get(
                    "SpacingBetweenSlices"
                )
            ),

            normalize_num(
                record.get(
                    "EchoTime"
                )
            ),

            normalize_num(
                record.get(
                    "RepetitionTime"
                )
            ),

            str(
                record.get(
                    "Manufacturer",
                    "NA"
                )
            ),

            str(
                record.get(
                    "ManufacturerModelName",
                    "NA"
                )
            )
        ]
    )

    return signature


# ============================================================
# 8. INSPECT DICOM HEADERS
# ============================================================

print("\n" + "=" * 90)
print("READING DICOM HEADERS")
print("=" * 90)

dicom_records = []

failed_series = []


for idx, row in train_series_labeled.iterrows():

    study_uid = str(
        row[
            "StudyInstanceUID"
        ]
    )

    series_uid = str(
        row[
            "SeriesInstanceUID"
        ]
    )

    series_dir = os.path.join(
        TRAIN_SERIES_ROOT,
        study_uid,
        series_uid
    )

    if not os.path.isdir(
        series_dir
    ):

        failed_series.append(
            {
                "StudyInstanceUID":
                    study_uid,

                "SeriesInstanceUID":
                    series_uid,

                "Reason":
                    "Series directory missing"
            }
        )

        continue


    dicom_paths = sorted(
        glob.glob(
            os.path.join(
                series_dir,
                "*.dcm"
            )
        )
    )


    if len(dicom_paths) == 0:

        failed_series.append(
            {
                "StudyInstanceUID":
                    study_uid,

                "SeriesInstanceUID":
                    series_uid,

                "Reason":
                    "No DICOM files"
            }
        )

        continue


    try:

        # ----------------------------------------------------
        # Read first slice only for acquisition metadata
        # ----------------------------------------------------

        ds = pydicom.dcmread(
            dicom_paths[0],
            stop_before_pixels=True,
            force=True
        )

    except Exception as e:

        failed_series.append(
            {
                "StudyInstanceUID":
                    study_uid,

                "SeriesInstanceUID":
                    series_uid,

                "Reason":
                    str(e)
            }
        )

        continue


    # --------------------------------------------------------
    # Header values
    # --------------------------------------------------------

    rows = safe_value(
        ds,
        "Rows"
    )

    columns = safe_value(
        ds,
        "Columns"
    )

    pixel_spacing = safe_value(
        ds,
        "PixelSpacing"
    )

    slice_thickness = safe_value(
        ds,
        "SliceThickness"
    )

    spacing_between_slices = safe_value(
        ds,
        "SpacingBetweenSlices"
    )

    orientation = safe_value(
        ds,
        "ImageOrientationPatient"
    )

    description = normalize_string(
        safe_value(
            ds,
            "SeriesDescription"
        )
    )

    sequence_name = normalize_string(
        safe_value(
            ds,
            "SequenceName"
        )
    )

    manufacturer = normalize_string(
        safe_value(
            ds,
            "Manufacturer"
        )
    )

    scanner_model = normalize_string(
        safe_value(
            ds,
            "ManufacturerModelName"
        )
    )

    field_strength = safe_value(
        ds,
        "MagneticFieldStrength"
    )

    echo_time = safe_value(
        ds,
        "EchoTime"
    )

    repetition_time = safe_value(
        ds,
        "RepetitionTime"
    )

    flip_angle = safe_value(
        ds,
        "FlipAngle"
    )

    scanning_sequence = safe_value(
        ds,
        "ScanningSequence"
    )

    sequence_variant = safe_value(
        ds,
        "SequenceVariant"
    )


    # --------------------------------------------------------
    # Plane from DICOM
    # --------------------------------------------------------

    dicom_plane = (
        infer_plane_from_orientation(
            ds
        )
    )


    # --------------------------------------------------------
    # train_series.csv metadata
    # --------------------------------------------------------

    competition_plane = str(
        row[
            "Anatomical_Plane"
        ]
    )

    fluid_sensitive = int(
        row[
            "Fluid_Sensitive"
        ]
    )

    fat_suppression = int(
        row[
            "Fat_Suppression"
        ]
    )


    # --------------------------------------------------------
    # Description type
    # --------------------------------------------------------

    description_type = (
        classify_description(
            description
        )
    )


    # --------------------------------------------------------
    # Main record
    # --------------------------------------------------------

    record = {

        # IDs
        "StudyInstanceUID":
            study_uid,

        "SeriesInstanceUID":
            series_uid,


        # ----------------------------------------------------
        # Competition metadata
        # ----------------------------------------------------

        "Fluid_Sensitive":
            fluid_sensitive,

        "Fat_Suppression":
            fat_suppression,

        "TrainSeries_AnatomicalPlane":
            competition_plane,


        # ----------------------------------------------------
        # DICOM metadata
        # ----------------------------------------------------

        "DICOM_AnatomicalPlane":
            dicom_plane,

        "PlaneMatch":
            (
                competition_plane
                == dicom_plane
            ),

        "NumberOfSlices":
            len(dicom_paths),

        "Rows":
            rows,

        "Columns":
            columns,

        "PixelSpacing":
            str(
                pixel_spacing
            ),

        "SliceThickness":
            slice_thickness,

        "SpacingBetweenSlices":
            spacing_between_slices,

        "ImageOrientationPatient":
            str(
                orientation
            ),

        "SeriesDescription":
            description,

        "DescriptionType":
            description_type,

        "SequenceName":
            sequence_name,

        "ScanningSequence":
            str(
                scanning_sequence
            ),

        "SequenceVariant":
            str(
                sequence_variant
            ),

        "Manufacturer":
            manufacturer,

        "ManufacturerModelName":
            scanner_model,

        "MagneticFieldStrength":
            field_strength,

        "EchoTime":
            echo_time,

        "RepetitionTime":
            repetition_time,

        "FlipAngle":
            flip_angle
    }


    # --------------------------------------------------------
    # Acquisition signature
    # --------------------------------------------------------

    record[
        "AcquisitionSignature"
    ] = (
        build_acquisition_signature(
            record
        )
    )


    dicom_records.append(
        record
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        len(dicom_records)
        % 25
        == 0
    ):

        print(
            f"Processed "
            f"{len(dicom_records):,}/"
            f"{len(train_series_labeled):,}"
        )


print(
    f"\nSuccessfully inspected: "
    f"{len(dicom_records):,} series"
)

print(
    f"Failed/missing series: "
    f"{len(failed_series):,}"
)


# ============================================================
# 9. MASTER SERIES DATAFRAME
# ============================================================

dicom_df = pd.DataFrame(
    dicom_records
)

failed_df = pd.DataFrame(
    failed_series
)


# ============================================================
# 10. JOIN GROUND-TRUTH LABELS
# ============================================================

label_lookup = (
    labeled_studies[
        [
            "StudyInstanceUID"
        ]
        +
        LABEL_COLUMNS
    ]
    .copy()
)

label_lookup[
    "StudyInstanceUID"
] = (
    label_lookup[
        "StudyInstanceUID"
    ]
    .astype(str)
)


master_df = dicom_df.merge(
    label_lookup,
    on="StudyInstanceUID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 11. SAVE MASTER DATASET
# ============================================================

master_path = os.path.join(
    OUTPUT_DIR,
    "01_labeled_series_joined.csv"
)

master_df.to_csv(
    master_path,
    index=False
)

print(
    "\nSaved:",
    master_path
)


# ============================================================
# 12. DUMMY / BLANK ANALYSIS
# ============================================================

dummy_blank_df = (
    master_df[
        master_df[
            "DescriptionType"
        ]
        .isin(
            [
                "Dummy",
                "Blank"
            ]
        )
    ]
    .copy()
)


dummy_blank_columns = [

    "StudyInstanceUID",
    "SeriesInstanceUID",

    "DescriptionType",
    "SeriesDescription",

    "Fluid_Sensitive",
    "Fat_Suppression",

    "TrainSeries_AnatomicalPlane",
    "DICOM_AnatomicalPlane",
    "PlaneMatch",

    "NumberOfSlices",
    "Rows",
    "Columns",

    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",

    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",

    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",

    "EchoTime",
    "RepetitionTime",
    "FlipAngle",

    "AcquisitionSignature"
]


dummy_blank_df[
    dummy_blank_columns
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "02_dummy_series_analysis.csv"
    ),
    index=False
)


# ============================================================
# 13. STUDY-LEVEL SEQUENCE MATRIX
# ============================================================

study_rows = []

for study_uid, group in master_df.groupby(
    "StudyInstanceUID"
):

    def count(
        mask
    ):
        return int(
            mask.sum()
        )


    # --------------------------------------------------------
    # Plane
    # --------------------------------------------------------

    plane = group[
        "DICOM_AnatomicalPlane"
    ]


    # --------------------------------------------------------
    # Competition sequence categories
    # --------------------------------------------------------

    sagittal = (
        group[
            "TrainSeries_AnatomicalPlane"
        ]
        == "Sagittal"
    )

    coronal = (
        group[
            "TrainSeries_AnatomicalPlane"
        ]
        == "Coronal"
    )

    axial = (
        group[
            "TrainSeries_AnatomicalPlane"
        ]
        == "Axial"
    )


    fluid = (
        group[
            "Fluid_Sensitive"
        ]
        == 1
    )

    non_fluid = (
        group[
            "Fluid_Sensitive"
        ]
        == 0
    )

    fat_sat = (
        group[
            "Fat_Suppression"
        ]
        == 1
    )

    non_fat_sat = (
        group[
            "Fat_Suppression"
        ]
        == 0
    )


    record = {

        "StudyInstanceUID":
            study_uid,

        "TotalSeries":
            len(group),

        "TotalSlices":
            int(
                group[
                    "NumberOfSlices"
                ].sum()
            ),


        # ----------------------------------------------------
        # Planes
        # ----------------------------------------------------

        "SagittalSeries":
            count(sagittal),

        "CoronalSeries":
            count(coronal),

        "AxialSeries":
            count(axial),

        "HasSagittal":
            int(
                sagittal.any()
            ),

        "HasCoronal":
            int(
                coronal.any()
            ),

        "HasAxial":
            int(
                axial.any()
            ),


        # ----------------------------------------------------
        # Sagittal sequence categories
        # ----------------------------------------------------

        "Sagittal_Fluid_FS":
            count(
                sagittal
                & fluid
                & fat_sat
            ),

        "Sagittal_Fluid_NonFS":
            count(
                sagittal
                & fluid
                & non_fat_sat
            ),

        "Sagittal_NonFluid_FS":
            count(
                sagittal
                & non_fluid
                & fat_sat
            ),

        "Sagittal_NonFluid_NonFS":
            count(
                sagittal
                & non_fluid
                & non_fat_sat
            ),


        # ----------------------------------------------------
        # Coronal sequence categories
        # ----------------------------------------------------

        "Coronal_Fluid_FS":
            count(
                coronal
                & fluid
                & fat_sat
            ),

        "Coronal_Fluid_NonFS":
            count(
                coronal
                & fluid
                & non_fat_sat
            ),

        "Coronal_NonFluid_FS":
            count(
                coronal
                & non_fluid
                & fat_sat
            ),

        "Coronal_NonFluid_NonFS":
            count(
                coronal
                & non_fluid
                & non_fat_sat
            ),


        # ----------------------------------------------------
        # Axial sequence categories
        # ----------------------------------------------------

        "Axial_Fluid_FS":
            count(
                axial
                & fluid
                & fat_sat
            ),

        "Axial_Fluid_NonFS":
            count(
                axial
                & fluid
                & non_fat_sat
            ),

        "Axial_NonFluid_FS":
            count(
                axial
                & non_fluid
                & fat_sat
            ),

        "Axial_NonFluid_NonFS":
            count(
                axial
                & non_fluid
                & non_fat_sat
            ),


        # ----------------------------------------------------
        # Description quality
        # ----------------------------------------------------

        "DummySeriesCount":
            count(
                group[
                    "DescriptionType"
                ]
                == "Dummy"
            ),

        "BlankDescriptionCount":
            count(
                group[
                    "DescriptionType"
                ]
                == "Blank"
            ),

        "NamedSeriesCount":
            count(
                group[
                    "DescriptionType"
                ]
                == "Named"
            ),


        # ----------------------------------------------------
        # Metadata consistency
        # ----------------------------------------------------

        "PlaneMatches":
            count(
                group[
                    "PlaneMatch"
                ]
                == True
            ),

        "PlaneMismatches":
            count(
                group[
                    "PlaneMatch"
                ]
                == False
            ),


        # ----------------------------------------------------
        # Scanner info
        # ----------------------------------------------------

        "Manufacturers":
            "|".join(
                sorted(
                    set(
                        group[
                            "Manufacturer"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )
            ),

        "ScannerModels":
            "|".join(
                sorted(
                    set(
                        group[
                            "ManufacturerModelName"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )
            ),

        "FieldStrengths":
            "|".join(
                sorted(
                    set(
                        group[
                            "MagneticFieldStrength"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )
            )
    }


    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    label_values = (
        group[
            LABEL_COLUMNS
        ]
        .drop_duplicates()
    )


    if len(label_values) == 1:

        label_values = (
            label_values
            .iloc[0]
        )

        for label in LABEL_COLUMNS:

            record[label] = int(
                label_values[label]
            )


    study_rows.append(
        record
    )


study_matrix_df = pd.DataFrame(
    study_rows
)


study_matrix_path = os.path.join(
    OUTPUT_DIR,
    "03_study_sequence_matrix.csv"
)


study_matrix_df.to_csv(
    study_matrix_path,
    index=False
)


print(
    "Saved:",
    study_matrix_path
)


# ============================================================
# 14. METADATA CONSISTENCY REPORT
# ============================================================

consistency_records = []


for study_uid, group in master_df.groupby(
    "StudyInstanceUID"
):

    consistency_records.append(
        {
            "StudyInstanceUID":
                study_uid,

            "SeriesCount":
                len(group),

            "PlaneMatches":
                int(
                    group[
                        "PlaneMatch"
                    ].sum()
                ),

            "PlaneMismatches":
                int(
                    (
                        ~group[
                            "PlaneMatch"
                        ]
                    ).sum()
                ),

            "MissingDICOMPlane":
                int(
                    (
                        group[
                            "DICOM_AnatomicalPlane"
                        ]
                        == "Unknown"
                    ).sum()
                ),

            "DummyDescriptions":
                int(
                    (
                        group[
                            "DescriptionType"
                        ]
                        == "Dummy"
                    ).sum()
                ),

            "BlankDescriptions":
                int(
                    (
                        group[
                            "DescriptionType"
                        ]
                        == "Blank"
                    ).sum()
                )
        }
    )


consistency_df = pd.DataFrame(
    consistency_records
)


consistency_path = os.path.join(
    OUTPUT_DIR,
    "04_metadata_consistency_report.csv"
)


consistency_df.to_csv(
    consistency_path,
    index=False
)


print(
    "Saved:",
    consistency_path
)


# ============================================================
# 15. DUPLICATE / REPEATED ACQUISITION ANALYSIS
# ============================================================

duplicate_records = []


for study_uid, group in master_df.groupby(
    "StudyInstanceUID"
):

    signature_counts = (
        group[
            "AcquisitionSignature"
        ]
        .value_counts()
    )


    duplicates = (
        signature_counts[
            signature_counts > 1
        ]
    )


    for signature, count_value in (
        duplicates.items()
    ):

        matching = group[
            group[
                "AcquisitionSignature"
            ]
            == signature
        ]


        duplicate_records.append(
            {
                "StudyInstanceUID":
                    study_uid,

                "AcquisitionSignature":
                    signature,

                "RepeatedCount":
                    int(
                        count_value
                    ),

                "SeriesInstanceUIDs":
                    "|".join(
                        matching[
                            "SeriesInstanceUID"
                        ]
                        .astype(str)
                    ),

                "SeriesDescriptions":
                    " | ".join(
                        matching[
                            "SeriesDescription"
                        ]
                        .fillna(
                            "<NULL>"
                        )
                        .astype(str)
                    ),

                "Planes":
                    "|".join(
                        matching[
                            "DICOM_AnatomicalPlane"
                        ]
                        .astype(str)
                    ),

                "FluidSensitive":
                    "|".join(
                        matching[
                            "Fluid_Sensitive"
                        ]
                        .astype(str)
                    ),

                "FatSuppression":
                    "|".join(
                        matching[
                            "Fat_Suppression"
                        ]
                        .astype(str)
                    )
            }
        )


duplicate_df = pd.DataFrame(
    duplicate_records
)


duplicate_path = os.path.join(
    OUTPUT_DIR,
    "duplicate_acquisition_signatures.csv"
)


duplicate_df.to_csv(
    duplicate_path,
    index=False
)


print(
    "Saved:",
    duplicate_path
)


# ============================================================
# 16. DATASET-LEVEL SUMMARY
# ============================================================

dummy_count = int(
    (
        master_df[
            "DescriptionType"
        ]
        == "Dummy"
    ).sum()
)

blank_count = int(
    (
        master_df[
            "DescriptionType"
        ]
        == "Blank"
    ).sum()
)

named_count = int(
    (
        master_df[
            "DescriptionType"
        ]
        == "Named"
    ).sum()
)

plane_matches = int(
    master_df[
        "PlaneMatch"
    ].sum()
)

plane_mismatches = int(
    (
        ~master_df[
            "PlaneMatch"
        ]
    ).sum()
)


# ------------------------------------------------------------
# Dummy breakdown
# ------------------------------------------------------------

dummy_group = (
    master_df[
        master_df[
            "DescriptionType"
        ]
        == "Dummy"
    ]
)


dummy_summary = {

    "count":
        int(
            len(dummy_group)
        ),

    "studies":
        int(
            dummy_group[
                "StudyInstanceUID"
            ]
            .nunique()
        ),

    "by_plane":
        dummy_group[
            "DICOM_AnatomicalPlane"
        ]
        .value_counts()
        .to_dict(),

    "fluid_sensitive":
        dummy_group[
            "Fluid_Sensitive"
        ]
        .value_counts()
        .to_dict(),

    "fat_suppression":
        dummy_group[
            "Fat_Suppression"
        ]
        .value_counts()
        .to_dict(),

    "manufacturers":
        dummy_group[
            "Manufacturer"
        ]
        .value_counts()
        .to_dict(),

    "scanner_models":
        dummy_group[
            "ManufacturerModelName"
        ]
        .value_counts()
        .to_dict(),

    "field_strengths":
        dummy_group[
            "MagneticFieldStrength"
        ]
        .value_counts()
        .to_dict()
}


# ------------------------------------------------------------
# Blank breakdown
# ------------------------------------------------------------

blank_group = (
    master_df[
        master_df[
            "DescriptionType"
        ]
        == "Blank"
    ]
)


blank_summary = {

    "count":
        int(
            len(blank_group)
        ),

    "studies":
        int(
            blank_group[
                "StudyInstanceUID"
            ]
            .nunique()
        ),

    "by_plane":
        blank_group[
            "DICOM_AnatomicalPlane"
        ]
        .value_counts()
        .to_dict(),

    "manufacturers":
        blank_group[
            "Manufacturer"
        ]
        .value_counts()
        .to_dict()
}


# ------------------------------------------------------------
# Plane cross-tab
# ------------------------------------------------------------

plane_cross_tab = pd.crosstab(
    master_df[
        "TrainSeries_AnatomicalPlane"
    ],
    master_df[
        "DICOM_AnatomicalPlane"
    ]
)


# ------------------------------------------------------------
# Study count distributions
# ------------------------------------------------------------

study_summary = {

    "series_per_study": {

        "mean":
            float(
                study_matrix_df[
                    "TotalSeries"
                ].mean()
            ),

        "median":
            float(
                study_matrix_df[
                    "TotalSeries"
                ].median()
            ),

        "min":
            int(
                study_matrix_df[
                    "TotalSeries"
                ].min()
            ),

        "max":
            int(
                study_matrix_df[
                    "TotalSeries"
                ].max()
            )
    },

    "slices_per_study": {

        "mean":
            float(
                study_matrix_df[
                    "TotalSlices"
                ].mean()
            ),

        "median":
            float(
                study_matrix_df[
                    "TotalSlices"
                ].median()
            ),

        "min":
            int(
                study_matrix_df[
                    "TotalSlices"
                ].min()
            ),

        "max":
            int(
                study_matrix_df[
                    "TotalSlices"
                ].max()
            )
    }
}


# ============================================================
# 17. MASTER JSON SUMMARY
# ============================================================

summary = {

    "dataset": {

        "total_train_studies":
            int(
                len(train)
            ),

        "fully_labeled_studies":
            int(
                len(labeled_studies)
            ),

        "series_in_labeled_studies":
            int(
                len(master_df)
            )
    },


    "study_summary":
        study_summary,


    "metadata_consistency": {

        "total_series":
            int(
                len(master_df)
            ),

        "plane_matches":
            plane_matches,

        "plane_mismatches":
            plane_mismatches,

        "plane_match_rate":
            (
                plane_matches
                / len(master_df)
                if len(master_df) > 0
                else None
            )
    },


    "description_quality": {

        "named":
            named_count,

        "dummy":
            dummy_count,

        "blank":
            blank_count,

        "dummy_percentage":
            (
                dummy_count
                / len(master_df)
                * 100
                if len(master_df) > 0
                else None
            ),

        "blank_percentage":
            (
                blank_count
                / len(master_df)
                * 100
                if len(master_df) > 0
                else None
            )
    },


    "dummy_analysis":
        dummy_summary,


    "blank_analysis":
        blank_summary,


    "plane_cross_tab":
        plane_cross_tab.to_dict(),


    "anatomical_planes":
        master_df[
            "DICOM_AnatomicalPlane"
        ]
        .value_counts()
        .to_dict(),


    "competition_planes":
        master_df[
            "TrainSeries_AnatomicalPlane"
        ]
        .value_counts()
        .to_dict(),


    "manufacturers":
        master_df[
            "Manufacturer"
        ]
        .value_counts()
        .to_dict(),


    "scanner_models":
        master_df[
            "ManufacturerModelName"
        ]
        .value_counts()
        .to_dict(),


    "field_strengths":
        master_df[
            "MagneticFieldStrength"
        ]
        .value_counts()
        .to_dict(),


    "description_types":
        master_df[
            "DescriptionType"
        ]
        .value_counts()
        .to_dict(),


    "series_descriptions":
        master_df[
            "SeriesDescription"
        ]
        .fillna(
            "<NULL>"
        )
        .value_counts()
        .to_dict(),


    "duplicate_acquisition_signatures":
        {

            "studies_with_repeated_signatures":
                int(
                    duplicate_df[
                        "StudyInstanceUID"
                    ]
                    .nunique()
                )
                if len(duplicate_df) > 0
                else 0,

            "number_of_repeated_signature_groups":
                int(
                    len(
                        duplicate_df
                    )
                )
        },


    "failed_series":
        {

            "count":
                int(
                    len(
                        failed_df
                    )
                ),

            "records":
                failed_df.to_dict(
                    orient="records"
                )
        }
}


json_path = os.path.join(
    OUTPUT_DIR,
    "05_labeled_dataset_analysis.json"
)


with open(
    json_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        default=str
    )


print(
    "Saved:",
    json_path
)


# ============================================================
# 18. PRINT IMPORTANT FINDINGS
# ============================================================

print("\n")
print("=" * 90)
print("INTEGRATED ANALYSIS SUMMARY")
print("=" * 90)


print(
    f"\nGold-labeled studies : "
    f"{len(labeled_studies)}"
)

print(
    f"Series analyzed      : "
    f"{len(master_df)}"
)


print(
    f"\nPlane metadata agreement:"
)

print(
    f"  Matches    : "
    f"{plane_matches}"
)

print(
    f"  Mismatches : "
    f"{plane_mismatches}"
)

print(
    f"  Match rate : "
    f"{plane_matches / len(master_df):.2%}"
)


print(
    f"\nSeriesDescription:"
)

print(
    f"  Named : "
    f"{named_count}"
)

print(
    f"  Dummy : "
    f"{dummy_count}"
)

print(
    f"  Blank : "
    f"{blank_count}"
)


print(
    "\nDummy series by plane:"
)

print(
    dummy_group[
        "DICOM_AnatomicalPlane"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nDummy series by Fluid_Sensitive:"
)

print(
    dummy_group[
        "Fluid_Sensitive"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nDummy series by Fat_Suppression:"
)

print(
    dummy_group[
        "Fat_Suppression"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nDummy series by manufacturer:"
)

print(
    dummy_group[
        "Manufacturer"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nDummy series by scanner:"
)

print(
    dummy_group[
        "ManufacturerModelName"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nPotential repeated acquisition signatures:"
)

if len(
    duplicate_df
) == 0:

    print(
        "  None found."
    )

else:

    print(
        duplicate_df[
            [
                "StudyInstanceUID",
                "RepeatedCount",
                "SeriesDescriptions"
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nOutput directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print(
    "\nDONE."
)


# ============================================================
# 19. OPTIONAL: DISPLAY SMALL TABLES
# ============================================================

print(
    "\nFirst 20 rows of the master joined dataset:"
)

display(
    master_df.head(20)
)


print(
    "\nDummy / blank series:"
)

display(
    dummy_blank_df[
        dummy_blank_columns
    ]
)


print(
    "\nStudy sequence matrix:"
)

display(
    study_matrix_df.head(20)
)