# ============================================================
# RSNA KNEE ABNORMALITY DETECTION - V2.1
#
# Hierarchical Multi-Series MRI Baseline
#
# DICOM
#   -> per-series normalization
#   -> K uniformly sampled slices
#   -> pretrained 2D CNN per slice
#   -> slice attention
#   -> series embedding
#   -> metadata fusion
#   -> variable-series attention
#   -> study embedding
#   -> 12-label multi-label classifier
#
# Phase: V2.1 MRI adaptation with stabilized training protocol
#   - Single visual backbone
#   - No radiology-report/NLP
#   - No multi-model fusion yet
#   - Study-level cross validation
#   - StudyInstanceUID is the split unit
#
# Designed for Kaggle GPU.
# ============================================================

from __future__ import annotations

import os
import json
import math
import random
import hashlib
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

import pydicom

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path("/kaggle/input/competitions/" "rsna-knee-abnormality-detection")

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"

WORK_ROOT = Path("/kaggle/working/rsna_v2_1")

# V2 uses exactly the same DICOM preprocessing as V1.
# Reuse V1's cache when it is still available in the current
# Kaggle working session. Otherwise V2 creates its own cache.
V1_CACHE_ROOT = Path("/kaggle/working/rsna_v1/series_cache")

CACHE_ROOT = V1_CACHE_ROOT if V1_CACHE_ROOT.exists() else WORK_ROOT / "series_cache"
CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"
RESULT_ROOT = WORK_ROOT / "results"

for path in [
    WORK_ROOT,
    CACHE_ROOT,
    CHECKPOINT_ROOT,
    RESULT_ROOT,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# Dataset / model
# ------------------------------------------------------------

IMAGE_SIZE = 224
SLICES_PER_SERIES = 16

# Default batch size is deliberately small because one study
# may contain up to 9 series x 16 slices.
BATCH_SIZE = 1

NUM_WORKERS = 2

# IMPORTANT FOR V1/V2 COMPARABILITY:
# V1 declared GRAD_ACCUM_STEPS=4, but its actual training loop
# zeroed and stepped the optimizer every batch. Its effective
# gradient accumulation was therefore 1. V2 intentionally
# preserves that actual behavior so the controlled change is
# ResNet layer4 fine-tuning.
GRAD_ACCUM_STEPS = 1

NUM_FOLDS = 5
RANDOM_SEED = 42

NUM_EPOCHS = 12

# Same absolute cosine floor as V3.1.
COSINE_ETA_MIN = 1e-6

# Differential learning rates:
# hierarchy/head learns at the original V1 rate;
# ResNet layer4 adapts conservatively to knee MRI.
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

# V2 change from V1:
# conv1 + layer1 + layer2 + layer3 remain frozen.
# Only the top ResNet stage (layer4) is trainable.
FINE_TUNE_LAYER4 = True

USE_AMP = True

# Cache normalized series tensors as float16.
USE_CACHE = True

# Small metadata vector:
# plane one-hot (3)
# Fluid_Sensitive (1)
# Fat_Suppression (1)
# PixelSpacingX (1)
# PixelSpacingY (1)
# SliceSpacing (1)
# SliceCount normalized (1)
SERIES_META_DIM = 9

NUM_LABELS = 12

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
# 2. REPRODUCIBILITY
# ============================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism is useful for a 58-study research baseline.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(RANDOM_SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ============================================================
# 3. LOAD / BUILD GOLD-LABELED STUDY TABLE
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)

train_series_df = pd.read_csv(TRAIN_SERIES_CSV)

train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)

train_series_df["StudyInstanceUID"] = train_series_df["StudyInstanceUID"].astype(str)

train_series_df["SeriesInstanceUID"] = train_series_df["SeriesInstanceUID"].astype(str)


# Only studies with all 12 ground-truth labels.
gold_df = train_df[train_df[LABEL_COLUMNS].notna().all(axis=1)].copy()

gold_df = gold_df.sort_values("StudyInstanceUID").reset_index(drop=True)


if len(gold_df) != 58:
    raise RuntimeError("Expected 58 fully labeled studies, " f"found {len(gold_df)}.")


gold_uids = set(gold_df["StudyInstanceUID"])


gold_series_df = train_series_df[
    train_series_df["StudyInstanceUID"].isin(gold_uids)
].copy()


# Make sure every gold study has at least one series.
series_counts = gold_series_df.groupby("StudyInstanceUID").size()

missing_studies = [uid for uid in gold_uids if uid not in series_counts.index]

if missing_studies:
    raise RuntimeError(
        "Gold studies without train_series.csv records:\n" + "\n".join(missing_studies)
    )


print(f"Gold studies : {len(gold_df)}")

print(f"Gold series  : {len(gold_series_df)}")


# ============================================================
# 4. DICOM / SERIES PREPROCESSING HELPERS
# ============================================================


def get_scalar_slice_position(ds) -> float | None:
    orientation = getattr(
        ds,
        "ImageOrientationPatient",
        None,
    )

    position = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    if orientation is None or position is None:
        return None

    try:
        row = np.asarray(
            orientation[:3],
            dtype=np.float64,
        )

        col = np.asarray(
            orientation[3:],
            dtype=np.float64,
        )

        normal = np.cross(
            row,
            col,
        )

        position = np.asarray(
            position,
            dtype=np.float64,
        )

        return float(
            np.dot(
                position,
                normal,
            )
        )

    except Exception:
        return None


def parse_pixel_spacing(
    value,
) -> Tuple[float, float]:
    try:
        if value is None:
            return 1.0, 1.0

        return (
            float(value[0]),
            float(value[1]),
        )

    except Exception:
        return 1.0, 1.0


def robust_series_normalize(
    images: np.ndarray,
) -> np.ndarray:
    """
    Normalize a sampled series using pooled robust
    percentiles.

    Each input slice is H x W.
    Output remains H x W float32 in [0, 1].
    """

    finite = images[np.isfinite(images)]

    if finite.size == 0:
        return np.zeros_like(
            images,
            dtype=np.float32,
        )

    low = np.percentile(
        finite,
        1.0,
    )

    high = np.percentile(
        finite,
        99.0,
    )

    if high <= low:
        return np.zeros_like(
            images,
            dtype=np.float32,
        )

    images = np.nan_to_num(
        images,
        nan=low,
        posinf=high,
        neginf=low,
    )

    images = np.clip(
        images,
        low,
        high,
    )

    images = (images - low) / (high - low)

    return images.astype(np.float32)


def resize_and_pad(
    image: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """
    Preserve aspect ratio and pad to square.

    Input:
        [1, H, W]

    Output:
        [1, size, size]
    """

    if image.ndim != 3:
        raise ValueError(f"Expected [1,H,W], got {image.shape}")

    _, h, w = image.shape

    scale = min(
        size / max(h, 1),
        size / max(w, 1),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    canvas = torch.zeros(
        1,
        size,
        size,
        dtype=resized.dtype,
    )

    top = (size - new_h) // 2

    left = (size - new_w) // 2

    canvas[
        :,
        top : top + new_h,
        left : left + new_w,
    ] = resized

    return canvas


def make_series_cache_key(
    study_uid: str,
    series_uid: str,
) -> Path:

    key = f"{study_uid}_{series_uid}"

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()

    return CACHE_ROOT / f"{digest}.pt"


def load_sampled_series(
    study_uid: str,
    series_uid: str,
    num_slices: int,
) -> Tuple[
    torch.Tensor,
    Dict[str, float],
]:
    """
    Read and preprocess one DICOM series.

    Returns:
        images:
            [K, 3, IMAGE_SIZE, IMAGE_SIZE]

        geometry:
            metadata used for the series metadata vector.
    """

    cache_path = make_series_cache_key(
        study_uid,
        series_uid,
    )

    if USE_CACHE and cache_path.exists():
        cached = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        return (
            cached["images"],
            cached["geometry"],
        )

    series_dir = TRAIN_SERIES_ROOT / study_uid / series_uid

    dcm_paths = sorted(series_dir.glob("*.dcm"))

    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found:\n{series_dir}")

    # --------------------------------------------------------
    # Read headers and spatial ordering
    # --------------------------------------------------------

    headers = []

    for path in dcm_paths:

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )

        position = get_scalar_slice_position(ds)

        instance_number = getattr(
            ds,
            "InstanceNumber",
            0,
        )

        headers.append(
            {
                "path": str(path),
                "position": position,
                "instance": instance_number,
            }
        )

    all_spatial = all(item["position"] is not None for item in headers)

    if all_spatial:

        headers.sort(key=lambda x: x["position"])

    else:

        headers.sort(
            key=lambda x: float(x["instance"] if x["instance"] is not None else 0)
        )

    # --------------------------------------------------------
    # Uniform slice sampling
    # --------------------------------------------------------

    n = len(headers)

    if n >= num_slices:

        selected_indices = (
            np.linspace(
                0,
                n - 1,
                num_slices,
            )
            .round()
            .astype(int)
        )

    else:

        # Repeat boundary / nearest samples so every series
        # always produces exactly K slices.
        selected_indices = (
            np.linspace(
                0,
                n - 1,
                num_slices,
            )
            .round()
            .astype(int)
        )

    selected_headers = [headers[i] for i in selected_indices]

    # --------------------------------------------------------
    # Decode selected slices
    # --------------------------------------------------------

    raw_images = []

    geometry_ds = None

    for item in selected_headers:

        ds = pydicom.dcmread(
            item["path"],
            force=True,
        )

        if geometry_ds is None:
            geometry_ds = ds

        image = ds.pixel_array.astype(np.float32)

        slope = float(
            getattr(
                ds,
                "RescaleSlope",
                1.0,
            )
        )

        intercept = float(
            getattr(
                ds,
                "RescaleIntercept",
                0.0,
            )
        )

        image = image * slope + intercept

        raw_images.append(image)

    volume = np.stack(
        raw_images,
        axis=0,
    )

    # --------------------------------------------------------
    # Per-series robust normalization
    # --------------------------------------------------------

    volume = robust_series_normalize(volume)

    # --------------------------------------------------------
    # Resize/pad and RGB conversion
    # --------------------------------------------------------

    processed = []

    for image in volume:

        tensor = torch.from_numpy(image).unsqueeze(0)

        tensor = resize_and_pad(
            tensor,
            IMAGE_SIZE,
        )

        # MRI is grayscale. Duplicate to 3 channels for
        # the ImageNet-pretrained backbone.
        tensor = tensor.repeat(
            3,
            1,
            1,
        )

        processed.append(tensor)

    images = torch.stack(
        processed,
        dim=0,
    ).contiguous()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    pixel_y, pixel_x = parse_pixel_spacing(
        getattr(
            geometry_ds,
            "PixelSpacing",
            None,
        )
    )

    slice_thickness = float(
        getattr(
            geometry_ds,
            "SliceThickness",
            0.0,
        )
    )

    # Calculate median spatial spacing where possible.
    positions = [item["position"] for item in headers if item["position"] is not None]

    if len(positions) >= 2:

        positions = np.sort(
            np.asarray(
                positions,
                dtype=np.float64,
            )
        )

        diffs = np.abs(np.diff(positions))

        slice_spacing = float(np.median(diffs))

    else:

        slice_spacing = float(
            getattr(
                geometry_ds,
                "SpacingBetweenSlices",
                slice_thickness,
            )
        )

    geometry = {
        "pixel_spacing_x": float(pixel_x),
        "pixel_spacing_y": float(pixel_y),
        "slice_spacing": float(slice_spacing),
        "slice_thickness": float(slice_thickness),
        "slice_count": float(n),
    }

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    if USE_CACHE:

        torch.save(
            {
                "images": images.half(),
                "geometry": geometry,
            },
            cache_path,
        )

    return (
        images,
        geometry,
    )


# ============================================================
# 5. STUDY DATASET
# ============================================================


class KneeStudyDataset(Dataset):
    """
    One sample = one STUDY.

    Output:
        images:
            [S, K, 3, H, W]

        metadata:
            [S, META_DIM]

        labels:
            [12]

        series_mask:
            [S]
    """

    def __init__(
        self,
        study_uids: List[str],
        gold_df: pd.DataFrame,
        series_df: pd.DataFrame,
    ):
        self.study_uids = [str(x) for x in study_uids]

        self.gold_df = gold_df.set_index("StudyInstanceUID")

        self.series_lookup = {}

        for study_uid, group in series_df.groupby("StudyInstanceUID"):

            self.series_lookup[str(study_uid)] = group.sort_values(
                "SeriesInstanceUID"
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.study_uids)

    @staticmethod
    def build_metadata_vector(
        series_row: pd.Series,
        geometry: Dict[str, float],
    ) -> torch.Tensor:
        """
        Metadata vector:

        [0:3] plane one-hot
        [3]   Fluid_Sensitive
        [4]   Fat_Suppression
        [5]   normalized pixel spacing X
        [6]   normalized pixel spacing Y
        [7]   normalized slice spacing
        [8]   normalized log slice count
        """

        plane = str(series_row["Anatomical_Plane"]).lower()

        plane_vec = [
            1.0 if plane == "sagittal" else 0.0,
            1.0 if plane == "coronal" else 0.0,
            1.0 if plane == "axial" else 0.0,
        ]

        fluid = float(series_row["Fluid_Sensitive"])

        fat = float(series_row["Fat_Suppression"])

        # These values are standardized around approximate
        # ranges observed in the labeled cohort.
        #
        # They are NOT learned from the labels.
        px = float(geometry["pixel_spacing_x"]) / 0.5

        py = float(geometry["pixel_spacing_y"]) / 0.5

        sz = float(geometry["slice_spacing"]) / 4.0

        log_slices = math.log1p(float(geometry["slice_count"])) / math.log1p(300.0)

        values = plane_vec + [
            fluid,
            fat,
            px,
            py,
            sz,
            log_slices,
        ]

        return torch.tensor(
            values,
            dtype=torch.float32,
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        study_uid = self.study_uids[index]

        series_rows = self.series_lookup[study_uid]

        series_images = []
        series_metadata = []

        for _, series_row in series_rows.iterrows():

            series_uid = str(series_row["SeriesInstanceUID"])

            images, geometry = load_sampled_series(
                study_uid,
                series_uid,
                SLICES_PER_SERIES,
            )

            metadata = self.build_metadata_vector(
                series_row,
                geometry,
            )

            # Cache stores float16. Convert to float32
            # before entering the model.
            series_images.append(images.float())

            series_metadata.append(metadata)

        images = torch.stack(
            series_images,
            dim=0,
        )

        metadata = torch.stack(
            series_metadata,
            dim=0,
        )

        labels = torch.tensor(
            self.gold_df.loc[
                study_uid,
                LABEL_COLUMNS,
            ].values.astype(np.float32),
            dtype=torch.float32,
        )

        series_mask = torch.ones(
            len(series_images),
            dtype=torch.bool,
        )

        return {
            "study_uid": study_uid,
            "images": images,
            "metadata": metadata,
            "series_mask": series_mask,
            "labels": labels,
        }


# ============================================================
# 6. COLLATE VARIABLE NUMBER OF SERIES
# ============================================================


def study_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    batch_size = len(batch)

    max_series = max(item["images"].shape[0] for item in batch)

    _, num_slices, channels, height, width = batch[0]["images"].shape

    images = torch.zeros(
        batch_size,
        max_series,
        num_slices,
        channels,
        height,
        width,
        dtype=torch.float32,
    )

    metadata = torch.zeros(
        batch_size,
        max_series,
        SERIES_META_DIM,
        dtype=torch.float32,
    )

    series_mask = torch.zeros(
        batch_size,
        max_series,
        dtype=torch.bool,
    )

    labels = torch.stack([item["labels"] for item in batch])

    study_uids = []

    for b, item in enumerate(batch):

        s = item["images"].shape[0]

        images[
            b,
            :s,
        ] = item["images"]

        metadata[
            b,
            :s,
        ] = item["metadata"]

        series_mask[
            b,
            :s,
        ] = True

        study_uids.append(item["study_uid"])

    return {
        "study_uid": study_uids,
        "images": images,
        "metadata": metadata,
        "series_mask": series_mask,
        "labels": labels,
    }


# ============================================================
# 7. MULTI-LABEL STRATIFIED FOLD ASSIGNMENT
# ============================================================


def greedy_multilabel_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """
    A lightweight greedy iterative multilabel split.

    This avoids requiring an external iterative-stratification
    package.

    The algorithm tries to distribute rare labels first.
    With only 58 studies, fold composition is printed so it
    can be audited before training.
    """

    rng = np.random.default_rng(seed)

    n_samples, n_labels = y.shape

    fold_assignments = -np.ones(n_samples, dtype=int)

    # Rarer labels receive priority.
    label_frequency = y.sum(axis=0) + 1e-8

    sample_rarity = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):

        positive_labels = np.where(y[i] > 0)[0]

        if len(positive_labels) == 0:

            sample_rarity[i] = 0

        else:

            sample_rarity[i] = float(np.sum(1.0 / label_frequency[positive_labels]))

    tie_noise = rng.random(n_samples) * 1e-6

    order = np.argsort(-sample_rarity - tie_noise)

    fold_label_counts = np.zeros((n_splits, n_labels), dtype=np.float64)

    fold_sizes = np.zeros(n_splits, dtype=int)

    desired_fold_label_counts = label_frequency / n_splits

    for sample_idx in order:

        sample = y[sample_idx]

        scores = []

        for fold in range(n_splits):

            # Penalize folds already containing too much
            # of the sample's positive-label mass.
            label_score = 0.0

            positive_labels = np.where(sample > 0)[0]

            if len(positive_labels) > 0:

                ratios = fold_label_counts[fold, positive_labels] / (
                    desired_fold_label_counts[positive_labels] + 1e-8
                )

                label_score = float(np.mean(ratios))

            # Slight preference for smaller folds.
            size_score = fold_sizes[fold] / max(1, math.ceil(n_samples / n_splits))

            scores.append(label_score + 0.05 * size_score)

        best_fold = int(np.argmin(scores))

        fold_assignments[sample_idx] = best_fold

        fold_sizes[best_fold] += 1

        fold_label_counts[best_fold] += sample

    return fold_assignments


# ============================================================
# 8. MODEL COMPONENTS
# ============================================================


class SliceAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        hidden = max(
            64,
            embedding_dim // 2,
        )

        self.score = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:
            [N, K, D]

        returns:
            pooled:
                [N, D]

            attention:
                [N, K]
        """

        scores = self.score(x).squeeze(-1)

        attention = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * attention.unsqueeze(-1),
            dim=1,
        )

        return (
            pooled,
            attention,
        )


class SeriesAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        hidden = max(
            64,
            embedding_dim // 2,
        )

        self.score = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:
            [B, S, D]

        mask:
            [B, S]
        """

        scores = self.score(x).squeeze(-1)

        # AMP can make scores float16. The old hard-coded -1e9
        # is outside float16 range and causes an overflow error.
        # Use float32 for masking/softmax for numerical stability.
        scores = scores.float()
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        attention = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * attention.unsqueeze(-1),
            dim=1,
        )

        return (
            pooled,
            attention,
        )


class ResNet18UpperFineTuneEncoder(nn.Module):
    """
    V2 visual encoder.

    Frozen:
        conv1, bn1, layer1, layer2, layer3

    Trainable:
        layer4

    This is deliberately conservative because the gold cohort
    contains only 58 studies.
    """

    def __init__(
        self,
        fine_tune_layer4: bool = True,
    ):
        super().__init__()

        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            print("WARNING: pretrained ResNet18 " "weights could not be loaded.")
            print(f"Reason: {exc}")
            print("Falling back to random initialization.")
            backbone = resnet18(weights=None)

        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fine_tune_layer4 = fine_tune_layer4

        # Freeze the complete pretrained network first.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # V2: adapt only the highest ResNet stage.
        if self.fine_tune_layer4:
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True

        # Keep the same ImageNet normalization as V1.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        """
        Keep frozen lower stages in eval mode so their BatchNorm
        running statistics do not drift. layer4 trains in V2.
        """
        super().train(mode)

        if mode:
            self.backbone.conv1.eval()
            self.backbone.bn1.eval()
            self.backbone.relu.eval()
            self.backbone.maxpool.eval()
            self.backbone.layer1.eval()
            self.backbone.layer2.eval()
            self.backbone.layer3.eval()

            if self.fine_tune_layer4:
                self.backbone.layer4.train(True)
            else:
                self.backbone.layer4.eval()

        return self

    def layer4_parameters(self):
        return [
            parameter
            for parameter in self.backbone.layer4.parameters()
            if parameter.requires_grad
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        # Unlike V1, layer4 participates in autograd.
        return self.backbone(x)


class V2HierarchicalModel(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        num_labels: int,
        fine_tune_layer4: bool = True,
    ):
        super().__init__()

        self.image_encoder = ResNet18UpperFineTuneEncoder(
            fine_tune_layer4=fine_tune_layer4
        )
        visual_dim = self.image_encoder.feature_dim
        self.embedding_dim = 256

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )

        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
        )

        self.series_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim + 128, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.slice_attention = SliceAttention(self.embedding_dim)
        self.series_attention = SeriesAttention(self.embedding_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.15),
            nn.Linear(self.embedding_dim, num_labels),
        )

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, S, K, C, H, W = images.shape

        flat_images = images.reshape(B * S * K, C, H, W)
        flat_features = self.image_encoder(flat_images)
        flat_features = self.visual_projection(flat_features)

        slice_features = flat_features.reshape(B * S, K, self.embedding_dim)

        series_visual, slice_attention = self.slice_attention(slice_features)
        series_visual = series_visual.reshape(B, S, self.embedding_dim)
        slice_attention = slice_attention.reshape(B, S, K)

        metadata_features = self.metadata_projection(metadata)
        combined = torch.cat(
            [series_visual, metadata_features],
            dim=-1,
        )
        series_features = self.series_fusion(combined)

        study_embedding, series_attention = self.series_attention(
            series_features,
            series_mask,
        )

        logits = self.classifier(study_embedding)

        return {
            "logits": logits,
            "study_embedding": study_embedding,
            "slice_attention": slice_attention,
            "series_attention": series_attention,
        }


# ============================================================
# 9. TRAINING HELPERS
# ============================================================


def make_pos_weight(
    y_train: np.ndarray,
) -> torch.Tensor:
    positive = y_train.sum(axis=0)

    negative = y_train.shape[0] - positive

    # Prevent extreme weights with this tiny cohort.
    weights = negative / np.maximum(positive, 1)

    weights = np.clip(
        weights,
        1.0,
        5.0,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def autocast_context():
    if USE_AMP and DEVICE.type == "cuda":

        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )

    return nullcontext()


def safe_auc(
    y_true,
    y_score,
):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    except Exception:
        return np.nan


def safe_ap(
    y_true,
    y_score,
):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return float(
            average_precision_score(
                y_true,
                y_score,
            )
        )

    except Exception:
        return np.nan


# ============================================================
# 10. ONE EPOCH
# ============================================================


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None

    if training:
        model.train()

    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    for batch in loader:

        images = batch["images"].to(
            DEVICE,
            non_blocking=True,
        )

        metadata = batch["metadata"].to(
            DEVICE,
            non_blocking=True,
        )

        series_mask = batch["series_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context():

            outputs = model(
                images,
                metadata,
                series_mask,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits,
                labels,
            )

        if training:

            if scaler is not None:

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probabilities)

        all_targets.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)

    y_prob = np.concatenate(all_probs, axis=0)

    metrics = {"loss": total_loss / max(total_samples, 1)}

    aucs = []
    aps = []

    for idx, label in enumerate(LABEL_COLUMNS):

        auc = safe_auc(
            y_true[:, idx],
            y_prob[:, idx],
        )

        ap = safe_ap(
            y_true[:, idx],
            y_prob[:, idx],
        )

        metrics[f"{label}_AUROC"] = auc

        metrics[f"{label}_AP"] = ap

        if not np.isnan(auc):
            aucs.append(auc)

        if not np.isnan(ap):
            aps.append(ap)

    metrics["macro_AUROC"] = float(np.mean(aucs)) if aucs else np.nan

    metrics["macro_AP"] = float(np.mean(aps)) if aps else np.nan

    # Threshold 0.5 for baseline F1 only.
    try:

        y_pred = (y_prob >= 0.5).astype(np.int32)

        metrics["macro_F1"] = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

    except Exception:

        metrics["macro_F1"] = np.nan

    return (
        metrics,
        y_true,
        y_prob,
    )


# ============================================================
# 11. FOLD REPORTING
# ============================================================


def print_fold_distribution(
    y,
    fold_ids,
):
    print("\nFold label distribution:")

    for fold in range(NUM_FOLDS):

        indices = np.where(fold_ids == fold)[0]

        print(f"\nFold {fold + 1}: " f"{len(indices)} studies")

        values = y[indices].sum(axis=0)

        parts = []

        for idx, label in enumerate(LABEL_COLUMNS):

            parts.append(f"{label}={int(values[idx])}")

        print(" | ".join(parts))


# ============================================================
# 12. TRAINING CONFIG
# ============================================================

y_all = gold_df[LABEL_COLUMNS].values.astype(np.int64)


study_uids = gold_df["StudyInstanceUID"].astype(str).tolist()


fold_ids = greedy_multilabel_folds(
    y_all,
    n_splits=NUM_FOLDS,
    seed=RANDOM_SEED,
)


print_fold_distribution(
    y_all,
    fold_ids,
)


# ============================================================
# 13. PRE-CHECK THAT EVERY SERIES DIRECTORY EXISTS
# ============================================================

print("\n")
print("=" * 90)
print("CHECKING DICOM SERIES")
print("=" * 90)


missing_series = []


for _, row in gold_series_df.iterrows():

    path = (
        TRAIN_SERIES_ROOT / str(row["StudyInstanceUID"]) / str(row["SeriesInstanceUID"])
    )

    if not path.exists():

        missing_series.append(str(path))


if missing_series:

    raise FileNotFoundError(
        "Missing series directories. "
        f"Count={len(missing_series)}\n" + "\n".join(missing_series[:20])
    )


print("All gold-labeled series directories exist.")


# ============================================================
# 14. 5-FOLD TRAINING
# ============================================================

oof_records = []
fold_summaries = []
epoch_oof_records = []


for fold in range(NUM_FOLDS):

    print("\n\n")
    print("#" * 90)
    print(f"FOLD {fold + 1}/{NUM_FOLDS}")
    print("#" * 90)

    train_indices = np.where(fold_ids != fold)[0]

    val_indices = np.where(fold_ids == fold)[0]

    train_uids = [study_uids[i] for i in train_indices]

    val_uids = [study_uids[i] for i in val_indices]

    train_y = y_all[train_indices]

    pos_weight = make_pos_weight(train_y)

    print("\nPos weights:")

    for label, weight in zip(LABEL_COLUMNS, pos_weight.detach().cpu().numpy()):

        print(f"  {label:20s}: " f"{weight:.3f}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = KneeStudyDataset(
        train_uids,
        gold_df,
        gold_series_df,
    )

    val_dataset = KneeStudyDataset(
        val_uids,
        gold_df,
        gold_series_df,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=study_collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=study_collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = V2HierarchicalModel(
        metadata_dim=SERIES_META_DIM,
        num_labels=NUM_LABELS,
        fine_tune_layer4=FINE_TUNE_LAYER4,
    ).to(DEVICE)

    # Differential learning rates:
    # head stays at V1's 1e-3; layer4 adapts at 2e-5.
    backbone_parameters = model.image_encoder.layer4_parameters()
    backbone_parameter_ids = {id(p) for p in backbone_parameters}
    head_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_parameter_ids
    ]

    if not backbone_parameters:
        raise RuntimeError("No trainable ResNet layer4 parameters found for V2.")
    if not head_parameters:
        raise RuntimeError("No trainable hierarchy/head parameters found for V2.")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": HEAD_LEARNING_RATE,
            },
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=COSINE_ETA_MIN,
    )

    print("\nV2 fine-tuning setup:")
    print(f"  Head parameters   : " f"{sum(p.numel() for p in head_parameters):,}")
    print(f"  Layer4 parameters : " f"{sum(p.numel() for p in backbone_parameters):,}")
    print(f"  Head LR           : {HEAD_LEARNING_RATE:.2e}")
    print(f"  Layer4 LR         : {BACKBONE_LEARNING_RATE:.2e}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = (
        torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))
        if DEVICE.type == "cuda"
        else None
    )

    final_checkpoint = CHECKPOINT_ROOT / f"fold_{fold + 1}_epoch_{NUM_EPOCHS}.pt"

    history = []
    fold_epoch_prediction_records = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, NUM_EPOCHS + 1):

        print(f"\nFold {fold + 1} " f"Epoch {epoch}/{NUM_EPOCHS}")

        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        with torch.no_grad():

            val_metrics, y_true, y_prob = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                scaler=None,
            )

        # Validation metrics are reporting-only.
        score = val_metrics["macro_AUROC"]

        scheduler.step()

        current_head_lr = optimizer.param_groups[0]["lr"]
        current_backbone_lr = optimizer.param_groups[1]["lr"]

        row = {
            "fold": fold + 1,
            "epoch": epoch,
            "head_lr": current_head_lr,
            "backbone_lr": current_backbone_lr,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_macro_AUROC": val_metrics["macro_AUROC"],
            "val_macro_AP": val_metrics["macro_AP"],
            "val_macro_F1": val_metrics["macro_F1"],
        }

        history.append(row)

        print(
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"AUROC={row['val_macro_AUROC']:.4f} | "
            f"AP={row['val_macro_AP']:.4f} | "
            f"F1={row['val_macro_F1']:.4f}"
        )

        # ----------------------------------------------------
        # Save this epoch's validation predictions.
        # They are never used to choose a checkpoint.
        # ----------------------------------------------------

        for row_idx, study_uid in enumerate(val_uids):

            prediction_record = {
                "StudyInstanceUID": study_uid,
                "Fold": fold + 1,
                "Epoch": epoch,
            }

            for label_idx, label in enumerate(LABEL_COLUMNS):

                prediction_record[f"{label}_true"] = int(y_true[row_idx, label_idx])

                prediction_record[f"{label}_prob"] = float(y_prob[row_idx, label_idx])

            fold_epoch_prediction_records.append(prediction_record)

        # ----------------------------------------------------
        # Fixed final-epoch checkpoint.
        # ----------------------------------------------------

        if epoch == NUM_EPOCHS:

            torch.save(
                {
                    "fold": fold + 1,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "final_val_macro_AUROC": val_metrics["macro_AUROC"],
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "slices_per_series": SLICES_PER_SERIES,
                        "series_meta_dim": SERIES_META_DIM,
                        "fine_tune_layer4": FINE_TUNE_LAYER4,
                        "head_learning_rate": HEAD_LEARNING_RATE,
                        "backbone_learning_rate": BACKBONE_LEARNING_RATE,
                        "training_protocol": "fixed_12_epochs_no_early_stopping_cosine_lr",
                        "cosine_eta_min": COSINE_ETA_MIN,
                    },
                },
                final_checkpoint,
            )

    # --------------------------------------------------------
    # Fold history
    # --------------------------------------------------------

    pd.DataFrame(history).to_csv(
        RESULT_ROOT / f"fold_{fold + 1}_history.csv",
        index=False,
    )

    pd.DataFrame(fold_epoch_prediction_records).to_csv(
        RESULT_ROOT / f"fold_{fold + 1}_epoch_predictions.csv",
        index=False,
    )

    epoch_oof_records.extend(fold_epoch_prediction_records)

    # --------------------------------------------------------
    # Reload fixed final-epoch model
    # --------------------------------------------------------

    checkpoint = torch.load(
        final_checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------------
    # Final validation predictions
    # --------------------------------------------------------

    with torch.no_grad():

        final_metrics, y_true, y_prob = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            scaler=None,
        )

    # --------------------------------------------------------
    # OOF predictions
    # --------------------------------------------------------

    for row_idx, study_uid in enumerate(val_uids):

        record = {"StudyInstanceUID": study_uid, "Fold": fold + 1}

        for label_idx, label in enumerate(LABEL_COLUMNS):

            record[f"{label}_true"] = int(y_true[row_idx, label_idx])

            record[f"{label}_prob"] = float(y_prob[row_idx, label_idx])

        oof_records.append(record)

    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    fold_summary = {
        "fold": fold + 1,
        "final_epoch": NUM_EPOCHS,
        "final_macro_AUROC": final_metrics["macro_AUROC"],
        "final_macro_AP": final_metrics["macro_AP"],
        "final_macro_F1": final_metrics["macro_F1"],
    }

    for label in LABEL_COLUMNS:

        fold_summary[f"{label}_AUROC"] = final_metrics[f"{label}_AUROC"]

        fold_summary[f"{label}_AP"] = final_metrics[f"{label}_AP"]

    fold_summaries.append(fold_summary)

    print("\nFold final:")

    print(f"  AUROC : " f"{fold_summary['final_macro_AUROC']:.4f}")

    print(f"  AP    : " f"{fold_summary['final_macro_AP']:.4f}")

    print(f"  F1    : " f"{fold_summary['final_macro_F1']:.4f}")

    # Clean up GPU between folds.
    del model
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 15. SAVE FINAL RESULTS
# ============================================================

oof_df = pd.DataFrame(oof_records)

fold_metrics_df = pd.DataFrame(fold_summaries)


oof_path = RESULT_ROOT / "oof_predictions.csv"

fold_metrics_path = RESULT_ROOT / "fold_metrics.csv"


oof_df.to_csv(
    oof_path,
    index=False,
)

fold_metrics_df.to_csv(
    fold_metrics_path,
    index=False,
)


# ============================================================
# 16. OOF SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("FINAL V2.1 FIXED-EPOCH OOF RESULTS")
print("=" * 90)


print("\nFold metrics:")

print(
    fold_metrics_df[
        ["fold", "final_epoch", "final_macro_AUROC", "final_macro_AP", "final_macro_F1"]
    ].to_string(index=False)
)


print("\nMean / std:")


for metric in [
    "final_macro_AUROC",
    "final_macro_AP",
    "final_macro_F1",
]:

    values = fold_metrics_df[metric].dropna().values

    if len(values):

        print(
            f"{metric:25s}: " f"{np.mean(values):.4f} " f"+/- " f"{np.std(values):.4f}"
        )


# ============================================================
# 17. EPOCH-WISE COMPLETE OOF CURVE
# ============================================================

epoch_oof_df = pd.DataFrame(epoch_oof_records)

epoch_oof_path = RESULT_ROOT / "epoch_oof_predictions.csv"

epoch_oof_df.to_csv(
    epoch_oof_path,
    index=False,
)

epoch_summary_rows = []

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_df = epoch_oof_df[epoch_oof_df["Epoch"] == epoch].copy()

    if len(epoch_df) != len(gold_df):
        raise RuntimeError(
            f"Epoch {epoch}: expected "
            f"{len(gold_df)} OOF rows, "
            f"found {len(epoch_df)}."
        )

    if epoch_df["StudyInstanceUID"].nunique() != len(gold_df):
        raise RuntimeError(
            f"Epoch {epoch}: duplicate or missing "
            "StudyInstanceUID in OOF predictions."
        )

    aucs = []
    aps = []
    true_columns = []
    prob_columns = []

    for label in LABEL_COLUMNS:

        y_true_epoch = epoch_df[f"{label}_true"].values

        y_prob_epoch = epoch_df[f"{label}_prob"].values

        auc = safe_auc(
            y_true_epoch,
            y_prob_epoch,
        )

        ap = safe_ap(
            y_true_epoch,
            y_prob_epoch,
        )

        if not np.isnan(auc):
            aucs.append(auc)

        if not np.isnan(ap):
            aps.append(ap)

        true_columns.append(y_true_epoch)

        prob_columns.append(y_prob_epoch)

    y_true_matrix = np.stack(
        true_columns,
        axis=1,
    )

    y_prob_matrix = np.stack(
        prob_columns,
        axis=1,
    )

    y_pred_matrix = (y_prob_matrix >= 0.5).astype(np.int32)

    epoch_summary_rows.append(
        {
            "Epoch": epoch,
            "OOF_Macro_AUROC": float(np.mean(aucs)),
            "OOF_Macro_AP": float(np.mean(aps)),
            "OOF_Macro_F1": float(
                f1_score(
                    y_true_matrix,
                    y_pred_matrix,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    )

epoch_summary_df = pd.DataFrame(epoch_summary_rows)

epoch_summary_path = RESULT_ROOT / "epoch_oof_summary.csv"

epoch_summary_df.to_csv(
    epoch_summary_path,
    index=False,
)

print("\nComplete epoch-wise OOF curve:")

print(epoch_summary_df.to_string(index=False))


# ============================================================
# 18. OUT-OF-FOLD PER-LABEL METRICS
# ============================================================

per_label_rows = []


if len(oof_df) > 0:

    for label in LABEL_COLUMNS:

        y_true = oof_df[f"{label}_true"].values

        y_prob = oof_df[f"{label}_prob"].values

        per_label_rows.append(
            {
                "Label": label,
                "PositiveCount": int(y_true.sum()),
                "AUROC": safe_auc(y_true, y_prob),
                "AveragePrecision": safe_ap(y_true, y_prob),
            }
        )


per_label_df = pd.DataFrame(per_label_rows)


per_label_df.to_csv(
    RESULT_ROOT / "oof_per_label_metrics.csv",
    index=False,
)


print("\nPer-label OOF metrics:")

print(per_label_df.to_string(index=False))


# ============================================================
# 18. SAVE COMPLETE CONFIG
# ============================================================

config = {
    "data_root": str(DATA_ROOT),
    "gold_studies": len(gold_df),
    "gold_series": len(gold_series_df),
    "image_size": IMAGE_SIZE,
    "slices_per_series": SLICES_PER_SERIES,
    "batch_size": BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "num_folds": NUM_FOLDS,
    "num_epochs": NUM_EPOCHS,
    "training_protocol": "fixed_12_epochs_no_early_stopping",
    "lr_scheduler": "CosineAnnealingLR",
    "cosine_eta_min": COSINE_ETA_MIN,
    "epoch_wise_oof_saved": True,
    "head_learning_rate": HEAD_LEARNING_RATE,
    "backbone_learning_rate": BACKBONE_LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "fine_tune_layer4": FINE_TUNE_LAYER4,
    "frozen_resnet_stages": ["conv1", "bn1", "layer1", "layer2", "layer3"],
    "trainable_resnet_stages": ["layer4"],
    "use_amp": USE_AMP,
    "cache_series": USE_CACHE,
    "backbone": "ResNet18 ImageNet - layer4 fine-tuned",
    "metadata_dim": SERIES_META_DIM,
    "labels": LABEL_COLUMNS,
}


with open(
    RESULT_ROOT / "v2_1_config.json",
    "w",
) as f:

    json.dump(
        config,
        f,
        indent=4,
    )


# ============================================================
# 19. DONE
# ============================================================

print("\n")
print("=" * 90)
print("V2.1 TRAINING COMPLETE")
print("=" * 90)


print(f"\nResults directory:\n" f"{RESULT_ROOT}")

print(f"\nOOF predictions:\n" f"{oof_path}")

print(f"\nFold metrics:\n" f"{fold_metrics_path}")

print(f"\nEpoch-wise OOF predictions:\n" f"{epoch_oof_path}")

print(f"\nEpoch-wise OOF summary:\n" f"{epoch_summary_path}")


print("\nCheckpoints:")

for fold in range(1, NUM_FOLDS + 1):

    print(CHECKPOINT_ROOT / f"fold_{fold}_epoch_{NUM_EPOCHS}.pt")

print("\nDone.")
