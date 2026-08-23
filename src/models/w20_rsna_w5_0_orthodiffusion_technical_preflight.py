#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection — W5.0 OrthoDiffusion technical preflight.

This is NOT the W5 training pipeline. It tests the official OrthoDiffusion
source/weights against representative RSNA DICOM series before we spend time
building a full feature cache or Knee-MoE.

No challenge labels, reports, or W2.3 artifacts are used.

Notebook:
    run_w5_ortho_preflight("status")
    run_w5_ortho_preflight("download")   # only when internet is enabled
    run_w5_ortho_preflight("preflight")
    run_w5_ortho_preflight("all")

Recommended attached resources:
    W50_ORTHO_CODE_ROOT    -> official lt-0123/OrthoDiffusion source directory
    W50_ORTHO_WEIGHTS_ROOT -> directory containing:
                               sagittal_model.pt
                               coronal_model.pt
                               axial_model.pt
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch

# ---------------------------------------------------------------------
# Configuration locked to official OrthoDiffusion classification config
# ---------------------------------------------------------------------

DATA_ROOT = Path(
    os.environ.get(
        "W50_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)
TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"

WORK_ROOT = Path(
    os.environ.get(
        "W50_WORK_ROOT",
        "/kaggle/working/rsna_w5_0_ortho_preflight",
    )
)
RESULT_ROOT = WORK_ROOT / "results"
LOCAL_CODE_ROOT = WORK_ROOT / "official" / "OrthoDiffusion"
LOCAL_WEIGHTS_ROOT = WORK_ROOT / "official" / "weights"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

EXPLICIT_CODE_ROOT = os.environ.get("W50_ORTHO_CODE_ROOT", "").strip()
EXPLICIT_WEIGHTS_ROOT = os.environ.get("W50_ORTHO_WEIGHTS_ROOT", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

OFFICIAL_GITHUB_REPO = "https://github.com/lt-0123/OrthoDiffusion.git"
OFFICIAL_HF_REPO = "lanstat0123/orthodiffusion"
OFFICIAL_HF_REVISION = os.environ.get(
    "W50_ORTHO_HF_REVISION",
    "d6920d49e7d1b3dd0774eed6b8e47e386a54dd81",
).strip()

WEIGHT_FILES = {
    "Sagittal": "sagittal_model.pt",
    "Coronal": "coronal_model.pt",
    "Axial": "axial_model.pt",
}
PLANES = ("Sagittal", "Coronal", "Axial")

INPUT_SIZE = 256
DEPTH_SIZE = 16
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE_CHANNELS = 64
NUM_RES_BLOCKS = 1
TIMESTEPS = 1000
FEATURE_TIMESTEP = int(os.environ.get("W50_FEATURE_TIMESTEP", "100"))
FEATURE_BLOCK = os.environ.get("W50_FEATURE_BLOCK", "mid_2").strip()

N_SAMPLE_STUDIES = int(os.environ.get("W50_PREFLIGHT_STUDIES", "3"))
BENCH_BATCHES = [
    int(x)
    for x in os.environ.get("W50_PREFLIGHT_BATCH_SIZES", "1,2,4").split(",")
    if x.strip()
]
BENCH_ITERS = int(os.environ.get("W50_PREFLIGHT_BENCH_ITERS", "3"))
SEED = int(os.environ.get("W50_SEED", "20260823"))

UID = "StudyInstanceUID"
SERIES_UID = "SeriesInstanceUID"
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda:0" if GPU_COUNT else "cpu")


def log(msg: str = ""):
    print(msg, flush=True)


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def elapsed(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------
# Bounded resource discovery: never recursively scan the RSNA DICOM tree
# ---------------------------------------------------------------------


def shallow_dirs(root: Path = Path("/kaggle/input"), depth: int = 3) -> List[Path]:
    if not root.exists():
        return []
    out = [root]
    frontier = [(root, 0)]
    while frontier:
        current, d = frontier.pop(0)
        if d >= depth:
            continue
        try:
            children = [x for x in current.iterdir() if x.is_dir()]
        except Exception:
            continue
        for child in children:
            if child.name in {"train_series", "test_series"}:
                continue
            out.append(child)
            frontier.append((child, d + 1))
    return out


def is_code_root(path: Path) -> bool:
    needed = [
        path / "diffusion_model" / "trainer.py",
        path / "diffusion_model" / "unet.py",
        path / "configs" / "config_finetune.yaml",
        path / "dataset.py",
        path / "finetune_classifier.py",
    ]
    return all(x.exists() for x in needed)


def is_weights_root(path: Path) -> bool:
    return all((path / f).exists() for f in WEIGHT_FILES.values())


def discover_code_root() -> Optional[Path]:
    candidates: List[Path] = []
    if EXPLICIT_CODE_ROOT:
        base = Path(EXPLICIT_CODE_ROOT)
        candidates += [base, base / "OrthoDiffusion"]
    candidates += [LOCAL_CODE_ROOT, Path("/kaggle/working/OrthoDiffusion")]
    for root in shallow_dirs():
        candidates += [root, root / "OrthoDiffusion"]
    for x in dict.fromkeys(candidates):
        if is_code_root(x):
            return x
    return None


def discover_weights_root() -> Optional[Path]:
    candidates: List[Path] = []
    if EXPLICIT_WEIGHTS_ROOT:
        base = Path(EXPLICIT_WEIGHTS_ROOT)
        candidates += [base, base / "orthodiffusion", base / "weights"]
    candidates += [LOCAL_WEIGHTS_ROOT, Path("/kaggle/working/orthodiffusion")]
    for root in shallow_dirs():
        candidates += [root, root / "orthodiffusion", root / "weights"]
    for x in dict.fromkeys(candidates):
        if is_weights_root(x):
            return x
    return None


def git_head(root: Optional[Path]) -> Optional[str]:
    if root is None or not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return None


def dependency_status() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name in ["torch", "numpy", "pandas", "pydicom", "yaml", "einops", "cv2"]:
        try:
            mod = importlib.import_module(name)
            result[name] = {"ok": True, "version": getattr(mod, "__version__", None)}
        except Exception as e:
            result[name] = {"ok": False, "error": repr(e)}
    return result


def download_official_resources() -> Dict[str, str]:
    """Requires Kaggle internet. Does not install Python packages."""
    import shutil

    log("=" * 88)
    log("DOWNLOADING OFFICIAL ORTHODIFFUSION RESOURCES")
    log("=" * 88)

    if not is_code_root(LOCAL_CODE_ROOT):
        if LOCAL_CODE_ROOT.exists():
            shutil.rmtree(LOCAL_CODE_ROOT)
        LOCAL_CODE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                OFFICIAL_GITHUB_REPO,
                str(LOCAL_CODE_ROOT),
            ],
            check=True,
        )

    if not is_weights_root(LOCAL_WEIGHTS_ROOT):
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError("huggingface_hub is required for download mode.") from e
        LOCAL_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=OFFICIAL_HF_REPO,
            revision=OFFICIAL_HF_REVISION,
            local_dir=str(LOCAL_WEIGHTS_ROOT),
            token=HF_TOKEN or None,
        )

    if not is_code_root(LOCAL_CODE_ROOT) or not is_weights_root(LOCAL_WEIGHTS_ROOT):
        raise RuntimeError(
            "Downloaded official OrthoDiffusion resources are incomplete."
        )

    payload = {
        "code_root": str(LOCAL_CODE_ROOT),
        "code_git_head": git_head(LOCAL_CODE_ROOT),
        "weights_root": str(LOCAL_WEIGHTS_ROOT),
    }
    log(json.dumps(payload, indent=2))
    return payload


def import_official_code(code_root: Path):
    root = str(code_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        trainer = importlib.import_module("diffusion_model.trainer")
        unet = importlib.import_module("diffusion_model.unet")
    except Exception as e:
        raise RuntimeError(
            "Official OrthoDiffusion modules failed to import. "
            "Run status and inspect dependency_status."
        ) from e
    return trainer.GaussianDiffusion, unet.create_model


# ---------------------------------------------------------------------
# DICOM geometry and physical ordering
# ---------------------------------------------------------------------


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-8:
        raise ValueError("Zero orientation vector")
    return a / n


def geometry_plane(iop) -> Tuple[str, float]:
    a = np.asarray(iop, dtype=np.float64)
    if a.shape != (6,):
        raise ValueError(f"Expected 6 IOP values, got {a.shape}")
    normal = unit(np.cross(unit(a[:3]), unit(a[3:])))
    scores = {
        "Sagittal": abs(float(normal[0])),
        "Coronal": abs(float(normal[1])),
        "Axial": abs(float(normal[2])),
    }
    plane = max(scores, key=scores.get)
    return plane, float(scores[plane])


def scalar_position(ds) -> Optional[float]:
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        return None
    try:
        a = np.asarray(iop, dtype=np.float64)
        normal = unit(np.cross(a[:3], a[3:]))
        return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))
    except Exception:
        return None


def sorted_series_records(study_uid: str, series_uid: str) -> List[Dict[str, Any]]:
    folder = TRAIN_SERIES_ROOT / study_uid / series_uid
    rows = []
    for path in sorted(folder.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            rows.append(
                {
                    "path": str(path),
                    "position": scalar_position(ds),
                    "instance": getattr(ds, "InstanceNumber", 0),
                }
            )
        except Exception:
            rows.append({"path": str(path), "position": None, "instance": 0})

    if rows and all(x["position"] is not None for x in rows):
        rows.sort(key=lambda x: float(x["position"]))
    else:

        def k(x):
            try:
                return float(x["instance"])
            except Exception:
                return 0.0

        rows.sort(key=k)
    return rows


def decode_slice(path: str) -> Tuple[np.ndarray, Any]:
    ds = pydicom.dcmread(path, force=True)
    image = ds.pixel_array
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise RuntimeError(f"Expected 2D DICOM, got {image.shape}: {path}")
    image = image.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return image * slope + intercept, ds


def count_dicoms(study_uid: str, series_uid: str) -> int:
    folder = TRAIN_SERIES_ROOT / study_uid / series_uid
    try:
        return sum(
            1
            for x in os.scandir(folder)
            if x.is_file() and x.name.lower().endswith(".dcm")
        )
    except Exception:
        return 0


def choose_representative_series() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_SERIES_CSV)
    required = {UID, SERIES_UID, "Anatomical_Plane"}
    if required - set(df.columns):
        raise RuntimeError(
            f"train_series.csv missing {sorted(required - set(df.columns))}"
        )
    df[UID] = df[UID].astype(str)
    df[SERIES_UID] = df[SERIES_UID].astype(str)

    selected_studies = []
    for uid, group in df.groupby(UID):
        good = True
        for plane in PLANES:
            pg = group[group["Anatomical_Plane"].astype(str) == plane]
            if not any(
                count_dicoms(str(uid), str(row[SERIES_UID])) >= DEPTH_SIZE
                for _, row in pg.iterrows()
            ):
                good = False
                break
        if good:
            selected_studies.append(str(uid))
        if len(selected_studies) >= N_SAMPLE_STUDIES:
            break

    if len(selected_studies) < N_SAMPLE_STUDIES:
        raise RuntimeError("Not enough studies have >=16 slices in all 3 planes.")

    rows = []
    for uid in selected_studies:
        group = df[df[UID] == uid]
        for plane in PLANES:
            pg = group[group["Anatomical_Plane"].astype(str) == plane]
            choices = []
            for _, row in pg.iterrows():
                suid = str(row[SERIES_UID])
                n = count_dicoms(uid, suid)
                if n >= DEPTH_SIZE:
                    choices.append((n, suid))
            choices.sort(reverse=True)
            n, suid = choices[0]
            rows.append(
                {
                    UID: uid,
                    "Anatomical_Plane": plane,
                    SERIES_UID: suid,
                    "NDicom": n,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_ROOT / "representative_series.csv", index=False)
    return out


# ---------------------------------------------------------------------
# Official-style RSNA -> 3-D volume conversion
# ---------------------------------------------------------------------


def center_indices(n: int) -> List[int]:
    if n < DEPTH_SIZE:
        raise ValueError(f"Need >= {DEPTH_SIZE} slices, got {n}")
    if n == DEPTH_SIZE:
        return list(range(n))
    center = n // 2
    half = DEPTH_SIZE // 2
    start = center - half
    end = center + half if DEPTH_SIZE % 2 == 0 else center + half + 1
    start = max(0, start)
    end = min(n, end)
    if end - start < DEPTH_SIZE:
        if start == 0:
            end = DEPTH_SIZE
        else:
            start = n - DEPTH_SIZE
    idx = list(range(start, end))
    if len(idx) != DEPTH_SIZE:
        raise RuntimeError(f"Depth crop produced {len(idx)}, expected {DEPTH_SIZE}")
    return idx


def make_volume(
    study_uid: str,
    series_uid: str,
    metadata_plane: str,
) -> Tuple[torch.Tensor, Dict[str, Any], np.ndarray]:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "cv2 is required; official code uses INTER_LINEAR resize."
        ) from e

    records = sorted_series_records(study_uid, series_uid)
    idx = center_indices(len(records))
    images, dsets = [], []

    for i in idx:
        image, ds = decode_slice(records[i]["path"])
        if image.shape != (INPUT_SIZE, INPUT_SIZE):
            image = cv2.resize(
                image,
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        images.append(image.astype(np.float32, copy=False))
        dsets.append(ds)

    # Official dataset represents volume as [H,W,D] before transform.
    hwd = np.stack(images, axis=2).astype(np.float32, copy=False)
    lo, hi = float(hwd.min()), float(hwd.max())
    norm = (hwd - lo) / (hi - lo + 1e-8)
    norm = (norm * 2.0 - 1.0).astype(np.float32, copy=False)

    # Exact official transform: [H,W,D] -> [D,H,W] -> [C,D,H,W]
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).contiguous().float()
    expected = (1, DEPTH_SIZE, INPUT_SIZE, INPUT_SIZE)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(
            f"Unexpected tensor shape {tuple(tensor.shape)} != {expected}"
        )

    ds = dsets[len(dsets) // 2]
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None:
        gp, plane_cos = geometry_plane(iop)
    else:
        gp, plane_cos = None, float("nan")

    positions = [
        records[i]["position"] for i in idx if records[i]["position"] is not None
    ]
    spacing = (
        float(np.median(np.abs(np.diff(np.asarray(positions, dtype=np.float64)))))
        if len(positions) > 1
        else float("nan")
    )
    ps = getattr(ds, "PixelSpacing", None)
    ps0 = float(ps[0]) if ps is not None and len(ps) >= 2 else float("nan")
    ps1 = float(ps[1]) if ps is not None and len(ps) >= 2 else float("nan")

    diag = {
        UID: study_uid,
        SERIES_UID: series_uid,
        "MetadataPlane": metadata_plane,
        "GeometryPlane": gp,
        "PlaneMatch": bool(gp == metadata_plane) if gp else False,
        "GeometryPlaneCosine": plane_cos,
        "SourceSliceCount": len(records),
        "SelectedStartIndex": idx[0],
        "SelectedEndIndex": idx[-1],
        "SelectedSliceCount": len(idx),
        "MedianSliceSpacingMM": spacing,
        "PixelSpacing0MM": ps0,
        "PixelSpacing1MM": ps1,
        "RawMin": lo,
        "RawMax": hi,
        "NormalizedMin": float(tensor.min()),
        "NormalizedMax": float(tensor.max()),
        "NormalizedMean": float(tensor.mean()),
        "NormalizedStd": float(tensor.std()),
        "TensorShape": str(list(tensor.shape)),
    }
    preview = tensor[0, DEPTH_SIZE // 2].numpy()
    return tensor, diag, preview


def prepare_volumes(
    representative: pd.DataFrame,
) -> Tuple[
    Dict[str, List[torch.Tensor]], pd.DataFrame, List[Tuple[str, str, np.ndarray]]
]:
    volumes = {plane: [] for plane in PLANES}
    diagnostics, previews = [], []
    for _, row in representative.iterrows():
        uid = str(row[UID])
        suid = str(row[SERIES_UID])
        plane = str(row["Anatomical_Plane"])
        tensor, diag, preview = make_volume(uid, suid, plane)
        volumes[plane].append(tensor)
        diagnostics.append(diag)
        previews.append((uid, plane, preview))
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(RESULT_ROOT / "volume_diagnostics.csv", index=False)
    return volumes, diag_df, previews


def save_preview(previews) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    studies = list(dict.fromkeys(uid for uid, _, _ in previews))
    lookup = {(uid, plane): image for uid, plane, image in previews}
    fig, axes = plt.subplots(
        len(studies),
        len(PLANES),
        figsize=(12, 4 * len(studies)),
        squeeze=False,
    )
    for r, uid in enumerate(studies):
        for c, plane in enumerate(PLANES):
            ax = axes[r, c]
            image = lookup.get((uid, plane))
            if image is not None:
                ax.imshow(image, cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"{uid[-12:]}\n{plane} center")
            ax.axis("off")
    fig.tight_layout()
    path = RESULT_ROOT / "orthodiffusion_rsna_volume_preview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
# Exact official model construction / EMA checkpoint loading
# ---------------------------------------------------------------------


def load_diffusion(
    code_root: Path,
    weight_path: Path,
    device: torch.device,
):
    GaussianDiffusion, create_model = import_official_code(code_root)

    denoise = create_model(
        INPUT_SIZE,
        BASE_CHANNELS,
        NUM_RES_BLOCKS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
    ).to(device)

    diffusion = GaussianDiffusion(
        denoise,
        image_size=INPUT_SIZE,
        depth_size=DEPTH_SIZE,
        timesteps=TIMESTEPS,
        loss_type="l2",
    ).to(device)

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "ema" not in checkpoint:
        raise RuntimeError(f"{weight_path} does not contain checkpoint['ema'].")

    original = checkpoint["ema"]
    state = {}
    for key, value in original.items():
        if key.startswith("denoise_fn.module."):
            key = key.replace("denoise_fn.module.", "denoise_fn.", 1)
        state[key] = value

    incompatible = diffusion.load_state_dict(state, strict=False)
    diffusion.eval()
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)

    diag = {
        "WeightFilename": weight_path.name,
        "WeightSizeMB": weight_path.stat().st_size / (1024**2),
        "WeightSHA256": sha256_file(weight_path),
        "CheckpointKeys": sorted(checkpoint.keys()),
        "EMAStateEntries": len(original),
        "MissingKeys": list(incompatible.missing_keys),
        "UnexpectedKeys": list(incompatible.unexpected_keys),
        "ParameterCount": sum(x.numel() for x in diffusion.parameters()),
    }
    del checkpoint, original, state
    gc.collect()
    return diffusion, diag


def reset_feature_rng(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def get_feature(
    diffusion,
    batch: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    seed: int,
) -> torch.Tensor:
    reset_feature_rng(seed, device)
    image = batch.to(device, non_blocking=True)
    t = torch.full(
        (image.shape[0],),
        FEATURE_TIMESTEP,
        dtype=torch.long,
        device=device,
    )
    feature = diffusion.get_feature(
        image,
        t,
        name=FEATURE_BLOCK,
        use_amp=bool(use_amp),
    )
    if not isinstance(feature, torch.Tensor):
        raise RuntimeError(f"get_feature returned {type(feature)}")
    return feature


def pool_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 5:
        return feature.float().mean(dim=(2, 3, 4))
    if feature.ndim == 2:
        return feature.float()
    if feature.ndim > 2:
        return feature.float().mean(dim=tuple(range(2, feature.ndim)))
    raise RuntimeError(f"Unsupported feature shape {tuple(feature.shape)}")


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a = a.float().cpu()
    b = b.float().cpu()
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (a * b).sum(dim=1).numpy()


def cross_volume_cosine(pooled: torch.Tensor) -> Dict[str, float]:
    x = pooled.float().cpu()
    if x.shape[0] < 2:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = x @ x.T
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
    return {
        "mean": float(off.mean()),
        "min": float(off.min()),
        "max": float(off.max()),
    }


def benchmark(
    diffusion,
    volume: torch.Tensor,
    plane: str,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    if device.type != "cuda":
        return pd.DataFrame(rows)

    for batch_size in BENCH_BATCHES:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batch = volume.repeat(batch_size, 1, 1, 1, 1).contiguous()

            _ = get_feature(
                diffusion,
                batch,
                device,
                use_amp=True,
                seed=SEED + batch_size,
            )
            torch.cuda.synchronize(device)

            started = time.perf_counter()
            for iteration in range(BENCH_ITERS):
                _ = get_feature(
                    diffusion,
                    batch,
                    device,
                    use_amp=True,
                    seed=SEED + 1000 + iteration,
                )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            n = batch_size * BENCH_ITERS
            alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
            reserve = torch.cuda.max_memory_reserved(device) / (1024**3)

            rows.append(
                {
                    "Plane": plane,
                    "BatchSize": batch_size,
                    "Iterations": BENCH_ITERS,
                    "Seconds": seconds,
                    "VolumesPerSecond": n / max(seconds, 1e-9),
                    "MillisecondsPerBatch": 1000 * seconds / BENCH_ITERS,
                    "PeakAllocatedGB": alloc,
                    "PeakReservedGB": reserve,
                    "Status": "PASS",
                }
            )
            log(
                f"  {plane:8s} batch={batch_size:2d} "
                f"rate={rows[-1]['VolumesPerSecond']:.3f} vol/s "
                f"peak={alloc:.2f} GB"
            )
            del batch, _
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            rows.append(
                {
                    "Plane": plane,
                    "BatchSize": batch_size,
                    "Iterations": BENCH_ITERS,
                    "Seconds": float("nan"),
                    "VolumesPerSecond": float("nan"),
                    "MillisecondsPerBatch": float("nan"),
                    "PeakAllocatedGB": float("nan"),
                    "PeakReservedGB": float("nan"),
                    "Status": "OOM",
                }
            )
            log(f"  {plane:8s} batch={batch_size:2d}: OOM")
            gc.collect()
            torch.cuda.empty_cache()
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Status and preflight orchestration
# ---------------------------------------------------------------------


def status() -> Dict[str, Any]:
    code_root = discover_code_root()
    weights_root = discover_weights_root()
    weights = {}
    if weights_root:
        for plane, filename in WEIGHT_FILES.items():
            path = weights_root / filename
            if path.exists():
                weights[plane] = {
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024**2),
                }

    payload = {
        "device": str(DEVICE),
        "gpu_count": GPU_COUNT,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)],
        "train_series_csv_exists": TRAIN_SERIES_CSV.exists(),
        "train_series_root_exists": TRAIN_SERIES_ROOT.exists(),
        "code_root": str(code_root) if code_root else None,
        "code_git_head": git_head(code_root),
        "weights_root": str(weights_root) if weights_root else None,
        "weights": weights,
        "dependency_status": dependency_status(),
        "official_config": {
            "input_size": INPUT_SIZE,
            "depth_size": DEPTH_SIZE,
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "num_channels": BASE_CHANNELS,
            "num_res_blocks": NUM_RES_BLOCKS,
            "timesteps": TIMESTEPS,
            "feature_timestep": FEATURE_TIMESTEP,
            "feature_block": FEATURE_BLOCK,
        },
        "work_root": str(WORK_ROOT),
    }
    log(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def run_preflight() -> Dict[str, Any]:
    seed_all()
    started = time.time()

    log("=" * 88)
    log("RSNA W5.0 ORTHODIFFUSION TECHNICAL PREFLIGHT")
    log("=" * 88)

    current_status = status()
    code_root = discover_code_root()
    weights_root = discover_weights_root()

    if not TRAIN_SERIES_CSV.exists() or not TRAIN_SERIES_ROOT.exists():
        raise FileNotFoundError(
            "RSNA train_series.csv/train_series directory not found."
        )
    if code_root is None:
        raise FileNotFoundError(
            "Official OrthoDiffusion code not found. "
            "Set W50_ORTHO_CODE_ROOT or run download mode."
        )
    if weights_root is None:
        raise FileNotFoundError(
            "Official OrthoDiffusion weights not found. "
            "Set W50_ORTHO_WEIGHTS_ROOT or run download mode."
        )

    missing_deps = [
        name
        for name, item in current_status["dependency_status"].items()
        if not item.get("ok", False)
    ]
    if missing_deps:
        raise RuntimeError(
            f"Missing dependencies: {missing_deps}. "
            "This script deliberately does not silently install packages."
        )

    # Fail early if official model modules cannot import.
    GaussianDiffusion, create_model = import_official_code(code_root)
    log(f"Official code root     : {code_root}")
    log(f"Official code git HEAD : {git_head(code_root)}")
    log(f"Official weights root  : {weights_root}")
    log(f"GaussianDiffusion      : {GaussianDiffusion}")
    log(f"create_model           : {create_model}")

    representative = choose_representative_series()
    log("\nRepresentative series:")
    log(representative.to_string(index=False))

    volumes, volume_diag, previews = prepare_volumes(representative)
    preview_path = save_preview(previews)

    log("\nRSNA -> official-style volume diagnostics:")
    cols = [
        UID,
        "MetadataPlane",
        "GeometryPlane",
        "PlaneMatch",
        "GeometryPlaneCosine",
        "SourceSliceCount",
        "SelectedStartIndex",
        "SelectedEndIndex",
        "MedianSliceSpacingMM",
        "NormalizedMin",
        "NormalizedMax",
        "NormalizedStd",
    ]
    log(volume_diag[cols].to_string(index=False))

    plane_match_ok = bool(volume_diag["PlaneMatch"].astype(bool).all())
    shape_ok = all(
        tuple(v.shape) == (1, DEPTH_SIZE, INPUT_SIZE, INPUT_SIZE)
        for plane in PLANES
        for v in volumes[plane]
    )

    model_rows = []
    precision_rows = []
    bench_frames = []
    feature_ok_flags = []
    batch1_ok_flags = []

    for plane_index, plane in enumerate(PLANES):
        log("\n" + "=" * 88)
        log(f"OFFICIAL {plane.upper()} MODEL")
        log("=" * 88)

        device = DEVICE
        weight_path = weights_root / WEIGHT_FILES[plane]
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        diffusion, load_diag = load_diffusion(code_root, weight_path, device)
        load_seconds = time.perf_counter() - t0

        log(f"Loaded in             : {load_seconds:.1f}s")
        log(f"Parameters            : {load_diag['ParameterCount']:,}")
        log(f"Missing keys          : {len(load_diag['MissingKeys'])}")
        log(f"Unexpected keys       : {len(load_diag['UnexpectedKeys'])}")

        batch = torch.stack(volumes[plane], dim=0)
        expected_tail = (1, DEPTH_SIZE, INPUT_SIZE, INPUT_SIZE)
        if tuple(batch.shape[1:]) != expected_tail:
            raise RuntimeError(f"{plane}: bad input batch shape {tuple(batch.shape)}")

        fp32 = get_feature(
            diffusion,
            batch,
            device,
            use_amp=False,
            seed=SEED + plane_index * 100,
        )
        fp32_finite = bool(torch.isfinite(fp32).all().item())
        fp32_std = float(fp32.float().std().cpu())
        pooled32 = pool_feature(fp32)
        cross = cross_volume_cosine(pooled32)

        feature_ok = fp32_finite and fp32_std > 1e-8
        feature_ok_flags.append(feature_ok)

        amp_finite = True
        mean_cos = min_cos = mean_diff = max_diff = float("nan")

        if device.type == "cuda":
            try:
                amp = get_feature(
                    diffusion,
                    batch,
                    device,
                    use_amp=True,
                    seed=SEED + plane_index * 100,
                )
                amp_finite = bool(torch.isfinite(amp).all().item())
                pooled_amp = pool_feature(amp)
                cos = cosine_rows(pooled32, pooled_amp)
                diff = (pooled32.float().cpu() - pooled_amp.float().cpu()).abs()
                mean_cos = float(cos.mean())
                min_cos = float(cos.min())
                mean_diff = float(diff.mean())
                max_diff = float(diff.max())
            except Exception as exc:
                amp_finite = False
                log(f"AMP probe failed      : {repr(exc)}")

        model_rows.append(
            {
                "Plane": plane,
                "WeightFilename": weight_path.name,
                "WeightSizeMB": load_diag["WeightSizeMB"],
                "WeightSHA256": load_diag["WeightSHA256"],
                "ParameterCount": load_diag["ParameterCount"],
                "MissingKeyCount": len(load_diag["MissingKeys"]),
                "UnexpectedKeyCount": len(load_diag["UnexpectedKeys"]),
                "LoadSeconds": load_seconds,
                "InputBatchShape": str(list(batch.shape)),
                "FP32FeatureShape": str(list(fp32.shape)),
                "FeatureNDIM": int(fp32.ndim),
                "FeatureChannelsOrDim": int(fp32.shape[1]) if fp32.ndim >= 2 else -1,
                "FP32FeatureFinite": fp32_finite,
                "FP32FeatureMean": float(fp32.float().mean().cpu()),
                "FP32FeatureStd": fp32_std,
                "PooledShape": str(list(pooled32.shape)),
                "PooledMeanL2Norm": float(pooled32.norm(dim=1).mean().cpu()),
                "CrossVolumeCosineMean": cross["mean"],
                "CrossVolumeCosineMin": cross["min"],
                "CrossVolumeCosineMax": cross["max"],
            }
        )

        precision_rows.append(
            {
                "Plane": plane,
                "AMPFeatureFinite": amp_finite,
                "MeanCosineFP32vsAMP": mean_cos,
                "MinCosineFP32vsAMP": min_cos,
                "MeanAbsoluteDifferencePooled": mean_diff,
                "MaxAbsoluteDifferencePooled": max_diff,
            }
        )

        log(f"Input batch shape     : {tuple(batch.shape)}")
        log(f"Feature shape         : {tuple(fp32.shape)}")
        log(f"Feature finite/std    : {fp32_finite} / {fp32_std:.6f}")
        log(
            "Cross-volume cosine   : "
            f"mean={cross['mean']:.4f} min={cross['min']:.4f} max={cross['max']:.4f}"
        )
        if device.type == "cuda":
            log(f"FP32 vs AMP cosine    : mean={mean_cos:.6f} min={min_cos:.6f}")
            log("GPU benchmark:")
            bdf = benchmark(diffusion, batch[0:1], plane, device)
            bench_frames.append(bdf)
            b1 = bdf[bdf["BatchSize"] == 1] if len(bdf) else pd.DataFrame()
            batch1_ok_flags.append(bool(len(b1) and b1.iloc[0]["Status"] == "PASS"))
        else:
            batch1_ok_flags.append(True)

        del diffusion, batch, fp32, pooled32
        if "amp" in locals():
            del amp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(
        RESULT_ROOT / "plane_model_feature_diagnostics.csv",
        index=False,
    )

    precision_df = pd.DataFrame(precision_rows)
    precision_df.to_csv(
        RESULT_ROOT / "fp32_amp_feature_comparison.csv",
        index=False,
    )

    bench_df = (
        pd.concat(bench_frames, ignore_index=True) if bench_frames else pd.DataFrame()
    )
    bench_df.to_csv(
        RESULT_ROOT / "gpu_volume_benchmark.csv",
        index=False,
    )

    best_batches: Dict[str, Any] = {}
    if len(bench_df):
        for plane in PLANES:
            ok = bench_df[(bench_df["Plane"] == plane) & (bench_df["Status"] == "PASS")]
            if len(ok):
                best = ok.sort_values(
                    "VolumesPerSecond",
                    ascending=False,
                ).iloc[0]
                best_batches[plane] = {
                    "batch_size": int(best["BatchSize"]),
                    "volumes_per_second": float(best["VolumesPerSecond"]),
                    "peak_allocated_gb": float(best["PeakAllocatedGB"]),
                }

    amp_ok = (
        all(bool(x["AMPFeatureFinite"]) for x in precision_rows) if GPU_COUNT else True
    )

    overall_pass = bool(
        plane_match_ok
        and shape_ok
        and all(feature_ok_flags)
        and all(batch1_ok_flags)
        and amp_ok
    )

    summary = {
        "overall_pass": overall_pass,
        "code_root": str(code_root),
        "code_git_head": git_head(code_root),
        "weights_root": str(weights_root),
        "official_hf_repo": OFFICIAL_HF_REPO,
        "official_hf_revision": OFFICIAL_HF_REVISION,
        "sample_studies": N_SAMPLE_STUDIES,
        "sample_volumes": len(volume_diag),
        "all_metadata_geometry_plane_matches": plane_match_ok,
        "minimum_geometry_plane_cosine": float(
            pd.to_numeric(
                volume_diag["GeometryPlaneCosine"],
                errors="coerce",
            ).min()
        ),
        "all_native_tensor_shapes_ok": shape_ok,
        "feature_extraction_all_planes_ok": all(feature_ok_flags),
        "amp_features_all_finite": amp_ok,
        "batch1_all_planes_ok": all(batch1_ok_flags),
        "best_gpu_batches": best_batches,
        "feature_timestep": FEATURE_TIMESTEP,
        "feature_block": FEATURE_BLOCK,
        "preview_path": str(preview_path) if preview_path else None,
        "runtime_seconds": time.time() - started,
    }

    (RESULT_ROOT / "ORTHODIFFUSION_PREFLIGHT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    log("\n" + "=" * 88)
    log(
        "W5.0 ORTHODIFFUSION TECHNICAL PREFLIGHT: "
        + ("PASS" if overall_pass else "REVIEW REQUIRED")
    )
    log("=" * 88)
    log(
        f"Plane matches         : "
        f"{int(volume_diag['PlaneMatch'].sum())}/{len(volume_diag)}"
    )
    log(f"Minimum plane cosine  : " f"{summary['minimum_geometry_plane_cosine']:.4f}")
    log(f"Native volume shape   : " f"[B,1,{DEPTH_SIZE},{INPUT_SIZE},{INPUT_SIZE}]")
    log(f"Feature point         : " f"t={FEATURE_TIMESTEP}, block={FEATURE_BLOCK}")
    log(f"Best GPU batches      : {json.dumps(best_batches)}")
    log(f"Results               : {RESULT_ROOT}")
    log(f"Runtime               : {elapsed(time.time() - started)}")
    return summary


def run_w5_ortho_preflight(mode: str = "preflight"):
    mode = str(mode).strip().lower()
    valid = {"status", "download", "preflight", "all"}
    if mode not in valid:
        raise ValueError(f"mode must be one of {sorted(valid)}, got {mode!r}")
    if mode == "status":
        return status()
    if mode == "download":
        return download_official_resources()
    if mode == "all":
        if discover_code_root() is None or discover_weights_root() is None:
            download_official_resources()
        return run_preflight()
    return run_preflight()


def parse_args():
    parser = argparse.ArgumentParser(
        description="RSNA W5.0 OrthoDiffusion technical preflight"
    )
    parser.add_argument(
        "--mode",
        choices=["status", "download", "preflight", "all"],
        default="preflight",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_w5_ortho_preflight(args.mode)


if __name__ == "__main__":
    main()
