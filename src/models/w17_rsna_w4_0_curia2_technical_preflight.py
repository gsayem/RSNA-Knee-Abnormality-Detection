#!/usr/bin/env python3
# RSNA Knee Abnormality Detection - W4.0 Curia-2 technical preflight
#
# This is intentionally NOT the full 4,407-study training pipeline.
# It verifies model loading, official preprocessing, DICOM orientation,
# embedding shape, FP32-vs-FP16 agreement, and safe T4 batch sizes.
#
# Notebook:
#   run_curia2_preflight("download")   # optional, internet + accepted HF terms
#   run_curia2_preflight("preflight")
# or:
#   run_curia2_preflight("all")
#
# If Curia-2 is attached as a Kaggle Dataset, set BEFORE executing this file:
#   os.environ["W40_CURIA_ROOT"] = "/kaggle/input/datasets/isayem/curia-2-model/curia-2-model"
#
# No challenge labels, reports, W1/W2/W2.3 are used here.

from __future__ import annotations
import os, gc, json, time, hashlib, argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch

DATA_ROOT = Path(
    os.environ.get(
        "W40_DATA_ROOT",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    )
)
TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"

WORK_ROOT = Path(
    os.environ.get(
        "W40_CURIA_PREFLIGHT_WORK_ROOT",
        "/kaggle/working/rsna_w4_0_curia2_preflight",
    )
)
RESULT_ROOT = WORK_ROOT / "results"
LOCAL_CURIA_ROOT = WORK_ROOT / "pretrained" / "curia-2"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

CURIA_REPO_ID = "raidium/curia-2"
CURIA_REVISION = os.environ.get(
    "W40_CURIA_REVISION",
    "645f566dd9e002505691178917cee265b491c7f7",
).strip()
EXPLICIT_CURIA_ROOT = os.environ.get(
    "W40_CURIA_ROOT", "/kaggle/input/datasets/isayem/curia-2-model/curia-2-model"
).strip()

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

N_SAMPLE_STUDIES = int(os.environ.get("W40_PREFLIGHT_STUDIES", "3"))
BENCH_BATCH_SIZES = [
    int(x)
    for x in os.environ.get("W40_PREFLIGHT_BATCH_SIZES", "1,2,4,8,16").split(",")
    if x.strip()
]
BENCH_ITERS = int(os.environ.get("W40_PREFLIGHT_BENCH_ITERS", "3"))

EXPECTED_HIDDEN = 768
EXPECTED_CHANNELS = 1
EXPECTED_IMAGE = 512
EXPECTED_PATCH = 16
EXPECTED_MODEL_SHA256 = (
    "403a02e27531d2858ecd1e9b1ec2d5ea" "7bfa909f10ff0d9e8416090a6a8c96ef"
)

UID = "StudyInstanceUID"
TARGET_AXES = {
    "Axial": ("P", "L"),
    "Coronal": ("I", "L"),
    "Sagittal": ("I", "P"),
}
VEC = {
    "L": np.array([1.0, 0.0, 0.0]),
    "R": np.array([-1.0, 0.0, 0.0]),
    "P": np.array([0.0, 1.0, 0.0]),
    "A": np.array([0.0, -1.0, 0.0]),
    "S": np.array([0.0, 0.0, 1.0]),
    "I": np.array([0.0, 0.0, -1.0]),
}

GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda:0" if GPU_COUNT else "cpu")


def log(x):
    print(x, flush=True)


def sha256_file(path: Path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def immediate_inputs():
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    try:
        return [p for p in root.iterdir() if p.is_dir()]
    except Exception:
        return []


def looks_curia2(path: Path):
    required = [
        path / "config.json",
        path / "model.safetensors",
        path / "preprocessor_config.json",
        path / "curia_image_processor.py",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        c = json.loads((path / "config.json").read_text())
        return (
            c.get("model_type") == "dinov2"
            and int(c.get("hidden_size", -1)) == EXPECTED_HIDDEN
            and int(c.get("num_channels", -1)) == EXPECTED_CHANNELS
            and int(c.get("image_size", -1)) == EXPECTED_IMAGE
        )
    except Exception:
        return False


def discover_curia2():
    c = []
    if EXPLICIT_CURIA_ROOT:
        p = Path(EXPLICIT_CURIA_ROOT)
        c += [p, p / "curia-2"]
    c += [LOCAL_CURIA_ROOT, Path("/kaggle/working/curia-2")]
    # shallow only: never recursively scan the competition's huge input tree
    for root in immediate_inputs():
        c.append(root)
        try:
            c += [p for p in root.iterdir() if p.is_dir()]
        except Exception:
            pass
    for p in dict.fromkeys(c):
        if looks_curia2(p):
            return p
    return None


def download_curia2():
    from huggingface_hub import snapshot_download

    LOCAL_CURIA_ROOT.mkdir(parents=True, exist_ok=True)
    log("=" * 88)
    log("DOWNLOADING CURIA-2")
    log("=" * 88)
    log(f"Repo     : {CURIA_REPO_ID}")
    log(f"Revision : {CURIA_REVISION}")
    try:
        snapshot_download(
            repo_id=CURIA_REPO_ID,
            revision=CURIA_REVISION,
            local_dir=str(LOCAL_CURIA_ROOT),
            token=HF_TOKEN or None,
        )
    except Exception as e:
        raise RuntimeError(
            "Download failed. Accept raidium/curia-2 terms on Hugging Face, "
            "enable internet, and provide HF_TOKEN if required."
        ) from e
    if not looks_curia2(LOCAL_CURIA_ROOT):
        raise RuntimeError("Downloaded Curia-2 directory is incomplete.")
    return LOCAL_CURIA_ROOT


# --------------------------- DICOM orientation ---------------------------


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("zero orientation vector")
    return v / n


def letter(v):
    v = unit(v)
    scores = {k: float(np.dot(v, t)) for k, t in VEC.items()}
    k = max(scores, key=scores.get)
    return k, scores[k]


def array_axes(iop):
    # pixel_array [row, col]:
    # axis0 follows DICOM column direction = IOP[3:6]
    # axis1 follows DICOM row direction    = IOP[0:3]
    a = np.asarray(iop, dtype=np.float64)
    if a.shape != (6,):
        raise ValueError(f"IOP shape {a.shape}")
    return unit(a[3:]), unit(a[:3])


def geometry_plane(iop):
    a0, a1 = array_axes(iop)
    normal = unit(np.cross(a1, a0))
    scores = {
        "Axial": abs(float(normal[2])),
        "Coronal": abs(float(normal[1])),
        "Sagittal": abs(float(normal[0])),
    }
    p = max(scores, key=scores.get)
    return p, scores[p]


def canonicalize(image, iop, plane):
    if plane not in TARGET_AXES:
        raise ValueError(plane)
    tletters = TARGET_AXES[plane]
    t0, t1 = VEC[tletters[0]], VEC[tletters[1]]
    a0, a1 = array_axes(iop)
    b0, c0 = letter(a0)
    b1, c1 = letter(a1)
    identity = abs(np.dot(a0, t0)) + abs(np.dot(a1, t1))
    transposed = abs(np.dot(a1, t0)) + abs(np.dot(a0, t1))
    out = np.asarray(image)
    ops = []
    if transposed > identity:
        out = out.T
        n0, n1 = a1.copy(), a0.copy()
        ops.append("transpose")
    else:
        n0, n1 = a0.copy(), a1.copy()
    if np.dot(n0, t0) < 0:
        out = np.flip(out, 0)
        n0 *= -1
        ops.append("flip_axis0")
    if np.dot(n1, t1) < 0:
        out = np.flip(out, 1)
        n1 *= -1
        ops.append("flip_axis1")
    a0l, a0c = letter(n0)
    a1l, a1c = letter(n1)
    diag = {
        "TargetOrientation": "".join(tletters),
        "BeforeAxis0": b0,
        "BeforeAxis1": b1,
        "BeforeAxis0DominantCosine": c0,
        "BeforeAxis1DominantCosine": c1,
        "AfterAxis0": a0l,
        "AfterAxis1": a1l,
        "TargetAxis0AlignmentCosine": float(np.dot(unit(n0), t0)),
        "TargetAxis1AlignmentCosine": float(np.dot(unit(n1), t1)),
        "Operations": "+".join(ops) if ops else "none",
    }
    return np.ascontiguousarray(out), diag


def scalar_position(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        return None
    try:
        row = np.asarray(iop[:3], float)
        col = np.asarray(iop[3:], float)
        return float(np.dot(np.asarray(ipp, float), np.cross(row, col)))
    except Exception:
        return None


def sorted_headers(study, series):
    paths = sorted((TRAIN_SERIES_ROOT / study / series).glob("*.dcm"))
    rec = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            rec.append(
                {
                    "path": str(p),
                    "pos": scalar_position(ds),
                    "inst": getattr(ds, "InstanceNumber", 0),
                }
            )
        except Exception:
            rec.append({"path": str(p), "pos": None, "inst": 0})
    if rec and all(x["pos"] is not None for x in rec):
        rec.sort(key=lambda x: x["pos"])
    else:

        def ik(x):
            try:
                return float(x["inst"])
            except:
                return 0.0

        rec.sort(key=ik)
    return rec


def decode_nearest(rec, target):
    cand = [target]
    for d in range(1, len(rec)):
        if target - d >= 0:
            cand.append(target - d)
        if target + d < len(rec):
            cand.append(target + d)
    last = None
    for i in cand:
        try:
            ds = pydicom.dcmread(rec[i]["path"], force=True)
            arr = ds.pixel_array
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise RuntimeError(f"Expected 2D, got {arr.shape}")
            arr = arr.astype(np.float32)
            arr = arr * float(getattr(ds, "RescaleSlope", 1.0) or 1.0) + float(
                getattr(ds, "RescaleIntercept", 0.0) or 0.0
            )
            return arr, ds, rec[i]["path"]
        except Exception as e:
            last = e
    raise RuntimeError("No decodable slice in representative series.") from last


def dcm_count(study, series):
    p = TRAIN_SERIES_ROOT / study / series
    try:
        return sum(
            1 for e in os.scandir(p) if e.is_file() and e.name.lower().endswith(".dcm")
        )
    except Exception:
        return 0


def choose_representative_series():
    s = pd.read_csv(TRAIN_SERIES_CSV)
    s[UID] = s[UID].astype(str)
    s["SeriesInstanceUID"] = s["SeriesInstanceUID"].astype(str)
    need = {UID, "SeriesInstanceUID", "Anatomical_Plane"}
    if need - set(s.columns):
        raise RuntimeError(f"Missing series columns: {sorted(need-set(s.columns))}")
    req = {"Axial", "Coronal", "Sagittal"}
    studies = []
    for uid, g in s.groupby(UID):
        if req.issubset(set(g["Anatomical_Plane"].dropna().astype(str))):
            studies.append(str(uid))
    studies = sorted(studies)[:N_SAMPLE_STUDIES]
    if len(studies) < N_SAMPLE_STUDIES:
        raise RuntimeError("Not enough three-plane studies.")
    rows = []
    for uid in studies:
        sg = s[s[UID] == uid]
        for plane in ("Axial", "Coronal", "Sagittal"):
            g = sg[sg["Anatomical_Plane"] == plane]
            scored = [
                (
                    dcm_count(uid, str(r["SeriesInstanceUID"])),
                    str(r["SeriesInstanceUID"]),
                )
                for _, r in g.iterrows()
            ]
            scored.sort(reverse=True)
            nd, suid = scored[0]
            rows.append(
                {
                    UID: uid,
                    "Anatomical_Plane": plane,
                    "SeriesInstanceUID": suid,
                    "NDicom": nd,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------- Curia model -----------------------------


def load_curia(root):
    import transformers
    from transformers import AutoModel, AutoImageProcessor

    log(f"Transformers version : {transformers.__version__}")
    processor = AutoImageProcessor.from_pretrained(
        str(root), trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(str(root), local_files_only=True)
    return processor, model


def validate_config(model, processor):
    c = model.config
    d = {
        "hidden_size": int(c.hidden_size),
        "num_channels": int(c.num_channels),
        "image_size": int(c.image_size),
        "patch_size": int(c.patch_size),
        "num_hidden_layers": int(c.num_hidden_layers),
        "num_attention_heads": int(c.num_attention_heads),
        "processor_crop_size": int(processor.crop_size),
        "processor_clip_below_air": bool(processor.clip_below_air),
        "processor_do_resize": bool(processor.do_resize),
        "processor_do_normalize": bool(processor.do_normalize),
    }
    expected = {
        "hidden_size": EXPECTED_HIDDEN,
        "num_channels": EXPECTED_CHANNELS,
        "image_size": EXPECTED_IMAGE,
        "patch_size": EXPECTED_PATCH,
        "processor_crop_size": EXPECTED_IMAGE,
    }
    for k, v in expected.items():
        if d[k] != v:
            raise RuntimeError(f"Curia config mismatch {k}: {d[k]} != {v}")
    return d


def prepare_samples(rep, processor):
    images = []
    rows = []
    previews = []
    for _, r in rep.iterrows():
        study = str(r[UID])
        series = str(r["SeriesInstanceUID"])
        plane = str(r["Anatomical_Plane"])
        h = sorted_headers(study, series)
        if not h:
            raise RuntimeError(f"No DICOMs: {study}/{series}")
        raw, ds, path = decode_nearest(h, len(h) // 2)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None:
            raise RuntimeError(f"Missing IOP: {path}")
        gplane, gcos = geometry_plane(iop)
        canon, odiag = canonicalize(raw, iop, plane)
        pv = processor(images=canon, return_tensors="pt")["pixel_values"]
        if tuple(pv.shape) != (1, 1, EXPECTED_IMAGE, EXPECTED_IMAGE):
            raise RuntimeError(f"Processor output shape {tuple(pv.shape)}")
        rows.append(
            {
                UID: study,
                "SeriesInstanceUID": series,
                "MetadataPlane": plane,
                "GeometryPlane": gplane,
                "GeometryPlaneCosine": gcos,
                "MetadataGeometryPlaneMatch": plane == gplane,
                "NDicom": int(r["NDicom"]),
                "RawRows": raw.shape[0],
                "RawColumns": raw.shape[1],
                "CanonicalRows": canon.shape[0],
                "CanonicalColumns": canon.shape[1],
                "RawMin": float(np.nanmin(raw)),
                "RawMax": float(np.nanmax(raw)),
                "RawMean": float(np.nanmean(raw)),
                "RawStd": float(np.nanstd(raw)),
                "ProcessorMean": float(pv.mean()),
                "ProcessorStd": float(pv.std()),
                "ProcessorMin": float(pv.min()),
                "ProcessorMax": float(pv.max()),
                "DICOMPath": path,
                **odiag,
            }
        )
        images.append(canon)
        previews.append((study, plane, canon))
    return images, pd.DataFrame(rows), previews


def save_preview(previews):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    studies = list(dict.fromkeys(x[0] for x in previews))
    planes = ["Axial", "Coronal", "Sagittal"]
    lookup = {(s, p): im for s, p, im in previews}
    fig, ax = plt.subplots(
        len(studies), 3, figsize=(12, 4 * len(studies)), squeeze=False
    )
    for i, s in enumerate(studies):
        for j, p in enumerate(planes):
            im = lookup[(s, p)]
            finite = im[np.isfinite(im)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
            ax[i, j].imshow(im, cmap="gray", vmin=lo, vmax=hi)
            ax[i, j].set_title(f"{s[-12:]}\n{p} -> {''.join(TARGET_AXES[p])}")
            ax[i, j].axis("off")
    fig.tight_layout()
    path = RESULT_ROOT / "curia2_canonical_orientation_preview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.inference_mode()
def forward(model, pv, fp16=False):
    if fp16 and DEVICE.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return model(pixel_values=pv)
    return model(pixel_values=pv)


def embeddings(output):
    d = {}
    if hasattr(output, "last_hidden_state"):
        t = output.last_hidden_state
        d["cls_token"] = t[:, 0]
        if t.shape[1] > 1:
            d["mean_patch_token"] = t[:, 1:].mean(1)
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        d["pooler_output"] = output.pooler_output
    if not d:
        raise RuntimeError("No usable Curia embedding output.")
    return d


def cosine_rows(a, b):
    a = a.float().cpu()
    b = b.float().cpu()
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (a * b).sum(1).numpy()


def embedding_test(images, processor, model):
    pv = processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)
    model = model.to(DEVICE).eval()
    o32 = forward(model, pv, False)
    e32 = embeddings(o32)
    e16 = None
    if DEVICE.type == "cuda":
        e16 = embeddings(forward(model, pv, True))
    shapes = {k: list(v.shape) for k, v in e32.items()}
    if hasattr(o32, "last_hidden_state"):
        shapes["last_hidden_state"] = list(o32.last_hidden_state.shape)
    prec = []
    if e16:
        for k in sorted(set(e32) & set(e16)):
            c = cosine_rows(e32[k], e16[k])
            diff = (e32[k].float().cpu() - e16[k].float().cpu()).abs().numpy()
            prec.append(
                {
                    "Embedding": k,
                    "MeanCosineFP32vsFP16": float(c.mean()),
                    "MinCosineFP32vsFP16": float(c.min()),
                    "MeanAbsoluteDifference": float(diff.mean()),
                    "MaxAbsoluteDifference": float(diff.max()),
                }
            )
    selected = "pooler_output" if "pooler_output" in e32 else "cls_token"
    z = e32[selected].float().cpu()
    zn = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = zn @ zn.T
    off = sim[~torch.eye(len(z), dtype=torch.bool)]
    summary = {
        "processor_pixel_values_shape": list(pv.shape),
        "output_shapes": shapes,
        "selected_embedding": selected,
        "selected_embedding_dim": int(z.shape[1]),
        "selected_embedding_mean_norm": float(z.norm(dim=1).mean()),
        "cross_slice_cosine_mean_off_diagonal": (
            float(off.mean()) if off.numel() else float("nan")
        ),
        "cross_slice_cosine_min_off_diagonal": (
            float(off.min()) if off.numel() else float("nan")
        ),
        "cross_slice_cosine_max_off_diagonal": (
            float(off.max()) if off.numel() else float("nan")
        ),
    }
    return summary, pd.DataFrame(prec), pv[:1].detach(), model


def benchmark(model, one):
    rows = []
    if DEVICE.type != "cuda":
        return pd.DataFrame(rows)
    for bs in BENCH_BATCH_SIZES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)
            x = one.repeat(bs, 1, 1, 1).contiguous()
            _ = forward(model, x, True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(BENCH_ITERS):
                _ = forward(model, x, True)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t
            rows.append(
                {
                    "BatchSize": bs,
                    "Iterations": BENCH_ITERS,
                    "Seconds": sec,
                    "SlicesPerSecond": bs * BENCH_ITERS / max(sec, 1e-9),
                    "MillisecondsPerBatch": 1000 * sec / BENCH_ITERS,
                    "PeakAllocatedGB": torch.cuda.max_memory_allocated(DEVICE)
                    / (1024**3),
                    "PeakReservedGB": torch.cuda.max_memory_reserved(DEVICE)
                    / (1024**3),
                    "Status": "PASS",
                }
            )
            log(
                f"FP16 batch={bs:2d} slices/s={rows[-1]['SlicesPerSecond']:.2f} "
                f"peak={rows[-1]['PeakAllocatedGB']:.2f} GB"
            )
            del x
        except torch.cuda.OutOfMemoryError:
            rows.append(
                {
                    "BatchSize": bs,
                    "Iterations": BENCH_ITERS,
                    "Seconds": float("nan"),
                    "SlicesPerSecond": float("nan"),
                    "MillisecondsPerBatch": float("nan"),
                    "PeakAllocatedGB": float("nan"),
                    "PeakReservedGB": float("nan"),
                    "Status": "OOM",
                }
            )
            log(f"FP16 batch={bs:2d}: OOM")
            gc.collect()
            torch.cuda.empty_cache()
            break
    return pd.DataFrame(rows)


# ----------------------------- orchestration -----------------------------


def run_preflight():
    t0 = time.time()
    log("=" * 88)
    log("RSNA W4.0 CURIA-2 TECHNICAL PREFLIGHT")
    log("=" * 88)
    log(f"Device               : {DEVICE}")
    log(f"CUDA GPU count       : {GPU_COUNT}")
    if DEVICE.type == "cuda":
        for i in range(GPU_COUNT):
            log(f"GPU {i}                : {torch.cuda.get_device_name(i)}")
    if not TRAIN_SERIES_CSV.exists():
        raise FileNotFoundError(TRAIN_SERIES_CSV)
    if not TRAIN_SERIES_ROOT.exists():
        raise FileNotFoundError(TRAIN_SERIES_ROOT)

    root = discover_curia2()
    if root is None:
        raise FileNotFoundError(
            "Curia-2 not found. Set W40_CURIA_ROOT to the attached model directory "
            "or run run_curia2_preflight('download') first."
        )
    log(f"Curia-2 root         : {root}")
    weights = root / "model.safetensors"
    log(f"Curia weights size   : {weights.stat().st_size/(1024**2):.2f} MB")
    log("Computing model SHA256 ...")
    wsha = sha256_file(weights)
    sha_match = wsha == EXPECTED_MODEL_SHA256
    log(f"Model SHA256          : {wsha}")
    log(f"Expected SHA match    : {sha_match}")

    processor, model = load_curia(root)
    cfg = validate_config(model, processor)
    params = sum(p.numel() for p in model.parameters())
    log(f"Parameters            : {params:,}")
    log("Model/processor config:\n" + json.dumps(cfg, indent=2))

    rep = choose_representative_series()
    rep.to_csv(RESULT_ROOT / "representative_series.csv", index=False)
    log("\nRepresentative series:\n" + rep.to_string(index=False))

    images, diag, previews = prepare_samples(rep, processor)
    diag.to_csv(
        RESULT_ROOT / "slice_orientation_processor_diagnostics.csv", index=False
    )
    preview = save_preview(previews)
    cols = [
        UID,
        "MetadataPlane",
        "GeometryPlane",
        "MetadataGeometryPlaneMatch",
        "TargetOrientation",
        "BeforeAxis0",
        "BeforeAxis1",
        "Operations",
        "TargetAxis0AlignmentCosine",
        "TargetAxis1AlignmentCosine",
        "ProcessorMean",
        "ProcessorStd",
    ]
    log("\nOrientation diagnostics:\n" + diag[cols].to_string(index=False))

    esum, pdf, one, model = embedding_test(images, processor, model)
    pdf.to_csv(RESULT_ROOT / "fp32_fp16_embedding_comparison.csv", index=False)
    log("\nEmbedding summary:\n" + json.dumps(esum, indent=2))
    if len(pdf):
        log("\nFP32 vs FP16:\n" + pdf.to_string(index=False))

    bdf = benchmark(model, one)
    bdf.to_csv(RESULT_ROOT / "gpu_throughput_benchmark.csv", index=False)
    good = bdf[bdf["Status"] == "PASS"] if len(bdf) else bdf
    if len(good):
        best = good.sort_values("SlicesPerSecond", ascending=False).iloc[0]
        rec_batch = int(best["BatchSize"])
        best_rate = float(best["SlicesPerSecond"])
    else:
        rec_batch = 1
        best_rate = float("nan")

    plane_matches = int(diag["MetadataGeometryPlaneMatch"].sum())
    min_cos = min(
        float(diag["TargetAxis0AlignmentCosine"].min()),
        float(diag["TargetAxis1AlignmentCosine"].min()),
    )
    processor_ok = bool((diag["ProcessorStd"].between(0.5, 1.5)).all())
    embed_ok = esum["selected_embedding_dim"] == EXPECTED_HIDDEN
    orientation_ok = min_cos >= 0.70
    plane_ok = plane_matches == len(diag)
    fp16_ok = True if not len(pdf) else bool((pdf["MinCosineFP32vsFP16"] > 0.995).all())
    overall = bool(
        processor_ok and embed_ok and orientation_ok and plane_ok and fp16_ok
    )

    summary = {
        "overall_pass": overall,
        "curia_repo_id": CURIA_REPO_ID,
        "curia_revision": CURIA_REVISION,
        "curia_root": str(root),
        "model_sha256": wsha,
        "expected_model_sha256_match": sha_match,
        "parameter_count": params,
        "device": str(DEVICE),
        "gpu_count": GPU_COUNT,
        "sample_studies": N_SAMPLE_STUDIES,
        "sample_series": len(diag),
        "metadata_geometry_plane_matches": plane_matches,
        "metadata_geometry_plane_total": len(diag),
        "minimum_target_orientation_alignment_cosine": min_cos,
        "processor_std_ok": processor_ok,
        "embedding_dim_ok": embed_ok,
        "orientation_ok": orientation_ok,
        "plane_match_ok": plane_ok,
        "fp16_ok": fp16_ok,
        "embedding": esum,
        "recommended_single_gpu_fp16_batch_size": rec_batch,
        "best_single_gpu_slices_per_second": best_rate,
        "preview_path": str(preview) if preview else None,
        "runtime_seconds": time.time() - t0,
    }
    (RESULT_ROOT / "CURIA2_PREFLIGHT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    log("\n" + "=" * 88)
    log("CURIA-2 TECHNICAL PREFLIGHT: " + ("PASS" if overall else "REVIEW REQUIRED"))
    log("=" * 88)
    log(f"Selected embedding    : {esum['selected_embedding']}")
    log(f"Embedding dim         : {esum['selected_embedding_dim']}")
    log(f"Plane matches         : {plane_matches}/{len(diag)}")
    log(f"Min orientation cos   : {min_cos:.4f}")
    log(f"Recommended FP16 batch: {rec_batch}")
    if np.isfinite(best_rate):
        log(f"Best single-GPU rate  : {best_rate:.2f} slices/s")
        if GPU_COUNT >= 2:
            log(
                f"Rough 2-GPU upper bound: {2*best_rate:.2f} slices/s "
                "(independent workers; not yet benchmarked)"
            )
    log(f"Results               : {RESULT_ROOT}")
    log(f"Runtime               : {time.time()-t0:.1f} s")
    return summary


def run_curia2_preflight(mode="preflight"):
    mode = str(mode).strip().lower()
    if mode not in {"download", "preflight", "all"}:
        raise ValueError(mode)
    if mode == "download":
        return download_curia2()
    if mode == "all" and discover_curia2() is None:
        download_curia2()
    return run_preflight()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["download", "preflight", "all"], default="preflight"
    )
    return p.parse_args()


def main():
    run_curia2_preflight(parse_args().mode)


if __name__ == "__main__":
    main()
