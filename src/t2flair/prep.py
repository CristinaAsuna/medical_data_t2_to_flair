"""Preprocess BraTS 2021 volumes into per-subject .npz files (contract: docs/DATA.md).

    python -m t2flair.prep --raw $LAB_DATA_DIR/brats2021/raw --splits configs/splits/main \
        --protocol main --out $LAB_DATA_DIR/brats2021/processed/main --workers 16

Each <subject>.npz holds a contiguous block of axial slices (one slice of margin around the
kept ones, so 2.5D neighbours exist): t2, flair (float16 in [-1, 1], background -1),
brain, tumor (uint8 masks), keep (bool: slice is a training/evaluation sample), z (index in
the original 155-slice volume). manifest.json records counts and sha256 per file.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .splits import load_split

FULL = 240
PERCENTILE = 99.5
MIN_BRAIN_FRACTION = 0.10
LIT_SLICES = (27, 127)


def normalize(volume, brain):
    """Clip at the 99.5th percentile of this modality inside the brain, map [0, p] -> [-1, 1].

    Voxels outside the brain mask are set to -1. Works per volume, never per slice.
    """
    volume = volume.astype(np.float32)
    inside = volume[(volume > 0) & brain]
    if inside.size == 0:
        raise ValueError("modality is empty inside the brain mask")
    top = float(np.percentile(inside, PERCENTILE))
    out = np.clip(volume, 0.0, top) / top * 2.0 - 1.0
    out[~brain] = -1.0
    return out


def kept_slices(brain, protocol):
    """Boolean per axial slice (last axis) saying whether it is a sample."""
    depth = brain.shape[-1]
    if protocol == "main":
        fraction = brain.reshape(-1, depth).mean(axis=0)
        return fraction > MIN_BRAIN_FRACTION
    if protocol == "lit":
        keep = np.zeros(depth, dtype=bool)
        keep[LIT_SLICES[0]:LIT_SLICES[1]] = True
        return keep
    raise ValueError(protocol)


def center_crop(array, size):
    """Crop the two in-plane axes (first two) of an (H, W, Z) array to size x size."""
    h, w = array.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return array[top:top + size, left:left + size]


def process_arrays(t2, flair, seg, protocol="main", crop=192):
    """Pure-array core of preprocessing; returns the dict stored in the .npz plus stats."""
    if not (t2.shape == flair.shape == seg.shape):
        raise ValueError("shape mismatch t2 %s flair %s seg %s" % (t2.shape, flair.shape, seg.shape))
    brain = (t2 > 0) | (flair > 0)
    t2n, flairn = normalize(t2, brain), normalize(flair, brain)
    tumor = seg > 0
    keep = kept_slices(brain, protocol)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("no slice passes the selection rule")
    lo, hi = max(0, idx[0] - 1), min(brain.shape[-1] - 1, idx[-1] + 1)
    block = slice(lo, hi + 1)
    cropped_brain = center_crop(brain, crop)
    lost = 1.0 - cropped_brain[..., keep].sum() / max(1, brain[..., keep].sum())
    lost_tumor = 1.0 - center_crop(tumor, crop)[..., keep].sum() / max(1, tumor[..., keep].sum())

    def stack(volume, dtype):  # (H, W, Z) -> (Z, crop, crop)
        return np.ascontiguousarray(np.moveaxis(center_crop(volume, crop)[..., block], -1, 0)).astype(dtype)

    arrays = {
        "t2": stack(t2n, np.float16),
        "flair": stack(flairn, np.float16),
        "brain": stack(brain, np.uint8),
        "tumor": stack(tumor, np.uint8),
        "keep": keep[block].copy(),
        "z": np.arange(lo, hi + 1, dtype=np.int16),
    }
    stats = {"n_keep": int(keep.sum()), "n_block": int(hi - lo + 1), "brain_lost_by_crop": float(lost),
             "tumor_lost_by_crop": float(lost_tumor), "tumor_voxels": int(tumor[..., keep].sum())}
    return arrays, stats


def load_subject(raw, subject):
    import nibabel as nib
    folder = Path(raw) / subject
    get = lambda name: np.asanyarray(nib.load(str(folder / "{}_{}.nii.gz".format(subject, name))).dataobj)
    return get("t2"), get("flair"), get("seg")


def _one(job):
    raw, subject, out, protocol, crop = job
    target = Path(out) / (subject + ".npz")
    t2, flair, seg = load_subject(raw, subject)
    arrays, stats = process_arrays(np.asarray(t2), np.asarray(flair), np.asarray(seg), protocol, crop)
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)
    stats["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return subject, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True)
    parser.add_argument("--splits", required=True, help="prefix, e.g. configs/splits/main")
    parser.add_argument("--protocol", choices=["main", "lit"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subjects = {name: load_split(args.splits, name) for name in ("train", "val", "test")}
    jobs = [(args.raw, s, str(out), args.protocol, args.crop) for ids in subjects.values() for s in ids]
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (subject, stats) in enumerate(pool.map(_one, jobs, chunksize=4), 1):
            results[subject] = stats
            if i % 50 == 0 or i == len(jobs):
                print("prep %d/%d" % (i, len(jobs)), flush=True)
    worst = max(results.values(), key=lambda s: s["tumor_lost_by_crop"])
    manifest = {
        "protocol": args.protocol, "crop": args.crop, "percentile": PERCENTILE,
        "min_brain_fraction": MIN_BRAIN_FRACTION if args.protocol == "main" else None,
        "slices": None if args.protocol == "main" else list(LIT_SLICES),
        "counts": {k: len(v) for k, v in subjects.items()},
        "kept_slices": {k: sum(results[s]["n_keep"] for s in v) for k, v in subjects.items()},
        "max_brain_lost_by_crop": max(s["brain_lost_by_crop"] for s in results.values()),
        "max_tumor_lost_by_crop": worst["tumor_lost_by_crop"],
        "subjects": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "subjects"}, indent=1))


if __name__ == "__main__":
    main()
