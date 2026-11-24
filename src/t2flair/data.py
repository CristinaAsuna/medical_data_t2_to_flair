"""Read the per-subject .npz files written by t2flair.prep (contract: docs/DATA.md)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

KEYS = ("t2", "flair", "brain", "tumor", "keep", "z")


def load_volume(processed, subject):
    with np.load(Path(processed) / (subject + ".npz")) as f:
        return {k: f[k] for k in KEYS}


def neighbours(n, k):
    """Indices k-1, k, k+1 clamped to [0, n-1] (edge slices repeat the nearest one)."""
    return [min(max(k + d, 0), n - 1) for d in (-1, 0, 1)]


def volume_inputs(vol):
    """All kept slices of one subject as tensors (N, C, H, W), in slice order."""
    n = vol["t2"].shape[0]
    idx = np.flatnonzero(vol["keep"])
    t2 = vol["t2"].astype(np.float32)
    cond = np.stack([t2[neighbours(n, k)] for k in idx])
    as_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    return {
        "cond": as_t(cond),
        "source": as_t(t2[idx][:, None]),
        "target": as_t(vol["flair"][idx].astype(np.float32)[:, None]),
        "brain": as_t(vol["brain"][idx][:, None]),
        "tumor": as_t(vol["tumor"][idx][:, None]),
        "z": vol["z"][idx],
    }


class SliceDataset(Dataset):
    """Every kept slice of the given subjects, held in memory.

    Item: cond (3,H,W) = T2 slices z-1,z,z+1; source (1,H,W) = T2 at z; target = FLAIR at z;
    brain, tumor (1,H,W) masks. flip=True applies the same random horizontal flip to all.
    """

    def __init__(self, processed, subjects, flip=False):
        self.vols = [load_volume(processed, s) for s in subjects]
        self.subjects = list(subjects)
        self.index = [(v, int(k)) for v, vol in enumerate(self.vols) for k in np.flatnonzero(vol["keep"])]
        self.flip = flip

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        v, k = self.index[i]
        vol = self.vols[v]
        n = vol["t2"].shape[0]
        item = {
            "cond": vol["t2"][neighbours(n, k)],
            "source": vol["t2"][k][None],
            "target": vol["flair"][k][None],
            "brain": vol["brain"][k][None],
            "tumor": vol["tumor"][k][None],
        }
        flip = self.flip and torch.rand(()) < 0.5
        out = {}
        for key, a in item.items():
            a = a[..., ::-1] if flip else a
            out[key] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return out
