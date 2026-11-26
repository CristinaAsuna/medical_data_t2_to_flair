"""Image metrics inside the brain mask and lesion-level metrics (definitions: docs/PIPELINE.md §4).

Images arrive in [-1, 1] and are mapped to [0, 1] (data range 1). Volumes are (N, H, W) stacks
of the kept slices of one subject; masks are boolean arrays of the same shape.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def to_unit(a):
    return (np.asarray(a, dtype=np.float64) + 1.0) / 2.0


def psnr(pred, gt, mask):
    err = (to_unit(pred) - to_unit(gt))[mask]
    mse = float(np.mean(err ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def mae(pred, gt, mask):
    return float(np.mean(np.abs(to_unit(pred) - to_unit(gt))[mask]))


def _gauss(size=11, sigma=1.5):
    x = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(pred, gt, mask):
    """Mean of the per-pixel SSIM map (11x11 Gaussian, sigma 1.5) over brain pixels of all slices."""
    a = torch.from_numpy(to_unit(pred))[:, None]
    b = torch.from_numpy(to_unit(gt))[:, None]
    w = _gauss()
    blur = lambda z: F.conv2d(F.pad(z, (5, 5, 5, 5), mode="reflect"), w)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(smap[:, 0].numpy()[mask].mean())


def _ball(radius):
    r = np.arange(-radius, radius + 1)
    zz, yy, xx = np.meshgrid(r, r, r, indexing="ij")
    return zz ** 2 + yy ** 2 + xx ** 2 <= radius ** 2


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(p, g).sum() / s)


# Diagonal of the 240x240x155 BraTS volume in voxels, sqrt(240^2 + 240^2 + 155^2): the HD95 that
# BraTS assigns when exactly one of prediction and ground truth is empty (D-006).
HD95_EMPTY = 373.13


def hd95(p, g, empty_penalty=HD95_EMPTY):
    """Symmetric 95th-percentile Hausdorff distance in voxels (1 mm isotropic).

    Both masks empty: 0. Exactly one empty: empty_penalty (never nan, so no subject is dropped).
    """
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float(empty_penalty)
    surf = lambda m: np.logical_xor(m, ndimage.binary_erosion(m))
    sp, sg = surf(p), surf(g)
    d_pg = ndimage.distance_transform_edt(~sg)[sp]
    d_gp = ndimage.distance_transform_edt(~sp)[sg]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def _components(mask):
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    index = np.arange(1, n + 1)
    sizes = ndimage.sum(np.ones_like(mask, dtype=np.int64), labels, index=index) if n else np.zeros(0)
    return labels, index, np.asarray(sizes)


def _touches(labels, index, other):
    """For each component, whether it overlaps the boolean mask `other`."""
    if index.size == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(ndimage.maximum(other.astype(np.uint8), labels, index=index)) > 0


def lesion_metrics(p, g, fp_dilate=5, small_voxels=1000, empty_tau=20, fp_lesion_voxels=10, gt_lesion_voxels=50,
                   hd95_empty=HD95_EMPTY):
    """p: predicted tumor mask, g: GT mask, both (N, H, W) bool. Components use 26-connectivity.

    Per subject: Dice, HD95, FP volume (voxels of p farther than fp_dilate from g), miss rate;
    lesion counts: fp_lesions = components of p with >= fp_lesion_voxels that do not touch
    dilate(g, fp_dilate) (invented lesions); missed_lesions / gt_lesions = components of g with
    >= gt_lesion_voxels that p does not touch; lesion_detection = 1 - missed/gt. Raw counts for
    the pooled small-lesion recall and empty-slice false alarms (exploratory) are also returned.
    """
    near = ndimage.binary_dilation(g, structure=_ball(fp_dilate))
    g_lab, g_idx, g_size = _components(g)
    p_lab, p_idx, p_size = _components(p)
    g_hit = _touches(g_lab, g_idx, p)
    p_near = _touches(p_lab, p_idx, near)
    small = g_size < small_voxels
    counted = g_size >= gt_lesion_voxels
    gt_lesions = int(counted.sum())
    missed = int((counted & ~g_hit).sum())
    empty = ~g.reshape(g.shape[0], -1).any(axis=1)
    false_slices = p.reshape(p.shape[0], -1).sum(axis=1) > empty_tau
    return {
        "dice": dice(p, g),
        "hd95": hd95(p, g, hd95_empty),
        "fp_volume": int(np.logical_and(p, ~near).sum()),
        "miss_rate": float(1.0 - np.logical_and(p, g).sum() / g.sum()) if g.any() else float("nan"),
        "fp_lesions": int(((p_size >= fp_lesion_voxels) & ~p_near).sum()),
        "missed_lesions": missed,
        "gt_lesions": gt_lesions,
        "lesion_detection": 1.0 - missed / gt_lesions if gt_lesions else float("nan"),
        "empty_prediction": bool(not p.any()),
        "pred_volume": int(p.sum()),
        "gt_volume": int(g.sum()),
        "small_lesions": int(small.sum()),
        "small_detected": int((small & g_hit).sum()),
        "empty_slices": int(empty.sum()),
        "empty_slices_flagged": int(np.logical_and(empty, false_slices).sum()),
    }


def summarize(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size)}
