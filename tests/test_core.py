"""L0 tests: data contract, splits, bridge maths, metrics. Run: python -m pytest tests -q"""
import numpy as np
import pytest
import torch

from t2flair import bridge, metrics, prep, splits


def synthetic_volume(size=48, depth=20, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.35) ** 2
    brain = np.repeat(disk[:, :, None], depth, axis=2)
    brain[:, :, :3] = False  # first slices empty
    t2 = (rng.uniform(50, 400, brain.shape) * brain).astype(np.int16)
    flair = (rng.uniform(30, 300, brain.shape) * brain).astype(np.int16)
    seg = np.zeros(brain.shape, np.uint8)
    seg[size // 2 - 3:size // 2 + 3, size // 2 - 3:size // 2 + 3, 8:12] = 2
    return t2, flair, seg


def test_normalize_range_and_background():
    t2, flair, _ = synthetic_volume()
    brain = (t2 > 0) | (flair > 0)
    out = prep.normalize(t2, brain)
    assert out.min() >= -1 and out.max() <= 1
    assert np.all(out[~brain] == -1)
    assert np.isclose(out[brain].max(), 1.0)


def test_process_arrays_contract():
    t2, flair, seg = synthetic_volume()
    arrays, stats = prep.process_arrays(t2, flair, seg, protocol="main", crop=40)
    assert arrays["t2"].shape[1:] == (40, 40) and arrays["t2"].dtype == np.float16
    keep, z = arrays["keep"], arrays["z"]
    assert keep.sum() == stats["n_keep"] == 17
    assert z[0] == 2 and not keep[0]  # one slice of margin before the first kept slice
    assert arrays["tumor"].sum() == 6 * 6 * 4
    assert stats["tumor_lost_by_crop"] == 0


def test_process_arrays_lit_slices():
    t2, flair, seg = synthetic_volume(depth=155)
    arrays, _ = prep.process_arrays(t2, flair, seg, protocol="lit", crop=40)
    assert arrays["z"][arrays["keep"]].tolist() == list(range(27, 127))


def test_splits_disjoint_and_deterministic():
    ids = ["BraTS2021_%05d" % i for i in range(1251)]
    a, b = splits.main_split(ids), splits.main_split(ids)
    assert a == b
    assert {k: len(v) for k, v in a.items()} == splits.MAIN
    assert not (set(a["train"]) & set(a["val"]) or set(a["train"]) & set(a["test"]) or set(a["val"]) & set(a["test"]))
    lit = splits.lit_split(ids)
    assert {k: len(v) for k, v in lit.items()} == splits.LIT


def test_committed_main_split_is_disjoint():
    s = {n: set(splits.load_split("configs/splits/main", n)) for n in ("train", "val", "test")}
    assert [len(s[n]) for n in ("train", "val", "test")] == [300, 30, 100]
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"])


def test_interpolate_endpoints():
    y, x, noise = torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    assert torch.allclose(bridge.interpolate(y, x, torch.zeros(2), 0.5, noise), y)
    assert torch.allclose(bridge.interpolate(y, x, torch.ones(2), 0.5, noise), x)


class Oracle(torch.nn.Module):
    """Always predicts the true endpoint; records x_t at a chosen step."""

    def __init__(self, x, watch_t):
        super().__init__()
        self.x, self.watch_t, self.seen = x, watch_t, None

    def forward(self, x_t, t, cond):
        if abs(float(t[0]) - self.watch_t) < 1e-6:  # t arrives as float32
            self.seen = x_t.clone()
        return self.x


def test_sampler_marginal_matches_training_interpolation():
    n, sigma, steps = 20000, 0.5, 10
    y = torch.full((n, 1, 1, 1), -0.6)
    x = torch.full((n, 1, 1, 1), 0.4)
    oracle = Oracle(x, watch_t=0.6)
    out = bridge.sample(oracle, y, None, steps, sigma, torch.Generator().manual_seed(0))
    assert torch.allclose(out, x)
    seen = oracle.seen.flatten()
    t = 0.6
    assert abs(seen.mean().item() - ((1 - t) * -0.6 + t * 0.4)) < 0.01
    assert abs(seen.var().item() - sigma ** 2 * t * (1 - t)) < 0.003


def test_noise_fm_path_and_sampler():
    """F0: t=0 is pure noise, t=1 the target; with an exact x_hat the Euler sampler lands on x and,
    at an intermediate step, sits on the training path (1 - t) eps + t x."""
    x, noise = torch.ones(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    assert torch.allclose(bridge.noise_interpolate(x, torch.zeros(2), noise), noise)
    assert torch.allclose(bridge.noise_interpolate(x, torch.ones(2), noise), x)
    n, steps = 20000, 10
    x = torch.full((n, 1, 1, 1), 0.4)
    oracle = Oracle(x, watch_t=0.6)
    out = bridge.noise_sample(oracle, x, None, steps, torch.Generator().manual_seed(0))
    assert torch.allclose(out, x)
    seen = oracle.seen.flatten()
    assert abs(seen.mean().item() - 0.6 * 0.4) < 0.01
    assert abs(seen.var().item() - 0.4 ** 2) < 0.01


def test_image_metrics():
    gt = np.random.default_rng(0).uniform(-1, 1, (3, 32, 32))
    mask = np.ones_like(gt, dtype=bool)
    assert metrics.psnr(gt, gt, mask) == float("inf")
    assert np.isclose(metrics.ssim(gt, gt, mask), 1.0)
    noisy = np.clip(gt + 0.1, -1, 1)
    assert 20 < metrics.psnr(noisy, gt, mask) < 30
    assert metrics.ssim(noisy, gt, mask) < 1.0


def test_lesion_metrics():
    g = np.zeros((10, 40, 40), bool)
    g[3:6, 10:20, 10:20] = True
    g[8, 30, 30] = True  # one small lesion
    p = g.copy()
    p[8, 30, 30] = False  # miss the small lesion
    p[2, 35:38, 0:3] = True  # far false positive on an empty slice
    m = metrics.lesion_metrics(p, g, fp_dilate=3, small_voxels=10, empty_tau=5)
    assert m["small_lesions"] == 1 and m["small_detected"] == 0
    assert m["fp_volume"] == 9
    assert m["empty_slices_flagged"] == 1
    assert 0.9 < m["dice"] < 1.0
    assert metrics.lesion_metrics(g, g)["hd95"] == 0.0


def test_generator_shapes():
    pytest.importorskip("diffusers")
    from t2flair.models import Generator
    cfg = {"base_channels": 16, "channel_mult": [1, 2, 2, 4], "layers_per_block": 1, "attn_levels": [2, 3]}
    net = Generator(cfg, sample_size=64)
    out = net(torch.randn(2, 1, 64, 64), torch.rand(2), torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_generator_channels_last_same_output():
    pytest.importorskip("diffusers")
    from t2flair.models import Generator
    cfg = {"base_channels": 16, "channel_mult": [1, 2, 2, 4], "layers_per_block": 1, "attn_levels": [2, 3],
           "attn_head_dim": 8}
    torch.manual_seed(0)
    a = Generator(cfg, sample_size=64).eval()
    b = Generator(dict(cfg, channels_last=True), sample_size=64).eval()
    b.load_state_dict(a.state_dict())
    args = torch.randn(2, 1, 64, 64), torch.rand(2), torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(a(*args), b(*args), atol=1e-4)


def test_hd95_empty_penalty_and_lesion_counts():
    g = np.zeros((10, 40, 40), bool)
    g[3:7, 5:15, 5:15] = True                      # one lesion, 400 voxels
    assert metrics.hd95(np.zeros_like(g), g) == metrics.HD95_EMPTY
    assert metrics.hd95(np.zeros_like(g), np.zeros_like(g)) == 0.0
    p = np.zeros_like(g)
    p[4, 30:34, 30:34] = True                      # 16-voxel invented lesion far away
    p[1, 0, 39] = True                             # 1-voxel speck: below fp_lesion_voxels
    m = metrics.lesion_metrics(p, g, fp_dilate=2, fp_lesion_voxels=10, gt_lesion_voxels=50)
    assert m["fp_lesions"] == 1 and m["missed_lesions"] == 1 and m["gt_lesions"] == 1
    assert m["lesion_detection"] == 0.0 and not m["empty_prediction"]
    p[5, 8, 8] = True                              # now touches the GT lesion
    m = metrics.lesion_metrics(p, g, fp_dilate=2, fp_lesion_voxels=10, gt_lesion_voxels=50)
    assert m["missed_lesions"] == 0 and m["lesion_detection"] == 1.0 and m["fp_lesions"] == 1


def test_compare_paired_stats_and_holm():
    from t2flair import compare
    rng = np.random.default_rng(0)
    base = rng.normal(20, 2, 30)
    a = {"s%d" % i: {"subject": "s%d" % i, "psnr": base[i] + 1.0, "empty_slices": 10, "empty_slices_flagged": 1}
         for i in range(30)}
    b = {"s%d" % i: {"subject": "s%d" % i, "psnr": base[i], "empty_slices": 10, "empty_slices_flagged": 3}
         for i in range(30)}
    st = compare.compare_metric(a, b, "psnr")
    assert abs(st["diff"] - 1.0) < 1e-9 and st["p"] < 1e-5 and st["ci95"][0] <= 1.0 <= st["ci95"][1]
    assert compare.noninferior(st, "higher", "0.5", st["mean_b"])["passes"]
    pooled = compare.compare_metric(a, b, "empty_slice_false_alarm")
    assert abs(pooled["diff"] - (-0.2)) < 1e-9
    assert compare.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
