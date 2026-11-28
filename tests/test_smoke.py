"""End-to-end smoke on synthetic 64x64 volumes: train -> train_seg -> infer -> evaluate -> figures.

Outputs go to scratch/smoke/ and a temporary PROJECT_RUNS_ROOT; nothing lands in runs/.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from t2flair import prep
from tests.test_core import synthetic_volume

ROOT = Path(__file__).resolve().parents[1]
EXP = "smoke-e2e"


def run(module, *args, env):
    cmd = [sys.executable, "-m", "t2flair." + module, *args]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def test_end_to_end(tmp_path):
    pytest.importorskip("diffusers")
    shutil.rmtree(ROOT / "scratch" / "smoke" / EXP, ignore_errors=True)  # never resume a stale smoke run
    processed = tmp_path / "processed"
    processed.mkdir()
    ids = {"train": ["BraTS2021_9000%d" % i for i in range(4)], "val": ["BraTS2021_90010", "BraTS2021_90011"],
           "test": ["BraTS2021_90020"]}
    for n, (name, subjects) in enumerate(ids.items()):
        (tmp_path / ("smoke_%s.txt" % name)).write_text("\n".join(subjects) + "\n")
        for j, subject in enumerate(subjects):
            arrays, _ = prep.process_arrays(*synthetic_volume(size=80, depth=16, seed=10 * n + j), crop=64)
            np.savez_compressed(processed / (subject + ".npz"), **arrays)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PROJECT_RUNS_ROOT=str(tmp_path / "runs"))
    sets = ["exp=%s" % EXP, "data.processed=%s" % processed, "data.splits=%s" % (tmp_path / "smoke"),
            "data.crop=64", "model.base_channels=16", "model.channel_mult=[1,2,2,4]", "model.layers_per_block=1", "model.attn_head_dim=8",
            "train.steps=6", "train.batch_size=4", "train.num_workers=0", "train.log_every=2", "train.val_every=3",
            "train.val_subjects=2", "train.val_slices_per_subject=3", "train.val_steps=2",
            "seg.runs_exp=%s" % EXP, "seg.steps=4", "seg.batch_size=4", "seg.val_every=2",
            "seg.s_eval.arch=tiny", "seg.s_train.arch=tiny", "eval.lpips=false"]
    common = ["--set", *sets, "--smoke"]
    run("train", *common, env=env)
    run("train", *common, env=env)  # resumes from last.pt and stops immediately
    run("train_seg", "--role", "s_eval", *common, env=env)
    run("infer", "--split", "val", "--steps", "3", *common, env=env)
    out = tmp_path / "result.json"
    run("evaluate", "--split", "val", "--steps", "3", "--out", str(out), *common, env=env)
    result = json.loads(out.read_text())
    assert result["eval_set"] == "main_val"
    assert result["metrics"]["model"]["psnr"]["n"] == 2
    assert "dice" in result["baseline"]["upper_bound_real_flair"]
    assert result["checkpoints"]["model"]["ckpt"] == "last" and "fp_lesions" in result["metrics"]["model"]
    run("compare", "pair", "--a", str(out), "--b", str(out), "--b-key", "identity", "--name", "model_vs_identity",
        "--primary", "psnr:higher", "--noninf", "fp_volume:lower:10%", "--metrics", "empty_slice_false_alarm",
        "--into", str(out), env=env)
    assert "model_vs_identity" in json.loads(out.read_text())["comparisons"]
    run("figures", "--steps", "3", "--out", str(tmp_path / "figs"), *common, env=env)
    assert (tmp_path / "figs" / "train_curve.png").exists() and (tmp_path / "figs" / "val_samples.png").exists()
    log = (ROOT / "scratch" / "smoke" / EXP / "model" / "log.jsonl").read_text().splitlines()
    assert any(json.loads(line).get("event") == "val" for line in log)


@pytest.mark.parametrize("method", ["regression", "pix2pix", "noise_fm"])
def test_baseline_end_to_end(tmp_path, method):
    """Baselines and F0 share the trunk pipeline: train (and resume) -> infer -> evaluate."""
    pytest.importorskip("diffusers")
    exp = "smoke-baseline-" + method
    shutil.rmtree(ROOT / "scratch" / "smoke" / exp, ignore_errors=True)
    processed = tmp_path / "processed"
    processed.mkdir()
    ids = {"train": ["BraTS2021_9100%d" % i for i in range(3)], "val": ["BraTS2021_91010"], "test": ["BraTS2021_91020"]}
    for n, (name, subjects) in enumerate(ids.items()):
        (tmp_path / ("smoke_%s.txt" % name)).write_text("\n".join(subjects) + "\n")
        for j, subject in enumerate(subjects):
            arrays, _ = prep.process_arrays(*synthetic_volume(size=80, depth=12, seed=50 + 10 * n + j), crop=64)
            np.savez_compressed(processed / (subject + ".npz"), **arrays)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PROJECT_RUNS_ROOT=str(tmp_path / "runs"))
    steps = "3" if method == "noise_fm" else "1"  # F0 integrates several Euler steps
    sets = ["exp=%s" % exp, "method.name=%s" % method, "infer.steps=%s" % steps, "data.processed=%s" % processed,
            "data.splits=%s" % (tmp_path / "smoke"), "data.crop=64", "model.base_channels=16",
            "model.channel_mult=[1,2,2,4]", "model.layers_per_block=1", "model.attn_head_dim=8", "train.steps=4",
            "train.batch_size=4", "train.num_workers=0", "train.log_every=2", "train.val_every=2",
            "train.val_subjects=1", "train.val_slices_per_subject=3", "seg.runs_exp=%s" % exp, "seg.steps=2",
            "seg.batch_size=4", "seg.val_every=2", "seg.s_eval.arch=tiny", "eval.lpips=false"]
    if method == "pix2pix":
        sets += ["train.lr=2e-4", "train.betas=[0.5,0.999]"]
    common = ["--set", *sets, "--smoke"]
    run("train", *common, env=env)
    common = ["--set", *sets, "train.steps=6", "--smoke"]
    run("train", *common, env=env)  # resumes (incl. discriminator) and continues to step 6
    run("train_seg", "--role", "s_eval", *common, env=env)
    run("infer", "--split", "val", "--steps", steps, *common, env=env)
    out = tmp_path / "result.json"
    run("evaluate", "--split", "val", "--steps", steps, "--out", str(out), *common, env=env)
    result = json.loads(out.read_text())
    assert result["checkpoints"]["model"] == {"ckpt": "last", "ckpt_step": 6}
    log = [json.loads(line) for line in (ROOT / "scratch" / "smoke" / exp / "model" / "log.jsonl").read_text().splitlines()]
    if method == "pix2pix":
        assert any("g_adv" in e for e in log if e.get("event") == "train")
