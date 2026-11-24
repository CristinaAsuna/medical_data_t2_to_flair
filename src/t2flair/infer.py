"""Generate FLAIR for every kept slice of a split with the EMA weights of best.pt.

    python -m t2flair.infer --config configs/trunk.yaml --split val --steps 50 --seed 0

Output: artifacts.cache_dir("samples", ...) keyed by exp, checkpoint, split, steps, seed and
sigma; one <subject>.npz (pred float16 (N,H,W), z) per subject. The same key is generated
once; t2flair.evaluate finds it with samples_dir().
"""
import argparse
import json
import os

import numpy as np
import torch

from . import artifacts, bridge, config
from .common import device, run_path, to_device
from .data import load_volume, volume_inputs
from .models import Generator
from .splits import load_split


def samples_dir(cfg, split, steps, seed, smoke=False):
    run = run_path(cfg, "model", smoke=smoke)
    best = json.loads((run / "checkpoints" / "best.json").read_text())
    if smoke:  # smoke outputs stay in scratch/, never in runs/_cache (AGENTS.md §5)
        path = run / "samples" / "{}-steps{}-seed{}".format(split, steps, seed)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return artifacts.cache_dir("samples", exp=cfg["exp"], smoke=smoke, ckpt="best", ckpt_metric=best["metric"],
                               split=split, steps=steps, seed=seed, sigma=cfg["method"]["sigma"])


def subject_generator(dev, seed, subject):
    return torch.Generator(device=dev).manual_seed(seed * 1_000_003 + int(subject[-5:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config.from_args(args)
    steps = args.steps or cfg["infer"]["steps"]
    seed = cfg["infer"]["seed"] if args.seed is None else args.seed
    dev = device()
    out = samples_dir(cfg, args.split, steps, seed, smoke=args.smoke)
    if artifacts.is_complete(out):
        print("already complete:", out)
        return
    state = artifacts.load_checkpoint(run_path(cfg, "model", smoke=args.smoke), "best")
    model = Generator(cfg["model"], sample_size=cfg["data"]["crop"]).to(dev)
    model.load_state_dict(state["ema"])
    model.eval()
    subjects = load_split(cfg["data"]["splits"], args.split)
    for i, subject in enumerate(subjects, 1):
        target = out / (subject + ".npz")
        if target.exists():
            continue
        inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
        gen = subject_generator(dev, seed, subject)
        preds = []
        for j in range(0, inp["cond"].shape[0], args.batch):
            b = to_device({k: inp[k][j:j + args.batch] for k in ("cond", "source")}, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = bridge.sample(model, b["source"], b["cond"], steps, cfg["method"]["sigma"], gen)
            preds.append(pred[:, 0].float().cpu().numpy())
        tmp = out / (subject + ".tmp.npz")
        np.savez_compressed(tmp, pred=np.concatenate(preds).astype(np.float16), z=inp["z"])
        os.replace(tmp, target)
        print("infer %d/%d %s" % (i, len(subjects), subject), flush=True)
    (out / "meta.json").write_text(json.dumps({"ckpt_step": state["step"], "subjects": len(subjects)}))
    artifacts.mark_complete(out)
    print("samples:", out)


if __name__ == "__main__":
    main()
