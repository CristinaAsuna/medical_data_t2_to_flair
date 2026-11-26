"""Generate FLAIR for every kept slice of a split with the EMA weights of one checkpoint.

    python -m t2flair.infer --config configs/trunk.yaml --split val --steps 50 --seed 0

Uses the checkpoint named by infer.ckpt (trunk: last = final step, no selection on main_val;
D-006). Output: artifacts.cache_dir("samples", ...) keyed by exp, checkpoint (name plus its
step or best metric), split, steps, seed and sigma; one <subject>.npz (pred float16 (N,H,W), z)
per subject. The same key is generated once; t2flair.evaluate finds it with samples_dir().
"""
import argparse
import json
import os

import numpy as np
import torch

from . import artifacts, baselines, config
from .common import device, run_path, to_device
from .data import load_volume, volume_inputs
from .models import Generator
from .splits import load_split


def ckpt_name(cfg):
    return cfg["infer"].get("ckpt", "best")


def checkpoint_key(run, which):
    """Identify a checkpoint for cache keys without loading it.

    best: its validation metric (best.json); last: the step of the last "val" event in log.jsonl
    (train.py writes last.pt right before logging that event); any other name is a frozen snapshot.
    """
    if which == "best":
        return {"ckpt": "best", "ckpt_metric": json.loads((run / "checkpoints" / "best.json").read_text())["metric"]}
    if which == "last":
        steps = [json.loads(line)["step"] for line in (run / "log.jsonl").read_text().splitlines()
                 if json.loads(line).get("event") == "val"]
        return {"ckpt": "last", "ckpt_step": steps[-1]}
    return {"ckpt": which}


def samples_dir(cfg, split, steps, seed, smoke=False):
    run = run_path(cfg, "model", smoke=smoke)
    which = ckpt_name(cfg)
    key = checkpoint_key(run, which)
    if smoke:  # smoke outputs stay in scratch/, never in runs/_cache (AGENTS.md §5)
        path = run / "samples" / "{}-{}-steps{}-seed{}".format(split, which, steps, seed)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return artifacts.cache_dir("samples", exp=cfg["exp"], smoke=smoke, split=split, steps=steps, seed=seed,
                               sigma=cfg["method"]["sigma"], **key)


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
    state = artifacts.load_checkpoint(run_path(cfg, "model", smoke=args.smoke), ckpt_name(cfg))
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
                pred = baselines.generate(model, b["source"], b["cond"], cfg, steps, gen)
            preds.append(pred[:, 0].float().cpu().numpy())
        tmp = out / (subject + ".tmp.npz")
        np.savez_compressed(tmp, pred=np.concatenate(preds).astype(np.float16), z=inp["z"])
        os.replace(tmp, target)
        print("infer %d/%d %s" % (i, len(subjects), subject), flush=True)
    (out / "meta.json").write_text(json.dumps({"ckpt": ckpt_name(cfg), "ckpt_step": state["step"], "subjects": len(subjects)}))
    artifacts.mark_complete(out)
    print("samples:", out)


if __name__ == "__main__":
    main()
