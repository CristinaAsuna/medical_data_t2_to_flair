"""Generate main_val FLAIR predictions with SelfRDB into t2flair's sample cache contract.

    python -u experiments/2026-11-27-baseline-selfrdb/infer.py \
        --config configs/trunk.yaml --exp configs/exp/2026-11-27-baseline-selfrdb.yaml \
        --split val --steps 10 --seed 0 --batch 32

Writes <subject>.npz (pred float16 (N, 192, 192), z), meta.json (including sampling steps
and actual generator forward calls), and .complete into the exact cache path expected by
t2flair.evaluate and t2flair.figures.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import lightning as L
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapter import project_prediction, upstream_input
from t2flair import artifacts, config
from t2flair.common import device, run_path
from t2flair.data import load_volume, volume_inputs
from t2flair.splits import load_split

UPSTREAM = config.ROOT / "third_party" / "SelfRDB"
UPSTREAM_COMMIT = "6262edeff9ca7afe14c55e226db5cb23411f55d9"
sys.path.insert(0, str(UPSTREAM))
from main import BridgeRunner


def ckpt_name(cfg):
    return cfg["infer"].get("ckpt", "best")


def checkpoint_key(run, which):
    """Identical to t2flair.infer.checkpoint_key without importing diffusers."""
    if which == "best":
        return {"ckpt": "best", "ckpt_metric": json.loads((run / "checkpoints" / "best.json").read_text())["metric"]}
    if which == "last":
        steps = [json.loads(line)["step"] for line in (run / "log.jsonl").read_text().splitlines()
                 if json.loads(line).get("event") == "val"]
        return {"ckpt": "last", "ckpt_step": steps[-1]}
    return {"ckpt": which}


def samples_dir(cfg, split, steps, seed, smoke=False):
    """Identical to t2flair.infer.samples_dir without importing diffusers."""
    run = run_path(cfg, "model", smoke=smoke)
    which = ckpt_name(cfg)
    key = checkpoint_key(run, which)
    if smoke:
        path = run / "samples" / "{}-{}-steps{}-seed{}".format(split, which, steps, seed)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return artifacts.cache_dir("samples", exp=cfg["exp"], smoke=smoke, split=split, steps=steps, seed=seed,
                               sigma=cfg["method"]["sigma"], **key)


def load_official_config():
    with (UPSTREAM / "config.yaml").open() as handle:
        official = yaml.safe_load(handle)
    params = dict(official["model"])
    params["generator_params"] = dict(params["generator_params"])
    params["generator_params"]["image_size"] = official["data"]["image_size"]
    return official, params


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config.from_args(args)
    official, params = load_official_config()

    steps = args.steps or int(cfg["infer"]["steps"])
    if steps != int(params["diffusion_params"]["n_steps"]):
        raise ValueError(f"Expected steps={params['diffusion_params']['n_steps']} for official SelfRDB, got {steps}")
    seed = int(cfg["infer"]["seed"]) if args.seed is None else int(args.seed)

    dev = device()
    run = run_path(cfg, "model", smoke=args.smoke)
    out = samples_dir(cfg, args.split, steps, seed, smoke=args.smoke)
    if artifacts.is_complete(out):
        print("already complete:", out, flush=True)
        return

    which = ckpt_name(cfg)
    state = artifacts.load_checkpoint(run, which)
    model = BridgeRunner(**params)
    model.generator.load_state_dict(state["generator"])
    model.eval().to(dev)

    # Count actual generator forward passes without modifying upstream code.
    nfe = {"batch_calls": 0, "slice_calls": 0}

    def _count_hook(_mod, inputs, _output):
        nfe["batch_calls"] += 1
        nfe["slice_calls"] += int(inputs[0].shape[0])

    hook = model.generator.register_forward_hook(_count_hook)

    subjects = load_split(cfg["data"]["splits"], args.split)
    if args.smoke:
        subjects = subjects[:2]

    t0 = perf_counter()
    total_slices = 0
    pred_min, pred_max = float("inf"), float("-inf")
    try:
        for i, subject in enumerate(subjects, 1):
            target = out / (subject + ".npz")
            inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
            n_slices = int(inp["source"].shape[0])
            total_slices += n_slices
            if target.exists():
                with np.load(target) as existing:
                    if not np.array_equal(existing["z"], inp["z"]):
                        raise RuntimeError(f"Cached z mismatch for {subject}")
                    arr = existing["pred"].astype(np.float32)
                    if arr.shape != (n_slices, 192, 192) or not np.isfinite(arr).all():
                        raise RuntimeError(f"Invalid cached prediction for {subject}")
                    pred_min = min(pred_min, float(arr.min()))
                    pred_max = max(pred_max, float(arr.max()))
                continue

            subj_seed = seed * 1_000_003 + int(subject[-5:])
            L.seed_everything(subj_seed, workers=False)
            preds = []
            for j in range(0, n_slices, args.batch):
                src_192 = inp["source"][j:j + args.batch]
                src_256 = upstream_input(src_192)
                if not torch.equal(project_prediction(src_256), src_192):
                    raise RuntimeError(f"Pad/crop roundtrip mismatch for {subject} at slice offset {j}")
                pred_256 = model.diffusion.sample_x0(src_256.to(dev), model.generator)
                pred_192 = project_prediction(pred_256)
                if pred_192.shape != (src_192.shape[0], 1, 192, 192) or not torch.isfinite(pred_192).all():
                    raise RuntimeError(f"Invalid prediction shape or non-finite values for {subject}")
                arr = pred_192[:, 0].float().cpu().numpy()
                pred_min = min(pred_min, float(arr.min()))
                pred_max = max(pred_max, float(arr.max()))
                preds.append(arr)

            pred_vol = np.concatenate(preds, axis=0)
            tmp = out / (subject + ".tmp.npz")
            np.savez_compressed(tmp, pred=pred_vol.astype(np.float16), z=inp["z"])
            os.replace(tmp, target)
            print("infer %d/%d %s (slices=%d)" % (i, len(subjects), subject, n_slices), flush=True)
    finally:
        hook.remove()

    infer_sec = perf_counter() - t0

    # Final verification pass before marking complete (CARD.md §评测).
    for subject in subjects:
        inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
        with np.load(out / (subject + ".npz")) as f:
            if not np.array_equal(f["z"], inp["z"]):
                raise RuntimeError(f"Final verification z mismatch for {subject}")
            if f["pred"].shape != (inp["source"].shape[0], 192, 192) or not np.isfinite(f["pred"]).all():
                raise RuntimeError(f"Final verification invalid pred for {subject}")

    meta = {
        "ckpt": which,
        "ckpt_step": int(state["step"]),
        "subjects": len(subjects),
        "total_slices": total_slices,
        "n_steps": int(params["diffusion_params"]["n_steps"]),
        "n_recursions": int(params["diffusion_params"]["n_recursions"]),
        "consistency_threshold": float(params["diffusion_params"]["consistency_threshold"]),
        "test_batch_size": int(args.batch),
        "nfe_batch_calls": nfe["batch_calls"],
        "nfe_slice_calls": nfe["slice_calls"],
        "avg_nfe_per_slice": round(nfe["slice_calls"] / max(1, total_slices), 4),
        "infer_sec": round(infer_sec, 2),
        "train_sec": round(float(state.get("elapsed_sec", 0.0)), 1),
        "train_steps": int(state["step"]),
        "train_batch_size": int(official["data"]["train_batch_size"]),
        "pred_range": [round(pred_min, 5), round(pred_max, 5)],
        "upstream_commit": UPSTREAM_COMMIT,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    artifacts.mark_complete(out)
    print("meta:", json.dumps(meta), flush=True)
    print("samples:", out, flush=True)


if __name__ == "__main__":
    main()
