"""Train a whole-tumor segmenter on real FLAIR slices of main_train.

    python -m t2flair.train_seg --config configs/trunk.yaml --role s_eval [--smoke]

s_train (BasicUNet, seed 1) is for the segmentation-consistency ablation; s_eval (SegResNet,
seed 2) is the independent evaluator and is never used to train a generator. Writes
runs/<seg.runs_exp>/<role>/{log.jsonl, checkpoints/{last,best}.pt}; best by mean 3D Dice
over main_val subjects.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import artifacts, config
from .common import JsonlLog, device, infinite, seed_all, seg_run, to_device
from .data import SliceDataset, load_volume, volume_inputs
from .metrics import dice
from .models import build_segmenter, count_params
from .splits import load_split


def seg_loss(logits, target):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    soft_dice = 1 - (2 * inter + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return soft_dice.mean() + F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def predict_volume(net, images, dev, threshold, batch=64):
    """images: (N,1,H,W) float in [-1,1] -> bool mask (N,H,W)."""
    net.eval()
    out = []
    for i in range(0, images.shape[0], batch):
        x = images[i:i + batch].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append((torch.sigmoid(net(x).float()) > threshold)[:, 0].cpu().numpy())
    return np.concatenate(out)


def load_segmenter(cfg, role, dev, smoke=False):
    run = seg_run(cfg, role, smoke=smoke)
    state = artifacts.load_checkpoint(run, "best")
    net = build_segmenter(state["arch"]).to(dev)
    net.load_state_dict(state["model"])
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--role", choices=["s_train", "s_eval"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config.from_args(args)
    s = cfg["seg"]
    arch, seed = s[args.role]["arch"], s[args.role]["seed"]
    seed_all(seed)
    dev = device()
    run = seg_run(cfg, args.role, smoke=args.smoke)
    run.mkdir(parents=True, exist_ok=True)
    log = JsonlLog(run / "log.jsonl")

    data = SliceDataset(cfg["data"]["processed"], load_split(cfg["data"]["splits"], "train"), flip=True)
    loader = DataLoader(data, batch_size=s["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["train"]["num_workers"], pin_memory=dev.type == "cuda",
                        persistent_workers=cfg["train"]["num_workers"] > 0)
    val = [volume_inputs(load_volume(cfg["data"]["processed"], v)) for v in load_split(cfg["data"]["splits"], "val")]

    net = build_segmenter(arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"])
    step = 0
    if (run / "checkpoints" / "last.pt").exists():
        state = artifacts.load_checkpoint(run, "last")
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
    log.write(event="start", role=args.role, arch=arch, seed=seed, step=step, params=count_params(net))

    batches = infinite(loader)
    running = []
    net.train()
    while step < s["steps"]:
        b = to_device(next(batches), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = net(b["target"])
        loss = seg_loss(logits.float(), b["tumor"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        running.append(loss.item())
        if step % cfg["train"]["log_every"] == 0:
            log.write(event="train", step=step, loss=round(float(np.mean(running)), 5))
            running = []
        if step % s["val_every"] == 0 or step == s["steps"]:
            scores = [dice(predict_volume(net, v["target"], dev, s["threshold"]), v["tumor"][:, 0].numpy() > 0)
                      for v in val]
            score = float(np.mean(scores))
            state = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "arch": arch, "seed": seed}
            improved = artifacts.save_checkpoint(run, state, metric=score, lower_is_better=False)
            log.write(event="val", step=step, val_dice=round(score, 4), best=improved)
            net.train()
    log.write(event="done", step=step)


if __name__ == "__main__":
    main()
