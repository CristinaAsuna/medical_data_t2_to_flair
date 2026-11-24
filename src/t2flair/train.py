"""Train the generator (F1 bridge FM on the trunk config).

    python -m t2flair.train --config configs/trunk.yaml [--exp configs/exp/<id>.yaml] [--smoke]

Writes runs/<exp>/model/{log.jsonl, checkpoints/{last,best}.pt}. best.pt is chosen by
brain-mask PSNR of EMA samples on a fixed main_val subset (never main_test). Resumes from
last.pt when it exists.
"""
import argparse
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import artifacts, bridge, config
from .common import JsonlLog, device, infinite, run_path, seed_all, to_device
from .data import SliceDataset, load_volume, volume_inputs
from .metrics import psnr
from .models import Generator, count_params
from .splits import load_split


def val_subset(cfg):
    """Fixed subset: first N val subjects, evenly spaced kept slices."""
    t = cfg["train"]
    out = []
    for subject in load_split(cfg["data"]["splits"], "val")[: t["val_subjects"]]:
        inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
        n = inp["cond"].shape[0]
        pick = np.unique(np.linspace(0, n - 1, t["val_slices_per_subject"]).round().astype(int))
        out.append({k: v[pick] for k, v in inp.items() if k != "z"})
    return out


@torch.no_grad()
def validate(model, subset, cfg, dev):
    model.eval()
    gen = torch.Generator(device=dev).manual_seed(1234)
    scores = []
    for item in subset:
        b = to_device(item, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg["train"]["bf16"] and dev.type == "cuda"):
            pred = bridge.sample(model, b["source"], b["cond"], cfg["train"]["val_steps"], cfg["method"]["sigma"], gen)
        scores.append(psnr(pred[:, 0].cpu().numpy(), item["target"][:, 0].numpy(), item["brain"][:, 0].numpy() > 0))
    model.train()
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--smoke", action="store_true", help="write to scratch/smoke instead of runs/")
    args = parser.parse_args()
    cfg = config.from_args(args)
    if cfg["method"]["name"] != "bridge":
        raise NotImplementedError("only method.name=bridge (F1) is implemented on the trunk")
    t = cfg["train"]
    seed_all(t["seed"])
    dev = device()
    run = run_path(cfg, "model", smoke=args.smoke)
    log = JsonlLog(run / "log.jsonl")

    data = SliceDataset(cfg["data"]["processed"], load_split(cfg["data"]["splits"], "train"), flip=True)
    loader = DataLoader(data, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                        num_workers=t["num_workers"], pin_memory=dev.type == "cuda",
                        persistent_workers=t["num_workers"] > 0)
    subset = val_subset(cfg)

    model = Generator(cfg["model"], sample_size=cfg["data"]["crop"]).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, t["warmup"])))
    step = 0
    if (run / "checkpoints" / "last.pt").exists():
        state = artifacts.load_checkpoint(run, "last")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        print("resumed from step", step, flush=True)
    log.write(event="start", step=step, train_slices=len(data), params=count_params(model), device=str(dev))

    def checkpoint():
        score = validate(ema, subset, cfg, dev)
        state = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step, "config": cfg}
        improved = artifacts.save_checkpoint(run, state, metric=score, lower_is_better=False)
        log.write(event="val", step=step, val_psnr=round(score, 4), best=improved)

    batches = infinite(loader)
    model.train()
    running = []
    while step < t["steps"]:
        batch = to_device(next(batches), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=t["bf16"] and dev.type == "cuda"):
            loss = bridge.training_loss(model, batch, cfg["method"]["sigma"])
        if not torch.isfinite(loss):
            log.write(event="nan", step=step)
            raise FloatingPointError("non-finite loss at step %d" % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        sched.step()
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1.0 - t["ema"])
            for b_ema, b in zip(ema.buffers(), model.buffers()):
                b_ema.copy_(b)
        step += 1
        running.append(loss.item())
        if step % t["log_every"] == 0:
            log.write(event="train", step=step, loss=round(float(np.mean(running)), 6), lr=sched.get_last_lr()[0])
            running = []
        if step % t["val_every"] == 0 or step == t["steps"]:
            checkpoint()
    log.write(event="done", step=step)


if __name__ == "__main__":
    main()
