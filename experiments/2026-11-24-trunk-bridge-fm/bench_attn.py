"""Training throughput of the trunk generator under a few attention settings (card budget check).

    python experiments/2026-11-24-trunk-bridge-fm/bench_attn.py --config configs/trunk.yaml [--loader]

Synthetic tensors at the trunk batch size and crop, same step as t2flair.train (bf16 autocast,
AdamW, grad clip). Prints one JSON line per variant: it/s, ms/step, peak GPU memory, params,
and the hours 100k steps would take. --loader also times the real DataLoader alone.
Nothing is saved; read the numbers from the Slurm log.
"""
import argparse
import copy
import json
import time

import torch

from t2flair import bridge, config
from t2flair.models import Generator, count_params

VARIANTS = {  # name -> overrides of cfg["model"]
    "trunk_hd8_attn23": {},
    "hd64_attn23": {"attn_head_dim": 64},
    "hd32_attn23": {"attn_head_dim": 32},
    "hd64_attn3": {"attn_head_dim": 64, "attn_levels": [3]},
    "no_attn": {"attn_levels": []},
}


def bench(cfg, name, extra, steps, warmup, batch, size, dev):
    mcfg = dict(cfg["model"], **extra)
    model = Generator(mcfg, sample_size=size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    data = {"cond": torch.randn(batch, 3, size, size, device=dev),
            "source": torch.rand(batch, 1, size, size, device=dev) * 2 - 1,
            "target": torch.rand(batch, 1, size, size, device=dev) * 2 - 1}
    sigma = cfg["method"]["sigma"]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = bridge.training_loss(model, data, sigma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.time() - start) / steps
    out = {"variant": name, "tf32": torch.backends.cuda.matmul.allow_tf32, "model": mcfg, "batch": batch, "size": size, "it_s": round(1 / sec, 3),
           "ms_step": round(sec * 1000, 1), "hours_100k": round(1e5 * sec / 3600, 1),
           "params": count_params(model), "loss": round(float(loss), 4)}
    if dev.type == "cuda":
        out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
        out["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(out), flush=True)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def bench_loader(cfg, batches):
    from torch.utils.data import DataLoader
    from t2flair.data import SliceDataset
    from t2flair.splits import load_split
    t = cfg["train"]
    t0 = time.time()
    data = SliceDataset(cfg["data"]["processed"], load_split(cfg["data"]["splits"], "train"), flip=True)
    load_sec = time.time() - t0
    loader = DataLoader(data, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                        num_workers=t["num_workers"], pin_memory=True, persistent_workers=t["num_workers"] > 0)
    it = iter(loader)
    next(it)
    t0 = time.time()
    for _ in range(batches):
        next(it)
    sec = (time.time() - t0) / batches
    print(json.dumps({"variant": "loader_only", "load_volumes_sec": round(load_sec, 1),
                      "batches_s": round(1 / sec, 2), "workers": t["num_workers"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--loader", action="store_true")
    parser.add_argument("--tf32", action="store_true", help="also enable TF32 and cudnn.benchmark")
    args = parser.parse_args()
    cfg = config.from_args(args)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and args.tf32:  # t2flair.train does not set these; off by default to match it
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    batch = args.batch or cfg["train"]["batch_size"]
    size = args.size or cfg["data"]["crop"]
    for name in args.variants:
        bench(cfg, name, copy.deepcopy(VARIANTS[name]), args.steps, args.warmup, batch, size, dev)
    if args.loader:
        bench_loader(cfg, 50)


if __name__ == "__main__":
    main()
