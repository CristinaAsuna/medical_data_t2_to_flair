"""Train official SelfRDB from scratch on project main_train slices (50k G/D updates).

    python -u experiments/2026-11-27-baseline-selfrdb/train.py \
        --config configs/trunk.yaml --exp configs/exp/2026-11-27-baseline-selfrdb.yaml

Preserves third_party/SelfRDB untouched (D-004). Writes only through t2flair.artifacts
(runs/<exp>/model/{log.jsonl, checkpoints/{last,best}.pt}), resumes from last.pt when
present, and logs val-subset brain-mask PSNR in the same schema as t2flair.train.
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

import lightning as L
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapter import BridgeSlices, project_prediction, upstream_input
from t2flair import artifacts, config
from t2flair.common import JsonlLog, run_path
from t2flair.data import load_volume, volume_inputs
from t2flair.metrics import psnr
from t2flair.splits import load_split

UPSTREAM = config.ROOT / "third_party" / "SelfRDB"
UPSTREAM_COMMIT = "6262edeff9ca7afe14c55e226db5cb23411f55d9"
sys.path.insert(0, str(UPSTREAM))
from main import BridgeRunner


def load_official_config():
    with (UPSTREAM / "config.yaml").open() as handle:
        official = yaml.safe_load(handle)
    params = dict(official["model"])
    params["generator_params"] = dict(params["generator_params"])
    params["generator_params"]["image_size"] = official["data"]["image_size"]
    return official, params


def val_subset(cfg, max_subjects=None):
    """Fixed subset: first N val subjects, evenly spaced kept slices (same as t2flair.train)."""
    t = cfg["train"]
    n_subj = max_subjects if max_subjects is not None else t["val_subjects"]
    out = []
    for subject in load_split(cfg["data"]["splits"], "val")[:n_subj]:
        inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
        n = inp["cond"].shape[0]
        pick = np.unique(np.linspace(0, n - 1, t["val_slices_per_subject"]).round().astype(int))
        out.append({k: v[pick] for k, v in inp.items() if k != "z"})
    return out


@torch.no_grad()
def validate(module, subset, batch_size, dev):
    was_training = module.training
    module.eval()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
    try:
        torch.manual_seed(1234)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(1234)
        scores = []
        for item in subset:
            src = upstream_input(item["source"])
            preds = []
            for j in range(0, src.shape[0], batch_size):
                x = src[j:j + batch_size].to(dev)
                pred_256 = module.diffusion.sample_x0(x, module.generator)
                pred_192 = project_prediction(pred_256)
                preds.append(pred_192[:, 0].float().cpu().numpy())
            pred_vol = np.concatenate(preds, axis=0)
            scores.append(psnr(pred_vol, item["target"][:, 0].numpy(), item["brain"][:, 0].numpy() > 0))
    finally:
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        if was_training:
            module.train()
    return float(np.mean(scores))


class SelfRDBCallback(L.Callback):
    def __init__(self, run, cfg, subset, target_steps, val_batch, resume_state=None):
        super().__init__()
        self.run = run
        self.cfg = cfg
        self.subset = subset
        self.target_steps = int(target_steps)
        self.val_batch = int(val_batch)
        self.resume_state = resume_state
        self.log = JsonlLog(run / "log.jsonl")
        self.elapsed_offset = float(resume_state.get("elapsed_sec", 0.0)) if resume_state else 0.0
        self.t0 = None
        self.last_saved_step = int(resume_state["step"]) if resume_state else 0

    def _elapsed(self):
        return self.elapsed_offset + (perf_counter() - self.t0)

    def on_train_start(self, trainer, pl_module):
        self.t0 = perf_counter()
        if self.resume_state is not None:
            trainer.optimizers[0].load_state_dict(self.resume_state["opt_g"])
            trainer.optimizers[1].load_state_dict(self.resume_state["opt_d"])
            trainer.lr_scheduler_configs[0].scheduler.load_state_dict(self.resume_state["sched_g"])
            trainer.lr_scheduler_configs[1].scheduler.load_state_dict(self.resume_state["sched_d"])
            # Move optimizer states to active device.
            dev = pl_module.device
            for opt in trainer.optimizers:
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(dev)
            g_global = int(self.resume_state["global_step"])
            progress = trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress
            progress.total.ready = g_global
            progress.total.completed = g_global
            print(f"resumed SelfRDB from step {self.resume_state['step']} (global_step={g_global})", flush=True)
            self.resume_state = None

    def _save_and_eval(self, trainer, pl_module, g_step):
        val_psnr = validate(pl_module, self.subset, self.val_batch, pl_module.device)
        elapsed = self._elapsed()
        state = {
            "step": g_step,
            "global_step": int(trainer.global_step),
            "generator": pl_module.generator.state_dict(),
            "discriminator": pl_module.discriminator.state_dict(),
            "opt_g": trainer.optimizers[0].state_dict(),
            "opt_d": trainer.optimizers[1].state_dict(),
            "sched_g": trainer.lr_scheduler_configs[0].scheduler.state_dict(),
            "sched_d": trainer.lr_scheduler_configs[1].scheduler.state_dict(),
            "val_psnr": val_psnr,
            "elapsed_sec": elapsed,
            "config": self.cfg,
            "upstream_commit": UPSTREAM_COMMIT,
        }
        improved = artifacts.save_checkpoint(self.run, state, metric=val_psnr, lower_is_better=False)
        self.last_saved_step = g_step
        rec = {"event": "val", "step": g_step, "val_psnr": round(val_psnr, 4),
               "best": bool(improved), "time": round(elapsed, 1)}
        self.log.write(rec)
        print(rec, flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        for key in ("d_loss", "g_loss/rec", "g_loss/adv", "g_loss/total"):
            val = metrics[key]
            if not torch.isfinite(val):
                raise RuntimeError(f"Non-finite {key}={val} at global_step={trainer.global_step}")

        g_step = int(trainer.global_step) // 2
        t = self.cfg["train"]
        if g_step == 1 or g_step % t["log_every"] == 0 or g_step == self.target_steps:
            rec = {
                "event": "train",
                "step": g_step,
                "loss": round(float(metrics["g_loss/total"].item()), 6),
                "g_rec": round(float(metrics["g_loss/rec"].item()), 6),
                "g_adv": round(float(metrics["g_loss/adv"].item()), 6),
                "d_loss": round(float(metrics["d_loss"].item()), 6),
                "lr_g": float(trainer.optimizers[0].param_groups[0]["lr"]),
                "lr_d": float(trainer.optimizers[1].param_groups[0]["lr"]),
                "time": round(self._elapsed(), 1),
            }
            self.log.write(rec)
            print(rec, flush=True)

        if g_step % t["val_every"] == 0 or g_step == self.target_steps:
            self._save_and_eval(trainer, pl_module, g_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--smoke", action="store_true", help="write to scratch/smoke instead of runs/")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    cfg = config.from_args(args)
    official, params = load_official_config()

    t = cfg["train"]
    target_steps = int(t["steps"])
    batch_size = int(official["data"]["train_batch_size"])
    if int(t["batch_size"]) != batch_size:
        raise ValueError(f"Expected train.batch_size={batch_size} to match official SelfRDB, got {t['batch_size']}")

    run = run_path(cfg, "model", smoke=args.smoke)
    resume_state = None
    if (run / "checkpoints" / "last.pt").exists():
        resume_state = artifacts.load_checkpoint(run, "last")
        if int(resume_state["step"]) >= target_steps:
            print(f"already trained to step {resume_state['step']} >= {target_steps}: {run}", flush=True)
            return

    L.seed_everything(official["seed_everything"], workers=True)
    model = BridgeRunner(**params)
    if resume_state is not None:
        model.generator.load_state_dict(resume_state["generator"])
        model.discriminator.load_state_dict(resume_state["discriminator"])

    train_subjects = load_split(cfg["data"]["splits"], "train")
    if args.smoke:
        train_subjects = train_subjects[:2]
    train_ds = BridgeSlices(cfg["data"]["processed"], train_subjects)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    subset = val_subset(cfg, max_subjects=1 if args.smoke else None)
    val_batch = int(official["data"]["test_batch_size"])
    cb = SelfRDBCallback(run, cfg, subset, target_steps, val_batch, resume_state=resume_state)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=32,
        max_epochs=int(official["trainer"]["max_epochs"]),
        max_steps=2 * target_steps,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        deterministic=True,
        default_root_dir=str(config.ROOT / "scratch" / "selfrdb-lightning"),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, train_dataloaders=loader)

    if trainer.global_step != 2 * target_steps:
        raise RuntimeError(f"Expected global_step={2 * target_steps}, got {trainer.global_step}")
    if cb.last_saved_step != target_steps:
        raise RuntimeError(f"Expected last.pt at step={target_steps}, got {cb.last_saved_step}")
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"done SelfRDB training: steps={target_steps}, peak_gib={peak_gib:.2f}, run={run}", flush=True)


if __name__ == "__main__":
    main()
