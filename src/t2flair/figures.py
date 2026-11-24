"""The two figures the trunk card lists: train_curve.png and val_samples.png.

    python -m t2flair.figures --config configs/trunk.yaml --out experiments/<id>/figures
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .common import run_path
from .data import load_volume, volume_inputs
from .infer import samples_dir
from .splits import load_split


def train_curve(log_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    tr = [(r["step"], r["loss"]) for r in rows if r.get("event") == "train"]
    va = [(r["step"], r["val_psnr"]) for r in rows if r.get("event") == "val"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    if tr:
        ax[0].plot(*zip(*tr), lw=1)
    ax[0].set(xlabel="step", ylabel="train MSE", yscale="log", title="training loss")
    if va:
        ax[1].plot(*zip(*va), marker="o")
    ax[1].set(xlabel="step", ylabel="PSNR (dB)", title="val subset PSNR (EMA)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def val_samples(cfg, samples, out, n_subjects=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subjects = load_split(cfg["data"]["splits"], "val")[:n_subjects]
    fig, axes = plt.subplots(n_subjects, 4, figsize=(10, 2.6 * n_subjects), squeeze=False)
    for row, subject in zip(axes, subjects):
        inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
        with np.load(Path(samples) / (subject + ".npz")) as f:
            pred = f["pred"].astype(np.float32)
        tumor = inp["tumor"][:, 0].numpy()
        k = int(tumor.reshape(tumor.shape[0], -1).sum(1).argmax())  # slice with the largest tumor area
        t2, gt, p = inp["source"][k, 0].numpy(), inp["target"][k, 0].numpy(), pred[k]
        panels = [(t2, "T2 (input)"), (gt, "FLAIR (real)"), (p, "FLAIR (F1)"), (np.abs(p - gt) / 2, "|error|")]
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img.T, cmap="magma" if title == "|error|" else "gray", origin="lower",
                      vmin=0 if title == "|error|" else -1, vmax=0.5 if title == "|error|" else 1)
            if title == "FLAIR (real)":
                ax.contour(tumor[k].T, levels=[0.5], colors="c", linewidths=0.6, origin="lower")
            ax.set_title("%s  %s z=%d" % (title, subject[-5:], inp["z"][k]) if title.startswith("T2") else title,
                         fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config.from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = run_path(cfg, "model", smoke=args.smoke)
    train_curve(run / "log.jsonl", out / "train_curve.png")
    steps = args.steps or cfg["infer"]["steps"]
    seed = cfg["infer"]["seed"] if args.seed is None else args.seed
    val_samples(cfg, samples_dir(cfg, "val", steps, seed, smoke=args.smoke), out / "val_samples.png")
    print("figures:", out)


if __name__ == "__main__":
    main()
