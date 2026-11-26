"""The single evaluation entry point (docs/PIPELINE.md §4).

    python -m t2flair.evaluate --config configs/trunk.yaml --split val --steps 50 --seed 0 \
        --out experiments/<id>/result.json [--run-id <lab run id>]

Reads the cached samples of t2flair.infer and reports, per subject and as mean ± std over
subjects: image metrics in the brain mask (PSNR, SSIM, MAE, LPIPS) and lesion metrics from
the independent segmenter S_eval (Dice, HD95 with a fixed penalty for empty predictions,
false-positive volume, miss rate, invented and missed lesion counts, lesion detection rate;
pooled small-lesion recall and empty-slice false alarms as exploratory). The same numbers are
computed for the identity baseline (output = T2) and S_eval on the real FLAIR (reference).
Paired comparisons between two result.json files: t2flair.compare.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import artifacts, config
from .common import device, git_commit, machine, run_path, seg_run
from .data import load_volume, volume_inputs
from .infer import checkpoint_key, ckpt_name, samples_dir
from .metrics import lesion_metrics, mae, psnr, ssim, summarize
from .splits import load_split
from .train_seg import load_segmenter, predict_volume

IMAGE = ("psnr", "ssim", "mae", "lpips")
LESION = ("dice", "hd95", "fp_volume", "miss_rate", "fp_lesions", "missed_lesions", "lesion_detection")


@torch.no_grad()
def lpips_score(net, pred, gt, dev, batch=32):
    vals = []
    for i in range(0, pred.shape[0], batch):
        a = torch.from_numpy(pred[i:i + batch]).float()[:, None].repeat(1, 3, 1, 1).to(dev)
        b = torch.from_numpy(gt[i:i + batch]).float()[:, None].repeat(1, 3, 1, 1).to(dev)
        vals.append(net(a, b).flatten().cpu().numpy())
    return float(np.concatenate(vals).mean())


def image_metrics(pred, gt, brain, lp_net, dev):
    out = {"psnr": psnr(pred, gt, brain), "ssim": ssim(pred, gt, brain), "mae": mae(pred, gt, brain)}
    out["lpips"] = lpips_score(lp_net, pred, gt, dev) if lp_net is not None else None
    return out


def aggregate(rows):
    summary = {k: summarize([r.get(k) for r in rows]) for k in IMAGE + LESION if any(k in r for r in rows)}
    small = sum(r.get("small_lesions", 0) for r in rows)
    empty = sum(r.get("empty_slices", 0) for r in rows)
    if any("empty_prediction" in r for r in rows):
        summary["empty_predictions"] = int(sum(bool(r.get("empty_prediction")) for r in rows))
    if any("small_lesions" in r for r in rows):
        summary["small_lesion_recall"] = {
            "pooled": sum(r["small_detected"] for r in rows) / small if small else None, "n_lesions": small}
        summary["empty_slice_false_alarm"] = {
            "pooled": sum(r["empty_slices_flagged"] for r in rows) / empty if empty else None, "n_slices": empty}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.add_args(parser)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config.from_args(args)
    e = cfg["eval"]
    split = args.split or e["split"]
    steps = args.steps or cfg["infer"]["steps"]
    seed = cfg["infer"]["seed"] if args.seed is None else args.seed
    start = time.time()
    dev = device()
    samples = samples_dir(cfg, split, steps, seed, smoke=args.smoke)
    if not artifacts.is_complete(samples):
        raise SystemExit("samples not complete: run t2flair.infer with the same split/steps/seed first (%s)" % samples)
    s_eval = load_segmenter(cfg, "s_eval", dev, smoke=args.smoke)
    lp_net = None
    if e["lpips"]:
        import lpips
        lp_net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    rows = {"model": [], "identity": [], "real_flair": []}
    lesion_kw = dict(fp_dilate=e["fp_dilate"], small_voxels=e["small_lesion_voxels"], empty_tau=e["empty_slice_tau"],
                     fp_lesion_voxels=e.get("fp_lesion_voxels", 10), gt_lesion_voxels=e.get("gt_lesion_voxels", 50),
                     hd95_empty=e.get("hd95_empty", 373.13))
    thr = cfg["seg"]["threshold"]
    for subject in load_split(cfg["data"]["splits"], split):
        inp = volume_inputs(load_volume(cfg["data"]["processed"], subject))
        with np.load(samples / (subject + ".npz")) as f:
            pred, z = f["pred"].astype(np.float32), f["z"]
        if not np.array_equal(z, inp["z"]):
            raise ValueError("slice index mismatch for %s" % subject)
        gt = inp["target"][:, 0].numpy()
        t2 = inp["source"][:, 0].numpy()
        brain = inp["brain"][:, 0].numpy() > 0
        g = inp["tumor"][:, 0].numpy() > 0
        for name, image in (("model", pred), ("identity", t2)):
            row = {"subject": subject, **image_metrics(image, gt, brain, lp_net, dev)}
            mask = predict_volume(s_eval, torch.from_numpy(image)[:, None], dev, thr)
            row.update(lesion_metrics(mask, g, **lesion_kw))
            rows[name].append(row)
        real = predict_volume(s_eval, inp["target"], dev, thr)
        rows["real_flair"].append({"subject": subject, **lesion_metrics(real, g, **lesion_kw)})
        print("eval", subject, "psnr %.2f dice %.3f" % (rows["model"][-1]["psnr"], rows["model"][-1]["dice"]), flush=True)

    metrics = {name: aggregate(r) for name, r in rows.items()}
    result = {
        "exp": cfg["exp"],
        "commit": git_commit(),
        "run_id": args.run_id,
        "config": args.exp or args.config,
        "overrides": args.set,
        "eval_set": "main_%s" % split,
        "steps": steps,
        "seed": seed,
        "checkpoints": {"model": checkpoint_key(run_path(cfg, "model", smoke=args.smoke), ckpt_name(cfg)),
                        "s_eval": cfg["seg"].get("ckpt", "best")},
        "metrics": {"model": metrics["model"]},
        "baseline": {"identity_t2": metrics["identity"], "upper_bound_real_flair": metrics["real_flair"]},
        "per_subject": rows,
        "runtime_sec": round(time.time() - start, 1),
        "outputs": "%s:%s ; samples %s:%s ; s_eval %s:%s" % (
            machine(), run_path(cfg, "model", smoke=args.smoke), machine(), samples,
            machine(), seg_run(cfg, "s_eval", smoke=args.smoke)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    m, b = metrics["model"], metrics["identity"]
    print(json.dumps({"psnr": m["psnr"], "dice": m["dice"], "identity_psnr": b["psnr"],
                      "upper_dice": metrics["real_flair"]["dice"]}, indent=1))


if __name__ == "__main__":
    main()
