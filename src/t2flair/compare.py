"""Paired, per-subject comparison of two result.json files from t2flair.evaluate (D-006).

    python -m t2flair.compare pair --a experiments/<A>/result.json --b experiments/<B>/result.json \
        --name F1_vs_F0 --primary fp_volume:lower [--noninf psnr:higher:0.5] [--noninf fp_volume:lower:10%] \
        [--metrics dice miss_rate ...] [--into experiments/<A>/result.json]
    python -m t2flair.compare holm experiments/<id1>/result.json:<name> experiments/<id2>/result.json:<name> ...

This is a statistics tool over the per-subject rows of the single evaluation entry; it computes
no metric itself. Differences are A - B. For each metric: paired Wilcoxon signed-rank p (two
sided) and the 95% bootstrap CI of the mean difference (subjects resampled, fixed seed). A
non-inferiority margin m passes when the one-sided 95% bound of A - B stays on the good side of
-m (higher is better) or +m (lower is better); "m%" means m percent of B's mean. Pooled ratios
(empty_slice_false_alarm, small_lesion_recall) are compared with a subject-level bootstrap.
`--into` stores the comparison under result["comparisons"][name], so no extra file is committed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

POOLED = {"empty_slice_false_alarm": ("empty_slices_flagged", "empty_slices"),
          "small_lesion_recall": ("small_detected", "small_lesions")}
N_BOOT = 10000


def _rows(result, key):
    group = "model" if key == "model" else key
    return {r["subject"]: r for r in result["per_subject"][group]}


def paired(a_rows, b_rows, metric):
    subjects = sorted(set(a_rows) & set(b_rows))
    pairs = [(a_rows[s].get(metric), b_rows[s].get(metric)) for s in subjects]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    return np.asarray(pairs, dtype=np.float64).reshape(-1, 2)


def bootstrap_mean(diff, seed=0, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    return diff[idx].mean(axis=1)


def pooled_diff(a_rows, b_rows, num, den, seed=0, n_boot=N_BOOT):
    subjects = sorted(set(a_rows) & set(b_rows))
    arr = np.asarray([[a_rows[s][num], a_rows[s][den], b_rows[s][num], b_rows[s][den]] for s in subjects], float)
    ratio = lambda x: x[..., 0].sum(-1) / x[..., 1].sum(-1) - x[..., 2].sum(-1) / x[..., 3].sum(-1)
    rng = np.random.default_rng(seed)
    boots = ratio(arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))])
    return float(ratio(arr[None])[0]), boots


def wilcoxon_p(diff):
    from scipy.stats import wilcoxon
    if diff.size < 2 or np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)


def compare_metric(a_rows, b_rows, metric, seed=0):
    if metric in POOLED:
        est, boots = pooled_diff(a_rows, b_rows, *POOLED[metric], seed=seed)
        return {"n": len(set(a_rows) & set(b_rows)), "diff": est, "ci95": [float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5))], "p": None, "pooled": True, "_boots": boots}
    pairs = paired(a_rows, b_rows, metric)
    diff = pairs[:, 0] - pairs[:, 1]
    boots = bootstrap_mean(diff, seed=seed)
    return {"n": int(diff.size), "mean_a": float(pairs[:, 0].mean()), "mean_b": float(pairs[:, 1].mean()),
            "diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p": wilcoxon_p(diff), "_boots": boots}


def noninferior(stat, direction, margin, b_mean):
    m = float(margin[:-1]) / 100.0 * abs(b_mean) if str(margin).endswith("%") else float(margin)
    boots = stat["_boots"]
    if direction == "higher":
        bound = float(np.percentile(boots, 5))  # one-sided 95% lower bound of A - B
        return {"margin": m, "bound": bound, "passes": bound > -m}
    bound = float(np.percentile(boots, 95))  # one-sided 95% upper bound of A - B
    return {"margin": m, "bound": bound, "passes": bound < m}


def holm(pvalues):
    """Holm step-down adjusted p-values, same order as the input."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def strip(stat):
    return {k: v for k, v in stat.items() if not k.startswith("_")}


def cmd_pair(args):
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    if a["eval_set"] != b["eval_set"]:
        raise SystemExit("different eval sets: %s vs %s" % (a["eval_set"], b["eval_set"]))
    a_rows, b_rows = _rows(a, args.a_key), _rows(b, args.b_key)
    metric, direction = args.primary.split(":")
    out = {"a": args.a, "b": args.b, "a_key": args.a_key, "b_key": args.b_key, "eval_set": a["eval_set"],
           "primary": {"metric": metric, "direction": direction}, "metrics": {}, "noninferiority": {}}
    stats = {}
    for m in [metric] + [n.split(":")[0] for n in args.noninf] + list(args.metrics):
        if m not in stats:
            stats[m] = compare_metric(a_rows, b_rows, m, seed=args.seed)
            out["metrics"][m] = strip(stats[m])
    better = stats[metric]["diff"] > 0 if direction == "higher" else stats[metric]["diff"] < 0
    out["primary"].update(p=stats[metric]["p"], better=bool(better))
    for spec in args.noninf:
        m, d, margin = spec.split(":")
        b_mean = stats[m].get("mean_b", 0.0)
        out["noninferiority"][m] = {"direction": d, **noninferior(stats[m], d, margin, b_mean)}
    print(json.dumps(out, indent=1))
    if args.into:
        target = json.loads(Path(args.into).read_text())
        target.setdefault("comparisons", {})[args.name] = out
        Path(args.into).write_text(json.dumps(target, indent=1))


def cmd_holm(args):
    entries = []
    for spec in args.items:
        path, name = spec.rsplit(":", 1)
        comp = json.loads(Path(path).read_text())["comparisons"][name]
        entries.append((spec, comp["primary"]["p"]))
    adj = holm([p for _, p in entries])
    print(json.dumps([{"comparison": s, "p": p, "p_holm": q} for (s, p), q in zip(entries, adj)], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--a", required=True)
    pair.add_argument("--b", required=True)
    pair.add_argument("--a-key", default="model", help="per_subject group of A: model, identity or real_flair")
    pair.add_argument("--b-key", default="model")
    pair.add_argument("--name", required=True)
    pair.add_argument("--primary", required=True, help="metric:higher|lower")
    pair.add_argument("--noninf", nargs="*", default=[], help="metric:higher|lower:margin (margin may end in %%)")
    pair.add_argument("--metrics", nargs="*", default=[], help="extra (exploratory) metrics")
    pair.add_argument("--seed", type=int, default=0)
    pair.add_argument("--into", default=None)
    pair.set_defaults(func=cmd_pair)
    h = sub.add_parser("holm")
    h.add_argument("items", nargs="+", help="<result.json>:<comparison name>")
    h.set_defaults(func=cmd_holm)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
