"""Figures and generated tables for paper/main.tex. Inputs are small JSON files in paper/data/ only.

    python paper/figures/make_figures.py

Writes paper/figures/*.pdf and paper/tables/*.tex. Numbers come from:
  data/identity_main_val.json  baseline + per-subject metrics of the single eval entry on main_val (smoke job 4362831)
  data/literature.json         published T2->FLAIR results with their protocols (verified against the sources listed)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
FIG, TAB, DATA = PAPER / "figures", PAPER / "tables", PAPER / "data"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
INK, ACCENT, MUTED = "#1f2937", "#2563eb", "#9ca3af"


def ms(stat, digits, scale=1.0):
    return "%.*f$\\pm$%.*f" % (digits, stat["mean"] * scale, digits, stat["std"] * scale)


def ours_table(d):
    idt, up = d["baseline"]["identity_t2"], d["baseline"]["upper_bound_real_flair"]
    p = "\\prelim"
    head = ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccc|cccccc}\n\\toprule\n"
            "& \\multicolumn{4}{c|}{Image quality (brain mask; LPIPS full slice)} & \\multicolumn{6}{c}{Lesion level ($S_{\\mathrm{eval}}$)} \\\\\n"
            "Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & MAE$\\downarrow$ & LPIPS$\\downarrow$ & Dice$\\uparrow$ & HD95$\\downarrow$ & "
            "FP vol.\\ ($10^3$)$\\downarrow$ & Miss$\\downarrow$ & SLR$\\uparrow$ & EFA$\\downarrow$ \\\\\n\\midrule\n")
    rows = ["Identity (T2) & %s & %s & %s & %s & %s%s & %.1f%s & %.1f%s & %.2f%s & %.2f%s & %.2f%s \\\\" % (
        ms(idt["psnr"], 2), ms(idt["ssim"], 3), ms(idt["mae"], 3), ms(idt["lpips"], 3),
        ms(idt["dice"], 2), p, idt["hd95"]["mean"], p, idt["fp_volume"]["mean"] / 1e3, p, idt["miss_rate"]["mean"], p,
        idt["small_lesion_recall"]["pooled"], p, idt["empty_slice_false_alarm"]["pooled"], p),
        "Real FLAIR (reference) & -- & -- & -- & -- & %s%s & %.1f%s & %.1f%s & %.2f%s & %.2f%s & %.2f%s \\\\" % (
        ms(up["dice"], 2), p, up["hd95"]["mean"], p, up["fp_volume"]["mean"] / 1e3, p, up["miss_rate"]["mean"], p,
        up["small_lesion_recall"]["pooled"], p, up["empty_slice_false_alarm"]["pooled"], p),
        "\\midrule"]
    for name in ["UNet-$\\ell_1$", "pix2pix", "SelfRDB (retrained)", "F0 (noise start)", "F1 (ours, trunk)",
                 "F1 + lesion weight", "F1 + seg.\\ consistency"]:
        rows.append(name + " & " + " & ".join(["\\pend"] * 10) + " \\\\")
    foot = ("\\bottomrule\n\\end{tabular}\n}\n\n\\vspace{2pt}\n{\\footnotesize HD95 in voxels; FP vol.: voxels of $P$ farther than 5 voxels "
            "from the tumor; SLR: small-lesion recall (%d GT lesions $<$1000 voxels, pooled); EFA: false-alarm rate on %d "
            "tumor-free slices (pooled).}\n") % (idt["small_lesion_recall"]["n_lesions"], idt["empty_slice_false_alarm"]["n_slices"])
    (TAB / "ours.tex").write_text(head + "\n".join(rows) + "\n" + foot)
    (TAB / "identity_psnr_inline.tex").write_text("%.1f\\unskip" % idt["psnr"]["mean"])


def identity_figure(d):
    ident, real = d["per_subject"]["identity"], d["per_subject"]["real_flair"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    rng = np.random.default_rng(0)
    for ax, key, label in [(axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")]:
        v = np.array([s[key] for s in ident])
        ax.boxplot(v, widths=0.5, showfliers=False, medianprops={"color": INK}, boxprops={"color": MUTED},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        ax.scatter(1 + rng.uniform(-0.12, 0.12, v.size), v, s=8, color=ACCENT, alpha=0.8, zorder=3)
        ax.set_xticks([1], ["identity (T2)"])
        ax.set_ylabel(label)
        ax.set_title("%s: %.2f $\\pm$ %.2f" % (label.split()[0], v.mean(), v.std(ddof=1)), fontsize=8)
    by = {s["subject"]: s for s in real}
    pairs = np.array([(s["dice"], by[s["subject"]]["dice"]) for s in ident if s["subject"] in by])
    ax = axes[2]
    for a, b in pairs:
        ax.plot([0, 1], [a, b], color=MUTED, lw=0.6, zorder=1)
    ax.scatter(np.zeros(len(pairs)), pairs[:, 0], s=8, color=INK, zorder=3)
    ax.scatter(np.ones(len(pairs)), pairs[:, 1], s=8, color=ACCENT, zorder=3)
    ax.set_xticks([0, 1], ["identity (T2)", "real FLAIR"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("$S_{\\mathrm{eval}}$ Dice (preliminary)")
    ax.set_title("per-subject Dice, n=%d" % len(pairs), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "identity_baseline.pdf")
    plt.close(fig)


def literature(lit):
    """Table I and Fig. 2 from data/literature.json (see its 'rows' and 'protocols')."""
    protos = {p["paper"]: p for p in lit["protocols"]}
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Reported BraTS T2$\\to$FLAIR results. Values are copied from each paper's table; protocols differ, so "
             "rows from different papers are \\emph{not} comparable. SSIM in \\%%. %s}" % lit.get("caption_note", ""),
             "\\label{tab:lit}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llccl|l}", "\\toprule",
             "Source & Method & PSNR (dB) & SSIM (\\%) & Source table & Protocol of the source \\\\", "\\midrule"]
    for paper in lit["order"]:
        rows = [r for r in lit["rows"] if r["paper"] == paper]
        for i, r in enumerate(rows):
            proto = protos[paper]["short"] if i == 0 else ""
            cite = "%s~\\cite{%s}" % (protos[paper]["name"], protos[paper]["cite"]) if i == 0 else ""
            lines.append("%s & %s & %s & %s & %s & %s \\\\" % (cite, r["method"], r["psnr"], r["ssim"], r["table"] if i == 0 else "", proto))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table*}", ""]
    lines += ["\\begin{figure}[t]", "\\centering", "\\includegraphics[width=\\columnwidth]{figures/literature_protocols.pdf}",
              "\\caption{%s}" % lit["figure_caption"], "\\label{fig:lit}", "\\end{figure}", "", lit["paragraph"], ""]
    (TAB / "literature.tex").write_text("\n".join(lines))

    fams = {}
    for r in lit["rows"]:
        fams.setdefault(r["family"], []).append(r)
    fams = {k: v for k, v in fams.items() if len({r["paper"] for r in v}) >= 2}
    order = sorted(fams, key=lambda k: np.ptp([r["psnr_value"] for r in fams[k]]))
    colors = dict(zip(lit["order"], ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for y, fam in enumerate(order):
        vals = [r["psnr_value"] for r in fams[fam]]
        ax.hlines(y, min(vals), max(vals), color=MUTED, lw=1, zorder=1)
        for r in fams[fam]:
            dup = "ddagger" in r["method"]
            ax.scatter(r["psnr_value"], y, color=colors[r["paper"]], s=30 if dup else 18, zorder=5 if dup else 3,
                       marker="x" if dup else "o", linewidths=1.4)
        ax.annotate("%.1f dB" % np.ptp(vals), (max(vals), y), textcoords="offset points", xytext=(5, -3), fontsize=6.5, color=INK)
    ax.set_xlim(right=max(r["psnr_value"] for r in lit["rows"]) + 1.8)
    ax.set_yticks(range(len(order)), [f.replace("$^2$", "2") for f in order])
    ax.set_xlabel("reported PSNR on BraTS T2$\\to$FLAIR (dB)")
    for paper in lit["order"]:
        if any(r["paper"] == paper and r["family"] in fams for r in lit["rows"]):
            ax.scatter([], [], color=colors[paper], s=18, label=paper)
    ax.scatter([], [], color=MUTED, marker="x", s=18, label="digits match SelfRDB row")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(FIG / "literature_protocols.pdf")
    plt.close(fig)


def main():
    TAB.mkdir(exist_ok=True)
    d = json.loads((DATA / "identity_main_val.json").read_text())
    ours_table(d)
    identity_figure(d)
    if (DATA / "literature.json").exists():
        literature(json.loads((DATA / "literature.json").read_text()))
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")), sorted(p.name for p in TAB.glob("*.tex")))


if __name__ == "__main__":
    main()
