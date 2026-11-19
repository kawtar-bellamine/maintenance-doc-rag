"""Figures for the README, from results/metrics.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREY, ORANGE = "#1f5fa8", "#9aa7b5", "#e69138"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    r = json.loads((ROOT / "results/metrics.json").read_text())
    out = ROOT / "results/figures"; out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), dpi=140)
    names = ["bm25", "dense", "hybrid", "rerank"]
    labels = ["BM25", "Word vectors\n(spaCy, SIF)", "Hybrid\n(RRF)", "Page routing +\nlearned reranker"]
    for ax, m, t in zip(axes, ["recall_at_1", "recall_at_5"], ["Right passage ranked 1st", "Right passage in top 5"]):
        v = [r["retrieval"][n][m] for n in names]
        b = ax.bar(labels, v, color=[GREY, GREY, GREY, BLUE])
        ax.bar_label(b, labels=[f"{x:.0%}" for x in v], fontsize=9)
        if m == "recall_at_1":
            lo, hi = r["retrieval"]["rerank"]["recall_at_1_ci95"]
            ax.errorbar(3, v[3], yerr=[[v[3] - lo], [hi - v[3]]], color="k", capsize=4, lw=1)
        ax.set_ylim(0, 1.08); ax.set_title(t, fontsize=11); ax.tick_params(axis="x", labelsize=8.5); style(ax)
    c = r["corpus"]
    fig.suptitle(f"{r['n_questions']} handwritten questions on the PX4 User Guide ({c['pages']} pages, "
                 f"{c['chunks']} passages), 5-fold cross-validation by page", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "fig1_retrieval.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=140)
    meths = ["top_passage", "direct", "learned"]
    lab = ["Images of the\ntop passage", "Direct search on\nalt text + OCR", "Learned fusion\n(shown if confident)"]
    x = range(len(meths)); w = 0.38
    p = [r["images"][m]["precision"] for m in meths]; rc = [r["images"][m]["recall"] for m in meths]
    b1 = ax.bar([i - w / 2 for i in x], p, w, color=BLUE, label="precision of the images shown")
    b2 = ax.bar([i + w / 2 for i in x], rc, w, color=ORANGE, label="right image shown (recall)")
    ax.bar_label(b1, labels=[f"{v:.0%}" for v in p], fontsize=9)
    ax.bar_label(b2, labels=[f"{v:.0%}" for v in rc], fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(lab, fontsize=9); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, frameon=False); style(ax)
    ax.set_title(f"Which images to show ({r['images']['learned']['questions_with_a_figure']} questions need one, "
                 f"{c['figures']} images in the guide)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig2_images.png"); plt.close(fig)


if __name__ == "__main__":
    main()
