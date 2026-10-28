"""P3 figure: probe accuracy and macro-F1 at every layer with 95% CI bands, the
no-model baselines as flat lines, and the class-mean decoder dashed.

Usage:
    uv run python scripts/p3_figure.py [--recording esm2_t30_150M]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a9d5d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    args = ap.parse_args()
    d = ROOT / "results" / "p3" / args.recording
    m = json.loads((d / "metrics.json").read_text())
    c = np.load(d / "layer_curves.npz")
    layers = c["layers"]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    panels = [("accuracy", "probe_acc", "probe_acc_ci", "centroid_acc", "accuracy (Q3)"),
              ("macro_f1", "probe_f1", "probe_f1_ci", "centroid_f1", "macro-F1")]
    for ax, (key, pk, ck, nk, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for name, label, color in (("majority", "majority class", GRID), ("identity", "one-hot identity (no model)", INK2),
                                   ("window3", "one-hot ±3 window (no model)", ORANGE)):
            v = m["baselines"][name][key]
            ax.axhline(v, color=color, lw=1.2, zorder=1)
            ax.text(layers[-1], v + 0.004, f"{label} {v:.2f}", color=INK2 if color == GRID else color, fontsize=8, ha="right", va="bottom")
        ci = c[ck]
        ax.fill_between(layers, ci[:, 0], ci[:, 1], color=BLUE, alpha=0.18, lw=0, zorder=2)
        ax.plot(layers, c[pk], color=BLUE, lw=2, zorder=3, label="logistic probe (held-out chains)")
        ax.plot(layers, c[nk], color=GREEN, lw=1.5, ls="--", zorder=3, label="class-mean decoder")
        for f, L in enumerate(m["chosen_layer_per_fold"]):
            ax.axvline(L, color=BLUE, lw=0.6, alpha=0.35, zorder=1)
        ax.set_xlabel("layer (0 = embedding output)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(layers[0], layers[-1])
    h = m["headline"]
    fig.suptitle(f"ESM-2 150M, per-residue secondary structure (helix / strand / coil): {m['n_residues']:,} residues in "
                 f"{m['n_proteins']} held-out chains\nheadline probe at the layer chosen inside training ({m['chosen_layer_per_fold']}): "
                 f"accuracy {h['accuracy']:.3f} [{h['accuracy_ci'][0]:.3f}, {h['accuracy_ci'][1]:.3f}], "
                 f"macro-F1 {h['macro_f1']:.3f} [{h['macro_f1_ci'][0]:.3f}, {h['macro_f1_ci'][1]:.3f}]",
                 color=INK, fontsize=9.5, x=0.01, ha="left")
    axes[0].set_title("accuracy by layer; shaded = 95% CI from resampling chains", color=INK2, fontsize=9, loc="left")
    axes[1].set_title("macro-F1 by layer; vertical line = layer chosen inside training", color=INK2, fontsize=9, loc="left")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = d / "layer_curves.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
