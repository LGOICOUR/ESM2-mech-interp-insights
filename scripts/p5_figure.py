"""P5 figure: dose-response of the steering. Left and middle: change in the
expected helix- and strand-propensity of the model's prediction at the steered
coil residues, per direction, with 95% CIs, against the level the model's own
helix / strand residues sit at. Right: how far the prediction moved (KL).

Usage:
    uv run python scripts/p5_figure.py [--recording esm2_t30_150M]
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
STYLE = {"dm_H": ("#2a78d6", "-", "o", "class-mean helix dir."),
         "dm_H_strat": ("#2a78d6", "--", "s", "within-amino-acid helix dir."),
         "random_H": ("#8a8782", "-", "o", "random dir., helix norm"),
         "dm_E": ("#eb6834", "-", "o", "class-mean strand dir."),
         "dm_E_strat": ("#eb6834", "--", "s", "within-amino-acid strand dir."),
         "random_E": ("#8a8782", "--", "s", "random dir., strand norm")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    args = ap.parse_args()
    d = ROOT / "results" / "p5" / args.recording
    m = json.loads((d / "metrics.json").read_text())
    t = m["target_class"]
    tname = {"H": "helix", "E": "strand", "C": "coil"}[t]
    doses = m["doses"]
    kinds = [k for k in STYLE if any(f"{k}@{x:g}" in m["conditions"] for x in doses)]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), facecolor=SURFACE)
    panels = [("H", f"Δ helix-propensity of the prediction at {tname} residues (nats)"),
              ("E", f"Δ strand-propensity of the prediction at {tname} residues (nats)")]
    for ax, (sc, label) in zip(axes[:2], panels):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        G = m["baseline"]["gap_G"].get(sc)
        if G:
            ax.axhline(G[0], color=INK2, lw=1, ls=":", zorder=1)
            ax.text(doses[0] * 0.97, G[0], f"level of the model's own {'helix' if sc == 'H' else 'strand'} residues",
                    fontsize=7.5, color=INK2, va="bottom", ha="left")
        for k in kinds:
            color, ls, marker, lab = STYLE[k]
            xs, ys, lo, hi = [], [], [], []
            for x in doses:
                r = m["conditions"].get(f"{k}@{x:g}")
                if r is None:
                    continue
                v = r["delta_S_by_score_class_by_true_class"][sc][t]
                xs.append(x); ys.append(v[0]); lo.append(v[1]); hi.append(v[2])
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], color=color, ls=ls,
                        marker=marker, ms=4.5, lw=1.4, capsize=2, label=lab, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(doses)
        ax.set_xticklabels([f"{x:g}" for x in doses])
        ax.set_xlabel("dose (class-vs-coil gaps along the direction)", fontsize=9)
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK)
    ax = axes[2]
    ax.set_facecolor(SURFACE)
    for k in kinds:
        color, ls, marker, lab = STYLE[k]
        xs, ys = [], []
        for x in doses:
            r = m["conditions"].get(f"{k}@{x:g}")
            if r is not None:
                xs.append(x); ys.append(r["kl_by_class"][t][0])
        ax.plot(xs, ys, color=color, ls=ls, marker=marker, ms=4.5, lw=1.4, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{x:g}" for x in doses])
    ax.set_xlabel("dose", fontsize=9)
    ax.set_ylabel(f"KL(steered ‖ unsteered) at {tname} residues (nats)", fontsize=8.5)
    ax.set_title("how far the prediction moved", color=INK, fontsize=9.5, loc="left")
    ax.grid(axis="y", color=GRID, lw=1, which="both")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle(f"ESM-2 150M: adding a secondary-structure direction to {tname} residues at layer {m['L_star']} during the forward pass "
                 f"({m['n_chains']} held-out chains, {m['n_target_residues']:,} {tname} residues; bars = 95% CI over chains)",
                 color=INK, fontsize=9.5, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.15, wspace=0.32)
    out = d / "steering.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
