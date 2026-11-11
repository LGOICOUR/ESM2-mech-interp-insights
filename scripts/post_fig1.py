"""Figure 1 of the write-up: what removing the class-mean plane costs by depth.

A: the plane removed at one layer at a time (P9), with the identity-controlled
   plane at the three layers where it was run and the random-plane controls.
B: the plane removed at every layer of a band (P10): cost.
C: the same bands: what a new linear probe still reads at layer 30.

Reads only saved metrics; runs no model. Cost is plotted as a percentage drop in
the likelihood of the true residue, 1 - exp(mean change in log-likelihood).

Usage:
    uv run python scripts/post_fig1.py [--recording esm2_t30_150M]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, validated all-pairs
BANDS = ["29-30", "26-30", "20-30"]


def pct(d):
    """Mean change in log-likelihood (nats) -> percentage drop in likelihood."""
    return (1.0 - np.exp(np.asarray(d, dtype=float))) * 100.0


def dot(ax, x, y, color, r=4.5, z=5):
    ax.plot(x, y, "o", ms=2 * r, mfc=color, mec=SURFACE, mew=1.6, zorder=z, clip_on=False)


def frame(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)


def panel_title(ax, letter, title, sub=None):
    ax.text(0, 1.11 if sub else 1.05, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="bottom")
    ax.text(0.045 if ax.get_subplotspec().colspan.stop - ax.get_subplotspec().colspan.start > 1 else 0.09, 1.11 if sub else 1.05,
            title, transform=ax.transAxes, fontsize=10.5, color=INK, va="bottom")
    if sub:
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    args = ap.parse_args()
    rec = args.recording
    p9 = json.loads((ROOT / "results" / "p9" / rec / "metrics.json").read_text())["conditions"]
    p9s = json.loads((ROOT / "results" / "p9" / f"{rec}_strat" / "metrics.json").read_text())["conditions"]
    p10 = json.loads((ROOT / "results" / "p10" / rec / "metrics.json").read_text())
    p10s = json.loads((ROOT / "results" / "p10" / f"{rec}_strat" / "metrics.json").read_text())["conditions"]

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(7.6, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.02], hspace=0.6, wspace=0.34, left=0.085, right=0.955, top=0.93, bottom=0.2)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- A: one layer at a time
    layers = np.arange(1, 31)
    plain = np.array([p9[f"dm_sub@{L}"]["delta_pll"]["all"] for L in layers])  # mean, lo, hi (nats)
    frame(ax_a, "drop in true-residue likelihood")
    ax_a.fill_between(layers, pct(plain[:, 2]), pct(plain[:, 1]), color=BLUE, alpha=0.14, lw=0, zorder=2)
    ax_a.plot(layers, pct(plain[:, 0]), color=BLUE, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    rnd_layers = sorted(int(k.split("@")[1]) for k in p9 if k.startswith("random_sub@"))
    for L in rnd_layers:
        dot(ax_a, L, pct(p9[f"random_sub@{L}"]["delta_pll"]["all"][0]), MUTED, r=3.2, z=4)
    for L in (1, 29, 30):
        m, lo, hi = p9s[f"dm_strat_sub@{L}"]["delta_pll"]["all"]
        dot(ax_a, L, pct(m), ORANGE, z=6)
        dot(ax_a, L, pct(plain[L - 1, 0]), BLUE, z=5)
    lab = dict(fontsize=8.5, color=INK, zorder=7)
    ax_a.text(1.7, pct(plain[0, 0]), f"{pct(plain[0, 0]):.1f}%", ha="left", va="center", **lab)
    ax_a.plot([1.25, 3.05], [0.45, 3.0], color=AXIS, lw=0.8, zorder=4)
    ax_a.text(3.2, 3.25, f"{pct(p9s['dm_strat_sub@1']['delta_pll']['all'][0]):.1f}%", ha="left", va="center", **lab)
    ax_a.text(28.35, pct(plain[28, 0]) + 0.75, f"{pct(plain[28, 0]):.1f}%", ha="right", va="center", **lab)
    ax_a.text(28.35, pct(p9s["dm_strat_sub@29"]["delta_pll"]["all"][0]) - 1.0,
              f"{pct(p9s['dm_strat_sub@29']['delta_pll']['all'][0]):.1f}%", ha="right", va="center", **lab)
    ax_a.text(30.7, pct(plain[29, 0]), f"{pct(plain[29, 0]):.1f}%", ha="left", va="center", **lab)
    ax_a.text(30.7, pct(p9s["dm_strat_sub@30"]["delta_pll"]["all"][0]),
              f"{pct(p9s['dm_strat_sub@30']['delta_pll']['all'][0]):.1f}%", ha="left", va="center", **lab)
    ax_a.set_xlim(0.2, 32.6)
    ax_a.set_ylim(0, 16)
    ax_a.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax_a.set_yticks([0, 5, 10, 15], ["0", "5%", "10%", "15%"])
    ax_a.set_xlabel("layer where the plane is removed", fontsize=8.5)
    panel_title(ax_a, "A", "Plane removed at one layer")

    # ---- B and C: every layer of a band
    xs = np.arange(3)
    series = [  # label key, source, colour
        ("sequential", p10["conditions"], BLUE),
        ("stale", p10["conditions"], AQUA),
        ("sequential_strat", p10s, ORANGE),
        ("random", p10["conditions"], MUTED),
    ]
    frame(ax_b, "drop in true-residue likelihood")
    frame(ax_c, "new linear probe at layer 30, macro-F1")
    base_f1 = p10["baseline"]["decodability"]["refit_probe"]["macro_f1"]
    for y, text, dy, va in ((base_f1, f"unedited {base_f1:.2f}", 0.02, "bottom"), (0.20, "floor 0.20, always coil", -0.022, "top")):
        ax_c.axhline(y, color=AXIS, lw=0.9, zorder=1)
        ax_c.text(-0.28, y + dy, text, fontsize=7.8, color=MUTED, ha="left", va=va, zorder=2)
    for mode, src, color in series:
        cost = np.array([src[f"{mode}@{b}"]["delta_pll"]["all"] for b in BANDS])
        f1 = np.array([src[f"{mode}@{b}"]["decodability"]["refit_probe"]["macro_f1"] for b in BANDS])
        r, z = (3.4, 4) if mode == "random" else (4.5, 5)
        dx = {"sequential": -0.045, "stale": 0.045}.get(mode, 0.0)  # these two overlap at the narrow band
        for ax, vals in ((ax_b, pct(cost[:, 0])), (ax_c, f1)):
            ax.plot(xs + dx, vals, color=color, lw=2 if mode != "random" else 1.4, solid_joinstyle="round", zorder=z - 1)
            for x, v in zip(xs + dx, vals):
                dot(ax, x, v, color, r=r, z=z)
        ax_b.text(2.17, pct(cost[2, 0]), f"{pct(cost[2, 0]):.0f}%", ha="left", va="center", **lab)
        if mode != "random":
            ax_c.text(2.17, f1[2], f"{f1[2]:.2f}", ha="left", va="center", **lab)
    band_labels = ["29–30\n2 layers", "26–30\n5 layers", "20–30\n11 layers"]
    for ax in (ax_b, ax_c):
        ax.set_xlim(-0.32, 2.62)
        ax.set_xticks(xs, band_labels)
    ax_b.set_ylim(0, 56)
    ax_b.set_yticks([0, 10, 20, 30, 40, 50], ["0", "10%", "20%", "30%", "40%", "50%"])
    ax_c.set_ylim(0, 0.92)
    ax_c.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    panel_title(ax_b, "B", "Plane removed at every layer of a band", "cost to the model")
    panel_title(ax_c, "C", "The same bands", "what a new linear probe still reads at layer 30")

    handles = [
        Line2D([], [], color=BLUE, lw=0, marker="o", ms=7, mec=SURFACE, mew=1.2, label="class-mean plane (B, C: each plane refit after the edits below it)"),
        Line2D([], [], color=AQUA, lw=0, marker="o", ms=7, mec=SURFACE, mew=1.2, label="class-mean plane, every plane fit before any edit (B, C)"),
        Line2D([], [], color=ORANGE, lw=0, marker="o", ms=7, mec=SURFACE, mew=1.2, label="identity-controlled plane (B, C: refit as above)"),
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label="random plane"),
    ]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.012), ncol=1, frameon=False, fontsize=8.3,
                     handlelength=1.2, labelspacing=0.32, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK2)

    out = ROOT / "results" / "post"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig1_depth.png", dpi=220, facecolor=SURFACE)
    print(f"wrote {out / 'fig1_depth.png'}")


if __name__ == "__main__":
    main()
