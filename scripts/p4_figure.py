"""P4 figure: change in the model's own masked log-likelihood, by true class, for
each removed direction at each scope, with 95% CIs; random controls alongside.
Second panel: what a refit probe still decodes at the ablated layer.

Usage:
    uv run python scripts/p4_figure.py [--recording esm2_t30_150M]
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
CLASS_COLORS = {"H": "#2a78d6", "E": "#eb6834", "C": "#8a8782"}
CLASS_LABEL = {"H": "helix residues", "E": "strand residues", "C": "coil residues"}
KIND_LABEL = {"probe_H": "probe helix dir.", "probe_E": "probe strand dir.", "dm_H": "class-mean helix dir.",
              "dm_E": "class-mean strand dir.", "probe_sub": "probe plane (rank 2)", "dm_sub": "class-mean plane (rank 2)",
              "random": "random dir. (control)", "random_sub": "random plane (control)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    args = ap.parse_args()
    d = ROOT / "results" / "p4" / args.recording
    m = json.loads((d / "metrics.json").read_text())
    scopes = list(m["scopes"])
    kinds = [k for k in KIND_LABEL if any(f"{s}/{k}" in m["conditions"] for s in scopes)]
    L_star = m["L_star"]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(scopes) + 1, figsize=(4.0 * (len(scopes) + 1), 5.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1] * len(scopes) + [0.85]})
    ypos = np.arange(len(kinds))[::-1]
    for ax, scope in zip(axes, scopes):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=GRID, lw=1, zorder=1)
        for j, c in enumerate(("H", "E", "C")):
            off = (1 - j) * 0.24
            for i, k in enumerate(kinds):
                r = m["conditions"].get(f"{scope}/{k}")
                if r is None:
                    continue
                mean, lo, hi = r["delta_pll_by_class"][c]
                ax.plot([lo, hi], [ypos[i] + off] * 2, color=CLASS_COLORS[c], lw=1.4, zorder=2)
                ax.plot(mean, ypos[i] + off, "o", color=CLASS_COLORS[c], ms=4.5, zorder=3,
                        label=CLASS_LABEL[c] if i == 0 else None)
        layers = m["scopes"][scope]
        lay = f"layer {layers[0]}" if len(layers) == 1 else f"layers {layers[0]}–{layers[-1]}"
        ax.set_title(f"removed at {lay} ({scope})", color=INK, fontsize=10, loc="left")
        ax.set_yticks(ypos)
        ax.set_yticklabels([KIND_LABEL[k] for k in kinds] if ax is axes[0] else [""] * len(kinds), fontsize=8.5)
        ax.set_xlabel("Δ masked log-likelihood of the true residue (nats)", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    # decodability panel: refit vs original probe macro-F1 at L*, per scope/kind
    ax = axes[-1]
    ax.set_facecolor(SURFACE)
    base = m["baseline"]["decodability"][str(L_star)]["refit_probe"]["macro_f1"]
    ax.axvline(base, color=INK2, lw=1, zorder=1)
    markers = {"single": "o", "band": "s", "mid": "^", "wide": "D"}
    scope_name = {"single": f"layer {L_star}", "band": f"layers {m['scopes'].get('band', [0])[0]}–{m['scopes'].get('band', [0])[-1]}",
                  "mid": f"layers {m['scopes'].get('mid', [0])[0]}–{m['scopes'].get('mid', [0])[-1]}", "wide": "wide"}
    for scope in scopes:
        xo, xr = [], []
        for k in kinds:
            r = m["conditions"].get(f"{scope}/{k}")
            dec = r["decodability"][str(L_star)] if r else None
            xo.append(dec["original_probe"]["macro_f1"] if dec else np.nan)
            xr.append(dec["refit_probe"]["macro_f1"] if dec else np.nan)
        ax.plot(xo, ypos, markers[scope], color=SURFACE, mec=INK2, ms=5.5, ls="none", zorder=2)
        ax.plot(xr, ypos, markers[scope], color=CLASS_COLORS["H"], ms=5.5, ls="none", zorder=3, label=f"removed at {scope_name[scope]}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.set_xlim(0.1, 0.9)
    ax.set_xlabel(f"probe macro-F1 at layer {L_star} after ablation\n(line = no ablation, {base:.2f})", fontsize=9)
    ax.set_title("is the concept still there?\nhollow = original probe, filled = refit probe", color=INK, fontsize=9.5, loc="left")
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK, handletextpad=0.3)
    fig.suptitle(f"ESM-2 150M: projecting a secondary-structure direction out of the residual stream during the forward pass "
                 f"({m['n_chains']} held-out chains, {m['n_residues']:,} residues; bars = 95% CI over chains)",
                 color=INK, fontsize=9.5, x=0.01, ha="left")
    fig.subplots_adjust(left=0.115, right=0.99, top=0.84, bottom=0.17, wspace=0.14)
    out = d / "ablation.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
