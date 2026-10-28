"""Post-hoc check for the remainder follow-up: how much of the residual stream does
each removed direction carry at the headline layer, and how much class signal?

A random unit direction in a 640-wide stream carries about 1/640 of the variance,
so the random control does not match the size of the perturbation a fitted
direction makes. This prints, per direction kind, the RMS size of the removed
component on held-out residues (directions fit on the training folds, as in P4)
as a share of the total variance, and for single directions the helix-vs-rest and
strand-vs-rest class-mean gaps along the direction in within-class standard
deviations (d'). Writes results/p4/<out>/variance_check.json.

Usage:
    uv run python scripts/check_variance.py            # layer L*, into results/p4/esm2_t30_150M_perp/
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from esm2_probe.probe import grouped_folds  # noqa: E402
from p4_ablate import KINDS, fit_directions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--out", default="esm2_t30_150M_perp")
    ap.add_argument("--layer", type=int, default=None, help="default: L*, the P3 folds' most chosen layer")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    tbl = np.load(src / "residues.npz")
    y, g = tbl["label"].astype(int), tbl["protein_idx"]
    folds = grouped_folds(g, 5, args.seed)
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L = args.layer or min(l for l, c in counts.items() if c == max(counts.values()))
    p3_C = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}
    cache, _ = fit_directions(acts, y, g, folds, [L], args.n_train, args.seed, lambda l, f: p3_C[l][f])

    X = np.asarray(acts[L], dtype=np.float32)
    total_var = float(((X - X.mean(0)) ** 2).sum(1).mean())
    out = {"layer": L, "n_residues": int(len(y)), "total_variance": total_var, "width": int(X.shape[1]), "per_kind": {}}
    for kind in KINDS:
        rms2, dprime = [], {"H": [], "E": []}
        for f in range(5):
            te = folds == f
            U = cache[f][L][kind]
            U2 = U[None] if U.ndim == 1 else U
            coord = (X[te] - cache[f][L]["mu"]) @ U2.T  # [n, d]
            rms2.append(float((coord ** 2).sum(1).mean()))
            if U.ndim == 1:
                c1 = coord[:, 0]
                for ci, name in ((0, "H"), (1, "E")):
                    a, b = c1[y[te] == ci], c1[y[te] != ci]
                    pooled = np.sqrt(0.5 * (a.var() + b.var()))
                    dprime[name].append(float((a.mean() - b.mean()) / pooled))
        out["per_kind"][kind] = {"rms_removed": float(np.sqrt(np.mean(rms2))),
                                 "share_of_total_variance": float(np.mean(rms2) / total_var),
                                 "dprime_H_vs_rest": float(np.mean(dprime["H"])) if dprime["H"] else None,
                                 "dprime_E_vs_rest": float(np.mean(dprime["E"])) if dprime["E"] else None}
    out_dir = ROOT / "results" / "p4" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variance_check.json").write_text(json.dumps(out, indent=1))

    print(f"layer {L}: total variance per residue {total_var:.1f}; a random direction's share would be 1/{X.shape[1]} = {1 / X.shape[1]:.4f}")
    print("| kind | RMS of removed component | share of total variance | d' helix vs rest | d' strand vs rest |")
    print("|---|---|---|---|---|")
    for kind, r in out["per_kind"].items():
        dh = f"{r['dprime_H_vs_rest']:+.2f}" if r["dprime_H_vs_rest"] is not None else "–"
        de = f"{r['dprime_E_vs_rest']:+.2f}" if r["dprime_E_vs_rest"] is not None else "–"
        print(f"| {kind} | {r['rms_removed']:.2f} | {r['share_of_total_variance']:.4f} | {dh} | {de} |")
    print(f"wrote {out_dir / 'variance_check.json'}")


if __name__ == "__main__":
    main()
