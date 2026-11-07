"""P9: the cost of the class-mean plane as a function of the layer it is removed at.

P4 removed the plane at L* (and in bands); P7/P8 stayed at L*. This sweep removes
the rank-2 class-mean plane (dm_sub) at one layer at a time, every layer 1..n,
and reads the model's own output only (masked pseudo-log-likelihood of the true
residue and top-1, k-pass masking), no recorded activations and no refits. The
plane at each layer is fit per fold on that layer's training rows, the same
subsample as P3/P4. A random rank-2 plane is run as the control at every
--control-every-th layer.

Put beside P3's per-layer decodability (probe F1 and class-mean-decoder F1, read
from results/p3), the sweep gives decodable-vs-used across depth in one table:
removing the plane at the last layer leaves the output head no chance to rebuild,
removing it earlier lets later layers recover.

Outputs in results/p9/<recording>[_limitN][_<tag>]/: metrics.json, summary.md, sweep.png

Usage:
    uv run python scripts/p9_layer_sweep.py
    uv run python scripts/p9_layer_sweep.py --limit 12 --layers 29 30   # quick check
    uv run python scripts/p9_layer_sweep.py --layers 1 29 30 --kinds dm_sub dm_strat_sub --control-every 1000 --tag strat
        # composition control: the within-amino-acid class-mean plane at the layers where the plain plane costs most
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.ablate import AblationContext, attach, orthonormal_rows, random_unit  # noqa: E402
from esm2_probe.data import CLASSES, read_jsonl  # noqa: E402
from esm2_probe.probe import diff_in_means, grouped_bootstrap_mean, grouped_folds, subsample  # noqa: E402
from esm2_probe.reader import load_model, masked_readout  # noqa: E402
from esm2_probe.steer import stratified_diff_in_means  # noqa: E402


def fit_planes(acts, y, folds, layers, n_train, seed, aa=None):
    """Per fold and layer: training mean, class-mean plane (rank 2), random plane (rank 2), and when
    `aa` is given the within-amino-acid class-mean plane (rank 2; no composition difference by construction)."""
    cache = {}
    for f in range(int(folds.max()) + 1):
        cache[f] = {}
        tr = np.where(folds != f)[0]
        fit_idx = subsample(tr, n_train, seed + 100 + f)  # same residues as P3/P4
        for L in layers:
            X = np.asarray(acts[L][fit_idx], dtype=np.float32)
            dm, mu = diff_in_means(X, y[fit_idx])
            W = X.shape[1]
            cache[f][L] = {"mu": mu, "dm_sub": orthonormal_rows(dm[:2]),
                           "random_sub": orthonormal_rows(np.stack([random_unit(W, 7000 + 1000 * f + L),
                                                                     random_unit(W, 8000 + 1000 * f + L)]))}
            if aa is not None:
                st = np.stack([stratified_diff_in_means(X, y[fit_idx], aa[fit_idx], c)[0] for c in (0, 1)])
                cache[f][L]["dm_strat_sub"] = orthonormal_rows(st)
    return cache


def by_class(values, y, g, n_boot, seed=0):
    out = {name: list(grouped_bootstrap_mean(values[y == c], g[y == c], n_boot=n_boot, seed=seed))
           for c, name in enumerate(CLASSES)}
    out["all"] = list(grouped_bootstrap_mean(values, g, n_boot=n_boot, seed=seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="layers to ablate at (default: 1..n_layers)")
    ap.add_argument("--control-every", type=int, default=5, help="run the random-plane control at every Nth layer (and the last)")
    ap.add_argument("--kinds", nargs="+", default=["dm_sub"], choices=["dm_sub", "dm_strat_sub"],
                    help="planes to remove: class-mean (dm_sub) and/or within-amino-acid class-mean (dm_strat_sub, the composition control)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    n_layers = acts.shape[0] - 1
    tbl = np.load(src / "residues.npz")
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    p3_by_layer = {r["layer"]: r for r in p3["per_layer"]}
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L_star = min(L for L, c in counts.items() if c == max(counts.values()))
    layers = args.layers or list(range(1, n_layers + 1))
    control_layers = sorted({L for L in layers if L % args.control_every == 0} | ({n_layers} if n_layers in layers else set()))

    y_all, g_all, aa_all = tbl["label"].astype(int), tbl["protein_idx"], tbl["aa"].astype(int)
    folds_all = grouped_folds(g_all, 5, args.seed)
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g_all, minlength=len(proteins)), out=starts[1:])
    chain_ids = list(range(len(proteins)))[: args.limit] if args.limit else list(range(len(proteins)))
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chain_ids])
    y, g = y_all[rows], g_all[rows]
    lab_pos = {i: tbl["position"][starts[i]:starts[i + 1]] for i in chain_ids}
    out_dir = ROOT / "results" / "p9" / (args.recording + (f"_limit{args.limit}" if args.limit else "")
                                         + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = fit_planes(acts, y_all, folds_all, layers, args.n_train, args.seed, aa=aa_all if "dm_strat_sub" in args.kinds else None)
    print(f"L* = {L_star}; sweep layers {layers}; controls at {control_layers}; planes fit in {time.time() - t0:.0f}s", flush=True)

    tok, model, device = load_model()
    ctx = AblationContext()

    def run_condition(dirs_by_fold):
        logp, top1 = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
        for f in range(5):
            handles = attach(model, dirs_by_fold[f], ctx, device) if dirs_by_fold is not None else []
            try:
                for i in chain_ids:
                    if folds_all[starts[i]] != f:
                        continue
                    sel = np.where(g == i)[0]
                    r = masked_readout(tok, model, device, proteins[i].sequence, args.k, ctx=ctx)
                    logp[sel], top1[sel] = r.logp_true[lab_pos[i]], r.top1[lab_pos[i]]
            finally:
                for h in handles:
                    h.remove()
        return logp, top1

    logp0, top10 = run_condition(None)
    true_tok = None
    # top-1 accuracy needs the true token; masked_readout's top1 is compared against it below via logp>log(0.5)? no:
    # recover it from the tokenizer once (cheap)
    true_tok = np.concatenate([np.asarray(tok(proteins[i].sequence)["input_ids"])[1:-1][lab_pos[i]] for i in chain_ids])
    print(f"baseline read in {time.time() - t0:.0f}s; top-1 {np.mean(top10 == true_tok):.3f}", flush=True)

    results = {"recording": args.recording, "L_star": L_star, "layers": layers, "control_layers": control_layers,
               "k": args.k, "n_chains": len(chain_ids), "n_residues": int(len(y)),
               "baseline": {"top1": float(np.mean(top10 == true_tok))}, "conditions": {}}
    per_residue = {"y": y, "g": g, "logp_none": logp0, "top1_none": top10, "true_tok": true_tok}

    def record(name, L, kind):
        dirs = {f: {L: (cache[f][L][kind], cache[f][L]["mu"])} for f in cache}
        logp, top1 = run_condition(dirs)
        d = logp - logp0
        results["conditions"][name] = {"layer": L, "kind": kind, "delta_pll": by_class(d, y, g, args.n_boot),
                                       "top1": float(np.mean(top1 == true_tok)),
                                       "top1_by_class": {c: float(np.mean((top1 == true_tok)[y == ci])) for ci, c in enumerate(CLASSES)}}
        per_residue[f"logp_{name}"] = logp
        r = results["conditions"][name]
        print(f"{name:>16}: dPLL {r['delta_pll']['all'][0]:+.4f} "
              f"[H {r['delta_pll']['H'][0]:+.3f} E {r['delta_pll']['E'][0]:+.3f} C {r['delta_pll']['C'][0]:+.3f}] "
              f"top-1 {r['top1']:.3f}; {time.time() - t0:.0f}s", flush=True)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))

    for L in layers:
        for kind in args.kinds:
            record(f"{kind}@{L}", L, kind)
        if L in control_layers:
            record(f"random_sub@{L}", L, "random_sub")

    results["seconds"] = round(time.time() - t0)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out_dir / "per_residue.npz", **per_residue)

    # summary table: cost by layer beside P3's decodability at that layer
    def f(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"

    lines = [f"# P9 layer sweep, `{args.recording}`: class-mean plane removed at one layer at a time; "
             f"{len(chain_ids)} held-out chains, {len(y)} labelled residues; L* = {L_star}", "",
             f"Baseline top-1 {results['baseline']['top1']:.1%}. ΔPLL in nats, mean [95% CI, chains resampled]; "
             "P3 columns are decodability at that layer before any edit (macro-F1).", "",
             "| layer | ΔPLL plane | helix | strand | coil | top-1 | ΔPLL random plane | P3 probe F1 | P3 class-mean F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    main_kind = args.kinds[0]
    lines[4] = lines[4].replace("ΔPLL plane", f"ΔPLL {main_kind}") + ("" if len(args.kinds) == 1 else " other kinds |")
    lines[5] = lines[5] + ("" if len(args.kinds) == 1 else "---|")
    for L in layers:
        r = results["conditions"][f"{main_kind}@{L}"]
        rc = results["conditions"].get(f"random_sub@{L}")
        p = p3_by_layer.get(L)
        lines.append(f"| {L}{' (L*)' if L == L_star else ''} | {f(r['delta_pll']['all'])} | {r['delta_pll']['H'][0]:+.3f} | "
                     f"{r['delta_pll']['E'][0]:+.3f} | {r['delta_pll']['C'][0]:+.3f} | {r['top1']:.1%} | "
                     f"{f(rc['delta_pll']['all']) if rc else ''} | "
                     f"{p['probe']['macro_f1']:.3f} | {p['centroid']['macro_f1']:.3f} |"
                     + ("" if len(args.kinds) == 1 else " " + "; ".join(f"{k} {results['conditions'][f'{k}@{L}']['delta_pll']['all'][0]:+.3f}" for k in args.kinds[1:]) + " |") if p else
                     f"| {L} | {f(r['delta_pll']['all'])} | {r['delta_pll']['H'][0]:+.3f} | {r['delta_pll']['E'][0]:+.3f} | "
                     f"{r['delta_pll']['C'][0]:+.3f} | {r['top1']:.1%} | {f(rc['delta_pll']['all']) if rc else ''} | | |")
    lines += ["", f"- {results['seconds']}s on {device}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4))
        Ls = layers
        cost = [-results["conditions"][f"{main_kind}@{L}"]["delta_pll"]["all"][0] for L in Ls]
        lo = [-results["conditions"][f"{main_kind}@{L}"]["delta_pll"]["all"][2] for L in Ls]
        hi = [-results["conditions"][f"{main_kind}@{L}"]["delta_pll"]["all"][1] for L in Ls]
        ax1.plot(Ls, cost, "o-", color="C3", label="model cost of removing the plane (nats)")
        ax1.fill_between(Ls, lo, hi, color="C3", alpha=0.2)
        cl = [L for L in control_layers if f"random_sub@{L}" in results["conditions"]]
        ax1.plot(cl, [-results["conditions"][f"random_sub@{L}"]["delta_pll"]["all"][0] for L in cl], "x", color="gray", label="random plane")
        ax1.set_xlabel("layer the plane is removed at")
        ax1.set_ylabel("−ΔPLL (nats)")
        ax2 = ax1.twinx()
        pl = [L for L in Ls if L in p3_by_layer]
        ax2.plot(pl, [p3_by_layer[L]["probe"]["macro_f1"] for L in pl], "s--", color="C0", label="probe F1 at that layer (P3)")
        ax2.plot(pl, [p3_by_layer[L]["centroid"]["macro_f1"] for L in pl], "^--", color="C2", label="class-mean decoder F1 (P3)")
        ax2.set_ylabel("macro-F1")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        ax1.set_title("decodable vs used, by layer")
        fig.tight_layout()
        fig.savefig(out_dir / "sweep.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}", flush=True)
    print(f"done in {results['seconds']}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
