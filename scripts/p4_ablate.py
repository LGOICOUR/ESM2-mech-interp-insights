"""P4: is the decodable direction load-bearing for the model's own predictions?

For each held-out fold, directions are fit on the training chains at each in-scope
layer, then projected out of the residual stream during the forward pass on the
held-out chains (residue positions only; see src/esm2_probe/ablate.py). Read-outs:
  * the model's own output: masked pseudo-log-likelihood of the true residue and
    its top-1 prediction at every labelled residue (k-pass masking), compared with
    the unablated run, split by true class (helix / strand / coil);
  * decodability afterwards: the fold's original probe applied to the ablated
    activations, and a probe refit on the ablated activations, at the ablated
    layer and at the last layer.

Directions: probe_H / probe_E (the logistic probe's class weight vector),
dm_H / dm_E (class mean minus the rest), probe_sub / dm_sub (the rank-2 span of
all three class directions), random (one unit direction) and random_sub (two),
the controls. Scopes: single = the headline layer L*; band = L* +/- 3; mid = layers 12-18.

Outputs in results/p4/<recording>/: metrics.json, summary.md, per_residue.npz

Usage:
    uv run python scripts/p4_ablate.py
    uv run python scripts/p4_ablate.py --limit 12 --kinds probe_H random --scopes single   # quick check
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.ablate import AblationContext, attach, orthonormal_rows, random_unit  # noqa: E402
from esm2_probe.data import CLASSES, read_jsonl  # noqa: E402
from esm2_probe.probe import (cv_probe, diff_in_means, fit_lr, grouped_bootstrap_ci, grouped_bootstrap_mean,  # noqa: E402
                              grouped_folds, metrics, probe_direction, subsample)
from esm2_probe.reader import Recorder, encode, load_model, masked_readout  # noqa: E402

KINDS = ["probe_H", "probe_E", "dm_H", "dm_E", "probe_sub", "dm_sub", "random", "random_sub"]


def scopes_for(L_star: int, n_layers: int) -> dict[str, list[int]]:
    return {"single": [L_star],
            "band": [L for L in range(L_star - 3, L_star + 4) if 1 <= L <= n_layers],
            "mid": [L for L in range(12, 19) if 1 <= L <= n_layers],
            "wide": [L for L in range(L_star - 3, n_layers + 1) if L >= 1]}


def fit_directions(acts, y, g, folds, layers, n_train, seed, C_of):
    """Per fold and layer: unit directions of every kind, the training mean, and the
    fold's probe (scaler, classifier) for the original-probe check."""
    cache, probes = {}, {}
    for f in range(int(folds.max()) + 1):
        cache[f], probes[f] = {}, {}
        tr = np.where(folds != f)[0]
        fit_idx = subsample(tr, n_train, seed + 100 + f)  # same residues as P3's outer fit
        for L in layers:
            X = np.asarray(acts[L][fit_idx], dtype=np.float32)
            sc, clf = fit_lr(X, y[fit_idx], C_of(L, f))
            probes[f][L] = (sc, clf)
            pw = np.stack([probe_direction(sc, clf, c) for c in range(3)])
            dm, mu = diff_in_means(X, y[fit_idx])
            W = X.shape[1]
            # the three class directions of either kind sum to zero (multinomial weights; weighted
            # class-mean deviations), so their span is the plane of the H and E directions
            cache[f][L] = {"mu": mu, "probe_H": pw[0], "probe_E": pw[1], "dm_H": dm[0], "dm_E": dm[1],
                           "probe_sub": orthonormal_rows(pw[:2]), "dm_sub": orthonormal_rows(dm[:2]),
                           "random": random_unit(W, seed=1000 * f + L),
                           "random_sub": orthonormal_rows(np.stack([random_unit(W, 7000 + 1000 * f + L), random_unit(W, 8000 + 1000 * f + L)]))}
    return cache, probes


def by_class(values, y, g, n_boot, seed=0):
    out = {}
    for c, name in enumerate(CLASSES):
        m = y == c
        out[name] = list(grouped_bootstrap_mean(values[m], g[m], n_boot=n_boot, seed=seed))
    out["all"] = list(grouped_bootstrap_mean(values, g, n_boot=n_boot, seed=seed))
    return out


def probe_block(y, yhat, g, n_boot):
    m = metrics(y, yhat)
    ci = grouped_bootstrap_ci(y, yhat, g, n_boot=n_boot)
    return {**m, "accuracy_ci": list(ci["accuracy"]), "macro_f1_ci": list(ci["macro_f1"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--scopes", nargs="+", default=["single", "band", "mid"], choices=["single", "band", "mid", "wide"])
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--k", type=int, default=8, help="masking passes per chain (1/k of residues masked per pass)")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="first N chains only (check runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    n_layers, _, width = acts.shape[0] - 1, acts.shape[1], acts.shape[2]
    tbl = np.load(src / "residues.npz")
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L_star = min(L for L, c in counts.items() if c == max(counts.values()))
    scopes = {s: scopes_for(L_star, n_layers)[s] for s in args.scopes}
    record_layers = sorted({L_star, n_layers})
    p3_C = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}

    def C_of(L, f):
        return p3_C[L][f] if L in p3_C else 1e-2

    # residue bookkeeping (the full table defines folds and directions; --limit only restricts the chains read)
    y_all, g_all = tbl["label"].astype(int), tbl["protein_idx"]
    folds_all = grouped_folds(g_all, 5, args.seed)
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g_all, minlength=len(proteins)), out=starts[1:])
    chain_ids = list(range(len(proteins)))[: args.limit] if args.limit else list(range(len(proteins)))
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chain_ids])
    y, g, folds = y_all[rows], g_all[rows], folds_all[rows]
    lab_pos = {i: tbl["position"][starts[i]:starts[i + 1]] for i in chain_ids}
    out_dir = ROOT / "results" / "p4" / (args.recording + (f"_limit{args.limit}" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache_layers = sorted(set(L for s in scopes.values() for L in s) | set(record_layers))
    cache, probes = fit_directions(acts, y_all, g_all, folds_all, cache_layers, args.n_train, args.seed, C_of)
    ranks = {k: cache[0][L_star][k].shape[0] for k in ("probe_sub", "dm_sub", "random_sub")}
    cos = {f"cos(probe_{c}, dm_{c}) at L*": float(cache[0][L_star][f"probe_{c}"] @ cache[0][L_star][f"dm_{c}"]) for c in ("H", "E")}
    print(f"L* = {L_star}; scopes {scopes}; subspace ranks {ranks}; {cos}; directions fit in {time.time() - t0:.0f}s", flush=True)

    tok, model, device = load_model()
    ctx = AblationContext()
    true_tok = np.zeros(len(y), dtype=np.int64)

    def run_condition(dirs_by_fold, record):
        """dirs_by_fold[f] = {layer: (U, mu)} or None for the unablated run. Returns
        logp [n], top1 [n], and recorded activations {layer: [n, W] float16}."""
        logp, top1 = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
        A = {L: np.zeros((len(y), width), dtype=np.float16) for L in record}
        for f in range(5):
            handles = attach(model, dirs_by_fold[f], ctx, device) if dirs_by_fold is not None else []
            try:
                for i in chain_ids:
                    if folds_all[starts[i]] != f:
                        continue
                    seq = proteins[i].sequence
                    sel = np.where(g == i)[0]
                    r = masked_readout(tok, model, device, seq, args.k, ctx=ctx)
                    logp[sel], top1[sel] = r.logp_true[lab_pos[i]], r.top1[lab_pos[i]]
                    batch = encode(tok, [seq], device)
                    ctx.residue_mask = batch.residue_mask
                    rec = Recorder(model, record)
                    try:
                        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                    finally:
                        rec.remove()
                    for L in record:
                        A[L][sel] = rec.out[L][0, 1:len(seq) + 1][lab_pos[i]].to(torch.float16).cpu().numpy()
                    if dirs_by_fold is None:
                        true_tok[sel] = batch.input_ids[0, 1:len(seq) + 1][lab_pos[i]].cpu().numpy()
            finally:
                for h in handles:
                    h.remove()
        return logp, top1, A

    # --- baseline (no ablation) ---
    logp0, top10, A0 = run_condition(None, record_layers)
    base_match = float(np.abs(A0[n_layers].astype(np.float32) - np.asarray(acts[n_layers][rows], dtype=np.float32)).max())
    print(f"baseline read in {time.time() - t0:.0f}s; max |P4 unablated - P2 recording| at layer {n_layers} = {base_match:.3f} "
          f"(float16 round-off expected)", flush=True)
    assert base_match < 1.0, "P4's unablated activations do not match the P2 recording"

    def decodability(A):
        out = {}
        for L in record_layers:
            X = A[L].astype(np.float32)
            yhat = np.zeros(len(y), dtype=int)
            for f in range(5):
                sc, clf = probes[f][L]
                te = folds == f
                if te.any():
                    yhat[te] = clf.predict(sc.transform(X[te]))
            C_fix = collections.Counter(p3_C[L]).most_common(1)[0][0] if L in p3_C else 1e-2
            refit = cv_probe(X, y, g, folds, n_train=args.n_train, seed=args.seed, C=C_fix)["yhat"]
            out[str(L)] = {"original_probe": probe_block(y, yhat, g, args.n_boot), "refit_probe": probe_block(y, refit, g, args.n_boot)}
        return out

    results = {"recording": args.recording, "L_star": L_star, "scopes": scopes, "record_layers": record_layers, "k": args.k,
               "n_chains": len(chain_ids), "n_residues": int(len(y)), "subspace_ranks": ranks, "cosines_at_L_star": cos,
               "baseline": {"pll_by_class": by_class(logp0, y, g, args.n_boot),
                            "top1_acc_by_class": by_class((top10 == true_tok).astype(float), y, g, args.n_boot),
                            "decodability": decodability(A0)},
               "conditions": {}}
    per_res = {"y": y, "g": g, "logp_base": logp0, "top1_base": top10, "true_tok": true_tok}
    print(f"baseline PLL {results['baseline']['pll_by_class']['all'][0]:.3f}, top-1 acc {results['baseline']['top1_acc_by_class']['all'][0]:.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)

    # --- ablations ---
    for scope, layers in scopes.items():
        for kind in args.kinds:
            dirs_by_fold = {f: {L: (cache[f][L][kind], cache[f][L]["mu"]) for L in layers} for f in range(5)}
            logp, top1, A = run_condition(dirs_by_fold, record_layers)
            # projection check: at the ablated layer, the removed coordinate must be ~0 on held-out residues
            chk = []
            for f in range(5):
                te = folds == f
                if te.any() and L_star in layers:
                    U = cache[f][L_star][kind]
                    U2 = U[None] if U.ndim == 1 else U
                    coord = (A[L_star][te].astype(np.float32) - cache[f][L_star]["mu"]) @ U2.T
                    chk.append(float(np.abs(coord).max()))
            name = f"{scope}/{kind}"
            results["conditions"][name] = {
                "scope": scope, "kind": kind, "layers": layers,
                "delta_pll_by_class": by_class(logp - logp0, y, g, args.n_boot),
                "top1_changed_by_class": by_class((top1 != top10).astype(float), y, g, args.n_boot),
                "top1_acc_by_class": by_class((top1 == true_tok).astype(float), y, g, args.n_boot),
                "max_abs_removed_coordinate_at_L_star": max(chk) if chk else None,
                "decodability": decodability(A)}
            per_res[f"logp_{scope}_{kind}"], per_res[f"top1_{scope}_{kind}"] = logp, top1
            d = results["conditions"][name]["delta_pll_by_class"]
            print(f"{name}: dPLL H {d['H'][0]:+.3f} E {d['E'][0]:+.3f} C {d['C'][0]:+.3f}; "
                  f"refit F1 at L* {results['conditions'][name]['decodability'][str(L_star)]['refit_probe']['macro_f1']:.3f}; "
                  f"{time.time() - t0:.0f}s", flush=True)
            if device == "mps":
                torch.mps.empty_cache()

    results["seconds"] = time.time() - t0
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez(out_dir / "per_residue.npz", **per_res)

    # --- summary table ---
    def pc(block, key="delta_pll_by_class"):
        return " | ".join(f"{block[key][c][0]:+.3f} [{block[key][c][1]:+.3f}, {block[key][c][2]:+.3f}]" for c in CLASSES)

    def tc(block):
        return " / ".join(f"{100 * block['top1_changed_by_class'][c][0]:.1f}%" for c in CLASSES)

    def dec(block, L):
        o, r = block["decodability"][str(L)]["original_probe"], block["decodability"][str(L)]["refit_probe"]
        return (f"{o['macro_f1']:.2f} (H {o['f1_per_class'][0]:.2f}, E {o['f1_per_class'][1]:.2f}) | "
                f"{r['macro_f1']:.2f} [{r['macro_f1_ci'][0]:.2f}, {r['macro_f1_ci'][1]:.2f}] (H {r['f1_per_class'][0]:.2f}, E {r['f1_per_class'][1]:.2f})")

    b = results["baseline"]
    lines = [f"# P4 ablation, `{args.recording}`: L* = {L_star}, {len(chain_ids)} held-out chains, {len(y)} labelled residues", "",
             f"Baseline (no ablation): masked PLL per residue {b['pll_by_class']['all'][0]:.3f} nats "
             f"(H {b['pll_by_class']['H'][0]:.3f}, E {b['pll_by_class']['E'][0]:.3f}, C {b['pll_by_class']['C'][0]:.3f}); "
             f"masked top-1 accuracy {100 * b['top1_acc_by_class']['all'][0]:.1f}%. k = {args.k} masking passes per chain.", "",
             "## Effect on the model's own output", "",
             "Change in masked log-likelihood of the true residue, mean over residues of each class [95% CI, chains resampled]; "
             "top-1 changed = share of residues whose top-1 prediction differs from the unablated run.", "",
             "| condition | layers | ΔPLL helix | ΔPLL strand | ΔPLL coil | top-1 changed H / E / C |", "|---|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        lay = f"{r['layers'][0]}" if len(r["layers"]) == 1 else f"{r['layers'][0]}–{r['layers'][-1]}"
        lines.append(f"| {name} | {lay} | {pc(r)} | {tc(r)} |")
    lines += ["", "## Is the concept gone? Decodability after ablation", "",
              "Original probe = the fold's unablated probe applied to the ablated activations; refit = a new probe trained on "
              "the ablated activations (same chain-grouped folds). Macro-F1 (H, E = per-class F1).", "",
              f"| condition | layer {L_star}: original \\| refit | layer {n_layers}: original \\| refit |", "|---|---|---|",
              f"| baseline, no ablation | {dec(b, L_star)} | {dec(b, n_layers)} |"]
    for name, r in results["conditions"].items():
        lines.append(f"| {name} | {dec(r, L_star)} | {dec(r, n_layers)} |")
    lines += ["", f"- subspace ranks: {ranks}; {cos}",
              f"- projection check: largest |removed coordinate| left at L* across conditions = "
              f"{max(v['max_abs_removed_coordinate_at_L_star'] or 0 for v in results['conditions'].values()):.4f}",
              f"- {results['seconds']:.0f}s on {device}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
