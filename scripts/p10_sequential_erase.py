"""P10: sequential multi-layer erasure of the class-mean plane.

P4's band condition removed the plane at layers 26-30 at once, every plane fit on
the *unedited* recording. After the planes at 26-29 are removed, the class means
at layer 30 move, so the pre-fit plane at 30 is stale and misses them: the refit
reader at layer 30 still read 0.70 in P4. Here the planes are fit in order, each
on activations already edited by the planes below it:

  fit plane at L1 on the recording -> forward pass on the training chains with
  the L1 hook, record L2 -> fit plane at L2 on those rows -> pass with hooks at
  L1, L2, record L3 -> ... up to the top of the band.

Per fold, planes are fit on the same 40k-row subsample as P3/P4. Conditions per
band: sequential (this), stale (P4's way, all planes fit on the recording) and a
random rank-2 plane at every layer of the band. Read-outs: the model's own
masked output (ΔPLL by class, top-1) and decodability at the last layer after the
edit (linear refit and QDA refit, chain-grouped cv), to check that the plane is
gone everywhere it was removed.

Outputs in results/p10/<recording>[_limitN][_<tag>]/: metrics.json, summary.md, per_residue.npz

Usage:
    uv run python scripts/p10_sequential_erase.py --bands 29-30 26-30 20-30
    uv run python scripts/p10_sequential_erase.py --limit 12 --bands 29-30 --n-boot 20 --tag check
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.ablate import AblationContext, attach, orthonormal_rows, random_unit  # noqa: E402
from esm2_probe.data import CLASSES, read_jsonl  # noqa: E402
from esm2_probe.probe import (cv_probe, diff_in_means, grouped_bootstrap_ci, grouped_bootstrap_mean,  # noqa: E402
                              grouped_folds, metrics, subsample)
from esm2_probe.reader import Recorder, encode, load_model, masked_readout  # noqa: E402


def parse_band(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def by_class(values, y, g, n_boot, seed=0):
    out = {name: list(grouped_bootstrap_mean(values[y == c], g[y == c], n_boot=n_boot, seed=seed))
           for c, name in enumerate(CLASSES)}
    out["all"] = list(grouped_bootstrap_mean(values, g, n_boot=n_boot, seed=seed))
    return out


def probe_block(y, yhat, g, n_boot):
    m = metrics(y, yhat)
    ci = grouped_bootstrap_ci(y, yhat, g, n_boot=n_boot)
    return {**m, "accuracy_ci": list(ci["accuracy"]), "macro_f1_ci": list(ci["macro_f1"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--bands", nargs="+", default=["29-30", "26-30", "20-30"], help="inclusive layer ranges, e.g. 26-30")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    n_layers, n_rows_all, width = acts.shape[0] - 1, acts.shape[1], acts.shape[2]
    tbl = np.load(src / "residues.npz")
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L_star = min(L for L, c in counts.items() if c == max(counts.values()))
    p3_C = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}
    C_fix = collections.Counter(p3_C[n_layers]).most_common(1)[0][0]
    bands = {b: parse_band(b) for b in args.bands}
    assert all(1 <= L <= n_layers for band in bands.values() for L in band)

    y_all, g_all = tbl["label"].astype(int), tbl["protein_idx"]
    folds_all = grouped_folds(g_all, 5, args.seed)
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g_all, minlength=len(proteins)), out=starts[1:])
    all_ids = list(range(len(proteins)))
    chain_ids = all_ids[: args.limit] if args.limit else all_ids
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chain_ids])
    y, g, folds = y_all[rows], g_all[rows], folds_all[rows]
    lab_pos_all = {i: tbl["position"][starts[i]:starts[i + 1]] for i in all_ids}
    fold_of_chain = {i: int(folds_all[starts[i]]) for i in all_ids}
    out_dir = ROOT / "results" / "p10" / (args.recording + (f"_limit{args.limit}" if args.limit else "")
                                          + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tok, model, device = load_model()
    ctx = AblationContext()

    @torch.no_grad()
    def record_layer(ids, hooks_by_layer, L):
        """One unmasked pass per chain in `ids` with the given hooks attached; returns the
        residual at layer L for every labelled row of those chains as {chain: [n_i, W] f32}."""
        handles = attach(model, hooks_by_layer, ctx, device) if hooks_by_layer else []
        out = {}
        try:
            for i in ids:
                seq = proteins[i].sequence
                batch = encode(tok, [seq], device)
                ctx.residue_mask = batch.residue_mask
                rec = Recorder(model, [L])
                try:
                    model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                finally:
                    rec.remove()
                out[i] = rec.out[L][0, 1:len(seq) + 1][lab_pos_all[i]].float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        return out

    def plane(X, yy):
        dm, mu = diff_in_means(X, yy)
        return orthonormal_rows(dm[:2]), mu

    def fit_band(band, mode):
        """Per fold: {layer: (U, mu)} for the whole band. mode: sequential | stale | random."""
        dirs = {}
        for f in range(5):
            tr = np.where(folds_all != f)[0]
            fit_idx = subsample(tr, args.n_train, args.seed + 100 + f)
            dirs[f] = {}
            if mode == "random":
                for L in band:
                    U = orthonormal_rows(np.stack([random_unit(width, 7000 + 1000 * f + L), random_unit(width, 8000 + 1000 * f + L)]))
                    dirs[f][L] = (U, np.asarray(acts[L][fit_idx], dtype=np.float32).mean(0))
                continue
            if mode == "stale":
                for L in band:
                    dirs[f][L] = plane(np.asarray(acts[L][fit_idx], dtype=np.float32), y_all[fit_idx])
                continue
            # sequential: the first plane from the recording, each next one from edited activations
            dirs[f][band[0]] = plane(np.asarray(acts[band[0]][fit_idx], dtype=np.float32), y_all[fit_idx])
            train_chains = [i for i in all_ids if fold_of_chain[i] != f]
            for L in band[1:]:
                rec = record_layer(train_chains, dirs[f], L)
                X_all = np.zeros((n_rows_all, width), dtype=np.float32)
                for i in train_chains:
                    X_all[starts[i]:starts[i + 1]] = rec[i]
                dirs[f][L] = plane(X_all[fit_idx], y_all[fit_idx])
                # how far the stale plane at L is from the sequential one: principal angles' cosines
                U_stale, _ = plane(np.asarray(acts[L][fit_idx], dtype=np.float32), y_all[fit_idx])
                s = np.linalg.svd(dirs[f][L][0] @ U_stale.T, compute_uv=False)
                print(f"  band {band[0]}-{band[-1]} fold {f} layer {L}: plane fit on edited rows; "
                      f"cos principal angles vs stale plane {np.round(s, 3).tolist()}; {time.time() - t0:.0f}s", flush=True)
        return dirs

    def run_condition(dirs_by_fold):
        logp, top1 = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
        A = np.zeros((len(y), width), dtype=np.float16)
        for f in range(5):
            handles = attach(model, dirs_by_fold[f], ctx, device) if dirs_by_fold is not None else []
            try:
                for i in chain_ids:
                    if fold_of_chain[i] != f:
                        continue
                    seq = proteins[i].sequence
                    sel = np.where(g == i)[0]
                    r = masked_readout(tok, model, device, seq, args.k, ctx=ctx)
                    logp[sel], top1[sel] = r.logp_true[lab_pos_all[i]], r.top1[lab_pos_all[i]]
                    batch = encode(tok, [seq], device)
                    ctx.residue_mask = batch.residue_mask
                    rec = Recorder(model, [n_layers])
                    try:
                        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                    finally:
                        rec.remove()
                    A[sel] = rec.out[n_layers][0, 1:len(seq) + 1][lab_pos_all[i]].to(torch.float16).cpu().numpy()
            finally:
                for h in handles:
                    h.remove()
        return logp, top1, A

    def qda_cv(X):
        yhat = np.zeros(len(y), dtype=int)
        for f in range(5):
            tr, te = folds != f, folds == f
            if te.any():
                if min(np.bincount(y[tr], minlength=3)) <= X.shape[1]:
                    yhat[te] = np.bincount(y[tr]).argmax()
                    continue
                yhat[te] = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(X[tr], y[tr]).predict(X[te])
        return yhat

    def decodability(A):
        X = A.astype(np.float32)
        refit = cv_probe(X, y, g, folds, n_train=args.n_train, seed=args.seed, C=C_fix)["yhat"]
        return {"refit_probe": probe_block(y, refit, g, args.n_boot), "qda": probe_block(y, qda_cv(X), g, args.n_boot)}

    true_tok = np.concatenate([np.asarray(tok(proteins[i].sequence)["input_ids"])[1:-1][lab_pos_all[i]] for i in chain_ids])
    logp0, top10, A0 = run_condition(None)
    results = {"recording": args.recording, "L_star": L_star, "bands": bands, "k": args.k, "n_chains": len(chain_ids),
               "n_residues": int(len(y)), "C_fix_last_layer": C_fix,
               "baseline": {"top1": float(np.mean(top10 == true_tok)), "decodability": decodability(A0)}, "conditions": {}}
    per_residue = {"y": y, "g": g, "logp_none": logp0, "true_tok": true_tok}
    print(f"baseline read in {time.time() - t0:.0f}s; top-1 {results['baseline']['top1']:.3f}; "
          f"layer {n_layers} refit F1 {results['baseline']['decodability']['refit_probe']['macro_f1']:.3f}", flush=True)

    for b, band in bands.items():
        for mode in ("sequential", "stale", "random"):
            name = f"{mode}@{b}"
            dirs = fit_band(band, mode)
            logp, top1, A = run_condition(dirs)
            d = logp - logp0
            results["conditions"][name] = {"band": band, "mode": mode, "delta_pll": by_class(d, y, g, args.n_boot),
                                           "top1": float(np.mean(top1 == true_tok)), "decodability": decodability(A)}
            per_residue[f"logp_{name}"] = logp
            r = results["conditions"][name]
            print(f"{name:>18}: dPLL {r['delta_pll']['all'][0]:+.4f} [H {r['delta_pll']['H'][0]:+.3f} E {r['delta_pll']['E'][0]:+.3f} "
                  f"C {r['delta_pll']['C'][0]:+.3f}] top-1 {r['top1']:.3f}; layer {n_layers} refit F1 "
                  f"{r['decodability']['refit_probe']['macro_f1']:.3f}, QDA {r['decodability']['qda']['macro_f1']:.3f}; {time.time() - t0:.0f}s", flush=True)
            (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))

    results["seconds"] = round(time.time() - t0)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out_dir / "per_residue.npz", **per_residue)

    def f3(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"

    def dec(d):
        return f"{d['refit_probe']['macro_f1']:.3f} [{d['refit_probe']['macro_f1_ci'][0]:.3f}, {d['refit_probe']['macro_f1_ci'][1]:.3f}]"

    b0 = results["baseline"]
    lines = [f"# P10 sequential multi-layer erasure, `{args.recording}`: {len(chain_ids)} held-out chains, {len(y)} labelled residues; L* = {L_star}", "",
             "Class-mean plane (rank 2) removed at every layer of a band. `sequential`: each plane fit on activations already "
             "edited by the planes below it. `stale`: every plane fit on the unedited recording (P4's band). `random`: a random "
             "rank-2 plane at each layer. ΔPLL in nats, mean [95% CI, chains resampled]; decodability = readers refit on the "
             f"edited layer-{n_layers} rows (chain-grouped cv), macro-F1.", "",
             f"Baseline top-1 {b0['top1']:.1%}; layer {n_layers} refit linear {dec(b0['decodability'])}, QDA {b0['decodability']['qda']['macro_f1']:.3f}.", "",
             f"| condition | layers | ΔPLL all | helix | strand | coil | top-1 | layer {n_layers} linear refit | layer {n_layers} QDA |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        lines.append(f"| {name} | {r['band'][0]}–{r['band'][-1]} | {f3(r['delta_pll']['all'])} | {r['delta_pll']['H'][0]:+.3f} | "
                     f"{r['delta_pll']['E'][0]:+.3f} | {r['delta_pll']['C'][0]:+.3f} | {r['top1']:.1%} | {dec(r['decodability'])} | "
                     f"{r['decodability']['qda']['macro_f1']:.3f} |")
    lines += ["", f"- {results['seconds']}s on {device}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"done in {results['seconds']}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
