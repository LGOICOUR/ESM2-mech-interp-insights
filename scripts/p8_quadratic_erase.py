"""P8: does the model use the shape? Quadratic erasure of secondary structure at L*.

P6 showed that after the class-mean plane is projected out at L* the three class
clouds share a centre but keep different covariances, and QDA reads the class from
the covariances alone at the unablated ceiling. P4 showed the model pays about
0.06 nats for the plane. Here the shape is erased too, during the forward pass, and
the model's own output is read out again.

The edit at L*, per residue row h with class c (DSSP label):
  1. project out the class-mean plane (P4's dm_sub), as before;
  2. h <- mu + T_c (h - mu_c), where mu_c is class c's mean after step 1 and T_c is
     the symmetric optimal-transport map between Gaussians (Bures map) that sends
     class c's covariance to the pooled within-class covariance:
         T_c = S_c^{-1/2} (S_c^{1/2} S_pool S_c^{1/2})^{1/2} S_c^{-1/2},
     so after the edit every class has mean mu and covariance S_pool on the
     training rows: no quadratic reader can beat chance there. This is the map of
     the closed-form quadratic erasure of Quirke and Belrose (QLEACE, arXiv
     2502.02820), whose target is the Wasserstein barycenter of the class Gaussians
     where this one is S_pool. It is applied here as a forward-pass intervention.
     The edit needs the row's label; residue
     positions without a DSSP label get step 1 only.
Control: the same maps T_c applied to rows with a shuffled class assignment (drawn
per row from the class frequencies), so the edits are of the same kind and size but
remove no class-specific shape.

Read-outs per condition: masked PLL and top-1 of the true residue vs the unablated
run; mean displacement of the edited rows at L*; decodability at L* and the last
layer by a refit linear probe and a refit QDA (chain-grouped folds, same as P6).

Cross-validation caveat. With per-fold maps, a held-out chain's rows are mapped by a map
fit on the other folds, and every training row of a refit reader was mapped by a map whose
fit set contained the held-out fold. Second-order sampling noise is then anti-correlated
between a fold's test rows and its training rows, and a cross-validated QDA scores below
the floor (0.03 in the per-fold run). `--disjoint-fit` removes this: one erasure (and one
reference probe) is fit on the chains of folds 0-2 and applied to the chains of folds 3-4,
and the refit readers are cross-validated over folds 3-4 only. That is the clean
decodability number; the model-output read-outs do not depend on the choice.

Outputs in results/p8/<recording>[_limitN][_disjoint][_<tag>]/: metrics.json, summary.md, per_residue.npz

Usage:
    .venv/bin/python scripts/p8_quadratic_erase.py
    .venv/bin/python scripts/p8_quadratic_erase.py --limit 12 --n-boot 30 --tag smoke
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
from esm2_probe.ablate import AblationContext, orthonormal_rows, project_out  # noqa: E402
from esm2_probe.data import CLASSES, read_jsonl  # noqa: E402
from esm2_probe.probe import (cv_probe, diff_in_means, fit_lr, grouped_bootstrap_ci, grouped_bootstrap_mean,  # noqa: E402
                              grouped_folds, metrics, subsample)
from esm2_probe.reader import Recorder, encode, layer_modules, load_model, masked_readout  # noqa: E402


def _eigh_fn(S, fn):
    w, V = np.linalg.eigh(S)
    return (V * fn(np.clip(w, 0, None))) @ V.T


def bures_map(S_c: np.ndarray, S_pool: np.ndarray) -> np.ndarray:
    """Symmetric PSD T with T S_c T = S_pool (the optimal-transport map between the
    two zero-mean Gaussians: the smallest mean-squared displacement that does it)."""
    S_c_h = _eigh_fn(S_c, np.sqrt)
    S_c_ih = _eigh_fn(S_c, lambda w: 1.0 / np.sqrt(w))
    M = _eigh_fn(S_c_h @ S_pool @ S_c_h, np.sqrt)
    T = S_c_ih @ M @ S_c_ih
    err = np.linalg.norm(T @ S_c @ T - S_pool) / np.linalg.norm(S_pool)
    assert err < 1e-3, f"Bures map does not match the pooled covariance: rel err {err:.2e}"
    return T


def fit_erasure(acts, y, folds, L, n_train, seed, ridge_rel=1e-3, fit_rows=None):
    """Per fold: the class-mean plane and reference mean (as P4), then the class means
    and Bures maps fit on the same training rows after the plane is removed. With
    `fit_rows`, one erasure is fit on those rows and used for every fold."""
    out = {}
    for f in range(int(folds.max()) + 1):
        if fit_rows is not None and f > 0:
            out[f] = out[0]
            continue
        tr = fit_rows if fit_rows is not None else np.where(folds != f)[0]
        fit_idx = subsample(tr, n_train, seed + 100 + f)
        X = np.asarray(acts[L][fit_idx], dtype=np.float32)
        yy = y[fit_idx]
        dm, mu = diff_in_means(X, yy)
        U = orthonormal_rows(dm[:2])
        X1 = (X - ((X - mu) @ U.T) @ U).astype(np.float64)
        W = X.shape[1]
        mus, covs, ns = [], [], []
        for c in range(3):
            Z = X1[yy == c]
            mus.append(Z.mean(0))
            covs.append(np.cov(Z.T, bias=True))
            ns.append(len(Z))
        S_pool = sum(n * S for n, S in zip(ns, covs)) / sum(ns)
        ridge = ridge_rel * np.trace(S_pool) / W * np.eye(W)
        S_pool = S_pool + ridge
        maps = [bures_map(S + ridge, S_pool) for S in covs]
        # size of the edit on the training rows, per class: mean |T_c (x - mu_c) + mu - x|
        disp = [float(np.linalg.norm((X1[yy == c] - mus[c]) @ maps[c] + mu - X1[yy == c], axis=1).mean()) for c in range(3)]
        out[f] = {"mu": mu, "U": U, "mu_c": np.stack(mus).astype(np.float32), "T": np.stack(maps).astype(np.float32),
                  "train_disp_by_class": disp, "train_radius": float(np.linalg.norm(X1 - mu, axis=1).mean())}
    return out


class QuadContext(AblationContext):
    def __init__(self) -> None:
        super().__init__()
        self.cls: torch.Tensor | None = None  # [T] long, class per position or -1


def make_quad_hook(er, ctx: QuadContext, device, quadratic: bool):
    U = torch.tensor(er["U"], device=device)
    mu = torch.tensor(er["mu"], device=device)
    mu_c = torch.tensor(er["mu_c"], device=device)
    T = torch.tensor(er["T"], device=device)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = project_out(h, U, mu, ctx.residue_mask)
        if quadratic:
            for c in range(3):
                m = ctx.residue_mask & (ctx.cls[None, :] == c)
                if m.any():
                    x = h[m].float()
                    h[m] = ((x - mu_c[c]) @ T[c] + mu).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--disjoint-fit", action="store_true",
                    help="fit the erasure and the reference probe once on the chains of folds 0-2; evaluate on folds 3-4 (2-fold cv)")
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    n_layers, width = acts.shape[0] - 1, acts.shape[2]
    tbl = np.load(src / "residues.npz")
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L_star = min(L for L, c in counts.items() if c == max(counts.values()))
    record = sorted({L_star, n_layers})
    p3_C = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}
    C_fix = {L: (collections.Counter(p3_C[L]).most_common(1)[0][0] if L in p3_C else 1e-2) for L in record}

    y_all, g_all = tbl["label"].astype(int), tbl["protein_idx"]
    folds_all = grouped_folds(g_all, 5, args.seed)
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g_all, minlength=len(proteins)), out=starts[1:])
    chain_ids = list(range(len(proteins)))[: args.limit] if args.limit else list(range(len(proteins)))
    fit_rows = None
    if args.disjoint_fit:
        fit_rows = np.where(folds_all <= 2)[0]
        chain_ids = [i for i in chain_ids if folds_all[starts[i]] >= 3]
        cv_folds_all = np.where(folds_all >= 3, folds_all - 3, -1)  # folds 3, 4 -> cv folds 0, 1; fit chains excluded
    else:
        cv_folds_all = folds_all
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chain_ids])
    y, g, folds = y_all[rows], g_all[rows], cv_folds_all[rows]
    n_cv = int(folds.max()) + 1
    lab_pos = {i: tbl["position"][starts[i]:starts[i + 1]] for i in chain_ids}
    class_freq = np.bincount(y_all, minlength=3) / len(y_all)
    out_dir = ROOT / "results" / "p8" / (args.recording + (f"_limit{args.limit}" if args.limit else "")
                                         + ("_disjoint" if args.disjoint_fit else "") + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    erase = fit_erasure(acts, y_all, folds_all, L_star, args.n_train, args.seed, fit_rows=fit_rows)
    probes = {}
    for f in range(5):
        if fit_rows is not None and f > 0:
            probes[f] = probes[0]
            continue
        tr = fit_rows if fit_rows is not None else np.where(folds_all != f)[0]
        fit_idx = subsample(tr, args.n_train, args.seed + 100 + f)
        probes[f] = {L: fit_lr(np.asarray(acts[L][fit_idx], dtype=np.float32), y_all[fit_idx],
                               p3_C[L][f] if L in p3_C else 1e-2) for L in record}
    fit_line = "; ".join(f"fold {f}: edit {np.round(e['train_disp_by_class'], 2).tolist()} vs radius {e['train_radius']:.1f}"
                         for f, e in erase.items())
    print(f"L* = {L_star}; erasure maps fit in {time.time() - t0:.0f}s; training-row displacement by class (H, E, C) {fit_line}", flush=True)
    labelled_share = len(y_all) / sum(len(p.sequence) for p in proteins)

    tok, model, device = load_model()
    ctx = QuadContext()
    mods = layer_modules(model)
    true_tok = np.zeros(len(y), dtype=np.int64)

    def run_condition(mode):
        """mode: None (no edit), 'linear' (plane only), 'quad' (plane + shape, true
        labels), 'quad_shuffled' (plane + shape maps with shuffled labels)."""
        logp, top1 = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
        A = {L: np.zeros((len(y), width), dtype=np.float16) for L in record}
        assigned = np.full(len(y), -1, dtype=int)
        for f in range(5):
            if not any(folds_all[starts[i]] == f for i in chain_ids):
                continue
            handle = mods[L_star].register_forward_hook(make_quad_hook(erase[f], ctx, device, mode in ("quad", "quad_shuffled"))) if mode else None
            try:
                for i in chain_ids:
                    if folds_all[starts[i]] != f:
                        continue
                    seq = proteins[i].sequence
                    n = len(seq)
                    sel = np.where(g == i)[0]
                    cls = np.full(n + 2, -1, dtype=np.int64)
                    if mode == "quad":
                        cls[1 + lab_pos[i]] = y[sel]
                    elif mode == "quad_shuffled":
                        cls[1 + lab_pos[i]] = np.random.default_rng(args.seed * 1_000_003 + 31 * i).choice(3, size=len(sel), p=class_freq)
                    assigned[sel] = cls[1 + lab_pos[i]]
                    ctx.cls = torch.tensor(cls, device=device)
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
                        A[L][sel] = rec.out[L][0, 1:n + 1][lab_pos[i]].to(torch.float16).cpu().numpy()
                    if mode is None:
                        true_tok[sel] = batch.input_ids[0, 1:n + 1][lab_pos[i]].cpu().numpy()
            finally:
                if handle is not None:
                    handle.remove()
        return logp, top1, A, assigned

    def by_class(values):
        out = {name: list(grouped_bootstrap_mean(values[y == c], g[y == c], n_boot=args.n_boot)) for c, name in enumerate(CLASSES)}
        out["all"] = list(grouped_bootstrap_mean(values, g, n_boot=args.n_boot))
        return out

    def probe_block(yhat):
        r = metrics(y, yhat)
        ci = grouped_bootstrap_ci(y, yhat, g, n_boot=args.n_boot)
        return {**r, "accuracy_ci": list(ci["accuracy"]), "macro_f1_ci": list(ci["macro_f1"])}

    def qda_cv(X):
        yhat = np.zeros(len(y), dtype=int)
        for f in range(n_cv):
            tr, te = folds != f, folds == f
            if te.any():
                if min(np.bincount(y[tr], minlength=3)) <= X.shape[1]:  # check runs only: too few rows per class
                    yhat[te] = np.bincount(y[tr]).argmax()
                    continue
                q = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(X[tr], y[tr])
                yhat[te] = q.predict(X[te])
        return yhat

    def decodability(A):
        out = {}
        for L in record:
            X = A[L].astype(np.float32)
            orig = np.zeros(len(y), dtype=int)
            for f in range(5):
                sc, clf = probes[f][L]
                te = folds_all[rows] == f
                if te.any():
                    orig[te] = clf.predict(sc.transform(X[te]))
            refit = cv_probe(X, y, g, folds, n_train=args.n_train, seed=args.seed, C=C_fix[L])["yhat"]
            out[str(L)] = {"original_probe": probe_block(orig), "refit_linear": probe_block(refit), "refit_qda": probe_block(qda_cv(X))}
        return out

    conditions = [("none", None), ("linear (plane)", "linear"), ("quadratic (plane + shape)", "quad"),
                  ("quadratic, shuffled labels", "quad_shuffled")]
    results = {"recording": args.recording, "L_star": L_star, "record_layers": record, "k": args.k, "n_chains": len(chain_ids),
               "disjoint_fit": bool(args.disjoint_fit), "cv_folds": n_cv,
               "n_residues": int(len(y)), "labelled_share_of_positions": labelled_share,
               "train_fit": {str(f): {k: v for k, v in e.items() if k in ("train_disp_by_class", "train_radius")} for f, e in erase.items()},
               "conditions": {}}
    per_res = {"y": y, "g": g}
    base = None
    for name, mode in conditions:
        logp, top1, A, assigned = run_condition(mode)
        if base is None:
            base = (logp, top1, A)
            per_res["true_tok"] = true_tok
        d = np.linalg.norm(A[L_star].astype(np.float32) - base[2][L_star].astype(np.float32), axis=1)
        radius = float(np.linalg.norm(base[2][L_star].astype(np.float32) - np.mean([erase[f]["mu"] for f in range(5)], axis=0), axis=1).mean())
        results["conditions"][name] = {
            "mode": mode,
            "delta_pll_by_class": by_class(logp - base[0]),
            "top1_changed_by_class": by_class((top1 != base[1]).astype(float)),
            "top1_acc_by_class": by_class((top1 == true_tok).astype(float)),
            "displacement_at_L_star": {"mean": float(d.mean()), "by_class": [float(d[y == c].mean()) for c in range(3)], "row_radius": radius},
            "decodability": decodability(A)}
        per_res[f"logp_{mode}"], per_res[f"top1_{mode}"], per_res[f"assigned_{mode}"] = logp, top1, assigned
        r = results["conditions"][name]
        print(f"{name}: dPLL {r['delta_pll_by_class']['all'][0]:+.3f}; top-1 acc {100 * r['top1_acc_by_class']['all'][0]:.1f}%; "
              f"displacement {d.mean():.2f} (radius {radius:.1f}); "
              + "; ".join(f"L{L} linear {r['decodability'][str(L)]['refit_linear']['macro_f1']:.3f} QDA {r['decodability'][str(L)]['refit_qda']['macro_f1']:.3f}" for L in record)
              + f"; {time.time() - t0:.0f}s", flush=True)
        if device == "mps":
            torch.mps.empty_cache()

    results["seconds"] = time.time() - t0
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez(out_dir / "per_residue.npz", **per_res)

    b = results["conditions"]["none"]
    lines = [f"# P8 quadratic erasure, `{args.recording}`: L* = {L_star}, {len(chain_ids)} held-out chains, {len(y)} labelled residues"
             + (" (erasure fit once on folds 0-2, evaluated on folds 3-4, 2-fold cv for the refits)" if args.disjoint_fit else ""), "",
             f"Baseline masked PLL {b['delta_pll_by_class']['all'][0]:+.3f} (by construction 0), top-1 {100 * b['top1_acc_by_class']['all'][0]:.1f}%. "
             f"Labelled share of residue positions {100 * labelled_share:.1f}% (unlabelled positions get the plane only).", "",
             "## The model's own output", "",
             "Change in masked log-likelihood of the true residue vs the unablated run (nats, mean [95% CI, chains resampled]); "
             "top-1 accuracy; mean displacement of the labelled rows at L* against the mean distance of a row from the training mean.", "",
             "| condition | ΔPLL all | ΔPLL helix | ΔPLL strand | ΔPLL coil | top-1 acc | displacement / radius |", "|---|---|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        d = r["delta_pll_by_class"]
        cells = " | ".join(f"{d[c][0]:+.3f} [{d[c][1]:+.3f}, {d[c][2]:+.3f}]" for c in ["all"] + list(CLASSES))
        lines.append(f"| {name} | {cells} | {100 * r['top1_acc_by_class']['all'][0]:.1f}% | "
                     f"{r['displacement_at_L_star']['mean']:.2f} / {r['displacement_at_L_star']['row_radius']:.1f} |")
    lines += ["", "## Decodability after the edit", "", "Macro-F1 [95% CI] of readers refit on the edited rows (chain-grouped folds): linear probe and QDA.", "",
              f"| condition | layer {L_star} linear | layer {L_star} QDA | layer {n_layers} linear | layer {n_layers} QDA |", "|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        cells = []
        for L in record:
            for rd in ("refit_linear", "refit_qda"):
                m = r["decodability"][str(L)][rd]
                cells.append(f"{m['macro_f1']:.3f} [{m['macro_f1_ci'][0]:.3f}, {m['macro_f1_ci'][1]:.3f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"- training-row displacement by class (H, E, C) and mean row radius, per fold: {fit_line}",
              f"- {results['seconds']:.0f}s on {device}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
