"""P6: after linear erasure at L*, how much does a NONLINEAR single-row reader recover?

P4 showed that removing the rank-2 class-mean plane at L* leaves a linear refit at
0.22 macro-F1 (floor 0.20) while the model's own next layer rebuilds 0.56. The
missing cell is a nonlinear reader of the same single rows, no neighbours, no
model: does the concept survive linear erasure in a form a curved boundary can read?

Everything here is post hoc on the P2 recording (no model passes): the fold's
directions are fit on its training chains exactly as in P4 (same residues, same
seeds, same random planes), projected out of the held-out chains' rows at L*, and
readers are trained with the same chain-grouped folds on the edited rows.

Readers: linear (multinomial logistic regression, the P4 refit, as a check that the
post-hoc edit reproduces the hook), mlp (one hidden layer of ReLU units), svm (RBF
support vector machine on a residue subsample; the reader d_model used). Kinds:
none (ceiling), dm_sub (class-mean plane), probe_sub (probe plane), random_sub
(random plane, control), dm_H / probe_H (single directions).

Outputs in results/p6/<recording>[_<tag>]/: metrics.json (updated after every
condition), summary.md, log.txt.

Usage:
    uv run python scripts/p6_nonlinear_refit.py
    uv run python scripts/p6_nonlinear_refit.py --quick            # pipeline check on a residue subsample
    uv run python scripts/p6_nonlinear_refit.py --readers mlp --kinds none dm_sub
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.ablate import orthonormal_rows, random_unit  # noqa: E402
from esm2_probe.data import CLASSES  # noqa: E402
from esm2_probe.probe import (diff_in_means, fit_lr, grouped_bootstrap_ci, grouped_folds, inner_select_c, metrics,  # noqa: E402
                              probe_direction, subsample)

KINDS = ["none", "dm_sub", "random_sub", "probe_sub", "dm_H", "probe_H"]
READERS = ["linear", "mlp", "svm", "quad", "qda"]
N_FOLDS = 5


def fit_planes(X: np.ndarray, y: np.ndarray, folds: np.ndarray, L: int, n_train: int, seed: int, C_of_f) -> dict:
    """Per fold: training mean and the directions of every kind, fit on the same
    residues P4 used (subsample(tr, n_train, seed + 100 + f))."""
    out = {}
    W = X.shape[1]
    for f in range(N_FOLDS):
        tr = np.where(folds != f)[0]
        fit_idx = subsample(tr, n_train, seed + 100 + f)
        Xf, yf = X[fit_idx], y[fit_idx]
        sc, clf = fit_lr(Xf, yf, C_of_f(f))
        pw = np.stack([probe_direction(sc, clf, c) for c in range(3)])
        dm, mu = diff_in_means(Xf, yf)
        out[f] = {"mu": mu,
                  "dm_H": dm[0][None], "probe_H": pw[0][None],
                  "dm_sub": orthonormal_rows(dm[:2]), "probe_sub": orthonormal_rows(pw[:2]),
                  "random_sub": orthonormal_rows(np.stack([random_unit(W, 7000 + 1000 * f + L),
                                                           random_unit(W, 8000 + 1000 * f + L)]))}
    return out


def edited_rows(X: np.ndarray, folds: np.ndarray, planes: dict, kind: str) -> tuple[np.ndarray, float]:
    """Project the fold's plane out of the fold's held-out rows (the hook's operation,
    post hoc). Returns the edited copy and the largest leftover coordinate along the
    removed plane on held-out rows (should be ~0)."""
    if kind == "none":
        return X, 0.0
    Xa = X.copy()
    worst = 0.0
    for f in range(N_FOLDS):
        te = folds == f
        U, mu = planes[f][kind], planes[f]["mu"]
        Z = Xa[te] - mu
        Xa[te] = Xa[te] - (Z @ U.T) @ U
        worst = max(worst, float(np.abs((Xa[te] - mu) @ U.T).max()))
    return Xa, worst


def cloud_shapes(X: np.ndarray, y: np.ndarray, folds: np.ndarray, planes: dict, kind: str) -> dict:
    """How the class clouds differ in shape once the plane is gone, on fold 0's held-out
    rows: for each class pair, the directions of largest variance ratio (generalized
    eigenvectors of the two covariances) and how many directions have a ratio beyond 2
    either way; plus the distance between class means (should be ~0 after dm_sub)."""
    te = folds == 0
    Xt, yt = X[te].astype(np.float64), y[te]
    mus = [Xt[yt == c].mean(0) for c in range(3)]
    pooled = np.cov(Xt.T, bias=True)
    ridge = 1e-3 * np.trace(pooled) / X.shape[1] * np.eye(X.shape[1])  # keeps near-empty directions from giving wild ratios
    covs = [np.cov(Xt[yt == c].T, bias=True) + ridge for c in range(3)]
    pooled_sd = float(np.sqrt(np.trace(pooled) / X.shape[1]))
    out = {"mean_distance_over_pooled_sd": {f"{CLASSES[a]}-{CLASSES[b]}": float(np.linalg.norm(mus[a] - mus[b]) / pooled_sd)
                                            for a, b in ((0, 2), (1, 2), (0, 1))}, "variance_ratio": {}}
    from scipy.linalg import eigh
    for a, b in ((0, 2), (1, 2), (0, 1)):
        w = eigh(covs[a], covs[b], eigvals_only=True)  # variance along u in class a / class b, over all directions
        out["variance_ratio"][f"{CLASSES[a]}/{CLASSES[b]}"] = {"max": float(w.max()), "min": float(w.min()),
                                                              "n_dirs_ratio_over_2": int((w > 2).sum()), "n_dirs_ratio_under_half": int((w < 0.5).sum()),
                                                              "median": float(np.median(w))}
    return out


def make_reader(name: str, seed: int, hidden: int):
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(hidden,), early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=10, max_iter=300, random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=2000, random_state=seed)
    if name == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=1e-3)
    raise ValueError(name)


def squares(X: np.ndarray) -> np.ndarray:
    """Coordinates and their squares [n, 2W]: what a flat plane over these can draw is an
    axis-aligned ellipsoid in the original space (second-order reader, no cross terms)."""
    return np.concatenate([X, X * X], axis=1)


def fold_job(name, X, y, folds, f, n_train, seed, C_fix, hidden):
    tr_idx, te_idx = np.where(folds != f)[0], np.where(folds == f)[0]
    fit_idx = subsample(tr_idx, n_train, seed + 100 + f) if n_train else tr_idx
    t0 = time.time()
    if name == "linear":
        sc, clf = fit_lr(X[fit_idx], y[fit_idx], C_fix)
        yhat = clf.predict(sc.transform(X[te_idx]))
        extra = {"C": C_fix}
    elif name == "quad":
        Q = squares(X)
        C, inner = inner_select_c(Q[fit_idx], y[fit_idx], np.zeros(len(fit_idx), dtype=int) + np.arange(len(fit_idx)) // 50, seed=seed + f)
        sc, clf = fit_lr(Q[fit_idx], y[fit_idx], C)
        yhat = clf.predict(sc.transform(Q[te_idx]))
        extra = {"C": C, "inner_f1": inner}
    else:
        sc = StandardScaler().fit(X[fit_idx])
        clf = make_reader(name, seed + f, hidden).fit(sc.transform(X[fit_idx]), y[fit_idx])
        yhat = clf.predict(sc.transform(X[te_idx]))
        extra = ({"n_iter": int(clf.n_iter_), "best_val": float(clf.best_validation_score_)} if name == "mlp"
                 else {"n_support": [int(v) for v in clf.n_support_]} if name == "svm" else {})
    return f, te_idx, yhat, {"n_fit": int(len(fit_idx)), "seconds": round(time.time() - t0, 1), **extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--readers", nargs="+", default=READERS, choices=READERS)
    ap.add_argument("--n-train", type=int, default=40000, help="residues per fold for direction fitting and the linear refit (P4's value)")
    ap.add_argument("--n-train-mlp", type=int, default=0, help="residues per fold for the MLP; 0 = every training residue")
    ap.add_argument("--n-train-svm", type=int, default=15000, help="residues per fold for the RBF-SVM (cost grows with the square)")
    ap.add_argument("--hidden", type=int, default=256, help="MLP hidden units")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=N_FOLDS)
    ap.add_argument("--quick", action="store_true", help="pipeline check: 6000 residues, 2000 per fold, tiny readers")
    ap.add_argument("--shape", action="store_true", help="also report how the class clouds differ in shape after each removal")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    tbl = np.load(src / "residues.npz")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L = min(l for l, c in counts.items() if c == max(counts.values()))
    C_per_fold = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}[L]
    C_fix = collections.Counter(C_per_fold).most_common(1)[0][0]

    y, g = tbl["label"].astype(int), tbl["protein_idx"]
    folds = grouped_folds(g, N_FOLDS, args.seed)
    X = np.asarray(acts[L], dtype=np.float32)
    if args.quick:
        keep = subsample(np.arange(len(y)), 6000, args.seed)
        X, y, g, folds = X[keep], y[keep], g[keep], folds[keep]
        args.n_train, args.n_train_mlp, args.n_train_svm, args.hidden, args.n_boot = 2000, 2000, 2000, 32, 100
    out_dir = ROOT / "results" / "p6" / (args.recording + ("_quick" if args.quick else "") + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    t0 = time.time()
    say(f"L* = {L}; C per fold {C_per_fold} -> C_fix {C_fix}; {len(y)} residues, {len(set(g.tolist()))} chains; "
        f"readers {args.readers}; kinds {args.kinds}; n_train linear {args.n_train} / mlp {args.n_train_mlp or 'all'} / svm {args.n_train_svm}; "
        f"hidden {args.hidden}")
    planes = fit_planes(X, y, folds, L, args.n_train, args.seed, lambda f: C_per_fold[f])
    cos = float(planes[0]["probe_H"][0] @ planes[0]["dm_H"][0])
    say(f"directions fit in {time.time() - t0:.0f}s; cos(probe_H, dm_H) fold 0 = {cos:.3f}")

    results = {"recording": args.recording, "L_star": L, "C_fix": C_fix, "n_residues": int(len(y)), "quick": args.quick,
               "hidden": args.hidden, "n_train": {"linear": args.n_train, "mlp": args.n_train_mlp or int((folds != 0).sum()),
                                                  "svm": args.n_train_svm, "quad": args.n_train, "qda": args.n_train_mlp or int((folds != 0).sum())},
               "cos_probe_H_dm_H_fold0": cos, "conditions": {}}
    n_train_of = {"linear": args.n_train, "mlp": args.n_train_mlp, "svm": args.n_train_svm, "quad": args.n_train, "qda": args.n_train_mlp}

    for kind in args.kinds:
        Xa, leftover = edited_rows(X, folds, planes, kind)
        results["conditions"][kind] = {"max_leftover_coordinate": leftover, "readers": {}}
        say(f"[{kind}] rows edited; max leftover coordinate along the removed plane on held-out rows = {leftover:.2e}")
        if args.shape:
            sh = cloud_shapes(Xa, y, folds, planes, kind)
            results["conditions"][kind]["shape_fold0"] = sh
            say(f"[{kind}] shape (fold 0 held-out): mean distances / pooled sd {sh['mean_distance_over_pooled_sd']}; "
                f"variance ratios {sh['variance_ratio']}")
        for name in args.readers:
            t1 = time.time()
            jobs = Parallel(n_jobs=args.jobs)(delayed(fold_job)(name, Xa, y, folds, f, n_train_of[name], args.seed, C_fix, args.hidden)
                                              for f in range(N_FOLDS))
            yhat = np.zeros(len(y), dtype=int)
            per_fold = {}
            for f, te_idx, yh, info in jobs:
                yhat[te_idx] = yh
                per_fold[str(f)] = info
            m = metrics(y, yhat)
            ci = grouped_bootstrap_ci(y, yhat, g, n_boot=args.n_boot, seed=args.seed)
            block = {**m, "macro_f1_ci": list(ci["macro_f1"]), "accuracy_ci": list(ci["accuracy"]), "per_fold": per_fold,
                     "seconds": round(time.time() - t1, 1)}
            results["conditions"][kind]["readers"][name] = block
            say(f"[{kind}] {name:6s} macro-F1 {m['macro_f1']:.3f} [{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}] "
                f"(H {m['f1_per_class'][0]:.2f}, E {m['f1_per_class'][1]:.2f}, C {m['f1_per_class'][2]:.2f}) "
                f"acc {m['accuracy']:.3f}; {time.time() - t1:.0f}s")
            (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
            write_summary(out_dir, results, args)
    say(f"done in {time.time() - t0:.0f}s")


def write_summary(out_dir: Path, results: dict, args) -> None:
    L = results["L_star"]
    lines = [f"# P6 nonlinear refit after linear erasure at layer {L}, `{results['recording']}`" + (" (QUICK CHECK)" if results["quick"] else ""), "",
             f"Post hoc on the P2 recording: the fold's plane (fit on training chains as in P4) is projected out of the held-out "
             f"chains' layer-{L} rows; each reader is then trained on the edited rows with the same chain-grouped folds and scored on "
             f"held-out chains. Macro-F1 [95% CI, chains resampled] (H, E, C = per-class F1). Floor 0.20 (always coil); "
             f"no-model baselines: residue identity 0.44, +/-3 window 0.57. Readers: linear = multinomial logistic regression "
             f"(C = {results['C_fix']}, {results['n_train']['linear']} residues per fold, the P4 refit); mlp = one hidden layer of "
             f"{results['hidden']} ReLU units, early stopping, {results['n_train']['mlp']} residues per fold; svm = RBF kernel, "
             f"C = 1, gamma = scale, {results['n_train']['svm']} residues per fold; quad = logistic regression on the coordinates and "
             f"their squares (axis-aligned ellipsoids, C by inner CV, {results['n_train']['quad']} residues per fold); qda = one "
             f"mean and one full covariance per class, {results['n_train']['qda']} residues per fold.", "",
             "| removed at layer " + str(L) + " | " + " | ".join(args.readers) + " |", "|---|" + "---|" * len(args.readers)]
    for kind, block in results["conditions"].items():
        cells = []
        for name in args.readers:
            r = block["readers"].get(name)
            cells.append("pending" if r is None else
                         f"{r['macro_f1']:.2f} [{r['macro_f1_ci'][0]:.2f}, {r['macro_f1_ci'][1]:.2f}] "
                         f"(H {r['f1_per_class'][0]:.2f}, E {r['f1_per_class'][1]:.2f}, C {r['f1_per_class'][2]:.2f})")
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    for kind, block in results["conditions"].items():
        if "shape_fold0" in block:
            sh = block["shape_fold0"]
            lines += ["", f"Shape after `{kind}` (fold 0 held-out rows): class-mean distances in pooled-sd units "
                      + ", ".join(f"{k} {v:.2f}" for k, v in sh["mean_distance_over_pooled_sd"].items())
                      + "; variance ratios (class a / class b along the generalized eigen-directions): "
                      + "; ".join(f"{k}: max {v['max']:.1f}, min {v['min']:.2f}, median {v['median']:.2f}, {v['n_dirs_ratio_over_2']} dirs > 2, "
                                  f"{v['n_dirs_ratio_under_half']} dirs < 0.5" for k, v in sh["variance_ratio"].items())]
    lines += ["", "Leftover coordinate along the removed plane on held-out rows (max |(x - mu) . u|): " +
              ", ".join(f"{k} {v['max_leftover_coordinate']:.1e}" for k, v in results["conditions"].items())]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
