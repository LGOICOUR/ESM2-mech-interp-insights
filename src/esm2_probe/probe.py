"""P3 probes: per-layer multinomial logistic regression on per-residue residual
activations, scored only on held-out proteins (grouped folds: every residue of a
chain is on one side). C is tuned inside the training rounds on a subsample.
Confidence intervals resample proteins, not residues.

Also the class-mean ("diff-in-means") directions, nearest-centroid decoding, and
the raw-space unit direction behind a probe class, for the P4 ablation.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
N_CLASSES = 3


def grouped_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold index per residue such that every residue of a protein shares one fold."""
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def subsample(idx: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if len(idx) <= n_max:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, size=n_max, replace=False))


def fit_lr(Xtr: np.ndarray, ytr: np.ndarray, C: float):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return sc, clf


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {"accuracy": float(np.mean(y == yhat)),
            "macro_f1": float(f1_score(y, yhat, average="macro", labels=list(range(N_CLASSES)), zero_division=0)),
            "f1_per_class": [float(v) for v in f1_score(y, yhat, average=None, labels=list(range(N_CLASSES)), zero_division=0)]}


def inner_select_c(X: np.ndarray, y: np.ndarray, g: np.ndarray, *, k_inner: int = 3, n_sub: int = 8000, seed: int = 0) -> tuple[float, float]:
    """Pick C by grouped inner CV (macro-F1) on a residue subsample of the training fold.
    Returns (C, inner macro-F1 at that C); the score is what layer choice inside the
    training rounds uses."""
    sub = subsample(np.arange(len(y)), n_sub, seed)
    Xs, ys, gs = X[sub], y[sub], g[sub]
    folds = grouped_folds(gs, k_inner, seed + 1)
    best, best_c = -1.0, C_GRID[0]
    for C in C_GRID:
        yhat = np.zeros(len(ys), dtype=int)
        for f in range(k_inner):
            tr, te = folds != f, folds == f
            if len(set(ys[tr].tolist())) < 2:
                continue
            sc, clf = fit_lr(Xs[tr], ys[tr], C)
            yhat[te] = clf.predict(sc.transform(Xs[te]))
        m = metrics(ys, yhat)["macro_f1"]
        if m > best:
            best, best_c = m, C
    return best_c, best


def cv_probe(X: np.ndarray, y: np.ndarray, g: np.ndarray, folds: np.ndarray, *, n_train: int = 40000,
             seed: int = 0, C: float | None = None) -> dict:
    """Held-out predictions for every residue. Per outer fold: choose C on the training
    residues (unless given), fit on up to n_train training residues, predict the fold."""
    k = int(folds.max()) + 1
    yhat = np.zeros(len(y), dtype=int)
    proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    chosen, inner = [], []
    for f in range(k):
        tr_idx, te_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        if len(te_idx) == 0:  # a fold with no held-out residues (small check runs)
            chosen.append(None); inner.append(float("nan"))
            continue
        if C is not None:
            c, score = C, float("nan")
        else:
            c, score = inner_select_c(X[tr_idx], y[tr_idx], g[tr_idx], seed=seed + f)
        inner.append(score)
        fit_idx = subsample(tr_idx, n_train, seed + 100 + f)
        sc, clf = fit_lr(X[fit_idx], y[fit_idx], c)
        Z = sc.transform(X[te_idx])
        yhat[te_idx] = clf.predict(Z)
        proba[te_idx] = clf.predict_proba(Z)
        chosen.append(c)
    return {"yhat": yhat, "proba": proba, "C_per_fold": chosen, "inner_f1_per_fold": inner}


def confusion_by_group(y: np.ndarray, yhat: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group 3x3 confusion counts [n_groups, true, pred] and the group ids."""
    uniq, inv = np.unique(g, return_inverse=True)
    cm = np.zeros((len(uniq), N_CLASSES, N_CLASSES))
    np.add.at(cm, (inv, y, yhat), 1)
    return cm, uniq


def metrics_from_cm(cm: np.ndarray) -> dict:
    """Accuracy and macro-F1 from a 3x3 confusion matrix (true x pred)."""
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1)), "macro_f1": float(f1.mean()), "f1_per_class": f1.tolist()}


def grouped_bootstrap_ci(y: np.ndarray, yhat: np.ndarray, g: np.ndarray, *, n_boot: int = 1000, seed: int = 0,
                         alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile CIs of accuracy and macro-F1 from resampling groups (proteins) with
    replacement. Exact: a resample's confusion matrix is the sum of its groups' matrices."""
    cm, uniq = confusion_by_group(y, yhat, g)
    rng = np.random.default_rng(seed)
    vals = {"accuracy": [], "macro_f1": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        m = metrics_from_cm(cm[pick].sum(0))
        for k in vals:
            vals[k].append(m[k])
    return {k: tuple(float(q) for q in np.quantile(v, [alpha / 2, 1 - alpha / 2])) for k, v in vals.items()}


def grouped_bootstrap_mean(values: np.ndarray, g: np.ndarray, *, n_boot: int = 1000, seed: int = 0,
                           alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of a per-residue quantity with a CI from resampling groups (proteins):
    returns (mean, lo, hi). A resample's mean is the ratio of summed values to summed counts."""
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        boots.append(sums[pick].sum() / cnts[pick].sum())
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def diff_in_means(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions [n_classes, width]: normalized (mean of class c - mean of the rest),
    plus the grand mean [width]. Raw activation space."""
    mu = X.mean(0)
    dirs = []
    for c in range(N_CLASSES):
        d = X[y == c].mean(0) - X[y != c].mean(0)
        dirs.append(d / np.linalg.norm(d))
    return np.stack(dirs).astype(np.float32), mu.astype(np.float32)


def nearest_centroid_predict(X: np.ndarray, Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    """Class-mean decoder: standardize on the training residues, assign each residue to
    the nearest class centroid."""
    sc = StandardScaler().fit(Xtr)
    Z, Ztr = sc.transform(X), sc.transform(Xtr)
    cents = np.stack([Ztr[ytr == c].mean(0) for c in range(N_CLASSES)])
    d2 = ((Z[:, None, :] - cents[None]) ** 2).sum(-1)
    return d2.argmin(1)


def probe_direction(sc: StandardScaler, clf: LogisticRegression, class_idx: int) -> np.ndarray:
    """Unit vector in raw activation space along which the probe's logit for class_idx
    grows fastest (its weight vector, undone through the scaler)."""
    w = clf.coef_[class_idx] / sc.scale_
    return (w / np.linalg.norm(w)).astype(np.float32)
