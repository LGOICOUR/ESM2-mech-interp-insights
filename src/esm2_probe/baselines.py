"""No-model baselines scored with the same grouped folds as the probes:
majority class, and logistic regression on one-hot amino-acid identity
(the residue alone, or a +/- window of neighbours)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def majority_predict(y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    yhat = np.zeros(len(y), dtype=int)
    for f in range(int(folds.max()) + 1):
        tr = folds != f
        yhat[folds == f] = np.bincount(y[tr], minlength=3).argmax()
    return yhat


def onehot_predict(X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 1.0) -> np.ndarray:
    yhat = np.zeros(len(y), dtype=int)
    for f in range(int(folds.max()) + 1):
        tr, te = folds != f, folds == f
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        yhat[te] = clf.predict(X[te])
    return yhat
