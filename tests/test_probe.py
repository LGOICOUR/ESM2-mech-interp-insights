"""Offline tests for the probe machinery on synthetic activations."""

import numpy as np

from esm2_probe.baselines import majority_predict, onehot_predict
from esm2_probe.probe import (confusion_by_group, cv_probe, diff_in_means, fit_lr, grouped_bootstrap_ci,
                              grouped_bootstrap_mean, grouped_folds, metrics, metrics_from_cm,
                              nearest_centroid_predict, probe_direction)


def _synthetic(n_prot=40, per=30, width=16, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_prot), per)
    y = rng.integers(0, 3, size=len(g))
    X_noise = rng.normal(size=(len(y), width))
    X_sig = rng.normal(size=(len(y), width))
    planted = np.zeros((3, width))
    planted[0, 0], planted[1, 1], planted[2, 2] = 3.0, 3.0, 3.0
    X_sig += planted[y]
    return X_noise, X_sig, y, g, planted


def test_folds_never_split_a_protein():
    g = np.repeat(np.arange(30), 7)
    folds = grouped_folds(g, 5, 0)
    for u in set(g.tolist()):
        assert len(set(folds[g == u].tolist())) == 1
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]


def test_probe_decodes_signal_layer_and_is_at_chance_on_noise():
    X_noise, X_sig, y, g, _ = _synthetic()
    folds = grouped_folds(g, 4, 0)
    sig = cv_probe(X_sig, y, g, folds, n_train=500)
    noise = cv_probe(X_noise, y, g, folds, n_train=500)
    assert metrics(y, sig["yhat"])["macro_f1"] > 0.9
    assert abs(metrics(y, noise["yhat"])["accuracy"] - 1 / 3) < 0.08
    assert len(sig["C_per_fold"]) == 4 and len(sig["inner_f1_per_fold"]) == 4
    assert min(sig["inner_f1_per_fold"]) > max(noise["inner_f1_per_fold"])


def test_bootstrap_ci_brackets_the_point_estimate():
    _, X_sig, y, g, _ = _synthetic()
    folds = grouped_folds(g, 4, 0)
    yhat = cv_probe(X_sig, y, g, folds, n_train=500, C=1.0)["yhat"]
    m = metrics(y, yhat)
    ci = grouped_bootstrap_ci(y, yhat, g, n_boot=200)
    for k in ("accuracy", "macro_f1"):
        lo, hi = ci[k]
        assert lo <= m[k] <= hi and hi - lo < 0.2
    # the confusion-matrix metrics agree with sklearn's
    cm, _ = confusion_by_group(y, yhat, g)
    mm = metrics_from_cm(cm.sum(0))
    assert abs(mm["macro_f1"] - m["macro_f1"]) < 1e-9 and abs(mm["accuracy"] - m["accuracy"]) < 1e-9
    v = np.where(y == yhat, 1.0, 0.0)
    mean, lo, hi = grouped_bootstrap_mean(v, g, n_boot=200)
    assert abs(mean - m["accuracy"]) < 1e-9 and lo <= mean <= hi


def test_diff_in_means_and_probe_direction_recover_planted_axes():
    _, X_sig, y, g, planted = _synthetic()
    dirs, mu = diff_in_means(X_sig, y)
    assert np.allclose(mu, X_sig.mean(0))
    for c in range(3):
        # one-vs-rest: expected direction is planted[c] minus the mean of the other planted axes
        expect = planted[c] - np.delete(planted, c, axis=0).mean(0)
        cos = float(dirs[c] @ expect / np.linalg.norm(expect))
        assert cos > 0.95
    sc, clf = fit_lr(X_sig, y, 1.0)
    for c in range(3):
        # multinomial weights are symmetric (sum to zero over classes), so the class-c
        # weight points along planted[c] minus the mean planted axis: parallel to `expect`
        expect = planted[c] - np.delete(planted, c, axis=0).mean(0)
        u = probe_direction(sc, clf, c)
        assert abs(np.linalg.norm(u) - 1) < 1e-5
        assert float(u @ expect / np.linalg.norm(expect)) > 0.95


def test_nearest_centroid_separates_planted_classes():
    _, X_sig, y, g, _ = _synthetic()
    yhat = nearest_centroid_predict(X_sig, X_sig, y)
    assert metrics(y, yhat)["accuracy"] > 0.9


def test_baselines_use_training_folds_only():
    y = np.array([2] * 25 + [0] * 5 + [2] * 25 + [0] * 5)
    folds = np.array([0] * 30 + [1] * 30)
    yhat = majority_predict(y, folds)
    assert set(yhat.tolist()) == {2}
    X = np.eye(3)[y].astype(np.float32)
    assert np.all(onehot_predict(X, y, folds) == y)


def test_cv_probe_skips_folds_without_held_out_residues():
    _, X_sig, y, g, _ = _synthetic(n_prot=12)
    folds = grouped_folds(g, 4, 0)
    folds[folds == 2] = 0  # fold 2 has no residues (as when --limit keeps only a few chains)
    res = cv_probe(X_sig, y, g, folds, n_train=500, C=1.0)
    assert res["C_per_fold"][2] is None and len(res["C_per_fold"]) == 4
    assert metrics(y, res["yhat"])["macro_f1"] > 0.9
