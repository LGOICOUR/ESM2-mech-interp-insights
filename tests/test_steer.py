"""Offline tests for P5: the steering hook, the composition-controlled class-mean
direction, the propensity scoring and the joint chain bootstrap."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from esm2_probe.probe import diff_in_means
from esm2_probe.steer import (SteerContext, aa_distribution, add_direction, attach_steer, chain_bootstrap, class_gap,
                              composition_subspace, former_mass, formers, kl_divergence, propensity_scores,
                              propensity_table, stratified_diff_in_means, subspace_share)


def test_add_direction_adds_v_only_at_masked_positions_and_copies():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 5, 4, generator=g)
    v = torch.tensor([1.0, -2.0, 0.5, 3.0])
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1] = mask[1, 3] = True
    out = add_direction(h, v, mask)
    assert torch.allclose(out[mask], h[mask] + v) and torch.equal(out[~mask], h[~mask])
    assert torch.equal(h, torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(0)))  # input untouched
    assert torch.equal(add_direction(h, v, torch.zeros(2, 5, dtype=torch.bool)), h)


class _Layer(nn.Module):
    def forward(self, x):
        return (x + 1.0,)  # EsmLayer returns a tuple


def _fake_model(n_layers=3):
    layers = nn.ModuleList([_Layer() for _ in range(n_layers)])
    return SimpleNamespace(esm=SimpleNamespace(embeddings=nn.Identity(), encoder=SimpleNamespace(layer=layers)))


def test_attach_steer_hooks_the_layer_and_offsets_residue_positions_for_cls():
    m = _fake_model()
    ctx = SteerContext()
    W = 5
    v = np.arange(1, W + 1, dtype=np.float32)
    handles = attach_steer(m, {2: v}, ctx, "cpu")
    x = torch.randn(3, 6, W)  # three copies of one chain, as in the k-pass read-out
    ctx.positions = np.array([0, 3])  # residues 0 and 3 sit at token positions 1 and 4
    y = m.esm.encoder.layer[1](x)[0]  # residual layer 2 = module index 1
    assert torch.allclose(y[:, [1, 4]], x[:, [1, 4]] + 1.0 + torch.tensor(v))
    assert torch.allclose(y[:, [0, 2, 3, 5]], x[:, [0, 2, 3, 5]] + 1.0)
    assert torch.allclose(m.esm.encoder.layer[0](x)[0], x + 1.0)  # other layers untouched
    ctx.positions = np.array([], dtype=int)
    assert torch.allclose(m.esm.encoder.layer[1](x)[0], x + 1.0)  # nothing to steer
    ctx.positions = None
    assert torch.allclose(m.esm.encoder.layer[1](x)[0], x + 1.0)
    for hd in handles:
        hd.remove()
    ctx.positions = np.array([0])
    assert torch.allclose(m.esm.encoder.layer[1](x)[0], x + 1.0)  # detached


def _composition_confounded(seed=0, n=6000, W=8):
    """Three 'amino acids' along e0, e1, e2; class 0 ('helix') is enriched in amino acid 0
    and, within every amino acid, shifted by the structure direction e5."""
    rng = np.random.default_rng(seed)
    aa = rng.integers(0, 3, n)
    p_h = np.where(aa == 0, 0.8, 0.2)
    y = np.where(rng.random(n) < p_h, 0, rng.integers(1, 3, n))
    v = np.zeros(W); v[5] = 1.0
    X = 5.0 * np.eye(W)[aa] + np.outer(y == 0, v) + 0.3 * rng.normal(size=(n, W))
    return X, y, aa, v


def test_stratified_diff_in_means_removes_the_composition_component():
    X, y, aa, v = _composition_confounded()
    plain = diff_in_means(X.astype(np.float32), y)[0][0]
    strat, info = stratified_diff_in_means(X, y, aa, 0, n_aa=3)
    assert abs(np.linalg.norm(strat) - 1) < 1e-5 and info["n_amino_acids"] == 3 and abs(info["weight_used"] - 1) < 1e-9
    assert strat @ v > 0.99                      # the within-amino-acid direction is the structure direction
    assert plain @ v < 0.9                       # the plain class-mean direction is pulled toward amino acid 0
    Q = composition_subspace(X, aa, n_aa=3)
    assert Q.shape[0] == X.shape[1] and np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-6)
    assert subspace_share(plain, Q) > 0.5 and subspace_share(strat, Q) < 0.1
    assert class_gap(X, y, strat, 0, 2) == pytest.approx(1.0, abs=0.1)  # helix sits one unit from the rest along e5
    assert class_gap(X, y, -strat, 0, 2) < 0
    # a rare amino acid is skipped; none left raises
    aa2 = aa.copy(); aa2[:3] = 3
    assert stratified_diff_in_means(X, y, aa2, 0, n_aa=4)[1]["n_amino_acids"] == 3
    with pytest.raises(ValueError):
        stratified_diff_in_means(X, y, aa, 0, n_aa=3, min_count=10 ** 6)


def test_propensity_table_is_the_smoothed_log_ratio_and_formers_its_positive_entries():
    aa = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    y = np.array([0, 0, 1, 0, 1, 1, 1, 2, 2, 2])
    T = propensity_table(aa, y, n_aa=4, n_classes=3)
    assert T.shape == (3, 4)
    # class 0, amino acid 0: n(a,c) = 2 of n(c) = 3; n(a) = 3 of n = 10; add-one with 4 amino acids
    assert T[0, 0] == pytest.approx(np.log(3 / 7) - np.log(4 / 14))
    assert np.array_equal(formers(T), T > 0)
    aa_x, y_x = np.append(aa, 4), np.append(y, 0)   # an X residue (index n_aa) is ignored
    assert np.allclose(propensity_table(aa_x, y_x, n_aa=4, n_classes=3), T)


def test_propensity_scores_are_expectations_under_the_renormalized_prediction():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(3, 20))
    logp = rng.normal(size=(5, 20)) - 3.0                 # unnormalized log-probabilities
    p = aa_distribution(logp)
    assert np.allclose(p.sum(1), 1) and np.all(p > 0)
    S = propensity_scores(logp, T)
    assert S.shape == (5, 3) and np.allclose(S, p @ T.T, atol=1e-6)
    assert np.allclose(propensity_scores(logp + 1.7, T), S, atol=1e-6)   # a constant offset drops out
    onehot = np.full((1, 20), -30.0); onehot[0, 7] = 0.0
    assert np.allclose(propensity_scores(onehot, T)[0], T[:, 7], atol=1e-6)
    assert np.allclose(former_mass(logp, T), p @ (T > 0).T, atol=1e-6)
    assert np.allclose(kl_divergence(logp, logp), 0, atol=1e-7)
    assert np.all(kl_divergence(logp, rng.normal(size=(5, 20))) > 0)
    two = np.array([[np.log(0.5), np.log(0.5)] + [-50.0] * 18]); one = np.array([[np.log(0.9), np.log(0.1)] + [-50.0] * 18])
    assert kl_divergence(two, one)[0] == pytest.approx(0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1), abs=1e-5)


def test_chain_bootstrap_resamples_chains_jointly_across_parts():
    # one residue per chain in each part and part_a = 0.1 * part_b chain by chain: every resample's ratio of
    # (summed values / summed counts) is 0.1, so the CI must collapse on 0.1 (independent resampling would not)
    part_a = (np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]))
    part_b = (np.array([10.0, 20.0, 30.0]), np.array([0, 1, 2]))
    point, lo, hi = chain_bootstrap(lambda m: m[0] / m[1], [part_a, part_b], n_boot=200, seed=0)
    assert point == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    # means are residue-weighted (summed values over summed counts), not means of chain means
    part_w = (np.array([1.0, 1.0, 2.0, 2.0, 3.0]), np.array([0, 0, 1, 1, 2]))
    assert chain_bootstrap(lambda m: m[0], [part_w], n_boot=10, seed=0)[0] == pytest.approx(9 / 5)
    # a difference of means with a chain missing from one part still runs and brackets the point estimate
    part_c = (np.array([5.0, 7.0]), np.array([0, 2]))
    point, lo, hi = chain_bootstrap(lambda m: m[0] - m[1], [part_a, part_c], n_boot=300, seed=1)
    assert point == pytest.approx(2.0 - 6.0) and lo <= point <= hi and lo < hi
