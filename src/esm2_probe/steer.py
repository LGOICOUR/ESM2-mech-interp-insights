"""P5 steering: add a direction to the residual stream during the forward pass,
at chosen residue positions, and score the model's masked predictions by
amino-acid propensity.

Where the P4 hook removes a direction (h <- h - ((h - mu) . u) u), the steering
hook adds one: h <- h + v at the steered positions, with v = d * g * u for a
unit direction u, a dose d and the gap g = (mu_class - mu_target) . u measured
on the training residues, so d = 1 moves the target residues' mean coordinate
along u onto the class mean's. <cls>, <eos>, padding and every other residue
are untouched. `stratified_diff_in_means` is the within-amino-acid class-mean
direction, the composition-controlled version of `probe.diff_in_means`.
`propensity_table` turns labelled residues into Chou-Fasman-style log-ratio
propensities per class; `propensity_scores` is the expected log-propensity of a
predicted amino-acid distribution under that table; `chain_bootstrap` gives a
CI for any statistic of chain-resampled means.
"""

from __future__ import annotations

import numpy as np
import torch

from .ablate import orthonormal_rows
from .probe import N_CLASSES
from .reader import layer_modules

N_AA = 20  # the standard amino acids, in data.AMINO_ACIDS order; index N_AA is X


class SteerContext:
    """Positions (0-based residue indices within the chain being processed) at
    which the hooks add their vector; the hook offsets by one for <cls>. None or
    empty = add nothing."""

    def __init__(self) -> None:
        self.positions: np.ndarray | None = None


def add_direction(h: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h [B, T, W]; v [W]; mask [B, T] bool. Returns a copy of h with v added at the
    masked positions. Pure function, used by the hook and the tests."""
    h = h.clone()
    h[mask] = (h[mask].float() + v).to(h.dtype)
    return h


def position_mask(B: int, T: int, positions: np.ndarray, device) -> torch.Tensor:
    """[B, T] bool, True at token position p + 1 for every residue index p (every
    row of the batch is the same chain, as in the k-pass masked read-out)."""
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    if len(positions):
        m[:, torch.as_tensor(np.asarray(positions, dtype=np.int64) + 1, device=device)] = True
    return m


def make_steer_hook(v: np.ndarray, ctx: SteerContext, device: str):
    v_t = torch.tensor(np.asarray(v, dtype=np.float32), device=device)

    def hook(_m, _i, out):
        if ctx.positions is None or len(ctx.positions) == 0:
            return None  # leave the output as it is
        h = out[0] if isinstance(out, tuple) else out
        new = add_direction(h, v_t, position_mask(h.shape[0], h.shape[1], ctx.positions, h.device))
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new

    return hook


def attach_steer(model, vectors: dict[int, np.ndarray], ctx: SteerContext, device: str) -> list:
    """vectors: layer index k (1..n_layers, as in the recordings) -> vector [W] to add.
    Hooks the module whose output is residual layer k."""
    mods = layer_modules(model)
    return [mods[k].register_forward_hook(make_steer_hook(v, ctx, device)) for k, v in vectors.items()]


def class_gap(X: np.ndarray, y: np.ndarray, u: np.ndarray, cls: int, ref: int) -> float:
    """(mean of class cls - mean of class ref) . u: the dose unit along u."""
    return float((X[y == cls].mean(0) - X[y == ref].mean(0)) @ np.asarray(u, dtype=np.float64))


def stratified_diff_in_means(X: np.ndarray, y: np.ndarray, aa: np.ndarray, cls: int, *, n_aa: int = N_AA,
                             min_count: int = 5) -> tuple[np.ndarray, dict]:
    """Within-amino-acid class-mean direction: for each amino acid a with at least
    min_count residues on both sides, mean of class-cls residues minus mean of the
    rest, averaged with the amino acid's overall frequency as weight, normalized.
    Residues whose amino acid is outside 0..n_aa-1 (X) are skipped. Carries no
    composition difference by construction."""
    X = np.asarray(X, dtype=np.float64)
    acc, wsum, used = np.zeros(X.shape[1]), 0.0, 0
    for a in range(n_aa):
        m = aa == a
        pos, neg = m & (y == cls), m & (y != cls)
        if pos.sum() < min_count or neg.sum() < min_count:
            continue
        w = float(m.mean())
        acc += w * (X[pos].mean(0) - X[neg].mean(0))
        wsum += w
        used += 1
    if used == 0:
        raise ValueError("no amino acid has enough residues on both sides")
    d = acc / wsum
    return (d / np.linalg.norm(d)).astype(np.float32), {"n_amino_acids": used, "weight_used": wsum}


def composition_subspace(X: np.ndarray, aa: np.ndarray, n_aa: int = N_AA) -> np.ndarray:
    """Orthonormal basis [W, r] of the span of the amino-acid means about the grand
    mean: the directions along which amino-acid identity moves the stream."""
    X = np.asarray(X, dtype=np.float64)
    rows = np.stack([X[aa == a].mean(0) - X.mean(0) for a in range(n_aa) if (aa == a).any()])
    return orthonormal_rows(rows).T.astype(np.float64)


def subspace_share(u: np.ndarray, Q: np.ndarray) -> float:
    """Squared norm of the unit direction u inside the subspace with orthonormal basis Q [W, r]."""
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    return float(np.linalg.norm(Q.T @ u) ** 2)


def propensity_table(aa: np.ndarray, y: np.ndarray, *, n_aa: int = N_AA, n_classes: int = N_CLASSES) -> np.ndarray:
    """[n_classes, n_aa] log propensities: log of amino acid a's frequency within
    class c over its overall frequency, add-one smoothed. Residues with aa >= n_aa
    (X) are excluded from the counts."""
    valid = aa < n_aa
    n_a = np.bincount(aa[valid], minlength=n_aa).astype(np.float64)
    overall = np.log((n_a + 1) / (n_a.sum() + n_aa))
    table = np.zeros((n_classes, n_aa))
    for c in range(n_classes):
        n_ac = np.bincount(aa[valid & (y == c)], minlength=n_aa).astype(np.float64)
        table[c] = np.log((n_ac + 1) / (n_ac.sum() + n_aa)) - overall
    return table


def formers(table: np.ndarray) -> np.ndarray:
    """[n_classes, n_aa] bool: the amino acids with positive propensity for each class."""
    return table > 0


def aa_distribution(logp_aa: np.ndarray) -> np.ndarray:
    """Renormalized distribution over the amino acids from their log-probabilities
    (any constant offset per row, e.g. mass on non-amino-acid tokens, drops out)."""
    z = np.asarray(logp_aa, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def propensity_scores(logp_aa: np.ndarray, table: np.ndarray) -> np.ndarray:
    """[n, n_classes] expected log-propensity of each predicted distribution under each class."""
    return (aa_distribution(logp_aa) @ table.T).astype(np.float32)


def former_mass(logp_aa: np.ndarray, table: np.ndarray) -> np.ndarray:
    """[n, n_classes] predicted probability mass on each class's formers."""
    return (aa_distribution(logp_aa) @ formers(table).T.astype(np.float64)).astype(np.float32)


def kl_divergence(logp_new: np.ndarray, logp_base: np.ndarray) -> np.ndarray:
    """[n] KL(p_new || p_base) between the renormalized amino-acid distributions."""
    p, q = aa_distribution(logp_new), aa_distribution(logp_base)
    return (p * (np.log(p) - np.log(q))).sum(1).astype(np.float32)


def chain_bootstrap(fn, parts: list[tuple[np.ndarray, np.ndarray]], *, n_boot: int = 1000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float, float]:
    """CI for a statistic of several means over different residue subsets of the
    same chains. parts: list of (values [m_i], chain index [m_i]). Chains (the union
    over parts) are resampled with replacement; each resample's means are the
    ratio of summed values to summed counts over the picked chains, and fn(list of
    means) -> float is the statistic. Returns (fn on the full data, lo, hi); a
    resample in which some part has no residues is skipped."""
    chains = np.unique(np.concatenate([np.asarray(g) for _, g in parts]))
    sums, cnts = [], []
    for v, g in parts:
        idx = np.searchsorted(chains, np.asarray(g))
        sums.append(np.bincount(idx, weights=np.asarray(v, dtype=np.float64), minlength=len(chains)))
        cnts.append(np.bincount(idx, minlength=len(chains)).astype(np.float64))
    point = float(fn([s.sum() / c.sum() for s, c in zip(sums, cnts)]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(chains), size=len(chains))
        c_b = [c[pick].sum() for c in cnts]
        if min(c_b) == 0:
            continue
        boots.append(fn([s[pick].sum() / cb for s, cb in zip(sums, c_b)]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
