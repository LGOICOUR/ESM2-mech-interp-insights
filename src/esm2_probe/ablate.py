"""P4 ablation: project one direction out of the residual stream during the
forward pass, at residue positions only.

At each in-scope layer, with unit direction u and reference mean mu (both fit on
training-fold residues), every residue vector h becomes
    h - ((h - mu) . u) u
so every residue has the same coordinate along u that mu has: the information
along u is gone for any linear reader downstream, and nothing else is touched.
With an orthonormal set U instead of u, the whole subspace is removed.
<cls>, <eos> and padding positions are left alone. Control: the same operation
along a random unit direction. `orthogonal_remainder` splits one direction into
what it shares with another and the unit remainder, so the remainder alone can
be removed. `variance_matched_random` draws a random direction that carries a
chosen share of the stream's variance, the control for a fitted direction's size.
"""

from __future__ import annotations

import numpy as np
import torch

from .reader import layer_modules


class AblationContext:
    """Holds the residue mask of the batch currently being processed, so hooks
    know which positions are residues."""

    def __init__(self) -> None:
        self.residue_mask: torch.Tensor | None = None


def project_out(h: torch.Tensor, U: torch.Tensor, mu: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    """h [B, T, W]; U [W] (one unit direction) or [d, W] (orthonormal rows); mu [W];
    residue_mask [B, T] bool. Returns h with the component in span(U) (about mu)
    removed at residue positions. Pure function, used by the hook and the tests."""
    U2 = U[None, :] if U.dim() == 1 else U
    x = h[residue_mask].float()
    proj = (x - mu) @ U2.T                 # [n, d]
    h = h.clone()
    h[residue_mask] = (x - proj @ U2).to(h.dtype)
    return h


def orthonormal_rows(vectors: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis (rows) of the span of the given vectors [d, W], by SVD with a
    rank cut-off, so linearly dependent inputs (e.g. the three multinomial weight
    vectors, which sum to zero) give a basis of the right size."""
    A = np.asarray(vectors, dtype=np.float64)
    _, sv, vt = np.linalg.svd(A, full_matrices=False)
    keep = sv > rel_tol * sv.max()
    return np.ascontiguousarray(vt[keep]).astype(np.float32)


def make_hook(U: np.ndarray, mu: np.ndarray, ctx: AblationContext, device: str):
    u_t = torch.tensor(np.asarray(U, dtype=np.float32), device=device)
    mu_t = torch.tensor(np.asarray(mu, dtype=np.float32), device=device)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        new = project_out(h, u_t, mu_t, ctx.residue_mask)
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new

    return hook


def attach(model, directions: dict[int, tuple[np.ndarray, np.ndarray]], ctx: AblationContext, device: str) -> list:
    """directions: layer index k (1..n_layers, as in the recordings) -> (U, mu), U one
    unit direction [W] or orthonormal rows [d, W]. Hooks the module whose output is
    residual layer k."""
    mods = layer_modules(model)
    return [mods[k].register_forward_hook(make_hook(u, mu, ctx, device)) for k, (u, mu) in directions.items()]


def random_unit(width: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed).normal(size=width)
    return (r / np.linalg.norm(r)).astype(np.float32)


def orthogonal_remainder(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Unit vector along the part of v that lies outside the direction w:
    (v - (v . w_hat) w_hat) / |...|, so that v_hat = cos(v, w) w_hat + sin(v, w) r.
    Used to split a direction into what it shares with another and what it adds,
    e.g. the probe direction minus its class-mean component. Raises if v and w
    are (nearly) parallel."""
    v = np.asarray(v, dtype=np.float64)
    w_hat = np.asarray(w, dtype=np.float64)
    w_hat = w_hat / np.linalg.norm(w_hat)
    r = v - (v @ w_hat) * w_hat
    n = np.linalg.norm(r)
    if n < 1e-6 * np.linalg.norm(v):
        raise ValueError("v is parallel to w: no remainder")
    return (r / n).astype(np.float32)


def variance_share(u: np.ndarray, cov: np.ndarray) -> float:
    """Variance of the coordinate along the unit direction u, as a share of the
    total variance tr(cov)."""
    u = np.asarray(u, dtype=np.float64)
    return float(u @ cov @ u / np.trace(cov))


def variance_matched_random(eigvals: np.ndarray, eigvecs: np.ndarray, target_share: float, seed: int,
                            p_max: float = 40.0, iters: int = 80) -> tuple[np.ndarray, dict]:
    """A random unit direction whose variance share (u' cov u / tr cov) equals
    `target_share`, for a covariance with eigenvalues `eigvals` and eigenvectors in
    the columns of `eigvecs` (as numpy's eigh returns them). Draw g ~ N(0, I) in the
    eigenbasis and tilt it toward the high-variance axes, c_i = g_i * lambda_i^(p/2):
    p = 0 is an isotropic random direction (share about 1/W), p -> inf the top
    eigenvector, and the share rises monotonically with p, so p is found by
    bisection. Returns the direction and a dict with the tilt, the achieved share,
    the cosine with the top eigenvector and the participation ratio (the effective
    number of eigen-axes the direction loads on). Raises ValueError if the target
    is outside the reachable range for this draw."""
    lam = np.asarray(eigvals, dtype=np.float64)
    V = np.asarray(eigvecs, dtype=np.float64)
    lam = np.clip(lam, 1e-9 * lam.max(), None)
    tr, loglam = lam.sum(), np.log(lam / lam.max())
    g = np.random.default_rng(seed).normal(size=lam.size)

    def share(p: float) -> float:
        w = g ** 2 * np.exp(p * loglam)
        return float(w @ lam / (w.sum() * tr))

    lo, hi = 0.0, float(p_max)
    if not share(lo) <= target_share <= share(hi):
        raise ValueError(f"target share {target_share:.4f} outside the reachable range [{share(lo):.4f}, {share(hi):.4f}]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if share(mid) < target_share:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    c = g * np.exp(0.5 * p * loglam)
    u = V @ c
    u /= np.linalg.norm(u)
    c2 = c ** 2 / (c ** 2).sum()
    info = {"tilt_p": float(p), "share": share(p), "target_share": float(target_share),
            "cos_top_eigvec": float(abs(u @ V[:, int(np.argmax(lam))])),
            "participation_ratio": float(1.0 / (c2 ** 2).sum())}
    return u.astype(np.float32), info
