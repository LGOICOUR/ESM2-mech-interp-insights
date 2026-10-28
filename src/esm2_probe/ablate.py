"""P4 ablation: project one direction out of the residual stream during the
forward pass, at residue positions only.

At each in-scope layer, with unit direction u and reference mean mu (both fit on
training-fold residues), every residue vector h becomes
    h - ((h - mu) . u) u
so every residue has the same coordinate along u that mu has: the information
along u is gone for any linear reader downstream, and nothing else is touched.
With an orthonormal set U instead of u, the whole subspace is removed.
<cls>, <eos> and padding positions are left alone. Control: the same operation
along a random unit direction.
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
