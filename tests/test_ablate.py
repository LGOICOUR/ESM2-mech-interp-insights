"""Offline tests for the projection ablation: the maths of project_out and the
hook attachment, on tensors and a stand-in model."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from esm2_probe.ablate import AblationContext, attach, orthonormal_rows, project_out, random_unit


def _setup(seed=0, B=2, T=7, W=8):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, W, generator=g)
    u = torch.randn(W, generator=g); u = u / u.norm()
    mu = torch.randn(W, generator=g)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 0] = False; mask[:, -1] = False; mask[1, 4:] = False
    return h, u, mu, mask


def test_project_out_removes_only_the_u_component_at_residue_positions():
    h, u, mu, mask = _setup()
    out = project_out(h, u, mu, mask)
    coord = ((out[mask] - mu) @ u)
    assert torch.allclose(coord, torch.zeros_like(coord), atol=1e-5)
    assert torch.equal(out[~mask], h[~mask])
    # orthogonal components untouched
    v = torch.randn(8); v = v - (v @ u) * u
    assert torch.allclose(out[mask] @ v, h[mask] @ v, atol=1e-4)
    # idempotent
    assert torch.allclose(project_out(out, u, mu, mask), out, atol=1e-5)
    assert torch.equal(h, _setup()[0])  # input not modified in place


def test_random_unit_is_seeded_and_normalized():
    a, b = random_unit(640, 3), random_unit(640, 3)
    assert np.array_equal(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    assert abs(float(a @ random_unit(640, 4))) < 0.2


class _Layer(nn.Module):
    def forward(self, x):
        return (x + 1.0,)  # EsmLayer returns a tuple


def _fake_model(n_layers=3):
    layers = nn.ModuleList([_Layer() for _ in range(n_layers)])
    emb = nn.Identity()
    return SimpleNamespace(esm=SimpleNamespace(embeddings=emb, encoder=SimpleNamespace(layer=layers)))


def test_attach_hooks_the_named_layer_and_leaves_other_positions():
    m = _fake_model()
    ctx = AblationContext()
    W = 5
    u = np.eye(W, dtype=np.float32)[0]
    mu = np.zeros(W, dtype=np.float32)
    handles = attach(m, {2: (u, mu)}, ctx, "cpu")
    x = torch.randn(1, 4, W)
    ctx.residue_mask = torch.tensor([[False, True, True, False]])
    y = m.esm.encoder.layer[1](x)[0]  # residual layer 2 = module index 1
    assert torch.allclose(y[0, 1:3, 0], torch.zeros(2))  # component along e0 removed (mu = 0)
    assert torch.allclose(y[0, 1:3, 1:], x[0, 1:3, 1:] + 1.0)
    assert torch.allclose(y[0, 0], x[0, 0] + 1.0) and torch.allclose(y[0, 3], x[0, 3] + 1.0)
    untouched = m.esm.encoder.layer[0](x)[0]
    assert torch.allclose(untouched, x + 1.0)
    for hd in handles:
        hd.remove()
    assert torch.allclose(m.esm.encoder.layer[1](x)[0], x + 1.0)


def test_project_out_subspace_removes_every_direction_in_the_span():
    h, u, mu, mask = _setup(seed=1)
    g = torch.Generator().manual_seed(5)
    raw = torch.randn(3, 8, generator=g).numpy()
    U = torch.tensor(orthonormal_rows(raw))
    assert torch.allclose(U @ U.T, torch.eye(3), atol=1e-5)
    out = project_out(h, U, mu, mask)
    for row in raw:
        coord = (out[mask] - mu) @ torch.tensor(row, dtype=torch.float32)
        assert torch.allclose(coord, torch.zeros_like(coord), atol=1e-4)
    v = torch.randn(8, generator=g); v = v - U.T @ (U @ v)  # orthogonal to the span
    assert torch.allclose(out[mask] @ v, h[mask] @ v, atol=1e-4)
    assert torch.equal(out[~mask], h[~mask])


def test_orthonormal_rows_drops_dependent_directions():
    a = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    rows = np.vstack([a, -a.sum(0, keepdims=True)])  # three vectors summing to zero: rank 2
    U = orthonormal_rows(rows)
    assert U.shape == (2, 4)
    assert np.allclose(U @ U.T, np.eye(2), atol=1e-6)
    assert np.allclose(np.abs(U[:, 2:]).sum(), 0, atol=1e-6)  # span is the first two axes
