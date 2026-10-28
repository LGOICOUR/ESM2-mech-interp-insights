"""P2 reader: run ESM-2 over each protein once and record the residual stream.

For every residue we keep the residual-stream vector after the embedding and
after each transformer layer, read from forward hooks on the embedding module and
on each `EsmLayer`. Index 0 = embedding output, k = output of layer k. (The last
entry of `output_hidden_states` has the final LayerNorm applied; the hook on the
last layer does not, so every layer here is the raw residual.)

The same hooks are the attachment point for the P4 ablation: a projection hook
on layer k rewrites that layer's output before layer k+1 reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

MODEL_ID = "facebook/esm2_t30_150M_UR50D"


def load_model(model_id: str = MODEL_ID, device: str | None = None):
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id).to(device).eval()
    return tok, model, device


def layer_modules(model) -> list:
    """Modules whose outputs are the residual stream: [embeddings, layer 1, ..., layer n]."""
    return [model.esm.embeddings] + list(model.esm.encoder.layer)


@dataclass
class Batch:
    input_ids: torch.Tensor      # [B, T]
    attention_mask: torch.Tensor  # [B, T]
    residue_mask: torch.Tensor    # [B, T] True at residue positions (not <cls>, <eos>, pad)
    lengths: list[int]


def encode(tok, seqs: list[str], device: str) -> Batch:
    enc = tok(seqs, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]
    residue = am.bool().clone()
    residue[:, 0] = False  # <cls>
    for i, s in enumerate(seqs):
        residue[i, len(s) + 1] = False  # <eos>
        residue[i, len(s) + 2:] = False
    return Batch(ids.to(device), am.to(device), residue.to(device), [len(s) for s in seqs])


class Recorder:
    """Collects each hooked module's output for the current forward pass."""

    def __init__(self, model, layers: list[int] | None = None):
        mods = layer_modules(model)
        self.layers = list(range(len(mods))) if layers is None else list(layers)
        self.out: dict[int, torch.Tensor] = {}
        self.handles = [mods[k].register_forward_hook(self._hook(k)) for k in self.layers]

    def _hook(self, k):
        def hook(_m, _i, out):
            self.out[k] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def residuals_for_batch(model, batch: Batch, layers: list[int] | None = None) -> list[np.ndarray]:
    """One forward pass. Returns, per sequence, float16 [len, n_recorded_layers, width]
    at the residue positions only."""
    rec = Recorder(model, layers)
    try:
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    finally:
        rec.remove()
    stacked = torch.stack([rec.out[k] for k in rec.layers], dim=2)  # [B, T, L, W]
    out = []
    for i, n in enumerate(batch.lengths):
        out.append(stacked[i, 1:n + 1].to(torch.float16).cpu().numpy())
    return out


def length_batches(lengths: list[int], token_budget: int) -> list[list[int]]:
    """Group sequence indices (longest first) so that batch_size * max_len <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur = [], []
    for i in order:
        n = lengths[i] + 2
        if cur and (len(cur) + 1) * (lengths[cur[0]] + 2) > token_budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def mask_schedule(length: int, k: int) -> np.ndarray:
    """[k, length] bool: pass j masks residues j, j+k, j+2k, ... Every residue is
    masked in exactly one pass, and masked residues within a pass are k apart."""
    sched = np.zeros((k, length), dtype=bool)
    for j in range(k):
        sched[j, j::k] = True
    return sched


@dataclass
class MaskedReadout:
    logp_true: np.ndarray  # [len] log p(true residue | rest of chain), residue masked
    top1: np.ndarray       # [len] token id of the model's top prediction at the masked residue
    k: int = field(default=8)


@torch.no_grad()
def masked_readout(tok, model, device: str, seq: str, k: int = 8, ctx=None) -> MaskedReadout:
    """Pseudo-log-likelihood by k passes: pass j masks every k-th residue starting at
    j; the readout at each residue comes from the pass in which it was masked. The
    full single-residue PLL would need len(seq) passes; k=8 masks 12.5% per pass,
    close to the 15% the model was trained with. `ctx`, if given, receives the
    residue mask of the k-pass batch so P4's projection hooks act on residues only."""
    ids = tok(seq, return_tensors="pt")["input_ids"]  # [1, len+2]
    n = len(seq)
    sched = torch.tensor(mask_schedule(n, k))
    batch_ids = ids.repeat(k, 1)
    for j in range(k):
        batch_ids[j, 1:n + 1][sched[j]] = tok.mask_token_id
    batch_ids = batch_ids.to(device)
    if ctx is not None:  # tell ablation hooks which positions are residues
        rm = torch.zeros(k, n + 2, dtype=torch.bool, device=device)
        rm[:, 1:n + 1] = True
        ctx.residue_mask = rm
    logits = model(input_ids=batch_ids).logits[:, 1:n + 1].float()  # [k, n, V]
    logp = logits.log_softmax(-1)
    true = ids[0, 1:n + 1].to(device)
    pass_of = torch.tensor(np.argmax(mask_schedule(n, k), axis=0), device=device)  # [n]
    pos = torch.arange(n, device=device)
    lp = logp[pass_of, pos, true]
    top1 = logits[pass_of, pos].argmax(-1)
    return MaskedReadout(lp.cpu().numpy(), top1.cpu().numpy(), k)
