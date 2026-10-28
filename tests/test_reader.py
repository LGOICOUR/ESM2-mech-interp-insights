"""Offline tests for the reader: masking schedule, batching, token alignment and
the masked read-out, on a fake model (and the real tokenizer when it is cached)."""

import numpy as np
import pytest
import torch

from esm2_probe.ablate import AblationContext
from esm2_probe.reader import MODEL_ID, encode, length_batches, mask_schedule, masked_readout

V = 33


class FakeTok:
    """ESM-like: <cls>=0, <pad>=1, <eos>=2, <mask>=32, letters from 4."""
    mask_token_id = 32

    def __call__(self, seqs, return_tensors="pt", padding=False):
        if isinstance(seqs, str):
            seqs = [seqs]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n + 2), 1)
        am = torch.zeros((len(seqs), n + 2), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, 0] = 0
            ids[i, 1:len(s) + 1] = torch.tensor([4 + (ord(c) - 65) % 20 for c in s])
            ids[i, len(s) + 1] = 2
            am[i, :len(s) + 2] = 1
        return {"input_ids": ids, "attention_mask": am}


class FakeModel:
    """Logits that prefer token (position mod V) at every position, plus a bump at the
    true token when the input is NOT masked, so the readout must come from the masked pass."""

    def __call__(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, V)
        for t in range(T):
            logits[:, t, t % V] = 10.0
        unmasked = input_ids != 32
        logits[unmasked.unsqueeze(-1).expand(-1, -1, V) & (torch.nn.functional.one_hot(input_ids.clamp(max=V - 1), V) > 0)] += 50.0
        class Out:
            pass
        o = Out()
        o.logits = logits
        return o


def test_mask_schedule_covers_each_residue_once_with_spacing():
    s = mask_schedule(19, 8)
    assert s.shape == (8, 19) and np.all(s.sum(0) == 1)
    for j in range(8):
        pos = np.where(s[j])[0]
        assert np.all(np.diff(pos) == 8)


def test_length_batches_respect_budget_and_cover_everything():
    lengths = [500, 10, 300, 300, 20, 5]
    b = length_batches(lengths, token_budget=700)
    assert sorted(i for grp in b for i in grp) == list(range(6))
    for grp in b:
        assert len(grp) * (max(lengths[i] for i in grp) + 2) <= 700


def test_encode_marks_residue_positions_only():
    b = encode(FakeTok(), ["ACDE", "KL"], "cpu")
    assert b.residue_mask.sum(1).tolist() == [4, 2]
    assert not b.residue_mask[0, 0] and not b.residue_mask[0, 5] and not b.residue_mask[1, 3]
    assert b.lengths == [4, 2]


def test_masked_readout_reads_each_residue_from_its_masked_pass():
    seq = "ACDEFGHIKLMNPQRSTVWYAC"
    r = masked_readout(FakeTok(), FakeModel(), "cpu", seq, k=4)
    n = len(seq)
    assert r.top1.tolist() == [(i + 1) % V for i in range(n)]  # token position i+1, masked pass
    # the +50 bump never leaks in: log p(true) is -10-ish unless true == (i+1) % V
    ids = FakeTok()(seq)["input_ids"][0, 1:n + 1]
    expect_hi = [(int(ids[i]) == (i + 1) % V) for i in range(n)]
    assert all((lp > -1.0) == hi for lp, hi in zip(r.logp_true, expect_hi))


def test_real_tokenizer_alignment_if_cached():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    b = encode(tok, [seq, "ACD"], "cpu")
    ids = b.input_ids[0][b.residue_mask[0]].tolist()
    assert tok.convert_ids_to_tokens(ids) == list(seq)
    assert b.input_ids[0, 0].item() == tok.cls_token_id and b.input_ids[0, len(seq) + 1].item() == tok.eos_token_id


def test_masked_readout_sets_residue_mask_on_the_context():
    ctx = AblationContext()
    seq = "ACDEFGHIK"
    masked_readout(FakeTok(), FakeModel(), "cpu", seq, k=3, ctx=ctx)
    rm = ctx.residue_mask
    assert rm.shape == (3, len(seq) + 2)
    assert rm[:, 1:-1].all() and not rm[:, 0].any() and not rm[:, -1].any()
