"""P2: run ESM-2 over every protein once and record the residual stream at every
layer for every labelled residue. Local, no API.

Outputs in results/p2/<model_tag>/:
    acts.npy       float16 [n_layers + 1, n_labelled_residues, width]; layer-major so
                   one layer is one contiguous block (index 0 = embedding output)
    residues.npz   protein_idx, position, label (0 H / 1 E / 2 C), aa, in the same order
    config.json, summary.md

Usage:
    uv run python scripts/p2_read.py
    uv run python scripts/p2_read.py --limit 20     # quick check (writes to a *_limit20 folder)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.data import read_jsonl, residue_table  # noqa: E402
from esm2_probe.reader import MODEL_ID, encode, length_batches, load_model, residuals_for_batch  # noqa: E402


def model_tag(model_id: str) -> str:
    return model_id.split("/")[-1].replace("_UR50D", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=None, help="first N proteins only")
    ap.add_argument("--token-budget", type=int, default=8192, help="max batch_size * max_len per forward pass")
    args = ap.parse_args()

    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    if args.limit:
        proteins = proteins[: args.limit]
    table = residue_table(proteins)
    n_res = len(table["label"])
    out_dir = ROOT / "results" / "p2" / (model_tag(args.model) + (f"_limit{args.limit}" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model, device = load_model(args.model)
    n_layers, width = model.config.num_hidden_layers, model.config.hidden_size
    acts = np.lib.format.open_memmap(out_dir / "acts.npy", mode="w+", dtype=np.float16, shape=(n_layers + 1, n_res, width))
    # row offsets of each protein's labelled residues in the flat table
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(table["protein_idx"], minlength=len(proteins)), out=starts[1:])
    labelled_pos = [table["position"][starts[i]:starts[i + 1]] for i in range(len(proteins))]

    lengths = [len(p.sequence) for p in proteins]
    batches = length_batches(lengths, args.token_budget)
    t0, done = time.time(), 0
    norms = np.zeros(n_layers + 1)
    for b_i, idx in enumerate(batches):
        batch = encode(tok, [proteins[i].sequence for i in idx], device)
        per_seq = residuals_for_batch(model, batch)  # list of [len, L+1, W] float16
        for i, A in zip(idx, per_seq):
            sel = A[labelled_pos[i]]  # [n_lab_i, L+1, W]
            acts[:, starts[i]:starts[i + 1], :] = np.transpose(sel, (1, 0, 2))
            norms += np.linalg.norm(sel.astype(np.float32), axis=-1).sum(0)
        done += len(idx)
        if (b_i + 1) % 5 == 0 or b_i + 1 == len(batches):
            print(f"batch {b_i + 1}/{len(batches)}, {done}/{len(proteins)} proteins, {time.time() - t0:.0f}s", flush=True)
        if device == "mps":
            torch.mps.empty_cache()
    acts.flush()
    np.savez(out_dir / "residues.npz", **table)
    elapsed = time.time() - t0
    cfg = {"model": args.model, "device": device, "n_proteins": len(proteins), "n_labelled_residues": n_res,
           "acts_shape": [n_layers + 1, n_res, width], "acts_dtype": "float16",
           "layer_index_note": "0 = embedding output, k = output of transformer layer k (raw residual, no final LayerNorm)",
           "positions": "labelled residues only; <cls>/<eos> never recorded", "token_budget": args.token_budget,
           "seconds": elapsed, "mean_residual_norm_by_layer": (norms / n_res).tolist()}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    lines = [f"# P2 recording, `{args.model}`", "",
             f"- {len(proteins)} proteins, {n_res} labelled residues, {n_layers + 1} layers x {width} width, "
             f"{acts.nbytes / 1e9:.2f} GB float16, {elapsed:.0f}s on {device}",
             "- mean residual norm by layer: " + ", ".join(f"{v:.0f}" for v in norms / n_res), ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    # checks: no NaN in a sample, every residue row written
    sample = acts[:, ::max(1, n_res // 2000), :].astype(np.float32)
    assert np.isfinite(sample).all(), "non-finite activations"
    assert np.abs(acts[5, -1]).sum() > 0, "last residue row empty"
    print(f"checks passed; wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
