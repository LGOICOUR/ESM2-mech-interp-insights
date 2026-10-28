"""Check: how far is the k-pass masked read-out (P4's default, k = 8) from the exact
pseudo-log-likelihood that masks one residue per pass? Runs both on a few chains and
writes results/p2/<recording>/masking_check.json.

Usage:
    uv run python scripts/check_masking.py [--n-chains 6] [--k 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.data import read_jsonl  # noqa: E402
from esm2_probe.reader import load_model, masked_readout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--n-chains", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(proteins), size=args.n_chains, replace=False).tolist())
    tok, model, device = load_model()
    rows = []
    for i in pick:
        seq = proteins[i].sequence
        fast = masked_readout(tok, model, device, seq, k=args.k)
        exact = masked_readout(tok, model, device, seq, k=len(seq))  # one residue per pass
        d = fast.logp_true - exact.logp_true
        rows.append({"chain": proteins[i].id, "length": len(seq), "pll_k": float(fast.logp_true.mean()),
                     "pll_exact": float(exact.logp_true.mean()), "mean_diff": float(d.mean()),
                     "mean_abs_diff": float(np.abs(d).mean()), "top1_agree": float(np.mean(fast.top1 == exact.top1))})
        print(rows[-1], flush=True)
    summary = {"k": args.k, "chains": rows,
               "mean_diff_k_minus_exact": float(np.mean([r["mean_diff"] for r in rows])),
               "mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in rows])),
               "top1_agree": float(np.mean([r["top1_agree"] for r in rows]))}
    out = ROOT / "results" / "p2" / args.recording / "masking_check.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"k={args.k} vs exact: mean PLL difference {summary['mean_diff_k_minus_exact']:+.4f} nats, "
          f"mean |diff| per residue {summary['mean_abs_diff']:.4f}, top-1 agreement {100 * summary['top1_agree']:.1f}%")


if __name__ == "__main__":
    main()
