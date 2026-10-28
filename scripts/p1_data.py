"""P1: fetch the labelled proteins and write one clean file. Network: Hugging Face only.

Source: NetSurfP training/test bundle (DTU Health Tech), CB513 and TS115 splits,
as mirrored in the Hugging Face dataset `lamm-mit/protein-secondary-structure-netsurfp`
(Q3 labels H/E/C, '.' = unlabelled). The exact dataset commit is recorded in
results/p1/config.json.

Outputs: results/p1/proteins.jsonl (id, split, sequence, labels), config.json, summary.md.

Usage:
    uv run python scripts/p1_data.py
    uv run python scripts/p1_data.py --max-proteins 40   # small set for checks
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.data import CLASSES, HF_FILES, HF_REPO, dedupe, filter_proteins, parse_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["ts115", "cb513"], choices=sorted(HF_FILES))
    ap.add_argument("--max-proteins", type=int, default=None, help="seeded random subset after cleaning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sha = HfApi().dataset_info(HF_REPO).sha
    proteins, per_split = [], {}
    for split in args.splits:
        path = hf_hub_download(HF_REPO, HF_FILES[split], repo_type="dataset", revision=sha, local_dir=raw)
        rows = parse_jsonl(Path(path), split)
        per_split[split] = {"rows": len(rows), "residues": sum(len(p.sequence) for p in rows),
                            "labelled_residues": sum(p.n_labelled for p in rows)}
        proteins += rows
    deduped, rep_d = dedupe(proteins)
    kept, rep_f = filter_proteins(deduped)
    if args.max_proteins and len(kept) > args.max_proteins:
        idx = np.sort(np.random.default_rng(args.seed).choice(len(kept), size=args.max_proteins, replace=False))
        kept = [kept[i] for i in idx]

    out = ROOT / "results" / "p1"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept, out / "proteins.jsonl")
    lengths = np.array([len(p.sequence) for p in kept])
    counts = collections.Counter(ch for p in kept for ch in p.labels if ch != ".")
    n_lab = sum(counts.values())
    class_frac = {c: counts[c] / n_lab for c in CLASSES}
    cfg = {"hf_repo": HF_REPO, "hf_revision": sha, "files": {s: HF_FILES[s] for s in args.splits},
           "per_split_raw": per_split, "dedupe": rep_d, "filter": rep_f,
           "max_proteins": args.max_proteins, "seed": args.seed,
           "n_proteins": len(kept), "n_residues": int(lengths.sum()), "n_labelled_residues": n_lab,
           "class_fraction": class_frac, "length": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                                                    "mean": float(lengths.mean()), "max": int(lengths.max())},
           "by_split": {s: sum(p.split == s for p in kept) for s in args.splits}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    lines = ["# P1 data", "",
             f"Source: `{HF_REPO}` @ `{sha[:12]}`, files {', '.join(HF_FILES[s] for s in args.splits)} "
             "(NetSurfP bundle: DSSP three-state labels; '.' = unlabelled).", ""]
    for s, r in per_split.items():
        lines.append(f"- {s}: {r['rows']} rows, {r['residues']} residues, {r['labelled_residues']} labelled")
    lines += [f"- de-duplication by sequence: {rep_d['n_in']} rows -> {rep_d['n_out']} chains "
              f"({rep_d['n_duplicate_rows']} duplicate rows merged, {rep_d['n_conflicting_chains_dropped']} chains dropped for label conflicts)",
              f"- filter: {rep_f['n_too_long']} chains over the length budget, {rep_f['n_without_labels']} without labels",
              f"- kept: **{len(kept)} chains** ({cfg['by_split']}), {int(lengths.sum())} residues, **{n_lab} labelled**",
              f"- class fraction among labelled residues: " + ", ".join(f"{c} {class_frac[c]:.3f}" for c in CLASSES),
              f"- chain length: min {lengths.min()}, median {np.median(lengths):.0f}, mean {lengths.mean():.0f}, max {lengths.max()}", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    # checks
    assert all(len(p.sequence) == len(p.labels) for p in kept)
    assert len({p.sequence for p in kept}) == len(kept), "duplicate sequences survived"
    assert abs(sum(class_frac.values()) - 1) < 1e-9
    print(f"checks passed; wrote {out.relative_to(ROOT)}/proteins.jsonl")


if __name__ == "__main__":
    main()
