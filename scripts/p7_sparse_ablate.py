"""P7: does the next layer rebuild the erased concept from the row itself, or from
its neighbours?

P4 projected the class-mean plane out of every residue row at L* during the forward
pass and found the next layer's rows still linearly decodable (refit macro-F1 0.56 at
layer 30 vs 0.81 unablated). P6 showed the single ablated row still carries the
concept in its shape (QDA 0.805), so no neighbours are needed in principle. Here the
question is tested directly: the plane is projected out at a random fraction p of
the residue positions of each held-out chain (the same draw for every p, so the
p = 0.1 set is inside the p = 0.5 set, inside the p = 0.9 set), and the read-outs at
ablated rows are split by how many of their sequence neighbours were ablated too.
If the rebuild at layer 30 comes from the row itself, ablated rows with intact
neighbours and ablated rows with ablated neighbours read the same; if it comes
from the neighbours, intact neighbours help.

Conditions: none; dm_sub at every position (P4's single/dm_sub); dm_sub at a
fraction p of positions; random_sub at the same p = 0.5 positions (control).
Read-outs per condition and row group (all / ablated / intact, by ablated
neighbours within +-1 and +-3): the model's own masked PLL and top-1 relative to
the unablated run; decodability at L* and the last layer by the fold's original
probe, by a probe refit on every row of the condition (P4's refit), and by a probe
refit on the ablated rows only.

Outputs in results/p7/<recording>[_limitN][_<tag>]/: metrics.json, summary.md, per_residue.npz

Usage:
    .venv/bin/python scripts/p7_sparse_ablate.py
    .venv/bin/python scripts/p7_sparse_ablate.py --limit 12 --fractions 0.5 --n-boot 50   # quick check
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.ablate import AblationContext, orthonormal_rows, project_out, random_unit  # noqa: E402
from esm2_probe.data import CLASSES, read_jsonl  # noqa: E402
from esm2_probe.probe import (cv_probe, diff_in_means, fit_lr, grouped_bootstrap_ci, grouped_bootstrap_mean,  # noqa: E402
                              grouped_folds, metrics, subsample)
from esm2_probe.reader import Recorder, encode, layer_modules, load_model, masked_readout  # noqa: E402


class SparseContext(AblationContext):
    """Residue mask of the current batch plus the positions (one chain, [T] bool)
    where the hook is allowed to edit."""

    def __init__(self) -> None:
        super().__init__()
        self.edit_mask: torch.Tensor | None = None


def make_sparse_hook(U, mu, ctx: SparseContext, device):
    u_t = torch.tensor(np.asarray(U, dtype=np.float32), device=device)
    mu_t = torch.tensor(np.asarray(mu, dtype=np.float32), device=device)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        mask = ctx.residue_mask & ctx.edit_mask[None, :]
        new = project_out(h, u_t, mu_t, mask)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    return hook


def neighbour_counts(chosen: np.ndarray, pos: np.ndarray, r: int) -> np.ndarray:
    """For each sequence position in `pos`, the number of ablated positions within +-r
    (self excluded; positions beyond the chain ends count as intact)."""
    pad = np.concatenate([np.zeros(r, dtype=int), chosen.astype(int), np.zeros(r, dtype=int)])
    cs = np.concatenate([[0], np.cumsum(pad)])
    return (cs[pos + 2 * r + 1] - cs[pos]) - chosen[pos].astype(int)


def fit_planes(acts, y, g, folds, L, n_train, seed):
    """Per fold at layer L: the class-mean plane (P4's dm_sub, same training rows),
    a random rank-2 plane, the training mean, and the fold's unablated probe."""
    out = {}
    for f in range(int(folds.max()) + 1):
        tr = np.where(folds != f)[0]
        fit_idx = subsample(tr, n_train, seed + 100 + f)
        X = np.asarray(acts[L][fit_idx], dtype=np.float32)
        dm, mu = diff_in_means(X, y[fit_idx])
        W = X.shape[1]
        out[f] = {"mu": mu, "dm_sub": orthonormal_rows(dm[:2]),
                  "random_sub": orthonormal_rows(np.stack([random_unit(W, 7000 + 1000 * f + L), random_unit(W, 8000 + 1000 * f + L)]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    n_layers, width = acts.shape[0] - 1, acts.shape[2]
    tbl = np.load(src / "residues.npz")
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L_star = min(L for L, c in counts.items() if c == max(counts.values()))
    record = sorted({L_star, n_layers})
    p3_C = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}
    C_fix = {L: (collections.Counter(p3_C[L]).most_common(1)[0][0] if L in p3_C else 1e-2) for L in record}

    y_all, g_all = tbl["label"].astype(int), tbl["protein_idx"]
    folds_all = grouped_folds(g_all, 5, args.seed)
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g_all, minlength=len(proteins)), out=starts[1:])
    chain_ids = list(range(len(proteins)))[: args.limit] if args.limit else list(range(len(proteins)))
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chain_ids])
    y, g, folds = y_all[rows], g_all[rows], folds_all[rows]
    lab_pos = {i: tbl["position"][starts[i]:starts[i + 1]] for i in chain_ids}
    out_dir = ROOT / "results" / "p7" / (args.recording + (f"_limit{args.limit}" if args.limit else "")
                                         + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    planes = fit_planes(acts, y_all, g_all, folds_all, L_star, args.n_train, args.seed)
    probes = {}
    for f in range(5):
        tr = np.where(folds_all != f)[0]
        fit_idx = subsample(tr, args.n_train, args.seed + 100 + f)
        probes[f] = {L: fit_lr(np.asarray(acts[L][fit_idx], dtype=np.float32), y_all[fit_idx],
                               p3_C[L][f] if L in p3_C else 1e-2) for L in record}
    print(f"L* = {L_star}; record layers {record}; planes and probes fit in {time.time() - t0:.0f}s", flush=True)

    # one uniform draw per chain position: the ablated set at fraction p is {u < p}
    draws = {i: np.random.default_rng(args.seed * 1_000_003 + 17 * i).random(len(proteins[i].sequence)) for i in chain_ids}

    tok, model, device = load_model()
    ctx = SparseContext()
    mods = layer_modules(model)
    true_tok = np.zeros(len(y), dtype=np.int64)

    def run_condition(kind, p):
        """Returns logp [n], top1 [n], A {layer: [n, W]}, ablated [n] bool, nb1 [n], nb3 [n]."""
        logp, top1 = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
        A = {L: np.zeros((len(y), width), dtype=np.float16) for L in record}
        abl, nb1, nb3 = np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
        for f in range(5):
            handle = mods[L_star].register_forward_hook(make_sparse_hook(planes[f][kind], planes[f]["mu"], ctx, device)) if kind else None
            try:
                for i in chain_ids:
                    if folds_all[starts[i]] != f:
                        continue
                    seq = proteins[i].sequence
                    n = len(seq)
                    sel = np.where(g == i)[0]
                    chosen = draws[i] < p if kind else np.zeros(n, dtype=bool)
                    em = torch.zeros(n + 2, dtype=torch.bool, device=device)
                    em[1:n + 1] = torch.tensor(chosen, device=device)
                    ctx.edit_mask = em
                    abl[sel], nb1[sel], nb3[sel] = chosen[lab_pos[i]], neighbour_counts(chosen, lab_pos[i], 1), neighbour_counts(chosen, lab_pos[i], 3)
                    r = masked_readout(tok, model, device, seq, args.k, ctx=ctx)
                    logp[sel], top1[sel] = r.logp_true[lab_pos[i]], r.top1[lab_pos[i]]
                    batch = encode(tok, [seq], device)
                    ctx.residue_mask = batch.residue_mask
                    rec = Recorder(model, record)
                    try:
                        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                    finally:
                        rec.remove()
                    for L in record:
                        A[L][sel] = rec.out[L][0, 1:n + 1][lab_pos[i]].to(torch.float16).cpu().numpy()
                    if not kind:
                        true_tok[sel] = batch.input_ids[0, 1:n + 1][lab_pos[i]].cpu().numpy()
            finally:
                if handle is not None:
                    handle.remove()
        return logp, top1, A, abl, nb1, nb3

    def groups(abl, nb1, nb3):
        return {"all": np.ones(len(y), dtype=bool), "ablated": abl, "intact": ~abl,
                "ablated, 0 of +-1 ablated": abl & (nb1 == 0), "ablated, 2 of +-1 ablated": abl & (nb1 == 2),
                "ablated, 0 of +-3 ablated": abl & (nb3 == 0), "ablated, 6 of +-3 ablated": abl & (nb3 == 6),
                "intact, 0 of +-1 ablated": ~abl & (nb1 == 0), "intact, 2 of +-1 ablated": ~abl & (nb1 == 2)}

    def mean_block(values, grp):
        return {name: ([int(m.sum())] + list(grouped_bootstrap_mean(values[m], g[m], n_boot=args.n_boot))) if m.sum() else None
                for name, m in grp.items()}

    def probe_block(yhat, m):
        if m.sum() < 50:
            return None
        r = metrics(y[m], yhat[m])
        ci = grouped_bootstrap_ci(y[m], yhat[m], g[m], n_boot=args.n_boot)
        return {"n": int(m.sum()), **r, "accuracy_ci": list(ci["accuracy"]), "macro_f1_ci": list(ci["macro_f1"])}

    def decodability(A, abl, grp):
        out = {}
        for L in record:
            X = A[L].astype(np.float32)
            orig = np.zeros(len(y), dtype=int)
            for f in range(5):
                sc, clf = probes[f][L]
                te = folds == f
                if te.any():
                    orig[te] = clf.predict(sc.transform(X[te]))
            refit = cv_probe(X, y, g, folds, n_train=args.n_train, seed=args.seed, C=C_fix[L])["yhat"]
            readers = {"original_probe": orig, "refit_all_rows": refit}
            if abl.any() and abl.sum() >= 500:
                ia = np.where(abl)[0]
                refit_abl = np.full(len(y), -1, dtype=int)
                refit_abl[ia] = cv_probe(X[ia], y[ia], g[ia], folds[ia], n_train=args.n_train, seed=args.seed, C=C_fix[L])["yhat"]
                readers["refit_ablated_rows"] = refit_abl
            out[str(L)] = {rd: {name: probe_block(yh, m & (yh >= 0)) for name, m in grp.items()} for rd, yh in readers.items()}
        return out

    conditions = [("none", 0.0), ("dm_sub", 1.0)] + [("dm_sub", p) for p in args.fractions] + [("random_sub", 0.5)]
    results = {"recording": args.recording, "L_star": L_star, "record_layers": record, "k": args.k, "fractions": args.fractions,
               "n_chains": len(chain_ids), "n_residues": int(len(y)), "conditions": {}}
    per_res = {"y": y, "g": g}
    base = None
    for kind, p in conditions:
        name = "none" if kind == "none" else f"{kind}/p{p:g}"
        logp, top1, A, abl, nb1, nb3 = run_condition(None if kind == "none" else kind, p)
        if base is None:
            base = (logp, top1)
            per_res["true_tok"] = true_tok
        grp = groups(abl, nb1, nb3)
        chk = None
        if kind != "none":
            chk = max(float(np.abs((A[L_star][(folds == f) & abl].astype(np.float32) - planes[f]["mu"]) @ planes[f][kind].T).max())
                      for f in range(5) if ((folds == f) & abl).any())
        results["conditions"][name] = {
            "kind": kind, "fraction": p, "share_ablated": float(abl.mean()),
            "delta_pll": mean_block(logp - base[0], grp),
            "top1_changed": mean_block((top1 != base[1]).astype(float), grp),
            "top1_acc": mean_block((top1 == true_tok).astype(float), grp),
            "max_abs_removed_coordinate_at_ablated_rows": chk,
            "decodability": decodability(A, abl, grp)}
        per_res[f"logp_{name}"], per_res[f"top1_{name}"], per_res[f"ablated_{name}"] = logp, top1, abl
        per_res[f"nb1_{name}"], per_res[f"nb3_{name}"] = nb1, nb3
        d = results["conditions"][name]["decodability"][str(n_layers)]
        rd = d.get("refit_ablated_rows") or d["refit_all_rows"]
        line = "; ".join(f"{k2}: {v['macro_f1']:.3f} (n {v['n']})" for k2, v in rd.items() if v)
        print(f"{name}: share ablated {abl.mean():.2f}; layer {n_layers} refit on ablated rows -> {line}; {time.time() - t0:.0f}s", flush=True)
        if device == "mps":
            torch.mps.empty_cache()

    results["seconds"] = time.time() - t0
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez(out_dir / "per_residue.npz", **per_res)

    # --- summary ---
    gnames = list(groups(np.zeros(1, bool), np.zeros(1, int), np.zeros(1, int)).keys())
    lines = [f"# P7 sparse-position ablation, `{args.recording}`: L* = {L_star}, {len(chain_ids)} held-out chains, {len(y)} labelled residues", "",
             "The class-mean plane (dm_sub) is projected out at L* at a random fraction p of each chain's residue positions "
             "(same draw for every p; random_sub = a random rank-2 plane at the same p = 0.5 positions). Rows are grouped by "
             "whether the row itself was ablated and how many of its sequence neighbours (within +-1 or +-3) were ablated.", "",
             "## Decodability of the next layer at ablated rows", "",
             f"Macro-F1 of a probe refit on the ablated rows only (cv over the same chain-grouped folds) at layer {n_layers}, "
             f"and at layer {L_star} [95% CI, chains resampled]. For p = 1 this is P4's refit.", "",
             "| condition | row group | n | layer %d | layer %d |" % (n_layers, L_star), "|---|---|---|---|---|"]

    def cell(b):
        return "—" if b is None else f"{b['macro_f1']:.3f} [{b['macro_f1_ci'][0]:.3f}, {b['macro_f1_ci'][1]:.3f}]"

    for name, r in results["conditions"].items():
        for gname in gnames:
            d_last, d_star = r["decodability"][str(n_layers)], r["decodability"][str(L_star)]
            rd = "refit_ablated_rows" if "refit_ablated_rows" in d_last else "refit_all_rows"
            b = d_last[rd].get(gname)
            if b is None or (gname.startswith("intact") and rd == "refit_ablated_rows"):
                continue
            lines.append(f"| {name} | {gname} | {b['n']} | {cell(b)} | {cell(d_star[rd].get(gname))} |")
    lines += ["", "## All readers, all rows", "", "Macro-F1 at layer %d: original probe | refit on all rows | refit on ablated rows." % n_layers, "",
              "| condition | row group | original | refit all | refit ablated |", "|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        d = r["decodability"][str(n_layers)]
        for gname in gnames:
            b = d["refit_all_rows"].get(gname)
            if b is None:
                continue
            ra = d.get("refit_ablated_rows", {}).get(gname)
            lines.append(f"| {name} | {gname} | {cell(d['original_probe'][gname])} | {cell(b)} | {cell(ra)} |")
    lines += ["", "## The model's own output", "",
              "Change in masked log-likelihood of the true residue vs the unablated run (nats, mean [95% CI]) and top-1 accuracy, by row group.", "",
              "| condition | row group | n | ΔPLL | top-1 acc |", "|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        for gname in gnames:
            b, a = r["delta_pll"].get(gname), r["top1_acc"].get(gname)
            if b is None:
                continue
            lines.append(f"| {name} | {gname} | {b[0]} | {b[1]:+.3f} [{b[2]:+.3f}, {b[3]:+.3f}] | {100 * a[1]:.1f}% |")
    lines += ["", f"- projection check: largest |removed coordinate| left at ablated rows = "
              f"{max((v['max_abs_removed_coordinate_at_ablated_rows'] or 0) for v in results['conditions'].values()):.4f}",
              f"- {results['seconds']:.0f}s on {device}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
