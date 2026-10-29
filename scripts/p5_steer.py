"""P5: is the class-mean direction *sufficient*? Steer coil residues toward helix
or strand during the forward pass and read what the model predicts there.

P4 removed directions and measured what the model's predictions lost
(necessity). Here a direction is added to the residual stream of every labelled
coil residue at layer L*, in doses, and the model's masked prediction at those
residues is scored by amino-acid propensity: does the predicted distribution
move toward helix-forming residues when the helix direction is added, toward
strand-formers for the strand direction, and nowhere for a random direction of
the same norm? Dose unit: the gap between the class mean and the coil mean
along the direction on the fold's training residues, so d = 1 moves the coil
residues' mean coordinate onto the class mean's.

Directions, fit per fold at L*: dm_H / dm_E (P4's class-mean directions);
dm_H_strat / dm_E_strat (within-amino-acid class-mean directions, the control
for composition: a plain class-mean direction partly encodes that helices are
rich in A/E/L/Q); random_H / random_E (a random unit direction scaled to the
norm dm_H or dm_E has at that dose).

Read-outs at every coil residue, from the pass in which it is masked: the
predicted distribution over the 20 amino acids scored with the fold's
training-chain propensities (expected log-propensity S_H, S_E, S_C and the
fraction of the helix-vs-coil or strand-vs-coil gap closed; mass on helix /
strand / coil formers), the KL divergence from the unsteered prediction, the
change in log-likelihood of the true residue and in the top-1 prediction, the
mean change in predicted probability per amino acid; side effects at helix and
strand residues of the same chains; and the fold's original L* and last-layer
probes applied to the steered activations (what share of coil residues they
now call helix / strand).

Outputs in results/p5/<recording>[_limitN][_<tag>]/: metrics.json, summary.md, per_residue.npz

Usage:
    uv run python scripts/p5_steer.py
    uv run python scripts/p5_steer.py --limit 12 --kinds dm_H random_H --doses 1 --n-boot 50   # quick check
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
sys.path.insert(0, str(ROOT / "scripts"))
from esm2_probe.ablate import random_unit  # noqa: E402
from esm2_probe.data import AMINO_ACIDS, CLASSES, read_jsonl  # noqa: E402
from esm2_probe.probe import grouped_bootstrap_mean, grouped_folds, subsample  # noqa: E402
from esm2_probe.reader import Recorder, encode, load_model, masked_readout  # noqa: E402
from esm2_probe.steer import (N_AA, SteerContext, aa_distribution, attach_steer, chain_bootstrap, class_gap,  # noqa: E402
                              composition_subspace, former_mass, formers, kl_divergence, propensity_scores,
                              propensity_table, stratified_diff_in_means, subspace_share)
from p4_ablate import by_class, fit_directions  # noqa: E402

KINDS = ["dm_H", "dm_E", "dm_H_strat", "dm_E_strat", "random_H", "random_E"]
CLASS_OF = {"dm_H": 0, "dm_E": 1, "dm_H_strat": 0, "dm_E_strat": 1, "random_H": 0, "random_E": 1}
FITTED = ["dm_H", "dm_E", "dm_H_strat", "dm_E_strat"]
REF_OF = {"random_H": "dm_H", "random_E": "dm_E"}  # the random directions borrow the class-mean direction's dose unit


def fit_steering(acts, y, g, aa, folds, L, n_layers, n_train, seed, C_of, target):
    """Per fold: unit directions of every kind at layer L with their dose units (gaps),
    the training-chain propensity table, geometry diagnostics, and the fold's probes
    at L and the last layer (from P4's fit_directions, for the probe check)."""
    cache, probes = fit_directions(acts, y, g, folds, [L, n_layers], n_train, seed, C_of)
    W = acts.shape[2]
    dirs, gaps, tables, geom = {}, {}, {}, {}
    for f in range(int(folds.max()) + 1):
        tr = np.where(folds != f)[0]
        fit_idx = subsample(tr, n_train, seed + 100 + f)  # the residues P3/P4 fit on
        X, yy, a = np.asarray(acts[L][fit_idx], dtype=np.float32), y[fit_idx], aa[fit_idx]
        d = {"dm_H": cache[f][L]["dm_H"], "dm_E": cache[f][L]["dm_E"],
             "dm_H_strat": stratified_diff_in_means(X, yy, a, 0)[0], "dm_E_strat": stratified_diff_in_means(X, yy, a, 1)[0],
             "random_H": cache[f][L]["random"], "random_E": random_unit(W, 3000 + 1000 * f + L)}
        gp = {k: class_gap(X, yy, d[k], CLASS_OF[k], target) for k in FITTED}
        for k, ref in REF_OF.items():
            gp[k] = gp[ref]
        Q = composition_subspace(X, a)
        dirs[f], gaps[f], tables[f] = d, gp, propensity_table(aa[tr], y[tr])
        geom[f] = {"gap": gp,
                   "cos_strat": {"H": float(d["dm_H"] @ d["dm_H_strat"]), "E": float(d["dm_E"] @ d["dm_E_strat"])},
                   "composition_share": {k: subspace_share(d[k], Q) for k in KINDS},
                   "within_class_sd": {k: float(np.sqrt(np.mean([((X[yy == c] - X[yy == c].mean(0)) @ d[k]).var()
                                                                for c in range(3)]))) for k in FITTED},
                   "mean_residual_norm": float(np.linalg.norm(X, axis=1).mean())}
    return dirs, gaps, tables, geom, probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--doses", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--target-class", default="C", choices=list(CLASSES), help="residues steered and read (default coil)")
    ap.add_argument("--k", type=int, default=8, help="masking passes per chain (1/k of residues masked per pass)")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="first N chains only (check runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="suffix for the output folder")
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts = np.load(src / "acts.npy", mmap_mode="r")
    n_layers, width = acts.shape[0] - 1, acts.shape[2]
    tbl = np.load(src / "residues.npz")
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    p3 = json.loads((ROOT / "results" / "p3" / args.recording / "metrics.json").read_text())
    counts = collections.Counter(p3["chosen_layer_per_fold"])
    L_star = min(L for L, c in counts.items() if c == max(counts.values()))
    record_layers = sorted({L_star, n_layers})
    p3_C = {r["layer"]: r["C_per_fold"] for r in p3["per_layer"]}
    target = CLASSES.index(args.target_class)

    def C_of(L, f):
        return p3_C[L][f] if L in p3_C else 1e-2

    # residue bookkeeping, as in P4 (the full table defines folds and directions; --limit restricts the chains read)
    y_all, g_all, aa_all = tbl["label"].astype(int), tbl["protein_idx"], tbl["aa"].astype(int)
    folds_all = grouped_folds(g_all, 5, args.seed)
    starts = np.zeros(len(proteins) + 1, dtype=np.int64)
    np.cumsum(np.bincount(g_all, minlength=len(proteins)), out=starts[1:])
    chain_ids = list(range(len(proteins)))[: args.limit] if args.limit else list(range(len(proteins)))
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in chain_ids])
    y, g, folds = y_all[rows], g_all[rows], folds_all[rows]
    lab_pos = {i: tbl["position"][starts[i]:starts[i + 1]] for i in chain_ids}
    tgt_pos = {i: lab_pos[i][y_all[starts[i]:starts[i + 1]] == target] for i in chain_ids}  # steered positions per chain
    is_tgt = y == target
    out_dir = ROOT / "results" / "p5" / (args.recording + (f"_limit{args.limit}" if args.limit else "")
                                         + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dirs, gaps, tables, geom, probes = fit_steering(acts, y_all, g_all, aa_all, folds_all, L_star, n_layers,
                                                    args.n_train, args.seed, C_of, target)
    form0 = {c: "".join(AMINO_ACIDS[a] for a in range(N_AA) if formers(tables[0])[ci, a]) for ci, c in enumerate(CLASSES)}
    geo_line = "; ".join(f"{k}: gap {np.mean([geom[f]['gap'][k] for f in geom]):.1f}"
                         + (f", within-class sd {np.mean([geom[f]['within_class_sd'][k] for f in geom]):.1f}" if k in FITTED else "")
                         + f", composition share {np.mean([geom[f]['composition_share'][k] for f in geom]):.2f}" for k in KINDS)
    cos_line = ", ".join(f"cos(dm_{c}, dm_{c}_strat) {np.mean([geom[f]['cos_strat'][c] for f in geom]):.3f}" for c in ("H", "E"))
    print(f"L* = {L_star}; target class {args.target_class}; mean residual norm at L* {np.mean([geom[f]['mean_residual_norm'] for f in geom]):.0f}; "
          f"{geo_line}; {cos_line}; fold-0 formers {form0}; fit in {time.time() - t0:.0f}s", flush=True)
    for f in geom:  # every fitted direction's dose unit is positive: adding it moves toward the class
        assert all(geom[f]["gap"][k] > 0 for k in FITTED), geom[f]["gap"]

    tok, model, device = load_model()
    aa_ids = tok.convert_tokens_to_ids(list(AMINO_ACIDS[:N_AA]))
    assert len(set(aa_ids)) == N_AA and tok.unk_token_id not in aa_ids, "tokenizer lacks an amino-acid token"
    sctx = SteerContext()
    true_tok = np.zeros(len(y), dtype=np.int64)

    def run_condition(vec_by_fold, record):
        """vec_by_fold[f] = vector [W] added at L* (None = unsteered). Returns logp_true [n],
        top1 [n], log-probs at the 20 amino acids [n, 20], recorded activations {layer: [n, W] float16}."""
        logp, top1 = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
        lpa = np.zeros((len(y), N_AA), dtype=np.float32)
        A = {L: np.zeros((len(y), width), dtype=np.float16) for L in record}
        for f in range(5):
            handles = attach_steer(model, {L_star: vec_by_fold[f]}, sctx, device) if vec_by_fold is not None else []
            try:
                for i in chain_ids:
                    if folds_all[starts[i]] != f:
                        continue
                    seq = proteins[i].sequence
                    sel = np.where(g == i)[0]
                    sctx.positions = tgt_pos[i]
                    r = masked_readout(tok, model, device, seq, args.k, token_ids=aa_ids)
                    logp[sel], top1[sel], lpa[sel] = r.logp_true[lab_pos[i]], r.top1[lab_pos[i]], r.logp_tokens[lab_pos[i]]
                    batch = encode(tok, [seq], device)
                    rec = Recorder(model, record)
                    try:
                        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                    finally:
                        rec.remove()
                    for L in record:
                        A[L][sel] = rec.out[L][0, 1:len(seq) + 1][lab_pos[i]].to(torch.float16).cpu().numpy()
                    if vec_by_fold is None:
                        true_tok[sel] = batch.input_ids[0, 1:len(seq) + 1][lab_pos[i]].cpu().numpy()
            finally:
                for h in handles:
                    h.remove()
                sctx.positions = None
        return logp, top1, lpa, A

    def scores(lpa):
        """Expected log-propensity [n, 3] and former mass [n, 3] under each residue's fold table."""
        S, M = np.zeros((len(y), 3), dtype=np.float32), np.zeros((len(y), 3), dtype=np.float32)
        for f in range(5):
            te = folds == f
            if te.any():
                S[te], M[te] = propensity_scores(lpa[te], tables[f]), former_mass(lpa[te], tables[f])
        return S, M

    def probe_calls(A):
        """Share of target residues the fold's original probe calls each class, per recorded layer."""
        out = {}
        for L in record_layers:
            yhat = np.zeros(len(y), dtype=int)
            for f in range(5):
                te = folds == f
                if te.any():
                    sc, clf = probes[f][L]
                    yhat[te] = clf.predict(sc.transform(A[L][te].astype(np.float32)))
            out[str(L)] = {c: list(grouped_bootstrap_mean((yhat[is_tgt] == ci).astype(float), g[is_tgt], n_boot=args.n_boot))
                           for ci, c in enumerate(CLASSES)}
        return out

    # --- baseline (no steering) ---
    logp0, top10, lpa0, A0 = run_condition(None, record_layers)
    base_match = float(np.abs(A0[n_layers].astype(np.float32) - np.asarray(acts[n_layers][rows], dtype=np.float32)).max())
    print(f"baseline read in {time.time() - t0:.0f}s; max |P5 unsteered - P2 recording| at layer {n_layers} = {base_match:.3f} "
          f"(float16 round-off expected)", flush=True)
    assert base_match < 1.0, "P5's unsteered activations do not match the P2 recording"
    S0, M0 = scores(lpa0)
    gap_G = {c: list(chain_bootstrap(lambda m: m[0] - m[1], [(S0[y == ci, ci], g[y == ci]), (S0[is_tgt, ci], g[is_tgt])],
                                     n_boot=args.n_boot, seed=args.seed)) for ci, c in enumerate(CLASSES) if ci != target}
    results = {"recording": args.recording, "L_star": L_star, "target_class": args.target_class, "doses": args.doses,
               "kinds": args.kinds, "k": args.k, "n_chains": len(chain_ids), "n_residues": int(len(y)),
               "n_target_residues": int(is_tgt.sum()), "record_layers": record_layers, "amino_acids": AMINO_ACIDS[:N_AA],
               "geometry_per_fold": {str(f): geom[f] for f in geom},
               "propensity_table_per_fold": {str(f): tables[f].tolist() for f in tables}, "formers_fold0": form0,
               "baseline": {"S_by_score_class_by_true_class": {c: by_class(S0[:, ci], y, g, args.n_boot) for ci, c in enumerate(CLASSES)},
                            "former_mass_at_target": {c: list(grouped_bootstrap_mean(M0[is_tgt, ci], g[is_tgt], n_boot=args.n_boot))
                                                      for ci, c in enumerate(CLASSES)},
                            "gap_G": gap_G,
                            "pll_by_class": by_class(logp0, y, g, args.n_boot),
                            "top1_acc_by_class": by_class((top10 == true_tok).astype(float), y, g, args.n_boot),
                            "probe_calls_at_target": probe_calls(A0),
                            "p_amino_acid_at_target": aa_distribution(lpa0[is_tgt]).mean(0).tolist()},
               "conditions": {}}
    per_res = {"y": y, "g": g, "logp_base": logp0, "top1_base": top10, "true_tok": true_tok, "logp_aa_base": lpa0.astype(np.float16)}
    b = results["baseline"]
    print(f"baseline PLL {b['pll_by_class']['all'][0]:.3f}; S_H at helix / coil residues "
          f"{b['S_by_score_class_by_true_class']['H']['H'][0]:+.3f} / {b['S_by_score_class_by_true_class']['H']['C'][0]:+.3f}; "
          f"S_E at strand / coil {b['S_by_score_class_by_true_class']['E']['E'][0]:+.3f} / {b['S_by_score_class_by_true_class']['E']['C'][0]:+.3f}; "
          f"G {({c: round(v[0], 3) for c, v in gap_G.items()})}; probe calls at target "
          f"{({L: {c: round(v[0], 3) for c, v in d.items()} for L, d in b['probe_calls_at_target'].items()})}; {time.time() - t0:.0f}s", flush=True)

    # --- steering ---
    shift_checks = []
    for kind in args.kinds:
        for dose in args.doses:
            vec_by_fold = {f: (dose * gaps[f][kind] * dirs[f][kind]).astype(np.float32) for f in range(5)}
            logp, top1, lpa, A = run_condition(vec_by_fold, record_layers)
            S, M = scores(lpa)
            dS, dM, kl = S - S0, M - M0, kl_divergence(lpa, lpa0)
            # hook check at L*: target residues moved by exactly the added vector along u, other residues not at all
            for f in range(5):
                te = folds == f
                if te.any():
                    u = dirs[f][kind].astype(np.float32)
                    shift = (A[L_star][te].astype(np.float32) - A0[L_star][te].astype(np.float32)) @ u
                    expect = float(dose * gaps[f][kind])
                    shift_checks.append((float(shift[is_tgt[te]].mean() / expect), float(np.abs(shift[~is_tgt[te]]).max()) if (~is_tgt[te]).any() else 0.0))
            name = f"{kind}@{dose:g}"
            frac = {c: list(chain_bootstrap(lambda m: m[0] / (m[1] - m[2]),
                                            [(dS[is_tgt, ci], g[is_tgt]), (S0[y == ci, ci], g[y == ci]), (S0[is_tgt, ci], g[is_tgt])],
                                            n_boot=args.n_boot, seed=args.seed)) for ci, c in enumerate(CLASSES) if ci != target}
            dp = aa_distribution(lpa[is_tgt]).mean(0) - aa_distribution(lpa0[is_tgt]).mean(0)
            results["conditions"][name] = {
                "kind": kind, "dose": dose, "class": CLASSES[CLASS_OF[kind]],
                "norm_added_per_fold": [float(dose * gaps[f][kind]) for f in range(5)],
                "delta_S_by_score_class_by_true_class": {c: by_class(dS[:, ci], y, g, args.n_boot) for ci, c in enumerate(CLASSES)},
                "fraction_closed_at_target": frac,
                "delta_former_mass_at_target": {c: list(grouped_bootstrap_mean(dM[is_tgt, ci], g[is_tgt], n_boot=args.n_boot))
                                                for ci, c in enumerate(CLASSES)},
                "kl_by_class": by_class(kl, y, g, args.n_boot),
                "delta_pll_by_class": by_class(logp - logp0, y, g, args.n_boot),
                "top1_changed_by_class": by_class((top1 != top10).astype(float), y, g, args.n_boot),
                "top1_acc_by_class": by_class((top1 == true_tok).astype(float), y, g, args.n_boot),
                "probe_calls_at_target": probe_calls(A),
                "delta_p_amino_acid_at_target": dp.tolist()}
            per_res[f"logp_{name}"], per_res[f"top1_{name}"], per_res[f"logp_aa_{name}"] = logp, top1, lpa.astype(np.float16)
            r = results["conditions"][name]
            t = args.target_class
            print(f"{name}: at {t} residues dS_H {r['delta_S_by_score_class_by_true_class']['H'][t][0]:+.3f} "
                  f"dS_E {r['delta_S_by_score_class_by_true_class']['E'][t][0]:+.3f} dS_C {r['delta_S_by_score_class_by_true_class']['C'][t][0]:+.3f}; "
                  f"F {({c: round(v[0], 2) for c, v in frac.items()})}; KL {r['kl_by_class'][t][0]:.3f}; dPLL {r['delta_pll_by_class'][t][0]:+.3f}; "
                  f"probe-{L_star} calls {({c: round(v[0], 2) for c, v in r['probe_calls_at_target'][str(L_star)].items()})}; "
                  f"{time.time() - t0:.0f}s", flush=True)
            if device == "mps":
                torch.mps.empty_cache()

    ratio_lo, ratio_hi = min(s[0] for s in shift_checks), max(s[0] for s in shift_checks)
    leak = max(s[1] for s in shift_checks)
    print(f"hook check at L*: mean shift of target residues along u / intended = {ratio_lo:.3f}..{ratio_hi:.3f}; "
          f"largest |shift| of a non-target residue along u = {leak:.3f}", flush=True)
    assert 0.97 < ratio_lo and ratio_hi < 1.03, "steered residues did not move by the intended amount"
    assert leak < 1.0, "a non-target residue moved at L*"
    results["hook_check"] = {"shift_ratio_range": [ratio_lo, ratio_hi], "max_non_target_shift": leak}
    results["seconds"] = time.time() - t0
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez(out_dir / "per_residue.npz", **per_res)

    # --- summary ---
    t = args.target_class
    tname = {"H": "helix", "E": "strand", "C": "coil"}[t]

    def ci(v, fmt="+.3f"):
        return f"{v[0]:{fmt}} [{v[1]:{fmt}}, {v[2]:{fmt}}]"

    def kind_label(k):
        return {"dm_H": "class-mean helix", "dm_E": "class-mean strand", "dm_H_strat": "within-amino-acid helix",
                "dm_E_strat": "within-amino-acid strand", "random_H": "random, helix norm", "random_E": "random, strand norm"}[k]

    Sb = b["S_by_score_class_by_true_class"]
    lines = [f"# P5 steering, `{args.recording}`: L* = {L_star}, {len(chain_ids)} held-out chains, {len(y)} labelled residues, "
             f"{int(is_tgt.sum())} {tname} residues steered and read", "",
             f"Directions are added at layer {L_star} to every labelled {tname} residue during the forward pass, in doses of the "
             f"class-vs-{tname} gap along the direction. Read-out: the masked prediction at those residues (k = {args.k}), scored by "
             f"the fold's training-chain amino-acid propensities.", "",
             "## Baseline (no steering)", "",
             "Expected log-propensity of the model's masked prediction, by the residue's true class (rows) and the scoring class (columns):", "",
             "| true class | S_helix | S_strand | S_coil |", "|---|---|---|---|"]
    for c in CLASSES:
        lines.append(f"| {c} residues | " + " | ".join(ci(Sb[sc][c]) for sc in CLASSES) + " |")
    lines += ["", "- gap G (class residues' own score minus " + tname + " residues' score): "
              + "; ".join(f"G_{c} = {ci(v)}" for c, v in gap_G.items()),
              f"- fold-0 formers (positive propensity): helix {form0['H']}, strand {form0['E']}, coil {form0['C']}",
              f"- mass on formers at {tname} residues: " + ", ".join(f"{c} {v[0]:.3f}" for c, v in b["former_mass_at_target"].items()),
              f"- masked PLL {b['pll_by_class']['all'][0]:.3f} nats (H {b['pll_by_class']['H'][0]:.3f}, E {b['pll_by_class']['E'][0]:.3f}, "
              f"C {b['pll_by_class']['C'][0]:.3f}); top-1 accuracy {100 * b['top1_acc_by_class']['all'][0]:.1f}%",
              "- original probes' calls at " + tname + " residues: " + "; ".join(
                  f"layer {L}: " + ", ".join(f"{c} {100 * v[0]:.1f}%" for c, v in d.items()) for L, d in b["probe_calls_at_target"].items()),
              f"- geometry at L* (mean over folds): residual norm {np.mean([geom[f]['mean_residual_norm'] for f in geom]):.0f}; {geo_line}; {cos_line}",
              "", "## Propensity of the prediction at " + tname + " residues", "",
              "ΔS = change in expected log-propensity (nats) [95% CI, chains resampled]; F = fraction of the gap G closed; "
              "Δ mass = change in predicted probability on the class's formers.", "",
              "| direction | dose | norm added | ΔS_helix | ΔS_strand | ΔS_coil | F_helix | F_strand | Δ mass helix-formers | Δ mass strand-formers |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        d = r["delta_S_by_score_class_by_true_class"]
        lines.append(f"| {kind_label(r['kind'])} | {r['dose']:g} | {np.mean(r['norm_added_per_fold']):.0f} | "
                     + " | ".join(ci(d[c][t]) for c in CLASSES) + " | "
                     + " | ".join(ci(r["fraction_closed_at_target"][c], "+.2f") for c in ("H", "E")) + " | "
                     + " | ".join(ci(r["delta_former_mass_at_target"][c]) for c in ("H", "E")) + " |")
    lines += ["", "## Size, cost and what the probes read", "",
              f"KL = KL(p_steered ‖ p_unsteered) at {tname} residues; ΔPLL = change in log-likelihood of the true residue, by true class; "
              f"probe calls = share of {tname} residues the fold's original probe labels helix / strand at the recorded layers.", "",
              f"| direction | dose | KL at {tname} | ΔPLL {tname} | ΔPLL helix | ΔPLL strand | top-1 changed at {tname} | "
              f"probe {L_star} calls H / E | probe {n_layers} calls H / E |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["conditions"].items():
        pc = r["probe_calls_at_target"]
        lines.append(f"| {kind_label(r['kind'])} | {r['dose']:g} | {ci(r['kl_by_class'][t], '.3f')} | {ci(r['delta_pll_by_class'][t])} | "
                     f"{r['delta_pll_by_class']['H'][0]:+.3f} | {r['delta_pll_by_class']['E'][0]:+.3f} | "
                     f"{100 * r['top1_changed_by_class'][t][0]:.1f}% | "
                     f"{100 * pc[str(L_star)]['H'][0]:.0f}% / {100 * pc[str(L_star)]['E'][0]:.0f}% | "
                     f"{100 * pc[str(n_layers)]['H'][0]:.0f}% / {100 * pc[str(n_layers)]['E'][0]:.0f}% |")
    lines += ["", "## Which amino acids gain and lose", "",
              f"Mean change in predicted probability at {tname} residues (percentage points), largest gains and losses, at dose 1 where run:", ""]
    for name, r in results["conditions"].items():
        if abs(r["dose"] - 1.0) > 1e-9:
            continue
        dp = np.array(r["delta_p_amino_acid_at_target"]) * 100
        order = np.argsort(dp)
        lines.append(f"- {kind_label(r['kind'])}: gains " + ", ".join(f"{AMINO_ACIDS[i]} {dp[i]:+.1f}" for i in order[::-1][:5])
                     + "; losses " + ", ".join(f"{AMINO_ACIDS[i]} {dp[i]:+.1f}" for i in order[:5]))
    lines += ["", f"- hook check at L*: steered residues moved along u by {ratio_lo:.3f}–{ratio_hi:.3f} of the intended amount; "
              f"largest move of a non-steered residue {leak:.3f}",
              f"- {results['seconds']:.0f}s on {device}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
