"""P3: layer-wise probes for secondary structure on the P2 recording, next to the
no-model baselines. Local, no API.

Per layer: chain-grouped 5-fold CV, C tuned inside each training fold, held-out
predictions for every labelled residue; accuracy and macro-F1 with 95% CIs from
resampling chains. Headline = the layer chosen inside each training fold by its
inner-CV macro-F1 (no hindsight). Baselines: majority class, one-hot identity,
one-hot +/-3 window. A class-mean (nearest-centroid) decoder is scored at every
layer as well.

Outputs in results/p3/<recording>/: metrics.json, layer_curves.npz, yhat_by_layer.npy, summary.md

Usage:
    uv run python scripts/p3_probe.py
    uv run python scripts/p3_probe.py --layers 0 10 20 --n-boot 50   # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esm2_probe.baselines import majority_predict, onehot_predict  # noqa: E402
from esm2_probe.data import CLASSES, one_hot_window, read_jsonl  # noqa: E402
from esm2_probe.probe import cv_probe, grouped_bootstrap_ci, grouped_folds, metrics, nearest_centroid_predict, subsample  # noqa: E402


def block(y, yhat, g, n_boot, seed=0):
    m = metrics(y, yhat)
    ci = grouped_bootstrap_ci(y, yhat, g, n_boot=n_boot, seed=seed)
    return {**m, "accuracy_ci": list(ci["accuracy"]), "macro_f1_ci": list(ci["macro_f1"])}


def one_layer(acts_path, layer, y, g, folds, n_train, n_boot, seed):
    acts = np.load(acts_path, mmap_mode="r")
    X = np.asarray(acts[layer], dtype=np.float32)
    res = cv_probe(X, y, g, folds, n_train=n_train, seed=seed)
    cent = np.zeros(len(y), dtype=int)
    for f in range(int(folds.max()) + 1):
        tr = np.where(folds != f)[0]
        fit = subsample(tr, n_train, seed + 200 + f)
        cent[folds == f] = nearest_centroid_predict(X[folds == f], X[fit], y[fit])
    return {"layer": layer, "probe": block(y, res["yhat"], g, n_boot, seed), "centroid": block(y, cent, g, n_boot, seed),
            "C_per_fold": res["C_per_fold"], "inner_f1_per_fold": res["inner_f1_per_fold"], "yhat": res["yhat"].astype(np.int8)}


def fmt(b):
    return (f"{b['accuracy']:.3f} [{b['accuracy_ci'][0]:.3f}, {b['accuracy_ci'][1]:.3f}] | "
            f"{b['macro_f1']:.3f} [{b['macro_f1_ci'][0]:.3f}, {b['macro_f1_ci'][1]:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="esm2_t30_150M")
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    src = ROOT / "results" / "p2" / args.recording
    acts_path = src / "acts.npy"
    tbl = np.load(src / "residues.npz")
    y, g = tbl["label"].astype(int), tbl["protein_idx"]
    proteins = read_jsonl(ROOT / "results" / "p1" / "proteins.jsonl")
    n_layers = np.load(acts_path, mmap_mode="r").shape[0]
    layers = list(range(n_layers)) if args.layers is None else args.layers
    folds = grouped_folds(g, 5, args.seed)
    out = ROOT / "results" / "p3" / args.recording
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- baselines (same folds) ---
    base = {"majority": block(y, majority_predict(y, folds), g, args.n_boot)}
    for name, hw in (("identity", 0), ("window3", 3)):
        X = one_hot_window(proteins, {k: tbl[k] for k in tbl.files}, hw)
        base[name] = block(y, onehot_predict(X, y, folds), g, args.n_boot)
    print(f"baselines done, {time.time() - t0:.0f}s: " + "; ".join(f"{k} {v['macro_f1']:.3f}" for k, v in base.items()), flush=True)

    # --- probes, one layer per worker ---
    per = Parallel(n_jobs=args.jobs)(delayed(one_layer)(acts_path, L, y, g, folds, args.n_train, args.n_boot, args.seed) for L in layers)
    print(f"{len(layers)} layers done, {time.time() - t0:.0f}s", flush=True)
    yhat_by_layer = np.full((n_layers, len(y)), -1, dtype=np.int8)
    for r in per:
        yhat_by_layer[r["layer"]] = r.pop("yhat")

    # --- headline: layer chosen inside each training fold by inner-CV macro-F1 ---
    inner = np.array([[r["inner_f1_per_fold"][f] for f in range(5)] for r in per])  # [n_layers_run, 5]
    chosen = [layers[int(np.argmax(inner[:, f]))] for f in range(5)]
    head_yhat = np.zeros(len(y), dtype=int)
    for f in range(5):
        head_yhat[folds == f] = yhat_by_layer[chosen[f]][folds == f]
    headline = block(y, head_yhat, g, args.n_boot)
    f1_curve = np.array([r["probe"]["macro_f1"] for r in per])
    hind = layers[int(np.argmax(f1_curve))]

    metrics_out = {"recording": args.recording, "seed": args.seed, "n_train": args.n_train, "n_boot": args.n_boot,
                   "n_residues": int(len(y)), "n_proteins": int(len(set(g.tolist()))), "layers": layers,
                   "baselines": base, "per_layer": per, "chosen_layer_per_fold": chosen,
                   "headline": headline, "hindsight": {"layer": hind, "macro_f1": float(f1_curve.max())}}
    (out / "metrics.json").write_text(json.dumps(metrics_out, indent=1))
    np.save(out / "yhat_by_layer.npy", yhat_by_layer)
    np.savez(out / "layer_curves.npz", layers=np.array(layers),
             probe_acc=np.array([r["probe"]["accuracy"] for r in per]), probe_acc_ci=np.array([r["probe"]["accuracy_ci"] for r in per]),
             probe_f1=f1_curve, probe_f1_ci=np.array([r["probe"]["macro_f1_ci"] for r in per]),
             centroid_acc=np.array([r["centroid"]["accuracy"] for r in per]), centroid_f1=np.array([r["centroid"]["macro_f1"] for r in per]),
             f1_per_class=np.array([r["probe"]["f1_per_class"] for r in per]))

    lines = [f"# P3 probes, recording `{args.recording}`", "",
             f"{len(y)} labelled residues in {metrics_out['n_proteins']} chains; 5 chain-grouped folds; "
             f"accuracy [95% CI] | macro-F1 [95% CI], CIs resample chains ({args.n_boot} draws).", "",
             "| reader | accuracy | macro-F1 |", "|---|---|---|"]
    for k, label in (("majority", "majority class"), ("identity", "one-hot identity, no model"), ("window3", "one-hot +/-3 window, no model")):
        lines.append(f"| {label} | {fmt(base[k])} |")
    lines.append(f"| **probe, layer chosen inside training** (folds chose {chosen}) | {fmt(headline)} |")
    lines += ["", "| layer | probe accuracy | probe macro-F1 | F1 H / E / C | class-mean decoder F1 | C per fold |", "|---|---|---|---|---|---|"]
    for r in per:
        pc = " / ".join(f"{v:.2f}" for v in r["probe"]["f1_per_class"])
        lines.append(f"| {r['layer']} | {r['probe']['accuracy']:.3f} [{r['probe']['accuracy_ci'][0]:.3f}, {r['probe']['accuracy_ci'][1]:.3f}] "
                     f"| {r['probe']['macro_f1']:.3f} [{r['probe']['macro_f1_ci'][0]:.3f}, {r['probe']['macro_f1_ci'][1]:.3f}] | {pc} "
                     f"| {r['centroid']['macro_f1']:.3f} | {r['C_per_fold']} |")
    lines += ["", f"- Layer picked in hindsight: {hind} (macro-F1 {f1_curve.max():.3f}); chosen inside training: {chosen}"]
    if 0 in layers:
        l0 = per[layers.index(0)]["probe"]
        lines.append(f"- Sanity, layer 0 (embedding = amino-acid identity) accuracy {l0['accuracy']:.3f} vs identity baseline "
                     f"{base['identity']['accuracy']:.3f} (must agree)")
    lines.append(f"- {time.time() - t0:.0f}s")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
