# Decisions

Every decision, dated, with who made it. Predictions are written before the
experiment they concern is run. "Claude" below is the coding agent (Claude
Fable 5.1, run from Claude Code) that built the repo; "Luis" is Luis Goicouria.

## 2026-10-28 — Project set up

- **Origin** (Luis, 2026-10-28; relayed to
  this session as a brief): run the probe-monitor-control loop ("decodable vs
  load-bearing") on a protein language model, with a ground truth whose answer
  is known. Luis has not yet reviewed anything here.
- **Repo** (Luis's brief): `~/Desktop/ML Projects/esm2-probe-ablation`, same
  conventions as probe-monitor-control: pinned `uv`, offline pytest, one script per
  phase that prints its own checks, dated DECISIONS.md, README with CIs and
  caveats first. No git remote; Luis decides whether it goes public.
- **Model** (Luis's brief; Claude confirmed it runs): `facebook/esm2_t30_150M_UR50D`
  (30 layers, width 640, MIT licence), loaded with `EsmForMaskedLM` on the Mac
  GPU (MPS). Residual stream read from forward hooks on the embedding module and
  each `EsmLayer`, so layer 30 is the raw residual, not the LayerNorm-ed last
  hidden state. A batch of 8 x 324 residues takes 0.5 s. The 650M model is a flag,
  not run (one clean result first).
- **Ground truth** (Luis's brief: a public DSSP-labelled set; Claude chose the
  copy): per-residue three-state secondary structure from the NetSurfP
  training/test bundle (DTU Health Tech; Q8 -> Q3 as GHI -> H, BE -> E, STC -> C),
  mirrored on Hugging Face as `lamm-mit/protein-secondary-structure-netsurfp`
  (JSONL, '.' = unlabelled). A second mirror, `proteinea/secondary_structure_prediction`
  (CSV, DOI 10.57967/hf/1104), carries the same sequences and labels (checked
  on 484 CB513 chains matched by sequence); neither mirror states a licence,
  nor does the DTU page. The structures and DSSP assignments come from the PDB
  (CC0). Recorded in the README as the exact source.
- **CB513 is messier than its reputation** (Claude, found while inspecting the
  file): 513 rows but only 434 distinct chains, because each row is a CB513
  *domain* but carries its whole PDB chain, with residues outside the domain
  unlabelled ('.'); 41% of CB513 residues are unlabelled. TS115 is clean (115
  chains, 29,704 residues, all labelled). Decision: use **both**, merge
  duplicate chains by sequence (union of labels; drop a chain if two rows
  disagree on a labelled residue), keep unlabelled residues in the sequence the
  model reads but out of every probe and metric. Chains are the grouping unit
  for cross-validation and bootstrap.
- **Metrics** (Claude, following the brief): accuracy and macro-F1 over the
  three classes, held-out by chain-grouped 5-fold CV, 95% CIs from 1,000
  bootstrap resamples of chains. Baselines next to every number: majority
  class, and logistic regression on one-hot amino-acid identity, as the residue
  alone and as a +/-3 window (the stronger no-model baseline; the analogue of
  probe-monitor-control's word counter).
- **Probe** (Claude): per layer, standardize, multinomial logistic regression, C
  from {0.01, 0.1, 1} by grouped inner 3-fold CV on an 8k-residue subsample of
  the training fold; outer fit on up to 40k training residues (compute cap);
  predictions on every held-out residue. Also a class-mean decoder (nearest
  centroid on standardized features) at every layer, since the class-mean
  direction is one of the two we ablate.
- **Sanity check built in** (Claude): the embedding output of ESM-2 is a
  function of amino-acid identity alone (rotary positions enter inside
  attention), so the layer-0 probe must land on the identity baseline.

## 2026-10-28 — P1, P2, P3 run (Claude)

- **P1.** `lamm-mit/protein-secondary-structure-netsurfp` @ `84141978b55a`. 628 rows
  -> 549 distinct chains (79 CB513 duplicate rows merged, 0 label conflicts); one
  TS115 chain over 1,022 residues dropped. **548 chains, 135,365 residues,
  113,279 labelled** (H 35.4%, E 21.4%, C 43.1%); length median 213, max 874.
- **P2.** 31 x 113,279 x 640 float16 = 4.5 GB, 22 s on the Mac GPU. Residual
  norm grows from 2 (embedding) to 315 (layer 30).
- **P3.** Probe accuracy / macro-F1 rise almost monotonically with depth: 0.49
  at layer 0 (= the identity baseline, as it must), 0.72 by layer 5, a plateau to
  layer 17, then a climb to **0.820 [0.814, 0.826] accuracy, 0.815 [0.809, 0.820]
  macro-F1 at layer 29**; layer 30 is a hair lower. Every fold chose layer 29
  inside training, so L* = 29 with no hindsight. Baselines: majority 0.431 /
  0.201; one-hot identity 0.491 / 0.439; one-hot +/-3 window 0.612 / 0.574.
  Class-mean decoder peaks at 0.77 macro-F1 (layer 28) and drops at layer 30.
  C grid widened to 1e-4..1 after the first pass chose its floor; folds now
  choose 1e-3. 54 s for 31 layers with 1,000-draw CIs.
- **Scopes for P4** (fixed now, from L* = 29): `single` = [29]; `band` = 26–30
  (L* +/- 3, clipped); `mid` = 12–18 (the plateau, where secondary structure is
  decodable at 0.71–0.72 but far from the read-out, so later layers could
  recompute it). `wide` (26–30) coincides with `band` here and is not run.

## 2026-10-28 — Pre-registered expectations for P4 (Claude, written before any ablation ran)

Luis was not available to dictate these; they are the agent's predictions, made
from probe-monitor-control's results and the P3 curve, and are to be judged
against the P4 table as written. Confidence in brackets.

1. **Random controls are null.** One or two random directions: ΔPLL within
   +/-0.01 nats for every class at every scope; top-1 changed < 3%; original and
   refit probes within the baseline CI. [90%]
2. **One direction at one layer is a line, not the concept.** probe_H or dm_H at
   layer 29: the *original* probe's helix F1 drops hard (it loses its own axis;
   F1_H < 0.5), but a *refit* probe at layer 29 and at layer 30 recovers to within
   0.03 macro-F1 of baseline. [75%]
3. **Output effects are small but class-specific at layer 29.** Removing the
   helix direction lowers masked log-likelihood more at helix residues than at
   strand residues, and the strand direction the reverse; targeted-class ΔPLL
   between -0.02 and -0.10 nats, off-target within +/-0.02. The full double
   dissociation (both directions, both classes, random control flat) appears
   [60%]; at least one single-direction effect separates from the random
   control [75%].
4. **The late band (26–30) moves the output more and keeps the specificity**:
   targeted-class ΔPLL 2–4x the single-layer value (-0.1 to -0.3 nats) with the
   dissociation intact. [70%]
5. **Class-mean directions move the output more than probe directions** for the
   same class and scope (Arditi et al.; probe-monitor-control). [65%]
6. **Removing the whole rank-2 class subspace (probe_sub, dm_sub) at the band**
   hurts every class (ΔPLL -0.2 to -0.5 nats everywhere), the original probe
   collapses toward chance (< 0.45 macro-F1) [85%], and a refit probe at the
   ablated layer still decodes above the window baseline (> 0.70 macro-F1):
   the concept is spread over more than a plane. [70%]
7. **The mid band (12–18) is recomputed.** After removing a direction or the
   subspace at layers 12–18, a refit probe at layer 30 is back within 0.02 of
   baseline [70%], and output effects are smaller than for the late band [80%].

Metrics are fixed as implemented in `scripts/p4_ablate.py` before this entry:
k = 8 masking passes, 40k-residue direction fits per fold with P3's chosen C,
chain-grouped folds (seed 0), 1,000-draw chain bootstraps.
