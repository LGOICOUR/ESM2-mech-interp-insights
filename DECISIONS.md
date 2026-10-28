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

## 2026-10-28 — Masking-scheme check (Claude, run while P4 was in progress; P4's design was not changed)

`scripts/check_masking.py`, 6 random chains: the k = 8 read-out (every 8th
residue masked per pass) gives a mean PLL 0.022 nats *below* the exact
one-residue-per-pass PLL (masked neighbours 8 apart cost a little context);
per-residue |difference| 0.17 nats; top-1 agreement 87%. Every P4 condition,
including the unablated baseline, uses the identical schedule, so the scheme's
bias cancels in the ΔPLL differences; the per-residue noise it reveals is why
effects are reported as class means over tens of thousands of residues with
chain-resampled CIs, never per residue.

## 2026-10-28 — P4 ran (Claude): predictions against the table

Full run: 3 scopes x 8 direction kinds, 548 held-out chains, 113,279 residues,
58 min on the Mac GPU. Baseline masked PLL -1.958 nats, top-1 39.2%. Table in
`results/p4/esm2_t30_150M/summary.md`; figure `ablation.png`.

1. **Random controls null: held.** ΔPLL -0.000 (layer 29), -0.002 (26–30),
   -0.003 (12–18); two-direction planes -0.001 / -0.004 / -0.006; top-1 changed
   1.5–6%; original and refit probes 0.81 = baseline everywhere.
2. **One direction is a line, not the concept: held in direction, missed in
   size.** probe_H at 29: original probe helix F1 0.86 → 0.15; refit 0.76 at 29
   and 0.76 at 30, a 0.05 drop, not the 0.03 predicted. The class-mean strand
   direction is the exception: removing it alone drops the refit at 29 to 0.59
   (strand F1 0.22), recovering to 0.74 at layer 30.
3. **Small, class-specific output effects at layer 29: held in direction,
   half the predicted size.** probe_H: helix -0.011 [-0.012, -0.010], strand
   -0.004, coil -0.001; probe_E: strand -0.009, helix -0.004, coil -0.000;
   random -0.000. The double dissociation is there with CIs ten times tighter
   than the effects, but the effects are 0.01 nats, below the predicted
   -0.02 to -0.10. dm_H is specific (helix -0.031 vs strand -0.004); dm_E is not
   (-0.037 / -0.040 / -0.032).
4. **Band 26–30 moves more, keeps specificity: held for the probe directions
   (2.5–3x: helix -0.028 vs strand -0.015; strand -0.030 vs helix -0.012) and
   for dm_H (-0.072 vs -0.018), at a quarter of the predicted size. dm_E at the
   band is large and unspecific: helix -0.154, strand -0.111, coil -0.161.**
5. **Class-mean directions move the output more than probe directions: held**
   at every scope and class (2–4x).
6. **Plane removal at the band: held for the class-mean plane** (ΔPLL -0.207 /
   -0.148 / -0.233; original probe 0.34; refit 0.70), **wrong by 10x for the
   probe plane** (-0.032 / -0.035 / -0.015, predicted -0.2 to -0.5; original
   0.20; refit 0.73). The probe's own plane is nearly irrelevant to the output.
7. **Mid band 12–18 is recomputed: held.** Refit at 29 and 30 back at 0.81
   (0.80 for dm_sub); even the *original* clean probe reads 0.81, so the late
   representation is rebuilt in the same directions. Output effects smaller than
   the late band (probe plane -0.013 vs -0.032; class-mean plane -0.08 to -0.12
   vs -0.15 to -0.23) but not zero: the mid-layer class-mean plane carries
   something the output uses that is not the final secondary-structure
   representation.

**Two things learnt that were not predicted.**
- Removing the class-mean plane *at a single layer* equalizes the three
  class-conditional means there, so by the LEACE result (Belrose et al. 2023)
  no linear classifier can beat a constant predictor at that layer: the refit
  reads 0.22 (chance 0.20). The next layer rebuilds part of it (0.56 at 30),
  and the output loses only 0.06 nats. Removing the probe's plane at the same
  layer leaves 0.71 decodable and costs 0.01 nats. Same lesson as
  probe-monitor-control: the model reads the class-mean plane, the probe reads
  a different line through the concept.
- Stacked projections with directions fit on clean activations are not
  cumulative: after projecting the class-mean plane out of layers 26–28, the
  clean-fit plane at 29 no longer equalizes the means there (refit 0.70 vs 0.22
  for 29 alone), because the upstream ablations move the class means off the
  clean plane. A multi-layer erasure has to refit on the ablated stream.

**Verdict on the brief's question.** Decodable: yes, 0.82 vs a 0.61 no-model
baseline. Load-bearing: yes, specifically and measurably, but weakly, and only
the class-mean geometry is what the model reads; the probe's own direction
barely figures in the model's predictions. Luis has not reviewed this.
