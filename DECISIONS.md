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

## 2026-10-28 — Closed out (Claude)

- **Clean-clone check passed.** `git clone` into a scratch folder, `uv sync`,
  `uv run pytest -q` with `HF_HUB_OFFLINE=1` (26 passed), then the full P1, a
  20-chain P2, a 3-layer P3 with figure, a 12-chain P4 (probe_H, dm_sub, random
  at layer 29) with figure, and the masking check: every script's own checks
  passed; the reduced P4 reproduced the full run's shape (probe_H helix −0.007
  vs coil −0.001; dm_sub refit 0.20; random null). The one failure on the way
  was the test command itself: a `--limit N` run writes to `<recording>_limitN`,
  which the next script must be given; now documented in the README.
- **Committed:** code, tests, lock file, README, this log, and every result
  except the large regenerable files: `results/p2/*/acts.npy` (4.5 GB) and
  `results/p4/*/per_residue.npz` (47 MB) are gitignored, as is `data/raw/`.
  `results/p3/*/yhat_by_layer.npy` (3.5 MB, held-out predictions per layer) is
  kept. Three local commits; no remote.
- **Compute, whole project:** P2 22 s, P3 54 s, P4 58 min, checks ~2 min; about
  1 h 5 min of GPU time on the M4 Pro, no paid calls.
- **Not done, by choice or scope:** ESM-2 650M; family-grouped folds; exact
  single-residue PLL; cumulative multi-layer erasure with refit directions; a
  review of any of this by Luis.

## 2026-10-28 — Follow-up: is what the probe reads *beyond* the class-mean direction used? Pre-registered (Claude, before the run)

Luis asked, after the walk-through, whether probe-monitor-control's finding
(the probe reads a line the model's output does not use) translates here. The
single-direction rows cannot say: the probe direction and the class-mean
direction overlap (cosine 0.66 helix, 0.61 strand), so removing the probe
direction also removes 44% of the class-mean coordinate. Split each direction
into what it shares with the other and the unit remainder orthogonal to it:

    probe_c = 0.66 dm_c + 0.75 probe_c_perp        dm_c = 0.66 probe_c + 0.75 dm_c_perp

(`orthogonal_remainder` in `src/esm2_probe/ablate.py`; new kinds in
`scripts/p4_ablate.py`). Removing `probe_c_perp` alone removes exactly the part
of the probe's direction that the class-mean direction does not contain;
`dm_c_perp` is the reverse split, added by Claude for symmetry (Luis asked for
the first). Run: kinds probe_H_perp, probe_E_perp, random, dm_H_perp, dm_E_perp
at scopes single (29) and band (26–30), everything else as in the main run,
written to `results/p4/esm2_t30_150M_perp/` so the main table stays as it is.

Predictions, confidence in brackets:

1. **The probe's remainder is close to inert.** probe_H_perp at 29: helix ΔPLL
   between −0.004 and 0, and no dissociation (|helix − strand| < 0.003); at
   26–30 helix between −0.010 and 0. The same for probe_E_perp on strand.
   [65%] If so, the part of the probe's direction the model uses is its overlap
   with the class-mean direction, which is the probe-monitor-control result in
   ESM-2 form: what the probe reads beyond the class-mean geometry is not read
   by the model. Alternative: the remainder carries a specific effect of
   ≥ 0.005 nats at 29, meaning the model reads a second line inside the probe
   direction. [35%]
2. **The class-mean remainder carries most of the class-mean effect.**
   dm_H_perp at 29: helix between −0.030 and −0.012, strand within [−0.008, 0]
   (specific). [65%] dm_E_perp: unspecific like dm_E, every class between
   −0.015 and −0.045. [60%]
3. **Decodability.** Refit at 29 after probe_H_perp or probe_E_perp ≥ 0.79
   (the remainder is orthogonal to its own class's mean-difference direction, so
   that class's means are not equalized) [70%]; after dm_H_perp 0.70–0.78 [60%].
   The original probe loses its own axis either way: helix F1 < 0.5 after
   probe_H_perp, since 56% of the probe's squared weight norm lies along it
   [70%].
4. **Random control null**, as before. [90%]

Arithmetic behind 1 and 2: if the model's sensitivity within the plane of the
two directions were along the class-mean direction alone, removing a unit
direction u would cost about cos²(u, dm) of the class-mean effect: probe_H
0.44 × 0.031 = 0.014 (observed 0.011), probe_H_perp 0, dm_H_perp 0.56 × 0.031
= 0.017. A 12-chain smoke run of three conditions will run after this entry to
check the code, before the full run; this entry is not edited afterwards.

## 2026-10-28 — The remainder follow-up ran (Claude): predictions against the table

Ten conditions, 548 chains, 26 min on the Mac GPU; baseline identical to the
main run (PLL −1.958, top-1 39.2%). Table in
`results/p4/esm2_t30_150M_perp/summary.md`. At layer 29 probe_H_perp has
cosine 0.75 with probe_H and 0 with dm_H, probe_E_perp 0.79 with probe_E
(orthogonality asserted in-script at every fold and layer).

1. **The probe's remainder is close to inert: held for helix; held in kind,
   missed in size, for strand.** probe_H_perp at 29: helix −0.002 [−0.003,
   −0.002], strand −0.003, coil −0.001 (random −0.000); at 26–30 helix −0.009,
   strand −0.009, coil −0.004. No dissociation anywhere. probe_E_perp at 29:
   strand −0.008, outside the predicted −0.004 to 0, but helix −0.008 and coil
   −0.009 as well, so no strand specificity; at 26–30 strand −0.017, helix
   −0.018, coil −0.033. The alternative (a class-specific remainder effect
   ≥ 0.005) occurred for neither class. What the probe reads beyond the
   class-mean direction is not read by the model, for either class.
2. **The class-mean remainder carries most of the class-mean effect: held for
   specificity, missed in size.** dm_H_perp at 29: helix −0.010 [−0.011,
   −0.008], strand −0.003, coil −0.006 (predicted −0.012 to −0.030); at 26–30
   helix −0.032, strand −0.012, coil −0.010 (dm_H itself −0.072 / −0.018 /
   −0.018). The remainder carries a third of the single-layer helix effect and
   44% of the band effect, not most of it. dm_E_perp: unspecific as predicted,
   −0.042 / −0.034 / −0.037 at 29; at 26–30 −0.195 / −0.118 / −0.220, *larger*
   than dm_E itself (−0.154 / −0.111 / −0.161).
3. **Decodability: half right.** Refit at 29 after probe_H_perp 0.79 [0.79,
   0.80] (held); after probe_E_perp 0.78 (missed by 0.01); after dm_H_perp
   0.81, baseline, not the predicted 0.70–0.78: removing 56% of the helix
   mean-gap variance leaves helix linearly separable from the rest. The
   original probe did not lose its axis either: 0.72 after probe_H_perp (helix
   F1 0.76, predicted < 0.5) and 0.62 after probe_E_perp.
4. **Random control null: held.** −0.000 at 29, −0.002 at 26–30; refit 0.81.

**Effects are sub-additive.** probe_H and dm_H_perp are orthogonal and span
the plane that contains dm_H; removing either costs helix residues about a
third of what removing dm_H costs (−0.011 and −0.010 against −0.031). The
model's dependence on the helix class-mean coordinate is super-linear in the
fraction of it removed: partial removals cost less than their share.

**Post-hoc check, `scripts/check_variance.py`** (not pre-registered; run after
the table, to read the unspecific effects). RMS of the removed component on
held-out residues as a share of the layer-29 variance, and the class-mean gap
along the direction in within-class standard deviations (d'), directions fit on
the training folds as in P4:

| direction | share of variance | d' helix vs rest | d' strand vs rest | ΔPLL at 29, targeted class |
|---|---|---|---|---|
| random | 0.0018 | 0.00 | +0.01 | −0.000 |
| probe_H | 0.0049 | +2.68 | −1.51 | −0.011 |
| probe_H_perp | 0.0054 | −0.01 | −0.91 | −0.002 |
| dm_H_perp | 0.0169 | +1.15 | +0.06 | −0.010 |
| dm_H | 0.0164 | +1.75 | −0.43 | −0.031 |
| probe_E | 0.0044 | −1.47 | +2.36 | −0.009 |
| probe_E_perp | 0.0143 | −0.51 | −0.03 | −0.008, every class |
| dm_E_perp | 0.0449 | +0.01 | +0.77 | −0.034, every class |
| dm_E | 0.0350 | −0.25 | +1.13 | −0.040, every class |

Three readings. (i) The probe's direction is the sharper separator (d' 2.7 vs
1.75) with a third of the class-mean direction's variance: a discriminative
probe points toward a low-variance direction where the classes separate
cleanly, and the model's output is insensitive to that direction. The helix
remainder carries three times a random direction's variance and a real
anti-strand gap (d' −0.9), and the output does not care. (ii) The unspecific
effects track variance, not class signal: the strand class-mean direction and
its remainder are the two highest-variance single directions (3.5% and 4.5%)
and the two most damaging, equally for every class; the strand probe's
remainder carries 1.4% and costs 0.008 everywhere. The random control, at
0.18%, is not a variance-matched control; a variance-matched random control
would be the right one and was not run. (iii) The helix effects follow the
helix class-mean gap: no gap, no effect (probe_H_perp); partial gap, partial
effect (probe_H, dm_H_perp); full gap, full effect (dm_H).

**Answer to Luis's question.** probe-monitor-control's finding translates in
the only form it can take here: what the probe reads beyond the class-mean
direction is not used by the model, for either class. The difference from the
Qwen monitor is the overlap. There the probe's line and the model's line were
nearly orthogonal, so removing the probe's line did nothing; here they share
0.6 of their direction, so removing the probe's line costs the model exactly
the part it shares. Luis has not reviewed this.

## 2026-10-28 — Variance-matched random control: pre-registered (Claude, before the run)

Luis asked for the control the remainder write-up names as missing. The
random control removes a direction carrying 0.18% of the layer's variance,
while the class-mean strand direction and its remainder carry 3.5% and 4.5%,
so their unspecific damage might be the size of the perturbation and nothing
else. Construction (`variance_matched_random` in `src/esm2_probe/ablate.py`):
draw a Gaussian direction in the eigenbasis of the training residues'
covariance and tilt it toward the high-variance axes, c_i = g_i λ_i^{p/2},
with the tilt p set by bisection so the direction's variance share equals, per
fold and layer, that of dm_H (≈ 1.6%), dm_E (≈ 3.5%) or dm_E_perp (≈ 4.5%).
One draw per fold and layer, so a band condition averages over 25 directions.
Kinds random_vm_dmH, random_vm_dmE, random_vm_dmEperp; scopes single (29) and
band (26–30); everything else as before; written to
`results/p4/esm2_t30_150M_vm/`.

Predictions, confidence in brackets:

1. **The matched directions carry no class signal and leave decodability at
   baseline.** |d'| < 0.3 for helix and strand on held-out residues
   (`check_variance.py`) [70%]; refit at 29 ≥ 0.80 [85%].
2. **At dm_H's share (1.6%), the matched random costs every class between
   −0.002 and −0.012 nats at 29, with no specificity** [65%]: it accounts for
   dm_H's off-target floor (strand −0.004, coil −0.009) and not for its helix
   effect (−0.031). dm_H's helix effect is signal, not size.
3. **At the strand directions' shares (3.5%, 4.5%), the matched random costs a
   substantial fraction of what dm_E and dm_E_perp cost, unspecifically.**
   Ratio of the matched random's class-averaged ΔPLL to its reference
   direction's, at 29: between 0.3 and 1.2 [55%]; below 0.3, meaning the
   class-mean strand directions carry something the model uses beyond their
   size [30%]; above 1.2, meaning a random direction of that size is more
   damaging than the class-mean one [15%]. Same ordering at 26–30 [60%].
4. **Effects scale with share**: for every class, the 4.5% match costs more
   than the 3.5% match, which costs more than the 1.6% match [75%].

Caveat known in advance: a direction tilted toward the high-variance axes is
random only within those axes. If a few outlier dimensions hold most of the
variance, the matched directions will load on them, and the control then
tests "a direction of this size along the stream's dominant axes" rather than
"any direction of this size". The cosine with the top eigenvector and the
participation ratio are recorded for every single direction, the class-mean
ones included, so the comparison can be judged. A 12-chain smoke run of one
condition will run after this entry to check the code, before the full run;
this entry is not edited afterwards.

## 2026-10-28 — Variance-matched random control ran (Claude): predictions against the table

Six conditions, 548 chains, 16 min; baseline identical again (PLL −1.958,
top-1 39.2%). Table in `results/p4/esm2_t30_150M_vm/summary.md`; held-out
check in its `variance_check.json`: shares 1.68 / 3.47 / 4.44% against the
references' 1.64 / 3.50 / 4.49%. Layer 29's spectrum is not dominated by one
axis (top eigenvalue 7.3% of the variance, then 3.4, 2.2, 1.7, 1.5%), but the
strand class-mean direction is largely that top axis (cosine 0.60; its
remainder 0.71), and the two matched randoms for it reproduce the loading
(0.61, 0.71) and the effective number of axes (5.7 vs 6.2; 3.3 vs 3.4). The
match for dm_H is cruder: its random loads more on the top axis (0.33 vs 0.15)
over more axes (21 vs 7).

1. **No class signal, decodability at baseline: held for decodability (refit
   0.80 at 29 for all three), borderline for signal.** d' 0.30 (helix) and
   0.33 (strand) for the two lower matches against the < 0.3 predicted: a
   direction tilted toward the top axis inherits a little of that axis's
   strand correlation.
2. **At dm_H's share, a small unspecific cost: held.** −0.009 / −0.007 /
   −0.008 at 29, the size of dm_H's off-target floor (−0.004 / −0.009) and
   less than a third of its helix effect (−0.031). At 26–30 the matched random
   costs helix −0.046 against dm_H's −0.072, and the other two classes −0.045
   / −0.038 against dm_H's −0.018 / −0.018: a random direction of dm_H's size
   does more general damage and less helix damage than dm_H. The helix effect
   is signal.
3. **At the strand directions' shares, a substantial fraction: held, middle
   option.** Ratio of the matched random's class-averaged ΔPLL to its
   reference's: dm_E 0.59 at 29 and 0.61 at 26–30; dm_E_perp 1.03 at 29 and
   0.71 at 26–30. The strand remainder's damage is its size: a random
   direction of the same share and the same loading on the top axis does the
   same at 29 (14–16% of top-1 predictions change in both). The strand
   class-mean direction keeps about 40% of its damage beyond size; that part
   is not explained.
4. **Effects scale with share: held** for every class at both scopes (4.5% >
   3.5% > 1.6%).

Two readings. The class pattern of the unspecific damage is itself a property
of the variance geometry, not of strand meaning: at 26–30 both dm_E and its
matched random hit strand residues least and coil most. And removing the
strand class-mean direction is, to first order, removing the stream's dominant
axis, which is why it hurts everything. The specific effects (helix for dm_H
and its remainder; the double dissociation of the probe directions) are the
part of the table that size does not explain. Luis has not reviewed this.
