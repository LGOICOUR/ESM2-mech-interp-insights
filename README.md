# esm2-probe-ablation

**A protein language model's residual stream decodes secondary structure. Is that direction load-bearing for what the model itself predicts?**

The same loop as [probe-monitor-control](https://github.com/LGOICOUR/probe-monitor-control),
run on a biological model with a ground truth whose answer is known. Model:
**ESM-2 150M** (`facebook/esm2_t30_150M_UR50D`, 30 layers, width 640), run locally
on an M-series Mac. Label: per-residue **three-state secondary structure**
(helix / strand / coil, DSSP-derived) on 548 PDB chains. Loop: (1) record the
residual stream at every layer for every labelled residue; (2) fit a linear probe
per layer, scored only on held-out chains, next to baselines that use no model;
(3) at the best layers, project the probe's direction out of the residual stream
*during the forward pass*, with a random direction as control, and measure what
the model's own masked-residue predictions lose; (4) refit a probe on the ablated
activations to ask whether the direction was the concept or one line through it.

This is a research exercise on a small open model, not a tool. Nothing here has
been reviewed by anyone but the agent that built it (see "How this was built").

## Headline

![Probe accuracy and macro-F1 at every layer, with the no-model baselines](results/p3/esm2_t30_150M/layer_curves.png)

**Decodable.** A logistic probe on the residual stream reads three-state
secondary structure at **0.820 accuracy [0.814, 0.826]** (macro-F1 0.815
[0.809, 0.820]) at layer 29, the layer every training fold chose on its own.
All numbers are on held-out chains (5 chain-grouped folds); 95% CIs resample
the 548 chains.

| reader | accuracy | macro-F1 |
|---|---|---|
| majority class (coil) | 0.431 [0.423, 0.440] | 0.201 [0.198, 0.204] |
| one-hot residue identity, no model | 0.491 [0.486, 0.497] | 0.439 [0.434, 0.444] |
| one-hot ±3-residue window, no model | 0.612 [0.606, 0.619] | 0.574 [0.569, 0.580] |
| **probe on ESM-2 150M, layer chosen inside training (29)** | **0.820 [0.814, 0.826]** | **0.815 [0.809, 0.820]** |
| class-mean (nearest-centroid) decoder, best layer (28) | 0.776 | 0.772 |

The layer-0 probe lands exactly on the identity baseline (0.491), as it must:
ESM-2's embedding output is a function of the residue alone (positions enter
inside attention). Accuracy is 0.72 by layer 5, flat through layer 17, then
climbs to the top of the network; layer 30 is a hair below 29.

**Load-bearing, but weakly, and through the class-mean geometry rather than
the probe's own direction.** For each held-out fold, a direction fit on the
training chains is projected out of every residue's residual vector during the
forward pass, and the model's masked prediction of each residue is re-read
(8 masking passes per chain, every residue masked once). Change in the masked
log-likelihood of the true residue, by the residue's true class, with a random
direction as control. Baseline: −1.958 nats, top-1 39.2%.

![Change in masked log-likelihood by class for each removed direction, and what a refit probe still decodes](results/p4/esm2_t30_150M/ablation.png)

| direction removed | layers | Δ log-lik., helix residues | strand residues | coil residues | refit probe at 29 (baseline 0.81) |
|---|---|---|---|---|---|
| probe's helix direction | 29 | **−0.011** [−0.012, −0.010] | −0.004 | −0.001 | 0.76 |
| probe's strand direction | 29 | −0.004 | **−0.009** [−0.010, −0.007] | −0.000 | 0.75 |
| class-mean helix direction | 29 | **−0.031** [−0.034, −0.028] | −0.004 | −0.009 | 0.68 |
| class-mean strand direction | 29 | −0.037 | −0.040 [−0.045, −0.035] | −0.032 | 0.59 |
| probe's plane (rank 2) | 29 | −0.011 | −0.009 | −0.003 | 0.71 |
| class-mean plane (rank 2) | 29 | −0.074 | −0.049 | −0.059 | **0.22** (0.56 at layer 30) |
| random direction (control) | 29 | −0.000 [−0.001, −0.000] | −0.000 | −0.000 | 0.81 |
| probe's helix direction | 26–30 | **−0.028** [−0.030, −0.026] | −0.015 | −0.005 | 0.77 |
| probe's strand direction | 26–30 | −0.012 | **−0.030** [−0.033, −0.027] | −0.005 | 0.76 |
| class-mean helix direction | 26–30 | **−0.072** [−0.077, −0.066] | −0.018 | −0.018 | 0.77 |
| class-mean strand direction | 26–30 | −0.154 | −0.111 | −0.161 | 0.76 |
| probe's plane (rank 2) | 26–30 | −0.032 | −0.035 | −0.015 | 0.73 |
| class-mean plane (rank 2) | 26–30 | −0.207 | −0.148 | −0.233 | 0.70 |
| random direction (control) | 26–30 | −0.002 | −0.002 | −0.002 | 0.81 |
| probe's plane (rank 2) | 12–18 | −0.013 | −0.017 | −0.012 | 0.81 |
| class-mean plane (rank 2) | 12–18 | −0.080 | −0.124 | −0.076 | 0.80 |

Full table with every CI, top-1 change rates and the original-probe column:
[`results/p4/esm2_t30_150M/summary.md`](results/p4/esm2_t30_150M/summary.md).

- **A double dissociation, small and clean.** The probe's helix direction hurts
  helix residues about three times more than strand residues and leaves coil
  alone; its strand direction does the reverse; the random direction does
  nothing (CIs are ten times tighter than the effects, on 113k residues). Over
  five layers the targeted-class effect grows to −0.03 nats and the dissociation
  holds. So the decoded direction is used by the model for the class it
  decodes, but the model leans on it very lightly: 0.01 nats at one layer on a
  −1.96 baseline, with 7% of top-1 predictions changing.
- **The class-mean direction moves the model two to four times more than the
  probe's direction** for the same class and scope (cosine between the two
  0.66 for helix, 0.61 for strand), the same ordering as in probe-monitor-control
  and Arditi et al. The helix one is specific; the strand one is not (over
  layers 26–30 it costs coil −0.16, helix −0.15, strand −0.11), so part of its
  extra leverage is that it carries something the model uses everywhere.
- **A line, not the concept.** After any single direction is removed, a probe
  refit on the ablated activations still decodes at 0.68–0.77 (baseline 0.81),
  and even after the probe's whole rank-2 plane is gone, 0.71–0.73. The
  class-mean plane is different: removing it at one layer makes the three
  class-conditional means coincide there, so by the LEACE result (Belrose et
  al. 2023) no linear reader can beat a constant predictor at that layer, and
  the refit reads 0.22 (chance 0.20). The next layer rebuilds part of it (0.56
  at layer 30), and the model's output loses only 0.06 nats. Linear erasure of
  the concept at one layer costs the model six times what erasing the probe's
  plane costs (−0.07 vs −0.01), and still little.
- **Mid-network removal is recomputed.** After removing either plane at layers
  12–18, a probe at layer 29 reads 0.81 again, and even the *original*
  clean-trained probe does, so the late representation is rebuilt in the same
  directions; the output still loses up to 0.12 nats for the class-mean plane,
  which the final secondary-structure representation does not explain.

Reading, in one sentence: in ESM-2 150M secondary structure is cleanly
decodable, the decoded direction is causally used for the right class, and
nearly all of what the model actually reads lives in the class-mean geometry,
not in the probe's weight vector.

**Follow-up: is what the probe reads *beyond* the class-mean direction used?**
The probe direction and the class-mean direction overlap (cosine 0.66 helix,
0.61 strand), so the rows above cannot say whether the model reads the probe's
own information or only the part the two share. Split each direction into its
component along the other and the unit remainder orthogonal to it, and remove
the remainder alone: "probe's helix direction, class-mean part removed" is the
probe's helix weight vector minus its component along the class-mean helix
direction. Pre-registered in `DECISIONS.md`; full table with CIs in
[`results/p4/esm2_t30_150M_perp/summary.md`](results/p4/esm2_t30_150M_perp/summary.md).

| direction removed | layers | Δ log-lik., helix residues | strand residues | coil residues | refit probe at 29 (baseline 0.81) |
|---|---|---|---|---|---|
| probe's helix direction, class-mean part removed | 29 | −0.002 [−0.003, −0.002] | −0.003 | −0.001 | 0.79 |
| probe's strand direction, class-mean part removed | 29 | −0.008 | −0.008 | −0.009 | 0.78 |
| class-mean helix direction, probe part removed | 29 | **−0.010** [−0.011, −0.008] | −0.003 | −0.006 | 0.81 |
| class-mean strand direction, probe part removed | 29 | −0.042 | −0.034 | −0.037 | 0.81 |
| random direction (control) | 29 | −0.000 | −0.000 | −0.000 | 0.81 |
| probe's helix direction, class-mean part removed | 26–30 | −0.009 | −0.009 | −0.004 | 0.80 |
| probe's strand direction, class-mean part removed | 26–30 | −0.018 | −0.017 | −0.033 | 0.79 |
| class-mean helix direction, probe part removed | 26–30 | **−0.032** [−0.035, −0.029] | −0.012 | −0.010 | 0.81 |
| class-mean strand direction, probe part removed | 26–30 | −0.195 | −0.118 | −0.220 | 0.81 |
| random direction (control) | 26–30 | −0.002 | −0.002 | −0.002 | 0.81 |

What the probe reads beyond the class-mean direction is not used by the model.
The helix remainder costs helix residues 0.002 nats (the full probe direction
0.011, the class-mean direction 0.031) with no specificity; the strand
remainder costs 0.008 on every class alike, the signature of variance removed
rather than of strand information. The probe's direction is the sharper
separator (d' 2.7 against 1.75 for the class-mean direction, on held-out
residues; `scripts/check_variance.py`) with a third of its variance: a
discriminative probe points toward a low-variance direction where the classes
separate cleanly, and the model's output is insensitive to that direction. The
remainder of the class-mean helix direction, by contrast, still carries helix
signal (d' 1.15) and still costs helix residues specifically. The strand
class-mean direction and its remainder are the two highest-variance single
directions (3.5% and 4.5% of the layer's variance) and the most damaging,
equally for every class. The random control, at 0.18% of the variance, is not
variance-matched; a variance-matched control was not run.

## Caveats, what did not work first

- **The effect sizes are small.** The strongest single-direction effect at the
  read-out layer is 0.03 nats on a −1.96 baseline; the probe's own direction
  moves the output by 0.01. "Separated from the random control" is not "the
  model depends on it". The whole class-mean plane over five layers (−0.15 to
  −0.23 nats, ~40% of top-1 predictions change) is the only ablation here that
  visibly degrades the model.
- **The pre-registered sizes were wrong; the directions held.** Before the run
  (DECISIONS.md) the agent predicted −0.02 to −0.10 nats for one direction at
  layer 29 (got −0.011 / −0.009 for the probe directions, −0.031 for the
  class-mean helix direction), −0.1 to −0.3 for the late band (got −0.03 for
  probe directions, −0.07 to −0.15 for class-mean), and that the probe's plane
  would hurt every class by −0.2 to −0.5 (got −0.03: wrong by ten times). All
  seven predictions held qualitatively; three missed quantitatively.
- **Stacked projections with clean-fit directions do not add up.** After the
  class-mean plane is projected out of layers 26–28, the clean-fit plane at 29
  no longer equalizes the class means there (refit 0.70, against 0.22 when 29
  is ablated alone): upstream ablation moves the class means off the clean
  plane. A real multi-layer erasure must refit on the ablated stream layer by
  layer; the band numbers here understate what a cumulative erasure would do.
- **The strand class-mean direction is not clean.** It hurts all three classes
  about equally, so "class-mean directions are more load-bearing" is partly
  "class-mean directions carry more than the concept".
- **The read-out layer is near the top (29 of 30)**, so "load-bearing" at the
  single layer is close to "read by the LM head"; the 12–18 scope is what shows
  that an earlier removal is recomputed.
- **The masked read-out is an approximation.** Masking every 8th residue per
  pass sits 0.022 nats below the exact one-residue-per-pass PLL on six test
  chains (top-1 agreement 87%); every condition uses the identical schedule, so
  the differences are fair, but the absolute PLLs are slightly pessimistic and
  per-residue values are noisy (mean |difference| 0.17 nats). Effects are
  therefore reported only as class means with chain-resampled CIs.
- **Folds are grouped by chain, not by family.** Homologous chains can sit on
  both sides of a fold, and ESM-2 saw these sequences in training (UniRef), so
  the probe accuracy is an upper bound on family-held-out generalization. The
  ablation does not depend on this.
- **One model, one seed.** ESM-2 150M only (650M is a flag, not run); one fold
  seed; one random direction per fold and layer as control (two for the planes).
  The controls are tight enough (−0.000 to −0.006, CIs ±0.001) that more draws
  would not change the reading.
- **Built and run by an agent; not yet reviewed by Luis** (see below).

## Data

- Labels and sequences: the **NetSurfP training/test bundle** (DTU Health Tech,
  [dataset page](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/5-Dataset.php)),
  splits **CB513** (Cuff & Barton 1999, re-annotated) and **TS115** (Klausen et al.
  2019), with Q8 DSSP states collapsed to Q3 as G/H/I → helix, B/E → strand,
  S/T/C → coil. Read from the Hugging Face mirror
  [`lamm-mit/protein-secondary-structure-netsurfp`](https://huggingface.co/datasets/lamm-mit/protein-secondary-structure-netsurfp)
  at commit `84141978b55a` (recorded by `scripts/p1_data.py` in
  `results/p1/config.json`). A second mirror, `proteinea/secondary_structure_prediction`
  (DOI 10.57967/hf/1104), carries the same sequences and labels.
- **Licence:** neither mirror nor the DTU page states one. The underlying
  structures and DSSP assignments are from the PDB (CC0). Treat the label files
  as academic-use data from DTU until they say otherwise; this repo does not
  redistribute them (`data/raw/` is ignored; the script re-downloads).
- **What the mirror's CB513 actually is:** 513 rows, but each row is a CB513
  *domain* carrying its whole PDB chain, with residues outside the domain
  unlabelled. So multi-domain chains repeat (434 distinct chains) and 41% of
  CB513 residues have no label. `p1_data.py` merges duplicate chains (union of
  labels; none conflicted) and keeps unlabelled residues as context the model
  reads but never as probe data. TS115 is clean. One TS115 chain over ESM-2's
  1,022-residue budget was dropped. Final set: **548 chains, 135,365 residues,
  113,279 labelled** (helix 35%, strand 21%, coil 43%).
- Model: [ESM-2](https://github.com/facebookresearch/esm) (Lin et al. 2023), MIT
  licence, via `transformers`.

## Reproduce

```bash
uv sync
uv run pytest -q                              # 27 offline tests, no network, ~6 s
uv run python scripts/p1_data.py              # downloads two label files from Hugging Face (~0.4 MB)
uv run python scripts/p2_read.py              # downloads ESM-2 150M (~0.6 GB); records 4.5 GB of activations in ~25 s on an M4 Pro
uv run python scripts/p3_probe.py && uv run python scripts/p3_figure.py     # ~1 min
uv run python scripts/p4_ablate.py && uv run python scripts/p4_figure.py    # ~1 h for 24 conditions (3 scopes × 8 directions)
uv run python scripts/p4_ablate.py --kinds probe_H_perp probe_E_perp random dm_H_perp dm_E_perp --scopes single band --tag perp   # ~26 min; the remainder follow-up, into results/p4/esm2_t30_150M_perp/
uv run python scripts/check_variance.py       # variance share and class gap of every direction at layer 29, ~1 min
```

Every script prints its own checks and refuses to continue if one fails
(label/sequence alignment, no duplicate chains, finite activations, layer-0
probe = identity baseline, P4's unablated pass matches the P2 recording, the
removed coordinate is zero after the hook). Nothing calls a paid API. Quick
versions (each `--limit N` run writes to its own `<recording>_limitN` folder,
which the next script takes as `--recording`):

```bash
uv run python scripts/p2_read.py --limit 20
uv run python scripts/p3_probe.py --recording esm2_t30_150M_limit20 --layers 0 15 29 --n-boot 50
uv run python scripts/p4_ablate.py --recording esm2_t30_150M_limit20 --limit 12 --kinds probe_H dm_sub random --scopes single --n-boot 50
uv run python scripts/p4_figure.py --recording esm2_t30_150M_limit20_limit12
```

That reduced chain, plus the tests and the full P1, was run from a fresh
`git clone` of this repository on 2026-10-28 and passed every check.

The 650M model is a flag (`--model facebook/esm2_t33_650M_UR50D` on `p2_read.py`,
then `--recording esm2_t33_650M` downstream) and was not run.

## Layout

```
src/esm2_probe/   data (labels, de-duplication, one-hot baselines), reader (hooks, masked read-out),
                  probe (grouped CV, CIs, directions), ablate (projection hook), baselines
scripts/          p1_data -> p2_read -> p3_probe (+ p3_figure) -> p4_ablate (+ p4_figure); each prints its checks;
                  check_masking and check_variance are diagnostics
tests/            offline tests on synthetic data and a fake model (no network)
results/          p1 proteins + config; p2 index + summary (activations are not committed);
                  p3 metrics, layer curves, figure; p4 metrics, per-residue read-outs, ablation table, figure;
                  p4 <recording>_perp: the remainder follow-up and its variance check
DECISIONS.md      dated log of every choice, with the ablation's expectations written before it ran
```

## How this was built

Coding agents (Claude, run from Claude Code) wrote nearly all of the code,
tests and text in this repository, in one session on 2026-10-28, under a design
brief written from Luis Goicouria's own notes: the model, the kind of ground
truth, the probe-then-ablate loop with a random-direction control and a
refit-probe check, the conventions (pinned environment, offline tests, one
script per phase, dated decision log, confidence intervals, caveats first), and
the rule that nothing is claimed until it exists and has been reviewed. The
agent chose the dataset copy, found and handled the CB513 duplication, chose
the metrics, the layer scopes, the direction kinds and the masking scheme, and
wrote the pre-registered expectations in `DECISIONS.md` (Luis was not available
to dictate them). The checks on that work are the offline tests, the
ground-truth labels, the baselines and controls that sit beside every number,
and the scripts' own assertions. The remainder follow-up (the last block of
the headline) was added the same day at Luis's request, after he was walked
through the first result and asked whether the probe's own information is
used; the agent pre-registered it, ran it and wrote it up.

**Luis has not yet reviewed the result.** Until he says otherwise, treat the
numbers as an agent's unreviewed run of an honest pipeline: reproducible from
this repo, but not yet checked by a person. The write-up voice follows his
other repos; the prose here is the agent's.

## Acknowledgements

ESM-2 (Lin et al., *Science* 2023); NetSurfP-2.0/3.0 data (Klausen et al. 2019;
Høie et al. 2022); CB513 (Cuff & Barton 1999). The ablation follows the
projection used in probe-monitor-control and in Arditi et al. (2024) for
refusal directions.

MIT License.
