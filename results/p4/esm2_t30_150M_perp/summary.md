# P4 ablation, `esm2_t30_150M`: L* = 29, 548 held-out chains, 113279 labelled residues

Baseline (no ablation): masked PLL per residue -1.958 nats (H -1.935, E -1.839, C -2.035); masked top-1 accuracy 39.2%. k = 8 masking passes per chain.

## Effect on the model's own output

Change in masked log-likelihood of the true residue, mean over residues of each class [95% CI, chains resampled]; top-1 changed = share of residues whose top-1 prediction differs from the unablated run.

| condition | layers | ΔPLL helix | ΔPLL strand | ΔPLL coil | top-1 changed H / E / C |
|---|---|---|---|---|---|
| single/probe_H_perp | 29 | -0.002 [-0.003, -0.002] | -0.003 [-0.004, -0.002] | -0.001 [-0.002, -0.001] | 3.3% / 4.9% / 4.0% |
| single/probe_E_perp | 29 | -0.008 [-0.009, -0.006] | -0.008 [-0.010, -0.007] | -0.009 [-0.010, -0.008] | 7.6% / 9.2% / 8.7% |
| single/random | 29 | -0.000 [-0.001, -0.000] | -0.000 [-0.001, -0.000] | -0.000 [-0.001, -0.000] | 1.5% / 1.7% / 1.7% |
| single/dm_H_perp | 29 | -0.010 [-0.011, -0.008] | -0.003 [-0.005, -0.002] | -0.006 [-0.007, -0.005] | 7.1% / 6.0% / 7.3% |
| single/dm_E_perp | 29 | -0.042 [-0.046, -0.039] | -0.034 [-0.038, -0.029] | -0.037 [-0.040, -0.035] | 15.9% / 14.4% / 15.4% |
| band/probe_H_perp | 26–30 | -0.009 [-0.011, -0.008] | -0.009 [-0.011, -0.007] | -0.004 [-0.005, -0.003] | 7.0% / 8.4% / 7.7% |
| band/probe_E_perp | 26–30 | -0.018 [-0.021, -0.015] | -0.017 [-0.020, -0.014] | -0.033 [-0.036, -0.031] | 14.7% / 18.7% / 17.9% |
| band/random | 26–30 | -0.002 [-0.003, -0.002] | -0.002 [-0.003, -0.002] | -0.002 [-0.002, -0.001] | 3.6% / 4.1% / 3.8% |
| band/dm_H_perp | 26–30 | -0.032 [-0.035, -0.029] | -0.012 [-0.014, -0.010] | -0.010 [-0.012, -0.008] | 12.7% / 10.0% / 11.4% |
| band/dm_E_perp | 26–30 | -0.195 [-0.204, -0.187] | -0.118 [-0.128, -0.108] | -0.220 [-0.230, -0.209] | 37.7% / 35.3% / 40.2% |

## Is the concept gone? Decodability after ablation

Original probe = the fold's unablated probe applied to the ablated activations; refit = a new probe trained on the ablated activations (same chain-grouped folds). Macro-F1 (H, E = per-class F1).

| condition | layer 29: original \| refit | layer 30: original \| refit |
|---|---|---|
| baseline, no ablation | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) |
| single/probe_H_perp | 0.72 (H 0.76, E 0.63) | 0.79 [0.79, 0.80] (H 0.84, E 0.75) | 0.75 (H 0.79, E 0.67) | 0.80 [0.79, 0.80] (H 0.84, E 0.75) |
| single/probe_E_perp | 0.62 (H 0.63, E 0.52) | 0.78 [0.77, 0.79] (H 0.83, E 0.71) | 0.65 (H 0.67, E 0.55) | 0.78 [0.78, 0.79] (H 0.83, E 0.73) |
| single/random | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) |
| single/dm_H_perp | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.87, E 0.77) | 0.81 (H 0.86, E 0.77) | 0.81 [0.80, 0.82] (H 0.86, E 0.77) |
| single/dm_E_perp | 0.81 (H 0.85, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.78) | 0.80 (H 0.85, E 0.76) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) |
| band/probe_H_perp | 0.74 (H 0.78, E 0.65) | 0.80 [0.79, 0.81] (H 0.84, E 0.75) | 0.72 (H 0.77, E 0.63) | 0.79 [0.78, 0.79] (H 0.83, E 0.74) |
| band/probe_E_perp | 0.62 (H 0.63, E 0.52) | 0.79 [0.78, 0.79] (H 0.84, E 0.73) | 0.57 (H 0.59, E 0.47) | 0.77 [0.76, 0.78] (H 0.82, E 0.70) |
| band/random | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) |
| band/dm_H_perp | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) | 0.80 (H 0.85, E 0.76) | 0.81 [0.80, 0.82] (H 0.86, E 0.77) |
| band/dm_E_perp | 0.79 (H 0.85, E 0.73) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) | 0.77 (H 0.83, E 0.71) | 0.81 [0.80, 0.81] (H 0.86, E 0.77) |

- subspace ranks: {'probe_sub': 2, 'dm_sub': 2, 'random_sub': 2}; {'cos(probe_H, dm_H) at L*': 0.66114342212677, 'cos(probe_E, dm_E) at L*': 0.6080994606018066, 'cos(probe_H_perp, probe_H) at L*': 0.750259518623352, 'cos(probe_E_perp, probe_E) at L*': 0.7938608527183533}
- projection check: largest |removed coordinate| left at L* across conditions = 0.0390
- 1535s on mps
