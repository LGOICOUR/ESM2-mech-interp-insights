# P4 ablation, `esm2_t30_150M`: L* = 29, 548 held-out chains, 113279 labelled residues

Baseline (no ablation): masked PLL per residue -1.958 nats (H -1.935, E -1.839, C -2.035); masked top-1 accuracy 39.2%. k = 8 masking passes per chain.

## Effect on the model's own output

Change in masked log-likelihood of the true residue, mean over residues of each class [95% CI, chains resampled]; top-1 changed = share of residues whose top-1 prediction differs from the unablated run.

| condition | layers | ΔPLL helix | ΔPLL strand | ΔPLL coil | top-1 changed H / E / C |
|---|---|---|---|---|---|
| single/random_vm_dmH | 29 | -0.009 [-0.010, -0.007] | -0.007 [-0.009, -0.005] | -0.008 [-0.010, -0.007] | 7.3% / 8.3% / 9.0% |
| single/random_vm_dmE | 29 | -0.022 [-0.024, -0.020] | -0.018 [-0.022, -0.014] | -0.024 [-0.026, -0.022] | 13.5% / 13.6% / 15.1% |
| single/random_vm_dmEperp | 29 | -0.036 [-0.040, -0.032] | -0.036 [-0.043, -0.030] | -0.044 [-0.048, -0.040] | 14.2% / 15.1% / 16.3% |
| band/random_vm_dmH | 26–30 | -0.046 [-0.050, -0.043] | -0.045 [-0.050, -0.041] | -0.038 [-0.040, -0.035] | 17.9% / 18.9% / 18.5% |
| band/random_vm_dmE | 26–30 | -0.083 [-0.089, -0.076] | -0.079 [-0.087, -0.071] | -0.099 [-0.108, -0.091] | 26.4% / 26.8% / 29.6% |
| band/random_vm_dmEperp | 26–30 | -0.133 [-0.144, -0.123] | -0.104 [-0.113, -0.095] | -0.141 [-0.151, -0.130] | 31.5% / 29.2% / 31.9% |

## Is the concept gone? Decodability after ablation

Original probe = the fold's unablated probe applied to the ablated activations; refit = a new probe trained on the ablated activations (same chain-grouped folds). Macro-F1 (H, E = per-class F1).

| condition | layer 29: original \| refit | layer 30: original \| refit |
|---|---|---|
| baseline, no ablation | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) | 0.81 (H 0.86, E 0.77) | 0.81 [0.81, 0.82] (H 0.86, E 0.77) |
| single/random_vm_dmH | 0.81 (H 0.85, E 0.77) | 0.80 [0.80, 0.81] (H 0.85, E 0.77) | 0.80 (H 0.85, E 0.77) | 0.80 [0.80, 0.81] (H 0.85, E 0.76) |
| single/random_vm_dmE | 0.80 (H 0.85, E 0.76) | 0.80 [0.80, 0.81] (H 0.85, E 0.76) | 0.80 (H 0.85, E 0.76) | 0.80 [0.79, 0.81] (H 0.85, E 0.76) |
| single/random_vm_dmEperp | 0.80 (H 0.85, E 0.76) | 0.80 [0.80, 0.81] (H 0.85, E 0.76) | 0.80 (H 0.85, E 0.76) | 0.80 [0.80, 0.81] (H 0.85, E 0.76) |
| band/random_vm_dmH | 0.79 (H 0.83, E 0.76) | 0.79 [0.78, 0.80] (H 0.84, E 0.76) | 0.79 (H 0.84, E 0.75) | 0.79 [0.78, 0.80] (H 0.83, E 0.75) |
| band/random_vm_dmE | 0.79 (H 0.85, E 0.74) | 0.80 [0.79, 0.80] (H 0.85, E 0.75) | 0.78 (H 0.84, E 0.72) | 0.79 [0.78, 0.80] (H 0.85, E 0.74) |
| band/random_vm_dmEperp | 0.79 (H 0.85, E 0.74) | 0.80 [0.79, 0.80] (H 0.85, E 0.75) | 0.78 (H 0.85, E 0.72) | 0.79 [0.79, 0.80] (H 0.84, E 0.74) |

- subspace ranks: {'probe_sub': 2, 'dm_sub': 2, 'random_sub': 2}; {'cos(probe_H, dm_H) at L*': 0.66114342212677, 'cos(probe_E, dm_E) at L*': 0.6080994606018066, 'cos(probe_H_perp, probe_H) at L*': 0.750259518623352, 'cos(probe_E_perp, probe_E) at L*': 0.7938608527183533}
- variance-matched random directions at L*, fold 0: random_vm_dmH: share 0.0160 (target 0.0160), tilt 1.4, cos top eigvec 0.37 (ref 0.10), PR 26.3; random_vm_dmE: share 0.0368 (target 0.0368), tilt 1.5, cos top eigvec 0.67 (ref 0.63), PR 4.7; random_vm_dmEperp: share 0.0468 (target 0.0468), tilt 5.3, cos top eigvec 0.66 (ref 0.74), PR 3.0
- projection check: largest |removed coordinate| left at L* across conditions = 0.0486
- 962s on mps
