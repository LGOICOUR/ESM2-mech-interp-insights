# P6 nonlinear refit after linear erasure at layer 29, `esm2_t30_150M`

Post hoc on the P2 recording: the fold's plane (fit on training chains as in P4) is projected out of the held-out chains' layer-29 rows; each reader is then trained on the edited rows with the same chain-grouped folds and scored on held-out chains. Macro-F1 [95% CI, chains resampled] (H, E, C = per-class F1). Floor 0.20 (always coil); no-model baselines: residue identity 0.44, +/-3 window 0.57. Readers: linear = multinomial logistic regression (C = 0.001, 40000 residues per fold, the P4 refit); mlp = one hidden layer of 256 ReLU units, early stopping, 89181 residues per fold; svm = RBF kernel, C = 1, gamma = scale, 15000 residues per fold.

| removed at layer 29 | linear | mlp | svm |
|---|---|---|---|
| none | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.81) | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.80) | 0.82 [0.81, 0.82] (H 0.87, E 0.77, C 0.81) |
| dm_sub | 0.22 [0.21, 0.22] (H 0.16, E 0.01, C 0.49) | 0.79 [0.78, 0.80] (H 0.84, E 0.75, C 0.78) | 0.78 [0.77, 0.78] (H 0.84, E 0.69, C 0.80) |
| random_sub | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.81) | 0.81 [0.80, 0.82] (H 0.86, E 0.77, C 0.80) | 0.82 [0.81, 0.82] (H 0.87, E 0.77, C 0.81) |
| probe_sub | 0.71 [0.70, 0.71] (H 0.75, E 0.62, C 0.75) | 0.80 [0.79, 0.80] (H 0.84, E 0.75, C 0.79) | 0.77 [0.76, 0.77] (H 0.82, E 0.70, C 0.78) |
| dm_H | 0.68 [0.67, 0.69] (H 0.60, E 0.73, C 0.72) | 0.80 [0.80, 0.81] (H 0.85, E 0.77, C 0.79) | 0.80 [0.80, 0.81] (H 0.84, E 0.77, C 0.80) |
| probe_H | 0.76 [0.75, 0.76] (H 0.77, E 0.71, C 0.78) | 0.80 [0.79, 0.81] (H 0.84, E 0.76, C 0.80) | 0.79 [0.78, 0.80] (H 0.83, E 0.75, C 0.80) |

Leftover coordinate along the removed plane on held-out rows (max |(x - mu) . u|): none 0.0e+00, dm_sub 1.5e-04, random_sub 3.1e-05, probe_sub 4.8e-05, dm_H 1.5e-05, probe_H 5.2e-06
