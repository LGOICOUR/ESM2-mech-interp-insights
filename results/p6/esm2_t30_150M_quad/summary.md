# P6 nonlinear refit after linear erasure at layer 29, `esm2_t30_150M`

Post hoc on the P2 recording: the fold's plane (fit on training chains as in P4) is projected out of the held-out chains' layer-29 rows; each reader is then trained on the edited rows with the same chain-grouped folds and scored on held-out chains. Macro-F1 [95% CI, chains resampled] (H, E, C = per-class F1). Floor 0.20 (always coil); no-model baselines: residue identity 0.44, +/-3 window 0.57. Readers: linear = multinomial logistic regression (C = 0.001, 40000 residues per fold, the P4 refit); mlp = one hidden layer of 256 ReLU units, early stopping, 89181 residues per fold; svm = RBF kernel, C = 1, gamma = scale, 15000 residues per fold; quad = logistic regression on the coordinates and their squares (axis-aligned ellipsoids, C by inner CV, 40000 residues per fold); qda = one mean and one full covariance per class, 89181 residues per fold.

| removed at layer 29 | linear | quad | qda |
|---|---|---|---|
| none | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.81) | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.80) | 0.81 [0.80, 0.81] (H 0.86, E 0.77, C 0.79) |
| dm_sub | 0.22 [0.21, 0.22] (H 0.16, E 0.01, C 0.49) | 0.56 [0.56, 0.57] (H 0.59, E 0.46, C 0.64) | 0.81 [0.80, 0.81] (H 0.86, E 0.77, C 0.79) |
| random_sub | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.81) | 0.81 [0.81, 0.82] (H 0.86, E 0.77, C 0.80) | 0.81 [0.80, 0.81] (H 0.86, E 0.77, C 0.79) |

Shape after `none` (fold 0 held-out rows): class-mean distances in pooled-sd units H-C 5.36, E-C 5.27, H-E 5.07; variance ratios (class a / class b along the generalized eigen-directions): H/C: max 16.0, min 0.07, median 0.98, 120 dirs > 2, 135 dirs < 0.5; E/C: max 19.6, min 0.08, median 0.98, 128 dirs > 2, 139 dirs < 0.5; H/E: max 35.7, min 0.03, median 0.99, 154 dirs > 2, 157 dirs < 0.5

Shape after `dm_sub` (fold 0 held-out rows): class-mean distances in pooled-sd units H-C 1.16, E-C 0.95, H-E 1.73; variance ratios (class a / class b along the generalized eigen-directions): H/C: max 15.5, min 0.07, median 0.98, 119 dirs > 2, 134 dirs < 0.5; E/C: max 15.2, min 0.09, median 0.98, 127 dirs > 2, 138 dirs < 0.5; H/E: max 34.5, min 0.03, median 0.99, 153 dirs > 2, 156 dirs < 0.5

Shape after `random_sub` (fold 0 held-out rows): class-mean distances in pooled-sd units H-C 5.36, E-C 5.28, H-E 5.07; variance ratios (class a / class b along the generalized eigen-directions): H/C: max 15.9, min 0.07, median 0.98, 118 dirs > 2, 134 dirs < 0.5; E/C: max 19.6, min 0.08, median 0.98, 127 dirs > 2, 139 dirs < 0.5; H/E: max 35.7, min 0.03, median 0.99, 154 dirs > 2, 157 dirs < 0.5

Leftover coordinate along the removed plane on held-out rows (max |(x - mu) . u|): none 0.0e+00, dm_sub 1.5e-04, random_sub 3.1e-05
