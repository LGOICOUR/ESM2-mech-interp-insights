# P7 sparse-position ablation, `esm2_t30_150M`: L* = 29, 548 held-out chains, 113279 labelled residues

The class-mean plane (dm_sub) is projected out at L* at a random fraction p of each chain's residue positions (same draw for every p; random_sub = a random rank-2 plane at the same p = 0.5 positions). Rows are grouped by whether the row itself was ablated and how many of its sequence neighbours (within +-1 or +-3) were ablated.

## Decodability of the next layer at ablated rows

Macro-F1 of a probe refit on the ablated rows only (cv over the same chain-grouped folds) at layer 30, and at layer 29 [95% CI, chains resampled]. For p = 1 this is P4's refit.

| condition | row group | n | layer 30 | layer 29 |
|---|---|---|---|---|
| none | all | 113279 | 0.812 [0.806, 0.818] | 0.815 [0.809, 0.820] |
| none | intact | 113279 | 0.812 [0.806, 0.818] | 0.815 [0.809, 0.820] |
| none | intact, 0 of +-1 ablated | 113279 | 0.812 [0.806, 0.818] | 0.815 [0.809, 0.820] |
| dm_sub/p1 | all | 113279 | 0.565 [0.557, 0.572] | 0.218 [0.213, 0.222] |
| dm_sub/p1 | ablated | 113279 | 0.565 [0.557, 0.572] | 0.218 [0.213, 0.222] |
| dm_sub/p1 | ablated, 2 of +-1 ablated | 112551 | 0.564 [0.556, 0.572] | 0.217 [0.213, 0.222] |
| dm_sub/p1 | ablated, 6 of +-3 ablated | 110976 | 0.564 [0.556, 0.572] | 0.216 [0.212, 0.221] |
| dm_sub/p0.1 | all | 11508 | 0.674 [0.663, 0.686] | 0.261 [0.251, 0.270] |
| dm_sub/p0.1 | ablated | 11508 | 0.674 [0.663, 0.686] | 0.261 [0.251, 0.270] |
| dm_sub/p0.1 | ablated, 0 of +-1 ablated | 9389 | 0.683 [0.672, 0.695] | 0.258 [0.248, 0.267] |
| dm_sub/p0.1 | ablated, 2 of +-1 ablated | 112 | 0.622 [0.496, 0.712] | 0.246 [0.189, 0.292] |
| dm_sub/p0.1 | ablated, 0 of +-3 ablated | 6082 | 0.693 [0.679, 0.707] | 0.259 [0.249, 0.270] |
| dm_sub/p0.5 | all | 56797 | 0.622 [0.614, 0.631] | 0.229 [0.224, 0.234] |
| dm_sub/p0.5 | ablated | 56797 | 0.622 [0.614, 0.631] | 0.229 [0.224, 0.234] |
| dm_sub/p0.5 | ablated, 0 of +-1 ablated | 14297 | 0.662 [0.651, 0.674] | 0.227 [0.221, 0.234] |
| dm_sub/p0.5 | ablated, 2 of +-1 ablated | 14069 | 0.576 [0.564, 0.589] | 0.233 [0.224, 0.239] |
| dm_sub/p0.5 | ablated, 0 of +-3 ablated | 861 | 0.692 [0.661, 0.726] | 0.243 [0.219, 0.263] |
| dm_sub/p0.5 | ablated, 6 of +-3 ablated | 788 | 0.548 [0.503, 0.595] | 0.241 [0.215, 0.267] |
| dm_sub/p0.9 | all | 101986 | 0.573 [0.565, 0.581] | 0.232 [0.226, 0.237] |
| dm_sub/p0.9 | ablated | 101986 | 0.573 [0.565, 0.581] | 0.232 [0.226, 0.237] |
| dm_sub/p0.9 | ablated, 0 of +-1 ablated | 1043 | 0.635 [0.603, 0.661] | 0.235 [0.215, 0.259] |
| dm_sub/p0.9 | ablated, 2 of +-1 ablated | 82051 | 0.567 [0.560, 0.576] | 0.232 [0.225, 0.237] |
| dm_sub/p0.9 | ablated, 6 of +-3 ablated | 52833 | 0.565 [0.557, 0.573] | 0.232 [0.227, 0.238] |
| random_sub/p0.5 | all | 56797 | 0.812 [0.806, 0.819] | 0.815 [0.809, 0.821] |
| random_sub/p0.5 | ablated | 56797 | 0.812 [0.806, 0.819] | 0.815 [0.809, 0.821] |
| random_sub/p0.5 | ablated, 0 of +-1 ablated | 14297 | 0.815 [0.807, 0.823] | 0.819 [0.811, 0.826] |
| random_sub/p0.5 | ablated, 2 of +-1 ablated | 14069 | 0.814 [0.805, 0.823] | 0.816 [0.806, 0.826] |
| random_sub/p0.5 | ablated, 0 of +-3 ablated | 861 | 0.830 [0.799, 0.856] | 0.838 [0.811, 0.863] |
| random_sub/p0.5 | ablated, 6 of +-3 ablated | 788 | 0.831 [0.799, 0.865] | 0.838 [0.810, 0.871] |

## All readers, all rows

Macro-F1 at layer 30: original probe | refit on all rows | refit on ablated rows.

| condition | row group | original | refit all | refit ablated |
|---|---|---|---|---|
| none | all | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.818] | — |
| none | intact | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.818] | — |
| none | intact, 0 of +-1 ablated | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.818] | — |
| dm_sub/p1 | all | 0.374 [0.367, 0.383] | 0.565 [0.557, 0.572] | 0.565 [0.557, 0.572] |
| dm_sub/p1 | ablated | 0.374 [0.367, 0.383] | 0.565 [0.557, 0.572] | 0.565 [0.557, 0.572] |
| dm_sub/p1 | ablated, 2 of +-1 ablated | 0.374 [0.366, 0.383] | 0.564 [0.556, 0.572] | 0.564 [0.556, 0.572] |
| dm_sub/p1 | ablated, 6 of +-3 ablated | 0.375 [0.367, 0.384] | 0.564 [0.556, 0.572] | 0.564 [0.556, 0.572] |
| dm_sub/p0.1 | all | 0.776 [0.770, 0.782] | 0.782 [0.776, 0.788] | 0.674 [0.663, 0.686] |
| dm_sub/p0.1 | ablated | 0.459 [0.447, 0.470] | 0.555 [0.545, 0.566] | 0.674 [0.663, 0.686] |
| dm_sub/p0.1 | intact | 0.811 [0.805, 0.817] | 0.806 [0.800, 0.812] | — |
| dm_sub/p0.1 | ablated, 0 of +-1 ablated | 0.464 [0.453, 0.476] | 0.564 [0.553, 0.577] | 0.683 [0.672, 0.695] |
| dm_sub/p0.1 | ablated, 2 of +-1 ablated | 0.481 [0.371, 0.577] | 0.545 [0.441, 0.632] | 0.622 [0.496, 0.712] |
| dm_sub/p0.1 | ablated, 0 of +-3 ablated | 0.468 [0.454, 0.481] | 0.574 [0.560, 0.585] | 0.693 [0.679, 0.707] |
| dm_sub/p0.1 | intact, 0 of +-1 ablated | 0.812 [0.806, 0.818] | 0.807 [0.800, 0.812] | — |
| dm_sub/p0.1 | intact, 2 of +-1 ablated | 0.801 [0.776, 0.823] | 0.792 [0.767, 0.813] | — |
| dm_sub/p0.5 | all | 0.616 [0.609, 0.622] | 0.685 [0.679, 0.692] | 0.622 [0.614, 0.631] |
| dm_sub/p0.5 | ablated | 0.422 [0.413, 0.430] | 0.591 [0.583, 0.599] | 0.622 [0.614, 0.631] |
| dm_sub/p0.5 | intact | 0.808 [0.802, 0.815] | 0.771 [0.763, 0.777] | — |
| dm_sub/p0.5 | ablated, 0 of +-1 ablated | 0.442 [0.432, 0.451] | 0.635 [0.624, 0.647] | 0.662 [0.651, 0.674] |
| dm_sub/p0.5 | ablated, 2 of +-1 ablated | 0.396 [0.384, 0.407] | 0.537 [0.524, 0.548] | 0.576 [0.564, 0.589] |
| dm_sub/p0.5 | ablated, 0 of +-3 ablated | 0.486 [0.450, 0.517] | 0.675 [0.640, 0.707] | 0.692 [0.661, 0.726] |
| dm_sub/p0.5 | ablated, 6 of +-3 ablated | 0.396 [0.364, 0.427] | 0.516 [0.474, 0.562] | 0.548 [0.503, 0.595] |
| dm_sub/p0.5 | intact, 0 of +-1 ablated | 0.813 [0.804, 0.821] | 0.776 [0.766, 0.785] | — |
| dm_sub/p0.5 | intact, 2 of +-1 ablated | 0.804 [0.796, 0.812] | 0.762 [0.752, 0.770] | — |
| dm_sub/p0.9 | all | 0.427 [0.420, 0.435] | 0.580 [0.573, 0.588] | 0.573 [0.565, 0.581] |
| dm_sub/p0.9 | ablated | 0.385 [0.377, 0.394] | 0.562 [0.555, 0.570] | 0.573 [0.565, 0.581] |
| dm_sub/p0.9 | intact | 0.805 [0.796, 0.813] | 0.729 [0.719, 0.739] | — |
| dm_sub/p0.9 | ablated, 0 of +-1 ablated | 0.423 [0.392, 0.453] | 0.613 [0.578, 0.648] | 0.635 [0.603, 0.661] |
| dm_sub/p0.9 | ablated, 2 of +-1 ablated | 0.380 [0.373, 0.390] | 0.557 [0.550, 0.566] | 0.567 [0.560, 0.576] |
| dm_sub/p0.9 | ablated, 6 of +-3 ablated | 0.379 [0.369, 0.390] | 0.555 [0.546, 0.564] | 0.565 [0.557, 0.573] |
| dm_sub/p0.9 | intact, 0 of +-1 ablated | 0.778 [0.678, 0.859] | 0.758 [0.671, 0.842] | — |
| dm_sub/p0.9 | intact, 2 of +-1 ablated | 0.804 [0.794, 0.813] | 0.726 [0.715, 0.736] | — |
| random_sub/p0.5 | all | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.819] |
| random_sub/p0.5 | ablated | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.818] | 0.812 [0.806, 0.819] |
| random_sub/p0.5 | intact | 0.812 [0.805, 0.818] | 0.812 [0.806, 0.819] | — |
| random_sub/p0.5 | ablated, 0 of +-1 ablated | 0.816 [0.808, 0.823] | 0.816 [0.808, 0.823] | 0.815 [0.807, 0.823] |
| random_sub/p0.5 | ablated, 2 of +-1 ablated | 0.812 [0.803, 0.822] | 0.811 [0.803, 0.821] | 0.814 [0.805, 0.823] |
| random_sub/p0.5 | ablated, 0 of +-3 ablated | 0.837 [0.808, 0.862] | 0.839 [0.811, 0.864] | 0.830 [0.799, 0.856] |
| random_sub/p0.5 | ablated, 6 of +-3 ablated | 0.829 [0.798, 0.862] | 0.825 [0.794, 0.858] | 0.831 [0.799, 0.865] |
| random_sub/p0.5 | intact, 0 of +-1 ablated | 0.814 [0.805, 0.822] | 0.815 [0.806, 0.823] | — |
| random_sub/p0.5 | intact, 2 of +-1 ablated | 0.810 [0.801, 0.817] | 0.810 [0.802, 0.818] | — |

## The model's own output

Change in masked log-likelihood of the true residue vs the unablated run (nats, mean [95% CI]) and top-1 accuracy, by row group.

| condition | row group | n | ΔPLL | top-1 acc |
|---|---|---|---|---|
| none | all | 113279 | +0.000 [+0.000, +0.000] | 39.2% |
| none | intact | 113279 | +0.000 [+0.000, +0.000] | 39.2% |
| none | intact, 0 of +-1 ablated | 113279 | +0.000 [+0.000, +0.000] | 39.2% |
| dm_sub/p1 | all | 113279 | -0.062 [-0.065, -0.059] | 37.9% |
| dm_sub/p1 | ablated | 113279 | -0.062 [-0.065, -0.059] | 37.9% |
| dm_sub/p1 | ablated, 2 of +-1 ablated | 112551 | -0.061 [-0.064, -0.059] | 37.9% |
| dm_sub/p1 | ablated, 6 of +-3 ablated | 110976 | -0.061 [-0.064, -0.059] | 38.1% |
| dm_sub/p0.1 | all | 113279 | -0.005 [-0.006, -0.005] | 39.1% |
| dm_sub/p0.1 | ablated | 11508 | -0.040 [-0.044, -0.035] | 38.6% |
| dm_sub/p0.1 | intact | 101771 | -0.001 [-0.002, -0.001] | 39.1% |
| dm_sub/p0.1 | ablated, 0 of +-1 ablated | 9389 | -0.040 [-0.044, -0.035] | 38.8% |
| dm_sub/p0.1 | ablated, 2 of +-1 ablated | 112 | -0.049 [-0.103, +0.006] | 37.5% |
| dm_sub/p0.1 | ablated, 0 of +-3 ablated | 6082 | -0.040 [-0.046, -0.034] | 38.8% |
| dm_sub/p0.1 | intact, 0 of +-1 ablated | 82130 | -0.001 [-0.001, -0.001] | 39.1% |
| dm_sub/p0.1 | intact, 2 of +-1 ablated | 1069 | -0.005 [-0.010, +0.000] | 42.0% |
| dm_sub/p0.5 | all | 113279 | -0.028 [-0.029, -0.026] | 38.6% |
| dm_sub/p0.5 | ablated | 56797 | -0.044 [-0.047, -0.042] | 38.5% |
| dm_sub/p0.5 | intact | 56482 | -0.011 [-0.012, -0.009] | 38.8% |
| dm_sub/p0.5 | ablated, 0 of +-1 ablated | 14297 | -0.043 [-0.047, -0.039] | 37.6% |
| dm_sub/p0.5 | ablated, 2 of +-1 ablated | 14069 | -0.051 [-0.056, -0.046] | 38.7% |
| dm_sub/p0.5 | ablated, 0 of +-3 ablated | 861 | -0.055 [-0.072, -0.040] | 36.0% |
| dm_sub/p0.5 | ablated, 6 of +-3 ablated | 788 | -0.059 [-0.080, -0.039] | 43.4% |
| dm_sub/p0.5 | intact, 0 of +-1 ablated | 13955 | -0.008 [-0.010, -0.006] | 38.9% |
| dm_sub/p0.5 | intact, 2 of +-1 ablated | 14140 | -0.013 [-0.016, -0.010] | 38.9% |
| dm_sub/p0.9 | all | 113279 | -0.054 [-0.057, -0.052] | 38.0% |
| dm_sub/p0.9 | ablated | 101986 | -0.057 [-0.060, -0.055] | 37.9% |
| dm_sub/p0.9 | intact | 11293 | -0.028 [-0.032, -0.023] | 38.1% |
| dm_sub/p0.9 | ablated, 0 of +-1 ablated | 1043 | -0.068 [-0.087, -0.052] | 40.7% |
| dm_sub/p0.9 | ablated, 2 of +-1 ablated | 82051 | -0.057 [-0.060, -0.055] | 37.9% |
| dm_sub/p0.9 | ablated, 6 of +-3 ablated | 52833 | -0.058 [-0.062, -0.055] | 38.1% |
| dm_sub/p0.9 | intact, 0 of +-1 ablated | 103 | -0.016 [-0.050, +0.011] | 38.8% |
| dm_sub/p0.9 | intact, 2 of +-1 ablated | 9151 | -0.028 [-0.033, -0.022] | 38.3% |
| random_sub/p0.5 | all | 113279 | -0.000 [-0.000, -0.000] | 39.2% |
| random_sub/p0.5 | ablated | 56797 | -0.001 [-0.001, -0.000] | 39.4% |
| random_sub/p0.5 | intact | 56482 | -0.000 [-0.000, -0.000] | 38.9% |
| random_sub/p0.5 | ablated, 0 of +-1 ablated | 14297 | -0.001 [-0.001, -0.000] | 38.7% |
| random_sub/p0.5 | ablated, 2 of +-1 ablated | 14069 | -0.000 [-0.001, +0.000] | 39.7% |
| random_sub/p0.5 | ablated, 0 of +-3 ablated | 861 | +0.001 [-0.001, +0.003] | 38.1% |
| random_sub/p0.5 | ablated, 6 of +-3 ablated | 788 | -0.001 [-0.003, +0.002] | 44.2% |
| random_sub/p0.5 | intact, 0 of +-1 ablated | 13955 | -0.000 [-0.000, +0.000] | 39.0% |
| random_sub/p0.5 | intact, 2 of +-1 ablated | 14140 | -0.000 [-0.000, -0.000] | 38.9% |

- projection check: largest |removed coordinate| left at ablated rows = 0.0334
- 846s on mps
