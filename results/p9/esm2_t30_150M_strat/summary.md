# P9 layer sweep, `esm2_t30_150M`: class-mean plane removed at one layer at a time; 548 held-out chains, 113279 labelled residues; L* = 29

Baseline top-1 39.2%. ΔPLL in nats, mean [95% CI, chains resampled]; P3 columns are decodability at that layer before any edit (macro-F1).

| layer | ΔPLL dm_sub | helix | strand | coil | top-1 | ΔPLL random plane | P3 probe F1 | P3 class-mean F1 | other kinds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.097 [-0.111, -0.083] | -0.086 | -0.149 | -0.081 | 36.3% |  | 0.496 | 0.486 | dm_strat_sub -0.001 |
| 29 (L*) | -0.062 [-0.065, -0.059] | -0.074 | -0.049 | -0.059 | 37.9% |  | 0.815 | 0.766 | dm_strat_sub -0.049 |
| 30 | -0.146 [-0.152, -0.140] | -0.124 | -0.076 | -0.199 | 36.4% | -0.000 [-0.000, +0.000] | 0.812 | 0.717 | dm_strat_sub -0.076 |

- 1240s on mps
