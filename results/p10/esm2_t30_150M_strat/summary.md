# P10 sequential multi-layer erasure, `esm2_t30_150M`: 548 held-out chains, 113279 labelled residues; L* = 29

Class-mean plane (rank 2) removed at every layer of a band. `sequential`: each plane fit on activations already edited by the planes below it. `stale`: every plane fit on the unedited recording (P4's band). `random`: a random rank-2 plane at each layer. `sequential_strat`: the within-amino-acid class-mean plane (composition control), fit sequentially. ΔPLL in nats, mean [95% CI, chains resampled]; decodability = readers refit on the edited layer-30 rows (chain-grouped cv), macro-F1.

Baseline top-1 39.2%; layer 30 refit linear 0.812 [0.806, 0.818], QDA 0.807.

| condition | layers | ΔPLL all | helix | strand | coil | top-1 | layer 30 linear refit | layer 30 QDA |
|---|---|---|---|---|---|---|---|---|
| sequential_strat@29-30 | 29–30 | -0.058 [-0.061, -0.056] | -0.055 | -0.038 | -0.071 | 37.8% | 0.406 [0.397, 0.414] | 0.806 |
| sequential_strat@26-30 | 26–30 | -0.128 [-0.133, -0.124] | -0.137 | -0.144 | -0.113 | 36.1% | 0.383 [0.375, 0.391] | 0.780 |
| sequential_strat@20-30 | 20–30 | -0.405 [-0.426, -0.387] | -0.426 | -0.497 | -0.341 | 28.2% | 0.386 [0.378, 0.394] | 0.738 |

- 1805s on mps
