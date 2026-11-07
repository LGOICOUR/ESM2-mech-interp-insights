# P10 sequential multi-layer erasure, `esm2_t30_150M`: 548 held-out chains, 113279 labelled residues; L* = 29

Class-mean plane (rank 2) removed at every layer of a band. `sequential`: each plane fit on activations already edited by the planes below it. `stale`: every plane fit on the unedited recording (P4's band). `random`: a random rank-2 plane at each layer. ΔPLL in nats, mean [95% CI, chains resampled]; decodability = readers refit on the edited layer-30 rows (chain-grouped cv), macro-F1.

Baseline top-1 39.2%; layer 30 refit linear 0.812 [0.806, 0.818], QDA 0.807.

| condition | layers | ΔPLL all | helix | strand | coil | top-1 | layer 30 linear refit | layer 30 QDA |
|---|---|---|---|---|---|---|---|---|
| sequential@29-30 | 29–30 | -0.146 [-0.150, -0.142] | -0.127 | -0.084 | -0.192 | 36.0% | 0.228 [0.223, 0.233] | 0.805 |
| stale@29-30 | 29–30 | -0.160 [-0.165, -0.155] | -0.158 | -0.070 | -0.208 | 35.7% | 0.535 [0.528, 0.543] | 0.806 |
| random@29-30 | 29–30 | -0.001 [-0.001, -0.001] | -0.001 | -0.001 | -0.001 | 39.2% | 0.811 [0.805, 0.817] | 0.807 |
| sequential@26-30 | 26–30 | -0.334 [-0.347, -0.323] | -0.350 | -0.352 | -0.312 | 32.1% | 0.230 [0.224, 0.236] | 0.780 |
| stale@26-30 | 26–30 | -0.205 [-0.210, -0.200] | -0.207 | -0.148 | -0.233 | 34.0% | 0.715 [0.708, 0.722] | 0.789 |
| random@26-30 | 26–30 | -0.004 [-0.004, -0.003] | -0.004 | -0.004 | -0.003 | 39.0% | 0.812 [0.806, 0.817] | 0.807 |
| sequential@20-30 | 20–30 | -0.713 [-0.753, -0.681] | -0.707 | -0.883 | -0.635 | 19.4% | 0.243 [0.237, 0.249] | 0.737 |
| stale@20-30 | 20–30 | -0.327 [-0.337, -0.317] | -0.316 | -0.307 | -0.346 | 30.6% | 0.744 [0.737, 0.751] | 0.778 |
| random@20-30 | 20–30 | -0.013 [-0.014, -0.012] | -0.013 | -0.015 | -0.012 | 38.8% | 0.811 [0.805, 0.817] | 0.806 |

- 3306s on mps
