# P1 data

Source: `lamm-mit/protein-secondary-structure-netsurfp` @ `84141978b55a`, files ts115.jsonl, cb513.jsonl (NetSurfP bundle: DSSP three-state labels; '.' = unlabelled).

- ts115: 115 rows, 29704 residues, 29704 labelled
- cb513: 513 rows, 144360 residues, 84686 labelled
- de-duplication by sequence: 628 rows -> 549 chains (79 duplicate rows merged, 0 chains dropped for label conflicts)
- filter: 1 chains over the length budget, 0 without labels
- kept: **548 chains** ({'ts115': 114, 'cb513': 434}), 135365 residues, **113279 labelled**
- class fraction among labelled residues: H 0.354, E 0.214, C 0.431
- chain length: min 20, median 213, mean 247, max 874
