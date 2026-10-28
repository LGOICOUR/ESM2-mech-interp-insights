"""P1 data: proteins with per-residue three-state secondary structure (helix /
strand / coil), DSSP-derived, from the NetSurfP training/test bundle (DTU Health
Tech) as mirrored on Hugging Face.

Q8 -> Q3 mapping used by NetSurfP (and here): G, H, I -> H (helix); B, E -> E
(strand); S, T, C -> C (coil). Unlabelled residues (the bundle's evaluation mask
= 0, written '.' in the mirror) stay in the sequence the model reads but are
excluded from every probe and every metric.

The mirror's CB513 split lists one row per CB513 *domain* but gives the whole
PDB chain as the sequence, so multi-domain chains appear several times with
different unlabelled regions. `dedupe()` merges identical sequences and unions
their labels; chains whose rows disagree on a labelled residue are dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HF_REPO = "lamm-mit/protein-secondary-structure-netsurfp"
HF_FILES = {"cb513": "cb513.jsonl", "ts115": "ts115.jsonl"}
CLASSES = ("H", "E", "C")  # label index 0, 1, 2
CLASS_NAMES = {"H": "helix", "E": "strand", "C": "coil"}
UNLABELLED = "."
Q8_TO_Q3 = {"G": "H", "H": "H", "I": "H", "B": "E", "E": "E", "S": "C", "T": "C", "C": "C"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"  # 20 standard + X (unknown); index into one-hot
MAX_LEN = 1022  # ESM-2 position budget minus <cls> and <eos>


@dataclass
class Protein:
    id: str
    split: str
    sequence: str
    labels: str  # one character per residue: H / E / C / '.'

    def __post_init__(self) -> None:
        if len(self.sequence) != len(self.labels):
            raise ValueError(f"{self.id}: sequence and labels differ in length")
        bad = set(self.labels) - set(CLASSES) - {UNLABELLED}
        if bad:
            raise ValueError(f"{self.id}: unknown label characters {sorted(bad)}")

    @property
    def n_labelled(self) -> int:
        return sum(ch != UNLABELLED for ch in self.labels)

    def label_array(self) -> np.ndarray:
        """int8 per residue: 0 = H, 1 = E, 2 = C, -1 = unlabelled."""
        return np.array([CLASSES.index(ch) if ch != UNLABELLED else -1 for ch in self.labels], dtype=np.int8)

    def aa_array(self) -> np.ndarray:
        """int8 per residue: index into AMINO_ACIDS, with any non-standard letter mapped to X."""
        return np.array([AMINO_ACIDS.index(ch) if ch in AMINO_ACIDS else AMINO_ACIDS.index("X")
                         for ch in self.sequence], dtype=np.int8)


def q8_to_q3(q8: str) -> str:
    return "".join(Q8_TO_Q3[ch] if ch != UNLABELLED else UNLABELLED for ch in q8)


def parse_jsonl(path: Path, split: str) -> list[Protein]:
    """One mirror row per line: {"sequence", "labels", ...}. Labels are already Q3 + '.'."""
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(Protein(id=f"{split}_{i:04d}", split=split, sequence=r["sequence"].upper(), labels=r["labels"].upper()))
    return out


def merge_labels(a: str, b: str) -> str | None:
    """Union of two label strings for the same sequence; None if they disagree on a labelled residue."""
    out = []
    for x, y in zip(a, b):
        if x == UNLABELLED:
            out.append(y)
        elif y == UNLABELLED or x == y:
            out.append(x)
        else:
            return None
    return "".join(out)


def dedupe(proteins: list[Protein]) -> tuple[list[Protein], dict]:
    """Merge rows with identical sequences (union of labels). Returns the merged
    list (first-seen order) and a report dict."""
    by_seq: dict[str, Protein] = {}
    conflicts: set[str] = set()
    n_merged = 0
    for p in proteins:
        if p.sequence in by_seq:
            q = by_seq[p.sequence]
            merged = merge_labels(q.labels, p.labels)
            if merged is None:
                conflicts.add(p.sequence)
            else:
                by_seq[p.sequence] = Protein(id=q.id, split=q.split, sequence=q.sequence, labels=merged)
            n_merged += 1
        else:
            by_seq[p.sequence] = p
    kept = [p for s, p in by_seq.items() if s not in conflicts]
    return kept, {"n_in": len(proteins), "n_duplicate_rows": n_merged, "n_conflicting_chains_dropped": len(conflicts), "n_out": len(kept)}


def filter_proteins(proteins: list[Protein], max_len: int = MAX_LEN, min_labelled: int = 1) -> tuple[list[Protein], dict]:
    kept, too_long, no_labels = [], 0, 0
    for p in proteins:
        if len(p.sequence) > max_len:
            too_long += 1
        elif p.n_labelled < min_labelled:
            no_labels += 1
        else:
            kept.append(p)
    return kept, {"n_too_long": too_long, "n_without_labels": no_labels, "n_out": len(kept)}


def write_jsonl(proteins: list[Protein], path: Path) -> None:
    with Path(path).open("w") as f:
        for p in proteins:
            f.write(json.dumps({"id": p.id, "split": p.split, "sequence": p.sequence, "labels": p.labels}) + "\n")


def read_jsonl(path: Path) -> list[Protein]:
    return [Protein(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def residue_table(proteins: list[Protein]) -> dict[str, np.ndarray]:
    """Flatten labelled residues in protein order. Returns arrays over labelled
    residues only: protein_idx, position (0-based within the chain), label, aa."""
    pidx, pos, lab, aa = [], [], [], []
    for i, p in enumerate(proteins):
        la, aa_i = p.label_array(), p.aa_array()
        keep = np.where(la >= 0)[0]
        pidx.append(np.full(len(keep), i, dtype=np.int32))
        pos.append(keep.astype(np.int32))
        lab.append(la[keep])
        aa.append(aa_i[keep])
    return {"protein_idx": np.concatenate(pidx), "position": np.concatenate(pos),
            "label": np.concatenate(lab).astype(np.int8), "aa": np.concatenate(aa).astype(np.int8)}


def one_hot_window(proteins: list[Protein], table: dict[str, np.ndarray], half_window: int) -> np.ndarray:
    """No-model baseline features for each labelled residue: one-hot amino-acid
    identity of the residue and its +/- half_window neighbours (a chain-edge slot
    marks positions past either end). half_window=0 is identity alone."""
    n_aa = len(AMINO_ACIDS) + 1  # + edge slot
    width = 2 * half_window + 1
    X = np.zeros((len(table["label"]), width * n_aa), dtype=np.float32)
    aa_by_protein = [p.aa_array() for p in proteins]
    for r, (pi, pos) in enumerate(zip(table["protein_idx"], table["position"])):
        seq = aa_by_protein[pi]
        for k, off in enumerate(range(-half_window, half_window + 1)):
            j = pos + off
            slot = int(seq[j]) if 0 <= j < len(seq) else n_aa - 1
            X[r, k * n_aa + slot] = 1.0
    return X
