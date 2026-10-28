"""Offline tests for P1 parsing, label mapping, de-duplication and baseline features."""

import numpy as np
import pytest

from esm2_probe.data import (AMINO_ACIDS, Protein, dedupe, filter_proteins, merge_labels, one_hot_window,
                             parse_jsonl, q8_to_q3, residue_table)


def test_q8_to_q3_mapping_is_netsurfp():
    assert q8_to_q3("GHIBESTC.") == "HHHEECCC."


def test_protein_rejects_length_mismatch_and_bad_labels():
    with pytest.raises(ValueError):
        Protein("x", "t", "ACD", "HH")
    with pytest.raises(ValueError):
        Protein("x", "t", "ACD", "HXC")


def test_merge_labels_unions_and_detects_conflicts():
    assert merge_labels("HH..", "..CC") == "HHCC"
    assert merge_labels("HH..", "H.CC") == "HHCC"
    assert merge_labels("HH..", "EH..") is None


def test_dedupe_merges_domain_rows_of_one_chain():
    rows = [Protein("a", "cb", "ACDEFG", "HHH..."), Protein("b", "cb", "ACDEFG", "...EEC"),
            Protein("c", "cb", "KLMN", "CCCC"), Protein("d", "cb", "KLMN", "CCCE")]
    kept, rep = dedupe(rows)
    assert [p.id for p in kept] == ["a"]  # the conflicting KLMN chain is dropped
    assert kept[0].labels == "HHHEEC"
    assert rep == {"n_in": 4, "n_duplicate_rows": 2, "n_conflicting_chains_dropped": 1, "n_out": 1}


def test_filter_drops_long_and_unlabelled_chains():
    rows = [Protein("a", "t", "ACD", "HEC"), Protein("b", "t", "ACDE", "...."), Protein("c", "t", "A" * 5, "H" * 5)]
    kept, rep = filter_proteins(rows, max_len=4)
    assert [p.id for p in kept] == ["a"] and rep["n_too_long"] == 1 and rep["n_without_labels"] == 1


def test_parse_jsonl_reads_mirror_rows(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"sequence": "acd", "labels": "h.c", "seq_length": 3}\n\n{"sequence": "KK", "labels": "EE"}\n')
    ps = parse_jsonl(f, "ts115")
    assert [(p.id, p.sequence, p.labels) for p in ps] == [("ts115_0000", "ACD", "H.C"), ("ts115_0002", "KK", "EE")]


def test_residue_table_keeps_only_labelled_residues_in_order():
    ps = [Protein("a", "t", "ACDX", "H.CE"), Protein("b", "t", "KL", "EE")]
    t = residue_table(ps)
    assert t["protein_idx"].tolist() == [0, 0, 0, 1, 1]
    assert t["position"].tolist() == [0, 2, 3, 0, 1]
    assert t["label"].tolist() == [0, 2, 1, 1, 1]
    assert t["aa"].tolist() == [AMINO_ACIDS.index(c) for c in "ADXKL"]


def test_one_hot_window_marks_identity_neighbours_and_edges():
    ps = [Protein("a", "t", "ACD", "HHH")]
    t = residue_table(ps)
    n_slot = len(AMINO_ACIDS) + 1
    X0 = one_hot_window(ps, t, 0)
    assert X0.shape == (3, n_slot) and X0.sum() == 3
    assert X0[1, AMINO_ACIDS.index("C")] == 1
    X1 = one_hot_window(ps, t, 1)
    assert X1.shape == (3, 3 * n_slot) and np.all(X1.sum(1) == 3)
    # first residue: left neighbour is the edge slot, centre is A, right is C
    assert X1[0, n_slot - 1] == 1 and X1[0, n_slot + AMINO_ACIDS.index("A")] == 1 and X1[0, 2 * n_slot + AMINO_ACIDS.index("C")] == 1
