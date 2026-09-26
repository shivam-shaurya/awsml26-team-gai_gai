import pytest

from submission import self_check_predictions, write_id_list_file


def test_write_id_list_file_dedupes_and_handles_empty(tmp_path):
    path = tmp_path / "out.tsv"
    write_id_list_file(str(path), ["S1-1", "S1-2"],
                       {"S1-1": ["S2-1", "S2-1", "S3-1"]}, "matched_entity_ids")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-1\tS2-1,S3-1"  # de-duplicated, order preserved
    assert lines[2] == "S1-2\t"           # no entry -> empty list, not a missing row


def test_self_check_rejects_prediction_not_in_valid_targets():
    with pytest.raises(ValueError, match="do not exist"):
        self_check_predictions(
            predictions={"S1-1": ["S2-999"]},
            candidates={"S1-1": ["S2-999"]},
            test_s1_ids=["S1-1"],
            valid_target_ids={"S2-1"},
        )


def test_self_check_rejects_prediction_not_in_own_candidates():
    # A matched id that was never a candidate signals a pipeline bug (matches must come from
    # blocking, never be invented downstream).
    with pytest.raises(ValueError, match="never a candidate"):
        self_check_predictions(
            predictions={"S1-1": ["S2-1"]},
            candidates={"S1-1": ["S2-2"]},  # S2-1 was never proposed as a candidate
            test_s1_ids=["S1-1"],
            valid_target_ids={"S2-1", "S2-2"},
        )


def test_self_check_passes_for_valid_subset_predictions():
    self_check_predictions(
        predictions={"S1-1": ["S2-1"]},
        candidates={"S1-1": ["S2-1", "S2-2"]},
        test_s1_ids=["S1-1", "S1-2"],  # S1-2 is an implicit singleton, not an error
        valid_target_ids={"S2-1", "S2-2"},
    )
