import csv
import os

import pandas as pd
import pytest

from data_loader import load_source_file, parse_ground_truth, read_tsv


def _write_tsv(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def test_read_tsv_preserves_literal_na_and_empty_strings(tmp_path):
    # A business literally named "NA", and a genuinely empty field, must survive as strings --
    # this is the #1 way pandas' default NA handling silently corrupts this dataset.
    path = tmp_path / "t.tsv"
    _write_tsv(path, [
        ("S1-1", "NA", "1 Main St", "US"),
        ("S1-2", "Acme", "", "US"),
    ], ["entity_id", "business_name", "business_address", "country"])
    df = read_tsv(str(path))
    assert df.loc[0, "business_name"] == "NA"
    assert df.loc[1, "business_address"] == ""
    assert df.entity_id.tolist() == ["S1-1", "S1-2"]  # never coerced to numeric


def test_load_source_file_rejects_comma_separated_file(tmp_path):
    # A comma-separated file read with sep="\t" keeps the same row count (read_tsv's own
    # line-count check can't see this), but collapses every column into one -- caught instead
    # by load_source_file's column-schema validation.
    (tmp_path / "train").mkdir()
    path = tmp_path / "train" / "train_source1.tsv"
    path.write_text("entity_id,business_name,business_address,country\nS1-1,Acme,1 Main St,US\n")
    with pytest.raises(ValueError, match="missing expected column"):
        load_source_file(str(tmp_path), "train", 1)


def test_load_source_file_rejects_wrong_prefix(tmp_path):
    (tmp_path / "train").mkdir()
    path = tmp_path / "train" / "train_source1.tsv"
    _write_tsv(path, [("S2-1", "Acme", "1 Main St", "US")],
              ["entity_id", "business_name", "business_address", "country"])
    with pytest.raises(ValueError, match="S1-"):
        load_source_file(str(tmp_path), "train", 1)


def test_load_source_file_rejects_duplicate_ids(tmp_path):
    (tmp_path / "train").mkdir()
    path = tmp_path / "train" / "train_source1.tsv"
    _write_tsv(path, [
        ("S1-1", "Acme", "1 Main St", "US"),
        ("S1-1", "Acme Robotics", "1 Main St", "US"),
    ], ["entity_id", "business_name", "business_address", "country"])
    with pytest.raises(ValueError, match="duplicate"):
        load_source_file(str(tmp_path), "train", 1)


def test_parse_ground_truth_empty_means_singleton(tmp_path):
    (tmp_path / "train").mkdir()
    path = tmp_path / "train" / "train_ground_truth.tsv"
    _write_tsv(path, [
        ("S1-1", "S2-1,S2-2"),
        ("S1-2", ""),
    ], ["source1_entity_id", "matched_entity_ids"])
    gt = parse_ground_truth(str(tmp_path), "train")
    assert gt.get("S1-1") == frozenset({"S2-1", "S2-2"})
    assert gt.is_singleton("S1-2")
    assert gt.is_singleton("S1-not-present")  # unknown id treated as no matches, not an error


def test_parse_ground_truth_rejects_self_match(tmp_path):
    (tmp_path / "train").mkdir()
    path = tmp_path / "train" / "train_ground_truth.tsv"
    _write_tsv(path, [("S1-1", "S1-2")], ["source1_entity_id", "matched_entity_ids"])
    with pytest.raises(ValueError, match="self-match"):
        parse_ground_truth(str(tmp_path), "train", s1_ids={"S1-1"}, s2_ids=set(), s3_ids=set())
