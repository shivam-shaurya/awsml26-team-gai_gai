"""Robust TSV loading, schema validation, and ground-truth parsing.

Every function here fails loudly on malformed input rather than silently producing a
partially-wrong DataFrame -- this is the #1 place competitions like this get quietly broken
(a `sep=","` typo, a business literally named "NA", an empty match list read as NaN).
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
SOURCE_PREFIXES = {1: "S1-", 2: "S2-", 3: "S3-"}


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV with the exact dtype/NA handling this format requires.

    - sep="\\t": these files use tabs specifically because addresses and ID lists contain
      commas; reading with the default comma separator would silently produce one column.
    - dtype=str + keep_default_na=False: entity_id must never be coerced to a number, and a
      business literally named "NA"/"NULL" or a genuinely empty match list must survive as
      the string "" / "NA", not become NaN.
    - quoting=csv.QUOTE_NONE: a stray `"` in a business name must not swallow the rest of the
      line into one field.
    - a line-count cross-check catches quoting/tab corruption pandas parsed through silently.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    with open(path, encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip()) - 1  # minus header
    if n_lines != len(df):
        raise ValueError(
            f"{path}: {n_lines} non-blank lines but pandas parsed {len(df)} rows -- "
            "this almost always means a stray quote or tab is corrupting the parse. "
            "Inspect the file before trusting anything downstream."
        )
    return df


def _validate_columns(df: pd.DataFrame, expected: list[str], path: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}; found {list(df.columns)}")


def load_source_file(data_dir: str, split: str, source_num: int) -> pd.DataFrame:
    """Load one of the three source files for a split, with prefix/schema validation."""
    path = os.path.join(data_dir, split, f"{split}_source{source_num}.tsv")
    df = read_tsv(path)
    _validate_columns(df, EXPECTED_SOURCE_COLUMNS, path)

    prefix = SOURCE_PREFIXES[source_num]
    bad_prefix = ~df.entity_id.str.startswith(prefix)
    if bad_prefix.any():
        examples = df.loc[bad_prefix, "entity_id"].head(5).tolist()
        raise ValueError(
            f"{path}: {bad_prefix.sum()} entity_id(s) do not start with '{prefix}' as "
            f"expected for source {source_num}, e.g. {examples}. Source identity comes from "
            "the entity_id prefix (there is no separate source column) -- do not assume it."
        )
    n_dupes = df.entity_id.duplicated().sum()
    if n_dupes:
        raise ValueError(f"{path}: {n_dupes} duplicate entity_id value(s) -- expected unique IDs.")

    log.info(
        "loaded %s: %s rows | countries: %s",
        path, f"{len(df):,}", df.country.value_counts().to_dict(),
    )
    return df


def load_split(data_dir: str, split: str) -> dict[int, pd.DataFrame]:
    """Load and validate all three source files for a split ("train" or "test")."""
    return {k: load_source_file(data_dir, split, k) for k in (1, 2, 3)}


@dataclass(frozen=True)
class GroundTruth:
    """source1_entity_id -> frozenset of matched S2/S3 entity_ids (empty means singleton)."""

    matches: dict[str, frozenset[str]]

    def get(self, s1_id: str) -> frozenset[str]:
        return self.matches.get(s1_id, frozenset())

    def __len__(self) -> int:
        return len(self.matches)

    def is_singleton(self, s1_id: str) -> bool:
        return len(self.get(s1_id)) == 0


def parse_ground_truth(
    data_dir: str, split: str = "train", s1_ids: set[str] | None = None,
    s2_ids: set[str] | None = None, s3_ids: set[str] | None = None,
) -> GroundTruth:
    """Parse train_ground_truth.tsv into a `GroundTruth` lookup.

    If the S1/S2/S3 id sets are provided, validates that every referenced id actually exists
    in those sources (catches a corrupted ground-truth file or a stale/mismatched download
    before it silently poisons training) and that every S1 entity has exactly one row.
    """
    path = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
    df = read_tsv(path)
    _validate_columns(df, EXPECTED_GT_COLUMNS, path)

    n_dupes = df.source1_entity_id.duplicated().sum()
    if n_dupes:
        raise ValueError(f"{path}: {n_dupes} duplicate source1_entity_id rows.")

    matches: dict[str, frozenset[str]] = {}
    for s1, raw in zip(df.source1_entity_id, df.matched_entity_ids):
        ids = frozenset(m for m in raw.split(",") if m)
        matches[s1] = ids

    if s1_ids is not None:
        missing = s1_ids - set(matches)
        if missing:
            raise ValueError(f"{path}: {len(missing)} S1 entities have no ground-truth row, e.g. {list(missing)[:5]}")
        extra = set(matches) - s1_ids
        if extra:
            raise ValueError(f"{path}: {len(extra)} ground-truth rows reference an S1 id not in {split}_source1.tsv.")

    if s2_ids is not None or s3_ids is not None:
        bad = set()
        for s1, ids in matches.items():
            for m in ids:
                if m.startswith("S1-"):
                    bad.add(m)
                elif m.startswith("S2-") and s2_ids is not None and m not in s2_ids:
                    bad.add(m)
                elif m.startswith("S3-") and s3_ids is not None and m not in s3_ids:
                    bad.add(m)
        if bad:
            raise ValueError(
                f"{path}: {len(bad)} matched id(s) are self-matches to S1 or don't exist in "
                f"source2/3, e.g. {list(bad)[:5]}"
            )

    log.info("parsed ground truth: %s S1 entities, %s singletons (%.1f%%)",
             f"{len(matches):,}", f"{sum(1 for v in matches.values() if not v):,}",
             100 * sum(1 for v in matches.values() if not v) / max(len(matches), 1))
    return GroundTruth(matches)
