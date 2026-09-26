"""Write and self-check the two required output files.

Both matching_results.tsv and candidate_pairs.tsv share the same format: one row per Source-1
entity, a comma-joined, de-duplicated, order-preserving list of S2/S3 ids (empty for no
matches/candidates). This module also runs the same structural checks the official
validator (utils/validate_submission.py) enforces, so a format problem is caught here rather
than discovered after spending a submission slot.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def write_id_list_file(path: str, s1_ids: list[str], mapping: dict, column_name: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{column_name}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s1, [])))  # de-dup, preserve order
            f.write(f"{s1}\t{','.join(ids)}\n")


def self_check_predictions(
    predictions: dict, candidates: dict, test_s1_ids: list[str], valid_target_ids: set[str],
) -> None:
    """Fail loudly, before writing anything, if the predictions violate a hard submission rule:
    every matched id must exist in test Source-2/3, and every prediction must be a subset of
    that entity's own candidate set (matches must come from blocking, never invented)."""
    missing_from_valid = {
        c for matches in predictions.values() for c in matches if c not in valid_target_ids
    }
    if missing_from_valid:
        raise ValueError(
            f"{len(missing_from_valid)} predicted id(s) do not exist in test source2/3, "
            f"e.g. {list(missing_from_valid)[:5]}"
        )
    not_subset = [
        s1 for s1 in test_s1_ids
        if not set(predictions.get(s1, [])) <= set(candidates.get(s1, []))
    ]
    if not_subset:
        raise ValueError(
            f"{len(not_subset)} S1 entities have a predicted match that was never a "
            f"candidate, e.g. {not_subset[:5]} -- this indicates a pipeline bug."
        )
    missing_rows = set(test_s1_ids) - set(predictions) - {s for s in test_s1_ids if not predictions.get(s)}
    # (predictions dict only contains entities with >=1 match; singletons are implicitly the rest)
    log.info(
        "self-check passed: %s/%s S1 entities have >=1 predicted match",
        f"{sum(1 for s in test_s1_ids if predictions.get(s)):,}", f"{len(test_s1_ids):,}",
    )
