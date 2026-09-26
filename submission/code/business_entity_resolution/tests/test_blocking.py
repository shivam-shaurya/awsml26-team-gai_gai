import pandas as pd

from blocking import blocking_audit


def _row(entity_id, src, country="US"):
    return dict(entity_id=entity_id, src=src, country=country)


def test_blocking_audit_zero_candidate_count_ignores_non_s1_rows():
    # Regression test: an earlier version reindexed the zero-candidate count over every row in
    # R (S1 + S2 + S3), which counts every S2/S3 row as if it were an "S1 with zero
    # candidates" and wildly overstates the number. It must only ever count real S1 rows.
    R = pd.DataFrame([
        _row("S1-1", "S1"), _row("S1-2", "S1"), _row("S1-3", "S1"),
        _row("S2-1", "S2"), _row("S2-2", "S2"), _row("S3-1", "S3"),
    ])
    # S1-1 (row 0) has one candidate; S1-2 and S1-3 have none.
    C = pd.DataFrame({"i": [0], "j": [3], "f_name": [1], "f_full": [0], "f_zip": [0],
                      "f_rev": [0], "f_emb": [0], "cos_name": [0.9], "cos_full": [0.9], "cos_emb": [0.0]})
    audit = blocking_audit(R, C, gold_pairs=set())
    assert audit["s1_with_zero_candidates"] == 2  # S1-2 and S1-3, NOT the S2/S3 rows too
