"""Apply a trained decision spec to test-set candidate probabilities and produce the final
per-entity prediction mapping (ordered by descending probability, matches only)."""
from __future__ import annotations

import pandas as pd

from evaluation import run_decision


def predict_test_matches(Ft: pd.DataFrame, test_pred_proba, decision_spec) -> tuple[dict, dict]:
    """Returns (predictions, candidates_by_s1_probability_order).

    `candidates_by_s1_probability_order` is used purely to order each entity's final match list
    by descending model confidence -- it has no effect on which candidates are selected, only
    on the order they're written in matching_results.tsv.
    """
    D = pd.DataFrame({"s1": Ft.s1, "cand": Ft.cand, "p": test_pred_proba})
    predictions = run_decision(D, decision_spec)
    ordered_candidates = (
        D.sort_values("p", ascending=False).groupby("s1")["cand"].apply(list).to_dict()
    )
    ordered_predictions = {
        s1: [c for c in ordered_candidates.get(s1, []) if c in matches]
        for s1, matches in predictions.items()
    }
    return ordered_predictions, ordered_candidates
