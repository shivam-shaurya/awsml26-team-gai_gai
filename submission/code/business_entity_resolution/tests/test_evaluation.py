import pandas as pd

from evaluation import apply_exclusivity, f05_score, macro_f05


def test_f05_matches_hand_worked_example_from_problem_statement():
    # Problem statement's own worked example: predicted {S2-47,S2-193,S3-812},
    # gold {S2-47,S3-812} -> precision=2/3, recall=1.0, F0.5 = 0.714.
    predicted = {"S2-47", "S2-193", "S3-812"}
    gold = {"S2-47", "S3-812"}
    assert abs(f05_score(predicted, gold) - 0.714) < 1e-3


def test_f05_singleton_all_or_nothing():
    assert f05_score(set(), set()) == 1.0        # correctly predicted empty -> full credit
    assert f05_score({"S2-1"}, set()) == 0.0      # any false merge on a true singleton -> zero
    assert f05_score(set(), {"S2-1"}) == 0.0      # missed the only true match entirely -> zero


def test_macro_f05_averages_per_entity_not_pooled():
    gold = {"S1-1": {"S2-1"}, "S1-2": set()}
    predictions = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert macro_f05(predictions, gold, ["S1-1", "S1-2"]) == 1.0


def test_apply_exclusivity_keeps_only_best_scoring_s1_per_candidate():
    D = pd.DataFrame({
        "s1": ["S1-1", "S1-2"],
        "cand": ["S2-1", "S2-1"],  # same candidate claimed by two different S1 entities
        "p": [0.9, 0.6],
    })
    out = apply_exclusivity(D)
    kept = out[out.p > 0]
    assert list(kept.s1) == ["S1-1"]
    assert (out.loc[out.s1 == "S1-2", "p"] == 0.0).all()
