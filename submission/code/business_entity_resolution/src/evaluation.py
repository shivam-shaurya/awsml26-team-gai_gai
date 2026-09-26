"""Scoring (macro F0.5) and the decision layer that turns pair probabilities into entity-level
match sets.

The decision layer matters as much as the classifier, because of how the metric's arithmetic
behaves:

- Singletons are all-or-nothing: an entity with no true match scores 1.0 for an empty
  prediction and 0.0 for any prediction at all.
- The break-even probability is NOT the same for an entity's first match as for later ones. A
  lone uncertain candidate is worth predicting once its probability exceeds roughly 0.5 (not
  the 0.8-0.9 "precision-heavy" intuition suggests), but once one match is already confirmed,
  adding a second only pays off above roughly 0.73, because a wrong addition drags a perfect
  score down further than a missed one would. A single global probability threshold is
  therefore the wrong tool -- it cannot express this asymmetry.

Two decision rules are implemented and evaluated empirically against each other per run:
threshold-based (with two distinct thresholds, reflecting the asymmetry above) and
expected-F0.5 set selection (threshold-free, and therefore the safer default for a country
with no training data -- see the leave-one-country-out check in ``pipeline.py``).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def f05_score(predicted: set, gold: set) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    tp = len(predicted & gold)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(gold) + len(predicted))


def macro_f05(predictions: dict, gold: dict, s1_ids) -> float:
    return float(np.mean([f05_score(predictions.get(s, set()), gold.get(s, set())) for s in s1_ids]))


def score_report(name: str, predictions: dict, gold: dict, s1_ids, country_of: dict) -> str:
    msg = f"{name}: macro F0.5 = {macro_f05(predictions, gold, s1_ids):.4f}"
    for c in sorted({country_of[s] for s in s1_ids}):
        ids = [s for s in s1_ids if country_of[s] == c]
        msg += f" | {c}: {macro_f05(predictions, gold, ids):.4f}"
    singletons = [s for s in s1_ids if not gold.get(s)]
    if singletons:
        msg += f" | singletons: {macro_f05(predictions, gold, singletons):.4f}"
    return msg


def apply_exclusivity(D: pd.DataFrame) -> pd.DataFrame:
    """Each S2/S3 record should belong to at most one S1 entity (Source 1 is deduplicated --
    verified against training ground truth in the blocking audit). Keep each candidate only
    under its single best-scoring S1 and zero it out elsewhere; removes a specific, common
    failure mode: two different businesses sharing an address or a similar name both drawing
    the same candidate."""
    D = D.copy()
    best = D.groupby("cand")["p"].transform("max")
    is_first = ~D.sort_values("p", ascending=False).duplicated("cand")
    D.loc[~((D.p == best) & is_first.reindex(D.index)), "p"] = 0.0
    return D


def decide_threshold(D: pd.DataFrame, t1: float, t2: float) -> dict:
    """t1: bar for an entity's best candidate to be accepted at all. t2 (usually > t1): bar
    for every additional candidate beyond the first."""
    out = {}
    for s1, g in D[D.p >= min(t1, t2)].groupby("s1"):
        g = g.sort_values("p", ascending=False)
        if g.p.iloc[0] < t1:
            continue
        out[s1] = {g.cand.iloc[0]} | set(g.cand.iloc[1:][g.p.iloc[1:] >= t2])
    return out


def decide_expected_f(
    D: pd.DataFrame, n_mc: int = 1000, p_floor: float = 0.02, max_k: int = 15,
    p_unseen: float = 0.0, seed: int = 42,
) -> dict:
    """For each entity, treat candidates' probabilities as independent match probabilities and
    choose the top-k (including k=0, i.e. singleton) maximising expected per-entity F0.5 under
    that model, via Monte Carlo simulation. Threshold-free given calibrated p.

    p_unseen: assumed chance the entity has a true match that blocking missed entirely --
    raise it if blocking recall is imperfect, since it makes "predict a small set" and "predict
    empty" both slightly riskier in a way that mirrors reality.
    """
    rng = np.random.default_rng(seed)
    E = D[D.p >= p_floor].sort_values(["s1", "p"], ascending=[True, False])
    E = E[E.groupby("s1").cumcount() < max_k]
    s1v, cand_v, p_v = E.s1.values, E.cand.values, E.p.values
    bounds = np.flatnonzero(np.r_[True, s1v[1:] != s1v[:-1], True])

    out = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = p_v[a:b]
        draws = rng.random((n_mc, len(p))) < p
        total_true = draws.sum(1) + (rng.random(n_mc) < p_unseen)
        tp = np.cumsum(draws, axis=1)
        k = np.arange(1, len(p) + 1)
        denom = 1.25 * tp + 0.25 * (total_true[:, None] - tp) + (k - tp)
        expected_f = np.where(tp > 0, 1.25 * tp / np.maximum(denom, 1e-9), 0.0).mean(0)
        empty_baseline = float(np.mean(total_true == 0))
        if expected_f.max() > empty_baseline:
            out[s1v[a]] = set(cand_v[a:a + int(expected_f.argmax()) + 1])
    return out


def _fast_threshold_scorer(D: pd.DataFrame, gold: dict, s1_ids):
    """Precompute once so each (t1, t2) grid point costs a few vectorised numpy ops instead of
    a full groupby pass -- makes the threshold grid search tractable."""
    D = D.sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
    pos = {s: k for k, s in enumerate(s1_ids)}
    code = D.s1.map(pos).values
    is_first = (D.groupby("s1").cumcount() == 0).values
    top_p = D.groupby("s1")["p"].transform("first").values
    y = np.array([c in gold.get(s, ()) for s, c in zip(D.s1, D.cand)], dtype=float)
    p = D.p.values
    n_gold = np.array([len(gold.get(s, ())) for s in s1_ids], dtype=float)
    n = len(s1_ids)

    def score(t1: float, t2: float) -> float:
        selected = (top_p >= t1) & (is_first | (p >= t2))
        n_pred = np.bincount(code[selected], minlength=n).astype(float)
        tp = np.bincount(code[selected], weights=y[selected], minlength=n)
        f = np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gold + n_pred, 1e-9), 0.0)
        f = np.where((n_gold == 0) & (n_pred == 0), 1.0, f)
        return float(f.mean())

    return score


def search_decisions(D: pd.DataFrame, gold: dict, s1_ids, t1_grid, t2_grid):
    """Evaluate both decision-rule families (with/without exclusivity) on out-of-fold
    predictions; returns results sorted best-first plus the spec needed to reproduce each one
    via `run_decision`."""
    results, variants = [], {}
    for exclusive in (False, True):
        Dx = apply_exclusivity(D) if exclusive else D
        tag = "excl" if exclusive else "raw"
        scorer = _fast_threshold_scorer(Dx, gold, s1_ids)
        grid = [(scorer(t1, t2), (round(float(t1), 3), round(float(t2), 3)))
                for t1 in t1_grid for t2 in t2_grid]
        best_score, (t1, t2) = max(grid)
        name = f"threshold[{tag}] t1={t1} t2={t2}"
        results.append((name, best_score))
        variants[name] = (exclusive, "threshold", (t1, t2))

        name = f"expected_f[{tag}]"
        results.append((name, macro_f05(decide_expected_f(Dx), gold, s1_ids)))
        variants[name] = (exclusive, "expected_f", None)

    results.sort(key=lambda r: -r[1])
    return results, variants


def run_decision(D: pd.DataFrame, spec) -> dict:
    exclusive, kind, params = spec
    Dx = apply_exclusivity(D) if exclusive else D
    return decide_threshold(Dx, *params) if kind == "threshold" else decide_expected_f(Dx)
