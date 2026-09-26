"""Pair-level feature engineering: ~30 similarity/overlap/rank features per (S1, candidate)
pair, grouped as name similarity, rarity-weighted overlap, structural, address similarity, and
rank/context features (this last group is empirically the most informative -- a pair's
absolute similarity matters less than whether it's clearly the best option relative to the
competition on both sides of the match).
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

log = logging.getLogger(__name__)

_WORKER_STATE: dict = {}  # populated in each forked worker; module-level so _pair_feats can see it


def compute_idf(R: pd.DataFrame) -> dict:
    """Per-country, per-field (name/address) token IDF, fit on all sources of the split.
    A rare shared token is much stronger evidence of a true match than a common one, and a rare
    token present on only one side is much stronger evidence AGAINST a match."""
    idf = {}
    for country, g in R.groupby("country"):
        n = len(g)
        for kind, col in (("name", "core_set"), ("addr", "addr_set")):
            doc_freq = Counter(t for s in g[col] for t in s)
            idf[(country, kind)] = (
                {t: math.log((n + 1) / (c + 1)) + 1 for t, c in doc_freq.items()},
                math.log(n + 1) + 1,  # default weight for an unseen token
            )
    return idf


def _idf_weighted_overlap(a: frozenset, b: frozenset, idf: dict, default: float):
    if not a and not b:
        return np.nan, np.nan
    w = lambda t: idf.get(t, default)
    inter = sum(w(t) for t in a & b)
    union = sum(w(t) for t in a | b)
    unmatched = max((w(t) for t in a ^ b), default=0.0)
    return (inter / union if union else np.nan), unmatched


def _pair_features(ij: tuple[int, int]) -> tuple:
    R, idf = _WORKER_STATE["R"], _WORKER_STATE["idf"]
    i, j = ij
    a, b = R[i], R[j]
    n1, n2 = a["name_core"], b["name_core"]
    ad1, ad2 = a["addr_norm"], b["addr_norm"]

    name_idf, name_default = idf[(a["country"], "name")]
    addr_idf, addr_default = idf[(a["country"], "addr")]
    n_idf, n_unmatched = _idf_weighted_overlap(a["core_set"], b["core_set"], name_idf, name_default)
    a_idf, a_unmatched = _idf_weighted_overlap(a["addr_set"], b["addr_set"], addr_idf, addr_default)

    cs1, cs2 = a["core_set"], b["core_set"]
    as1, as2 = a["addr_set"], b["addr_set"]
    acronym_hit = int(bool(a["acronym"]) and a["acronym"] in cs2) or int(bool(b["acronym"]) and b["acronym"] in cs1)

    if a["postal"] and b["postal"]:
        zip_eq = float(a["postal"] == b["postal"])
        zip_prefix_eq = float(a["postal"][:3] == b["postal"][:3])
    else:
        zip_eq = zip_prefix_eq = np.nan
    if a["nums"] and b["nums"]:
        num_jaccard = len(a["nums"] & b["nums"]) / len(a["nums"] | b["nums"])
    else:
        num_jaccard = np.nan

    return (
        fuzz.ratio(n1, n2), fuzz.partial_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
        fuzz.token_sort_ratio(n1, n2), JaroWinkler.normalized_similarity(n1, n2),
        Levenshtein.normalized_similarity(n1, n2),
        fuzz.token_set_ratio(a["name_norm"], b["name_norm"]),
        len(cs1 & cs2) / max(len(cs1 | cs2), 1), n_idf, n_unmatched,
        int(a["core_first"] == b["core_first"]), acronym_hit, len(cs1), len(cs2),
        int(bool(a["legal"]) and bool(b["legal"]) and not (a["legal"] & b["legal"])),
        int(bool(a["name_digits"] ^ b["name_digits"])),
        fuzz.ratio(ad1, ad2), fuzz.partial_ratio(ad1, ad2), fuzz.token_set_ratio(ad1, ad2),
        fuzz.token_sort_ratio(ad1, ad2), len(as1 & as2) / max(len(as1 | as2), 1), a_idf, a_unmatched,
        zip_eq, zip_prefix_eq, num_jaccard, len(as1), len(as2), a["landmark"], b["landmark"],
        int(b["src"] == "S3"),
        fuzz.token_set_ratio(n2, ad1),  # candidate's name appearing inside S1's address (rare, cheap)
    )


PAIR_FEATURE_COLUMNS = [
    "n_ratio", "n_pratio", "n_tset", "n_tsort", "n_jw", "n_lev", "n_full_tset", "n_jacc",
    "n_idf", "n_idf_unmatched_max", "n_first_eq", "n_acronym", "n_len1", "n_len2",
    "legal_conflict", "name_digit_conflict", "a_ratio", "a_pratio", "a_tset", "a_tsort",
    "a_jacc", "a_idf", "a_idf_unmatched_max", "zip_eq", "zip_prefix_eq", "num_jacc",
    "a_len1", "a_len2", "landmark1", "landmark2", "is_s3", "name_in_addr",
]


def featurize(C: pd.DataFrame, R: pd.DataFrame, idf: dict, n_jobs: int) -> pd.DataFrame:
    """Compute PAIR_FEATURE_COLUMNS for every candidate pair, then add rank/context features
    that compare each pair against the competition for the same S1 entity and the same
    candidate (empirically the strongest feature group -- see module docstring)."""
    _WORKER_STATE["R"] = R.to_dict("records")
    _WORKER_STATE["idf"] = idf
    pairs = list(zip(C.i.values, C.j.values))
    if n_jobs > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(n_jobs) as pool:
            rows = pool.map(_pair_features, pairs, chunksize=2000)
    else:
        rows = [_pair_features(p) for p in pairs]

    F = pd.DataFrame(rows, columns=PAIR_FEATURE_COLUMNS, index=C.index).astype(np.float32)
    F = pd.concat([C, F], axis=1)

    F["combo"] = 0.5 * F.cos_full + 0.25 * F.n_tset / 100 + 0.25 * F.a_tset / 100
    for col in ("cos_name", "cos_full", "combo"):
        by_s1 = F.groupby("i")[col]
        F[f"{col}_rank_s1"] = by_s1.rank(ascending=False, method="min")
        F[f"{col}_gap_s1"] = by_s1.transform("max") - F[col]
        by_cand = F.groupby("j")[col]
        F[f"{col}_rank_cand"] = by_cand.rank(ascending=False, method="min")
        F[f"{col}_gap_cand"] = by_cand.transform("max") - F[col]

    top2 = F.sort_values(["i", "combo"], ascending=[True, False]).groupby("i").head(2).groupby("i")["combo"]
    gap2 = (top2.max() - top2.min()).where(top2.size() > 1, 1.0)
    F["combo_second_gap_s1"] = F.i.map(gap2).values
    F["n_cands_s1"] = F.groupby("i")["j"].transform("size")
    F["n_s1_per_cand"] = F.groupby("j")["i"].transform("size")
    F["mutual_best"] = ((F.combo_rank_s1 == 1) & (F.combo_rank_cand == 1)).astype(np.int8)

    # how common is this S1's core name within its country? A generic name needs address
    # corroboration; a rare one is more trustworthy on name alone.
    key = R.country + "\x1f" + R.name_core
    freq = key[R.src == "S1"].value_counts()
    F["s1_name_freq"] = key.loc[F.i].map(freq).fillna(1).values
    return F
