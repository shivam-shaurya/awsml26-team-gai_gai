"""Candidate generation (blocking): union of independent, complementary blockers.

Comparing every Source-1 record against every Source-2/3 record is infeasible at this
dataset's real scale (millions of rows per country) -- blocking narrows that down to a
recall-oriented candidate set cheaply, before the (expensive, precise) pair classifier ever
runs. Blocking sets the ceiling on achievable recall: a true match never proposed as a
candidate cannot be found downstream, so this module's own recall is audited directly against
ground truth (see ``blocking_audit``) before any model training happens.

Scalability note (this is the one part of the pipeline that is NOT safe to implement the
"obvious" way at this dataset's size): computing an exact top-k cosine similarity between two
TF-IDF matrices via ``(Q @ T.T).toarray()`` in memory-bounded chunks looks reasonable, but the
chunk size that keeps that dense array within a fixed byte budget collapses to single digits
once a target set reaches millions of rows -- turning a couple of sparse matmuls into hundreds
of thousands of Python-loop iterations. Measured directly against this challenge's real
per-country data: that approach needs ~88 CPU-hours for a single blocker view/direction
against the ~3M-row US Source-2 file. A naive "just multiply the sparse matrices instead"
fix is not enough either -- it can OOM outright (confirmed: a 20k-row chunk against 3M targets
tried to allocate a 9-billion-nonzero array), because short business-name strings share enough
common character-trigrams that the "sparse" product isn't sparse at all at this scale. The fix
that actually works is `prune_common_columns`: an ABSOLUTE (not relative) document-frequency
cap on which trigrams are even allowed to participate, applied before the matmul.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import BlockingConfig

log = logging.getLogger(__name__)


def prune_common_columns(A: sparse.csr_matrix, B: sparse.csr_matrix, max_doc_freq: int):
    """Zero out TF-IDF columns (trigrams) whose combined document frequency across A and B
    exceeds an ABSOLUTE cap. See module docstring for why this must be absolute, not relative."""
    doc_freq = np.asarray((A > 0).sum(0)).ravel() + np.asarray((B > 0).sum(0)).ravel()
    keep = doc_freq <= max_doc_freq
    if keep.all():
        return A, B
    mask = sparse.diags(keep.astype(np.float32))
    return (A @ mask).tocsr(), (B @ mask).tocsr()


def topk_cosine(Q: sparse.csr_matrix, T: sparse.csr_matrix, k: int):
    """Row-wise top-k of Q @ T.T for L2-normalised sparse rows.

    True sparse-times-sparse product, chunked over Q only to bound peak memory -- it never
    densifies the T axis (see module docstring for why that matters at this scale). Rows with
    fewer than k matching (nonzero-similarity) targets are padded with index -1 / similarity
    0.0; callers must skip idx == -1.
    """
    k = min(k, T.shape[0])
    nq = Q.shape[0]
    idx = np.full((nq, k), -1, dtype=np.int64)
    sim = np.zeros((nq, k), dtype=np.float32)
    if k == 0 or nq == 0:
        return idx, sim
    Tt = T.T.tocsr()
    chunk = 20_000
    for a in range(0, nq, chunk):
        S = (Q[a:a + chunk] @ Tt).tocsr()
        indptr, indices, data = S.indptr, S.indices, S.data
        for r in range(S.shape[0]):
            start, end = indptr[r], indptr[r + 1]
            if end == start:
                continue
            row_idx, row_data = indices[start:end], data[start:end]
            part = np.argpartition(-row_data, k - 1)[:k] if end - start > k else np.arange(end - start)
            order = part[np.argsort(-row_data[part])]
            n = len(order)
            idx[a + r, :n] = row_idx[order]
            sim[a + r, :n] = row_data[order]
    return idx, sim


def _load_external_candidates(path: str, eid2idx: dict) -> dict:
    """Load an optional GPU-blocker candidate file (source1_entity_id, candidate_entity_id,
    cos_emb) and map entity_ids onto row positions in this split's R DataFrame."""
    scores = {}
    n_loaded = n_mapped = 0
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            s1_id, cand_id, score = line.rstrip("\n").split("\t")
            n_loaded += 1
            i, j = eid2idx.get(s1_id), eid2idx.get(cand_id)
            if i is None or j is None:
                continue
            scores[(i, j)] = max(scores.get((i, j), -1.0), float(score))
            n_mapped += 1
    log.info("embedding blocker: loaded %s pairs from %s, %s mapped onto this split",
             f"{n_loaded:,}", path, f"{n_mapped:,}")
    return scores


def generate_candidates(R: pd.DataFrame, cfg: BlockingConfig, embed_candidates_path: str | None = None) -> pd.DataFrame:
    """Union of blockers, run separately per (country, target source):

      (a) char-3gram TF-IDF on core name, S1->target top-k and target->S1 top-k (reverse)
      (b) char-3gram TF-IDF on name+address, both directions
      (c) exact key: (postal code, first core-name token)
      (d) optional: pre-computed dense-embedding candidates (GPU stage, see gpu_embed_blocker.py)

    Returns DataFrame [i, j, f_name, f_full, f_zip, f_rev, f_emb, cos_name, cos_full, cos_emb]
    with i = row position of an S1 record, j = row position of an S2/S3 record, both into ``R``.
    """
    cand: dict[tuple, set] = {}

    def add(i, j, flag):
        cand.setdefault((i, j), set()).add(flag)

    cos_store = {}
    for country, g in R.groupby("country"):
        s1 = g[g.src == "S1"]
        for src in ("S2", "S3"):
            tg = g[g.src == src]
            if len(s1) == 0 or len(tg) == 0:
                continue
            views = {}
            for view, text_fn in (("name", lambda d: d["name_core"]),
                                  ("full", lambda d: d["name_core"] + " | " + d["addr_norm"])):
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                      dtype=np.float32)
                vec.fit(pd.concat([text_fn(s1), text_fn(tg)]))
                A, B = vec.transform(text_fn(s1)), vec.transform(text_fn(tg))
                A, B = prune_common_columns(A, B, cfg.max_trigram_doc_freq)
                views[view] = (A, B)
                kf = cfg.k_name if view == "name" else cfg.k_full
                idx, _ = topk_cosine(A, B, kf)
                for a in range(idx.shape[0]):
                    for b in idx[a]:
                        if b != -1:
                            add(s1.index[a], tg.index[b], "f_" + view)
                idx, sim = topk_cosine(B, A, cfg.k_rev)
                for b in range(idx.shape[0]):
                    for a, sv in zip(idx[b], sim[b]):
                        if a != -1 and sv > 0.2:
                            add(s1.index[a], tg.index[b], "f_rev")
            # exact postal + first-token key: cheap, high-precision anchor that cross-checks
            # the fuzzy blockers above.
            buckets = defaultdict(list)
            for i, p, f in zip(s1.index, s1.postal, s1.core_first):
                if p and f:
                    buckets[(p, f)].append(i)
            for j, p, f in zip(tg.index, tg.postal, tg.core_first):
                b = buckets.get((p, f), [])
                if 0 < len(b) <= 50:
                    for i in b:
                        add(i, j, "f_zip")
            cos_store[(country, src)] = (views, {v: k for k, v in enumerate(s1.index)},
                                         {v: k for k, v in enumerate(tg.index)})

    emb_score = {}
    if embed_candidates_path:
        eid2idx = {v: k for k, v in R["entity_id"].items()}
        emb_score = _load_external_candidates(embed_candidates_path, eid2idx)
        for (i, j) in emb_score:
            add(i, j, "f_emb")

    rows = [(i, j, int("f_name" in f), int("f_full" in f), int("f_zip" in f), int("f_rev" in f),
             int("f_emb" in f)) for (i, j), f in cand.items()]
    C = pd.DataFrame(rows, columns=["i", "j", "f_name", "f_full", "f_zip", "f_rev", "f_emb"])

    C["cos_name"] = np.nan
    C["cos_full"] = np.nan
    key = list(zip(R.loc[C.i, "country"].values, R.loc[C.j, "src"].values))
    C["_g"] = key
    for g_key, sub in C.groupby("_g"):
        views, m1, m2 = cos_store[g_key]
        ia = np.array([m1[i] for i in sub.i])
        jb = np.array([m2[j] for j in sub.j])
        for view in ("name", "full"):
            A, B = views[view]
            C.loc[sub.index, "cos_" + view] = np.asarray(A[ia].multiply(B[jb]).sum(1)).ravel()
    C = C.drop(columns="_g")
    C["cos_emb"] = [emb_score.get((i, j), np.nan) for i, j in zip(C.i, C.j)]

    # Cap candidates per S1 by best evidence across all blockers -- keeps the candidate file
    # and downstream feature cost bounded (recall past ~99.5% mostly adds false-merge risk
    # under a precision-weighted metric, not real matches).
    C["_s"] = C[["cos_name", "cos_full", "cos_emb"]].max(axis=1) + 0.05 * C.f_zip
    C = C.sort_values(["i", "_s"], ascending=[True, False])
    C = C.groupby("i", sort=False).head(cfg.max_cands).drop(columns="_s").reset_index(drop=True)
    C["cos_emb"] = C["cos_emb"].fillna(0.0)
    return C


def blocking_audit(R: pd.DataFrame, C: pd.DataFrame, gold_pairs: set[tuple[str, str]]) -> dict:
    """Measure blocking quality against ground truth: recall, reduction ratio, per-blocker
    contribution. This is what `reports/blocking_baseline.md` is built from, and what must be
    checked BEFORE spending any time on model tuning -- a recall ceiling problem downstream of
    here cannot be fixed by a better classifier."""
    s1_ids = R.loc[R.src == "S1", "entity_id"]
    cand_pairs = set(zip(R.loc[C.i, "entity_id"].values, R.loc[C.j, "entity_id"].values))
    n_s1 = len(s1_ids)
    recall = len(gold_pairs & cand_pairs) / max(len(gold_pairs), 1)

    gold_by_s1: dict[str, set[str]] = defaultdict(set)
    for s, c in gold_pairs:
        gold_by_s1[s].add(c)
    cand_by_s1: dict[str, set[str]] = defaultdict(set)
    for s, c in cand_pairs:
        cand_by_s1[s].add(c)
    entities_fully_recalled = np.mean([
        gold_by_s1[s] <= cand_by_s1.get(s, set()) for s in gold_by_s1
    ]) if gold_by_s1 else float("nan")

    s1_row_positions = R.index[R.src == "S1"]
    n_zero_cand = (C.groupby("i").size().reindex(s1_row_positions).fillna(0) == 0).sum()

    per_blocker = {}
    F = C.copy()
    F["s1"] = R.loc[F.i, "entity_id"].values
    F["cand"] = R.loc[F.j, "entity_id"].values
    for flag in ("f_name", "f_full", "f_zip", "f_rev", "f_emb"):
        sub = set(zip(F.s1[F[flag] == 1], F.cand[F[flag] == 1]))
        per_blocker[flag] = {
            "pairs": len(sub),
            "recall_alone": len(gold_pairs & sub) / max(len(gold_pairs), 1),
        }

    return dict(
        n_s1=n_s1,
        n_candidate_pairs=len(C),
        candidates_per_s1=len(C) / max(n_s1, 1),
        pair_recall=recall,
        entities_fully_recalled=float(entities_fully_recalled),
        s1_with_zero_candidates=int(n_zero_cand),
        per_blocker=per_blocker,
    )
