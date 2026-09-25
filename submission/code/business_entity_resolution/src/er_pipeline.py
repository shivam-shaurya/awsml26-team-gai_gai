#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution: end-to-end baseline.

  normalize -> block (multi-view, both directions) -> pair features -> LightGBM (GroupKFold OOF)
  -> exclusivity (each S2/S3 record belongs to <= 1 S1) -> per-entity expected-F0.5 set selection
  -> matching_results.tsv + candidate_pairs.tsv

Usage (run from student_resource/):
  python er_pipeline.py --data dataset --out output --mode cv           # local validation only
  python er_pipeline.py --data dataset --out output --mode cv --loco    # + leave-one-country-out (France proxy)
  python er_pipeline.py --data dataset --out output --mode full         # CV + predict test + write outputs

Deps (tested): pandas 3.0 numpy scipy 1.17 scikit-learn 1.8 rapidfuzz 3.14 lightgbm 4.7

Diagnostic files (OOF / test scores) go to --work, so --out holds only the two submission files.

GPU stage (Ayush's L40, 96GB): two additive scripts in this same src/ folder produce optional
inputs this pipeline consumes if present, and are silently skipped otherwise so every teammate can
run the CPU path unmodified:
  * gpu_embed_blocker.py  -> --embed-candidates-train/-test  (dense multilingual-e5 kNN blocker;
    catches cross-script / transliteration matches char-3gram TF-IDF cannot see, e.g. a Devanagari
    S2 name vs a Latin S1 name)
  * cross_encoder.py      -> --xenc-train-probs/-test-probs  (fine-tuned mDeBERTa-v3 pair
    probability, stacked in as one extra LightGBM feature: p_xenc, plus p_xenc_present)
Still open for the team: rare-token inverted-index blocker, isotonic calibration of p before
expected-F selection, tuning p_unseen, stage-2 model on OOF-derived competitor/cluster features.
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

SEED = 42
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------------------------
def read_tsv(path):
    """dtype=str + keep_default_na=False keeps names like 'NA'/'NULL' and empty ID lists as strings.
    QUOTE_NONE stops a stray quote character from swallowing following lines."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    with open(path, encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip()) - 1
    if n_lines != len(df):
        log(f"WARNING {path}: {n_lines} lines but {len(df)} rows parsed - inspect quoting/tabs")
    return df


def write_id_lists(path, s1_ids, mapping, col):
    """One row per S1 id, comma-joined de-duplicated S2/S3 ids, no quoting."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s, [])))
            f.write(f"{s}\t{','.join(ids)}\n")


# ----------------------------------------------------------------------------------------------
# Normalization (language-agnostic core + small open-ended abbreviation table; no country hard-coding)
# ----------------------------------------------------------------------------------------------
# Long form -> short canonical token. Canonicalising to the SHORT form avoids ambiguity problems
# (e.g. 'st' = street in the US and saint in France: both sides collapse to the same token anyway).
TABLE = {
    # legal forms / business words
    "incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt",
    "compagnie": "cie", "societe": "ste", "etablissements": "ets", "etablissement": "ets",
    "enterprises": "ent", "enterprise": "ent", "industries": "ind", "industry": "ind",
    "international": "intl", "technologies": "tech", "technology": "tech", "services": "svc",
    "service": "svc", "brothers": "bros", "manufacturing": "mfg", "associates": "assoc",
    "solutions": "sol", "solution": "sol", "shree": "sri", "shri": "sri", "sree": "sri",
    "et": "and", "und": "and", "y": "and",
    # address words
    "street": "st", "saint": "st", "sainte": "ste", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln", "place": "pl", "suite": "ste",
    "floor": "fl", "flr": "fl", "building": "bldg", "apartment": "apt", "appartement": "apt",
    "highway": "hwy", "parkway": "pkwy", "court": "ct", "square": "sq", "sector": "sec",
    "near": "nr", "opposite": "opp", "north": "n", "south": "s", "east": "e", "west": "w",
    "number": "no", "num": "no", "chemin": "ch", "route": "rte", "impasse": "imp", "allee": "all",
    "faubourg": "fbg", "centre": "ctr", "center": "ctr", "mount": "mt", "district": "dist",
}
LEGAL = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "pte", "opc", "sarl",
         "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gmbh", "ag", "bv", "srl", "spa",
         "cie", "ste", "ets"}
NAME_STOP = {"and", "the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "s"}
LANDMARK = {"nr", "opp", "behind", "beside", "next", "adjacent", "facing", "pres", "cote", "vis"}

_re_ms = re.compile(r"\bm\s*/\s*s\b")          # Indian "M/s" prefix
_re_pin = re.compile(r"\b(\d{3})[\s-](\d{3})\b")  # "560 001" -> "560001"
_re_nonalnum = re.compile(r"[^0-9a-z]+")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokens(s):
    s = strip_accents(str(s)).lower()
    s = _re_ms.sub(" ", s).replace("&", " and ").replace("'", " ").replace("\u2019", " ")
    s = _re_pin.sub(r"\1\2", s)
    return [TABLE.get(t, t) for t in _re_nonalnum.sub(" ", s).split()]


def merge_initials(toks):
    """'j k s traders' -> 'jks traders' (Indian initials); keeps a lone letter as-is."""
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


def process_records(df, src):
    recs = []
    for eid, name, addr, country in zip(df["entity_id"], df["business_name"],
                                        df["business_address"], df["country"]):
        nt = merge_initials(tokens(name))
        legal = frozenset(t for t in nt if t in LEGAL)
        core = [t for t in nt if t not in LEGAL and t not in NAME_STOP] or nt
        at = tokens(addr)
        postal = ""
        for t in reversed(at[-4:]):  # postal code: 5-6 digit token near the end of the address
            if t.isdigit() and len(t) in (5, 6):
                postal = t
                break
        nums = frozenset(t for t in at if any(ch.isdigit() for ch in t) and t != postal)
        recs.append(dict(
            entity_id=eid, src=src, country=country.strip() or "UNK",
            name_norm=" ".join(nt), name_core=" ".join(core), core_set=frozenset(core),
            core_first=core[0] if core else "", legal=legal,
            acronym="".join(t[0] for t in core) if len(core) >= 2 else "",
            name_digits=frozenset(t for t in core if t.isdigit()),
            addr_norm=" ".join(at), addr_set=frozenset(at), postal=postal, nums=nums,
            landmark=int(any(t in LANDMARK for t in at)),
        ))
    return recs


def load_split(data_dir, split):
    recs = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f"{split}_source{k}.tsv"))
        recs += process_records(df, f"S{k}")
    return pd.DataFrame(recs)


def compute_idf(R):
    """Per-country token IDF for name-core and address tokens (fit on all sources of the split)."""
    idf = {}
    for country, g in R.groupby("country"):
        n = len(g)
        for kind, col in (("name", "core_set"), ("addr", "addr_set")):
            df_ = Counter(t for s in g[col] for t in s)
            idf[(country, kind)] = ({t: math.log((n + 1) / (c + 1)) + 1 for t, c in df_.items()},
                                    math.log(n + 1) + 1)
    return idf


# ----------------------------------------------------------------------------------------------
# Blocking
# ----------------------------------------------------------------------------------------------
def topk_cosine(Q, T, k):
    """Row-wise top-k of Q @ T.T for L2-normalised sparse rows. True sparse-times-sparse product,
    chunked over Q only to bound peak memory -- it never densifies the T axis. That densify-then-
    argpartition approach is the single biggest scalability trap here: with a `chunk` sized to keep
    a dense (chunk x T) array under a fixed byte budget, T=3e6 forces chunk down to single digits,
    which turns a couple of sparse matmuls into hundreds of thousands of Python-loop iterations --
    measured at ~90 minutes per 1,000 queries, i.e. multiple DAYS at this challenge's real
    per-country target sizes (e.g. US Source 2 alone is ~3M rows). Staying sparse end-to-end
    instead costs a per-row Python loop over the (typically tiny) nonzero run for each query, which
    is orders of magnitude cheaper. Rows with fewer than k matching (nonzero-similarity) targets
    are padded with index -1 / similarity 0.0; callers must skip idx == -1."""
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


def prune_common_columns(A, B, max_doc_freq):
    """Zero out TF-IDF columns (trigrams) whose combined document frequency across A and B
    exceeds an ABSOLUTE cap. This -- not a relative max_df -- is what keeps A @ B.T sparse at
    million-row scale: short business-name strings share enough common trigrams that even the 6%
    most-frequent ones turn the "sparse" product matrix effectively dense (measured: without this,
    a 20k x 3M chunk alone tries to allocate a 9-billion-nonzero array and OOMs). A relative
    max_df=0.05 is not aggressive enough once the corpus reaches millions of rows, because absolute
    collision counts -- not fractions -- are what the sparse matmul cost actually depends on."""
    doc_freq = np.asarray((A > 0).sum(0)).ravel() + np.asarray((B > 0).sum(0)).ravel()
    keep = doc_freq <= max_doc_freq
    if keep.all():
        return A, B
    mask = sparse.diags(keep.astype(np.float32))
    return (A @ mask).tocsr(), (B @ mask).tocsr()


def generate_candidates(R, args):
    """Union of blockers, all run separately per (country, target source):
       (a) char-3gram TF-IDF on core name, S1->target top-k and target->S1 top-k (reverse)
       (b) char-3gram TF-IDF on name+address, both directions
       (c) exact key: (postal code, first core-name token)
    Returns DataFrame [i, j, f_name, f_full, f_zip, f_rev, cos_name, cos_full] with i=S1 row, j=S2/S3 row."""
    cand = {}  # (i, j) -> flags

    def add(i, j, flag):
        cand.setdefault((i, j), set()).add(flag)

    cos_store = {}  # (country, src) -> (vectors & row maps) to score every candidate later
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
                A, B = prune_common_columns(A, B, args.max_trigram_doc_freq)
                views[view] = (A, B)
                kf = args.k_name if view == "name" else args.k_full
                idx, _ = topk_cosine(A, B, kf)
                for a in range(idx.shape[0]):
                    for b in idx[a]:
                        if b != -1:
                            add(s1.index[a], tg.index[b], "f_" + view)
                idx, sim = topk_cosine(B, A, args.k_rev)
                for b in range(idx.shape[0]):
                    for a, sv in zip(idx[b], sim[b]):
                        if a != -1 and sv > 0.2:
                            add(s1.index[a], tg.index[b], "f_rev")
            # exact postal + first-token key
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

    # GPU dense-embedding blocker (produced offline by gpu_embed_blocker.py on Ayush's L40; optional,
    # gracefully absent on dev machines without a GPU). Catches cross-script / transliteration matches
    # (e.g. Devanagari S2 name vs Latin S1 name) that char-3gram TF-IDF cannot see at all.
    emb_score = {}
    embed_path = getattr(args, "embed_candidates", None)
    if embed_path and os.path.exists(embed_path):
        eid2idx = {v: k for k, v in R["entity_id"].items()}
        n_loaded, n_mapped = 0, 0
        with open(embed_path, encoding="utf-8") as f:
            next(f)  # header: source1_entity_id  candidate_entity_id  cos_emb
            for line in f:
                s1_id, cand_id, score = line.rstrip("\n").split("\t")
                n_loaded += 1
                i, j = eid2idx.get(s1_id), eid2idx.get(cand_id)
                if i is None or j is None:
                    continue
                add(i, j, "f_emb")
                emb_score[(i, j)] = max(emb_score.get((i, j), -1.0), float(score))
                n_mapped += 1
        log(f"  embedding blocker: loaded {n_loaded:,} pairs from {embed_path}, "
            f"{n_mapped:,} mapped onto this split")

    rows = [(i, j, int("f_name" in f), int("f_full" in f), int("f_zip" in f), int("f_rev" in f),
             int("f_emb" in f)) for (i, j), f in cand.items()]
    C = pd.DataFrame(rows, columns=["i", "j", "f_name", "f_full", "f_zip", "f_rev", "f_emb"])
    # exact cosine for every candidate under both views
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
    # cap candidates per S1 by best cosine (keeps the candidate file and feature cost bounded)
    C["_s"] = C[["cos_name", "cos_full", "cos_emb"]].max(axis=1) + 0.05 * C.f_zip
    C = C.sort_values(["i", "_s"], ascending=[True, False])
    C = C.groupby("i", sort=False).head(args.max_cands).drop(columns="_s").reset_index(drop=True)
    C["cos_emb"] = C["cos_emb"].fillna(0.0)
    return C


# ----------------------------------------------------------------------------------------------
# Pair features
# ----------------------------------------------------------------------------------------------
_G = {}  # globals shared with forked workers


def _idf_overlap(a, b, idf, default):
    if not a and not b:
        return np.nan, np.nan
    w = lambda t: idf.get(t, default)
    inter = sum(w(t) for t in a & b)
    union = sum(w(t) for t in a | b)
    unmatched = max((w(t) for t in a ^ b), default=0.0)
    return inter / union if union else np.nan, unmatched


def _pair_feats(ij):
    R, idf = _G["R"], _G["idf"]
    i, j = ij
    a, b = R[i], R[j]
    n1, n2 = a["name_core"], b["name_core"]
    ad1, ad2 = a["addr_norm"], b["addr_norm"]
    ni, nd = idf[(a["country"], "name")]
    ai, adf = idf[(a["country"], "addr")]
    n_idf, n_unm = _idf_overlap(a["core_set"], b["core_set"], ni, nd)
    a_idf, a_unm = _idf_overlap(a["addr_set"], b["addr_set"], ai, adf)
    cs1, cs2 = a["core_set"], b["core_set"]
    as1, as2 = a["addr_set"], b["addr_set"]
    acr = int(bool(a["acronym"]) and a["acronym"] in cs2) or int(bool(b["acronym"]) and b["acronym"] in cs1)
    if a["postal"] and b["postal"]:
        zip_eq = float(a["postal"] == b["postal"])
        zip_pref = float(a["postal"][:3] == b["postal"][:3])
    else:
        zip_eq = zip_pref = np.nan
    if a["nums"] and b["nums"]:
        num_j = len(a["nums"] & b["nums"]) / len(a["nums"] | b["nums"])
    else:
        num_j = np.nan
    return (
        fuzz.ratio(n1, n2), fuzz.partial_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
        fuzz.token_sort_ratio(n1, n2), JaroWinkler.normalized_similarity(n1, n2),
        Levenshtein.normalized_similarity(n1, n2),
        fuzz.token_set_ratio(a["name_norm"], b["name_norm"]),
        len(cs1 & cs2) / max(len(cs1 | cs2), 1), n_idf, n_unm,
        int(a["core_first"] == b["core_first"]), acr, len(cs1), len(cs2),
        int(bool(a["legal"]) and bool(b["legal"]) and not (a["legal"] & b["legal"])),
        int(bool(a["name_digits"] ^ b["name_digits"])),
        fuzz.ratio(ad1, ad2), fuzz.partial_ratio(ad1, ad2), fuzz.token_set_ratio(ad1, ad2),
        fuzz.token_sort_ratio(ad1, ad2), len(as1 & as2) / max(len(as1 | as2), 1), a_idf, a_unm,
        zip_eq, zip_pref, num_j, len(as1), len(as2), a["landmark"], b["landmark"],
        int(b["src"] == "S3"),
        fuzz.token_set_ratio(n2, ad1),  # name of candidate appearing inside S1 address (rare but cheap)
    )


PAIR_COLS = ["n_ratio", "n_pratio", "n_tset", "n_tsort", "n_jw", "n_lev", "n_full_tset", "n_jacc",
             "n_idf", "n_idf_unmatched_max", "n_first_eq", "n_acronym", "n_len1", "n_len2",
             "legal_conflict", "name_digit_conflict", "a_ratio", "a_pratio", "a_tset", "a_tsort",
             "a_jacc", "a_idf", "a_idf_unmatched_max", "zip_eq", "zip_prefix_eq", "num_jacc",
             "a_len1", "a_len2", "landmark1", "landmark2", "is_s3", "name_in_addr"]


def featurize(C, R, idf, n_jobs):
    _G["R"] = R.to_dict("records")
    _G["idf"] = idf
    pairs = list(zip(C.i.values, C.j.values))
    if n_jobs > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(n_jobs) as pool:
            out = pool.map(_pair_feats, pairs, chunksize=2000)
    else:
        out = [_pair_feats(p) for p in pairs]
    F = pd.DataFrame(out, columns=PAIR_COLS, index=C.index).astype(np.float32)
    F = pd.concat([C, F], axis=1)

    # ---- context features: how does this pair compare with the competition? ----
    F["combo"] = 0.5 * F.cos_full + 0.25 * F.n_tset / 100 + 0.25 * F.a_tset / 100
    for col in ("cos_name", "cos_full", "combo"):
        g1 = F.groupby("i")[col]
        F[f"{col}_rank_s1"] = g1.rank(ascending=False, method="min")
        F[f"{col}_gap_s1"] = g1.transform("max") - F[col]
        g2 = F.groupby("j")[col]
        F[f"{col}_rank_cand"] = g2.rank(ascending=False, method="min")
        F[f"{col}_gap_cand"] = g2.transform("max") - F[col]
    top2 = F.sort_values(["i", "combo"], ascending=[True, False]).groupby("i").head(2).groupby("i")["combo"]
    gap2 = (top2.max() - top2.min()).where(top2.size() > 1, 1.0)
    F["combo_second_gap_s1"] = F.i.map(gap2).values
    F["n_cands_s1"] = F.groupby("i")["j"].transform("size")
    F["n_s1_per_cand"] = F.groupby("j")["i"].transform("size")
    F["mutual_best"] = ((F.combo_rank_s1 == 1) & (F.combo_rank_cand == 1)).astype(np.int8)
    # how common is this S1 core name in its country? (common names need address evidence)
    key = R.country + "\x1f" + R.name_core
    freq = key[R.src == "S1"].value_counts()
    F["s1_name_freq"] = key.loc[F.i].map(freq).fillna(1).values
    return F


FEATURES = None  # filled in main (all numeric columns except ids)


def merge_xenc_probs(F, path):
    """Stack in an out-of-fold / test cross-encoder probability (from cross_encoder.py, run on the
    GPU box) as one extra feature. Absent file -> neutral 0.5 prior everywhere, so the pipeline is
    fully runnable without a GPU; present file -> only pairs it scored get a real value, the rest
    (e.g. candidates the cross-encoder step subsampled away) keep the neutral prior."""
    if path and os.path.exists(path):
        X = read_tsv(path).rename(columns={"source1_entity_id": "s1", "candidate_entity_id": "cand"})
        X["p_xenc"] = X["p_xenc"].astype(np.float32)
        F = F.merge(X[["s1", "cand", "p_xenc"]], on=["s1", "cand"], how="left")
    else:
        F = F.copy()
        F["p_xenc"] = np.nan
    F["p_xenc_present"] = F["p_xenc"].notna().astype(np.int8)
    F["p_xenc"] = F["p_xenc"].fillna(0.5)
    return F


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED)


def train_oof(F, y, groups, n_folds, X_test=None):
    oof = np.zeros(len(F))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    gkf = GroupKFold(n_splits=n_folds)
    imp = np.zeros(len(FEATURES))
    for k, (tr, va) in enumerate(gkf.split(F, y, groups)):
        dtr = lgb.Dataset(F.iloc[tr][FEATURES], y[tr])
        dva = lgb.Dataset(F.iloc[va][FEATURES], y[va])
        m = lgb.train(LGB_PARAMS, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(F.iloc[va][FEATURES], num_iteration=m.best_iteration)
        imp += m.feature_importance("gain")
        if X_test is not None:
            test_pred += m.predict(X_test[FEATURES], num_iteration=m.best_iteration) / n_folds
        log(f"  fold {k}: best_iter={m.best_iteration}")
    return oof, test_pred, pd.Series(imp, FEATURES).sort_values(ascending=False)


# ----------------------------------------------------------------------------------------------
# Decision layer
# ----------------------------------------------------------------------------------------------
def f05(pred, gold):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(gold) + len(pred))


def macro_f05(pred, gold, s1_ids):
    return float(np.mean([f05(pred.get(s, set()), gold.get(s, set())) for s in s1_ids]))


def apply_exclusivity(D):
    """Each S2/S3 record belongs to at most one S1: zero out all but its best-scoring S1."""
    D = D.copy()
    best = D.groupby("cand")["p"].transform("max")
    first = ~D.sort_values("p", ascending=False).duplicated("cand")
    D.loc[~((D.p == best) & first.reindex(D.index)), "p"] = 0.0
    return D


def decide_threshold(D, t1, t2):
    """First (best) match needs p>=t1; every additional match needs p>=t2 (t2 usually > t1)."""
    out = {}
    for s, g in D[D.p >= min(t1, t2)].groupby("s1"):
        g = g.sort_values("p", ascending=False)
        if g.p.iloc[0] < t1:
            continue
        out[s] = {g.cand.iloc[0]} | set(g.cand.iloc[1:][g.p.iloc[1:] >= t2])
    return out


def decide_expected_f(D, n_mc=1000, p_floor=0.02, max_k=15, p_unseen=0.0, seed=SEED):
    """Pick the top-k set that maximises expected per-entity F0.5 under independent Bernoulli
    match probabilities (k=0 means 'predict singleton'). Threshold-free given calibrated p, which
    makes it the safer choice for countries never seen in training. p_unseen = chance the entity
    has a true match that blocking missed (raise it if blocking recall is imperfect)."""
    rng = np.random.default_rng(seed)
    E = D[D.p >= p_floor].sort_values(["s1", "p"], ascending=[True, False])
    E = E[E.groupby("s1").cumcount() < max_k]
    s1v, cv, pv = E.s1.values, E.cand.values, E.p.values
    bounds = np.flatnonzero(np.r_[True, s1v[1:] != s1v[:-1], True])
    out = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = pv[a:b]
        draws = rng.random((n_mc, len(p))) < p
        total = draws.sum(1) + (rng.random(n_mc) < p_unseen)
        tp = np.cumsum(draws, axis=1)                        # TP when predicting top-k, k=1..n
        k = np.arange(1, len(p) + 1)
        denom = 1.25 * tp + 0.25 * (total[:, None] - tp) + (k - tp)
        v = np.where(tp > 0, 1.25 * tp / np.maximum(denom, 1e-9), 0.0).mean(0)
        if v.max() > float(np.mean(total == 0)):             # beat "predict empty"?
            out[s1v[a]] = set(cv[a:a + int(v.argmax()) + 1])
    return out


def _fast_threshold_scorer(D, gold, s1_ids):
    """Precompute once so each (t1, t2) evaluation is a few vectorised numpy ops."""
    D = D.sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
    pos = {s: k for k, s in enumerate(s1_ids)}
    code = D.s1.map(pos).values
    first = (D.groupby("s1").cumcount() == 0).values
    top_p = D.groupby("s1")["p"].transform("first").values
    y = np.array([c in gold.get(s, ()) for s, c in zip(D.s1, D.cand)], dtype=float)
    p = D.p.values
    G = np.array([len(gold.get(s, ())) for s in s1_ids], dtype=float)
    n = len(s1_ids)

    def score(t1, t2):
        sel = (top_p >= t1) & (first | (p >= t2))
        npred = np.bincount(code[sel], minlength=n).astype(float)
        tp = np.bincount(code[sel], weights=y[sel], minlength=n)
        f = np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * G + npred, 1e-9), 0.0)
        f = np.where((G == 0) & (npred == 0), 1.0, f)
        return float(f.mean())
    return score


def search_decisions(D, gold, s1_ids):
    """Evaluate decision variants on OOF predictions; returns (sorted results, variant specs)."""
    res, variants = [], {}
    for excl in (False, True):
        Dx = apply_exclusivity(D) if excl else D
        tag = "excl" if excl else "raw"
        scorer = _fast_threshold_scorer(Dx, gold, s1_ids)
        grid = [(scorer(t1, t2), (round(float(t1), 3), round(float(t2), 3)))
                for t1 in np.arange(0.15, 0.86, 0.025) for t2 in np.arange(0.3, 0.98, 0.025)]
        sc, (t1, t2) = max(grid)
        name = f"threshold[{tag}] t1={t1} t2={t2}"
        res.append((name, sc))
        variants[name] = (excl, "thr", (t1, t2))
        name = f"expected_f[{tag}]"
        res.append((name, macro_f05(decide_expected_f(Dx), gold, s1_ids)))
        variants[name] = (excl, "expf", None)
    res.sort(key=lambda r: -r[1])
    return res, variants


def run_decision(D, spec):
    excl, kind, params = spec
    Dx = apply_exclusivity(D) if excl else D
    return decide_threshold(Dx, *params) if kind == "thr" else decide_expected_f(Dx)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def build(data_dir, split, args):
    log(f"loading {split}")
    R = load_split(data_dir, split)
    log(f"  {split}: " + ", ".join(f"{k}={v}" for k, v in R.groupby(['src']).size().items()) +
        " | countries: " + ", ".join(f"{k}={v}" for k, v in R.country.value_counts().items()))
    idf = compute_idf(R)
    args.embed_candidates = args.embed_candidates_train if split == "train" else args.embed_candidates_test
    C = generate_candidates(R, args)
    log(f"  {split}: {len(C):,} candidate pairs, {len(C) / max((R.src == 'S1').sum(), 1):.1f} per S1")
    F = featurize(C, R, idf, args.n_jobs)
    F["s1"] = R.loc[F.i, "entity_id"].values
    F["cand"] = R.loc[F.j, "entity_id"].values
    F["country"] = R.loc[F.i, "country"].values
    log(f"  {split}: features done")
    return R, F


def diagnostics(R, F, gold):
    s1_ids = R.loc[R.src == "S1", "entity_id"].tolist()
    country_of = dict(zip(R.entity_id, R.country))
    gold_pairs = {(s, c) for s, cs in gold.items() for c in cs}
    cand_pairs = set(zip(F.s1, F.cand))
    n_single = sum(1 for s in s1_ids if not gold.get(s))
    log("---- dataset / blocking diagnostics ----")
    log(f"  S1 entities: {len(s1_ids):,} | singletons: {n_single:,} ({n_single / len(s1_ids):.1%})")
    sizes = Counter(len(gold.get(s, ())) for s in s1_ids)
    log(f"  match-set size distribution: {dict(sorted(sizes.items())[:10])}")
    cross = sum(1 for s, c in gold_pairs if country_of.get(s) != country_of.get(c))
    log(f"  cross-country gold pairs: {cross} (if 0, per-country blocking is safe)")
    multi = sum(1 for v in Counter(c for _, c in gold_pairs).values() if v > 1)
    log(f"  S2/S3 records matched to >1 S1: {multi} (if 0, exclusivity constraint is valid)")
    rec = len(gold_pairs & cand_pairs) / max(len(gold_pairs), 1)
    ent_full = np.mean([gold[s] <= {c for c in gold[s] if (s, c) in cand_pairs}
                        for s in s1_ids if gold.get(s)])
    log(f"  blocking pair recall: {rec:.4f} | entities with ALL matches recalled: {ent_full:.4f}")
    for flag in ("f_name", "f_full", "f_zip", "f_rev", "f_emb"):
        sub = set(zip(F.s1[F[flag] == 1], F.cand[F[flag] == 1]))
        log(f"    {flag:7s}: recall alone {len(gold_pairs & sub) / max(len(gold_pairs), 1):.4f}, "
            f"{len(sub):,} pairs")


def score_report(name, pred, gold, s1_ids, country_of):
    msg = f"  {name}: macro F0.5 = {macro_f05(pred, gold, s1_ids):.4f}"
    for c in sorted({country_of[s] for s in s1_ids}):
        ids = [s for s in s1_ids if country_of[s] == c]
        msg += f" | {c}: {macro_f05(pred, gold, ids):.4f}"
    sing = [s for s in s1_ids if not gold.get(s)]
    if sing:
        msg += f" | singletons: {macro_f05(pred, gold, sing):.4f}"
    log(msg)


def main():
    global FEATURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work_dir", help="diagnostic files (OOF, test scores)")
    ap.add_argument("--mode", choices=["cv", "full"], default="cv")
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out check (France proxy)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k-name", type=int, default=10)
    ap.add_argument("--k-full", type=int, default=10)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--max-cands", type=int, default=50)
    ap.add_argument("--max-trigram-doc-freq", type=int, default=5000,
                     help="absolute cap on a char-trigram's combined S1+target document frequency "
                          "before it's dropped from blocking; keeps the TF-IDF blocker's sparse "
                          "matmul tractable at million-row country sizes (see prune_common_columns)")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--embed-candidates-train", default=None,
                     help="tsv of (source1_entity_id, candidate_entity_id, cos_emb) from "
                          "gpu_embed_blocker.py run on dataset/train; optional GPU-only blocker")
    ap.add_argument("--embed-candidates-test", default=None,
                     help="same, for dataset/test (only used with --mode full)")
    ap.add_argument("--xenc-train-probs", default=None,
                     help="tsv of (source1_entity_id, candidate_entity_id, p_xenc) from "
                          "cross_encoder.py OOF predictions on train; optional GPU-only feature")
    ap.add_argument("--xenc-test-probs", default=None,
                     help="same, for test-set cross-encoder inference (only used with --mode full)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    R, F = build(args.data, "train", args)
    gt = read_tsv(os.path.join(args.data, "train", "train_ground_truth.tsv"))
    gold = {s: set(filter(None, m.split(","))) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    s1_ids = R.loc[R.src == "S1", "entity_id"].tolist()
    country_of = dict(zip(R.entity_id, R.country))
    diagnostics(R, F, gold)
    gold_pairs = {(s, c) for s, cs in gold.items() for c in cs}
    y = np.array([(s, c) in gold_pairs for s, c in zip(F.s1, F.cand)], dtype=int)

    F = merge_xenc_probs(F, args.xenc_train_probs)
    if args.xenc_train_probs:
        log(f"  cross-encoder feature: {F.p_xenc_present.mean():.1%} of train pairs scored")

    ignore = {"i", "j", "s1", "cand", "country"}
    FEATURES = [c for c in F.columns if c not in ignore]

    Ft = None
    if args.mode == "full":
        Rt, Ft = build(args.data, "test", args)
        Ft = merge_xenc_probs(Ft, args.xenc_test_probs)

    log("training LightGBM (GroupKFold by S1 entity)")
    oof, test_pred, imp = train_oof(F, y, F.s1.values, args.folds, Ft)
    log("  top features: " + ", ".join(imp.index[:12]))
    D = pd.DataFrame({"s1": F.s1, "cand": F.cand, "p": oof})
    D.to_csv(os.path.join(args.work, "oof_train.tsv"), sep="\t", index=False)

    log("---- decision layer (OOF) ----")
    res, variants = search_decisions(D, gold, s1_ids)
    for name, sc in res:
        log(f"  {sc:.4f}  {name}")
    best_name = res[0][0]
    score_report(f"BEST {best_name}", run_decision(D, variants[best_name]), gold, s1_ids, country_of)
    cand_set = set(zip(F.s1, F.cand))
    oracle = {s: {c for c in gold.get(s, ()) if (s, c) in cand_set} for s in s1_ids}
    score_report("ceiling (perfect matcher on these candidates)", oracle, gold, s1_ids, country_of)

    if args.loco and F.country.nunique() > 1:
        log("---- leave-one-country-out (train on others, threshold-free expected-F decision) ----")
        for c in sorted(F.country.unique()):
            tr, te = (F.country != c).values, (F.country == c).values
            m = lgb.train(LGB_PARAMS, lgb.Dataset(F.loc[tr, FEATURES], y[tr]), 400)
            Dc = pd.DataFrame({"s1": F.s1[te], "cand": F.cand[te],
                               "p": m.predict(F.loc[te, FEATURES])})
            ids = [s for s in s1_ids if country_of[s] == c]
            for spec_name in (best_name, "expected_f[excl]"):
                sc = macro_f05(run_decision(Dc, variants[spec_name]), gold, ids)
                log(f"  held-out {c}: {sc:.4f} with {spec_name}")

    if args.mode == "full":
        log(f"---- predicting test with: {best_name} ----")
        Dt = pd.DataFrame({"s1": Ft.s1, "cand": Ft.cand, "p": test_pred})
        Dt.to_csv(os.path.join(args.work, "test_scores.tsv"), sep="\t", index=False)
        pred = run_decision(Dt, variants[best_name])
        t_s1 = Rt.loc[Rt.src == "S1", "entity_id"].tolist()
        cands = Dt.sort_values("p", ascending=False).groupby("s1")["cand"].apply(list).to_dict()
        pred = {s: [c for c in cands.get(s, []) if c in m] for s, m in pred.items()}  # keep p order
        write_id_lists(os.path.join(args.out, "candidate_pairs.tsv"), t_s1, cands, "candidate_entity_ids")
        write_id_lists(os.path.join(args.out, "matching_results.tsv"), t_s1, pred, "matched_entity_ids")
        valid = set(Rt.loc[Rt.src != "S1", "entity_id"])
        assert all(c in valid for v in pred.values() for c in v)
        assert all(set(pred.get(s, [])) <= set(cands.get(s, [])) for s in t_s1)
        n_match = sum(1 for s in t_s1 if pred.get(s))
        t_country = dict(zip(Rt.entity_id, Rt.country))
        log(f"  wrote outputs: {n_match:,}/{len(t_s1):,} S1 entities with >=1 match; by country: " +
            ", ".join(f"{c}={np.mean([bool(pred.get(s)) for s in t_s1 if t_country[s] == c]):.2f}"
                      for c in sorted(Rt.country.unique())))
    log("done")


if __name__ == "__main__":
    main()
