#!/usr/bin/env python3
"""
GPU dense-embedding blocker (Ayush's L40, 96GB VRAM). Additive, optional stage: er_pipeline.py
runs fine without ever calling this. Its whole reason to exist is recall that char-3gram TF-IDF
structurally cannot get:

  * cross-script matches: a Source-2 business name written in Devanagari against a Source-1 name
    transliterated into Latin script share almost no character n-grams, but a multilingual sentence
    embedding trained across scripts puts them close together in vector space.
  * heavy word-order shuffles / paraphrased legal form ("Robotics Acme Pvt Ltd" vs "Acme Robotics
    Private Limited") that survive semantic embedding better than character overlap.

Model: intfloat/multilingual-e5-base (278M params, MIT license, well within the ≤8B / MIT-Apache
constraint). Covers English, Hindi and French, which is exactly the training + test country mix.
No internet lookups happen at inference time beyond the one-time model download; nothing here
touches an external database or API to resolve identities, so it stays inside the fair-play rules.

Usage (run from student_resource/, requires the code/business_entity_resolution/src folder on
PYTHONPATH or run from inside it so `import er_pipeline` resolves):

  python gpu_embed_blocker.py --data dataset --split train \
      --out work_dir/embed_candidates_train.tsv
  python gpu_embed_blocker.py --data dataset --split test \
      --out work_dir/embed_candidates_test.tsv

Then feed the outputs back into er_pipeline.py:

  python er_pipeline.py --data dataset --out output --mode full \
      --embed-candidates-train work_dir/embed_candidates_train.tsv \
      --embed-candidates-test  work_dir/embed_candidates_test.tsv \
      ...

Output format: long TSV, one row per candidate pair, columns
  source1_entity_id  candidate_entity_id  cos_emb
This is an intermediate working file (goes under work_dir/, not output/) — it is NOT
candidate_pairs.tsv. It only widens what er_pipeline.py's own blocking union considers; the
submitted candidate_pairs.tsv is still whatever er_pipeline.py ends up feeding its model.
"""
import argparse
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import er_pipeline as ep

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def encode(model, texts, prefix, batch_size):
    """E5 models require a 'query: '/'passage: ' instruction prefix baked into the input text."""
    texts = [f"{prefix}{t}" for t in texts]
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)


def topk_by_matmul(Q, T, k, device, chunk=4096):
    """GPU brute-force top-k cosine (vectors already L2-normalised). At this scale (tens of
    thousands of candidates per country/source group after per-country partitioning) a dense
    matmul on a 96GB GPU is simpler and just as fast as standing up a FAISS index, and avoids an
    extra dependency; swap in faiss-gpu here if a country group ever gets too large for VRAM."""
    k = min(k, T.shape[0])
    if k == 0 or Q.shape[0] == 0:
        return np.zeros((Q.shape[0], 0), np.int64), np.zeros((Q.shape[0], 0), np.float32)
    Tt = torch.as_tensor(T, device=device).T
    idx_out = np.empty((Q.shape[0], k), np.int64)
    sim_out = np.empty((Q.shape[0], k), np.float32)
    for a in range(0, Q.shape[0], chunk):
        q = torch.as_tensor(Q[a:a + chunk], device=device)
        sim, idx = torch.topk(q @ Tt, k, dim=1)
        idx_out[a:a + chunk] = idx.cpu().numpy()
        sim_out[a:a + chunk] = sim.cpu().numpy()
    return idx_out, sim_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", required=True, help="output tsv path (goes under work_dir/)")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--k", type=int, default=15, help="top-k neighbours per direction")
    ap.add_argument("--batch-size", type=int, default=512,
                     help="96GB VRAM comfortably fits large batches for a 278M-param encoder")
    ap.add_argument("--min-cos", type=float, default=0.75,
                     help="drop weak neighbours below this cosine to keep the file small")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (WARNING: no GPU found)"))
    model = SentenceTransformer(args.model, device=device)

    log(f"loading {args.split}")
    R = ep.load_split(args.data, args.split)
    text = (R["name_core"] + " | " + R["addr_norm"]).tolist()

    pairs = []
    for country, g in R.groupby("country"):
        s1 = g[g.src == "S1"]
        if len(s1) == 0:
            continue
        s1_vecs = encode(model, [text[i] for i in s1.index], "query: ", args.batch_size)
        for src in ("S2", "S3"):
            tg = g[g.src == src]
            if len(tg) == 0:
                continue
            tg_vecs = encode(model, [text[j] for j in tg.index], "passage: ", args.batch_size)
            idx, sim = topk_by_matmul(s1_vecs, tg_vecs, args.k, device)
            for a in range(idx.shape[0]):
                for b, sv in zip(idx[a], sim[a]):
                    if sv >= args.min_cos:
                        pairs.append((s1.entity_id.iloc[a], tg.entity_id.iloc[b], float(sv)))
            # reverse direction: candidates that are a target record's own best S1 match, even if
            # they don't make that S1's own top-k list (mirrors er_pipeline's f_rev blocker)
            idx_r, sim_r = topk_by_matmul(tg_vecs, s1_vecs, min(args.k, 5), device)
            for b in range(idx_r.shape[0]):
                for a, sv in zip(idx_r[b], sim_r[b]):
                    if sv >= args.min_cos:
                        pairs.append((s1.entity_id.iloc[a], tg.entity_id.iloc[b], float(sv)))
            log(f"  {country}/{src}: {len(s1):,} x {len(tg):,} -> {len(pairs):,} pairs so far")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tcos_emb\n")
        seen = {}
        for s1_id, cand_id, sv in pairs:
            key = (s1_id, cand_id)
            if sv > seen.get(key, -1.0):
                seen[key] = sv
        for (s1_id, cand_id), sv in seen.items():
            f.write(f"{s1_id}\t{cand_id}\t{sv:.4f}\n")
    log(f"wrote {len(seen):,} unique (S1, candidate) pairs -> {args.out}")


if __name__ == "__main__":
    main()
