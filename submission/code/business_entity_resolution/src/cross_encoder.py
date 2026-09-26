#!/usr/bin/env python3
"""
GPU cross-encoder stacking feature (Ayush's L40, 96GB VRAM). Additive, optional stage.

Fine-tunes microsoft/mdeberta-v3-base (~280M params, MIT license — inside the ≤8B / MIT-Apache
constraint) as a binary "same business?" classifier directly on
  "name: <S1 name> | address: <S1 address>"  [SEP]  "name: <candidate name> | address: <candidate address>"
using the training ground truth as labels and er_pipeline's own blocking candidates as negatives
(mostly hard negatives: businesses similar enough to have been blocked together but not a true
match). This is the strongest single feature the team can add on top of the tree-model baseline: a
transformer reads the full normalized text jointly, rather than through hand-built similarity
scores, which is exactly what's needed for the noisiest cases (transliteration, word-order
shuffles, abbreviation patterns the feature table didn't anticipate) and for the unseen French
country, where token-frequency features (IDF, name_freq) trained on US/India carry the least
information.

It is deliberately a SEPARATE script from pipeline.py, not a rewrite of it: it produces a
probability column that pipeline.py merges in as one more LightGBM feature (--xenc-train-probs /
--xenc-test-probs), so the GBM keeps doing what it's good at (combining the rank/context/exclusivity
signals) while this supplies the one feature it structurally cannot compute itself.

Compute budget (read before running): candidate lists are already capped at ~50/entity by
the pipeline's blocking stage, which is still too many pairs to fine-tune/score at full 2M+ S1
entity scale end to end. This script narrows to the top --topk-per-entity candidates per S1 (ranked by
the existing TF-IDF/embedding "combo" score) before touching the transformer, since disambiguating
among a handful of already-plausible candidates is exactly the job a cross-encoder is for; letting
it also reject far-fetched candidates is what the cheap CPU features already do well. Test-set
scale (~1.7M S1 x 6 candidates ~= 10M pairs) at batch 256 / seq_len 128 / fp16 is roughly a 1-2
hour inference job on an L40; budget accordingly and consider raising --topk-per-entity only once
that's comfortably within the time left.

Usage (run from student_resource/):

  # 1) out-of-fold probabilities on train, for stacking into the pipeline's LightGBM
  python cross_encoder.py --data dataset --mode cv \
      --embed-candidates-train work_dir/embed_candidates_train.tsv \
      --out work_dir/xenc_oof.tsv

  # 2) train on all of train, score test candidates
  python cross_encoder.py --data dataset --mode full \
      --embed-candidates-train work_dir/embed_candidates_train.tsv \
      --embed-candidates-test  work_dir/embed_candidates_test.tsv \
      --out work_dir/xenc_test.tsv

  # 3) feed both into the baseline pipeline
  python pipeline.py --data dataset --out output --mode full \
      --xenc-train-probs work_dir/xenc_oof.tsv --xenc-test-probs work_dir/xenc_test.tsv ...

Output format: long TSV, columns  source1_entity_id  candidate_entity_id  p_xenc
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import data_loader
from config import PipelineConfig
from pipeline import build_split as _pipeline_build_split

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def build_split(data_dir, split, embed_path, args):
    """Reuse pipeline.py's own blocking + feature stages so candidate pairs are identical to
    what the GBM sees; we only need R (for text) and F (for s1/cand ids + the 'combo' rank score
    used to pick the top-k per entity)."""
    cfg = PipelineConfig(data_dir=data_dir, n_jobs=args.n_jobs,
                         embed_candidates_train=embed_path if split == "train" else None,
                         embed_candidates_test=embed_path if split == "test" else None)
    cfg.blocking.k_name, cfg.blocking.k_full, cfg.blocking.k_rev = args.k_name, args.k_full, args.k_rev
    cfg.blocking.max_cands = args.max_cands
    cfg.blocking.max_trigram_doc_freq = args.max_trigram_doc_freq
    R, _C, F = _pipeline_build_split(data_dir, split, cfg)
    return R, F


def select_topk(F, k):
    F = F.sort_values(["s1", "combo"], ascending=[True, False])
    return F.groupby("s1", sort=False).head(k).reset_index(drop=True)


def entity_texts(R):
    return dict(zip(R.entity_id, "name: " + R.name_norm + " | address: " + R.addr_norm))


class PairDataset(Dataset):
    def __init__(self, s1_ids, cand_ids, labels, texts):
        self.a = [texts[s] for s in s1_ids]
        self.b = [texts[c] for c in cand_ids]
        self.y = labels  # None at inference time

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], (None if self.y is None else self.y[idx])


def make_collate(tokenizer, max_len):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tokenizer(list(a), list(b), truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        if y[0] is None:
            return enc, None
        return enc, torch.tensor(y, dtype=torch.float32)
    return collate


def run_epoch(model, loader, device, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    probs, losses = [], []
    for enc, y in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = model(**enc).logits.squeeze(-1)
                if y is not None:
                    loss = F_.binary_cross_entropy_with_logits(logits, y.to(device))
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs), (float(np.mean(losses)) if losses else None)


def fit_predict(train_s1, train_cand, train_y, val_s1, val_cand, texts, args, device):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    collate = make_collate(tok, args.max_len)
    tr_ds = PairDataset(train_s1, train_cand, train_y, texts)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                           num_workers=args.num_workers, drop_last=True)
    val_ds = PairDataset(val_s1, val_cand, None, texts)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_steps = (len(tr_loader) // 1) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * n_steps), n_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for ep_i in range(args.epochs):
        _, loss = run_epoch(model, tr_loader, device, optimizer, scheduler, scaler)
        log(f"    epoch {ep_i + 1}/{args.epochs}: train BCE = {loss:.4f}")

    val_probs, _ = run_epoch(model, val_loader, device)
    del model
    torch.cuda.empty_cache()
    return val_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--mode", choices=["cv", "full"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--folds", type=int, default=3, help="fewer folds than the GBM: fine-tuning is expensive")
    ap.add_argument("--topk-per-entity", type=int, default=6,
                     help="candidates per S1 sent through the transformer (see module docstring)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--k-name", type=int, default=10)
    ap.add_argument("--k-full", type=int, default=10)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--max-cands", type=int, default=50)
    ap.add_argument("--max-trigram-doc-freq", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--embed-candidates-train", default=None)
    ap.add_argument("--embed-candidates-test", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (WARNING: no GPU found, this will be slow)"))

    log("building train candidates/features (reusing er_pipeline blocking)")
    R, F = build_split(args.data, "train", args.embed_candidates_train, args)
    texts = entity_texts(R)
    gt = data_loader.read_tsv(os.path.join(args.data, "train", "train_ground_truth.tsv"))
    gold_pairs = {(s, c) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)
                 for c in filter(None, m.split(","))}
    F = select_topk(F, args.topk_per_entity)
    y = np.array([(s, c) in gold_pairs for s, c in zip(F.s1, F.cand)], dtype=np.float32)
    log(f"  {len(F):,} train pairs after top-{args.topk_per_entity} selection, "
        f"{int(y.sum()):,} positives ({y.mean():.2%})")

    if args.mode == "cv":
        oof = np.zeros(len(F))
        gkf = GroupKFold(n_splits=args.folds)
        for k, (tr, va) in enumerate(gkf.split(F, y, F.s1.values)):
            log(f"  fold {k}: fine-tuning on {len(tr):,} pairs, predicting {len(va):,}")
            oof[va] = fit_predict(F.s1.values[tr], F.cand.values[tr], y[tr],
                                  F.s1.values[va], F.cand.values[va], texts, args, device)
        out = pd.DataFrame({"source1_entity_id": F.s1, "candidate_entity_id": F.cand, "p_xenc": oof})
        out.to_csv(args.out, sep="\t", index=False)
        log(f"wrote OOF probabilities -> {args.out}")

    else:  # full: train on ALL train pairs, score test candidates
        log("building test candidates/features")
        Rt, Ft = build_split(args.data, "test", args.embed_candidates_test, args)
        Ft = select_topk(Ft, args.topk_per_entity)
        texts_t = entity_texts(Rt)
        texts.update(texts_t)
        log(f"  fine-tuning final model on all {len(F):,} train pairs, "
            f"scoring {len(Ft):,} test pairs")
        test_probs = fit_predict(F.s1.values, F.cand.values, y,
                                 Ft.s1.values, Ft.cand.values, texts, args, device)
        out = pd.DataFrame({"source1_entity_id": Ft.s1, "candidate_entity_id": Ft.cand,
                            "p_xenc": test_probs})
        out.to_csv(args.out, sep="\t", index=False)
        log(f"wrote test probabilities -> {args.out}")


if __name__ == "__main__":
    main()
