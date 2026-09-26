#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution: end-to-end pipeline (modular rewrite).

  normalize -> block (multi-view, both directions) -> pair features -> LightGBM (GroupKFold OOF)
  -> exclusivity -> per-entity expected-F0.5 / threshold set selection
  -> matching_results.tsv + candidate_pairs.tsv

Usage (run from the directory containing dataset/):
  python pipeline.py --data dataset --out output --mode cv           # local validation only
  python pipeline.py --data dataset --out output --mode cv --loco    # + leave-one-country-out
  python pipeline.py --data dataset --out output --mode full         # CV + predict test + write outputs

Every run appends one row to experiments/experiment_log.csv and, if it's the best OOF F0.5 seen
so far, updates experiments/best_result.json -- see `experiment_tracking.py`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blocking
import data_loader
import evaluation
import experiment_tracking
import features
import inference
import preprocessing
import submission
import training
from config import PipelineConfig

log = logging.getLogger("pipeline")


def build_split(data_dir: str, split: str, cfg: PipelineConfig):
    """Load + normalize + block + featurize one split ("train" or "test")."""
    t0 = time.time()
    sources = data_loader.load_split(data_dir, split)
    R = pd.concat(
        [preprocessing.normalize_source(df, f"S{k}") for k, df in sources.items()],
        ignore_index=True,
    )
    log.info("%s: %s S1, %s S2, %s S3 | countries: %s", split,
             f"{(R.src=='S1').sum():,}", f"{(R.src=='S2').sum():,}", f"{(R.src=='S3').sum():,}",
             R.country.value_counts().to_dict())

    idf = features.compute_idf(R)
    embed_path = cfg.embed_candidates_train if split == "train" else cfg.embed_candidates_test
    C = blocking.generate_candidates(R, cfg.blocking, embed_path)
    log.info("%s: %s candidate pairs (%.1f per S1)", split, f"{len(C):,}",
             len(C) / max((R.src == "S1").sum(), 1))

    F = features.featurize(C, R, idf, cfg.n_jobs)
    F["s1"] = R.loc[F.i, "entity_id"].values
    F["cand"] = R.loc[F.j, "entity_id"].values
    F["country"] = R.loc[F.i, "country"].values
    log.info("%s: features done in %.1fs", split, time.time() - t0)
    return R, C, F


def merge_xenc_probs(F: pd.DataFrame, path: str | None) -> pd.DataFrame:
    """Stack in an optional GPU cross-encoder probability as one more LightGBM feature. Absent
    file -> neutral 0.5 prior everywhere, so the pipeline is fully runnable without a GPU."""
    if path and os.path.exists(path):
        X = data_loader.read_tsv(path).rename(
            columns={"source1_entity_id": "s1", "candidate_entity_id": "cand"})
        X["p_xenc"] = X["p_xenc"].astype(np.float32)
        F = F.merge(X[["s1", "cand", "p_xenc"]], on=["s1", "cand"], how="left")
    else:
        F = F.copy()
        F["p_xenc"] = np.nan
    F["p_xenc_present"] = F["p_xenc"].notna().astype(np.int8)
    F["p_xenc"] = F["p_xenc"].fillna(0.5)
    return F


def run_blocking_audit(R, C, gold, cfg: PipelineConfig) -> dict:
    gold_pairs = {(s, c) for s, cs in gold.matches.items() for c in cs}
    audit = blocking.blocking_audit(R, C, gold_pairs)

    s1_ids = R.loc[R.src == "S1", "entity_id"].tolist()
    country_of = dict(zip(R.entity_id, R.country))
    cross_country = sum(1 for s, c in gold_pairs if country_of.get(s) != country_of.get(c))
    match_target_counts: dict[str, int] = {}
    for _, c in gold_pairs:
        match_target_counts[c] = match_target_counts.get(c, 0) + 1
    exclusivity_violations = sum(1 for v in match_target_counts.values() if v > 1)

    log.info("---- blocking audit ----")
    log.info("S1 entities: %s | pair recall: %.4f | entities fully recalled: %.4f",
             f"{len(s1_ids):,}", audit["pair_recall"], audit["entities_fully_recalled"])
    log.info("candidates/S1: %.1f | S1 with zero candidates: %s",
             audit["candidates_per_s1"], audit["s1_with_zero_candidates"])
    log.info("cross-country gold pairs: %s (expect 0 -- validates per-country blocking)", cross_country)
    log.info("S2/S3 matched to >1 S1: %s (expect 0 -- validates exclusivity assumption)", exclusivity_violations)
    for flag, stats in audit["per_blocker"].items():
        log.info("  %-7s: recall alone %.4f, %s pairs", flag, stats["recall_alone"], f"{stats['pairs']:,}")

    os.makedirs(cfg.reports_dir, exist_ok=True)
    with open(os.path.join(cfg.reports_dir, "blocking_baseline.md"), "w", encoding="utf-8") as f:
        f.write("# Blocking Audit\n\n")
        f.write(f"- S1 entities: {len(s1_ids):,}\n")
        f.write(f"- Candidate pairs: {audit['n_candidate_pairs']:,} ({audit['candidates_per_s1']:.1f} per S1)\n")
        f.write(f"- Pair recall (union of blockers): {audit['pair_recall']:.4f}\n")
        f.write(f"- Entities with ALL matches recalled: {audit['entities_fully_recalled']:.4f}\n")
        f.write(f"- S1 entities with zero candidates: {audit['s1_with_zero_candidates']}\n")
        f.write(f"- Cross-country gold pairs: {cross_country} (expect 0)\n")
        f.write(f"- S2/S3 matched to >1 S1: {exclusivity_violations} (expect 0)\n\n")
        f.write("## Per-blocker recall (in isolation)\n\n")
        for flag, stats in audit["per_blocker"].items():
            f.write(f"- **{flag}**: recall {stats['recall_alone']:.4f}, {stats['pairs']:,} pairs\n")

    audit["cross_country_gold_pairs"] = cross_country
    audit["exclusivity_violations"] = exclusivity_violations
    return audit


def main():
    logging.basicConfig(level=logging.INFO, format="[%(relativeCreated)7.1fms] %(message)s")
    t_start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work_dir")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--experiments", default="experiments")
    ap.add_argument("--mode", choices=["cv", "full"], default="cv")
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out check (France proxy)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k-name", type=int, default=10)
    ap.add_argument("--k-full", type=int, default=10)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--max-cands", type=int, default=50)
    ap.add_argument("--max-trigram-doc-freq", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--embed-candidates-train", default=None)
    ap.add_argument("--embed-candidates-test", default=None)
    ap.add_argument("--xenc-train-probs", default=None)
    ap.add_argument("--xenc-test-probs", default=None)
    ap.add_argument("--experiment-notes", default="")
    args = ap.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data, out_dir=args.out, work_dir=args.work,
        reports_dir=args.reports, experiments_dir=args.experiments,
        mode=args.mode, loco=args.loco, n_jobs=args.n_jobs,
        embed_candidates_train=args.embed_candidates_train,
        embed_candidates_test=args.embed_candidates_test,
        xenc_train_probs=args.xenc_train_probs, xenc_test_probs=args.xenc_test_probs,
    )
    cfg.blocking.k_name, cfg.blocking.k_full, cfg.blocking.k_rev = args.k_name, args.k_full, args.k_rev
    cfg.blocking.max_cands = args.max_cands
    cfg.blocking.max_trigram_doc_freq = args.max_trigram_doc_freq
    cfg.model.n_folds = args.folds
    for d in (cfg.out_dir, cfg.work_dir, cfg.reports_dir, cfg.experiments_dir):
        os.makedirs(d, exist_ok=True)

    R, C, F = build_split(cfg.data_dir, "train", cfg)
    s1_ids_train = set(R.loc[R.src == "S1", "entity_id"])
    s2_ids = set(R.loc[R.src == "S2", "entity_id"])
    s3_ids = set(R.loc[R.src == "S3", "entity_id"])
    gold = data_loader.parse_ground_truth(cfg.data_dir, "train", s1_ids_train, s2_ids, s3_ids)
    gold_dict = {k: set(v) for k, v in gold.matches.items()}

    s1_ids = R.loc[R.src == "S1", "entity_id"].tolist()
    country_of = dict(zip(R.entity_id, R.country))
    audit = run_blocking_audit(R, C, gold, cfg)

    gold_pairs = {(s, c) for s, cs in gold.matches.items() for c in cs}
    y = np.array([(s, c) in gold_pairs for s, c in zip(F.s1, F.cand)], dtype=int)
    F = merge_xenc_probs(F, cfg.xenc_train_probs)

    ignore_cols = {"i", "j", "s1", "cand", "country"}
    feature_cols = [c for c in F.columns if c not in ignore_cols]

    Rt = Ct = Ft = None
    if cfg.mode == "full":
        Rt, Ct, Ft = build_split(cfg.data_dir, "test", cfg)
        Ft = merge_xenc_probs(Ft, cfg.xenc_test_probs)

    log.info("training LightGBM (GroupKFold by S1 entity, %d folds)", cfg.model.n_folds)
    oof, test_pred, importance = training.train_oof(F, y, F.s1.values, feature_cols, cfg.model, Ft)
    log.info("top features: %s", ", ".join(importance.index[:12]))

    D = pd.DataFrame({"s1": F.s1, "cand": F.cand, "p": oof})
    D.to_csv(os.path.join(cfg.work_dir, "oof_train.tsv"), sep="\t", index=False)

    log.info("---- decision layer (OOF) ----")
    results, variants = evaluation.search_decisions(
        D, gold_dict, s1_ids, cfg.decision.threshold_t1_grid, cfg.decision.threshold_t2_grid)
    for name, score in results:
        log.info("  %.4f  %s", score, name)
    best_name, best_oof_f05 = results[0]
    log.info(evaluation.score_report(f"BEST {best_name}",
             evaluation.run_decision(D, variants[best_name]), gold_dict, s1_ids, country_of))

    cand_set = set(zip(F.s1, F.cand))
    oracle = {s: {c for c in gold_dict.get(s, ()) if (s, c) in cand_set} for s in s1_ids}
    log.info(evaluation.score_report("ceiling (perfect matcher on these candidates)",
             oracle, gold_dict, s1_ids, country_of))

    loco_results = {}
    if cfg.loco and F.country.nunique() > 1:
        log.info("---- leave-one-country-out (proxy for the unseen France test country) ----")
        import lightgbm as lgb
        for held_out in sorted(F.country.unique()):
            train_mask, test_mask = (F.country != held_out).values, (F.country == held_out).values
            model = lgb.train(cfg.model.as_lgb_params(),
                              lgb.Dataset(F.loc[train_mask, feature_cols], y[train_mask]), 400)
            Dc = pd.DataFrame({"s1": F.s1[test_mask], "cand": F.cand[test_mask],
                               "p": model.predict(F.loc[test_mask, feature_cols])})
            ids = [s for s in s1_ids if country_of[s] == held_out]
            for spec_name in (best_name, "expected_f[excl]"):
                score = evaluation.macro_f05(evaluation.run_decision(Dc, variants[spec_name]), gold_dict, ids)
                log.info("  held-out %s: %.4f with %s", held_out, score, spec_name)
                loco_results[f"{held_out}/{spec_name}"] = score

    n_match = None
    if cfg.mode == "full":
        log.info("---- predicting test with: %s ----", best_name)
        predictions, candidates = inference.predict_test_matches(Ft, test_pred, variants[best_name])
        pd.DataFrame({"s1": Ft.s1, "cand": Ft.cand, "p": test_pred}).to_csv(
            os.path.join(cfg.work_dir, "test_scores.tsv"), sep="\t", index=False)

        test_s1_ids = Rt.loc[Rt.src == "S1", "entity_id"].tolist()
        valid_targets = set(Rt.loc[Rt.src != "S1", "entity_id"])
        submission.self_check_predictions(predictions, candidates, test_s1_ids, valid_targets)

        submission.write_id_list_file(os.path.join(cfg.out_dir, "candidate_pairs.tsv"),
                                       test_s1_ids, candidates, "candidate_entity_ids")
        submission.write_id_list_file(os.path.join(cfg.out_dir, "matching_results.tsv"),
                                       test_s1_ids, predictions, "matched_entity_ids")

        n_match = sum(1 for s in test_s1_ids if predictions.get(s))
        t_country = dict(zip(Rt.entity_id, Rt.country))
        log.info("wrote outputs: %s/%s S1 entities with >=1 match; by country: %s",
                 f"{n_match:,}", f"{len(test_s1_ids):,}",
                 ", ".join(f"{c}={np.mean([bool(predictions.get(s)) for s in test_s1_ids if t_country[s]==c]):.2f}"
                          for c in sorted(Rt.country.unique())))

    experiment_tracking.log_experiment(
        cfg,
        metrics=dict(
            blocking_pair_recall=audit["pair_recall"],
            candidate_recall_entities=audit["entities_fully_recalled"],
            avg_candidates_per_s1=audit["candidates_per_s1"],
            best_decision_rule=best_name,
            validation_f05=best_oof_f05,
            singleton_rate=sum(1 for s in s1_ids if not gold_dict.get(s)) / len(s1_ids),
            n_test_matched=n_match,
            loco=loco_results,
            runtime_seconds=time.time() - t_start,
        ),
        notes=args.experiment_notes,
    )
    log.info("done in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
