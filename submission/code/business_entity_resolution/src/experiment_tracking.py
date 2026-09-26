"""Experiment history: every run appends one row to experiments/experiment_log.csv, and
updates experiments/best_result.json if it's the best validation F0.5 seen so far.

This exists so "we improved F0.5" is always backed by a traceable row (config + measured
numbers), not a claim -- and so the best-known config is never silently lost to a later,
worse run.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import uuid
from datetime import datetime, timezone

from config import PipelineConfig

LOG_COLUMNS = [
    "experiment_id", "date", "description", "mode",
    "blocking_k_name", "blocking_k_full", "blocking_k_rev", "blocking_max_cands",
    "blocking_max_trigram_doc_freq", "model_num_leaves", "model_learning_rate", "n_folds",
    "best_decision_rule", "validation_f05", "blocking_pair_recall", "candidate_recall_entities",
    "avg_candidates_per_s1", "singleton_rate", "n_test_matched", "runtime_seconds", "notes",
]


def log_experiment(cfg: PipelineConfig, metrics: dict, notes: str = "") -> str:
    os.makedirs(cfg.experiments_dir, exist_ok=True)
    log_path = os.path.join(cfg.experiments_dir, "experiment_log.csv")
    best_path = os.path.join(cfg.experiments_dir, "best_result.json")

    experiment_id = uuid.uuid4().hex[:8]
    row = {
        "experiment_id": experiment_id,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": f"mode={cfg.mode}" + (" +loco" if cfg.loco else ""),
        "mode": cfg.mode,
        "blocking_k_name": cfg.blocking.k_name,
        "blocking_k_full": cfg.blocking.k_full,
        "blocking_k_rev": cfg.blocking.k_rev,
        "blocking_max_cands": cfg.blocking.max_cands,
        "blocking_max_trigram_doc_freq": cfg.blocking.max_trigram_doc_freq,
        "model_num_leaves": cfg.model.num_leaves,
        "model_learning_rate": cfg.model.learning_rate,
        "n_folds": cfg.model.n_folds,
        "best_decision_rule": metrics.get("best_decision_rule"),
        "validation_f05": metrics.get("validation_f05"),
        "blocking_pair_recall": metrics.get("blocking_pair_recall"),
        "candidate_recall_entities": metrics.get("candidate_recall_entities"),
        "avg_candidates_per_s1": metrics.get("avg_candidates_per_s1"),
        "singleton_rate": metrics.get("singleton_rate"),
        "n_test_matched": metrics.get("n_test_matched"),
        "runtime_seconds": round(metrics.get("runtime_seconds", 0), 1),
        "notes": notes,
    }

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    current_best = None
    if os.path.exists(best_path):
        with open(best_path, encoding="utf-8") as f:
            current_best = json.load(f)

    validation_f05 = metrics.get("validation_f05")
    if validation_f05 is not None and (current_best is None or validation_f05 > current_best.get("validation_f05", -1)):
        best_payload = dict(row)
        best_payload["config"] = dataclasses.asdict(cfg)
        best_payload["loco"] = metrics.get("loco")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2, default=str)

    return experiment_id
