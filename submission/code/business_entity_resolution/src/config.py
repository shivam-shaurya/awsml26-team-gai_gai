"""Central configuration for the entity-resolution pipeline.

All tunable parameters live here so experiments are driven by config, not scattered
hard-coded literals. `PipelineConfig` is what `pipeline.py` builds from CLI args and passes
through every stage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


SEED = 42


@dataclass
class BlockingConfig:
    """Candidate-generation (blocking) parameters.

    k_name / k_full / k_rev control the size of each TF-IDF top-k blocker.
    max_trigram_doc_freq is the scalability-critical one: it caps how common a character
    trigram is allowed to be before it's dropped from blocking entirely. This is an ABSOLUTE
    cap, not a fraction of corpus size -- a relative max_df is not aggressive enough once the
    corpus reaches millions of rows, because short business-name strings share enough common
    trigrams that even the top ~5% most-frequent ones make the Q @ T.T sparse product blow up
    towards dense. Measured on the real per-country data: without this cap, a single blocker
    view/direction against the full ~3M-row US Source-2 file needs ~88 CPU-hours (dense
    chunking) or OOMs outright (naive sparse matmul); with an absolute cap of 5000, the same
    call finishes in ~2.3 minutes.
    """

    k_name: int = 10
    k_full: int = 10
    k_rev: int = 3
    max_cands: int = 50
    max_trigram_doc_freq: int = 5000


@dataclass
class ModelConfig:
    """LightGBM hyperparameters for the pair-match classifier."""

    objective: str = "binary"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    n_folds: int = 5

    def as_lgb_params(self) -> dict:
        return dict(
            objective=self.objective,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            lambda_l2=self.lambda_l2,
            verbose=-1,
            seed=SEED,
        )


def _frange(start: float, stop: float, step: float):
    x = start
    while x < stop:
        yield x
        x += step


@dataclass
class DecisionConfig:
    """Entity-level decision-layer parameters (pair probabilities -> final match sets)."""

    threshold_t1_grid: tuple = tuple(round(x, 3) for x in _frange(0.15, 0.86, 0.025))
    threshold_t2_grid: tuple = tuple(round(x, 3) for x in _frange(0.3, 0.98, 0.025))
    expected_f_n_mc: int = 1000
    expected_f_p_floor: float = 0.02
    expected_f_max_k: int = 15
    expected_f_p_unseen: float = 0.0


@dataclass
class PipelineConfig:
    data_dir: str = "dataset"
    out_dir: str = "output"
    work_dir: str = "work_dir"
    reports_dir: str = "reports"
    experiments_dir: str = "experiments"
    mode: str = "cv"  # "cv" (local validation only) or "full" (+ predict test, write outputs)
    loco: bool = False  # leave-one-country-out check, used as a France-generalization proxy
    n_jobs: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    seed: int = SEED

    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # Optional GPU-stage inputs (produced by separate scripts); absent -> pipeline runs
    # identically without them, just without the extra signal.
    embed_candidates_train: str | None = None
    embed_candidates_test: str | None = None
    xenc_train_probs: str | None = None
    xenc_test_probs: str | None = None
