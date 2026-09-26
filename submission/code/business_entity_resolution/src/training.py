"""LightGBM pair-match classifier, trained under grouped cross-validation.

GroupKFold is keyed by Source-1 entity id specifically so that no entity's candidate pairs
span both the training and validation fold within a run -- an ungrouped split would let the
model see a different candidate for the same S1 entity during "validation", which leaks
entity-level context (e.g. `n_cands_s1`, the rank/gap features) and overstates the score.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import ModelConfig

log = logging.getLogger(__name__)


def train_oof(
    F: pd.DataFrame, y: np.ndarray, groups: np.ndarray, feature_cols: list[str],
    cfg: ModelConfig, X_test: pd.DataFrame | None = None,
):
    """GroupKFold train + out-of-fold prediction, optionally scoring a held-out test set too
    (test predictions are averaged across folds)."""
    oof = np.zeros(len(F))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    importance = np.zeros(len(feature_cols))
    lgb_params = cfg.as_lgb_params()

    gkf = GroupKFold(n_splits=cfg.n_folds)
    for fold, (tr, va) in enumerate(gkf.split(F, y, groups)):
        dtrain = lgb.Dataset(F.iloc[tr][feature_cols], y[tr])
        dvalid = lgb.Dataset(F.iloc[va][feature_cols], y[va])
        model = lgb.train(
            lgb_params, dtrain, cfg.num_boost_round, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        oof[va] = model.predict(F.iloc[va][feature_cols], num_iteration=model.best_iteration)
        importance += model.feature_importance("gain")
        if X_test is not None:
            test_pred += model.predict(X_test[feature_cols], num_iteration=model.best_iteration) / cfg.n_folds
        log.info("  fold %d: best_iter=%d", fold, model.best_iteration)

    importance_s = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    return oof, test_pred, importance_s
