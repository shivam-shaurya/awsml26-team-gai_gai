# Business Entity Resolution — Team Pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end to end from
`dataset/train` and `dataset/test`.

## Layout

```
src/
├── config.py               # every tunable parameter (blocking, model, decision layer)
├── data_loader.py           # robust TSV loading, schema validation, ground-truth parsing
├── preprocessing.py         # name/address normalization (multiple representations, not destructive)
├── blocking.py              # candidate generation: multi-view TF-IDF + exact key + optional embeddings
├── features.py               # ~30 pair similarity/overlap/rank features
├── training.py               # LightGBM under GroupKFold (entity-level, leakage-safe)
├── evaluation.py             # F0.5 scoring + the decision layer (exclusivity, threshold/expected-F rules)
├── inference.py              # apply a trained decision spec to test candidates
├── submission.py             # write + self-check the two required output files
├── experiment_tracking.py    # every run appends to experiments/experiment_log.csv
├── pipeline.py                # CLI entry point orchestrating all of the above
├── gpu_embed_blocker.py       # optional GPU stage: dense multilingual-embedding blocker
└── cross_encoder.py           # optional GPU stage: fine-tuned cross-encoder stacking feature
tests/                         # unit tests for the critical preprocessing/submission logic
requirements.txt
```

`pipeline.py` is fully runnable on a CPU-only laptop with no GPU dependencies at all. The GPU
scripts are additive: they write intermediate TSVs that `pipeline.py` merges in *if you pass
their paths on the command line* (`--embed-candidates-*`, `--xenc-*-probs`); leave those flags
out and the pipeline behaves exactly like the CPU-only baseline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # or your usual environment manager
pip install -r requirements.txt
```

On Ayush's L40 box, also install the CUDA build of torch (see the comment in
`requirements.txt`) before running the GPU scripts.

## Run — CPU-only baseline (any laptop)

From this challenge's `student_resource/` directory (so `dataset/` resolves):

```bash
cd student_resource
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --out output --work work_dir --mode cv --loco
```

Read the log: blocking pair recall, singleton rate, cross-country leakage, the ranked decision-rule
scores, and (with `--loco`) a France proxy score. Fix blocking before touching model
hyperparameters — it sets the recall ceiling everything downstream is capped by.
`reports/blocking_baseline.md` and `experiments/experiment_log.csv` are written automatically.

Once you're satisfied with `--mode cv`, generate the real submission:

```bash
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --out output --work work_dir --mode full
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Run — GPU stage (Ayush's L40 only)

```bash
cd student_resource
export PYTHONPATH=code/business_entity_resolution/src

# 1. Dense embedding blocker — widens recall on cross-script / transliteration matches
python code/business_entity_resolution/src/gpu_embed_blocker.py \
    --data dataset --split train --out work_dir/embed_candidates_train.tsv
python code/business_entity_resolution/src/gpu_embed_blocker.py \
    --data dataset --split test  --out work_dir/embed_candidates_test.tsv

# 2. Cross-encoder stacking feature — out-of-fold on train, then scored on test
python code/business_entity_resolution/src/cross_encoder.py \
    --data dataset --mode cv \
    --embed-candidates-train work_dir/embed_candidates_train.tsv \
    --out work_dir/xenc_oof.tsv
python code/business_entity_resolution/src/cross_encoder.py \
    --data dataset --mode full \
    --embed-candidates-train work_dir/embed_candidates_train.tsv \
    --embed-candidates-test  work_dir/embed_candidates_test.tsv \
    --out work_dir/xenc_test.tsv

# 3. Feed both into the baseline pipeline for the final submission
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --out output --work work_dir --mode full \
    --embed-candidates-train work_dir/embed_candidates_train.tsv \
    --embed-candidates-test  work_dir/embed_candidates_test.tsv \
    --xenc-train-probs work_dir/xenc_oof.tsv \
    --xenc-test-probs  work_dir/xenc_test.tsv
```

Read `gpu_embed_blocker.py` and `cross_encoder.py`'s module docstrings for the compute-budget
notes (candidate top-k caps, expected inference time on an L40) before scaling batch sizes up.

## Tests

```bash
cd code/business_entity_resolution
pip install pytest
python -m pytest tests/ -v
```

Covers the logic that's actually critical to get right silently: TSV parsing edge cases
(literal "NA", empty fields, comma-separated files), ground-truth parsing (self-matches,
missing rows), normalization (accent-folding, legal-suffix separation, postal-code extraction),
F0.5 scoring against the problem statement's own worked example, exclusivity enforcement, and
submission format self-checks.

## Validate before every leaderboard upload

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

## Experiment tracking

Every `pipeline.py` run appends one row to `experiments/experiment_log.csv` (config +
measured metrics) and updates `experiments/best_result.json` if it's the best validation F0.5
seen so far. Use `--experiment-notes "..."` to record what changed and why. Log a portal
submission's public-leaderboard score into `experiments/leaderboard_log.csv` against the
`experiment_id` that produced it, so every leaderboard number maps back to an exact local
config — required both by the challenge's own "maintain version history" rule and by basic
research hygiene under a 5-submission/day budget.

## Notes on reproducibility

- All random state is seeded (`SEED = 42` in `config.py`); reruns should match exactly on CPU.
- Diagnostic files (`work_dir/oof_train.tsv`, `work_dir/test_scores.tsv`, the embedding-blocker
  candidate files, the cross-encoder probability files) are intermediate working files, not
  part of the graded output — only `output/matching_results.tsv` and `output/candidate_pairs.tsv`
  are scored/audited.
- Every submitted leaderboard file should correspond to a tagged git commit + an
  `experiment_id` from `experiments/experiment_log.csv`, so the exact run that produced it can
  be reproduced from this folder alone.
