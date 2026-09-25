# Business Entity Resolution — Team Pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end to end from
`dataset/train` and `dataset/test`.

## Layout

```
src/
├── er_pipeline.py        # baseline: normalize -> block -> pair features -> LightGBM -> decide
├── gpu_embed_blocker.py  # optional GPU stage: dense multilingual-embedding blocker (Ayush's L40)
└── cross_encoder.py      # optional GPU stage: fine-tuned cross-encoder stacking feature
requirements.txt
```

`er_pipeline.py` is fully runnable on a CPU-only laptop with no GPU dependencies at all — this is
the shared baseline every teammate can run and iterate on. `gpu_embed_blocker.py` and
`cross_encoder.py` are additive: they write intermediate TSVs that `er_pipeline.py` merges in
*if you pass their paths on the command line*; leave those flags out and the pipeline behaves
exactly like the CPU-only baseline. This lets the team split work without anyone blocking on
GPU access.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # or your usual environment manager
pip install -r requirements.txt
```

On Ayush's L40 box, also install the CUDA build of torch (see the comment in
`requirements.txt`) before running the GPU scripts.

## Run — CPU-only baseline (any laptop)

From this challenge's `student_resource/` directory (so `dataset/` resolves), with
`src/` on your path:

```bash
cd student_resource
python code/business_entity_resolution/src/er_pipeline.py \
    --data dataset --out output --work work_dir --mode cv --loco
```

Read the log: blocking pair recall, singleton rate, cross-country leakage, the ranked decision-rule
scores, and (with `--loco`) a France proxy score. Fix blocking before touching model hyperparameters
— it sets the recall ceiling everything downstream is capped by.

Once you're satisfied with `--mode cv`, generate the real submission:

```bash
python code/business_entity_resolution/src/er_pipeline.py \
    --data dataset --out output --work work_dir --mode full
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Run — GPU stage (Ayush's L40 only)

```bash
cd student_resource
export PYTHONPATH=code/business_entity_resolution/src   # so cross_encoder.py can `import er_pipeline`

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
python code/business_entity_resolution/src/er_pipeline.py \
    --data dataset --out output --work work_dir --mode full \
    --embed-candidates-train work_dir/embed_candidates_train.tsv \
    --embed-candidates-test  work_dir/embed_candidates_test.tsv \
    --xenc-train-probs work_dir/xenc_oof.tsv \
    --xenc-test-probs  work_dir/xenc_test.tsv
```

Read `gpu_embed_blocker.py` and `cross_encoder.py`'s module docstrings for the compute-budget
notes (candidate top-k caps, expected inference time on an L40) before scaling batch sizes up —
scoring the full test set through a transformer without capping candidates per entity is not
tractable in the challenge window.

## Validate before every leaderboard upload

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Notes on reproducibility

- All random state is seeded (`SEED = 42` in `er_pipeline.py`, `--seed` in `cross_encoder.py`);
  reruns should match to within GPU non-determinism on the transformer stage.
- Diagnostic files (`work_dir/oof_train.tsv`, `work_dir/test_scores.tsv`, the two embedding-blocker
  candidate files, the two cross-encoder probability files) are intermediate working files, not
  part of the graded output — only `output/matching_results.tsv` and `output/candidate_pairs.tsv`
  are scored/audited.
- Every submitted leaderboard file should correspond to a tagged git commit (see the team
  methodology document, Section 10) so the exact run that produced it can be reproduced from this
  folder alone.
