# Amazon ML Challenge 2026 — Business Entity Resolution

Team repo for the challenge (25–27 Sep 2026). Link business records describing the same
real-world business across three noisy sources, using only names and addresses. Get the full
problem statement, guidelines, and video walkthrough directly from the official challenge
portal — they aren't duplicated in this repo (see below).

## What's in this repo

| Path | What it is |
|---|---|
| `submission/` | **The actual deliverable.** `code/business_entity_resolution/` (modular pipeline: `src/` + `tests/`), `reports/` (data audit, blocking recall), `experiments/` (experiment/leaderboard tracking), `output/` (where `matching_results.tsv` + `candidate_pairs.tsv` land), `Documentation_template.md` (methodology write-up, in progress). This is what goes in the final zip. |
| `Business_Entity_Resolution_Solution.docx` | Full strategy document — problem breakdown, blocking design, decision-layer math, team workflow/timeline. Read this first for the *why* behind the pipeline's design. |
| `Ml-proj1.ipynb` | Historical notebook — **do not use as-is**. Predates the scalability fix now in `submission/code/.../src/blocking.py` (the naive version's blocking stage does not finish in reasonable time at full dataset scale). Kept for reference only. |

**Not in this repo:** the actual dataset, and the official problem statement / guidelines / video
transcript PDFs. All of that is the organizers' own material — download it yourself from the
challenge portal. Place the dataset so `dataset/train/` and `dataset/test/` are reachable from
wherever you run the pipeline (see `submission/code/business_entity_resolution/README.md` for
exact paths).

## Quick start (any team member)

```bash
git clone <this repo>
cd amazon-ml-challenge-2026   # or whatever you named it
pip install -r submission/code/business_entity_resolution/requirements.txt
```

Download the dataset from the challenge portal and place it next to `submission/` (or symlink
it) so the layout looks like:

```
dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

Then follow `submission/code/business_entity_resolution/README.md` for exact run commands
(CPU-only baseline, and the optional GPU stage for whoever has GPU access).

## Validate before anyone uploads anything

The official validator (from the challenge's own `student_resource/utils/` — grab it from the
portal download, it's stdlib-only) checks your output format before you spend one of the 5
daily submissions on it:

```bash
python3 validate_submission.py \
    --matching submission/output/matching_results.tsv \
    --candidate submission/output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Must print `PASS`. If it doesn't, fix the listed issues before uploading — a format rejection
still costs you a submission slot.

## Team workflow

- **Trust local out-of-fold cross-validation over the public leaderboard.** Public LB is a
  subset of test data and you only get 5 uploads/day (15 total) — not enough to tune against.
  `pipeline.py --mode cv --loco` is the real signal.
- Every submission should follow a measured OOF improvement, not be sent speculatively.
- See Section 9 of `Business_Entity_Resolution_Solution.docx` for role split and the 3-day
  timeline.
