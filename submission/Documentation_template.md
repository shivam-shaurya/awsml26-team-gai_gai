# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [fill in]
**Team Members:** Shivam Shaurya, Ayush [surname], [teammate 3], [teammate 4]
**Submission Date:** [fill in, on/before 27 Sep 2026 23:59 IST]

---

## 1. Executive Summary

We link business records across three noisy sources using only names and addresses, through a
five-stage pipeline — normalize, block, score candidate pairs with a gradient-boosted model,
optionally re-rank with a fine-tuned multilingual cross-encoder, then convert pair probabilities
into entity-level predictions via a decision layer built directly around the macro-F0.5 metric
(exclusivity constraint + expected-F0.5 set selection, rather than a single global probability
threshold). Our main technical contribution beyond a standard blocking+classifier baseline is
using a GPU (L40, 96GB) for two additive stages most teams' time budget won't reach: a dense
multilingual-embedding blocker that recovers cross-script/transliteration matches char-n-gram
TF-IDF cannot see, and a fine-tuned mDeBERTa-v3 cross-encoder whose probability is stacked into the
LightGBM model as one more feature.

---

## 2. Methodology

### 2.1 Problem Analysis

- Roughly a third of Source-1 entities are expected to be singletons (no true match) — under
  macro F0.5, correctly predicting empty for these is worth exactly as much as correctly finding a
  real match, and a false merge on one costs a full point. [Replace with your team's measured
  singleton rate from `diagnostics()`'s log output once you've run the pipeline on the real data.]
- Name noise: legal-suffix variants (Corp/Corporation, Pvt/Private, Ltd/Limited), DBA/trade names,
  punctuation (& vs "and"), word-order transpositions, typos, and — critically — cross-script
  transliteration (Source 2 contains business names written in Devanagari against Latin-script
  names elsewhere for the same India-country records).
- Address noise: abbreviations (Rd/Road, St/Street), missing components (no PIN/postal code, no
  state), landmark references ("Near SBI ATM"), municipal numbering and component reordering.
- Training data covers US and India only; the test set adds France, unseen during training — the
  pipeline is built to generalize to it by construction (Section 7 below), not by tuning on it.
- [Add your own EDA findings here once run: match-set size distribution, cross-country gold-pair
  count (should be ~0, validating per-country blocking), how many S2/S3 records match more than
  one S1 entity (should be ~0, validating the exclusivity constraint).]

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pair classifier + optional GPU cross-encoder
stacking feature + F0.5-aware decision layer (hybrid, not end-to-end).

**Core Innovation:** Two things most teams under-invest in given a 72-hour window: (1) a
decision layer that treats the metric's arithmetic as a first-class design constraint rather than
picking one probability threshold (Section 6), and (2) spending the team's one GPU on the specific
gap the CPU pipeline has — cross-script matching — instead of re-deriving what TF-IDF/LightGBM
already do well.

---

## 3. Candidate Generation (Blocking)

Blocking sets the ceiling on achievable recall — a true match never proposed as a candidate cannot
be found downstream — so it is validated against training ground truth before any modeling time is
spent, and `candidate_pairs.tsv` is exactly the set the final model scores over (not an earlier,
looser pass), matching what the grading process audits.

**Blockers used (union of all, per country, per target source, both directions S1→candidate and
candidate→S1):**

| Blocker | Signal | Catches |
|---|---|---|
| Name TF-IDF (char 3-grams) | S1↔target top-k cosine on normalized core business name | Typos, minor rewording |
| Name+address TF-IDF | Same, on name and address concatenated | Cases where the name alone is too generic |
| Exact key | (postal code, first name token) | Cheap high-precision anchor cross-check |
| **Dense multilingual embeddings (GPU)** | intfloat/multilingual-e5-base, top-k cosine via GPU matmul | **Cross-script / transliteration matches TF-IDF cannot see at all** (e.g. Devanagari S2 name vs Latin S1 name), heavy word-order/legal-form paraphrase |

Both directions are run for every fuzzy blocker: a candidate that is any S1's best match is kept
even if it doesn't make that S1's own top-k, which matters when one S1 entity is generic-sounding
and would otherwise crowd out a genuine partner from its own ranked list.

- **Candidate pairs generated:** [fill in from the pipeline log — pairs per S1 entity after
  capping, both before and after the embedding blocker is merged in]
- **How true matches are not lost:** the pipeline reports blocking pair recall against ground
  truth (union and each blocker in isolation) before any model is trained (`diagnostics()`), so a
  recall gap is caught and fixed by adding/widening a blocker, not discovered after the fact.
  Target: recall ≥ ~99.5% at a bounded average candidate count; past that point extra low-quality
  candidates mostly add false-merge risk under a precision-weighted metric.

---

## 4. Matching Model

**Features used (~60 per candidate pair, computed in `featurize()`):**

- **Name similarity:** Levenshtein / Jaro-Winkler / token-set / token-sort ratios on normalized
  and core names
- **Rarity-weighted overlap:** IDF-weighted token overlap per country; max IDF among unmatched
  tokens (a rare token present in one name but not the other is the strongest "different business"
  signal)
- **Structural:** acronym match, legal-form conflict, digit/number-token conflict in the name
- **Address:** fuzzy ratios on normalized address; postal-code exact/prefix match (missing vs.
  mismatch kept distinct); house-number set overlap; landmark flags
- **Rank/context:** this pair's similarity rank and score gap vs. the same S1's other candidates
  and vs. the same candidate's other S1 entities; mutual-best-match flag; how common the S1's core
  name is within its country — empirically the most informative group, since a pair's absolute
  similarity matters less than whether it's clearly the best option on both sides of the match
- **Cross-encoder probability (GPU, optional):** `p_xenc` — fine-tuned mDeBERTa-v3-base's match
  probability on the full normalized name+address text pair, stacked in as one more feature (plus
  `p_xenc_present`, since it's only computed for the top-k candidates per entity; see Section 5.2)

**Model type:** LightGBM binary classifier, `GroupKFold` cross-validation keyed by Source-1 entity
id (no entity's candidates span both a train and validation fold, which would leak information
about that entity's other candidates and overstate the validation score).

**Threshold selection method:** not a single global threshold — see Section 6. The pipeline
grid-searches a two-threshold rule and computes an expected-F0.5 set-selection rule on
out-of-fold predictions, per run, and picks whichever scores higher.

### 5.1 GPU Cross-Encoder (Stage 2)

`microsoft/mdeberta-v3-base` (~280M params, MIT license — within the ≤8B / MIT-Apache-2.0
constraint), fine-tuned as a binary sequence-pair classifier directly on
`"name: ... | address: ..."` text for the S1 entity and for the candidate, using training ground
truth as labels and the blocking candidates as (mostly hard) negatives. To keep training/inference
tractable at this dataset's scale (millions of S1 entities), only the top-k candidates per entity
(ranked by the existing TF-IDF/embedding score) are sent through the transformer — disambiguating
among a handful of already-plausible candidates is exactly the job a cross-encoder is suited for;
the cheap CPU features already handle rejecting far-fetched candidates. Its out-of-fold probability
is stacked into the LightGBM model as a feature rather than used standalone, so the GBM keeps
combining the rank/context/exclusivity signals it's already good at.

[Fill in once run: number of candidates sent through the cross-encoder, fine-tuning time, held-out
AUC/log-loss of `p_xenc` alone vs. its marginal LightGBM feature-importance gain once stacked.]

---

## 5. Decision Layer: From Pair Probabilities to Entity Predictions

This is the stage most teams are expected to under-invest in — worth as much attention as the
model itself, because of how the metric's arithmetic behaves:

- **Singletons are all-or-nothing:** an entity with no true match scores 1.0 for an empty
  prediction and 0.0 for any prediction at all.
- **The break-even probability is not the same for the first match as for later ones:** a lone
  uncertain candidate is worth predicting once its probability exceeds roughly 0.5, not the
  0.8–0.9 "precision-heavy" intuition suggests — but once one match is confirmed, adding a second
  candidate only pays off above roughly 0.73, because a wrong addition drags a perfect score down
  further than a missed one would. A single global pair-probability threshold is therefore
  mathematically the wrong tool.

**Exclusivity constraint:** Source 1 is deduplicated, so a genuine S2/S3 record should belong to
at most one S1 entity (confirmed against training ground truth in diagnostics). Enforcing this —
keeping each candidate only under its single best-scoring S1 entity — removes a specific, common
failure mode: two different businesses sharing an address or a similar name both drawing the same
candidate.

**Two competing decision rules**, both implemented, selected empirically per run on held-out
out-of-fold predictions:

1. **Two-threshold rule:** a higher bar `t1` to accept an entity's best candidate at all, a
   second, higher bar `t2` for every additional candidate — reflecting the asymmetric break-evens
   above. Both grid-searched on out-of-fold predictions.
2. **Expected-F0.5 set selection:** for each entity, treat candidates' probabilities as
   independent match probabilities and choose the top-k (including k=0, i.e. singleton) that
   maximizes expected per-entity F0.5. Threshold-free, which makes it the safer default for a
   country with no training data (Section 7).

---

## 6. Generalizing to an Unseen Country (France)

France appears only in the test set, so any France-specific tuning is untestable until
submissions are spent on it. The pipeline generalizes by construction:

- No country is hard-coded anywhere; blocking, IDF weighting and features are computed per
  country from whatever labels are present in that split, so a new label needs no code change.
- Feature choices favor signals that are relative and language-light (rank/gap-vs-competitors,
  character n-grams, IDF-weighted overlap) over anything assuming English/Indian naming
  conventions.
- A leave-one-country-out check (train on the others, evaluate on the held-out one, `--loco` flag)
  is used throughout as a proxy for France; any change that helps in-domain but hurts the
  held-out country is treated as overfitting to the visible countries.
- The threshold-free expected-F0.5 rule is preferred specifically for France, since its
  thresholds were never tuned on French data.
- The multilingual embedding model and cross-encoder were both chosen for their French-language
  coverage (e5/mDeBERTa are trained across 100+ languages including French), not just
  English/Hindi.

[Fill in once run: `--loco` held-out score for France's stand-in vs. in-domain GroupKFold score.]

---

## 7. Results & Error Analysis

- **F_0.5 Score (macro), best validated decision rule:** [fill in]
- **Blocking pair recall / entities with all matches recalled:** [fill in from diagnostics log]
- **Common false positives (wrong merges):** [fill in — e.g. generic chain names sharing a city
  but different address, resolved by rank/context + address IDF features]
- **Common false negatives (missed matches):** [fill in — e.g. heavy transliteration cases the
  embedding blocker/cross-encoder were added specifically to address; report before/after]
- **Public leaderboard score, by submission:** [maintain a running table here — see Section 8 of
  the team playbook for the version-history requirement]

---

## 8. Conclusion

[Fill in: 2-3 sentences summarizing the final approach, the measured gain from the GPU stage vs.
the CPU-only baseline, and one lesson learned worth carrying into future entity-resolution work.]

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`:

- `src/er_pipeline.py` — normalize → block → featurize → LightGBM (GroupKFold OOF) → exclusivity
  → expected-F0.5 decision → `matching_results.tsv` + `candidate_pairs.tsv`. Fully CPU-runnable;
  the GPU stage is optional and additive (silently absent if not run).
- `src/gpu_embed_blocker.py` — dense multilingual-embedding blocker (GPU, Ayush's L40); widens
  blocking recall via `--embed-candidates-train/-test`.
- `src/cross_encoder.py` — fine-tuned cross-encoder stacking feature (GPU, Ayush's L40); adds the
  `p_xenc` LightGBM feature via `--xenc-train-probs/-test-probs`.
- `README.md` — exact, copy-pasteable run commands for both the CPU-only path and the GPU path.
- `requirements.txt` — pinned dependencies for both.

Entry point for the graded outputs (see `README.md` for the GPU-augmented version):

```bash
python code/business_entity_resolution/src/er_pipeline.py \
    --data dataset --out output --mode full
```

### B. Additional Results

[Attach any charts/tables: feature importance, precision/recall by country, candidate-count vs.
recall curve from blocking tuning, cross-encoder ablation.]

---

**Note:** Sections above map directly onto the required "methodology," "candidate
generation/blocking strategy," "model architecture and feature engineering" — fill in every
`[fill in]` placeholder with real numbers from your own run before submitting; a template with
unfilled brackets will read as incomplete to reviewers.
