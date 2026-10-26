# A32 protocol: GPU and transfer challengers, Stage 1 (developed-market backtest)

Fixed 26 October 2026, before any Stage 1 score was seen, at the owner's instruction: "Test the leaf refit and
TabPFN-3.5 on the developed backtest first." The design follows the A32 research shortlist (leaf refit, TabPFN-3.5;
the neural correction on C0 was dropped because Week 1's GPU CANN on the developed market gained at most +0.0007, A20).
Any change after the first Stage 1 score is a logged deviation.

## Question

Can either challenger rank risks better than C0 when it has a large source book and a target sample the size of the
emerging market? Stage 1 tests this on the developed market, where the outcome year is large enough to give power.
A method that cannot gain here cannot help on emerging.

## Data

- Developed market, portable (shared) features only, built by the pack's SQL exactly as for C0 (A30).
- Source: policy years 2021-22. Target draws: five random samples of developed 2023 rows, each the size of the
  emerging 2023 book for the line (buildings 12,580, contents 15,834, motor 21,680), drawn by customer with seeds
  2032-2036. Scored rows: a fixed random sample of 100,000 developed 2024 rows per line (seed 2032), the same rows
  for every arm.
- Target: C0's small-claim target for buildings and motor (each claim split at the line's large-claim threshold, as
  C0 does) and the full developed cost for contents (no split, as C0). The large-claim part of every arm is C0's.

## Comparator

C0 refitted on 2021-22 with its unchanged procedures (portable features; buildings XGBoost trees early-stopped on
2023 as C0's tuning does), predicting 2024 with its trend.

## Arms (the decision family, m = 2)

- **A2, leaf refit.** A Tweedie (p = 1.5) LightGBM on the small-claim target, fitted on 2021-22 with log exposure as
  offset (learning rate 0.05, 31 leaves, at least 200 rows a leaf, feature and bagging fraction 0.8, L2 1.0, trees
  early-stopped on all developed 2023 rows, at most 3,000). Its leaves are then refitted on each target draw, tree by
  tree in order: a leaf with draw rows takes d x its value + (1 - d) x the Newton step at the current margin (scaled by
  the learning rate); a leaf with no draw rows keeps its value; the initial score is kept. d is chosen per draw from
  {0.5, 0.8, 0.9, 0.95, 0.98} by five-fold cross-validation within the draw, grouped by customer. Arm prediction:
  the refitted LightGBM's small-claim prediction plus C0's large-claim part.
- **A3, TabPFN-3.5 occurrence stacker.** tabpfn 9.2.0, weights tabpfn-v3.5-20260909, run locally on the Apple GPU,
  default settings (8 estimators). Per draw: a TabPFN classifier for claim occurrence (any claim in the row), with
  the draw as context and features = the shared features, log of C0's annual rate and exposure, giving p(x); a logistic
  regression of occurrence on log C0's annual rate and log exposure, on the same draw, giving p0(x). Arm =
  C0 x p(x) / p0(x). Primary: C0^(1 - w) x arm^w with w = 0.5.

## Controls (reported, not eligible)

The LightGBM before refitting (source only); A3 at w = 0.25, 0.75 and 1.0; A2 at every d.

## Metric and gate

- Primary: pooled Tweedie D2 (p = 1.5) over the three lines on the scored rows against a constant-rate baseline,
  each line at its deviance-optimal level (level-matched, so ranking is judged apart from level), as the gain over
  C0, averaged over the five draws.
- Inference: paired bootstrap by customer, 2,000 resamples, the mean gain over draws recomputed in each.
- An arm passes Stage 1 only if all hold: Holm-adjusted one-sided p below 0.025 (m = 2); pooled gain at least +0.003;
  gain still positive with the ten largest scored claims removed; no loss on the new-business rows or on the 17-24
  age band.
- Also reported: per-line gains, raw (not level-matched) D2, and the spread over draws.

## Engineering checks before any score

MPS against CPU agreement for TabPFN on 2,000 rows (largest relative difference in p below 1e-3); TabPFN and
LightGBM run in their own processes (separate OpenMP runtimes, A20); network activity of the TabPFN process
checked after the weights are cached (Hugging Face offline mode on); the leaf refit checked on a toy case by hand.

## Afterwards

Stage 2 (the emerging transfer check) runs only for an arm that passes, under its own pre-registered rule. Every
result, including failures, goes in the audit log with this file's commit hash.

## Amendments before any score (26 October 2026, after an independent code review; no Stage 1 score had been computed)

1. **Draw customers are kept out of the scored rows.** Target draws are taken only from 2023 customers with no row in
   the 2024 scored sample, so no arm sees a scored row's own earlier outcome (C0 uses no claim history).
2. **Trend in the LightGBM offset.** The offset is log exposure + log rate + C0's small-claim trend (slope x (year -
   centre)), with the rate solved on the trended 2021-22 exposure, so the LightGBM small part sits on C0's 2024 basis
   before C0's large part is added.
3. **Decay cross-validation** uses five seeded random folds by customer and level-matched held-out deviance, as the
   primary metric is level-matched.
4. **A3 acts on the small-claim part only:** C0_small x (p / p0)^w + C0's large part, as the Data section requires
   (the large-claim part of every arm is C0's).
5. **A3's C0 feature** (and p0) uses C0 at the 2023 trend for both the context and the scored rows, so the feature does
   not shift by a year between them.
6. **TabPFN settings and checks:** n_estimators pinned at 8 (asserted after fitting); the arm runs at TabPFN's default
   precision on the Apple GPU; the engineering check compares the Apple GPU with the CPU at float32 on both (tolerance
   1e-3 in p) and also reports the default-precision difference; predictions in chunks of 10,000 rows; if the
   checkpoint's sample limit is below a draw's size, ignore_pretraining_limits is set (within the model's documented
   1,000,000-row range) and recorded. The weights are loaded from an absolute path, offline, with no browser, and their
   sha256 recorded; a missing file stops the run.
7. **Contents features** for both arms are C0's portable GLM inputs (log sum insured, raw elevation and river distance,
   C0's categoricals), not the within-market quantile bands, so an arm that passes can carry over to Stage 2.
8. **Scoring:** Holm with m = 2 fixed, and both families required with five draws each; the ten-largest-claims check
   re-solves each line's level on the remaining rows; raw D2 reported; per-line D2 against each line's own constant rate.
9. **Readings recorded:** LightGBM early stopping uses 100 rounds of patience on its default Tweedie metric; the 17-24
   band pools the home policyholder bands with the motor driver band; the 2024 outcomes are kept in a file only the
   scoring step reads.
10. **Caveats for the result:** C0's structure was adopted under the two-year rule using developed 2024, so 2024 is not
    a clean holdout for C0 (a bias towards C0); `missed_payments_12m` may carry same-period information (A21 bounded it
    at 0.0004 for C0), which a flexible arm could exploit more than C0.
