# A31 confirmation rule (fixed 26 October 2026, before any exploration result was seen)

1. Exploration: each variation is fitted on emerging 2023 (and the developed market if it wishes) and scored on the
   2024 explore half A (customers split 50/50 at random, seed 2031, across all lines). Configurations scored on A
   are counted.
2. Confirmation: every variation that the verify stage does not refute is scored once on the held-back half B,
   with its frozen configuration, fitted on emerging 2023 exactly as in exploration. No change after seeing B.
3. Metric: pooled Tweedie D2 (p = 1.5) over the three lines against a constant-rate baseline, each line's predictions
   at their deviance-optimal level on the scored rows (level-matched), against C0 the same way.
4. Adopt a variation if its gain over C0 on B has a paired customer-bootstrap (1,000 resamples) lower bound above
   zero at one-sided 0.025 / m (Holm, m = the variations confirmed). A combination is adopted only if each part is
   adopted and the combination's own B gain also clears 0.025 (scored once).
5. Owner decisions (written_premium, D-05; gender, D-36) are confirmed and reported the same way but never adopted
   by this rule.
6. The 2025 file: the adopted variation refitted on 2023 and all of 2024 (A and B), predicting the test rows, each
   line rescaled to the as-is file's line total, so the leaderboard measures the ranking change alone.

# Round 2 (fixed 26 October 2026, before running; from the data-suitability audit, not from any A2024 or B2024 outcome)

Data-harmonisation variants of C0 itself, each with no parameter chosen on 2024:
- R1 MI: fields C0 uses that are missing completely at random in emerging (missing in at least 5% of emerging
  training rows and under 5% of developed, flood zone excepted, as E5 found them) are imputed ten times from the
  same line's emerging training rows of the same region, and C0's predictions averaged (in place of the developed
  MISSING relativity); x_clay_dry is rebuilt from the imputed soil.
- R2 vehicle age: emerging motor vehicle_age minus 2 (floored at 0) before C0 predicts (the emerging distribution is
  the developed one shifted by two years).
- R3 calibration power: per line, log mu = a + b log c0 with b fitted on emerging 2023 (Tweedie p = 1.5, target capped
  at 50k for buildings and 30k for motor with the excess spread back pro rata; contents uncapped).
Each is scored on A2024 (reported) and confirmed once on B2024 under rule 4 above with Holm over m = 3.
- R4 customer_since_year in 2025: buildings C0 predicts the 2025 rows with customer_since_year minus 1, so the value
  means what it meant in the last training year. Not testable on emerging 2024; tested instead on the developed
  backtest (fit 2021-23, score 2024, whose new business has a year above the training maximum, as in 2025).
  Adopted if the shifted prediction's D2 gain on that backtest has a customer-bootstrap lower bound above zero
  (one-sided 0.025).
