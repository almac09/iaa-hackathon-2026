# Audit log — home contents analysis

**Project:** IAA AI-Assisted Hackathon 2026, Part 1 (Week 1)
**Owner:** Gerard Murphy (contents line)
**AI assistant:** Claude Code (Claude Opus 5.5), run locally on the owner's Mac
**Log started:** 2026-10-20 13:33 BST. Entries A01–A08 were reconstructed from the session transcript of the same day; times before 13:33 are approximate.

## Ground rules for this log

- **Source data is never modified.** Every fix is applied only in derived tables, and the original column is kept beside the fixed one (`exposure_raw` and `exposure`, `sum_insured_raw` and `sum_insured`), so before and after can always be compared row by row.
- Every entry records what was done, why, the inputs, the method, the before, the after, how the result was verified, and its status.
- **Status values:** `analysis` (read-only, nothing changed) · `applied-derived` (applied in a derived table only, awaiting human sign-off) · `proposed` (not applied) · `signed-off` (a human has approved it, recorded in DECISIONS.md).
- Inputs are pinned by SHA-256 in `MANIFEST.sha256`. Evidence files are in `evidence/`, prefixed with the entry ID.
- Nothing has been written to the Databricks workspace or to GitHub. All work is in a local scratch copy.

---

### A01 — Connect to Databricks and locate the data
- **When / who:** 2026-10-20 ~13:20, Claude, after the owner logged in through the Databricks CLI (OAuth, profile DEFAULT).
- **What:** Listed the workspace and found the Git folder `/Workspace/Users/<owner>/iaa-hackathon-2026` (user name redacted for publication) (remote `github.com/almac09/iaa-hackathon-2026`, branch `main`).
- **Result:** 11 parquet files in `data/` (about 210 MB), competition docs in `docs/`, notebook stubs, and an existing `submissions/home_insurance_submission_v001` and `workings/home_insurance_model` (not reviewed).
- **Status:** analysis.

### A02 — Read the rules before copying the data
- **What:** Exported and read `README.md`, `docs/kaggle_*.md`, `docs/DATA_SETUP.md` and `src/utils.py`.
- **Why:** To confirm that copying the competition data for AI processing is allowed.
- **Result:** Rules §5a explicitly allow processing with AI tools. §5b forbids redistribution. **Risk raised:** `.gitignore` does not exclude `data/`. If the parquet files have been pushed to a public GitHub repo, that would breach §5b. Not yet checked: the request to GitHub was blocked, so the owner needs to check this.
- **Status:** analysis. Open action for the owner.

### A03 — Copy the data to local scratch and profile all tables
- **What:** Exported the 11 parquet files with `databricks workspace export`, hashed them (MANIFEST.sha256), and profiled row counts, keys and claims with DuckDB.
- **Result:** Row IDs and claim IDs are unique. Claims are valued at 31 December 2024. Contents is 721,617 training rows, 187,498 test rows and 24,102 claims.
- **Status:** analysis.

### A04 — Exposure recorded in the wrong units
- **Finding:** `exposure` should be in years, but some rows hold days (values 2–365) and some use a 360-day year.
- **Method:** Recompute `exposure = (period_end − period_start + 1) / 365` for every row. The raw value is kept as `exposure_raw`.
- **Before → after (contents):**

  | | Rows > 1 | Total exposure | Max |
  |---|---|---|---|
  | Train before | 3,088 | 810,723 | 365.0 |
  | Train after | 0 | 590,859 | 1.0 |
  | Test before / after | 0 / 0 | 150,124 / 150,124 | 1.0 / 1.0 |

  The raw column overstates training exposure by **37%**. The test file is unaffected.
- **Verification:** Every row over 1 was classified: days (exactly matches the day count) or a 360-day basis (365/360 = 1.0139). Changed rows are listed in `evidence/A04_exposure_changed_rows.csv`.
- **Evidence:** `evidence/A04_exposure_before_after.csv`, `evidence/A04_exposure_changed_rows.csv`.
- **Status:** applied-derived. Decision D-01.

### A05 — Sum insured recorded 1,000 times too large
- **Finding:** 1,402 training rows and 360 test rows have sum insured between 8.2 million and 152.6 million. The median is 39,000.
- **Method:** If `sum_insured >= 1,000,000`, divide by 1,000. The raw value is kept as `sum_insured_raw`.
- **Before → after:** train mean 124,801 → 42,881, max 152.6m → 346,700. Test mean 132,536 → 45,003, max 146.8m → 298,800.
- **Verification:** For the same policy in another year, the ratio is about 1,000 (for example C86549631: 21.4m in 2021 and 22,300 in 2023). Written premium on these rows averages 135, the same as the rest of the book, so the insurer did not price them as high-value risks. The 107 rows between 200,000 and 1m are left unchanged: their premium averages 478, which is consistent with genuinely high sums insured.
- **Evidence:** `evidence/A05_sum_insured_before_after.csv`, `evidence/A05_sum_insured_changed_rows.csv`.
- **Status:** applied-derived. Decision D-02.

### A06 — Build the contents claim target and join the tables
- **Method:** `scripts/build_contents.sql`. Claims are matched to a policy row on `policy_id` with `loss_date` inside the row's period. Repudiated claims are excluded. Target = `incurred_to_date − recoveries` (incurred is already net of excess; contents recoveries are all zero). Customers, addresses and weather are joined on the matching year.
- **Before → after:** 24,102 contents claims → 300 repudiated removed → 49 outside any policy period dropped → **23,753 claims, total 65,966,579 CU**. Customer, address and weather joins are 100% complete for both training and test.
- **Development check:** Contents is short-tailed. Estimated unreported 2024 claims are about 2.9% of the count (theft 3.5%, accidental damage 2.9%, escape of water 2.4%, flood 2.4%), based on report lags from 2021–22. Open 2024 claims have average estimates similar to closed ones, so case reserves look adequate.
- **Status:** applied-derived. Decision D-03 covers the development uplift.

### A07 — One-way analysis of 56 rating factors
- **Method:** `scripts/oneway_contents.py`. For each factor: exposure, frequency, severity, loss cost and relativity, plus the relativity in 2021–22 and in 2023–24 as a stability check. Each factor is scored **out of time**: level rates are fitted on 2021–23, used to predict 2024 rows as rate × exposure, and scored with Tweedie D² (power 1.5). The score is the gain over an exposure-only model (D² = 0.031).
- **Result:** Excluding contacts (A08), the strongest factors are flood zone (+0.024), written premium (+0.019), elevation (+0.018), sum insured (+0.016), distance to river (+0.013), tenure, region, marketing segment, prior claims, policyholder age, missed payments, voluntary excess and credit band. Relativities are stable between the two halves of the period. About 20 factors show no out-of-time signal, including branch, previous insurer, campaign, occupation and channel.
- **Evidence:** `evidence/A07_oneway_contents_output.txt`.
- **Status:** analysis.

### A08 — Customer service contacts leak the target
- **Finding:** `customer_service_contacts` would add +0.577 to D² on its own, which is far too good to be true. Rows with 0 contacts are 61% of exposure but have only 54 claims in four years.
- **Verification:** Training rows with a claim average 3.05 contacts, against 0.41 for rows without one (about 2.6 contacts per claim). The test file averages 0.40, which matches training rows *without* claims, and its high-contact share is much smaller (4+ contacts: 0.08% in test against 1.21% in training). Conclusion: training contacts include contacts caused by claims, while test contacts do not. A model using this field would validate extremely well and then mis-price 2025.
- **Evidence:** `evidence/A08_contacts_leakage.csv`.
- **Status:** proposed: exclude the field. Decision D-04.

### A09 — Should claims be split into size bands or perils? (owner's question)
- **When / who:** 2026-10-20 ~13:45, Claude, at the owner's request ("separate high, mid and low claims into separate distributions").
- **Severity profile (23,753 contents claims, net):** pooled CV 1.68; within each peril the CV is about 1.1 (accidental damage 1.09, escape of water 1.13, flood 1.10, theft 1.14) and the log-skew is negative, which fits a gamma shape with a lighter tail than a lognormal. The largest claim is 111,900. The top 1% of claims make up 11.8% of cost, the top 5% make up 32.7%.
- **Size bands mostly reflect peril:** 62% of claims under 1k are accidental damage; 59% of claims of 15k and over are flood.
- **Data quality, minor:** 3 claims exceed the cleaned sum insured (maximum 1.24×). Logged; no action yet.
- **Test:** `scripts/split_test_contents.py` (v2). The same 19 rating factors (85 parameters) in every model, with contacts and written premium excluded per D-04 and D-05. Fitted on 2021–23 (534,126 rows) and scored on 2024 (187,491 rows) with Tweedie D², power 1.5.
- **Run 1 failed (kept for audit):** the statsmodels IRLS solver diverged on sparse per-peril levels and crashed in model D before any results were printed. The script is kept as `evidence/A09_split_test_contents_v1_failed.py`. Run 2 uses scikit-learn GLMs with a light L2 penalty (alpha 1e-4). Exposure enters exactly as an offset would: target cost/exposure with weight exposure^(2−p).
- **Result (run 2):**

  | Model | D² (2024) | Predicted/actual |
  |---|---|---|
  | Exposure only | 0.0310 | 0.941 |
  | **A single Tweedie** | **0.0933** | 0.975 |
  | B per-peril Tweedie | 0.0920 | 0.983 |
  | C per-size-band Tweedie | 0.0872 | 1.036 |
  | D per-peril frequency × severity | 0.0899 | 0.982 |

  Splitting did **not** improve the expected-cost score. Size bands were the worst option. The gaps between A, B and D (about 0.001–0.003) have not been tested for significance (a bootstrap is still to do). The predicted/actual figures exclude the roughly 3% IBNR uplift for 2024 (D-03).
- **But factor effects do differ by claim size** (relativities fitted per band): flood zone 5 gives 1.25× for claims under 1k and 20× for claims of 15k and over; log sum insured gives 0.93 and 3.71; accidental damage cover gives 2.90 and 1.06; voluntary excess of 1,000 gives 0.50 and 1.62. The single model averages these. More parameters spread across thin components cost more in noise than they gain. Interactions in a single model (GLM interaction terms or a GBM) are the likely way to capture this.
- **Status:** analysis.

### A10 — Does a policy's own claim history predict next year? (owner's Markov chain question)
- **Method:** `scripts/persistence_contents.sql`. Policy-years with exposure of at least 0.5. Year-to-year transition matrix on claim count, and next-year frequency by the previous two years' claim flags, with actual/expected (A/E) measured within flood zone.
- **Result:** claimed last year → next-year frequency 60–63 per 1,000, against 37–38 after a claim-free year. Large claim (5k+) last year → next-year loss cost 2.5× within flood zone.
- **Not first-order Markov:** a claim two years ago but none last year gives A/E 1.67, compared with 1.50 for a claim last year only and 2.16 for both. Older history carries as much information as last year's, which points to a stable hidden risk level per policy rather than state dependence. That suggests credibility-weighted experience rating (for example a Poisson–gamma posterior) rather than a one-step Markov chain.
- **Limitation:** only flood zone is controlled for so far. The next step is to re-measure A/E against the full GLM, because some of the persistence will be explained by observed factors.
- **Evidence:** `evidence/A10_transition_matrix.csv`, `evidence/A10_two_year_history.csv`.
- **Status:** analysis.

### A11 — Replication pack in SQL, Python and R, with automated reconciliation
- **When / who:** 2026-10-20 ~14:00–17:10 BST, Claude, at the owner's request ("SQL, Python and R … cleanly replicated with a full audit trail"). The owner chose a local folder for now, shared SQL with independent R modelling, and installing the R packages.
- **What:** `~/iaa-hackathon-2026-contents`. SQL (`sql/01`–`08`) is the single source for data preparation. `python/run_all.py` and `R/run_all.R` each run that SQL through DuckDB 1.5.6, then build the one-way analysis and the four GLM structures independently. `reconcile/reconcile.py` compares the two runs and checks the results against this session's exploratory numbers. Every parameter and decision switch is in `config.json`. Every run appends hashes, versions and results to `run_log.jsonl`. Unit tests: 8 in pytest (including hand-built cases for every data fix) and 7 expectations in testthat.
- **How the reconciliation got to green** (each run is preserved in `run_log.jsonl`; run 1 evidence is in `evidence/history/`):
  1. **Run 1** (R using glmnet, threshold 1e-12, 58 minutes): 83 of 84 checks passed. The flood model's coefficient for elevation band q07 differed (py −12.9716, r −12.9239). **Diagnosis:** a quasi-separated level (almost no flood claims at high elevation), held finite only by the ridge penalty. Python's L-BFGS and exact Newton solvers reached the same objective, so Python was at the optimum and glmnet stopped short. Predictions were unaffected (relativity about 2e-6).
  2. **Run 2** (threshold loosened to 1e-10 to save time): 5 failures, on marketing-segment and credit coefficients, which overlap heavily with age band. Loosening made R stop earlier. **The tolerance was not widened.** A diagnostic with tighter glmnet settings ran for over 40 minutes on a single fit and was stopped.
  3. **Resolution:** R's GLMs were rewritten as a penalised IRLS (Fisher scoring) in base R that minimises exactly the same objective. Python moved to scikit-learn's Newton-Cholesky solver. The reconciliation now also checks each model component's predicted total, and lists quasi-separated coefficients (|coef| > 8, intercepts excluded) as WARN rather than FAIL. A bug in the first version of that rule, which flagged Gamma intercepts, was fixed.
  4. **Final run:** **100 checks, 0 failed, 1 warning** (flood q07: py −12.9715, r −12.9715). All 34 SQL evidence files are byte-identical. Coefficients agree to 5 decimal places in every component. Model D² agrees to 4 decimal places. R runs in 88 seconds, Python in 39.
- **Known differences from the exploratory numbers** (all within ±0.002 D², checked automatically): missing flood zone kept as a level in A10 (A/E 1.67/1.50/2.16 becomes 1.669/1.517/2.098); solver tolerance tightened (model D 0.0899 becomes 0.0904); band definitions unified in SQL.
- **Status:** analysis. Pack held locally, not yet committed.

### A12 — Correction: written premium is annual, so loss ratios must use earned premium
- **When / who:** 2026-10-20 ~16:30 BST. Triggered by the owner ("expected claims should account for exposure and earned premium"). Investigated and corrected by Claude.
- **Finding:** `written_premium` is the annual premium for the row, not the amount for its period. It averages 135–137 whether the row covers a full year or a few weeks, its correlation with time on cover is 0.00, and policies split by a mid-term change show about 282 across their two rows (`evidence/*/ev_02_premium_basis.csv`). **Earned premium = written premium × exposure.**
- **Error corrected:** the loss ratios quoted earlier in the session, and on the team status page, divided claims by summed written premium. That understated them and produced a false downward trend.

  | Line | Before (on written premium) | After (on earned premium) |
  |---|---|---|
  | Contents 2021 / 22 / 23 / 24 | 0.749 / 0.649 / 0.673 / 0.638 | **0.844 / 0.812 / 0.842 / 0.798** |
  | Buildings 2021–24 | 1.05–1.17 | **1.29–1.31** |
  | Motor 2021–23 (2024) | 0.81–0.91 (0.61) | **1.04–1.07 (0.78, undeveloped)** |

  The one-way analysis, GLMs and D² scores were not affected: they use exposure, and written premium was only ever used as a banded factor (it is an annual rate, so that use stays valid).
- **New evidence (D-07):** persistence measured against expected claims based on earned premium, which controls for everything in the current price. A claim in both prior years gives next-year cost A/E **2.59**; last year only **1.84**; two years ago only **1.66**; neither **0.94**. A large claim (5k+) last year gives next-year cost A/E **3.35**. The current pricing does not reflect own-claims history, so experience rating would add information the rates don't already contain.
- **Changes:** `sql/02` (premium-basis evidence), `sql/03` (`earned_premium` column), `sql/04` (loss ratios on earned premium for all lines, with the written-premium figures kept as `_superseded` columns), `sql/06` (A/E against earned premium). Re-run and reconciled in A11's final run.
- **Status:** applied-derived. D-07 decided in session; formal sign-off pending.

### A13 — XGBoost on contents, against GLM A
- **When / who:** 2026-10-21 ~19:00–19:35 BST, Claude, at the owner's request ("I need XGBoost to be run on the contents dataset"). The owner approved the plan: both feature sets, with no submission file yet.
- **Why:** A09 found that factor effects differ by claim size and suggested a GBM as a way to capture interactions that a single GLM averages out.
- **Method:** `python/run_xgb.py` and `python/xgb_models.py`, using xgboost 3.2.0 with settings in `config.json` → `xgboost`. Tweedie objective with power 1.5. Exposure and the fit-period base rate enter as a log offset (`base_margin`). Checked: with `base_margin` set, XGBoost does not add `base_score`. Same split as A09: fit on 2021–23, score on 2024. Tuning never sees 2024: each grid point is fitted on 2021–22 and early-stopped on 2023 (Tweedie negative log-likelihood, which is the deviance up to a constant), then refitted on 2021–23 with the same number of trees. Two feature sets:
  - *banded*: GLM A's 19 factors plus `z_log_si`.
  - *raw*: 55 permitted columns at full resolution (D-08).

  A paired bootstrap (1,000 row resamples of 2024) compares each model with GLM A. GLM A is refitted inside the runner; it matches the pack's D² to 2e-7.
- **Run 1 (kept in `run_log.jsonl`):** grid of depth 2/4/6 × `min_child_weight` 10/200. The best point was depth 2, on the edge of the grid, and the two weights gave identical results. At the base rate the Tweedie hessian is about 5 per row (median), so both weights allowed leaves of only a handful of rows. Run 1 was set aside as under-specified, not reported as a finding.
- **Run 2 (reported):** grid of depth 1/2/3 × `min_child_weight` 10/2,000/20,000. No grid point hit the 3,000-tree cap. Two runs gave byte-identical evidence.

  | Model (2024) | D² | Predicted/actual | D² level-matched (diagnostic) |
  |---|---|---|---|
  | Exposure only | 0.0310 | 0.941 | 0.0317 |
  | **GLM A single Tweedie** | **0.0932** | 0.975 | **0.0935** |
  | XGBoost banded (depth 2, weight 20,000, 234 trees) | 0.0919 | 0.941 | 0.0924 |
  | XGBoost raw (depth 1, weight 20,000, 414 trees) | 0.0913 | 0.942 | 0.0916 |

  **Bootstrap, XGBoost minus GLM A:** banded −0.0014 (95% interval −0.0034 to +0.0006; not better in 92% of resamples). Raw −0.0019 (−0.0055 to +0.0019; 84%).
- **Conclusion:** XGBoost does **not** beat GLM A on contents. The gap is within noise, so the two are statistically level, and the GLM is the more explainable of the two. Shallow trees did best in tuning: depth 1, which is additive with no interactions, was within 0.0002 of the best on 2023. So on this data, interactions add nothing measurable to expected cost, despite the size-dependent relativities in A09. Level-matched to actual, XGBoost's decile A/E ranges 0.87–1.09 (banded) and 0.75–1.16 (raw), against GLM A's 0.94–1.10. The trees under-predict the top deciles.
- **Finding for the 2025 price, all models:** loss cost per unit of exposure steps up from 105.5 and 107.0 (2021, 2022) to 116.9 and 116.8 (2023, 2024). Claim frequency is flat at 0.040–0.041, so the step is mostly in severity. Fitted on 2021–22, GLM A under-predicts 2023 by 7%. 2024 is not yet developed (about 3% IBNR, D-03), so its underlying level is probably above 2023's. Any 2025 submission needs an explicit trend or level adjustment; none is applied here.
- **Not explained:** the raw set scores below the banded set at every depth on 2023. Dropping the three fields whose meaning shifts with calendar time (`customer_since_year`, `years_at_address`, `renewal_count`) did not change this, so time drift in those fields is not the cause. The likely cause is extra noise from 35 more features, but that is untested.
- **Limitations:** the bootstrap resamples rows, not policies or addresses, so it ignores within-address correlation and its intervals are, if anything, too narrow. That strengthens the conclusion that the gap is not significant. The model is Python only, with no R twin. Gain importance is descriptive, not a test.
- **Evidence:** `evidence/xgboost/ev_x1_split_test.csv`, `ev_x2_tuning.csv`, `ev_x3_importance.csv`, `ev_x4_bootstrap_vs_glm_a.csv`, `ev_x5_deciles.csv`. Tests: `tests/test_xgb_models.py` (offset recovers a known rate and scales exactly with exposure; categories fixed to training levels; feature guard; bootstrap).
- **Status:** analysis. D-08 awaits sign-off.

### A14 — Submission files: contents predictions, other lines at 0.1
- **When / who:** 2026-10-21 ~19:50 BST, Claude, at the owner's request: "an output file similar to the example given, but for the home and motor policies put in 0.1". Read as buildings and motor rows = 0.1, with contents rows predicted.
- **Method:** `python/make_submission.py`. GLM A and XGBoost banded (depth 2, `min_child_weight` 20,000, 234 trees, from A13) are each refitted on all of 2021–24 and predict the 187,498 contents rows of `test_contents`. Row order and IDs follow the competition's `sample_submission.csv` (read from the team repo's git history into `work/`; its SHA-256 is in the run log). The outputs are `work/submissions/submission_contents_{glm_a,xgb_banded}_20261021.csv`. They are not committed: `work/` is git-ignored.
- **Verification:** both files have 626,639 rows, with row_id order and header identical to the sample, no nulls and every prediction above zero (a zero would make the Tweedie deviance infinite on any row with a claim). There are 184,155 buildings and 254,986 motor rows at 0.1. No contents test row has a factor level unseen in training, and test sum insured is within the training range. Tests: `tests/test_submission.py`.
- **Level:** the predicted 2025 contents rate per unit of exposure is 116.8 for GLM A and 112.5 for XGBoost, against training years of 105.5 / 107.0 / 117.0 / 116.8. The GLM's continuous log sum insured picks up the year-on-year rise in sums insured; the trees cannot extrapolate it. **No trend or IBNR adjustment is applied** (A13), so both are probably too low for 2025.
- **Leaderboard caution:** a constant 0.1 for buildings and motor scores those rows far below a mean-only prediction, so the overall leaderboard score reflects that choice, not the contents model alone.
- **Status:** analysis. Submitting is the owner's decision.

### A15 — XGBoost on all three lines, with development, a large-claim cap and a trend, combined submission
- **When / who:** 2026-10-21 ~19:00–20:20 BST, Claude, at the owner's request ("run XGBoost across the three datasets independently and combine the submission"). The owner approved the plan, including development for all lines and a per-line trend to 2025. Bridget's `hackathon-bridget` branch already had XGBoost for all lines. It was reviewed and not reused: it uses exposure as a weight and a feature instead of an offset, trains on renewed policies only, uses `Year` as a feature and has no development. Its v008 predicted means per row are about half of this run's (buildings 98 against 395, contents 48 against 100, motor 244 against 476).
- **Data layer (`sql/09`):** buildings and motor are built with the contents rules (D-01 to D-03).
  - Exposure is recomputed from the period dates: 4,138 buildings rows and 5,666 motor rows change.
  - Buildings sum insured is divided by 1,000 at **10m and above** (1,395 training rows, 370 test rows). The rule is the same as D-02, but the threshold isn't: values from 52m to 594m are wrong, with a same-policy ratio of median 978 across 3,265 pairs. Nothing lies between 2m and 36m, and one genuine-looking 1.06m test risk (premium 545) must not be divided.
  - Buildings `construction_type` is normalised for casing (11 raw values become 5).
  - The new claim build reproduces sql/03's 23,753 contents claims exactly (`ev_09_contents_claims_match`).
  - Unmatched claims dropped: 67 buildings and 294 motor. 10% of motor rows have no customer record, in training and test alike.
  - **Contacts leak in every line** (D-04 extended): claimants average 3.0 (buildings) and 3.3 (motor) contacts, against 0.4 for non-claimants and 0.40 in the test files.
- **IBNR (`sql/10`, D-10):** a cost-weighted report-lag distribution for each line and peril. It is estimated from every loss year with the right-truncation product-limit estimator and applied as the mean factor for each line, peril and loss quarter. Applying it claim by claim gave one early-reported motor bodily injury claim ×60; buildings breached the cap that way in the first attempt.
  - **Validation:** the estimated pattern matches the empirical share reported by lag in fully observed loss years to about ±0.01 (contents theft and escape of water, motor own damage and bodily injury).
  - Hand-checked case in `tests/test_sql_lines.py`.
  - Factors for loss year 2024: **buildings 1.109, contents 1.028, motor 1.268**. Factors for 2023 are 1.002–1.004, except motor at 1.123.
  - The largest are motor bodily injury (2.04 for 2024 including IBNER) and buildings subsidence (1.68).
  - **Correction to A06:** contents theft IBNR is 1.2%, not 3.45%, and escape of water is 4.6%, not 2.4%. sql/04 pooled one lag distribution across perils, and theft is reported faster than average. The total is little changed.
- **IBNER (D-11):** one snapshot cannot show development on reported claims. Average current cost of claims reported within 90 days (losses January–September) rises with inflation for every peril except **motor bodily injury**: 21,049 (2021, 10% open), 19,564, 17,325, 18,098 (2024, 74% open).
  - Each bodily-injury loss year is lifted to the 2021 level with no inflation: 1.076, 1.215 and 1.163 for 2022–24.
  - This is a lower bound. Inflation is left to the trend term, so adding it here would count it twice.
  - Buildings subsidence (58% open in 2024) shows no clear shortfall and is not lifted.
- **Large claims (D-12, added during the run, not in the approved plan):** one open 2023 motor bodily injury claim is **11.89m**, ten times the next largest (1.12m).
  - It is genuine, not a ×1,000 error: 738,769 has already been paid.
  - Developed claims are capped at 1m and the excess is spread back over the line as a multiplicative loading: motor ×1.031. No buildings or contents claim reaches the cap.
- **Developed cost per unit of exposure, capped excess included (2021–24):**
  - Buildings 383.9 / 410.5 / 433.8 / 498.1.
  - Contents 105.5 / 107.0 / 117.2 / 120.0.
  - Motor 564.5 / 606.7 / 659.6 / 575.4. Capped, motor 2023 is 588.3.
- **Model (`python/run_lines.py`):** XGBoost per line on the developed capped target, with a log-exposure offset and the A13 settings.
  - **Features:** buildings 53 raw columns, motor 52 raw columns, contents the banded set from A13.
  - **Trend (D-13):** a log-linear calendar-year term fitted jointly with the trees, by backfitting it into the offset over 4 iterations. The remaining slope is at most 0.0006 a year. Tested on synthetic data with a known trend.
  - **Tuning:** fit on 2021–22, early-stopped on 2023, over depth 1–4 and leaf size of 50–50,000 rows of median hessian.
- **Run 1 (kept in the run log):** leaf size stopped at 5,000 rows and motor's best sat on that edge. In run 2, 20,000 and 50,000 were added. A side check on motor found 100,000 rows gives 0.0853 and 200,000 gives 0.0782, against 0.0886 at 50,000, so 50,000 is the optimum. Chosen settings:
  - buildings depth 3, 20,000 rows, 184 trees;
  - contents depth 1, 5,000 rows, 574 trees;
  - motor depth 1, 50,000 rows, 1,081 trees.
- **Backtest (fit 2021–23, score 2024):**

  | Line | Model | D² vs reported | D² vs developed | Predicted / developed |
  |---|---|---|---|---|
  | Buildings | exposure only | 0.0417 | 0.0378 | 0.823 |
  | | XGBoost, no trend | 0.1106 | 0.1020 | 0.825 |
  | | **XGBoost, trend 3.3%** | **0.1114** | **0.1045** | 0.874 |
  | Contents | exposure only | 0.0310 | 0.0303 | 0.916 |
  | | XGBoost, no trend | 0.0920 | 0.0919 | 0.933 |
  | | **XGBoost, trend 3.6%** | **0.0923** | **0.0927** | 0.999 |
  | Motor | exposure only | 0.0192 | 0.0303 | 1.062 |
  | | XGBoost, no trend | 0.0711 | 0.0818 | 1.021 |
  | | **XGBoost, trend 4.2%** | 0.0661 | 0.0816 | 1.088 |

  The trend improves buildings and contents. For motor it scores the same against developed 2024 and worse against reported 2024. Reported 2024 is the least developed year, and the Kaggle target is ultimate cost, so D² against developed is the relevant comparison.
- **2025 (fit 2021–24):**

  | Line | Trend a year | 2025 rate per unit of exposure | vs developed 2024 |
  |---|---|---|---|
  | Buildings | 6.4% | 484.1 | 0.972 |
  | Contents | 3.3% | 124.5 | 1.037 |
  | Motor | 3.6% | 607.4 (load ×1.031 included) | 1.056 |

  Buildings 2025 sits below 2024 despite its trend because the model uses each year's weather. 2024 had more freezing days (39.5 against 34.7) and its escape of water cost rose from 99.9 to 117.3; 2025 is milder (35.9).
- **Submission:** `work/submissions/submission_all_lines_xgb_20261021.csv` (git-ignored). It has 626,639 rows, row_ids and order identical to the sample, no nulls, and a minimum prediction of 0.063. Mean per row: buildings 395, contents 100, motor 476.
- **Main uncertainties, in order:**
  1. **Motor bodily injury development.** 2024 developed bodily injury is 214 per unit of exposure, against 246–311 in 2021–23, so 2024 is probably still under-developed. That would explain the motor trend's 9% over-prediction of 2024 in the backtest.
  2. **Buildings subsidence for 2024.** Its 1.68 factor rests on 69% open claims.
  3. **The trends rest on four years each.**
  4. **The open-claim uplift for bodily injury is a lower bound.**

  Backtest deciles (level-matched) are within 0.84–1.15 for each line, except buildings decile 1 (0.72).
- **Checks:** the GLM pack is unchanged. `run_all.py` evidence is byte-identical, and after re-running R in this checkout `reconcile.py` gives 100 checks, 0 failed. Tests: 24 pass (pytest). Lint clean.
- **Status:** analysis. D-10 to D-13 await sign-off. Submitting is the owner's decision.

### A16 — Split models with claim-history credibility; buildings–contents correlation
- **When / who:** 2026-10-22, Claude, at the owner's request: GLMs, with large ("catastrophic") and small claims modelled separately, accounting for correlation between buildings and contents claims. After seeing the plan the owner set the design: **contents** one GLM, not split; **buildings** small claims by XGBoost at depth 3 plus a large-claim model; **motor** small-claim GLM plus a large-claim model; and "take full account of claim history".
- **Correlation evidence:** buildings and contents claims barely move together.
  - **Same customer, address and year (both policies held):** contents frequency is 42.3 per 1,000 with a buildings claim against 40.3 without.
  - **Same address and loss date:** 4 claim pairs, out of about 24,000 contents and 26,000 buildings claims. There are no shared events, not even flood.
  - **Next year:**

    | Prior-year claim | Next-year contents frequency | Next-year buildings frequency |
    |---|---|---|
    | None | 38.8 | 42.6 |
    | Contents only | 67.6 | 49.9 |
    | Buildings only | 43.2 | 75.0 |

  - **After the models, 2024 backtest:** Pearson residual correlation of buildings and contents by customer and address is **−0.002**; claim-indicator correlation is **0.005** (128,878 pairs).
  - So the link that matters is a customer's own history. A copula, a joint model of the two lines' claims, was not fitted: it would change the spread of combined losses, not the expected cost per row that is scored.
- **Claim history (`sql/12`, `sql/13`, D-14):** for every row, the claims of the same customer that were lost *and reported* before the row's period started.
  - **Groups:** own line (customer and address for home, customer for motor), the other home line (customer and address), and motor/home (customer).
  - Frequency by own prior claims (0 / 1 / 2 / 3+), per 1,000: buildings 43 / 69 / 108 / 185, contents 38 / 64 / 81 / 140, motor 116 / 196 / 291 / 470.
  - Tested for no look-ahead: a late-reported claim is excluded until it is known.
  - **First attempt, rejected (kept in the run log):** history as model factors. Prior exposure grows with the calendar year (none in 2021, up to 4 years for 2025), so it confounded the trend and the 2025 rows fell into levels never seen in training. Contents scored 0.0614 against 0.0939 without history, and motor's trend inflated to 8.5% a year.
  - **Adopted:** a Poisson–gamma credibility factor (k + n) / (k + E). n is the prior claims; E is the claims a frequency GLM without history expects over the same prior rows. It is rebalanced so the fitted rows keep their total, and k is tuned per group on 2023 (infinite k = group off).
  - Chosen k: own line 4 (buildings), 2 (contents), 8 (motor); other home line off for buildings, 8 for contents; motor/home off for buildings and motor, 8 for contents.
  - 2025 factors range from 0.67 to 3.38.
- **Large claims (D-15):** each claim is split at a threshold on its *reported* cost, and both parts are then developed. Developing first would push slow-reporting claims over the threshold (motor bodily injury Q4 2024 is ×5.2).
  - The large part is a compact Tweedie GLM, scaled so its mean matches a generalised Pareto fit to the excess, with the small-claim trend applied.
  - Thresholds tuned on 2023 from 25k, 50k and 100k: **buildings 100k** (Pareto shape 0.03, adjustment 1.00), **motor 50k** (shape 0.48, adjustment 0.77). For motor, the fitted mean excess of 50.5k replaces the empirical 65.6k, which the 11.9m claim inflated.
  - The D-12 cap is not used here.
- **Backtest (fit 2021–23, score 2024; D² against developed actual):**

  | Line | Chosen | No history | No trend | No split | A15 XGBoost | Predicted / developed |
  |---|---|---|---|---|---|---|
  | Buildings | **0.1082** | 0.1064 | 0.1057 | 0.1063 | 0.1045 | 0.877 |
  | Contents | **0.0945** | 0.0939 | 0.0938 | — | 0.0927 | 1.045 |
  | Motor | **0.0839** | 0.0810 | 0.0837 | 0.0783 | 0.0816 | 1.058 |

  The chosen models beat A15's XGBoost on every line. History adds 0.0018 (buildings), 0.0006 (contents) and 0.0029 (motor). The split adds 0.0019 (buildings) and 0.0056 (motor). These differences have not been tested for significance.
- **2025 (fit 2021–24):**

  | Line | Trend a year | Rate per unit of exposure | vs developed 2024 |
  |---|---|---|---|
  | Buildings | 6.3% | 491.8 | 0.987 |
  | Contents | 4.1% | 129.6 | 1.080 |
  | Motor | 4.3% | 610.0 | 1.060 |

  Submission: `work/submissions/submission_all_lines_glm_20261022.csv` (git-ignored). It has 626,639 rows in sample order, no nulls and a minimum prediction of 0.043. Mean per row: buildings 401, contents 104, motor 478; these correlate 0.94–0.97 with the A15 file.
- **Uncertainties:**
  1. A15's development uncertainties (motor bodily injury 2024, buildings subsidence) carry over unchanged. Buildings still under-predicts 2024 (0.877) and motor over-predicts (1.058).
  2. History counts use reported claims while E uses expected claims, so the most recent prior year is slightly short for slow-reporting perils. The rebalancing absorbs this on average, not by row.
  3. The large-claim GLMs rest on 593 (motor, at 50k) and 260 (buildings, at 100k) reported large claims.
  4. Python only; no R twin.
- **Checks:** tests 29 pass, including split, Pareto, credibility, history without look-ahead and expected history over the right rows. Lint clean.
- **Status:** analysis. D-14 to D-16 await sign-off. Submitting is the owner's decision.

### A17 — Model review: development backtest, Bornhuetter–Ferguson, smoothing, two motor factors
- **When / who:** 2026-10-22, Claude. The owner asked what would improve the GLMs without overfitting, and whether the data is suitable, then asked for recommendations 1–3 to be implemented, each tested with the two-year rule.
- **Review findings, before the changes:**
  - **Data structure is clean:** no duplicate rows, zero exposure or overlapping periods; no 2025 category levels unseen in training.
  - **Left as is:**
    - 11 home claims above the sum insured (at most 1.48×).
    - 15,036 zero-mileage motor rows.
    - 161,822 motor rows (13%) with more NCB years than years licensed. They claim *less* at every licence band (frequency 259 against 312 per 1,000 at 0–2 years licensed), so it is signal, not error.
  - **No overfitting:** contents D² is 0.101 in sample against 0.094 out of time; coefficients barely move between fitting windows (correlation 0.99, no parameter moves by more than 0.2); the GLM penalty is immaterial.
  - **Written premium adds nothing:** −0.0007 to +0.0002 D² across both years and both lines tested, although it is strongly predictive on its own. D-05 costs no score.
  - **Features that look strong in one year but fail out of time:** previous insurer, campaign code and occupation (−0.001 to −0.003).
- **The two-year rule:** a change is kept only if D² against the developed actual improves both on 2023 (fit 2021–22) and on 2024 (fit 2021–23). Scores are comparable only within one development basis.
- **1. Development backtest (`python/backtest_development.py`, `ev_d1`):** the position is rebuilt as at 2023-12-31 from report dates, 2023 is developed, and the result is compared with 2023 as now known (current incurred, IBNER off on both sides).
  - **Chain ladder:** within ±3.5% for every peril except buildings **subsidence (1.179)** and motor **bodily injury (1.324)**. These are the two perils under 60% reported after a year.
  - Motor bodily injury is reported faster each year (within 365 days: 85.9% → 87.0% → 89.3%), so pooling the years understates how complete recent years are.
  - **Bornhuetter–Ferguson (D-17):** BF replaces the chain ladder for a loss year under 75% reported. The prior is the cost per exposure of the up to two preceding years that are at least 90% reported, with no trend and claims capped at 1m. In the backtest: **subsidence 1.014, bodily injury 1.139.**
  - Applied to 2024 only for subsidence (11.3m against 13.3m by chain ladder) and bodily injury (46.0m against 42.8m). Developed 2024 cost per unit of exposure: **buildings 498.1 → 484.8, motor 575.4 → 591.8.**
  - A hand-worked test was added (`tests/test_sql_lines.py`).
- **1b. Buildings weather perils:** separate small-claim models for weather perils (storm, flood, escape of water, subsidence) and the rest **failed** on both years (−0.0015, −0.0018). The buildings model already uses each year's weather, so its trend is already net of weather.
- **2. Smoothing (D-18):** cubic splines with 4 knots at training quantiles, flat beyond the training range, with a missing indicator. The knot count was not tuned.

  | Change | Gain in 2023 | Gain in 2024 | Result |
  |---|---|---|---|
  | Motor small-claim GLM: all numerics smoothed, plus credit band and missed payments | +0.0053 | +0.0023 | Adopted |
  | Contents: GLM A's age, elevation and river-distance bands replaced by splines on the raw values | +0.0011 | +0.0019 | Adopted |
  | Buildings large-claim GLM smoothed | +0.0010 | +0.0005 | Adopted |
  | Motor large-claim GLM smoothed | −0.0020 | +0.0004 | Rejected |

  Credit band and missed payments alone gave +0.0015 and +0.0008.
- **Full run (fit 2021–23, score 2024; D² against developed):**

  | Line | Chosen | No history | No trend | No split | Predicted / developed |
  |---|---|---|---|---|---|
  | Buildings | **0.1114** | 0.1097 | 0.1102 | 0.1095 | 0.917 |
  | Contents | **0.0958** | 0.0959 | 0.0951 | — | 1.047 |
  | Motor | **0.0801** | 0.0771 | 0.0791 | 0.0751 | 1.056 |

  - **Contents** uses the same development basis as A16, so the two compare directly: **0.0945 → 0.0958**.
  - **Buildings and motor** targets changed with D-17, so they cannot be compared with A16. Their improvement rests on the two-year tests above, run on the new basis.
  - **Buildings** now under-predicts 2024 by 8%, against 12% in A16.
  - **Contents claim history** now scores −0.0001 on 2024 (+0.0017 on 2023), so it would narrowly fail the two-year rule. It is kept under D-14 as the owner asked; the difference is within noise.
  - Retuned settings: buildings large-claim threshold 50k (was 100k); GLM penalties 1e-4 (contents) and 1e-5 (motor).
- **2025:**

  | Line | Rate per unit of exposure | A16 | vs developed 2024 | Trend a year |
  |---|---|---|---|---|
  | Buildings | 487.5 | 491.8 | 1.006 | 5.4% |
  | Contents | 129.6 | 129.6 | 1.080 | 4.2% |
  | Motor | 627.9 | 610.0 | 1.061 | 5.5% |

  Motor rises because Bornhuetter–Ferguson lifts 2024 bodily injury.
- **Submission:** `work/submissions/submission_all_lines_glm_20261022.csv` has 626,639 rows in sample order, no nulls and a minimum prediction of 0.043. It overwrites A16's same-day file, which can be regenerated from commit 522115f.
- **Caveats:**
  - The 2023 and 2024 backtests have now informed many choices, so the reported gains are somewhat optimistic.
  - The bodily-injury "actual" in the development backtest is itself 49% open.
- **Checks:** tests 31 pass (pytest); lint clean.
- **Status:** analysis. D-17 and D-18 await sign-off.
