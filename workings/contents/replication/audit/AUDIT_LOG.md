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

### A18 — Duplicate claims and unpadded motor customer IDs (found by checking the team's issues)
- **When / who:** 2026-10-22, Claude, while preparing documentation and GitHub issues at the owner's request. The team's data workstream had already found both problems (GitHub #55 duplicate claims and #51 motor customer_id padding, both closed for their pipeline). This pack had not applied either fix.
- **Duplicate claims (D-19):** 604 claims repeat another claim on every field except `claim_id`: policy, peril, loss and report dates, status, paid, incurred and recoveries. For example, two motor third-party property claims on policy M46478953 share a loss date (2022-01-27), an amount (4,160.43), a repairer and a third party. There are 411 in motor, 109 in buildings and 84 in contents (one of the 84 is repudiated). One copy of each is kept (`claims_clean`, sql/03); every later step reads it, while sql/01 and sql/02 still profile the raw table. Contents loses 229k of cost (0.35%).
- **Motor customer IDs:** 98,110 training rows and 25,554 test rows (10%) hold `customer_id` without leading zeros. sql/02 flagged this in A03 (`ev_02_motor_customer_id_padding`), but sql/09 never applied the fix. After padding to 8 characters, every motor row finds its customer record.
- **Correction to A15 and A16:** they said 10% of motor rows "have no customer record". That was this defect, not missing data. Until now, those rows lacked customer features and their motor/home history links.
- **Re-run:**
  - **GLM pack:** reconciliation 100 checks, 0 failed. GLM A's 2024 D² is 0.0933 before and after.
  - **Development backtest:** subsidence 1.012, bodily injury 1.134 under BF (chain ladder 1.177 and 1.326).
  - **Current model, 2024 backtest:** buildings 0.1114 → **0.1109**, contents 0.0958 → **0.0965**, motor 0.0801 → **0.0806**.
  - **Contents claim history** now improves 2024 as well (+0.0006 against no history).
  - **2025 rates:** buildings 488.8, contents 129.7, motor 627.4.
  - **Submission:** `work/submissions/submission_all_lines_glm_20261022.csv` regenerated (626,639 rows in sample order, no nulls, minimum 0.044).
- **Not changed:** A13–A15 (the XGBoost runs) were not re-run, so their evidence reflects claims before de-duplication. They are superseded by the current model.
- **Also confirmed:** the competition data is in the public GitHub repo (`data/*.parquet` on `main`, `hackathon-bridget` and `data-cian`), the risk A02 raised. This needs a team decision under rules §5b.
- **Tests:** 33 pass, including duplicate claims counted once and motor IDs padded.
- **Status:** applied-derived. D-19 awaits sign-off.

### A19 — Work recorded as GitHub issues
- **When / who:** 2026-10-22, Claude, at the owner's instruction: post the issues on the public repo as closed, and push `contents/gerard` so their links work. The owner gave this instruction after being told that the repo is public and holds the competition data (A18).
- **What:** 22 issues, #105–#121 and #123–#127, each created and closed as completed. They follow the team template (AI interaction log, tasks, notes), carry the team's labels and cross-reference the data workstream's issues (#51–#56, #59–#64, #66–#72, #83, #86–#87, #96, #100, #102, #104). #122 is not part of this set. `contents/gerard` was pushed (ef86154..abfed31); the push contained no data or submission files.
- **Status:** done. The public-data question (A02, A18) remains open for the team.

### A20 — CANN ensembles, bootstrap significance and the 2025 level under uncertainty
- **When / who:** 2026-10-22, Claude, at the owner's instruction to implement three items under the two-year rule: (1) CANN ensembles per line, (2) significance by bootstrap and (3) the level set under uncertainty. The owner had earlier asked for full use of the M3 Max GPU.
- **Probes before implementation (scratch, no evidence kept):** residual boosting on the GLM, isotonic recalibration, frequency boosting, credibility-encoded make and model, and a GLM + XGBoost blend each failed the two-year rule.
  - A CANN on *all* motor claims with 5 seeds failed: −0.0012 on 2023, +0.0008 on 2024.
  - A CANN on small claims with 20 seeds, trained one after another, passed for motor: +0.00015 and +0.00054.
- **(1) CANN (D-20):** the incumbent small-claim model's log prediction is the offset, and a network with a zero-initialised output layer learns a log correction.
  - **Network:** embeddings for categories; hidden layers 128/64/32 with SiLU and 10% dropout; Tweedie loss with p = 1.5; Adam at 0.001, batches of 16,384; early stopping on 10% of the rows with a patience of 4.
  - **Ensemble:** 20 seeds, mean log correction, rebalanced to the incumbent's total on the fitting rows. The architecture was fixed in advance, not tuned.
  - **Results (2023 / 2024 D², without → with):**

    | Line | 2023 | 2024 | Outcome |
    |---|---|---|---|
    | Buildings | 0.11170 → 0.11175 | 0.11090 → 0.11095 | Adopted |
    | Contents | 0.09438 → 0.09444 | 0.09645 → 0.09650 | Adopted |
    | Motor | 0.10097 → 0.10078 | 0.08062 → 0.08128 | Rejected |

    Evidence: `ev_g7_cann_two_year.csv`.
  - **Adopted gains are at the edge of noise:** only buildings 2023 is significant (below). With the batched training the motor probe's result reversed, a sign that the motor effect is seed noise.
- **Engineering, including two failures:**
  - **OpenMP conflict.** PyTorch and XGBoost each load their own OpenMP runtime on macOS. With both in one process, XGBoost segfaulted, and the test run deadlocked depending on import order. Torch was moved into a separate worker process.
  - **First worker design crashed.** It used a `multiprocessing` spawn pool, but a spawned child re-imports the calling script, so XGBoost's runtime was loaded beside torch's anyway. The worker segfaulted 4 s into the first fit (macOS crash report at 17:08:53). The pool replaced the dead worker and lost its task, so the run waited 78 minutes with no CPU use until it was stopped.
  - **Fix.** `cann.py` now starts `cann_torch.py` as a plain subprocess and talks to it over a pipe. If the worker dies, the next call raises immediately. A regression test imports XGBoost in the caller and checks that the worker never loads it.
  - **GPU.** All 20 seeds now train as one batched network, with weights stacked on a seed axis. Each seed keeps its own initialisation, batch order, per-seed gradient clipping and early stopping; a test checks that seeds cannot influence each other.
    - GPU utilisation is about 96% during training. A synthetic 900,000-row, 20-seed benchmark ran at about 1.45 s per epoch.
    - CANN takes about 6 of the full run's 20 minutes. The rest is SQL, GLM and XGBoost work on the CPU.
  - **Reproducibility.** Two full runs gave identical D² to five decimals.
- **(2) Significance:** a paired bootstrap that resamples whole customers (1,000 times) and compares D² of the chosen model with each variant on the same rows. Evidence: `ev_g8_bootstrap_significance.csv`.
  - **Significant (95% interval above zero):**
    - Claim history: buildings both years, contents 2023, motor 2024.
    - A17 changes: buildings both years, contents 2024, motor both years.
    - Trend: buildings 2024, contents 2023.
    - Motor large-claim split: 2024.
  - **Not significant:** CANN except buildings 2023; contents history on 2024; the trend on motor.
  - Motor CANN is significantly better on 2024 (+0.0007, interval 0.00003 to 0.0014), but the two-year rule rejects it on 2023.
- **(3) Level (D-21):**
  - **Level factor.** The deviance-optimal scale k = Σyμ^(1−p) / Σμ^(2−p) on each backtest year was tested on the *other* year. k (2023 / 2024) is 1.005 / 1.050 (buildings), 1.114 / 0.981 (contents) and 1.022 / 0.973 (motor). It fails on every line, so no factor is applied (`ev_g9_level_factor.csv`).
  - **Development uncertainty** (`python/level_uncertainty.py`, `ev_u1_development_bootstrap.csv`): 200 resamples of training rows, each re-running sql/10 in full, on 14 processes in 56 s. The CV of the development factor for 2024 / 2023 is 0.84% / 0.09% (buildings), 0.39% / 0.04% (contents) and 3.0% / 4.3% (motor). Motor 2024's factor is 1.303, with a 90% range of 1.238 to 1.369.
  - **Level variance for 2025:** CV₂₀₂₄² + 0.25·CV₂₀₂₃² + 2.5·σ_year², with σ_year from the final model's log actual/expected by year on 2 degrees of freedom. σ is 9.7% (buildings), 5.8% (contents) and 13.8% (motor).
  - **Tilt.** exp(σ²/2) is +0.48% (buildings), +0.17% (contents) and +0.96% (motor). The first full run applied it without a test, which was found on review. It was then put under the two-year rule and is kept for buildings only, because contents and motor over-predicted 2024 (`ev_g10_level_uncertainty.csv`).
  - **2025 rates (90% ranges):** buildings 491.4 (419–577), contents 129.9 (118–143), motor 627.4 (500–788).
  - **Caveats:**
    - The development bootstrap covers sampling error only, not method error (13% on bodily injury in the A17 backtest).
    - Buildings' year residuals are all positive (+1% to +6%), so σ_year there includes bias.
- **Correction:** `docs/MODEL_DOCUMENTATION.md` §8 said under-prediction costs far more D² than over-prediction (0.0015 against 0.0003 for ±8% on contents 2024). That was a misreading with no audit entry. For p = 1.5 the deviance excess of a scale error r is proportional to 2·cosh(ln r / 2) − 2, which is symmetric in ln r. So 8% low costs about 18% more than 8% high, not five times more. The document now says so.
- **Backtest 2024 (chosen model):** buildings 0.1110, contents 0.0965, motor 0.0806.
- **Submission:** `work/submissions/submission_all_lines_glm_20261022.csv` has 626,639 rows in sample order, no nulls and a minimum of 0.043. Mean prediction: buildings 400.9, contents 104.0, motor 492.1.
- **Checks:** 38 tests pass; ruff passes with the project settings.
- **Status:** analysis. D-20 and D-21 await sign-off. Recommendation: given the significance results, the owner may prefer to drop CANN for buildings and contents. The model would be simpler and score within 0.0001 of the current one.

### A21 — Inverting the model: what would make it wrong, and four gaps tested
- **When / who:** 2026-10-22, Claude. The owner asked what would make the model bad or less accurate, then asked for the open gaps to be tested under the two-year rule.
- **Checks on the data (ad hoc DuckDB queries, no evidence files):**
  - **Train and test rows are built the same way.** By start month, the 2024 training and 2025 test rows match in count and mean length (for example, motor January starts: 204,567 against 205,065 rows, 318 days each).
    - An apparent difference in the quarterly exposure mix was an artefact of sampling in the query.
    - 2024 rows end on 30 December because periods are 365 days and 2024 is a leap year. No policy covers 31 December 2024 and no claim is dated then, so nothing is lost.
  - **Leak scan.** For every numeric feature: the gap between claimants and others, and the shift from 2024 to 2025. A pure outcome leak shifts the test mean by −(claim share × gap).
    - Customer service contacts matches exactly (ratio 1.02–1.11 on all lines), which confirms D-04.
    - `missed_payments_12m` gives 0.5–1.3, but its shifts are within noise.
    - Same-year against next-year claims: the missed-payments gap is 1.3–2.2× larger for the same year on every line. For contacts it is 40× larger. For voluntary excess and telematics the two are equal; NCB is driven by the previous year, as it should be.
  - **Frequency and severity move in opposite directions** for several factors. For example, motor third-party-only cover has 0.55× the frequency and 1.75–2.6× the severity; contents accidental damage has 1.37× and 0.80×.
  - **Seasonality.** Claim frequency by month, relative to the year:
    - Buildings: 1.26–1.43 in December to February, 0.82–0.92 the rest of the year.
    - Contents: 1.09–1.16 in December to February.
    - Motor: 0.96–1.06, rising towards the year end.
  - **Premium index.** Like-for-like renewal premium rises by the same amount every year: buildings +6.5%, contents +5.8%, motor +3.2%. That suggests a fixed premium index in the generator. It mixes in risk ageing, so it is not used as evidence of claims inflation.
  - **Scoring.** The rules define the D² baseline as a constant *rate*; the pack uses a constant mean. Model comparisons are unaffected, but the pack's D² will not match the leaderboard. The leaderboard probably pools all three lines.
- **Gap tests** (`python/gap_tests.py`, `ev_g11_gap_tests.csv`; three runs, the last two identical to five decimals):
  - **Baseline:** the chosen model without CANN, with the last full run's tuned settings. It reproduces that run's D² exactly (buildings 0.11170 and 0.11090).
  - **Method:** each variant changes one thing and is compared with a paired customer bootstrap. Buildings' tree count is re-tuned for each variant as in the full run, and also held at the baseline's 144, because re-tuning alone moved the tree count from 133 to 175.
  - **G1, missed payments** (lagged = the previous row's value, known before the period; 71–72% of rows have one):

    | Line | Lagged 2023 / 2024 | Dropped 2023 / 2024 |
    |---|---|---|
    | Buildings (baseline trees) | −0.0004 / −0.0003 | −0.0011 / −0.0004 |
    | Contents | −0.0014 / +0.0000 | −0.0012 / −0.0008 |
    | Motor | −0.0003 / −0.0003 | −0.0005 / −0.0002 |

    All fail. Dropping loses more than lagging on every line, so the feature carries genuine signal. The backtest rows would carry any leak as well, so this cannot rule one out. But the most a leak could be adding is about the gap between the current and lagged value, at most 0.0004 on 2024. **Kept as is.**
  - **G2, adoption by significance** (`ev_g12_significance_rule.csv`): a component is kept only if it passes the two-year rule *and* its two-year mean gain has a 95% interval above zero. The interval is approximated from the per-year intervals, treating the years as independent; they share customers, so it is if anything too narrow.
    - **Would be dropped:** CANN for contents (−0.0002 to +0.0003), the buildings large-claim split (−0.0004 to +0.0033), and the trend for contents and motor.
    - **Stays:** CANN for buildings (+0.00002 to +0.00009: tiny but consistent).
    - **Also shown:** the motor large-claim split fails even the plain two-year rule (−0.0011 on 2023, the year of the 11.9m claim). It is kept under D-15.
    - **Conclusion:** the rule suits discretionary statistical additions (CANN), not structural components with a prior rationale. Dropping the trend would price 2025 at the 2021–24 average.
  - **G3, dispersion (double GLM):** contents +0.0009 / −0.0004, motor −0.0002 / −0.0001. Both fail. Reweighting rows by a fitted dispersion does not improve the mean at this volume of data.
  - **G4, seasonal exposure** (days weighted by the month's relative frequency, estimated on the fitting years; fitting-year exposure totals are preserved exactly; `ev_g13_seasonal_weights.csv`):
    - **Contents: +0.0010 (significant) / +0.0003. Passes** the two-year rule and the significance rule (two-year mean +0.00065, 95% interval +0.00008 to +0.00123).
    - **Buildings:** −0.0005 / −0.0004 with baseline trees, so it fails despite the strongest seasonality. A possible reason, not tested: the winter peak comes from year-specific weather that the annual weather features already carry, and frequency weights do not match cost seasonality.
    - **Motor:** +0.0002 / −0.0007 (2024 significantly worse). It fails.
- **Status:** analysis. Nothing is changed in the model. The one candidate is seasonal exposure for contents, which needs the owner's decision, a full run and a new submission. Proposed decision: D-22.

### A22 — Seasonal exposure for contents (D-22) and significance for CANN (D-23)
- **When / who:** 2026-10-22, Claude, at the owner's instruction to adopt D-22 and D-23, run the full pipeline and regenerate the submission.
- **D-22 in the runner:** `Line.use_window()` sets each row's exposure to its days by month (from `line_rows`) weighted by the month's claim frequency relative to the year. The weights come from the window's fitting years only:
  - tuning: 2021–22;
  - backtest: 2021–23;
  - final: 2021–24.

  The fitting years keep their total exposure, which is asserted in code. Reported rates use the plain exposure, kept as `exposure_plain`. The expected claims behind the credibility factor (sql/13) still use plain exposure, as in the A21 test.
  - **Weights:** for the backtest windows they are identical to A21's. The final (2021–24) weights are January 1.108, February 1.127, December 1.066 and March to November 0.943–1.002 (`ev_g14_seasonal_exposure_weights.csv`).
  - **Switch:** `glm_lines.models.contents.seasonal_exposure`.
- **D-23 in the runner:** CANN is adopted only if it passes the two-year rule *and* its two-year mean gain has a 95% interval above zero. The yearly paired bootstraps now run inside the CANN step (`ev_g7` gains their intervals). Switch: `glm_lines.cann.require_significance`.

  | Line | Two-year mean gain (95% interval) | Outcome |
  |---|---|---|
  | Buildings | +0.00005 (+0.00002, +0.00009) | Adopted |
  | Contents | +0.00005 (−0.00019, +0.00030) | Not adopted |
  | Motor | +0.00024 (−0.00023, +0.00071) | Not adopted (fails 2023) |

- **Contents results:** the D² baseline (without CANN) moves from 0.09438 to 0.09538 on 2023 and from 0.09645 to 0.09677 on 2024, exactly the A21 gap test.
  - The chosen model scores 0.0968 on the 2024 backtest (0.0965 in A20).
  - Predicted / actual on 2024 is 1.039 (was 1.047).
  - The uncertainty tilt still fails for contents. The trend is 4.2% a year.
- **Unchanged:** buildings and motor give the same D², settings and 2025 rates as A20. The only difference in their evidence is buildings' largest prediction, at the ninth significant figure (10,970.508372 against 10,970.508369): floating-point noise from the GPU.
- **2025 rates:** buildings 491.4, contents 129.2 (was 129.9; 90% range 117–142), motor 627.4.
- **Submission:** `work/submissions/submission_all_lines_glm_20261022.csv` has 626,639 rows in sample order, no nulls and a minimum of 0.049. Mean prediction: buildings 400.9, contents 103.4 (was 104.0), motor 492.1.
- **Code also changed:**
  - `prepare()` is shared with `gap_tests.py`; this run is its first full use in production.
  - `significance.two_year_interval()` is shared by the runner and the gap tests.
  - An empty calendar month no longer turns the seasonal weights into NaN; a unit test found this.
- **Checks:** 41 tests pass, including seasonal exposure keeping the fitting total, weighting winter rows up and leaving other lines untouched. Ruff passes.
- **Status:** applied. D-22 and D-23 were adopted by the owner in session; formal sign-off is outstanding.

### A23 — Work A20–A22 and open risks recorded as GitHub issues
- **When / who:** 2026-10-22, Claude, at the owner's instruction to make sure everything is documented and every issue encountered is raised on GitHub.
- **Closed issues #128–#139** (resolved work, created and closed as completed):
  - CANN (#128);
  - the PyTorch/XGBoost crash and the 78-minute hang (#129);
  - the customer bootstrap (#130);
  - the 2025 level and its ranges (#131);
  - the uncertainty adjustment applied without the two-year test (#132);
  - the wrong asymmetry claim in the documentation (#133);
  - the inversion and its data checks (#134);
  - missed payments (#135);
  - the double GLM (#136);
  - seasonal exposure (#137);
  - adoption by significance (#138);
  - documentation and the submission (#139).
- **Open issues #140–#147:**
  - level ranges missing method and tail error (#140);
  - categorical leak scan (#141);
  - the motor large-claim split (#142);
  - buildings under-prediction in total (#143);
  - buildings seasonality (#144);
  - no end-to-end test (#145);
  - the D² baseline (#146);
  - decision sign-off (#147).
- **Format:** all issues use the team template (AI interaction log, tasks, notes) and the team's labels. They cross-reference the team's #67, #70, #71, #91, #92, #101 and #110 where relevant, and link to the documentation on `contents/gerard-a20-a22`.
- **Not raised:** the competition data in the public repository (A02, A18). A public issue would draw attention to the possible breach, so it is left to the team.
- **Documentation:** a risk register (section 8.1) maps each risk to its issue.
- **Status:** done.

### A24 — Combining the best techniques from every branch
- **When / who:** 2026-10-22, Claude, at the owner's instruction to look through all the branches, combine the best ideas into a new model and run it. The plan was signed off as written.
- **Review** (two read-only agents, plus a check of each claim used):
  - Alan's `main`: GLM/XGBoost blends and chain-ladder development (Benktander selected).
  - Bridget's `hackathon-bridget`: GLM per line scaled to the 2024 level, GLM/XGBoost blend, XGBoost tuning harness.
  - Cian's `data-cian`: 2025 benchmark per line and the data contract.
  - The team's issues.
- **Recorded leaderboard scores**, all Bridget's: v008 −0.12546 (trained on renewed policies only), v009 +0.00101, v013 0.10829, v014 0.10916. The leader then was 0.12996. No score is recorded for any of this pack's files.
- **Data fix found (D-24):** buildings `construction_type` `T` appears only in 2021–22 and `W` only from 2023, about 22,000 rows a year each. Every 2025 test row is `W`.
  - Merged as `timber` in sql/09, raw value kept; switch `D24_merge_timber`.
  - A scan of all 31 text fields on the three lines found no other year-specific level.
  - Source: the data workstream's mapping (#54).
- **Tests** (`python/combined_tests.py`, `ev_c1_combined_tests.csv`, `ev_c2_blend_weight_grid.csv`). Baseline: the production model without CANN. Blend weights chosen on one year and scored on the other. Results as D² change on 2023 / 2024:
  - **Timber merge:** −0.0007 / −0.0004, not significant. Re-tuning took buildings from 144 to 131 trees. Applied as a data correction, as the plan set out.
  - **GLM + XGBoost blend:** best weights on the alternative were 0.0 / 0.2 (buildings), 0.0 / 0.0 (contents) and 0.0 / 0.2 (motor). Crosswise scoring: buildings −0.0004 / 0.0000, contents 0.0000 / 0.0000, motor −0.0020 / 0.0000. Fails everywhere.
  - **XGBoost with the harness's best settings** (depth 4, min_child_weight 10, colsample 0.7, lambda 3, 500 trees), on its own: contents −0.089 / −0.036, motor −0.039 / −0.012. Early stopping on 2023 would have kept only 55 and 99 trees.
    - A first run used those early-stopped counts (contents −0.018 / −0.015, motor −0.020 / −0.011). It was repeated with Bridget's fixed 500 trees to give her settings a fair test.
  - **Smoothed GLM for buildings small claims:** −0.0071 / −0.0041.
  - **Frequency × severity GLM** (Poisson × gamma, each with its own trend): contents −0.0006 / −0.0009, motor −0.0010 / +0.0007. Fails.
    - The first attempt crashed on motor rows whose claims cost nothing. Such claims are now left out of the count, consistently in both parts.
- **Why the other branches rated these higher:** random customer folds over all four years, and blend weights chosen on the scored year, both reward fitting year-specific noise.
- **Cross-check:** this pack's 2024 development factors (1.079 / 1.027 / 1.303) agree with the Benktander factors on `main` (1.081 / 1.037 / 1.293).
- **Combined model:** the A22 model plus D-24.
  - **Buildings:** CANN now gains +0.00014 / +0.00025, significant both years (two-year mean +0.00019, interval +0.00005 to +0.00034). The chosen model scores 0.11116 / 0.11071 against 0.11175 / 0.11095 before the merge. 131 trees, trend +5.9% a year, tilt ×1.0042.
  - **Contents and motor:** contents changes only through the cross-line history; motor is identical.
  - **2025 rates (90% range):** buildings 492.3 (423–573), contents 129.2 (117–142), motor 627.4 (500–788).
- **Submission:** `work/submissions/submission_combined_v1_20261022.csv`. It has 626,639 rows in sample order, no nulls and a minimum of 0.049. Buildings correlates 0.9989 with the A22 file.
  - **Leaderboard A/B file:** `submission_combined_v1_20261022_motor_at_v014_level.csv`, identical except that motor is ×0.8635 to v014's mean of 424.91 (`python/level_variant.py`, logged).
  - **Earlier output files** were renamed by the owner and are untouched.
- **Issues:** #148 (timber), #149 (team techniques), #150 (combined run) closed; #151 (motor leaderboard A/B) open.
- **Checks:** 43 tests pass, including the timber merge with its switch on and off and the level variant; ruff passes.
- **Status:** D-24 applied; the motor level is open (#151).

### A25 — Traditional development methods: offset fit, recent weighting and Mack
- **When / who:** 2026-10-22, Claude, at the owner's instruction to test whether chain-ladder and triangle methods improve the model. The owner also asked that the runs be labelled, the audit trail continued for every run, and the latest runs and code pushed to a new branch.
- **Run labels:** every runner now accepts `--run-label "<text>"`, stored as `label` in its run-log record (`common.run_label`). All A25 runs are labelled "A25 development methods: …". The trail includes:
  - one `level_uncertainty` run at 20:59 UTC in which my edit adding Mack had not applied, so it reproduced the bootstrap only;
  - one `development_tests` run at 21:03 that failed. Contents' exposure had already been switched to seasonal (D-22) when the line was copied, and the check in `use_window` caught the mismatch. It was rerun after the copy was built from plain exposure.
- **(3) Recent-weighted reporting patterns:** `sql/10` gains `pattern_decay`, with each claim weighted decay^(valuation year − loss year). At 1.0 the 2023 and 2024 factors match `ev_u1` to six decimals.
  - **Method:** `python/backtest_development.py` now rebuilds the position at 2022-12-31 and at 2023-12-31, for decays 1, 0.5 and 0.25 (`ev_d2_pattern_decay_backtest.csv`). `ev_d1` is unchanged.
  - **Developed / actual, by line:**

    | Line | As at 2022 | As at 2023 |
    |---|---|---|
    | Motor | 1.035 → 1.037 → 1.039 | 1.055 → 1.065 → 1.075 |
    | Contents | 1.002 → 1.003 → 1.003 | 0.988 → 0.988 → 0.987 |
    | Buildings | 0.992 → 0.993 → 0.994 | 0.993 → 0.997 → 1.000 |

  - **Not adopted:** motor and contents are worse in both years; buildings is better by under 1%, but its subsidence 2023 worsens from 1.012 to 1.065.
- **(1) Fitting on reported claims with a development offset** (`python/development_tests.py`):
  - **Method:** the small-claim model is fitted to reported cost with exposure × F as the offset. F is each row's expected reported share: reported / developed cost by line and loss quarter, weighted by the row's days in each quarter (`ev_dv2`). Its mean in 2024 is 0.92 (buildings), 0.97 (contents) and 0.77 (motor).
  - **Results** (`ev_dv1_offset_fit.csv`, 2023 / 2024): buildings 0.0000 / +0.0002 (interval −0.0003 to +0.0007), contents 0.0000 / 0.0000, motor −0.0002 (significant) / −0.0003. In the tuning window F is close to 1, so 2023 cannot show much.
  - **Not adopted.**
- **Mack** (`python/mack.py`): Mack's (1993) standard error of the chain-ladder reserve.
  - **Check:** the unit test reproduces the published example (Taylor and Ashe triangle, reserve 18,680,856, standard error 2,447,095).
  - **Method:** `level_uncertainty.py` builds quarterly reporting triangles per line (reported cost by loss quarter and quarters to report, 2021 Q1 to 2024 Q4) and writes the CV of the ultimate for loss years 2023 and 2024 (`ev_u2_mack_development.csv`).
  - **CV of the 2024 ultimate:** buildings 1.08%, contents 0.56%, motor 5.38%, against 0.84%, 0.39% and 2.96% from the bootstrap.
  - **CV of the 2024 reserve:** 14% / 18% / 24%, close to the Mack analysis on `main` (13% / 17% / 21%, #101).
  - **Adopted (D-25):** the level step uses the larger CV per line and loss year (switch `glm_lines.level.include_mack`).
- **Finding:** the motor development over-estimated the backtest year at both valuation dates (×1.035, ×1.055), mostly through bodily injury (×1.087, ×1.134). Raised on #151 and #140. It points the same way as the lower motor level of the best-scoring leaderboard file.
- **Full run** (labelled "A25 development methods: full run with Mack uncertainty"): `work/submissions/submission_combined_v2_dev_methods_20261022.csv`.
  - **Checks:** 626,639 rows in sample order, no nulls, minimum 0.049.
  - **Predictions:** identical to combined v1 for contents and motor; buildings within 0.003% (adjustment ×1.0043, was ×1.0042).
  - **2025 90% ranges:** buildings 423–573, contents 117–142, motor 494–797 (was 500–788). Motor σ is 14.5% (was 13.8%).
- **Issues:** #152 (closed) records A25; progress comments on #140 and #151.
- **Checks:** 45 tests pass (new: Mack against the published example, the offset line's reported targets); ruff passes.
- **Status:** D-25 applied; recent weighting and the offset fit are rejected. Pushed to the new branch `contents/gerard-a25-development`.

### A26 — Geolocation: location information below region
- **When / who:** 2026-10-22, Claude, at the owner's request to explore geolocation tagging from addresses, on a new branch (`contents/gerard-geolocation`).
- **What the data holds:**
  - **No geocoding possible:** `addresses` has 575,983 rows with region (12), urban density, flood zone, distance to river, elevation, soil, and property fields. There are no coordinates, postcode or street. `address_id` is opaque: every ID decile has all 12 regions, a top-region share of 0.158–0.164 and mean elevation 11.5–11.6 m.
  - **Weather as a location proxy:** `weather` has one row per address and year, 2021–25. Region explains this share of the variance of each address's 2021–25 average: rainfall 0.77, temperature 0.97, freezing days 0.91, wind 0.77, hail 0.59, dry summer 0.95, wildfire 0.83. Urban density adds nothing to region. One address's rainfall correlates 0.81–0.94 between years, so the within-region part is a persistent, sub-regional climate fingerprint.
  - **Shared addresses:** 553,117 addresses have one customer and 22,866 have more.
  - These profiles came from ad hoc DuckDB queries; there are no evidence files for them.
- **Tests** (`python/geo_tests.py`, run labelled "A26 geolocation: residuals and location features"; 357 s):
  - **L0 residuals:** actual / expected of the production model (without CANN) by region, region × urban and climate quintiles (`ev_geo1_location_residuals.csv`, 85 groups per line and year). z uses a Tweedie variance φμ^p with φ from the Pearson residuals.
    - No group is off in the same direction on both years with |z| > 2.
    - Share of groups with |z| > 2: buildings 5.9% / 10.6% (2023 / 2024, different groups each year), contents 0% / 0%, motor 8.2% / 0%. Motor 2023's maximum, z = 9.5, is the group holding the 11.9m claim.
  - **L1–L3 features** (`ev_geo2_location_feature_tests.csv`, D² change 2023 / 2024):
    - contents yearly weather +0.0002 / −0.0031 (2024 significant);
    - contents climate +0.0001 / −0.0002;
    - contents region × urban −0.0016 / −0.0010 (both significant);
    - motor yearly weather −0.0063 / −0.0004;
    - motor climate +0.0003 / −0.0012;
    - motor region × urban −0.0005 / −0.0005;
    - buildings climate, with the baseline's 131 trees, −0.0005 / −0.0001.

    None passes the two-year rule.
- **Conclusion:** the existing location factors already carry the location signal, and nothing is adopted. The model and submission are unchanged, so no full run was needed. With real coordinates, the next test would be spatial smoothing of residuals (territory credibility).
- **Issue:** #153 (closed).
- **Checks:** 46 tests pass (new: the feature helpers copy the config); ruff passes.
- **Status:** analysis only.

### A27 — Final review, leaderboard results and the visual report
- **When / who:** 2026-10-22 to 23, Claude, at the owner's instruction to do a thorough final pass for anything missed that is likely to improve the model, then write a report, with graphics, of what was done and what was and was not tested.
- **Leaderboard results** (public scores, shared by the owner; `ev_r4_leaderboard_scores.csv`; recorded on #151):

  | File | Public score |
  |---|---|
  | A22 file (seasonal contents, before the timber merge) | **0.12753** (best) |
  | Combined v1 (A24) | 0.12727 |
  | Combined v2, Mack ranges (A25) | 0.12727 |
  | Combined v1 with motor ×0.8635 | 0.12725 |
  | Competition leader | 0.13050 |

  - **Timber merge:** cost about 0.0003, in line with both backtests.
  - **Motor level:** the motor pair implies a score-optimal motor level of about ×0.93 of ours (×0.926–0.935 for ±0.0001 of noise on the difference), worth about +0.0004. This uses the pooled backtest's curvature of the score in the motor level, about 0.071 per (ln k)², similar in 2023 and 2024.
- **Timber recode confirmed** (D-24 kept at the owner's instruction, given it is correct): 16,954 of the 17,621 policies coded `T` in 2022 (96%) are `W` in 2023. Between other pairs of years, 98.2% of policies keep their code (294,312 of 299,771).
- **Review method:** a workflow of 26 agents. It ran no pack runner, so nothing was added to `run_log.jsonl`.
  - **Measurement tool:** row-level 2023 and 2024 backtest predictions of the production model without CANN were exported to scratch; they reproduce the backtest D² exactly. A probe fitted a credibility-shrunk correction per band on one year and scored it on the other. Calibration: a modelled factor gives −0.0012 / −0.0007, the contacts leak +0.37 / +0.37, written premium −0.0038 / +0.0011.
  - **Finders:** eight lenses (target and development, unused features, interactions, structure and fitting, level and calibration, customer history, test set and scoring, team and literature). Most measured their candidates by rebuilding the production model in memory, which reproduces the backtest to 1e-8 or better.
  - **Merge:** an agent removed duplicates and anything already tested; 24 candidates became 14, and 12 were dropped with reasons (`ev_r2`).
  - **Verification:** one skeptic per candidate re-measured it independently and tried to refute it on novelty, leaks, single-claim artefacts, seed noise and selection. 9 of 14 survived.
  - **Completeness critic:** it ran numeric heaping, unit-error, drift and sum-insured scans and added two data errors, both verified.
- **Findings** (`ev_r1_review_verdicts.csv`; D² change on 2023 / 2024; none applied):
  - **R11 elevation recorded in feet** (#154): buildings −0.0001 / +0.0008.
  - **`year_built = 1900` placeholder** (#155): +0.0003 / +0.0007; two-year mean +0.00051 (95% +0.00007 to +0.00086).
  - **The bodily-injury lift D-11 looks wrong** (#156): closed claims fell as much as open ones in 2023. Removing it lowers motor's 2025 level by about 5–8%, consistent with the leaderboard reading.
    - Documentation correction: the applied factors are 1.0854 / 1.2149 / 1.1631 (2022–24). The model documentation still quoted 1.076 for 2022, the value before de-duplication; it is corrected.
  - **Motor own-line credibility k = 4** (#157): −0.0002 / +0.0013 on the full target, where 2023 is driven by one 15.25m claim; capped at 1m, +0.0007 / +0.0012.
  - **Buildings peril interactions** (clay × dry summer, flood zone 3+ × rainfall; #158): large-claim GLM +0.0004 / +0.0006, two-year mean +0.00051 (95% +0.00022 to +0.00078); XGBoost +0.0012 / +0.0008.
  - **Buildings calibration power** (+0.00045 / +0.00056) and **short part-year rows** (+0.00047 / +0.00067) (#158).
  - **Contents direct-channel flag** (#159): +0.00056 / +0.00069, borderline.
  - **Net open motor own-damage and theft claims of expected recoveries** (#160): a correction to the target definition.
- **Rejected by the skeptics:**
  - motor large-claim GLM at power 1.9: one claim each year carries the gain;
  - buildings log knots: a selected peak;
  - more XGBoost trees and seed bagging: seed-specific;
  - count-weighted bodily-injury pattern: below the noise bar;
  - Bornhuetter–Ferguson by loss quarter: the 11.9m claim carries the gain.
- **Report:** `docs/FINAL_REPORT.html`, also published as a private artifact. It has charts of:
  - the pipeline;
  - backtest D² by stage;
  - the leaderboard;
  - every tested idea's 2023 / 2024 effect, with its outcome;
  - the development backtests and the uncertainty;
  - the 2025 level against the benchmarks;
  - seasonality;
  - the final review's candidates;
  - what was not tested and why.

  Its data come from the evidence files.
- **Issues:** #154–#160 (open), #161 (closed, this review).
- **Status:** analysis only. The model and submission are unchanged; the recommended final run awaits the owner's sign-off.

### A28 — Final run: the final review's items 1–6, tested and combined into a final candidate
- **When / who:** 2026-10-23, Claude, at the owner's instruction: "Go ahead with the recommended final run (items 1-6 in the report), each tested under the usual rules, then produce a final candidate submission."
- **Implementation:** every change has its own switch, so each can be turned off.
  - **D-26 and D-27, the address corrections:** a view `addresses_clean` in `sql/03`, used by `sql/03` and `sql/11`. It converts R11 elevation from feet and imputes year built 1900 from energy rating, keeping `elevation_m_raw` and `year_built_raw`.
  - **Buildings interactions:** `x_clay_dry` and `x_fz3_rain` in `sql/11`. They are used through `lines.models.buildings.numeric` and `glm_lines.models.buildings.excess.continuous`; the large-claim GLM now accepts continuous columns.
  - **Contents flag:** `f_direct` in `sql/08`.
  - **Motor credibility:** `glm_lines.credibility_tuning_cap` caps the target when tuning k.
  - **Prediction shape:** the step in `run_glm_lines.py` (`shape_adjust`, `fit_shape`; switch `glm_lines.posthoc`) applies a calibration power and a short-row factor. Each is estimated on one backtest year and scored on the other; 2025 uses the average.
  - **D-11:** switched off.
  - **Tests:** 51 pass. New tests cover the address view with its switch on and off, `shape_adjust` and `fit_shape`.
- **Tests:** `python/final_tests.py`, three labelled processes ("A28 final run: T1", "T2-T3", "T4-T6"); evidence `ev_f1`–`ev_f6`. The base is the production model without CANN. Buildings averages four XGBoost seeds with the tree count fixed. The SQL behind all three runs: `sql/03` df1d0989…, `sql/08` 73c3ee8b…, `sql/11` 52229dc9… (sha256 prefixes). These records lack the SQL hashes, a defect since fixed (below). Gains are D² change on 2023 / 2024.
  - **T1, address corrections:**
    - Buildings, both corrections: +0.00020 / +0.00131.
    - R11 elevation only: +0.00020 / +0.00061.
    - Year built only: +0.00046 / +0.00046, two-year mean significant (+0.00006 to +0.00086).
    - Contents: +0.00031 / +0.00017. Motor: no change (it uses neither field).
    - **Applied as corrections.**
  - **T2, buildings peril interactions:**
    - XGBoost only: +0.00105 / +0.00091, two-year mean +0.00098 (+0.00025 to +0.00171).
    - Large-claim GLM only: +0.00023 / +0.00045 (+0.00009 to +0.00059).
    - **Both: +0.00110 / +0.00121, two-year mean +0.00115 (+0.00030 to +0.00200). Adopted (D-29).**
  - **T3, buildings prediction shape** (fitted power 1.04 / 1.07, factor 0.48 / 0.42; 7,508 / 7,402 short rows):
    - Calibration power: +0.00003 / +0.00026, two-year mean +0.00015 (−0.00027 to +0.00057). **Fails significance.**
    - Short-row factor: +0.00047 / +0.00063, two-year mean +0.00055 (+0.000002 to +0.00111). **Passes, just.**
    - Both: +0.00047 / +0.00089 (−0.00004 to +0.00140). Fails.
    - Configured as the short-row factor only (`fit_power` false); the production step then re-tests it on the final model (below).
  - **T4, motor credibility:** tuned on 2023 capped at 1m, it chooses k = 4.
    - Against k = 8: −0.00017 / +0.00131 on the full target. Capped at 1m: +0.00066 / +0.00122. Without rows of 10m or more: +0.00072 / +0.00131.
    - k = 6: +0.00004 / +0.00057.
    - **The pre-registered rule takes k = 4 (D-30).**
  - **T5, contents direct-channel flag:** +0.00055 / +0.00069, two-year mean +0.00062 (−0.00004 to +0.00128). **Fails the significance rule; not adopted.**
  - **T6, D-11 off:** against the actual with the lift, −0.00118 / −0.00046; against the actual without it, +0.00045 / +0.00124. Each model wins on its own yardstick, so the backtest cannot settle it. **Adopted (D-28)** on the structural evidence (closed claims fell as much as open ones) and the leaderboard evidence (motor about 7% high).
- **Independent code review** (a 12-agent workflow: three reviewers, then a skeptic per finding). Two defects were confirmed and fixed:
  - **Low:** the "no A17 changes" variant would have dropped the large-claim interactions. It now keeps them (`Line.build`).
  - **Medium:** `final_tests.py` did not record the SQL hashes in its run records. It now merges what `prepare()` writes (`PROVENANCE`) and rejects unknown test names. The hashes for the three runs above are recorded in this entry.

  Seven other findings were refuted with reasons: the cap reproduces T4; the order of shape and tilt; ev_g8 coverage; T4 deployability; T2's tree count; T3's fitting; unknown test names, which is now fixed anyway.
- **Development uncertainty rerun** with D-11 off (labelled): motor's development factor falls from 1.1231 to 1.0294 for 2023 and from 1.3032 to 1.2361 for 2024. Motor's bootstrap CV falls from 4.31% to 0.42% (2023) and from 2.96% to 1.28% (2024). Mack is unchanged, since it uses reported claims.
- **GLM pack rerun** (the contents SQL changed): reconciliation 101 checks, 0 failed. GLM A 2024 D² moves from 0.0933 to 0.0934.
- **Final run** ("A28 final run: final candidate"):
  - **Buildings:** 194 trees, trend +5.3% a year. CANN is no longer significant (+0.00001) and is dropped. The short-row factor narrowly fails on the final model (two-year mean +0.00056, −0.00002 to +0.00114) and is not applied.
  - **Buildings level factor adopted** (×1.0334): k = 1.031 on 2023 and 1.036 on 2024, each improving the other year.
  - **Contents:** unchanged (k = 2, trend +4.2%, no CANN).
  - **Motor:** k_own = 4 and trend +3.8% a year (was +5.5% with the lift); CANN not adopted; uncertainty tilt adopted (×1.0054). Predicted / developed 2024 is now 1.005 (was 1.053).
- **2024 backtest D²:**
  - buildings **0.1138** (was 0.1107);
  - contents **0.0969** (was 0.0968);
  - motor **0.0832**. Motor is not comparable with 0.0806, because the actual changed with D-11.
- **2025 rates (90% range):** buildings 500.3 (437–573), contents 129.2 (117–142), motor **581.3** (490–690; was 627.4).
- **Submission:** `work/submissions/submission_final_candidate_20261023.csv`. 626,639 rows in sample order, no nulls, minimum 0.050.
  - Mean per row, against the best-scoring A22 file: buildings 408.2 (×1.018, correlation 0.977), contents 103.4 (×1.000), **motor 455.9 (×0.927)**.
  - The motor level lands on the ×0.93 that the leaderboard pair implied (A27), without being tuned to it.
- **Not adopted:** the contents direct-channel flag; the buildings calibration power; the short-row factor, on the final model's own test; CANN on every line.
- **Status:** final candidate ready. D-26 to D-30 were adopted on the owner's go-ahead and await formal sign-off. The public leaderboard is the independent check.
- **Leaderboard result** (reported by the owner, 23 October 2026): the final candidate scored **0.12805**, the best public score so far, against 0.12753 for the A22 file. The leader was still at 0.13050. The gain of +0.00052 is consistent in sign with the backtest gains and with the motor level the A27 pair implied, but a single public score on part of the test set cannot attribute it to any one change. Recorded in `ev_r4_leaderboard_scores.csv`.

### A29 — Statistical tests of the final model: Type III, chi-square, level merging and interactions
- **When / who:** 2026-10-23, Claude, at the owner's instruction: "yes commit and push the latest version and result the do all three steps and see is there any further improvement". The owner had asked whether the GLMs had been through formal hypothesis tests (p-values, chi-square tests, grouping levels, interactions); the three steps were Type III and chi-square tests, level merging tested out of time, and a pairwise interaction screen.
- **Harness:** `python/stat_tests.py`, tests `tests/test_stat_tests.py` (21). The adoption rules are fixed in the module docstring and were set before the labelled runs (one amendment, below). Evidence `ev_s1_*`, `ev_s2_*`, `ev_s3_*` in `audit/evidence/glm_lines/`.
  - **S1, documentation:** Type III tests of every term of every GLM component of the final model (fitted 2021–24): a quasi-likelihood F test on refits (Pearson dispersion, df the term's estimable columns) and a score test at the reduced fit (the smaller of the cluster-robust and the model-based statistic with the sandwich's effective dispersion). Then chi-square tests of actual against expected by level of every model factor on both backtest years.
  - **S2, simplification, scored out of time:** greedy level merging within each fitting window (Wald statistic for equal coefficients below 2, Takeuchi's criterion; aliased merges free), and dropping each term in turn.
  - **S3, interactions:** a score-test screen of every pair of terms in each GLM, and of buildings' XGBoost on its own residuals, on 2021–22 only; the three strongest pairs per model scored out of time.
  - **Adoption (S2, S3):** both backtest years improve and the two-year mean gain is significant (D-23); for dropped terms and interactions also Holm across the candidates scored for a model.
- **Checks on the tests themselves** (eight findings confirmed across the two reviews, seven distinct defects):
  - The Type III F test matches R's `glm` with `statmod` to a relative 1e-4 on simulated, unpenalised data (unit test, run).
  - **First independent review** (13 agents; six findings confirmed, four refuted). Fixed: the score-test guard used the penalised projection, so columns already in the model kept degrees of freedom at alpha 1e-3; a sandwich statistic collapses where a level has few or no claims (spuriously significant, and such levels never merged); a trend interaction on motor's small-claim GLM would have moved the large-claim trend; three tests too weak to catch regressions.
  - **Diagnostic on the real fits** (unlogged). Some production factors are **exactly aliased**: motor marketing segment's budget and mainstream levels with credit band, contents segment with contents credit band, and one direction of motor's voluntary-excess spline. Only the small ridge penalty identifies those coefficients, so per-coefficient Wald tests are meaningless there; predictions are unaffected. The effective dispersion the sandwich implies is 0.6 to 1.1 times the Pearson one, so the Pearson F tests are about right; the deviance-based dispersion (44 to 111) is far too small. An earlier reading in this session, that the Pearson dispersion was inflated many times, was wrong and is corrected here. A check of how the residual variance scales with exposure did not support changing the variance function, so the A/E tests keep the models' own.
  - **Redesign**, then a **second independent review** (5 agents; two findings confirmed, the same defect): merges of aliased levels were tested on a ridge-dependent projection, so they never merged. Each merge pass now re-expresses the model in the merges made so far; a non-estimable merge is free.
  - **Smoke runs** (unlogged) exercised all three tests on the real data before the labelled runs.
  - **Amendment before the labelled S3 run:** the rule was "Benjamini-Hochberg at 5%, then up to three survivors per model". A positive control showed that the screen cannot detect even the A28 peril interactions, which gain about 0.001 out of time: refitting buildings' XGBoost without them and screening the two pairs they came from gives p = 0.41 and 0.53 on the fitting years and, in an unlogged check, 0.03 and 0.05 on 2023 residuals, against a BH bar of about 3e-5. So the three strongest pairs per model are scored regardless of BH, with Holm at the scoring step. The fitting-year control is part of the labelled run (`ev_s3_power_check.csv`).
- **Runs** ("A29 statistical tests: S1 …", "S2 …", "S3 …"); the S1 and S2 records carry the harness before the S3 amendment, committed separately.
- **S1, Type III** (`ev_s1_type3.csv`, `ev_s1_component_fits.csv`):

  | Component | Rows | Columns (rank) | Pearson φ | Effective φ | Terms significant on both tests |
  |---|---|---|---|---|---|
  | Buildings large-claim GLM | 707,726 | 35 (35) | 15,033 | 13,160 | 4 of 9 |
  | Contents GLM | 721,617 | 79 (77) | 1,612 | 965 | 16 of 21 |
  | Motor small-claim GLM | 979,968 | 83 (80) | 1,077 | 896 | 18 of 19 |
  | Motor large-claim GLM | 979,968 | 28 (28) | 54,631 | 59,072 | 1 of 5 |

  - Not significant on either test (F p / score p): buildings large-claim flood zone (0.62 / 0.55), soil type (0.41 / 0.40), construction type (0.11 / 0.15), distance to river (0.14 / 0.05), x_fz3_rain (0.21 / 0.19); contents segment (0.97 / 0.88), bike cover (0.82 / 0.76), lodgers (0.19 / 0.09); motor marketing segment (1.00 / 1.00); motor large-claim driver age (0.93 / 0.95), years licensed (0.94 / 0.96), region (0.06 / 0.15).
  - Significant on one test only: contents policyholder age (0.15 / 0.03) and distance to river (0.28 / 0.050); motor large-claim annual mileage (< 0.001 / 0.05).
  - Segment adds nothing beyond credit band in either line: 9 of its 11 columns are estimable, and they are not significant.
- **S1, calibration by level** (`ev_s1_ae_chisq.csv`, `ev_s1_ae_by_level.csv`): 186 factor-years; 16 have p < 0.05 (about 9 expected by chance), 2 survive Benjamini-Hochberg within line and year: buildings basement in 2023 (A/E 1.33 for basements; 1.08 in 2024) and motor third-party-only cover in 2023 (A/E 2.25, the large claim; 1.04 in 2024); neither is significant in 2024 (p 0.31 and 0.75). Only one factor has p < 0.05 in both years, buildings distance to river (0.039 and 0.016; q 0.30 and 0.69, so not after Benjamini-Hochberg), and its deviations differ between the years. Within it, **buildings rows with no distance recorded** are under-predicted in both years (A/E 1.61 and 1.20, 170 and 155 claims). Found on the backtest years, so it is raised for a test on fresh data rather than adopted.
- **S2, simplification** (`ev_s2_level_merges.csv`, `ev_s2_merge_two_year.csv`, `ev_s2_drop_term_two_year.csv`):
  - **Level merging:** 9 to 33 merges per component and window, including 2 free merges each for contents and motor's small-claim GLM (their aliasing). Out of time: buildings +0.00034 / −0.00010, contents −0.00056 / −0.00019, motor −0.00025 / +0.00034. **Fails on every line.**
  - **Dropping terms:** 54 terms, none adopted. Two improve both years without significance: contents segment (+0.00013 / +0.00020; two-year mean +0.00017, −0.00029 to +0.00062) and motor payment method (+0.00130 / +0.00006; +0.00068, −0.00067 to +0.00203). Payment method is strongly significant in-sample (F 58.9) yet costs nothing when dropped, which suggests its relativity is not stable over time. Conversely, x_fz3_rain is not significant in-sample in the large-claim GLM, yet dropping it costs significantly out of time (−0.00011, −0.00022 to −0.00001). In-sample p-values do not decide predictive value.
- **S3, interactions** (`ev_s3_interaction_screen.csv`, `ev_s3_interaction_two_year.csv`, `ev_s3_power_check.csv`):
  - Screened (2021–22): buildings large-claim GLM 34 pairs, buildings XGBoost residuals 1,479, contents 210, motor small-claim 171, motor large-claim 10; 8 pairs that repeat the A28 features recorded but not screened. **No pair passes Benjamini–Hochberg** (smallest q 0.16, contents). Share of p below 0.05: 3%, 3%, 4%, 11% and 0%.
  - **Positive control** (in the labelled run): buildings' XGBoost without the A28 features gives p = 0.41 (soil type × dry summer) and 0.53 (flood zone × rainfall).
  - **Scored out of time:** 15 pairs, the three strongest per model. **None adopted.** One passes the two-year rule but not significance: buildings XGBoost email domain × branch, +0.00004 / +0.00033, two-year mean +0.00018 (−0.00014, +0.00050). The rest fail, and motor's small-claim candidates make things worse: cover type × credit band −0.00327 / −0.00058 (both years' intervals exclude zero), marketing segment × cover type −0.00667 / −0.00025, NCB × trend +0.00053 / −0.00087. Buildings large-claim: soil type × year built −0.00062 / −0.00026, soil type × sum insured −0.00028 / −0.00009, distance to river × x_clay_dry −0.00052 / −0.00001. XGBoost: energy rating × solar panels +0.00036 / −0.00003, wildfire index × basement +0.00058 / −0.00001. Contents: flood zone × bike +0.00023 / −0.00018, flood zone × property type +0.00046 / −0.00056, voluntary excess × elevation −0.00040 / +0.00047. Motor large-claim: NCB × years licensed −0.00038 / +0.00095, mileage × region −0.00011 / +0.00095, NCB × driver age −0.00045 / +0.00085.
- **Status:** **no change to the model; the A28 final candidate (public 0.12805) stands.** Issues [#163](https://github.com/almac09/iaa-hackathon-2026/issues/163) (buildings rows with no distance to river) and [#164](https://github.com/almac09/iaa-hackathon-2026/issues/164) (motor payment method not stable out of time) are open for follow-up. The lesson for further work: the gains that mattered came from reasoning about how claims arise, which no screen of this kind could have found, so new information is the more promising route than re-specifying the GLMs.
