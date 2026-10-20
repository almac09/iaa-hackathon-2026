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
