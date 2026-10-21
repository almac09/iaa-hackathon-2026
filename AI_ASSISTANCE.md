# AI Assistance Log — IAA Hackathon 2026

This file documents all AI-assisted contributions to this project, maintained using Databricks Genie Code (AI assistant).

## Format

Each entry records:
- **Date** — when the assistance occurred
- **Task** — what was being worked on
- **AI Contribution** — what the AI assistant did
- **Human Contribution** — what the author directed, reviewed, or modified

## Log

### 2026-10-20 — Project Setup
- **Task**: Connect the GitHub repo to Databricks workspace for version control
- **AI Contribution**: Cloned `almac09/iaa-hackathon-2026` into the Databricks workspace at `/Workspace/Users/bridget.macdonnell1@gmail.com/hackathon-project`; explained Git branching and team sync workflows
- **Human Contribution**: Set up GitHub credential via Linked Accounts, provided repo URL and workspace path, directed the setup
- **Files affected**: Repo clone (no code changes yet)

### 2026-10-20 — Fix seaborn import error in data_import_Bridget notebook
- **Task**: Resolve ModuleNotFoundError when importing seaborn
- **AI Contribution**: Diagnosed that seaborn is not pre-installed in the serverless compute environment; added a `%pip install seaborn` step at the top of the imports cell and verified successful execution
- **Human Contribution**: Reported the failing cell and requested the fix
- **Files affected**: `notebooks/data_import_Bridget` (Cell 2)

### 2026-10-20 — Load parquet datasets & export Excel files
- **Task**: Load all parquet datasets from the repo's data folder, create an overview Excel workbook (first 100 rows per dataset with total row counts), and export full datasets to Excel where size permits
- **AI Contribution**: Located the 11 parquet files in the data folder, wrote code to loop through and load each into its own Spark DataFrame (df_claims, df_customers, etc.), built Data_overview_100_rows.xlsx with one sheet per dataset, and exported full Excel files with a 250,000-row safety threshold (8 of 11 datasets exceeded it and were skipped)
- **Human Contribution**: Requested the loading and Excel export workflow, provided the folder location and naming requirements
- **Files affected**: `notebooks/data_import_Bridget` (Cells 2, 5, 6, 7), new `Excel files/` folder with 4 workbooks

### 2026-10-20 — Build Data_Summary.xlsx workbook
- **Task**: Create a summary spreadsheet with dataframe name, shape, describe statistics, and missing-value information per dataset
- **AI Contribution**: Added a cell that computes per-column dtypes, missing counts/percentages and numeric describe stats (count/mean/stddev/min/max) for all 11 datasets, with an Overview sheet plus one detail sheet per dataset
- **Human Contribution**: Requested the summary workbook and its contents
- **Files affected**: `notebooks/data_import_Bridget` (Cell 8), new `Excel files/Data_Summary.xlsx`

## Human in the Loop — Review Checklist

The following steps require human review and decision-making (not automated). Status: items 1–3 **completed** 2026-10-20 (review notes below); item 4 **pending** — awaiting human review of the master policy files.

1. **Review the Excel files created** — Open `Excel files/Data_overview_100_rows.xlsx`, `claims.xlsx`, and `Data_Summary.xlsx` to get a feel for the data: column meanings, distributions, and missing-value patterns (notably the 10.32% missing cells in `claims` and 3.4% in `customers`).
2. **Check for new files** — Confirm with the team whether any new data files have been received (e.g. new Kaggle uploads or teammate additions to the `data/` folder). New files are picked up automatically by re-running the loading cell in `notebooks/data_import_Bridget`.
3. **Review overall approach** — Based on the data reviewed, decide whether the current plan needs to change: which datasets to join for modeling, how to handle missing values in `claims`/`customers`, and whether the target/feature set still makes sense for the competition.
4. **Review the master policy files in Excel** (PENDING) — Open `Excel files/Master_Data_overview_100_rows.xlsx` (one sheet per line: motor, contents, buildings) and the supporting exports (`policies_*_master_first_100_rows.xlsx`, `earned_premium_examples.xlsx`, plus the claims-field examples in Cell 13 of the notebook) and check the masters are as expected: the joined columns (customer, address, weather and aggregated claims fields) look correct; the renewed-only filter is sensible; and the new fields behave as intended — `claim_indicator`, `n_claims`, `incurred_net_of_recoveries`, `ultimate_claims`, and the pro-rated `earned_premium` (incl. the flagged assumption that `written_premium` is annualized).

## Human Review Notes — 2026-10-20

Recorded after reviewing the Excel workbooks. These are thoughts only — nothing actioned yet.

**Data quality**
- A lot of missing data — for example the energy rating column in `addresses`
- Possible class imbalance — e.g. almost no listed buildings in the addresses file

**Claims development (target construction)**
- We have incurred-to-date claims data but no development patterns to use or obtain — all claims data is as at 31/12/2024 and not tracked year on year
- Will need to develop claims to ultimate using an external benchmark, or using expected loss ratios from earlier years
- Assume all closed claims are ultimates, especially for earlier years (2021–2022) which are more developed
- Note: most claims in the dataset are closed

**Feature thinking**
- A lot of the fields don't look predictive — e.g. email domain
- Open question: is it ethical to include number of customer service contacts?
- Weather probably matters more for home than motor — except possibly the hail peril for motor

**Modelling approach**
- Need to decide whether we model by peril or just by line of business
- Can model altogether or separately as frequency and severity, for explainability purposes
- Target for severity looks like ultimate claims — can use incurred-to-date less recoveries for now, until claims are more developed
- Target for frequency would need a claims flag added to the policies files — consider how to add up multiple claims on one policy row

### 2026-10-20 — Create master policy files (motor, contents, buildings)
- **Task**: Build policies_motor_master, policies_contents_master and policies_buildings_master by joining customers, addresses, weather (year-matched to period_start) and claims (line-matched, loss year matched to period year, aggregated per policy) onto each policies file
- **AI Contribution**: Inspected schemas and join-key uniqueness first; discovered addresses has no customer_id (joined on address_id instead), weather is unique per (address_id, year) 2021-2025, and policy_id repeats across years so claims were pre-aggregated per (policy_id, loss year) to prevent fan-out; wrote and ran the join code with sanity checks (all masters match their policy file row counts exactly)
- **Human Contribution**: Specified the join keys, year-matching rules and multiple-claims aggregation rules (sum paid/incurred/recoveries, peril="multiple" when mixed, first claim for other fields, claim count noted)
- **Files affected**: `notebooks/data_import_Bridget` (Cell 10)
- **Follow-up (2026-10-20)**: Added `claim_indicator` (1 if `n_claims` ≥ 1, else 0) to all three master files; verified the indicator totals exactly match the claims-matched counts in each file.
- **Human verification (completed 2026-10-20)**: Checked the master policy file creation has been done correctly — the output files look OK, the `n_claims` column was spot-checked, and confirmed no files were lost in the join/export process. Includes review of the first-100-row Excel exports of the three master files (Cell 11).

### 2026-10-20 — Add earned_premium to master files
- **Task**: Add an earned_premium field to each master dataframe for future loss-ratio work — equal to written_premium when the period sits within one calendar year, otherwise pro-rata by days in force during the start year (days to 31 Dec inclusive / total days inclusive, times written_premium, rounded to 2dp)
- **AI Contribution**: Implemented the calculation in the master build with sanity checks (zero nulls across all three files); discovered that every policy in the data sits within a single calendar year, so the pro-rata branch never fires on real rows — demonstrated it instead on two synthetic test rows (500.00 x 365/730 = 250.00; 500.00 x 184/365 = 252.05); exported verification examples to `Excel files/earned_premium_examples.xlsx` and refreshed the first-100-row master exports
- **Human Contribution**: Specified the earned premium rules (same-year vs cross-year pro-rata) and requested worked examples to hand-check
- **Files affected**: `notebooks/data_import_Bridget` (Cells 10, 11, 12), new `Excel files/earned_premium_examples.xlsx`

### 2026-10-20 — Investigate M53731187 (single-day policy)
- **Task**: Investigate why M53731187 carried a 433.46 premium for a 1-day period (user suspected a cancelled policy)
- **AI Contribution**: Confirmed the hypothesis — the 2022 row is a renewal that lapsed after one day (`renewal_decision` = "lapsed_by_customer", exposure = 0.0027 = 1/365); the policy's one claim (2021, third-party property, 979.32 paid, closed) attaches to the 2021 row correctly. Broader findings: `written_premium` is always the annualized quote (median ~480 across every period-length bucket from 1 day to 365); 795 motor policies are single-day and ~37% run shorter than 365 days; the `exposure` column has data-quality issues (3,991 motor rows > 1, mixed scales — some look like days, not years)
- **Human Contribution**: Flagged the anomaly in the earned_premium examples and hypothesised cancellation
- **Files affected**: none (ad-hoc exploration via chat)

### 2026-10-20 — Filter master files to renewed policies only (HUMAN DECISION)
- **Task**: Report the proportion of policies with `renewal_decision` = "renewed" and keep only those policies in the master files
- **AI Contribution**: Measured the proportions (motor 80.96%, contents 83.02%, buildings 84.24%; remainder split between lapsed_by_customer, mid_term_change, cancelled, declined_by_insurer); implemented the renewed-only filter in the master build with updated sanity checks (master rows match renewed policy rows exactly in all three lines) and refreshed the Excel exports. **Flagged for revisit**: ~20-23% of renewed policies still run shorter than a full year (motor: 184,595 of 793,414 renewed rows, including 16,375 under 30 days) and `written_premium` is the annualized quote regardless of period length, so `earned_premium` = `written_premium` still overstates the premium base for those rows (e.g. 2-day renewed policies earn a full annual premium under the current rule)
- **Human Contribution**: Decision to keep only renewed policies in the masters (noted as revisitable if needed); working assumption that only lapsed policies were affected by the earned premium issue — partially incorrect, see flag above
- **Files affected**: `notebooks/data_import_Bridget` (Cells 10, 11, 12)

### 2026-10-20 — Pro-rate earned_premium by days in force (DECISION / ASSUMPTION)
- **Task**: Change earned_premium so the premium base reflects actual exposure instead of treating same-calendar-year policies as fully earned
- **Decision / Assumption (human-approved, revisit if organisers clarify)**: `written_premium` is treated as the ANNUAL premium under the insurer's current pricing. The competition documentation is ambiguous ("Premium charged under the insurer's current pricing"), so this interpretation is data-driven, not confirmed: correlation between premium and days in force is ~0 in all three lines (motor -0.042, contents -0.012, buildings 0.003); 59,333 same-policy pairs show a short period (avg 104 days) priced at 99.6% of the following full year's premium; single-day policies carry ~100% of the annual median premium
- **AI Contribution**: Implemented earned_premium = written_premium x days_in_force / days_in_calendar_year (inclusive day counts; 366-day basis for the 2024 leap year; days capped at the calendar year end for the hypothetical cross-year case); extended sanity checks — zero nulls and earned never exceeds written in any line; refreshed verification examples (incl. leap-year and 90-day synthetic tests) and the Excel exports
- **Human Contribution**: Approved applying the fix and requested the interpretation be flagged as a decision/assumption in this log
- **Files affected**: `notebooks/data_import_Bridget` (Cells 10, 11, 12), `Excel files/earned_premium_examples.xlsx` and first-100-row exports refreshed

### 2026-10-20 — Add claims amount fields to master files
- **Task**: Add `incurred_net_of_recoveries` and `ultimate_claims` fields alongside the aggregated claims fields in each master file
- **AI Contribution**: Both fields = incurred_to_date − recoveries, computed on the per-policy-year aggregated sums (null for policy-years with no claims); sanity checks confirm the ultimate_claims totals match incurred minus recoveries exactly in all three lines (motor 261.1M, contents 46.6M, buildings 111.1M); added a verification cell showing single- and multi-claim rows (sums apply correctly); refreshed Excel exports. Note: compute had been terminated by inactivity, so the prerequisite cells (library install, data load, Excel folder setup) were re-run first
- **Human Contribution**: Specified both fields; ultimate_claims is deliberately a placeholder (incurred net of recoveries) that may be revised later once claims are developed to ultimate
- **Files affected**: `notebooks/data_import_Bridget` (Cells 10, 11, new Cell 13)
- **Follow-up (2026-10-20)**: Added `Master_Data_overview_100_rows.xlsx` to the Excel files folder — one sheet per master line (motor, contents, buildings) with the first 100 rows and a total-row summary line, mirroring the layout of `Data_overview_100_rows.xlsx` (new Cell 14).

### 2026-10-20 — Motor feature selection (HUMAN DECISION) & modeling dataset
- **Task**: Select features for the motor Tweedie XGBoost (expert-judgement first pass, to be revisited after EDA) and build the modeling dataset
- **Human Contribution**: Supplied the full include/exclude list — 41 features including a derived `Year` (from period_start); excluded vehicle_colour, discount_applied, customer_service_contacts (ethical grounds), renewal_decision (not available in test files), all property-risk attributes and non-hail weather (motor-irrelevant), and all claims-derived columns (leakage); chose to keep exposure, written_premium and earned_premium as features
- **AI Contribution**: Implemented the feature list in new Cell 16; derived `Year`; converted target nulls to 0 for claim-free rows; kept `customer_id` for GroupKFold grouping and `policy_id`/`row_id` for prediction tracking; verified every listed column exists in the master; profiled the assembled dataset (missingness left as-is for XGBoost: telematics_score 87.6%; customer-block features ~10% reflecting the customer join gap; vehicle fields ~2.5%)
- **Files affected**: `notebooks/data_import_Bridget` (Cells 5, 10 re-run after compute restart; new Cell 16)

### 2026-10-20 — Motor Tweedie XGBoost: first GroupKFold CV model
- **Task**: Drop earned_premium as a motor feature (keep annualized written_premium) and build and train the first Tweedie XGBoost for motor ultimate claims
- **Human Contribution**: Dropped earned_premium from the feature list; noted the hackathon scores with Tweedie power 1.5
- **AI Contribution**: Added Cells 17–21: pip install xgboost (3.4.1); pandas prep (15 categoricals kept as category dtype, 4 booleans → int); 4-fold GroupKFold (grouped by customer_id) XGBoost with objective reg:tweedie and tweedie_variance_power FIXED at 1.5 to match the hackathon scoring, exposure as sample weights; evaluation with mean Tweedie unit deviance at p=1.5 computed directly (the scoring loss form — XGBoost's built-in metric is a log-likelihood, not deviance); out-of-fold decile calibration, feature importance (gain); results exported to `Excel files/motor_tweedie_xgb_results.xlsx`
- **Results**: mean deviance 12,029 across folds (fold range 3,511–16,136 — the spread is driven by a few very large claims landing in different folds); deciles perfectly monotonic on actual claims, claim rate rising 3.8% → 22.9% from decile 1 to 10, top-decile lift 2.30x; top features by gain: exposure 23.0%, telematics_opt_in 6.7%, ncb_years 5.8%, make_model 4.6%, transmission 3.2%
- **Observations for next iteration**: OOF predictions under-predict the unweighted mean (≈265 vs ≈329) — consistent with exposure weighting plus the known exposure column issues (values > 1); exposure's top importance is partly mechanical (longer time in force → more chance of a claim). Both points feed the planned EDA/exposure cleanup
- **Files affected**: `notebooks/data_import_Bridget` (Cells 16–21), `Excel files/motor_tweedie_xgb_results.xlsx`

### 2026-10-20 — Test-file scoring & first motor submission (v001)
- **Task**: Refit the motor Tweedie XGBoost on 100% of the training rows, score the 2025 test files, and produce the first submission CSV (EDA/exposure cleanup and hyperparameter tuning deliberately deferred)
- **Human Contribution**: Confirmed the submission format (sample_submission skeleton matched on row_id, prediction = predicted ultimate_claims, motor M / buildings B / contents C prefixes); instructed that buildings/contents rows and any zero predictions use a dummy value of 10 because the scorer does not accept zeros
- **AI Contribution**: Built the motor test master mirroring the training joins (weather + customers + addresses; no claims join possible; NO renewal filter — renewal_decision absent from test files, so every test row is scored and the renewed-only training population mismatch stands); aligned categorical vocabularies with training (unseen levels → NaN; **none found** — perfect feature alignment); refit the final model on all 793,414 training rows; assembled the submission on the sample_submission skeleton with 2-dp rounding and the dummy-10 rule
- **Results**: 626,639 rows (M 254,986 / C 187,498 / B 184,155); motor mean prediction 267.39 (consistent with the OOF mean ~264), min 3.78, max 9,576.22; 371,653 dummy rows (B/C lines, no model yet); 1 motor zero-prediction replaced by the dummy; file verified on disk (header + 626,639 data rows)
- **Files affected**: `notebooks/data_import_Bridget` (Cells 22–24), `submissions/motor_tweedie_xgb_submission_v001.csv` (new file, not yet committed)

### 2026-10-20 — Submission v002: motor + teammate buildings
- **Task**: Combine the motor predictions (v001) with the teammate's buildings predictions from `main`'s `home_insurance_submission_v001.csv` into a v2 submission, on the `hackathon-bridget` branch only
- **Human Contribution**: Instructed to copy the buildings predictions (B-prefix row_ids) from the team submission file into her submission as v2; branch-local only, no merge of main
- **AI Contribution**: Extracted the teammate's CSV via `git show origin/main:...` (no merge needed); merged on row_id (left join preserving her row order); contents rows kept at the dummy 10; applied the zero→10 rule (no zeros found); verified motor rows identical to v001, B rows identical to the teammate's values, 626,639 rows, all predictions positive
- **Results**: B predictions from the teammate file: min 1.13, mean 342.89, max 420.28 (a notably narrow range — worth reviewing their buildings model later); file: `submissions/motor_buildings_submission_v002.csv`
- **Files affected**: `submissions/motor_buildings_submission_v002.csv` (new)

### 2026-10-20 — Persist master files to workings/
- **Task**: Save the three master policy files (motor, contents, buildings) as single-file parquet in the repo's `workings/` folder on the `hackathon-bridget` branch, so future sessions can `spark.read.parquet` them directly instead of re-running the data load + master build (~5 min) after every compute restart
- **Human Contribution**: Requested the persistence into a Workings folder in her branch
- **AI Contribution**: New notebook cell (after Cell 14) writing each master via `toPandas().to_parquet` with a pandas read-back row-count verification; two environment quirks solved along the way — Spark Connect attaches plan metrics to `.attrs` which pyarrow cannot serialize (cleared before writing), and a freshly written `/Workspace` file takes a minute or more to become visible to `spark.read.parquet` (verification switched to pandas; Spark readability confirmed separately afterwards)
- **Results**: policies_motor_master 793,414 rows (61 MB), policies_contents_master 599,110 rows (39 MB), policies_buildings_master 596,186 rows (40 MB) — all round-trip verified via pandas AND Spark; all under GitHub's 100 MB per-file limit; raw `data/` folder untouched
- **Files affected**: `notebooks/Motor_Bridget` (new cell), `workings/policies_motor_master.parquet`, `workings/policies_contents_master.parquet`, `workings/policies_buildings_master.parquet`

### 2026-10-20 — Contents Tweedie XGBoost model & submission v001
- **Task**: Build the contents model in the new `Contents_Bridget` notebook per Bridget's feature list (1 = use, 0 = exclude), score `test_contents` (2025), and fill the C-prefix rows of the sample_submission skeleton
- **Human Contribution**: Provided the contents feature list (Year derived from period_start; target = ultimate_claims; written_premium, discount_applied and all claims-derived columns excluded; contents rows carry the C prefix in the submission)
- **Human Decision (flagged)**: the seven weather features were left BLANK in the provided list — they are INCLUDED (consistent with the motor model and directly property-relevant); revisit if that was not the intent
- **AI Contribution**: Mirrored the validated motor pipeline — contents master read from `workings/` (fast-start pattern, no rebuild), 50 features verified against the master (~2.5% missingness on joined customer/address features), 4-fold GroupKFold (grouped by customer_id) Tweedie XGBoost at p=1.5 with exposure sample weights, OOF decile evaluation, test master built with the same weather/customers/addresses joins (no unseen categorical levels; 2.5% join gap), refit on 100% of training rows and C rows filled into the skeleton (M/B rows keep the dummy 10)
- **Results**: CV mean Tweedie deviance (p=1.5) 7,652 (folds 3,385 / 6,600 / 7,446 / 13,177 — tail-driven spread); mean pred ~52 vs actual ~78 (same under-prediction pattern as motor); deciles 100% monotonic, top-decile lift 2.20x, claim rate 1.7% → 6.3%; top gain features: exposure (13.5%), solar_panels, flood_zone, occupation, voluntary_excess; submission 626,639 rows — 187,498 C rows model-scored (mean 57.49, min 0.64, max 2,977.81), 20 zero-preds replaced with the dummy, 439,141 M/B dummy rows; evaluation exported to `Excel files/contents_tweedie_xgb_results.xlsx`
- **Files affected**: `notebooks/Contents_Bridget` (cells 5–14), `submissions/contents_tweedie_xgb_submission_v001.csv` (new), `Excel files/contents_tweedie_xgb_results.xlsx` (new)

### 2026-10-20 — Combined submission v3 (motor + buildings + contents)
- **Task**: Assemble the full three-line submission — every row of the skeleton now carries a real model prediction (no more dummy slots)
- **Human Contribution**: Requested the combined v3 assembly
- **AI Contribution**: New section in `Contents_Bridget` — reads the three source files from disk (deterministic, restart-proof), merges by direct `row_id` matching: M rows ← motor model v1, B rows ← teammate's buildings model (carried from the v2 merge), C rows ← contents model v001; asserts each line against its source file plus full-skeleton checks (626,639 rows, no nulls, no zeros, unique row_id)
- **Results**: M 254,986 rows (mean 267.39, identical to motor v1) | C 187,498 rows (mean 57.49, identical to contents v001) | B 184,155 rows (min 1.13 / mean 342.89 / max 420.28, identical to v2)
- **Files affected**: `notebooks/Contents_Bridget` (cells 15–16), `submissions/motor_buildings_contents_submission_v003.csv` (new)

### 2026-10-21 — Buildings Tweedie XGBoost model & combined v4 submission
- **Task**: Mirror the contents pipeline for buildings (Buildings_Bridget notebook), score test_buildings, replace the teammate's buildings predictions
- **Human Contribution**: Requested the buildings pipeline; confirmed weather matching question (2025 test rows join 2025 weather on address_id + policy year)
- **AI Contribution**: New notebook `notebooks/Buildings_Bridget` — feature list derived from Bridget's contents list (same exclusions) plus buildings-specific features (construction_type, occupancy, days_unoccupied, smoke_alarms, mortgage); one buildings-only fix: nullable boolean smoke_alarms (2.5% nulls) converted to 0/1/NaN instead of categorical (XGBoost categorical path rejects boolean-typed categories); same 4-fold GroupKFold Tweedie XGBoost (p=1.5, exposure weights), OOF decile lift analysis, test master with weather join on address_id + policy year, refit on 100%, submissions v001 + combined v4
- **Results**: 49 features, 596,186 rows, 3.4% claim rate; CV mean deviance 776 (folds 665 / 2,216 / 119 / 105 — fold 2 tail); deciles 100% monotonic, top-decile lift 2.15x, claim rate 1.8% → 5.9%; top gain features: exposure (9.9%), flood_zone (6.1%), elevation_m, wind_exposure_index, occupation, construction_type, new_business; test 184,155 rows scored (mean 111.92, min 0.13, max 8,035.82), no unseen categorical levels, 2.5% customer join gap; v4: M 254,986 (mean 267.39) | C 187,498 (mean 57.49) | B 184,155 (mean 111.92) — all three lines now Bridget's own models
- **Files affected**: `notebooks/Buildings_Bridget` (new, 16 cells), `submissions/buildings_tweedie_xgb_submission_v001.csv` (new), `submissions/motor_buildings_contents_submission_v004.csv` (new), `Excel files/buildings_tweedie_xgb_results.xlsx` (new)

### 2026-10-21 — Buildings fold-2 deviance outlier investigation
- **Task**: Investigate why fold 2's CV deviance (2,216) was 20x the other folds — suspected a large claim
- **Human Contribution**: Requested the investigation without re-running training
- **AI Contribution**: Added a diagnostic cell (now Cell 12) that re-derives the deterministic GroupKFold split (no retraining) and recomputes per-row unit deviance at p=1.5 from the cached OOF predictions; verified the re-derived folds reproduce fold_summary exactly
- **Findings**: NOT a large claim — the 2,216 was 95.6% driven by 5 rows: moderate claims ($463–$11,012) sitting on policies with OOF predictions ≈ 1e-9 (Tweedie deviance explodes when a positive actual meets a near-zero prediction; worst row = unit deviance 134M). Excluding the top 10 rows, fold 2's mean deviance was 93.4 — the best fold. All folds carry balanced tails (414–418 claims >$10k each); the dataset's 15 largest claims ($220k–$630k) are spread across all folds. Fold 1's 665 had the same fragility (top-10 rows = 85.7% of the fold). Two root-cause data smells found in the culprit rows: (a) the exposure column mixes scales (547 buildings rows carry day counts like 365 instead of year fractions), and (b) construction_type has inconsistent casing — 11 raw levels that are really 5 (Brick/BRICK/brick, stone/STONE, t/T, w/W, non_standard/NON_STANDARD) — rare case variants route validation rows into near-zero-prediction leaves
- **Files affected**: `notebooks/Buildings_Bridget` (new Cell 12)

### 2026-10-21 — Buildings model fixes (exposure, casing, prediction floor) & rebuilt model + v5 submission
- **Task**: Apply the three fixes suggested by the outlier investigation: recompute exposure from period dates, normalize construction_type casing, and floor near-zero predictions
- **Human Contribution**: Approved all three actions
- **AI Contribution**: (1) Exposure recomputed as days in force / days in calendar year (366 for leap-year 2024), capped at 1.0, in both the training (Cell 7) and test (Cell 14) masters; (2) construction_type normalized with lower(trim()) in both masters (applied consistently so train/test distributions stay aligned — no unseen categorical levels in test after the fix); (3) PRED_FLOOR = 1.0 applied to OOF and test predictions (Tweedie unit deviance at p=1.5 grows ~2y/√pred for small pred — one moderate claim on a near-zero prediction had produced unit deviances of 1e8); CV retrained end-to-end and the full pipeline re-scored, writing to NEW files (buildings v002 and combined v005) so v001/v004 remain untouched
- **Results (before → after)**: CV mean Tweedie deviance 776 → **103.4** (folds 665/2,216/119/105 → 107/103/101/102 — the outlier is gone and fold spread is tight); fold-2 re-diagnostic: top-10 deviance rows now only 6% of the fold (was 95.8%), all genuine large claims. Trade-off noted: decile monotonicity 100% → 78% (small dips between deciles 2/3 and 5/6) and top-decile lift 2.15x → 2.01x; exposure no longer a top-gain feature (flood_zone now #1 at 7.3%) — the deviance (the competition's scoring metric) improved ~7.5x, which is why the trade-off was accepted. Test predictions: 184,155 B rows, mean 111.92 → 95.94, min 1.00 (floor), max 12,442.74. Combined v5: M 254,986 (mean 267.39, motor v1) | C 187,498 (mean 57.49, contents v001) | B 184,155 (mean 95.94, buildings v002)
- **Files affected**: `notebooks/Buildings_Bridget` (Cells 7, 10, 14, 15, 16–17 retitled v5), `submissions/buildings_tweedie_xgb_submission_v002.csv` (new), `submissions/motor_buildings_contents_submission_v005.csv` (new), `Excel files/buildings_tweedie_xgb_results.xlsx` (refreshed)

### 2026-10-21 — EDA notebook started (motor master)
- **Task**: Start an EDA notebook by loading the motor policies master from the `workings` folder
- **Human Contribution**: Requested the import; chose the simplest direct-load approach
- **AI Contribution**: New notebook `notebooks/EDA - Bridget` — markdown title cell plus a load cell (`spark.read.parquet` on `workings/policies_motor_master.parquet` into `df_motor`, first 10 rows displayed; 93 columns loaded)
- **Files affected**: `notebooks/EDA - Bridget` (new)

### 2026-10-21 — Motor EDA notebook: feature selection, Year derivation, categorical value counts + Motor_EDA.xlsx
- **Task**: Trim the motor master to an EDA feature set, derive a policy year, and produce full categorical value counts saved to Excel
- **Human Contribution**: Supplied the full variable list (later trimmed to 42 core features plus `earned_premium`); requested pandas format, `Year` from `period_start`, categorical-only value counts with full output, and the Excel export; asked for explanations of f-strings
- **AI Contribution**: `notebooks/EDA - Bridget` — `variables_motor` list (compact formatting, per user request) + `df_motor_trim` as pandas (4 ID fields + `earned_premium` + variables, 793,414 rows); `Year` = year of `period_start` via `pd.to_datetime(...).dt.year`, appended to `variables_motor`; value counts cell restricted to categorical (object/bool + `Year`) columns, excluding `period_start`/`period_end`, full distributions printed; all counts written to `Excel files/Motor_EDA.xlsx` (one sheet per variable, 23 sheets). Fixes: initial `value_counts()` failed on the Spark DataFrame (replaced with `groupBy().count()` then converted to pandas per user request); openpyxl absent on serverless compute — added `%pip install openpyxl` cell and switched the Excel writer to openpyxl
- **Results**: 47-column pandas frame; categorical distributions include make_model (300 levels), occupation (21, 12.2% null), marital_status (9.8% null), credit_band (A–E, 10% null), marketing_segment (13 levels, 10% null), region (12, R01 largest at 16%), Year evenly split ~198.5k per 2021–2024
- **Files affected**: `notebooks/EDA - Bridget` (new cells 2–6), `Excel files/Motor_EDA.xlsx` (new)

---

*This log is updated whenever AI assistance is used for project work. Each entry is committed to Git alongside the related code changes.*