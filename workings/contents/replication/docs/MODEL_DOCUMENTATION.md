# Home and motor claims cost model: technical documentation

IAA AI-Assisted Hackathon 2026, Part 1. This documents the model behind `work/submissions/submission_all_lines_glm_20261022.csv`, every data issue we found and how it was resolved, and how the model was validated. It reflects the pack as of audit entry A18 (22 October 2026).

- **Owner:** Gerard Murphy. **AI assistance:** Claude Code (Claude Opus 5.5).
- **Code:** `workings/contents/replication`, branch `contents/gerard`. Every figure here comes from a logged run (`audit/run_log.jsonl`) and an evidence file under `audit/evidence/`.
- **Decisions** are numbered D-01 to D-19 and are recorded in `audit/DECISIONS.md`. Their switches are in `config.json`. None is formally signed off yet.
- **Audit entries** A01 to A18 are in `audit/AUDIT_LOG.md`. This document summarises them; the log has the full detail.

## 1. Summary

The task is to predict the expected ultimate claim cost of every 2025 policy row in three lines: home contents, home buildings and private motor. Ultimate cost means all claims with a loss in the row's period, net of excess and recoveries, excluding declined claims. The score is Tweedie deviance D² with power 1.5.

The model has four parts:

1. **A cleaned target.** Data errors are fixed. Claims are developed to ultimate: a chain ladder by peril, Bornhuetter–Ferguson for the two slow perils, and an uplift for motor bodily injury open-claim estimates.
2. **One model per line,** with a log-exposure offset and a calendar-year trend:

   | Line | Small claims | Large claims |
   |---|---|---|
   | Contents | Tweedie GLM (no split) | — |
   | Buildings | XGBoost, trees of depth 3 | Tweedie GLM above 50k, scaled to a generalised Pareto tail |
   | Motor | Tweedie GLM | Tweedie GLM above 50k, scaled to a generalised Pareto tail |

3. **Claim history** as a Poisson–gamma credibility factor: claims a customer actually had, against claims the model expected for them.
4. **A 2025 projection:** the trend carried forward one year, using the 2025 weather and risk mix in the test file.

**2024 backtest** (fitted on 2021–23; D² against developed 2024):

| Line | D² | Without history | Without trend | Without large-claim split | Predicted / actual |
|---|---|---|---|---|---|
| Buildings | **0.1109** | 0.1092 | 0.1092 | 0.1094 | 0.915 |
| Contents | **0.0965** | 0.0959 | 0.0957 | — | 1.044 |
| Motor | **0.0806** | 0.0777 | 0.0796 | 0.0757 | 1.053 |

**2025 cost per unit of exposure:** buildings 488.8, contents 129.7, motor 627.4.

## 2. Data

### 2.1 Sources and how they are joined

| Table | Rows (training / test) | Joined on |
|---|---|---|
| `policies_contents`, `test_contents` | 721,617 / 187,498 | — |
| `policies_buildings`, `test_buildings` | 707,726 / 184,155 | — |
| `policies_motor`, `test_motor` | 979,968 / 254,986 | — |
| `claims` | 163,150 (all lines) | `policy_id`, with `loss_date` inside the row's period |
| `customers` | 545,240 | `customer_id` (motor IDs padded, issue 5) |
| `addresses` | — | `address_id` |
| `weather` | address × year, 2021–2025 | `address_id` and the row's calendar year |

Policy rows are split at the calendar year end, so no training row runs past the 31 December 2024 valuation date. Test rows all fall within 2025.

Every input file is pinned by SHA-256 (`audit/expected_inputs.sha256`) and checked before each run. All data preparation is in SQL (`sql/01`–`sql/13`), run through DuckDB.

### 2.2 Data issues and how each was resolved

| # | Issue | Evidence | Resolution | Decision / audit |
|---|---|---|---|---|
| 1 | **Exposure in the wrong units.** Some rows hold days (2–365), some use a 360-day year. | Contents training exposure overstated by 37%. Rows changed: 3,088 contents, 4,138 buildings, 5,666 motor. Test files are clean. | Recomputed as (period end − start + 1) / 365 for every row; the raw value is kept as `exposure_raw`. | D-01, D-09; A04, A15 |
| 2 | **Sum insured 1,000 times too large.** | Contents: 1,402 training and 360 test rows from 8.2m to 152.6m (median 39,000). Buildings: 1,395 and 370 rows from 52m to 594m. The same policy in another year shows a ratio of about 1,000 (median 978 over 3,265 buildings pairs), and premiums are normal. | Divided by 1,000 at ≥ 1m (contents) and ≥ 10m (buildings). Buildings needs the higher threshold because it has a genuine 1.06m risk and nothing between 2m and 36m. | D-02, D-09; A05, A15 |
| 3 | **Buildings `construction_type` in mixed case** (`Brick`, `BRICK`, `brick`…). | 11 raw values for 5 real ones. Rare variants sent rows into near-empty tree leaves in a teammate's model. | Lower-cased and trimmed; the raw value is kept. | A15 |
| 4 | **Duplicate claims.** | 604 claims (motor 411, buildings 109, contents 84) repeat another claim on every field except `claim_id`, down to the repairer and third party. | One copy kept (`claims_clean`); the raw table is still profiled. Found first by the team's data workstream (GitHub #55). | D-19; A18 |
| 5 | **Motor `customer_id` missing its leading zeros.** | 98,110 training and 25,554 test rows (10%) found no customer record; padded to 8 characters, all match. Flagged in A03 but applied to motor only in A18; A15–A16 had wrongly treated these as missing. | Padded to 8 characters; the raw value is kept. Found by the data workstream too (GitHub #51). | D-19; A03, A18 |
| 6 | **Customer service contacts leak the outcome.** | Contents: claimants average 3.05 contacts against 0.41, and the test file averages 0.40, matching non-claimants. On its own the field would add D² 0.577, which is impossible. Buildings (3.0 against 0.4) and motor (3.3 against 0.4) are the same. | Excluded from every model. | D-04, D-09; A08, A15 |
| 7 | **Written premium is annual, not for the period.** | Average 135–137 whatever the length of the row; correlation with time on cover 0.00. Loss ratios quoted on written premium were understated, giving a false downward trend. | Earned premium = written premium × exposure, used for loss ratios only. Written premium is also kept out of the models (D-05): it adds nothing beyond the risk factors (−0.0007 to +0.0002 D² in A17), though strongly predictive on its own. | D-05, D-07; A12, A17 |
| 8 | **Declined claims and claims outside any policy period.** | Contents: 24,102 claims → 84 duplicates → 299 repudiated → 49 outside any period → 23,670 in the target. Buildings: 67 outside. Motor: 1,406 repudiated, 293 outside. Some losses are dated 2020, before the first policy row. | Excluded. Target = incurred to date (already net of excess) − recoveries. Motor recoveries are 6% of incurred; home lines have none. | D-03; A06, A15 |
| 9 | **`renewal_decision` exists only in training** and is known only after the period. | Absent from every test file. | Never used as a feature. | A15 |
| 10 | **Missing values** (security level, high-value items, flood zone, river distance, vehicle fields, telematics score at 88%). | Missing flood zone has a relativity of 1.71, so missingness carries information. | GLMs: missing is its own level, or a separate flag for smoothed numerics. XGBoost: learned default direction. No imputation. | D-06; A07 |
| 11 | **Claims above the sum insured.** | 3 contents and 8 buildings claims, at most 1.48×. | Left as recorded; too few to matter. | A09, A17 |
| 12 | **Motor NCB years exceed years licensed** on 161,822 rows (13%); 3,717 rows licensed before age 16. | These rows claim *less* at every licence band (for example 259 against 312 per 1,000 at 0–2 years licensed), so the field carries real signal. | Left as recorded; correcting it would remove signal. | A17 |
| 13 | **Zero annual mileage** on 15,036 motor rows. | Small, and the smoothed mileage term handles it. | Left as recorded. | A17 |
| 14 | **One catastrophic claim:** motor bodily injury of 11.89m, open (claim CL30973963, 2023). | Ten times the next largest (1.12m), about 10% of 2023 motor cost. 738,769 already paid, so not a ×1,000 error. | Kept. Handled by the large-claim model and Pareto tail (section 4), and capped at 1m inside the Bornhuetter–Ferguson prior. | D-12, D-15, D-17; A15–A17 |
| 15 | **Recent claims not fully reported or settled.** | Valuation 31 December 2024. 2024 motor reported cost is 22% below 2023. Motor bodily injury reports with an average lag of 159 days (up to 1,361); subsidence 116. | Developed to ultimate (section 3). | D-10, D-11, D-17 |
| 16 | **A step in claim cost from 2023, and inflation.** | Contents cost per unit of exposure is 105 in 2021–22 and 117 in 2023–24, with flat frequency. Claim-size quantiles rise 3.9–4.4% a year. | Calendar-year trend in every model, projected to 2025. | D-13; A13–A14 |
| 17 | **Competition data in the public GitHub repo.** | `data/*.parquet` is on `main` (and on `hackathon-bridget` and `data-cian`) of a public repository. A02 raised the risk on day one. | **Open.** Needs a team decision: make the repo private, or remove the files and their history (rules §5b). | A02, A18 |

**Checks that found no problem:** unique row and claim IDs; no duplicate rows, zero or negative exposure, or overlapping periods for any policy; every buildings, contents and padded motor row joins to customer, address and weather; no category level in 2025 unseen in training; policyholder age matches date of birth on every home row; drift into 2025 matches the trends already modelled (sum insured about +2% a year, premium about +5%).

## 3. The target: ultimate claim cost

Per row, the target is the sum over its claims of net cost × development factor. Each claim's factor is set in `sql/10` as follows.

### 3.1 Not yet reported (IBNR): chain ladder by peril (D-10)

- **Pattern:** for each line and peril, a cost-weighted report-lag distribution F is estimated from all four loss years. The right-truncation product-limit estimator corrects for the fact that a recent claim is only visible if its lag is shorter than its age: F(l) = ∏ over k > l of (1 − d_k / n_k), with d_k the cost reported at lag k and n_k the cost at risk.
- **Factor:** each claim is grossed up by 1/F(age), averaged by loss quarter, so a single early-reported claim does not carry the whole quarter's late claims (claim by claim, one motor bodily injury claim would have been ×60).
- **Validation:** the estimated pattern matches the empirical reported share in fully observed years to about ±0.01.
- **Reporting speed:** after 90 days most perils have 98–99.7% of their cost reported. The exceptions are subsidence at 49% and motor bodily injury at 35%.
- **Correction (A15):** an earlier contents estimate pooled every peril into one pattern, which overstated theft and understated escape of water.

### 3.2 Immature slow perils: Bornhuetter–Ferguson (D-17)

- **When it applies:** to a loss year with under 75% of its cost reported. In practice that is 2024 subsidence (59% reported) and 2024 motor bodily injury (57%).
- **How:** BF = reported + unreported share × prior. The prior is the cost per unit of exposure of the up to two preceding years that are at least 90% reported, with no trend and with claims capped at 1m.
- **Result:** 2024 subsidence becomes 11.3m (the chain ladder gives 13.3m), and bodily injury 46.0m (42.8m).

**Backtest (`python/backtest_development.py`):** the position was rebuilt as at 31 December 2023 from report dates alone, then 2023 was developed and compared with 2023 as now known.

| Peril | Chain ladder / actual | Method used / actual |
|---|---|---|
| Buildings subsidence | 1.177 | **1.012** (BF) |
| Motor bodily injury | 1.326 | **1.134** (BF) |
| All other perils | 0.965–1.024 | same (chain ladder) |

The chain ladder failed for the slow perils for two reasons. It gives too much weight to the half of the claims already reported (the 11.9m claim included), and motor bodily injury is reported faster each year (within 365 days: 85.9% → 87.0% → 89.3%).

### 3.3 Open-claim estimates on motor bodily injury (IBNER, D-11)

- **The problem:** the data is one snapshot, so movement on reported claims cannot be observed directly.
- **Evidence:** for every peril except motor bodily injury, the current average cost of claims reported within 90 days rises with inflation from year to year. For bodily injury it falls as the open share rises: 21,049 (2021, 10% open) → 19,564 → 17,325 → 18,098 (2024, 74% open).
- **Fix:** each bodily injury loss year is lifted to the 2021 level with no inflation: ×1.076, ×1.215 and ×1.163 for 2022–24. This is a deliberate lower bound, because inflation is left to the trend.

### 3.4 Developed cost per unit of exposure

| Year | Buildings | Contents | Motor |
|---|---|---|---|
| 2021 | 383.9 | 105.5 | 564.5 |
| 2022 | 403.1 | 105.5 | 599.4 |
| 2023 | 433.8 | 117.2 | 659.8 (including the 11.9m claim) |
| 2024 | 484.7 | 120.0 | 591.4 |

## 4. The models

All three lines share:
- **Offset:** log of exposure, so every prediction is cost for the row's period.
- **Metric and objective:** Tweedie power 1.5, matching the score.
- **Fitting windows:** tune on 2021–22 against 2023; backtest on 2021–23 against 2024; final fit on 2021–24 to predict 2025.

Code: `python/run_glm_lines.py`, `python/glm_lines.py`, `python/xgb_models.py`, `python/models.py`.

### 4.1 Contents: one Tweedie GLM, no split

- **Factors:** 16 banded rating factors from GLM A: flood zone, tenure, region, marketing segment, prior claims, missed payments, voluntary excess, credit band, security, high-value items, accidental-damage cover, bicycle cover, works from home, lodgers, urban density, property type.
- **Smooth curves:** cubic splines on policyholder age, elevation and distance to river. These replaced GLM A's bands in A17.
- **Continuous:** standardised log sum insured.
- **Trend:** a linear calendar-year term.
- **Penalty:** L2 at 10⁻⁴, tuned on 2023; it makes almost no difference.
- **No large-claim split.** Contents claims top out at 113k, and A09 showed splitting by size or peril lowers D².

### 4.2 Buildings: XGBoost for small claims, GLM for large claims

- **Small claims** (each claim up to 50k):
  - XGBoost with depth-3 trees (the owner's choice) and a leaf minimum of 20,000 rows' worth of hessian. The other settings: η 0.05, subsampling 0.8, L2 1.
  - 144 trees, set by early stopping on 2023.
  - 53 raw features: policy, customer, address and that year's weather.
  - The year trend sits in the offset, fitted jointly with the trees by backfitting, because trees cannot extrapolate a year feature.
- **Large claims** (cost above 50k):
  - Tweedie GLM on spline curves for flood zone, sum insured, elevation, year built and distance to river, plus soil type and construction type. L2 penalty 10⁻³.
  - Its level is scaled to the mean of a generalised Pareto distribution fitted to the 950 reported excesses (shape 0.33, adjustment ×1.025). It takes the small-claim trend.
- **Thresholds:** 50k was chosen on 2023 from 25k, 50k and 100k.
- **Tried and rejected:** a separate model for weather perils (worse on both test years). The model already uses each year's weather.

### 4.3 Motor: GLM for small claims, GLM for large claims

- **Small claims** (each claim up to 50k), Tweedie GLM:
  - **Spline curves:** NCB years, driver age, annual mileage, years licensed, vehicle age, telematics score, vehicle value group, voluntary excess.
  - **Bands:** prior claims and missed payments (too few distinct values for a curve).
  - **Categories:** region, marketing segment, cover type, urban density, channel, vehicle use, payment method, credit band.
  - **Year trend;** L2 penalty 10⁻⁵.
  - Factors were taken from those with at least 1% of XGBoost gain. Credit band and missed payments were added in A17 after passing the two-year rule.
- **Large claims** (above 50k):
  - Tweedie GLM on banded NCB, driver age, mileage and years licensed, plus region. Smoothing this model failed the two-year rule.
  - Generalised Pareto fit to 591 reported excesses: shape 0.48, so a heavy tail. The fitted mean excess of 50.0k replaces the empirical 65.3k, which the 11.9m claim inflates (adjustment ×0.765).

### 4.4 How claims are split into small and large

Each claim is split at the threshold on its *reported* cost, and both parts then get the claim's development factor. Developing first would push slow-reporting claims over the threshold artificially: motor bodily injury in Q4 2024 is grossed up about ×4.5. The large-claim model's trend comes from its line's small-claim model.

### 4.5 Trend (D-13)

The trend is a log-linear calendar-year term fitted jointly with each line's small-claim model. It is projected from the fitted line to 2025: buildings +5.6% a year, contents +4.3%, motor +5.5%.

The trend is net of the rating factors and, for buildings, of weather. In the backtest it improves all three lines: buildings 0.1109 against 0.1092 without it, contents 0.0965 against 0.0957, motor 0.0806 against 0.0796. In the earlier A15 run the motor trend over-predicted 2024; the owner chose to keep it then.

### 4.6 Claim history: credibility (D-14)

- **Counts** (`sql/12`): for every row, the customer's prior claims, counting only claims both lost and reported before the row's start date. A test enforces that no later claim leaks in. There are three groups:
  - **Own line:** same customer and address for home, same customer for motor.
  - **Other home line:** same customer and address.
  - **Motor and home across:** same customer.
- **Expected** (`sql/13`): E is the number of claims a frequency GLM *without* history expects over the same prior rows.
- **Factor:** (k + n) / (k + E) per group, multiplied together and rebalanced so the fitted rows keep their total. k is tuned on 2023; infinite k switches a group off.
  - Own line: k = 4 (buildings), 2 (contents), 8 (motor).
  - Other home line: off for buildings, k = 8 for contents.
  - Motor and home across: off for buildings and motor, k = 8 for contents.
  - In 2025 the factor ranges from 0.67 to 3.40.
- **Why it matters:** claim frequency per 1,000 by prior own-line claims (0 / 1 / 2 / 3 or more) is 43 / 69 / 108 / 185 for buildings, 38 / 64 / 81 / 140 for contents and 116 / 196 / 291 / 470 for motor.
- **Rejected approach:** history as ordinary model factors. History length grows with the calendar year (none for 2021, up to four years for 2025), so the factors became a stand-in for the trend, and the 2025 rows fell outside the training range. Contents D² dropped to 0.0614 (A16).

### 4.7 Buildings–contents correlation (D-16)

There is no usable joint behaviour, so the two lines are modelled independently and no copula is fitted:

- **Same year:** with both policies held, contents frequency is 42.3 per 1,000 with a buildings claim and 40.3 without.
- **Same event:** 4 claim pairs share an address and a loss date, out of about 24,000 contents and 26,000 buildings claims. Even flood never hits both.
- **After the models:** the correlation of buildings and contents residuals in 2024 is −0.0025, and of claim indicators 0.005, over 128,878 customer-addresses.
- **What remains** is captured through cross-line claim history. Tuning keeps it only for contents.

A copula would change the spread of combined losses, which matters for capital or reinsurance, but not the expected cost per row, which is what is scored.

### 4.8 The 2025 prediction

For each 2025 row:

> prediction = [ small-claim model at 2025 + large-claim GLM × Pareto adjustment × trend to 2025 ] × claim-history factor × rebalancing constant

Contents has no large-claim term.

## 5. Validation

### 5.1 Protocol and the two-year rule

- **Out of time only.** 2024 is never used to tune anything inside a model.
- **The two-year rule:** a change is kept only if it improves D² against the developed actual on *both* 2023 (fit 2021–22) and 2024 (fit 2021–23). Scores are compared only within one development basis, because changing the development changes the actual itself.

### 5.2 Overfitting checks (A17)

- **In-sample against out-of-time:** contents scores 0.101 on its fitting years and 0.094 out of time. Large-claim noise explains the gap.
- **Stable coefficients:** between the 2021–23 and 2021–24 fits, coefficients correlate at 0.99 and none of 85 (contents) or 102 (motor) moves by more than 0.2.
- **Penalty strength makes no difference:** D² is unchanged from 10⁻⁵ to 10⁻³. With 700,000–980,000 rows per line, the parameters are well supported.

### 5.3 What was tried and rejected

| Alternative | Result | Where |
|---|---|---|
| Contents split by peril or claim size | D² 0.0920 and 0.0872, against 0.0933 for one model | A09 |
| XGBoost instead of a GLM for contents | 0.0919 against 0.0932 (bootstrap interval includes zero) | A13 |
| XGBoost for every line (first combined submission) | Beaten by the current models on every line | A15, A16 |
| Written premium as a factor | −0.0007 to +0.0002 | A17 |
| Previous insurer, campaign code, occupation | −0.001 to −0.003 out of time, despite large one-year swings | A17 |
| Claim history as model factors | Contents 0.0614; motor trend inflated to 8.5% | A16 |
| Separate weather-peril model for buildings | −0.0015 and −0.0018 | A17 |
| Smoothing the motor large-claim GLM | −0.0020 on 2023 | A17 |
| Correcting NCB to at most years licensed | Not fitted: the "impossible" rows carry signal | A17 |
| Copula between buildings and contents | Not fitted: residual correlation −0.0025 | A16 |

## 6. Submission

- **File:** `work/submissions/submission_all_lines_glm_20261022.csv`, kept out of git (`work/` is ignored). It has 626,639 rows, with row_ids and order identical to `sample_submission.csv`, no nulls, and every prediction above zero (minimum 0.044). A zero would make the deviance infinite on any row with a claim.
- **Mean prediction per row:** buildings 398.8, contents 103.8, motor 492.1.
- **2025 cost per unit of exposure against developed 2024:** buildings 488.8 against 484.7, contents 129.7 against 120.0, motor 627.4 against 591.4.

## 7. Decisions

| ID | Decision | Owner's position |
|---|---|---|
| D-01 | Exposure from the period dates | Applied |
| D-02 | Sum insured ÷1,000 above a threshold | Applied |
| D-03 | Target = incurred − recoveries; declined and out-of-period claims excluded | Applied |
| D-04 | Exclude customer service contacts | Applied |
| D-05 | Keep written premium out of the models | Applied; confirmed in A17 |
| D-06 | Missing values as their own level | Applied |
| D-07 | Exposure offset; earned premium for loss ratios only | Chosen by the owner |
| D-09 | D-01, D-02 and D-04 extended to buildings and motor | Applied |
| D-10 | IBNR by peril, product-limit pattern | Applied |
| D-11 | Motor bodily injury IBNER lift (lower bound) | Applied |
| D-12 | 1m cap and loading (XGBoost run only); replaced by D-15 | Superseded |
| D-13 | Trend to 2025 | Approved; motor trend confirmed after the backtest |
| D-14 | Claim history as credibility | Owner asked for full use of history |
| D-15 | Large-claim split with a Pareto tail (buildings, motor) | Owner's design |
| D-16 | Model structure per line; no copula | Owner's design |
| D-17 | Bornhuetter–Ferguson for immature slow perils | Owner asked for recommendation 1 |
| D-18 | Spline smoothing; two motor factors | Owner asked for recommendations 2 and 3 |
| D-19 | De-duplicate claims; pad motor customer IDs | Applied |

D-08 records the XGBoost feature set used in A13 only. No decision is formally signed off.

## 8. Limitations and open questions

1. **Development is the largest uncertainty.**
   - Motor bodily injury: even with BF, the backtest still over-predicts by 13%, and its "actual" is itself 49% open.
   - Subsidence: 2024 is 59% reported.
   - The IBNER lift is a lower bound.
2. **Level.** The backtest under-predicts buildings 2024 by 8.5% and over-predicts contents and motor by 4–5%. Under-prediction costs more D² than over-prediction. On contents 2024, 8% too low cost about 0.0015 and 8% too high about 0.0003. That comparison was made during the trend analysis of 21 October and is not in an audit entry.
3. **Trends rest on four years.** Year-to-year noise is about ±3.5%.
4. **Selection on the same years.** 2023 and 2024 have informed many choices, so the backtest gains are somewhat optimistic. The leaderboard is the independent check, but it should confirm choices, not drive them.
5. **Implementation coverage.** The new models are implemented in Python only. GLM A, the contents benchmark, has an independent R implementation that reconciles (100 checks, 0 failed).
6. **Synthetic data.** It has no shared events between buildings and contents. A real portfolio would, and the independence assumption would need revisiting for Week 2.
7. **Data in a public repository** (section 2.2, issue 17) is unresolved.

## 9. How to reproduce

From `workings/contents/replication`, with Python 3.11 and `requirements.txt` installed (and R 4.5 for the GLM pack cross-check):

```bash
python python/fetch_data.py
```
```bash
python python/run_all.py
```
```bash
Rscript R/run_all.R
```
```bash
python reconcile/reconcile.py
```
```bash
python python/backtest_development.py
```
```bash
python python/run_glm_lines.py
```
```bash
python -m pytest -q tests
```

- `run_glm_lines.py` needs `work/sample_submission.csv`, the competition's sample file. It takes about 10 minutes and writes the submission and the `audit/evidence/glm_lines/` tables.
- Each run appends a record to `audit/run_log.jsonl`: input, SQL and code hashes, decision switches, package versions and headline metrics.
- **Expected results:** reconciliation ends with `0 failed`; the 2024 backtest matches section 1; 33 tests pass.

## 10. Where to find things

| What | Where |
|---|---|
| Data preparation | `sql/01`–`sql/08` (contents), `sql/09`–`sql/13` (all lines, development, history) |
| Current model | `python/run_glm_lines.py`, `python/glm_lines.py` |
| Earlier XGBoost runs | `python/run_xgb.py` (A13), `python/run_lines.py` (A15) |
| Development backtest | `python/backtest_development.py` |
| Parameters and decision switches | `config.json` |
| Narrative audit log | `audit/AUDIT_LOG.md` |
| Decisions for sign-off | `audit/DECISIONS.md` |
| Evidence tables | `audit/evidence/glm_lines/` (current), `audit/evidence/lines/` (A15 and the development backtest), `audit/evidence/python`, `audit/evidence/r` (GLM pack) |
| Tests | `tests/` |
