# Home and motor claims cost model: technical documentation

IAA AI-Assisted Hackathon 2026, Part 1. This documents the model behind `work/submissions/submission_final_candidate_20261023.csv`, every data issue we found and how it was resolved, and how the model was validated. It reflects the pack as of audit entry A31 (26 October 2026); section 11 covers Week 2, the emerging market. A visual summary of the whole project, with charts, is in [`FINAL_REPORT.html`](FINAL_REPORT.html).

- **Owner:** Gerard Murphy. **AI assistance:** Claude Code (Claude Opus 5.5).
- **Code:** `workings/contents/replication`, branch `contents/gerard-week2-emerging` (Part 1 final review on `contents/gerard-final-review`). Every figure here comes from a logged run (`audit/run_log.jsonl`) and an evidence file under `audit/evidence/`.
- **Decisions** are numbered D-01 to D-39 and are recorded in `audit/DECISIONS.md`. Where a decision has a switch, it is in `config.json`; the Week 2 decisions (D-31 to D-39) have none and are set in `python/emerging.py` or by file. None is formally signed off yet.
- **Audit entries** A01 to A31 are in `audit/AUDIT_LOG.md`. This document summarises them; the log has the full detail.

## 1. Summary

The task is to predict the expected ultimate claim cost of every 2025 policy row in three lines: home contents, home buildings and private motor. Ultimate cost means all claims with a loss in the row's period, net of excess and recoveries, excluding declined claims. The score is Tweedie deviance D² with power 1.5.

The model has four parts:

1. **A cleaned target.** Data errors are fixed, including two in the address table found in the final review (R11 elevation in feet; a placeholder year built). Claims are developed to ultimate: a chain ladder by peril and Bornhuetter–Ferguson for the two slow perils. The earlier uplift for motor bodily-injury claim size was switched off in A28: closed claims fell as much as open ones, so the fall was real.
2. **One model per line,** with a log-exposure offset and a calendar-year trend:

   | Line | Small claims | Large claims |
   |---|---|---|
   | Contents | Tweedie GLM (no split), seasonal exposure | — |
   | Buildings | XGBoost, trees of depth 3, with two peril interactions | Tweedie GLM above 50k with the same interactions, scaled to a generalised Pareto tail |
   | Motor | Tweedie GLM | Tweedie GLM above 50k, scaled to a generalised Pareto tail |

3. **Claim history** as a Poisson–gamma credibility factor: claims a customer actually had, against claims the model expected for them.
4. **A 2025 projection:** the trend carried forward one year with the 2025 weather and risk mix, and a level adjustment where both backtest years support it: buildings ×1.033 and, for uncertainty, motor ×1.005. The level's 90% range is reported per line.

A neural correction (CANN) was built and tested on every line. In the final model its two-year mean gain is not significant on any line, and motor's gain, although significant on 2024, is worse on 2023 and fails the two-year rule, so none is used.

**2024 backtest** (fitted on 2021–23; D² against developed 2024):

| Line | D² | Without history | Without trend | Without large-claim split | Predicted / actual |
|---|---|---|---|---|---|
| Buildings | **0.1138** | 0.1122 | 0.1131 | 0.1117 | 0.934 |
| Contents | **0.0969** | 0.0964 | 0.0961 | — | 1.039 |
| Motor | **0.0832** | 0.0791 | 0.0823 | 0.0798 | 1.005 |

Motor's actual changed when the bodily-injury lift was switched off (A28), so its D² is not comparable with earlier entries; buildings (0.1107 before A28) and contents (0.0968) are.

**2025 cost per unit of exposure** (90% range for the level in brackets): buildings 500.3 (437–573), contents 129.2 (117–142), motor 581.3 (490–690).

**Leaderboard:** the final candidate scored **0.12805**, the best public score so far, against 0.12753 for the A22 file; the leader had 0.13050. The final candidate's motor level is ×0.927 of the A22 file's, close to the ×0.93 the leaderboard implied (audit log A27; scores in section 6), without being tuned to it.

**Statistical tests (A29):** Type III tests, chi-square tests by level, level merging, dropping terms and a pairwise interaction screen, all scored out of time where they could change the model. Nothing passes the rules, so the model is unchanged (section 5.9).

**The team's other branches** were reviewed for techniques to combine (A24). Blends of GLM and XGBoost, tuned XGBoost and frequency × severity models all failed out-of-time testing; one data fix (timber construction recoded in 2023) was adopted. Section 5.6 has the detail.

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
| 18 | **Buildings timber construction recoded in 2023.** | `T` appears only in 2021–22 and `W` only from 2023 (about 22,000 rows a year each); every 2025 test row is `W`. The same buildings switch code: 16,954 of the 17,621 policies coded `T` in 2022 (96%) are `W` in 2023, against about 2% code changes between any other pair of years (A27). A scan of all 31 text fields on the three lines found no other year-specific recode. | Merged as `timber`; the raw value is kept. Left apart, `W` stands in for "2023 or later". Found by the data workstream (#54). | D-24; A24 |
| 19 | **Region R11 elevation recorded in feet.** | All 49,046 R11 addresses run 3.25–3.30× every other region at the 10th, 50th and 90th percentiles and within every flood zone; the other regions cap at 60.0 and R11 at 196.9 (60 × 3.28). R11's river distance and flood zones match the others. | Converted to metres in a cleaned address view, raw value kept ([#154](https://github.com/almac09/iaa-hackathon-2026/issues/154)). Found in the final review. | D-26; A27, A28 |
| 20 | **`year_built = 1900` is a placeholder.** | 19,014 addresses (10.8× any neighbouring year); their energy ratings and claim rates match houses of all ages, yet the models price them as pre-1900. | Replaced by the typical year built for the energy rating, raw value kept ([#155](https://github.com/almac09/iaa-hackathon-2026/issues/155)). Found in the final review. | D-27; A27, A28 |

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
- **Evidence:** for every peril except motor bodily injury, the current average cost of claims reported within 90 days rises with inflation from year to year. For bodily injury it falls as the open share rises: 21,049 (2021, 10% open) → 19,392 → 17,325 → 18,098 (2024, 74% open).
- **Fix:** each bodily injury loss year is lifted to the 2021 level with no inflation: ×1.085, ×1.215 and ×1.163 for 2022–24 (×1.076 for 2022 before the duplicate claims were removed in A18; corrected in A27).
- **Switched off (A28, D-28):** closed claims, which cannot develop, fell in size by as much as open ones in 2023, so the fall is real severity rather than under-reserving. Bodily injury is now developed for IBNR only. Motor's development factor falls from ×1.303 to ×1.236 for 2024 and from ×1.123 to ×1.029 for 2023 ([#156](https://github.com/almac09/iaa-hackathon-2026/issues/156)).

### 3.4 Traditional methods tested (A25)

Three further chain-ladder ideas were tested; runs are labelled "A25 development methods" in the run log.

- **Recent-weighted reporting patterns** (`pattern_decay` in `sql/10`: each claim weighted decay^(valuation year − loss year)). The development backtest was rebuilt at two valuation dates. Developed / actual for the backtest year, at decay 1, 0.5 and 0.25:

  | Line | As at 2022-12-31 | As at 2023-12-31 |
  |---|---|---|
  | Buildings | 0.992 / 0.993 / 0.994 | 0.993 / 0.997 / 1.000 |
  | Contents | 1.002 / 1.003 / 1.003 | 0.988 / 0.988 / 0.987 |
  | Motor | 1.035 / 1.037 / 1.039 | 1.055 / 1.065 / 1.075 |

  Motor and contents get worse in both years. Buildings improves slightly, but its subsidence 2023 worsens from 1.012 to 1.065, so this is **not adopted** (decay stays at 1). Recent weighting does not carry over from the chain ladder on `main` to this reporting-lag method.
- **Fitting on reported claims with development as an offset** (`python/development_tests.py`) instead of grossing claims up: buildings 0.0000 / +0.0002 (2023 / 2024), contents 0.0000 / 0.0000, motor −0.0002 (significant) / −0.0003. **Not adopted.** Development is applied as quarter averages, so grossing up adds less noise than claim-by-claim factors would.
- **Mack's chain-ladder error** in the level ranges: **adopted** (D-25, section 4.9).
- **Finding:** in both backtests the motor development over-estimates the year (×1.035 and ×1.055), mostly through bodily injury (×1.087 and ×1.134). Raised on [#151](https://github.com/almac09/iaa-hackathon-2026/issues/151), the motor-level question.

### 3.5 Developed cost per unit of exposure

| Year | Buildings | Contents | Motor |
|---|---|---|---|
| 2021 | 383.9 | 105.5 | 564.5 |
| 2022 | 403.1 | 105.5 | 578.8 |
| 2023 | 433.8 | 117.2 | 604.8 (including the 11.9m claim) |
| 2024 | 484.7 | 120.0 | 561.0 |

Motor is developed without the bodily-injury lift (D-28).

## 4. The models

All three lines share:
- **Offset:** log of exposure, so every prediction is cost for the row's period. For contents, exposure is seasonal (section 4.1).
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
- **Seasonal exposure (D-22, A22):** each day of a row counts by its month's claim frequency relative to the year, estimated on the fitting years only. The fitting years keep their total exposure, and reported rates use plain exposure.
  - Weights from 2021–24, used for 2025: January 1.11, February 1.13, December 1.07; March to November 0.94–1.00.
  - It matters for part-year rows: a row covering only winter months expects up to about 13% more than plain days imply.
  - D² gain: +0.0010 on 2023, +0.0003 on 2024. Buildings and motor were tested too and failed (A21).
  - Weights: `audit/evidence/glm_lines/ev_g14_seasonal_exposure_weights.csv`.

### 4.2 Buildings: XGBoost for small claims, GLM for large claims

- **Small claims** (each claim up to 50k):
  - XGBoost with depth-3 trees (the owner's choice) and a leaf minimum of 20,000 rows' worth of hessian. The other settings: η 0.05, subsampling 0.8, L2 1.
  - 194 trees, set by early stopping on 2023 (131 before the A28 changes, 144 before the timber merge).
  - 53 raw features (policy, customer, address and that year's weather) plus two peril interactions (A28, D-29): clay soil × dry-summer index (subsidence) and flood zone 3 or above, or missing, × rainfall (flood). The large-claim GLM takes the same two as linear terms. Together they add +0.0011 / +0.0012 (two-year mean +0.00115, significant).
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

The trend is a log-linear calendar-year term fitted jointly with each line's small-claim model. It is projected from the fitted line to 2025: buildings +5.3% a year, contents +4.2%, motor +3.8% (final model, A28; motor was +5.5% with the bodily-injury lift).

The trend is net of the rating factors and, for buildings, of weather. In the final model's backtest it improves all three lines: buildings 0.1138 against 0.1131 without it, contents 0.0969 against 0.0961, motor 0.0832 against 0.0823. In the earlier A15 run the motor trend over-predicted 2024; the owner chose to keep it then.

### 4.6 Claim history: credibility (D-14)

- **Counts** (`sql/12`): for every row, the customer's prior claims, counting only claims both lost and reported before the row's start date. A test enforces that no later claim leaks in. There are three groups:
  - **Own line:** same customer and address for home, same customer for motor.
  - **Other home line:** same customer and address.
  - **Motor and home across:** same customer.
- **Expected** (`sql/13`): E is the number of claims a frequency GLM *without* history expects over the same prior rows.
- **Factor:** (k + n) / (k + E) per group, multiplied together and rebalanced so the fitted rows keep their total. k is tuned on 2023; infinite k switches a group off.
  - Own line: k = 4 (buildings), 2 (contents), 4 (motor). Motor's k is tuned on 2023 claims capped at 1m per row (A28, D-30), so the single 15.25m claim does not choose it; it was 8 before.
  - Other home line: off for buildings, k = 8 for contents.
  - Motor and home across: off for buildings and motor, k = 8 for contents.
  - In 2025 the factor ranges from 0.59 to 3.70 (buildings 0.84–2.58, contents 0.68–3.40, motor 0.59–3.70).
- **Why it matters:** claim frequency per 1,000 by prior own-line claims (0 / 1 / 2 / 3 or more) is 42 / 68 / 117 / 166 for buildings, 38 / 64 / 84 / 134 for contents and 115 / 195 / 292 / 476 for motor.
- **Rejected approach:** history as ordinary model factors. History length grows with the calendar year (none for 2021, up to four years for 2025), so the factors became a stand-in for the trend, and the 2025 rows fell outside the training range. Contents D² dropped to 0.0614 (A16).

### 4.7 Buildings–contents correlation (D-16)

There is no usable joint behaviour, so the two lines are modelled independently and no copula is fitted:

- **Same year:** with both policies held, contents frequency is 42.3 per 1,000 with a buildings claim and 40.3 without.
- **Same event:** 4 claim pairs share an address and a loss date, out of about 24,000 contents and 26,000 buildings claims. Even flood never hits both.
- **After the models:** the correlation of buildings and contents residuals in 2024 is −0.0025, and of claim indicators 0.005, over 128,878 customer-addresses.
- **What remains** is captured through cross-line claim history. Tuning keeps it only for contents.

A copula would change the spread of combined losses, which matters for capital or reinsurance, but not the expected cost per row, which is what is scored.

### 4.8 Neural correction of the small-claim model: CANN (D-20)

- **Method** (Wüthrich and Merz's combined actuarial neural network): the incumbent small-claim model's log prediction is a fixed offset. A network adds a log correction learned from the same rows. Its output layer starts at zero, so training starts exactly at the incumbent and only moves where the data supports it.
  - **Inputs:** each line's model factors (buildings: the 53 raw features), with categories as learned embeddings and numerics standardised, with a flag for missing values.
  - **Architecture**, fixed in advance and not tuned: hidden layers of 128, 64 and 32 units, SiLU activation, 10% dropout.
  - **Training:** Tweedie loss with p = 1.5, Adam at learning rate 0.001, batches of 16,384 rows, gradients clipped at norm 1. Training stops early on a random 10% of the fitting rows, with a patience of 4 epochs.
- **Ensemble:** 20 networks that differ only in their random seed. The correction is their mean log output (the "nagging" predictor of Richman and Wüthrich), rebalanced so the fitting rows keep the incumbent's total.
- **Adoption:** under the two-year rule and, as a discretionary addition, only if the two-year mean gain is significant (D-23):

  | Line | 2023 without / with | 2024 without / with | Two-year mean gain (95% interval) | Adopted |
  |---|---|---|---|---|
  In the final model (A28, `ev_g7_cann_two_year.csv`):

  | Line | 2023 without / with | 2024 without / with | Two-year mean gain (95% interval) | Adopted |
  |---|---|---|---|---|
  | Buildings | 0.11277 / 0.11278 | 0.11378 / 0.11378 | +0.00001 (−0.00001, +0.00002) | No (not significant) |
  | Contents (seasonal exposure) | 0.09568 / 0.09574 | 0.09694 / 0.09697 | +0.00005 (−0.00021, +0.00030) | No |
  | Motor | 0.09981 / 0.09975 | 0.08318 / 0.08409 | +0.00043 (−0.00008, +0.00093) | No (fails 2023) |

  Before A28, buildings passed (0.11102 / 0.11116 and 0.11046 / 0.11071, +0.00019 (+0.00005, +0.00034)) and was adopted. After the address corrections and peril interactions its gain is no longer significant, so CANN is dropped on every line. Whether the interactions capture what the network found was not tested directly.

- **How much it matters:** very little. In the final model the buildings gain is +0.000009 on 2023 and +0.000004 on 2024. Before A28 it was +0.00014 and +0.00025, and +0.00005 before the timber merge, which that partly made up for. Contents' identical-looking gain was dropped because its interval is about seven times wider. When it was adopted (A20), CANN had also raised contents' 2024 prediction by 0.3%. Motor's swing between years (−0.0001, +0.0009) is larger than the gain that was ever adopted for any line. An earlier probe trained the seeds one after another rather than in one batch, and it passed for motor (+0.00015 and +0.00054). The motor effect is at the level of seed noise.
- **Engineering:**
  - The networks train on the Apple GPU (PyTorch, MPS backend). All 20 seeds train at once as one batched network, with weights stacked along a seed axis. Each seed keeps its own initialisation, batch order, gradient clipping and early stopping. During training GPU utilisation is about 96%, and one 20-seed fit on 900,000 rows takes roughly 10–60 seconds, depending on when early stopping ends it.
  - PyTorch runs in a separate worker process (`python/cann_torch.py`), which is started as a plain subprocess. PyTorch and XGBoost each load their own OpenMP runtime on macOS, and the two crash or deadlock in one process (A20).
  - Results are reproducible: two full runs gave identical D² to five decimals.

### 4.9 Level and its uncertainty (D-21)

- **Level factor:** the deviance-optimal scale k = Σ y μ^(1−p) / Σ μ^(2−p) is measured on each backtest year. This is not the ratio of totals, because the log link is not canonical for p = 1.5. A level factor is adopted only if each year's k improves the *other* year.
  - In the final model (A28), k is 1.031 and 1.036 for buildings, 1.111 and 0.987 for contents, 1.003 and 1.021 for motor (2023 and 2024).
  - Until A28 no line passed. With the address corrections and interactions, buildings' two years agree, and each improves the other (+0.00012 / +0.00016), so buildings takes ×1.033. Contents and motor still fail.
- **Uncertainty in the 2025 level** has three parts:
  - **Development:** 200 bootstrap resamples of the training rows, each re-running the whole development step (`python/level_uncertainty.py`). The coefficient of variation of the development factor for 2024 is 0.8% (buildings), 0.4% (contents) and 1.3% (motor); for 2023 it is 0.1%, 0.04% and 0.4%.
  - **Year-to-year noise:** σ_year, from the final model's log actual/expected by year (2 degrees of freedom lost to the level and trend). It is 5.1% (buildings), 3.7% (contents) and 5.6% (motor).
  - **Projection:** the least-squares trend projects 2025 from weights of 1.0 on 2024 and 0.5 on 2023. The variance of the 2025 level is therefore CV₂₀₂₄² + 0.25·CV₂₀₂₃² + 2.5·σ_year².
  - **Mack (A25, D-25):** the development CV used for each line and loss year is the larger of the bootstrap and Mack's standard error of the chain-ladder reserve on quarterly reporting triangles (`python/mack.py`, `ev_u2_mack_development.csv`). Mack adds the process error of claims still to be reported. For 2024 it gives 1.1% (buildings), 0.6% (contents) and **5.4% (motor, against 1.3% from the bootstrap)**. Its CV of the 2024 reserve (14% / 18% / 24%) is close to the Mack analysis on `main` (13% / 17% / 21%).
  - This gives σ of 8.2% (buildings), 5.8% (contents) and 10.4% (motor) in the final model, and the 90% ranges in section 1. Motor's falls from 14.5% because, without the lift, the 11.9m claim no longer swings its development.
- **Mean adjustment ("tilt"):** if the projected level is the median of a lognormal, the mean is higher by exp(σ²/2). With p = 1.5 the deviance-optimal prediction is the mean of the uncertain level (section 8, item 2).
  - In the final model the tilt is +0.34% (buildings), +0.17% (contents) and +0.54% (motor).
  - Like any level change it faces the two-year rule. In the final model it is kept for motor only (×1.0054); buildings' now worsens 2023 (−0.000003) and gains only +0.000001 on 2024, so it fails the rule.
- **Caveats:**
  - Neither the bootstrap nor Mack measures method error: the development backtests over-estimated motor by 3.5% and 5.5% (section 3.4), so the motor range is still somewhat narrow ([#140](https://github.com/almac09/iaa-hackathon-2026/issues/140)).
  - Buildings' year residuals are all positive (+2.3% to +4.8%), so its σ_year mixes in a bias as well as noise.

### 4.10 The 2025 prediction

For each 2025 row:

> prediction = [ small-claim model at 2025 + large-claim GLM × Pareto adjustment × trend to 2025 ] × claim-history factor × rebalancing constant × level factor × uncertainty tilt

Contents has no large-claim term, and its exposure is seasonal. The level factor applies to buildings (×1.0334) and the tilt to motor (×1.0054); no CANN correction is used.

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
| CANN on motor | 2023 −0.0002, 2024 +0.0007 | A20 |
| CANN on contents | +0.00005 a year, but the two-year interval includes zero (D-23) | A22 |
| CANN on all claims, not just small ones (5 seeds, motor probe) | −0.0012 and +0.0008 | A20 |
| Residual boosting on the GLM, isotonic recalibration, frequency boosting, credibility-encoded make and model, a GLM + XGBoost blend | Each failed the two-year rule in scratch probes; no evidence files kept | A20 |
| A level factor from the other backtest year | Worse on the other year, every line | A20 |
| Uncertainty tilt for contents and motor | Worse on 2024 | A20 |
| Missed payments from the previous row (a guard against a same-year leak) | Contents −0.0014 / +0.0000, motor −0.0003 / −0.0003, buildings −0.0004 / −0.0003 | A21 |
| Missed payments dropped | Worse on both years for contents and motor; buildings −0.0011 / −0.0004 | A21 |
| Double GLM (dispersion modelled) | Contents +0.0009 / −0.0004, motor −0.0002 / −0.0001 | A21 |
| Seasonal exposure for buildings and motor | Buildings −0.0005 / −0.0004, motor +0.0002 / −0.0007 | A21 |
| Recent-weighted reporting patterns | Motor and contents developed worse in both backtests | A25 |
| Fitting on reported claims with a development offset | Motor −0.0002 / −0.0003, others neutral | A25 |

### 5.4 Significance: paired customer bootstrap (A20; figures from the final model, A28)

Each kept component is compared with the chosen model without it, on the same rows. Whole customers are resampled (1,000 times), so claims of one customer are never split across resamples. The table gives the D² gain and its 95% interval; **bold** means the interval excludes zero.

| Line | Year | CANN | Claim history | Trend | Large-claim split | A17 changes |
|---|---|---|---|---|---|---|
| Buildings | 2023 | not adopted: +0.00001 (−0.00000, 0.00002) | +0.0008 (−0.0000, 0.0017) | +0.0001 (−0.0005, 0.0007) | +0.0011 (−0.0014, 0.0035) | **+0.0011 (0.0002, 0.0022)** |
| Buildings | 2024 | not adopted: +0.00000 (−0.00002, 0.00003) | **+0.0016 (0.0006, 0.0028)** | +0.0007 (−0.0004, 0.0019) | +0.0021 (−0.0004, 0.0044) | +0.0006 (−0.0005, 0.0016) |
| Contents | 2023 | not adopted: +0.00006 (−0.00001, 0.00013) | **+0.0016 (0.0003, 0.0031)** | **+0.0001 (0.0001, 0.0002)** | — | **+0.0014 (0.0000, 0.0030)** |
| Contents | 2024 | not adopted: +0.00004 (−0.00045, 0.00055) | +0.0005 (−0.0010, 0.0020) | +0.0008 (−0.0003, 0.0021) | — | **+0.0020 (0.0003, 0.0040)** |
| Motor | 2023 | not adopted: −0.00006 (−0.00080, 0.00053) | +0.0009 (−0.0024, 0.0035) | +0.0017 (−0.0021, 0.0069) | −0.0010 (−0.0030, 0.0007) | +0.0048 (−0.0001, 0.0117) |
| Motor | 2024 | **not adopted: +0.00091 (0.00020, 0.00170)** | **+0.0041 (0.0021, 0.0066)** | +0.0009 (−0.0002, 0.0023) | **+0.0034 (0.0004, 0.0066)** | **+0.0023 (0.0011, 0.0036)** |

Reading: claim history and the A17 changes (smoothing, the motor factors) are real on most line-years. The trend and the large-claim split are mostly real but noisy, because single large claims dominate the interval. CANN is at the edge of detection: for motor it is significantly better on 2024 and (not significantly) worse on 2023, which is why the two-year rule rejects it. Motor's split loses on 2023 (the year of the 11.9m claim) and wins clearly on 2024. Source: `audit/evidence/glm_lines/ev_g8_bootstrap_significance.csv`.

### 5.5 Inverting the model: four gaps tested (A21)

The model was examined for what would make it wrong (section 8 lists what remains). Four gaps were not covered, and each was tested under the two-year rule against the chosen model without CANN (`python/gap_tests.py`, `ev_g11`–`ev_g13`):

| Gap | What was tested | Result |
|---|---|---|
| Missed payments may partly record the period's outcome: its claimant gap is 1.3–2.2× larger for the same year's claims than for next year's | The previous row's value (known before the period), and dropping the feature | Both worse on every line; dropping is worse than lagging. The feature carries genuine signal. A leak can't be ruled out from the backtest, but it would be worth at most about 0.0004 on 2024. **Kept.** |
| Gains near noise pass the two-year rule about a quarter of the time | Adoption only if the two-year mean gain is also significant | Would drop CANN for contents, the buildings split and the contents and motor trends. Suitable for discretionary additions such as CANN, not for structural components like the trend. **Adopted for CANN (D-23, A22): contents CANN dropped.** |
| Frequency and severity move in opposite directions for several factors, so one Tweedie dispersion weights rows wrongly | A double GLM: the mean refitted with weights from a gamma GLM of its deviances | Contents +0.0009 / −0.0004, motor −0.0002 / −0.0001. **Fails.** |
| Claims are seasonal (buildings 1.26–1.43 in December to February), but exposure counts every day the same | Exposure weighted by the month's relative frequency, estimated on the fitting years | **Contents passes: +0.0010 / +0.0003**, and its two-year mean (+0.00065, interval +0.00008 to +0.00123) is significant. Buildings and motor fail. **Adopted for contents (D-22, A22).** |

### 5.6 Combining the team's techniques (A24)

Every branch was reviewed (Alan's on `main`, Bridget's on `hackathon-bridget`, Cian's on `data-cian`) and each modelling technique was tested on this model under the same rules (`python/combined_tests.py`, `ev_c1`, `ev_c2`). Blend weights were chosen on one year and scored on the other.

| Technique (from) | 2023 | 2024 | Result |
|---|---|---|---|
| GLM + XGBoost blend (Alan, Bridget) | best weight 0–0.1; motor −0.0020 with 2024's weight | best weight 0–0.2 | Fails on every line |
| XGBoost with the tuning harness's best settings (depth 4, min_child_weight 10, 500 trees) alone (Bridget) | contents −0.089, motor −0.039 | −0.036, −0.012 | Overfits when scored on a later year; early stopping would keep only 55 and 99 trees |
| Smoothed GLM instead of XGBoost for buildings | −0.0071 | −0.0041 | Fails |
| Frequency × severity GLM (Bridget's queue) | contents −0.0006, motor −0.0010 | −0.0009, +0.0007 | Fails |
| Timber codes merged (Cian) | −0.0007 | −0.0004 | Not significant; applied as a data correction (D-24) |

- **Why the other branches rated these techniques higher:** their validation mixed all four years (random customer folds) or chose blend weights on the year they scored. Both reward models that follow year-specific noise. Out-of-time testing does not.
- **Independent check:** this model's 2024 development factors (1.079 / 1.027 / 1.303) agree with the chain-ladder Benktander factors on `main` (1.081 / 1.037 / 1.293).
- **Level:** buildings and contents agree with Bridget's best-scoring file (v014, 0.10916) within 2%. Motor is 16% higher, and it sits at the data workstream's central 2025 benchmark (#91) while v014's motor is below the low end. A leaderboard A/B that differs only in motor is set up in [#151](https://github.com/almac09/iaa-hackathon-2026/issues/151).

### 5.7 Location below region (A26)

- **What the data holds:** the address table has no coordinates, postcode or street, and `address_id` carries no geography (every ID decile has the same region mix and mean elevation). True geocoding is therefore not possible.
- **What varies by place:** each address's weather. Region explains 59–97% of an address's 2021–25 averages; the rest is within-region variation that persists (one address's rainfall correlates 0.81–0.94 between years). It works as a sub-regional climate fingerprint.
- **Residual check:** the production model's actual / expected was examined by region, region × urban density, and quintiles of each address's climate. No group is off in the same direction on both 2023 and 2024 with |z| > 2, on any line.
- **Features tested** (`python/geo_tests.py`, `ev_geo1`, `ev_geo2`; D² change on 2023 / 2024):

  | Feature | Contents | Motor | Buildings |
  |---|---|---|---|
  | Yearly weather (five splines) | +0.0002 / **−0.0031** | −0.0063 / −0.0004 | already in the model |
  | Address climate, 2021–25 average | +0.0001 / −0.0002 | +0.0003 / −0.0012 | −0.0005 / −0.0001 |
  | Region × urban density | **−0.0016 / −0.0010** | −0.0005 / −0.0005 | — |

  Bold means significantly worse. None passes.
- **Shared addresses:** 96% of addresses only ever have one customer, so address-level history from other customers would add little.
- **Conclusion:** region, urban density, flood zone, elevation, distance to river, soil and, for buildings, each year's weather already carry the location signal. With real coordinates, the next test would be spatial smoothing of residuals (territory credibility). [#153](https://github.com/almac09/iaa-hackathon-2026/issues/153).

### 5.8 Final review (A27)

A last sweep for anything missed: eight reviewers with different lenses (target and development, unused features, interactions, model structure, level and calibration, customer history, test set and scoring, team and literature ideas) proposed and measured candidates by rebuilding the production model in memory. A skeptic then re-measured each and tried to refute it, and a completeness critic looked for gaps. Evidence: `audit/evidence/final_review/`.

| Candidate | Line | 2023 / 2024 | Verdict | Issue |
|---|---|---|---|---|
| R11 elevation in feet (data error) | Buildings | −0.0001 / +0.0008 | Worth testing | [#154](https://github.com/almac09/iaa-hackathon-2026/issues/154) |
| `year_built = 1900` placeholder (data error) | Buildings | +0.0003 / +0.0007 | Worth testing | [#155](https://github.com/almac09/iaa-hackathon-2026/issues/155) |
| Remove the bodily-injury lift (D-11) | Motor | changes the target; level −5% to −8% | Worth testing | [#156](https://github.com/almac09/iaa-hackathon-2026/issues/156) |
| Own-line credibility k = 4 | Motor | −0.0002 (one claim) / +0.0013 | Worth testing | [#157](https://github.com/almac09/iaa-hackathon-2026/issues/157) |
| Peril interactions (large-claim GLM; XGBoost) | Buildings | +0.0004 / +0.0006; +0.0012 / +0.0008 | Worth testing | [#158](https://github.com/almac09/iaa-hackathon-2026/issues/158) |
| Calibration power; short part-year rows | Buildings | +0.0005 / +0.0006; +0.0005 / +0.0007 | Worth testing | [#158](https://github.com/almac09/iaa-hackathon-2026/issues/158) |
| Direct-channel flag | Contents | +0.0006 / +0.0007 | Borderline | [#159](https://github.com/almac09/iaa-hackathon-2026/issues/159) |
| Net open own-damage/theft claims of expected recoveries | Motor | target definition; −1.2% on 2024 | Worth testing | [#160](https://github.com/almac09/iaa-hackathon-2026/issues/160) |
| Named young driver and quoted-on-inception-day flag (small-claim GLM) | Motor | −0.00015 / −0.00004 on the full target; +0.00057 / +0.00037 with claims capped at 50k | Narrow: passes only on the capped basis | — (no issue raised) |
| Large-claim GLM at power 1.9; log knots; more trees and seed bagging; count-weighted or quarterly BF development | Motor / buildings | various | Rejected: one-claim or seed artefacts, selected peaks | [#161](https://github.com/almac09/iaa-hackathon-2026/issues/161) |

**Final run (A28).** Items 1–6 were then implemented behind switches and tested under the usual rules (`python/final_tests.py`, `ev_f1`–`ev_f6`; buildings averaged over four XGBoost seeds with the tree count fixed):

| Change | 2023 / 2024 | Two-year mean (95%) | Outcome |
|---|---|---|---|
| Address corrections (D-26, D-27) | buildings +0.0002 / +0.0013; contents +0.0003 / +0.0002 | buildings +0.00075 (−0.00012, +0.00163) | Applied as corrections |
| Buildings peril interactions, both models (D-29) | +0.0011 / +0.0012 | +0.00115 (+0.00030, +0.00200) | **Adopted** |
| Bodily-injury lift off (D-28) | −0.0012 / −0.0005 on the lifted actual; +0.0005 / +0.0012 on the unlifted | — | **Adopted** on the structural and leaderboard evidence |
| Motor own-line k = 4 (D-30) | −0.0002 / +0.0013 (capped at 1m +0.0007 / +0.0012) | — | **Adopted** by the pre-registered rule |
| Buildings short-row factor | +0.0005 / +0.0006 | +0.00055 (+0.000002, +0.00111) in the test; +0.00056 (−0.00002, +0.00114) on the final model | Not adopted: fails on the final model |
| Buildings calibration power | +0.00003 / +0.0003 | +0.00015 (−0.00027, +0.00057) | Not adopted |
| Contents direct-channel flag | +0.0006 / +0.0007 | +0.00062 (−0.00004, +0.00128) | Not adopted |


### 5.9 Statistical tests of the final model (A29)

The GLMs of the final model were put through formal hypothesis tests, at the owner's request: Type III tests of each term, chi-square tests of actual against expected by level, level merging and term dropping scored out of time, and a screen of every pair of terms for interactions (`python/stat_tests.py`, evidence `ev_s1_*`, `ev_s2_*`, `ev_s3_*`). The adoption rules were fixed before the labelled runs. The Type III F test matches R's `glm` with `statmod` to a relative 1e-4 on simulated data (a unit test), and two independent code reviews checked the tests (eight findings confirmed, seven distinct defects, all fixed before the runs).

**Type III tests** on the final fits (2021–24). Each term is dropped and the model refitted. The quasi-likelihood F test divides the deviance change by the Pearson dispersion, on the term's estimable columns. Beside it, a score test at the reduced fit takes the smaller of the cluster-robust statistic and the model-based one with the dispersion the sandwich implies.

| Component | Rows | Columns (rank) | Pearson φ | Effective φ | Terms significant on both tests |
|---|---|---|---|---|---|
| Buildings large-claim GLM | 707,726 | 35 (35) | 15,033 | 13,160 | 4 of 9 |
| Contents GLM | 721,617 | 79 (77) | 1,612 | 965 | 16 of 21 |
| Motor small-claim GLM | 979,968 | 83 (80) | 1,077 | 896 | 18 of 19 |
| Motor large-claim GLM | 979,968 | 28 (28) | 54,631 | 59,072 | 1 of 5 |

- **Not significant on either test** (F p / score p): buildings large-claim flood zone (0.62 / 0.55), soil type (0.41 / 0.40), construction type (0.11 / 0.15), distance to river (0.14 / 0.05) and x_fz3_rain (0.21 / 0.19); contents segment (0.97 / 0.88), bike cover (0.82 / 0.76) and lodgers (0.19 / 0.09); motor marketing segment (1.00 / 1.00); motor large-claim driver age (0.93 / 0.95), years licensed (0.94 / 0.96) and region (0.06 / 0.15).
- **Significant on one test only:** contents policyholder age (0.15 / 0.03) and distance to river (0.28 / 0.050); motor large-claim annual mileage (< 0.001 / 0.05).
- **Aliasing.** Some factors are exactly collinear: marketing segment's budget and mainstream tiers with motor credit band, contents segment with contents credit band, and one direction of motor's voluntary-excess spline. Only the small ridge penalty pins those coefficients, so they cannot be tested one at a time; the tests count estimable columns only. Predictions are unaffected. Segment adds nothing beyond credit band in either line.
- **Dispersion.** The dispersion the sandwich implies is 0.6 to 1.1 times the Pearson one, so the Pearson F tests are about right. The deviance-based dispersion (44 to 111) is far too small and should not be used for these models.

**Calibration by level** on both backtest years: 186 factor-years, 16 with p < 0.05 (about 9 expected by chance) and 2 after Benjamini–Hochberg within line and year. Those are buildings basement in 2023 (A/E 1.33; 1.08 in 2024) and motor third-party-only cover in 2023 (A/E 2.25, the large claim; 1.04 in 2024); neither is significant in 2024 (p 0.31 and 0.75). Only one factor has p < 0.05 in both years, buildings distance to river (0.039 and 0.016), and it does not survive Benjamini–Hochberg in either (q 0.30 and 0.69). Within it, buildings rows with no distance recorded are under-predicted in both years (A/E 1.61 and 1.20; 170 and 155 claims). It was found on the backtest years, so it is raised for testing rather than adopted ([#163](https://github.com/almac09/iaa-hackathon-2026/issues/163)).

**Simplification, scored out of time:**

- **Level merging** (greedy, while the Wald statistic for equal coefficients is below 2; aliased merges free): buildings +0.00034 / −0.00010, contents −0.00056 / −0.00019, motor −0.00025 / +0.00034. Fails on every line.
- **Dropping each term** (54 terms): none adopted. Two improve both years without significance: contents segment (+0.00013 / +0.00020) and motor payment method (+0.00130 / +0.00006). Payment method is strongly significant in-sample (F 58.9) yet costs nothing when dropped, which suggests its relativity is not stable over time ([#164](https://github.com/almac09/iaa-hackathon-2026/issues/164)). Conversely, x_fz3_rain is not significant in-sample in the large-claim GLM, yet dropping it costs significantly out of time (−0.00011, 95% −0.00022 to −0.00001). In-sample p-values do not decide predictive value; the out-of-time test does.

**Interaction screen.** Every pair of terms in each GLM, and of buildings' XGBoost on its own residuals, was screened by the score test on 2021–22 only, so both backtest years stayed unseen. Pairs that only repeat the A28 peril interactions were recorded but not screened (8).

| Model | Pairs screened | p < 0.05 | Smallest p | Smallest BH q |
|---|---|---|---|---|
| Buildings large-claim GLM | 34 | 3% | 0.027 | 0.91 |
| Buildings XGBoost (residuals) | 1,479 | 3% | 0.0052 | 1.00 |
| Contents GLM | 210 | 4% | 0.0010 | 0.16 |
| Motor small-claim GLM | 171 | 11% | 0.0018 | 0.28 |
| Motor large-claim GLM | 10 | 0% | 0.32 | 0.91 |

- **No pair passes Benjamini–Hochberg.** The p-values are close to uniform. Motor's small-claim GLM has more small p-values than chance (11% below 0.05), but no single pair stands out.
- **Positive control.** The screen cannot see the kind of interaction that matters. Refit without the A28 features, buildings' XGBoost gives p = 0.41 for soil type × dry summer and 0.53 for flood zone × rainfall, the two pairs that gain about 0.001 out of time as engineered features (D-29). In an unlogged check before the labelled run, the same pairs reached only 0.03 and 0.05 on 2023 residuals; the BH bar is about 3e-5 (0.05 / 1,479). The trees absorb most of an interaction on the years they are fitted to; its value comes from carrying over to other years' weather. So the three strongest pairs per model were scored regardless of BH, with Holm at the scoring step (an amendment made before the labelled run).
- **Scored out of time:** 15 pairs, none adopted. One passes the two-year rule: buildings XGBoost email domain × branch (+0.00004 / +0.00033; two-year mean +0.00018, 95% −0.00014 to +0.00050), two administrative fields (email provider and servicing branch). That is the chance pass the significance rule exists to stop. The motor small-claim candidates made things worse: cover type × credit band −0.00327 / −0.00058 and marketing segment × cover type −0.00667 / −0.00025.

**Conclusion.** The model's structure stands up to formal testing, and nothing in A29 changes it: the final candidate (A28, public score 0.12805) stands. Terms that look weak in-sample are mostly neutral out of time, and some (x_fz3_rain) are needed. Merging levels does not improve the fit: it fails on every line, and only contents is worse in both years. The screen finds no interaction it can confirm, and the positive control shows why a screen of this kind could not have found the gains that did matter. Those came from reasoning about how claims arise (subsidence on clay in dry summers, flooding in high zones in wet years), so any further gain is more likely to come from new information than from re-specifying the GLMs.

## 6. Submission

- **File:** `work/submissions/submission_final_candidate_20261023.csv` (A28), kept out of git (`work/` is ignored). It has 626,639 rows, with row_ids and order identical to `sample_submission.csv`, no nulls, and every prediction above zero (minimum 0.050). A zero would make the deviance infinite on any row with a claim.
- **Mean prediction per row:** buildings 408.2, contents 103.4, motor 455.9. Against the best-scoring A22 file: ×1.018, ×1.000 and ×0.927.
- **A/B file for the leaderboard:** `submission_combined_v1_20261022_motor_at_v014_level.csv` is combined v1 (`submission_combined_v1_20261022.csv`) with motor scaled ×0.8635 to a mean of 424.91; its buildings and contents are identical to v1. It is not a variant of the final candidate above, which differs from it on all three lines ([#151](https://github.com/almac09/iaa-hackathon-2026/issues/151)).
- **Earlier files** (A20–A22 outputs, renamed by the owner) are kept beside it.
- **Public leaderboard:** the A28 final candidate scored **0.12805** (best). Earlier (A27): the A22 file 0.12753, combined v1 0.12727, the A25 file (`submission_combined_v2_dev_methods_20261022.csv`) 0.12727 (the same predictions as v1 apart from buildings within 0.003%), and combined v1 with motor ×0.8635 0.12725. The leader was 0.13050.
- **2025 cost per unit of exposure against developed 2024:** buildings 500.3 against 484.7, contents 129.2 against 120.0, motor 581.3 against 561.0 (motor's developed 2024 is lower without the bodily-injury lift).

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
| D-11 | Motor bodily injury IBNER lift (lower bound); switched off by D-28 | Superseded (switched off in A28) |
| D-12 | 1m cap and loading (XGBoost run only); replaced by D-15 | Superseded |
| D-13 | Trend to 2025 | Approved; motor trend confirmed after the backtest |
| D-14 | Claim history as credibility | Owner asked for full use of history |
| D-15 | Large-claim split with a Pareto tail (buildings, motor) | Owner's design |
| D-16 | Model structure per line; no copula | Owner's design |
| D-17 | Bornhuetter–Ferguson for immature slow perils | Owner asked for recommendation 1 |
| D-18 | Spline smoothing; two motor factors | Owner asked for recommendations 2 and 3 |
| D-19 | De-duplicate claims; pad motor customer IDs | Applied |
| D-20 | CANN correction per line under the two-year rule: adopted for buildings and contents in A20, contents dropped in A22 (D-23), buildings dropped in A28; not used on any line in the final model | Owner asked for it (implementation 1) |
| D-21 | Level factor and uncertainty tilt, each under the two-year rule (level factor kept for buildings, tilt kept for motor only) | Owner asked for it (implementation 3) |
| D-22 | Seasonal exposure for contents | Adopted by the owner (A22) |
| D-23 | Significance required for discretionary additions such as CANN (contents CANN dropped) | Adopted by the owner (A22) |
| D-24 | Merge buildings timber codes T and W | Adopted with the combined-model plan (A24) |
| D-25 | Development uncertainty: the larger of the bootstrap and Mack | Owner asked for Mack in the ranges (A25) |
| D-26 | R11 elevation converted from feet | Adopted (A28) |
| D-27 | Year built 1900 placeholder imputed | Adopted (A28) |
| D-28 | Motor bodily-injury lift (D-11) switched off | Adopted (A28) |
| D-29 | Buildings peril interactions | Adopted (A28) |
| D-30 | Motor credibility k tuned on capped claims | Adopted (A28) |
| D-31 | Week 2: carry the developed relativities over through a portable model (shared factors only) | Awaiting sign-off (A30) |
| D-32 | Week 2: reset each line's level to the emerging 2023-24 experience (buildings pooled as a judgement) | Awaiting sign-off (A30) |
| D-33 | Week 2: developed reporting pattern for emerging motor; own pattern for the home lines | Awaiting sign-off (A30) |
| D-34 | Week 2: no emerging factor corrections, emerging-only model or claim history | Awaiting sign-off (A30) |
| D-35 | Week 2: keep the developed treatment of missing values | Awaiting sign-off (A30) |
| D-36 | Week 2: gender excluded | **Owner's decision** (A30) |
| D-37 | Week 2: new-business loading in a scenario file only (reopened as motor-specific, A31) | **Owner's decision** (A30) |
| D-38 | Week 2: C0 stays the ranking model; no A31 variation adopted | Awaiting sign-off (A31) |
| D-39 | Week 2: levels by segment (candidate file, probes, final selection) | **Owner's decision** (A31) |

D-08 records the XGBoost feature set used in A13 only. No decision is formally signed off.

## 8. Limitations and open questions

1. **Development is the largest uncertainty.**
   - Motor bodily injury: even with BF, the backtest still over-predicts by 13%, and its "actual" is itself 49% open.
   - Subsidence: 2024 is 59% reported.
   - The bodily-injury IBNER lift was switched off in A28 (D-28): closed claims fell as much as open ones, so the fall in claim size is treated as real.
2. **Level.** The backtest under-predicts buildings 2024 by 6.6% in total and over-predicts contents by 3.9% and motor by 0.5%. Level is the largest single source of error: on contents, moving 2023's level by 2% costs about 0.0006 D², ten times the CANN gain. Only buildings' level factor survives the two-year rule (×1.033); contents and motor fail it (section 4.9).
   - **Correction (A20):** an earlier version of this document said under-prediction costs far more D² than over-prediction (8% low 0.0015 against 8% high 0.0003). That was wrong and had no audit entry. For p = 1.5 the deviance of predictions scaled by r from their optimum grows as 2·cosh(ln r / 2) − 2 ≈ (ln r)²/4, which is symmetric in log terms. So 8% too low costs only about 18% more than 8% too high. The deviance-optimal level is the *mean* of the uncertain level, which is what the tilt in section 4.9 targets. There is no case for deliberately predicting high.
   - The deviance-optimal scale is not the ratio of totals, so "predicted / actual" in section 1 is not the measure to correct.
3. **Trends rest on four years.** Year-to-year noise around the fitted trend is 3.7% (contents) to 5.6% (motor); see section 4.9.
   - The 90% range for the 2025 level runs from −13% to +14% (buildings), −9% to +10% (contents) and −16% to +19% (motor). This includes Mack's chain-ladder error but not the development method error.
4. **Selection on the same years.** 2023 and 2024 have informed many choices, so the backtest gains are somewhat optimistic. The leaderboard is the independent check, but it should confirm choices, not drive them.
5. **Seasonality** is modelled for contents only. Buildings has the stronger winter peak, but seasonal exposure made it worse on both years (A21). The reason is not established.
6. **Small gains near the noise.** The final review's gains are each a few ten-thousandths of D², found and judged on the same two years. Under the two-year rule, a change with no real effect passes roughly a quarter of the time (if the two years behave independently), so significance is reported alongside it (section 5.4).
7. **Implementation coverage.** The new models are implemented in Python only. GLM A, the contents benchmark, has an independent R implementation that reconciles (101 checks, 0 failed).
8. **Synthetic data.** It has no shared events between buildings and contents. A real portfolio would, and the independence assumption would need revisiting for Week 2.
9. **Data in a public repository** (section 2.2, issue 17) is unresolved.
10. **What formal tests can and cannot show (A29).** In-sample p-values did not decide predictive value. A term that fails in-sample can be needed out of time, and one that passes can add nothing. An exhaustive interaction screen at the usual thresholds could not detect the interactions that matter here (section 5.9). The out-of-time test remains the arbiter. Two of the GLMs carry exactly aliased factors, held only by the ridge penalty; that is harmless for prediction but means their coefficients cannot be read one at a time.

### 8.1 Risk register: what would make the model wrong (A21)

Each risk found by inverting the model, what has been done, and the GitHub issue that tracks it. Resolved work is in closed issues [#128](https://github.com/almac09/iaa-hackathon-2026/issues/128)–[#139](https://github.com/almac09/iaa-hackathon-2026/issues/139).

| Risk | Status | Issue |
|---|---|---|
| Wrong 2025 level | Measured: 90% ranges per line. Only buildings' level factor passes the two-year rule and is adopted (×1.0334); motor carries a +0.5% mean adjustment (uncertainty tilt ×1.0054). | [#131](https://github.com/almac09/iaa-hackathon-2026/issues/131) (closed) |
| Ranges too narrow: development method error and tail-fit uncertainty left out | Mack's error added (A25); method error and tail fit **open** | [#140](https://github.com/almac09/iaa-hackathon-2026/issues/140), [#152](https://github.com/almac09/iaa-hackathon-2026/issues/152) (closed) |
| Leakage | Contacts excluded; missed payments tested and kept; categorical features **not yet scanned** | [#135](https://github.com/almac09/iaa-hackathon-2026/issues/135) (closed), [#141](https://github.com/almac09/iaa-hackathon-2026/issues/141) |
| Selection on the same two years | Significance now required for discretionary additions (D-23); structural components stay under the two-year rule | [#138](https://github.com/almac09/iaa-hackathon-2026/issues/138) (closed) |
| A kept component that fails the two-year rule: motor large-claim split | **Open**: owner decision on D-15 | [#142](https://github.com/almac09/iaa-hackathon-2026/issues/142) |
| Buildings under-predicts every fitted year in total | **Open** | [#143](https://github.com/almac09/iaa-hackathon-2026/issues/143) |
| Dispersion (frequency and severity move in opposite directions) | Double GLM tested and rejected; a separate frequency × severity model not tried | [#136](https://github.com/almac09/iaa-hackathon-2026/issues/136) (closed) |
| Seasonality | Adopted for contents (D-22); buildings and motor rejected, reason for buildings **open** | [#137](https://github.com/almac09/iaa-hackathon-2026/issues/137) (closed), [#144](https://github.com/almac09/iaa-hackathon-2026/issues/144) |
| Train and test rows built differently | Checked: no difference | [#134](https://github.com/almac09/iaa-hackathon-2026/issues/134) (closed) |
| Silent failure of the full run | Worker fails loudly; torch kept out of the main process; **no end-to-end test yet** | [#129](https://github.com/almac09/iaa-hackathon-2026/issues/129) (closed), [#145](https://github.com/almac09/iaa-hackathon-2026/issues/145) |
| Techniques from the team's other branches | Tested out of time; none passes; one data fix adopted (D-24) | [#148](https://github.com/almac09/iaa-hackathon-2026/issues/148), [#149](https://github.com/almac09/iaa-hackathon-2026/issues/149) (closed) |
| Motor 2025 level (final candidate ×0.927 of the A22 file, about 7% below it; ×1.073 of v014's level) | A/B scored (A27): motor at v014's level 0.12725 against 0.12727 for combined v1, so the lower level did not help; now covered by the D-28 row below | [#151](https://github.com/almac09/iaa-hackathon-2026/issues/151) |
| Location information below region | No coordinates in the data; weather-based proxies and region × urban tested; none passes and no residual pattern | [#153](https://github.com/almac09/iaa-hackathon-2026/issues/153) (closed) |
| Data errors in the address table (R11 elevation in feet; year built 1900) | Corrected (A28, D-26, D-27) | [#154](https://github.com/almac09/iaa-hackathon-2026/issues/154), [#155](https://github.com/almac09/iaa-hackathon-2026/issues/155) (closed) |
| Motor level and the bodily-injury lift | Lift switched off (D-28); motor now ×0.927 of the A22 file's level, near the leaderboard's ×0.93. The final candidate scored 0.12805, the best so far | [#156](https://github.com/almac09/iaa-hackathon-2026/issues/156) (closed), [#151](https://github.com/almac09/iaa-hackathon-2026/issues/151) |
| Further gains measured in the final review | Interactions and motor credibility adopted; calibration power, short-row factor and contents channel failed the rules (A28). Recoveries on open claims **open**; motor named-young-driver and quote-day flags (narrow pass, needs an owner ruling on the capped basis; not tested in A28) **open** | [#157](https://github.com/almac09/iaa-hackathon-2026/issues/157)–[#159](https://github.com/almac09/iaa-hackathon-2026/issues/159) (closed), [#160](https://github.com/almac09/iaa-hackathon-2026/issues/160) |
| Model structure untested by formal hypothesis tests | Done (A29): Type III, chi-square by level, level merging, term dropping and an interaction screen; nothing passes the rules; two observations open | [#163](https://github.com/almac09/iaa-hackathon-2026/issues/163), [#164](https://github.com/almac09/iaa-hackathon-2026/issues/164) |
| Pack D² not comparable with the leaderboard | **Open** (model choices unaffected) | [#146](https://github.com/almac09/iaa-hackathon-2026/issues/146) |
| Decisions not signed off | **Open** | [#147](https://github.com/almac09/iaa-hackathon-2026/issues/147) |

The public-repository data question (item 9) is not raised as a public issue, because a public issue would draw attention to the possible breach. It is for the team to settle privately (A02, A18).

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
python python/level_uncertainty.py
```
```bash
python python/run_glm_lines.py
```
```bash
python -m pytest -q tests
```

- `level_uncertainty.py` must run first: it writes the development bootstrap (`ev_u1`) that `run_glm_lines.py` reads. It takes about a minute on 14 processes.
- `run_glm_lines.py` needs `work/sample_submission.csv`, the competition's sample file. It takes about 20 minutes on an M3 Max and writes the submission and the `audit/evidence/glm_lines/` tables. The CANN networks use the Apple GPU when PyTorch finds it (MPS) and fall back to the CPU otherwise, which is much slower.
- `python python/gap_tests.py` (optional, about 10 minutes) repeats the A21 gap tests; it reads `ev_g6` and `ev_g8` from the last full run.
- `python python/combined_tests.py` (optional, about 30 minutes) repeats the A24 tests of the team's techniques; `python python/development_tests.py` (about 5 minutes) repeats the A25 offset-fit test.
- `python python/geo_tests.py` (about 6 minutes) repeats the A26 location tests; `python python/final_tests.py` (about 40 minutes; `--tests T1` etc. to run parts) repeats the A28 tests.
- Any runner accepts `--run-label "<text>"`, which is stored in its run-log record (A25).
- PyTorch runs only in its own worker process (`python/cann_torch.py`). Do not import torch in any process that uses XGBoost (A20).
- Each run appends a record to `audit/run_log.jsonl`: input, SQL and code hashes, decision switches, package versions and headline metrics.
- **Expected results:** reconciliation ends with `0 failed`; the 2024 backtest matches section 1; 51 tests pass.

## 10. Where to find things

| What | Where |
|---|---|
| Data preparation | `sql/01`–`sql/08` (contents), `sql/09`–`sql/13` (all lines, development, history) |
| Current model | `python/run_glm_lines.py`, `python/glm_lines.py` |
| CANN correction | `python/cann.py` (client), `python/cann_torch.py` (GPU worker) |
| Significance and level uncertainty | `python/significance.py`, `python/level_uncertainty.py` |
| Gap tests from inverting the model | `python/gap_tests.py` |
| Earlier XGBoost runs | `python/run_xgb.py` (A13), `python/run_lines.py` (A15) |
| Development backtest | `python/backtest_development.py` |
| Parameters and decision switches | `config.json` |
| Narrative audit log | `audit/AUDIT_LOG.md` |
| Decisions for sign-off | `audit/DECISIONS.md` |
| Evidence tables | `audit/evidence/glm_lines/` (current), `audit/evidence/lines/` (A15 and the development backtest), `audit/evidence/python`, `audit/evidence/r` (GLM pack) |
| Tests | `tests/` |

## 11. Week 2: the emerging market (A30)

**The task.** Predict the 2025 claim cost of every row in the three emerging-market test files (209,063 rows). The insurer wrote business there in 2023 and 2024, and takes on a broker portfolio at the start of 2025, so 76% of 2025 rows are new business (`docs/kaggle_part2_brief.md` at the repository root). The work is in `python/emerging.py` on branch `contents/gerard-week2-emerging`; the evidence is in `audit/evidence/emerging/`.

### 11.1 What changed

| | Developed (Part 1) | Emerging |
|---|---|---|
| History | 2021-24 | 2023-24 |
| Claims a year | about 6,600 buildings, 6,000 contents, 28,000 motor | about 520, 500 and 3,600 |
| Rating information | exact ages, mileage, telematics, credit band, regions R01-R12 | age bands, gender, regions E01-E06, construction coded BRK/MIX/STN/WOD |
| Missing values | 2.5% in each field | about 12% in 15 fields, apparently at random |
| Claim size | | motor about 0.61 of developed; home lines 0.68-0.79 |
| 2025 book | like the training years | 3.9 times 2024 by exposure (3.5 by rows), 77% new business by exposure (76% by rows), younger and less experienced |

The Part 1 data errors recur (exposure in days, sum insured 1,000 times too large, year built 1900), and the Part 1 fixes cover them. Buildings fire is the largest single difference: 45% of emerging buildings cost, total losses at 4.7 times the developed rate, and 49 fires in 2023 against 21 in 2024, concentrated in region E02.

### 11.2 What carries over

The same SQL runs on both markets through views that give each market the other's columns. Each line's model is refitted on the developed market with only the shared factors (the portable model, D-31) and tested on the emerging years:

- **Relativities carry over; the level does not.** As it is, the portable model's A/E is 1.19 and 0.70 for buildings (2023, 2024), 0.80 and 0.73 for contents and 0.71 and 0.77 for motor, while it ranks emerging risks (level-matched D2 up to 0.116).
- **Losing the motor factors costs ranking.** On the developed 2024 backtest, motor's D2 falls from 0.056 to 0.044 without mileage, telematics, exact age, credit band and region.
- **The level reset is what helps** (D-32). Fitted on one year and scored on the other, it improves contents and motor in both directions (D2 +0.009 to +0.012). Buildings' single-year levels (×1.19 and ×0.68) do not carry from one year to the other, so its pooled level is a judgement.
- **Nothing else passes** (D-34, D-35): corrections for region, new business, age band and channel, an emerging-only model, claim history (motor +0.0026, 95% −0.0008 to +0.0071) and filling the missing values with developed values.
- **Development** (D-33): motor uses the developed reporting pattern. As at 2023-12-31, the emerging pattern over-projected 2023 bodily injury by 23% (developed pattern −5.5%). The home lines keep their own patterns: their projection errors (buildings −1.9%, contents +0.6%) are within 2% of the developed pattern's (+1.0% and +1.8%).

### 11.3 The 2025 prediction

Each line's portable model, projected to 2025 with the developed trend, times its level:

| Line | Level (90% range) | 2025 cost per unit of exposure (90% range) | Emerging 2024 actual |
|---|---|---|---|
| Buildings | ×0.917 (0.42-1.98) | 460.7 (213-995) | 326.6 |
| Contents | ×0.750 (0.66-0.85) | 98.2 (86-112) | 88.3 |
| Motor | ×0.768 (0.63-0.93) | 545.6 (450-661) | 454.9 |

- **Why 2025 is above 2024:** the developed trend (×1.037 to ×1.047), the mix of the 2025 book, mostly the broker portfolio (motor ×1.143: younger and less experienced on the developed relativities), and, for buildings, a level pooled over a heavy and a light fire year.
- **The ranges** combine the sampling error of the level, year noise at the larger of the developed and emerging between-year values, development error scaled up for the smaller book, and trend error. Buildings is wide because its two years disagree far more than the developed market's year noise allows. With 2023 fire at the 2024 rate its level would be ×0.716.
- **Files** (in `work/`): the adopted prediction (`submission_emerging_level_reset_20261026.csv`), the portable model as it is (`…_as_is_…`), and a new-business scenario (`…_level_reset_nb_scenario_…`).

### 11.4 Uncertainty and the owner's decisions

1. **New business** (D-37), the largest. In 2023-24, emerging new business claimed 1.30 times renewals for motor (95% 1.01-1.68) and 1.27 for contents (0.82-1.77), beyond what the rating factors explain. The developed market shows 1.03 and 1.01. Blended by credibility (Z = n/(n + 1,000), an assumption), the loadings would move the 2025 lines by ×1.089 (motor), ×1.020 (contents) and ×1.035 (buildings); the emerging-only loadings by ×1.207, ×1.182 and ×0.872. The 2024 holdout is 9% new business and cannot test this, so it is not in the adopted file.
2. **Buildings fire.** About 70 fires drive the buildings level. The 2023 excess is unexplained, and 22% of 2025 buildings exposure lies beyond the developed wildfire range.
3. **Gender** (D-36). It is predictive for motor frequency (×1.32 for men) and excluded here. Whether it may be used is a legal and ethical question for the owner.
4. **Leaderboard result** (corrected in A31). The portable model as it is scored 0.15601, the new-business scenario 0.15451 and the level reset 0.15021; the leader 0.17303. The files are rescales of as is by line and new business. Read that way, the board puts motor new business at 0.99-1.18 of as is, motor renewals at 0.70-0.95 and contents at 0.70-0.93 (buildings not identified): the level reset suits renewals and contents but is too low for motor new business. An earlier reading (one common scale, ×0.91, worth +0.003) was wrong. D-32 and D-37 are reopened (section 11.5).
5. **Not built, and why:** a separate fire model (too few claims to validate); peril-level rebasing (the line models are not peril-based); a motor large-claim threshold rescaled to emerging costs (the level reset scales the large-claim part with the rest); written premium as an offset (D-05).

### 11.5 Model variations, data suitability and levels (A31)

**Can the ranking be improved? Not on the 2023-24 evidence.** Nine variations were each explored on half of the 2024 emerging customers and confirmed once on the other half, under a rule fixed in advance. They were: written premium (as a diagnostic), an emerging residual XGBoost, a joint developed and emerging XGBoost, frequency × severity, Bühlmann–Straub credibility, a buildings treatment, corrections on capped claims, an emerging-only XGBoost with blends, and grouping sparse levels with effects pooled across lines. Four fixes to C0's inputs from the data audit followed: imputing the fields missing at random, vehicle age minus 2, a calibration power and the out-of-range customer-since year. None beats C0 on the held-back half; the best on all of 2024 is +0.0003. Four adversarial reviews found no measurement fault that could hide a gain. Corrections fitted on one year chase its few large claims: motor new business ran 1.81 times C0 in 2023 and 0.93 in 2024.

**Is the data suitable?** Yes, with the treatments already in place. C0 uses no leaky field (customer contacts, renewal decision). New business is encoded five ways, and the one value outside the training range is inert in the buildings trees. The fields about 12% missing are missing at random. The 2025 new business is January, full-year business, unlike training's mid-year starts. A handful of claims carries much of any one year's score, so small gains on 2024 are noise.

**Where the score can move: levels by segment.** Read by line and new business, the public scores put motor new business well above the level reset (0.99-1.18 of as is, against 0.768), with motor renewals and contents near it. Levels by segment are worth at least +0.0024 on the public board, plausibly up to about +0.009. A candidate halfway between as is and the level reset with the emerging motor new-business loading (×1.296) is predicted at 0.1562-0.1586 (median 0.1573), better than as is in 99-100% of consistent solutions. Submitted by the owner, it scored **0.15742**, the best emerging score so far, close to the predicted median. With that fourth score the board pins motor new business at 1.06-1.11 of as is; raising only that segment to ×1.05 was predicted at 0.1578-0.1582 and scored **0.15805**, the best emerging score so far (`submission_emerging_level_midpoint_motornb105_20261026.csv`; D-39). Further level changes would mostly fit the public board. Probe submissions could identify the levels; the private board decides, so probed levels should be shrunk towards 2023-24.

**The gap to the leader (0.017)** is unlikely to be noise alone or levels alone. Most of it is probably risk ranking that the 2023-24 data cannot reveal: the 2025 book, 77% new business, behaves differently from 2024.

**Office premium.** `python/office_premium.py` builds the annual office premium from the risk premium and the expense table (cat reinsurance on property lines, fixed expense, then expense, commission, tax and profit as shares of premium). For the 20 reference policies, C0's office premium is a median 4.5, 2.8 and 2.5 times the written premium (buildings, contents, motor). The emerging table lacks a comparison-site commission; the developed 10% stands in, an assumption for the owner.

