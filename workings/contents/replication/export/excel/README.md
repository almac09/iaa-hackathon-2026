# Export for the Excel replica of the Week 2 model

C0, the portable developed model, with the segment levels of the final file. Written by `python/export_excel.py`
(run `7db04db6` in `audit/run_log.jsonl`, label "A34 export: buildings trees added"). Every number is at full
precision; nothing is rounded.

## The final file

`work/submissions/submission_emerging_level_midpoint_motornb105_20261026.csv`, sha256
`7799b90105d361d7df29d3979193826d760bbf00bd51a97f91ca123a0a65b3bf`, public score 0.15805, the best of the five
emerging submissions. The competition rules (the repository's `docs/kaggle_rules.md`, item 2.2.b) use the best
public score when no Final Submission is selected; they are written for Part 1, and the same is assumed for Part 2.
The formal choice is the owner's (D-39).

It was built in three steps, and the export reproduces each one bit for bit:

1. **As is** (`submission_emerging_as_is_20261026.csv`, sha256 `61ab17cc…`): C0's 2025 prediction, from
   `python/emerging.py` E4 in run `f5458f45` ("A30 emerging: all steps on the committed code (window fix)",
   commit `363408e5`). The model code is unchanged since then apart from comments, and the refit here matches the
   file on every row, bit for bit.
2. **Midpoint** (`submission_emerging_level_midpoint_20261026.csv`): each segment (line x new business or renewal)
   times its halfway scale (A31 `audit/exploratory/a31/levels/candidates.py`).
3. **Final**: motor new business times 1.05, then divided by its halfway scale, so it ends at 1.05 of as is
   (A31 `write4.py`). Every other row is the midpoint value as `write4.py` read it (see Precision).

## How one prediction is built

```
final = (small + large) x step 1                          buildings, contents, motor renewals
final = (small + large) x step 1 x 1.05 / step 1          motor new business (in that order)

small  (contents, motor)  = exp(intercept + sum of term contributions + trend in 2025) x exposure
small  (buildings)        = exp(ln(exposure) + ln(base rate) + trend in 2025 + one leaf from each of 193 trees)
                            (see The buildings trees)
large  (buildings, motor) = exp(intercept + sum of term contributions + trend in 2025 + ln(Pareto adjustment))
                            x exposure
large  (contents)         = 0 (contents has no large-claim split)
exposure                  = (period end - period start + 1) / 365, plain (D-01; no seasonal weights)
```

Take each trend from the `log contribution in 2025` row of `terms.csv` (or the multiplier in `scalars.csv`); do not
recompute slope x (2025 - origin) in Excel, because the origin needs 17 significant digits.

C0 applies **no** claim-history factor, history rebalance, level-uncertainty tilt, developed level factor,
rebalancing constant, CANN correction, seasonal exposure weights or missing-value fill. `scalars.csv` lists each one
with `applied` = False, and gives the developed production value where one is recorded (motor tilt 1.005425,
buildings level factor 1.03338, the credibility k by group).

## Files

| File | Item | Rows | Committed |
|---|---|---|---|
| `terms.csv` | 2 | 8,100 | yes |
| `spline_pieces.csv` | 2 | 54 | yes |
| `buildings_xgboost.json` | trees | 193 trees | yes |
| `buildings_xgboost_features.csv` | trees | 218 | yes |
| `buildings_xgboost_nodes.csv` | trees | 2,577 | yes |
| `input_mappings.csv` | 3 | 162 | yes |
| `scalars.csv` | 4 | 80 | yes |
| `reconciliation.csv` | 8 | 46 (40 checks, 6 for information) | yes |
| `rows_components.parquet` | 5 | 209,063 | no |
| `rows_inputs_<line>.parquet` | 6 | 52,295 / 66,247 / 90,521 | no |
| `rows_glm_terms_<line>.parquet` | extra | as above | no |
| `reference_office_premium.parquet` | 7 | 20 | no |

The Parquet files hold competition data (rules section 5b) and are git-ignored here and in the pack. Do not commit
or share them outside the team.

## terms.csv (item 2)

One long table: `line, component, term, kind, level, value, band_lower, band_upper, contribution, note`.
`contribution` is on the log scale. Components: `small` (contents, motor) and `large` (buildings, motor).

| kind | How to use it |
|---|---|
| `intercept` | add `contribution` |
| `categorical` | map the input to its model level with `input_mappings.csv` (section "categorical level"), then add that level's `contribution`. The base level has 0; a level the fit did not see, including `MISSING` where the term has no `MISSING` row, also scores 0 (`(any other level)`) |
| `band` | a numeric cut into bands: the band with `band_lower < x <= band_upper`, or the `MISSING` row if x is missing |
| `spline value` | look up x in `value` and add `contribution`. Each lookup runs from the smallest to the largest value in that line's emerging rows, at the column's recorded precision: elevation in steps of 0.1, distance to river 0.01, year built and the motor counts 1, voluntary excess 10, sum insured 1,000. Every 2025 value is on its grid. For any other value use `spline_pieces.csv` |
| `spline missing` | if x is missing, add this **instead of** the curve. Compare it with the curve's values, not with 0: a missing value switches the curve off |
| `continuous` | `contribution` x the value, on the raw scale. `log_si` = ln(max(sum insured, 1)) with sum insured after D-02 (contents; see `input_mappings.csv`); `x_clay_dry` and `x_fz3_rain` (buildings) are defined in `note` |
| `trend` | three rows: the slope per year, its origin (the exposure-weighted mean fitting year), and the log contribution in 2025, whose `value` is 2025 - origin and whose `note` gives the multiplier. A large-claim GLM has no year term of its own: it takes the small-claim slope from its own origin, which equals the small-claim origin (same fitting rows) |
| `offset` | add ln(exposure) |
| `scale` | the Pareto adjustment of a large-claim GLM: `value`, and ln of it in `contribution` |

Buildings small claims are XGBoost (193 trees, depth 3) and have no terms: see The buildings trees.

## spline_pieces.csv

Each spline as cubic polynomials: `f(x) = a0 + a1 d + a2 d^2 + a3 d^3`, `d = x - origin`, for `x` between `lower`
and `upper` (`lower_included` / `upper_included` say whether each end belongs to the piece). Below the first knot
and above the last the curve is constant. The pieces match the fitted curves to within 2e-15 on the log scale, and
the export rebuilds the final file through them as well as through the lookups.

## The buildings trees

`buildings_xgboost.json` is the buildings small-claim model, saved by XGBoost 3.2.0 (`booster.save_model`) from
the refit that reproduces the as-is file. It holds the tree structure, split points, leaf values, the category
levels and per-node training totals (hessian sums and gains, each leaf at least `min_child_weight`); no rows. Load
it with XGBoost 3.2.0; other versions are untested. Its prediction for a row in 2025:

```
margin = ln(exposure) + ln(350.1811758423766) + 0.04587525922327018 x (2025 - 2022.5232987657048)
small  = exp(margin + one leaf from each of the 193 trees)
```

The base rate, slope and origin are in `scalars.csv`. The file also stores a `base_score` (292.2078), which is not
part of the prediction: XGBoost ignores it when the margin is given as `base_margin`. Adding its log to the margin,
as XGBoost does when `base_margin` is left out, would make the prediction 292 times too high.

With XGBoost, as `python/xgb_models.py` does (`matrix`, then `predict`), which reproduces `small_claim` bit for bit:

```python
booster = xgb.Booster(model_file="buildings_xgboost.json")
# X: the 52 features in the order of buildings_xgboost_features.csv; numeric columns as float with NaN for missing;
# each categorical as pd.Categorical(values, categories=its levels in code order), the values as text exactly as
# in the table ("True", not True or 1; branch_id "3", not 3.0), a missing value first replaced by the text
# "MISSING" and any value that is not a level set to NaN
pred = booster.predict(xgb.DMatrix(X, base_margin=margin, enable_categorical=True))
```

Without XGBoost (another engine, or Excel), use the two tables:

- `buildings_xgboost_features.csv`: `position, feature, type, code, level, note`. The 52 features in the model's
  order, 26 numeric then 26 categorical. A categorical input enters as its level's `code`, matched on the exact
  text of `level`: booleans are `True` and `False` here (not `true`/`false` as in `terms.csv`, nor 1/0), and
  integer levels such as `branch_id` have no decimals (`3`, not `3.0`; reading a CSV column that has a blank turns
  `3` into `3.0`). A missing value is first the text `MISSING`, which has a code where the fit saw it; a value with
  no code (`(any other level)`) is missing. Every 2025 value has a code. `x_clay_dry` and `x_fz3_rain` are defined
  in `note`, and `sum_insured` is after D-02.
- `buildings_xgboost_nodes.csv`: `tree, node, kind, position, feature, threshold_float32, threshold, right_codes,
  right_levels, missing_goes, left, right, leaf_value`, one row per node. Start each tree at node 0:
  - `split`: go to `left` if x < `threshold`, otherwise to `right`;
  - `category split`: go to `right` if the code is in `right_codes` (the levels are in `right_levels`), otherwise
    to `left`. Match whole codes: in Excel, look for `";"&code&";"` in `";"&right_codes&";"`, because a plain
    search finds 2 inside 12 and would reprice most rows;
  - a missing input (a missing number, or a categorical with no code) goes to `missing_goes` instead;
  - `leaf`: take `leaf_value`; add one leaf per tree.

  A child always comes after its parent, and no tree has more than 15 nodes (depth 3).

XGBoost holds inputs, thresholds and leaves in single precision and adds the leaves in single precision:

- Compare x with `threshold`, not `threshold_float32`. `threshold` is the double that gives XGBoost's branch for
  every input. With `threshold_float32` (the threshold as XGBoost holds it), an input just below it that rounds up
  to it in single precision, such as 12.3, would go left, where XGBoost sends it right. On the 2025 rows that would
  reprice 10,598 of the 52,295 rows, by up to 36% (median 1.2%), mostly through discount, elevation and wind
  exposure; with `threshold` every row takes XGBoost's branches.
- `leaf_value` and `threshold_float32` are the single-precision values written out exactly. To reproduce
  `small_claim` bit for bit, start a single-precision total at the margin rounded to single precision, add each
  tree's leaf in turn, tree 0 to 192, and take exp in single precision as XGBoost does (C `expf`). Checked on
  macOS (arm64), where numpy and XGBoost both call the system `expf`; an exp rounded correctly instead differs in
  the last bit on 333 rows. Adding the leaves in double precision, as Excel does, differs from `small_claim` by up
  to 7.3e-6 relative.

For quotes: the trees were fitted on the developed market, 2021-24. A quote must go through the same input
mappings as the 2025 rows (`input_mappings.csv`: the conforming view and the data fixes, D-02 at 10m), and its
year enters only through the trend: for 2026, slope x (2026 - origin) raises the price by exp(0.04588) = 1.047
against 2025. Whether to carry the trend forward is a pricing decision. A categorical input with no code is scored
as missing without any error, which changes the price (a `branch_id` of `3.0` for `3` reprices almost every row),
so a quote engine should count such inputs and reject or flag them; for the 2025 rows the count is 0
(`reconciliation.csv`).

## input_mappings.csv (item 3)

`section, line, component, term, source_value, mapped_value, note`. Sections:

- `conforming view`: how each emerging table is given the developed columns (missing ages, mileage, telematics
  and credit band; construction codes mapped);
- `construction code`: BRK/MIX/STN/WOD to brick/non-standard/stone/timber. This happens in the view, so
  `construction_type_raw` already holds the category;
- `age band`: the emerging bands are used as given; the developed rows were cut the same way;
- `dropped factor`: region, occupation, credit band, exact ages, mileage and telematics, which the portable model
  leaves out (D-31);
- `data fix D-01` to `D-27`: exposure from the dates; sum insured divided by 1,000 at or above 1m for contents
  (135 of the 66,247 2025 rows, raw 9.2m to 98.1m) or 10m for buildings (104 of 52,295, raw 60m to 568m); R11
  elevation (no emerging address is in R11); the year-built placeholder by energy rating, from the emerging
  addresses. Apply D-02 to the raw test-file value before anything else: a contents row left undivided prices
  about 215 times too high (1,000 to the power of the `log_si` coefficient, 0.777);
- `contents feature`: the SQL behind each contents `f_` input;
- `missing values`: the rules below;
- `categorical level`: every value in the 2025 rows and the model level it scores as. Only contents prior claims
  and missed payments of 3 are merged, into OTHER.

## scalars.csv (item 4)

`line, segment, item, value, applied, note`: the small-claim model and its trend (slope, origin, 2025
multiplier); the XGBoost base rate and settings, and the `base_score` its file stores but the prediction does not
use; the generalised Pareto fit and adjustment (buildings x1.0254,
motor x0.7651); the GLM penalties; the level steps by segment; and what C0 does not apply.

The halfway scales are:

- buildings 0.9574847257267345;
- contents 0.8657493863699818;
- motor renewals 0.8614202024859916;
- motor new business 0.9807983046341449, then x1.05 / that scale.

The buildings and contents scales are the square roots of the 2023-24 levels as written, rounded to 6 dp, in
`audit/evidence/emerging/ev_w4_level_2025.csv` (0.916777 and 0.749522), as `candidates.py` read them. The motor scales are recovered from
the as-is and midpoint files, because A31's cache was deleted. In each segment exactly one of the nine nearest
doubles reproduces the midpoint file.

## rows_components.parquet (item 5)

One row per 2025 emerging test row, in the final file's order:

| Column | Meaning |
|---|---|
| `row_id, line, new_business, exposure, segment, reference_policy` | the row; `segment` is line_nb or line_ren; 20 reference policies |
| `small_claim` | small-claim expected cost in 2025, trend included |
| `small_trend_multiplier` | divide `small_claim` by it for the value at the trend origin |
| `large_claim_glm` | exp(intercept + terms) x exposure: the large-claim GLM before scaling (blank for contents) |
| `pareto_adjustment, large_trend_multiplier` | the two multipliers on it |
| `large_claim` | `large_claim_glm` x Pareto adjustment x trend multiplier (0 for contents) |
| `as_is` | `small_claim` + `large_claim`, from the refit |
| `as_is_file` | the as-is file as written (identical to `as_is`) |
| `history_n_<group>`, `history_e_<group>` | claims in the row's history before its start (sql/12), and the claims a frequency GLM without history, fitted on emerging 2023-24, expects over the same rows (sql/13). Groups: own, home and cross; motor has no home group, so those columns are blank |
| `history_factor` | 1: claim history is not applied (D-34) |
| `level_step1_scale, level_step2_times, level_step2_divide` | the level steps |
| `as_is_file_as_read` | the as-is file as `candidates.py` read it (see Precision) |
| `midpoint_prediction` | `as_is_file_as_read` x step 1: the midpoint file, bit for bit |
| `midpoint_file_as_read` | the midpoint file as `write4.py` read it |
| `final_prediction` | the final file, bit for bit |
| `effective_level` | `final_prediction` / `as_is_file` |
| `final_from_components` | (small + large) x the steps: the final file to within 1.8e-15 |

To replace the buildings trees with a substitute, fit it to `small_claim`.

## rows_inputs_<line>.parquet (item 6)

Each scored row as the model saw it, after the data fixes and mappings:

- identifiers, period dates, `exposure` (the one to use) and `exposure_plain` (identical);
- every feature any component uses (for buildings, also the XGBoost's features);
- the sources of the continuous terms;
- the raw values of fixed features (`*_raw`; `exposure_raw` is the raw exposure column, which D-01 replaces and
  the model does not use);
- for each GLM term, the level or band it was given (`<component>__<term>__level`).

Contents rows take their period dates from the test file, because the contents model table keeps none. For the
same reason the contents file has no `*_raw` columns: its `sum_insured` is after D-02, and the raw value is in the
test file.

## rows_glm_terms_<line>.parquet (extra)

Each row's log contribution from every GLM term (`<component>__<term>`), the offset, the trend and, for a large
claim, ln of the Pareto adjustment. They add up to `<component>__total_log`, the log of the component's
prediction. Use them to find where a spreadsheet row departs from the model.

## reference_office_premium.parquet (item 7)

`python/office_premium.py` on the final file, for the 20 reference policies (17 broker, 3 direct). Columns:
`row_id, line, channel, new_business, exposure` (the reference file's own), `written_premium`, then the risk premium
for the period (the final file's value, exactly) and annualised, cat reinsurance, fixed and variable expense,
commission, premium tax, profit margin and office premium, which the components add up to. It annualises with the reference file's own exposure
column, which differs from the recomputed exposure by at most 0.000044 years. The emerging table has no
comparison-site commission (the developed rate stands in), but no reference policy sells through that channel.

## reconciliation.csv (item 8)

`check, line, component, term, value, passed, note, type`. Rows of type `information` are not checks and leave
`passed` blank. All 40 checks pass:

- the refit reproduces the as-is file on every row, bit for bit (0 rows differ on each line);
- the two level steps reproduce the midpoint and final files bit for bit;
- the GLMs re-evaluated from the exported tables alone, as a spreadsheet would, match the model to within 6.0e-15,
  through the lookups and through the pieces;
- the buildings trees reproduce `small_claim` bit for bit, both from `buildings_xgboost.json` reloaded and from
  the two tree tables scored in single precision; scored in double precision they are within 7.3e-6, against a
  limit of 1e-4 (float32 rounding over 193 additions can approach it, while a wrong branch moves a row far more);
- the final prediction rebuilt from the exported tables alone, the buildings trees included (scored in single
  precision), matches the final file to within 6.2e-15;
- the components recombine to the final file to within 1.8e-15 (relative); 34,967 rows are not bit-exact, for the
  reason below.

## Precision

- The CSVs carry every float at round-trip precision. Read them in pandas with `float_precision="round_trip"`.
  pandas' default parser returns some values a few units in the last place away: 10.5% of the as-is file (up to
  4 units), 10.4% of the midpoint file (up to 5) and 3.4% of the final file (1 unit).
- A31's scripts read the as-is and midpoint files with the default parser, so the midpoint and final files carry
  those shifts. `as_is_file_as_read` and `midpoint_file_as_read` hold the values as those scripts saw them, and they
  are why the components match the final file to 1.8e-15 rather than exactly.
- Excel keeps 15 significant digits. A spreadsheet that follows this README should match the final file to about
  1e-14 relative (up to 3e-14 in a simulation of 15-digit inputs; not tested in Excel itself), against about 6e-15
  at full double precision. That holds for contents, motor and the buildings large claims, and for buildings small
  claims taken from `small_claim`. Buildings rows whose small claims are scored from the trees in double
  precision, as Excel must, match to about 7e-6 instead (see The buildings trees).
- Open bounds are written `inf` and `-inf`, and `scalars.csv` gives `inf` for a credibility k that gives history no
  weight (not applied in C0 anyway). Excel has no infinity: load the CSVs through Data > From Text/CSV and replace
  them with -1E+307 and 1E+307 before use. Opened directly, `-inf` may read as a formula and show `#NAME?`, and as
  text it compares above every number, so the first band and the first spline piece would never match.
- Category levels are text: some look like numbers (`0` to `5`, `250`, `500`, `1000`) or booleans (`true`, `false`
  in `terms.csv` and `input_mappings.csv`; `True`, `False` in `buildings_xgboost_features.csv`).
  Import the `level`, `source_value` and `mapped_value` columns, and the input columns you look up, as text, and
  count the rows that fall back to `(any other level)`: with the 2025 data the count should be 0.

## Missing values

- A missing value takes the term's `MISSING` level where the fit saw one. Among the categoricals, those are
  buildings `soil_type` and contents `f_flood_zone`, `f_hvi` and `f_security`. Elsewhere it scores as the base level;
  each band's `MISSING` row says which applies.
- Of the fields about 12% missing in the emerging market, these reach a GLM:
  - `soil_type`: buildings large claims;
  - `security_level` and `high_value_items`: contents, through their `MISSING` levels;
  - `distance_to_river_km`: buildings large claims and contents;
  - `vehicle_age`: motor small claims.

  The last two are splines and take their `spline missing` value. `smoke_alarms`, `marital_status`, `roof_type`,
  `energy_rating`, `days_unoccupied`, `household_size` and `years_at_address` are features of the buildings XGBoost
  only, and no tree splits on `smoke_alarms` or `roof_type`. A missing energy rating also sets the year-built
  placeholder (D-27), which feeds the buildings large-claim GLM.
  Occupation, `overnight_parking` and `engine_size_cc` reach no model.
- A missing flood zone counts as zone 3 or above in `x_fz3_rain`, and `x_clay_dry` is 0 when soil is missing.

## Regenerating

From the pack root: `python python/export_excel.py`. It refits C0 on the developed market (about two minutes),
stops if the refit is more than rounding away from the as-is file or the level steps do not reproduce the files,
and appends a record to `audit/run_log.jsonl`.
