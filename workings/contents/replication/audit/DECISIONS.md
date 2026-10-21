# Decisions awaiting sign-off — home contents

Same format as the team's `notebooks/00_decision_log`: what the AI suggested, what the human decided, and why. Claude fills in the first column only. The **Human decision** and **Signed off** fields are for Gerard, or for whoever the team nominates.

| ID | AI suggestion | Evidence | Human decision | Rationale (human) | Signed off (name, date) |
|---|---|---|---|---|---|
| D-01 | Recompute exposure from period dates for every row, keeping the raw value. | A04: 3,088 contents rows in days; raw exposure overstates training exposure by 37%. | | | |
| D-02 | Divide sum insured by 1,000 where it is 1 million or more; leave 200k–1m unchanged. | A05: about a 1,000× ratio against the same policy's other years; premium consistent with ordinary risks. | | | |
| D-03 | Target = incurred − recoveries, excluding repudiated claims and claims outside any period. Uplift 2024 by about 3% for unreported claims, by peril. No uplift for case reserves. | A06 | | | |
| D-04 | Exclude `customer_service_contacts` from all models. | A08: claim-driven in training but not in test. | | | |
| D-05 | Keep `written_premium` out of the technical model, but use it as a benchmark. It encodes the incumbent's pricing, so including it would partly copy the current rates rather than build an independent technical price. | A07: second strongest factor (+0.019). | | | |
| D-06 | Treat missing values in `security_level`, `high_value_items`, `flood_zone` and `distance_to_river_km` as their own level rather than imputing them. | A07: missing flood zone has a relativity of 1.71, so the missingness carries information. | | | |
| D-07 | Expected claims: model claims per unit of **exposure** (log-exposure offset), giving an independent technical price. Use **earned premium** (= annual written premium × exposure) for loss ratios, A/E monitoring and the benchmark against current rates, not as the model offset. | A12: premium is annual; earned-premium A/E shows claim history is not priced. | Accepted: chosen by Gerard Murphy in session, 2026-10-20, from three options (exposure offset; earned premium as offset; test both). | | |
| D-08 | Keep **GLM A** as the contents candidate and use XGBoost as a challenger only. If XGBoost is used, use the banded features: the raw set (55 columns, excluding IDs, dates, `yr`, contacts per D-04, written and earned premium per D-05, `renewal_decision` and claim columns) scored lower and calibrated worse. Numeric missing values are left to XGBoost's learned default direction; categorical missing values are a level, as in D-06. | A13: XGBoost banded −0.0014 D² against GLM A (95% interval −0.0034 to +0.0006), raw −0.0019. Depth 1 was as good as deeper trees. | | | |
