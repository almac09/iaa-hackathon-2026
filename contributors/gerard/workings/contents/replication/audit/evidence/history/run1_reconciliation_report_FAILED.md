# Reconciliation report

Generated 2026-10-20T15:01:06+00:00. 84 checks, 1 failed.

| Check | Item | Result | Detail |
|---|---|---|---|
| sql evidence | same set of files | PASS | 32 python, 32 r |
| sql evidence | ev_01_claims_by_line_year.csv | PASS | byte-identical |
| sql evidence | ev_01_claims_status_checks.csv | PASS | byte-identical |
| sql evidence | ev_01_key_uniqueness.csv | PASS | byte-identical |
| sql evidence | ev_01_row_counts.csv | PASS | byte-identical |
| sql evidence | ev_02_claim_matching.csv | PASS | byte-identical |
| sql evidence | ev_02_contents_missing_values.csv | PASS | byte-identical |
| sql evidence | ev_02_contents_sum_insured_bands.csv | PASS | byte-identical |
| sql evidence | ev_02_exposure_units.csv | PASS | byte-identical |
| sql evidence | ev_02_motor_customer_id_padding.csv | PASS | byte-identical |
| sql evidence | ev_02_sum_insured_same_policy_ratio.csv | PASS | byte-identical |
| sql evidence | ev_03_claim_reconciliation.csv | PASS | byte-identical |
| sql evidence | ev_03_exposure_before_after.csv | PASS | byte-identical |
| sql evidence | ev_03_join_completeness.csv | PASS | byte-identical |
| sql evidence | ev_03_sum_insured_before_after.csv | PASS | byte-identical |
| sql evidence | ev_04_contents_experience_by_year.csv | PASS | byte-identical |
| sql evidence | ev_04_contents_ibnr_2024.csv | PASS | byte-identical |
| sql evidence | ev_04_contents_open_vs_closed.csv | PASS | byte-identical |
| sql evidence | ev_04_report_lag_by_line.csv | PASS | byte-identical |
| sql evidence | ev_05_claims_above_sum_insured.csv | PASS | byte-identical |
| sql evidence | ev_05_cost_concentration.csv | PASS | byte-identical |
| sql evidence | ev_05_severity_by_peril.csv | PASS | byte-identical |
| sql evidence | ev_05_size_band_mix.csv | PASS | byte-identical |
| sql evidence | ev_06_next_year_by_state.csv | PASS | byte-identical |
| sql evidence | ev_06_transition_matrix.csv | PASS | byte-identical |
| sql evidence | ev_06_two_year_history.csv | PASS | byte-identical |
| sql evidence | ev_07_contacts_by_claim_status.csv | PASS | byte-identical |
| sql evidence | ev_07_contacts_oneway.csv | PASS | byte-identical |
| sql evidence | ev_07_contacts_train_vs_test.csv | PASS | byte-identical |
| sql evidence | ev_08_feature_bounds.csv | PASS | byte-identical |
| sql evidence | evr_01_largest_claims.csv | PASS | byte-identical |
| sql evidence | evr_03_exposure_changed_rows.csv | PASS | byte-identical |
| sql evidence | evr_03_sum_insured_changed_rows.csv | PASS | byte-identical |
| split test D2 | exposure only | PASS | py 0.03099 r 0.03099 diff 0.00000 |
| split test pred/actual | exposure only | PASS | diff 0.00000 |
| split test D2 | A single Tweedie | PASS | py 0.09325 r 0.09325 diff 0.00000 |
| split test pred/actual | A single Tweedie | PASS | diff 0.00000 |
| split test D2 | B per-peril Tweedie | PASS | py 0.09196 r 0.09196 diff 0.00000 |
| split test pred/actual | B per-peril Tweedie | PASS | diff 0.00000 |
| split test D2 | C per-size-band Tweedie | PASS | py 0.08721 r 0.08721 diff 0.00000 |
| split test pred/actual | C per-size-band Tweedie | PASS | diff 0.00009 |
| split test D2 | D per-peril freq x sev | PASS | py 0.09041 r 0.09041 diff 0.00000 |
| split test pred/actual | D per-peril freq x sev | PASS | diff 0.00001 |
| coefficients | same parameters in every component | PASS | 0 unmatched |
| coefficients | A total | PASS | max abs diff 0.00325 (f_segment=mature_mainstream) |
| coefficients | B accidental_damage | PASS | max abs diff 0.00187 (f_segment=young_mainstream) |
| coefficients | B escape_of_water | PASS | max abs diff 0.00171 (f_segment=senior_budget) |
| coefficients | B flood | FAIL | max abs diff 0.04767 (f_elev_band=q07) |
| coefficients | B theft | PASS | max abs diff 0.00215 (f_segment=mature_mainstream) |
| coefficients | C cost_band_1 | PASS | max abs diff 0.00151 (f_credit=D) |
| coefficients | C cost_band_2 | PASS | max abs diff 0.00216 (f_segment=mature_budget) |
| coefficients | C cost_band_3 | PASS | max abs diff 0.00233 (f_segment=mature_mainstream) |
| coefficients | C cost_band_4 | PASS | max abs diff 0.00183 (f_credit=E) |
| coefficients | D freq accidental_damage | PASS | max abs diff 0.00002 (f_segment=senior_mainstream) |
| coefficients | D freq escape_of_water | PASS | max abs diff 0.00010 (f_credit=E) |
| coefficients | D freq flood | PASS | max abs diff 0.00008 (f_flood_zone=5) |
| coefficients | D freq theft | PASS | max abs diff 0.00009 (f_segment=young_mainstream) |
| coefficients | D sev accidental_damage | PASS | max abs diff 0.00086 (f_segment=young_budget) |
| coefficients | D sev escape_of_water | PASS | max abs diff 0.00082 (f_segment=senior_mainstream) |
| coefficients | D sev flood | PASS | max abs diff 0.00104 (f_elev_band=q05) |
| coefficients | D sev theft | PASS | max abs diff 0.00089 (f_segment=young_mainstream) |
| one-way scores | 56 factors, d2_gain | PASS | max abs diff 0.00e+00 |
| one-way scores | levels per factor | PASS |  |
| one-way tables | same factor levels | PASS | 394 rows |
| one-way tables | exposure | PASS | max abs diff 0.00e+00 |
| one-way tables | n_claims | PASS | max abs diff 0.00e+00 |
| one-way tables | loss_cost | PASS | max abs diff 0.00e+00 |
| one-way tables | relativity | PASS | max abs diff 0.00e+00 |
| one-way tables | rel_2021_22 | PASS | max abs diff 0.00e+00 |
| one-way tables | rel_2023_24 | PASS | max abs diff 0.00e+00 |
| design | rows, parameters and reference levels | PASS |  |
| vs exploratory | split D2: exposure only | PASS | exploratory 0.0310 pack 0.0310 |
| vs exploratory | split D2: A single Tweedie | PASS | exploratory 0.0933 pack 0.0932 |
| vs exploratory | split D2: B per-peril Tweedie | PASS | exploratory 0.0920 pack 0.0920 |
| vs exploratory | split D2: C per-size-band Tweedie | PASS | exploratory 0.0872 pack 0.0872 |
| vs exploratory | split D2: D per-peril freq x sev | PASS | exploratory 0.0899 pack 0.0904 |
| vs exploratory | one-way gain: f_cs_contacts | PASS | exploratory 0.5774 pack 0.5774 |
| vs exploratory | one-way gain: f_flood_zone | PASS | exploratory 0.0239 pack 0.0239 |
| vs exploratory | one-way gain: f_wp_band | PASS | exploratory 0.0194 pack 0.0194 |
| vs exploratory | one-way gain: f_elev_band | PASS | exploratory 0.0175 pack 0.0175 |
| vs exploratory | one-way gain: f_si_band | PASS | exploratory 0.0164 pack 0.0164 |
| vs exploratory | one-way gain: f_river_band | PASS | exploratory 0.0132 pack 0.0132 |
| vs exploratory | one-way gain: f_tenure | PASS | exploratory 0.0050 pack 0.0050 |
| vs exploratory | one-way gain: f_region | PASS | exploratory 0.0037 pack 0.0037 |
