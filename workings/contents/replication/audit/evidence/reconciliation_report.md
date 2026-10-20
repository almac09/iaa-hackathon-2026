# Reconciliation report

Generated 2026-10-20T16:23:57+00:00. 100 checks, 0 failed, 1 warnings (see audit A11).

| Check | Item | Result | Detail |
|---|---|---|---|
| sql evidence | same set of files | PASS | 34 python, 34 r |
| sql evidence | ev_01_claims_by_line_year.csv | PASS | byte-identical |
| sql evidence | ev_01_claims_status_checks.csv | PASS | byte-identical |
| sql evidence | ev_01_key_uniqueness.csv | PASS | byte-identical |
| sql evidence | ev_01_row_counts.csv | PASS | byte-identical |
| sql evidence | ev_02_claim_matching.csv | PASS | byte-identical |
| sql evidence | ev_02_contents_missing_values.csv | PASS | byte-identical |
| sql evidence | ev_02_contents_sum_insured_bands.csv | PASS | byte-identical |
| sql evidence | ev_02_exposure_units.csv | PASS | byte-identical |
| sql evidence | ev_02_motor_customer_id_padding.csv | PASS | byte-identical |
| sql evidence | ev_02_premium_basis.csv | PASS | byte-identical |
| sql evidence | ev_02_sum_insured_same_policy_ratio.csv | PASS | byte-identical |
| sql evidence | ev_03_claim_reconciliation.csv | PASS | byte-identical |
| sql evidence | ev_03_exposure_before_after.csv | PASS | byte-identical |
| sql evidence | ev_03_join_completeness.csv | PASS | byte-identical |
| sql evidence | ev_03_sum_insured_before_after.csv | PASS | byte-identical |
| sql evidence | ev_04_contents_experience_by_year.csv | PASS | byte-identical |
| sql evidence | ev_04_contents_ibnr_2024.csv | PASS | byte-identical |
| sql evidence | ev_04_contents_open_vs_closed.csv | PASS | byte-identical |
| sql evidence | ev_04_loss_ratio_by_line_year.csv | PASS | byte-identical |
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
| split test pred/actual | C per-size-band Tweedie | PASS | diff 0.00000 |
| split test D2 | D per-peril freq x sev | PASS | py 0.09041 r 0.09041 diff 0.00000 |
| split test pred/actual | D per-peril freq x sev | PASS | diff 0.00000 |
| coefficients | same parameters in every component | PASS | 0 unmatched |
| coefficients | B flood | f_elev_band=q07 | WARN | quasi-separated: py -12.9715 r -12.9715 (relativity ~2.3e-06) |
| coefficients | A total | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | B accidental_damage | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | B escape_of_water | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | B flood | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | B theft | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | C cost_band_1 | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | C cost_band_2 | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | C cost_band_3 | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | C cost_band_4 | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D freq accidental_damage | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D freq escape_of_water | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D freq flood | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D freq theft | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D sev accidental_damage | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D sev escape_of_water | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D sev flood | PASS | max abs diff 0.00000 ((intercept)) |
| coefficients | D sev theft | PASS | max abs diff 0.00000 ((intercept)) |
| component predictions | A total | PASS | py 17,077,309 r 17,077,309 rel diff 5.87e-14 |
| component predictions | B accidental_damage | PASS | py 1,879,483 r 1,879,483 rel diff 5.32e-13 |
| component predictions | B escape_of_water | PASS | py 3,968,983 r 3,968,983 rel diff 2.52e-13 |
| component predictions | B flood | PASS | py 4,198,501 r 4,198,501 rel diff 1.91e-12 |
| component predictions | B theft | PASS | py 7,168,483 r 7,168,483 rel diff 1.39e-13 |
| component predictions | C cost_band_1 | PASS | py 1,169,393 r 1,169,393 rel diff 0.00e+00 |
| component predictions | C cost_band_2 | PASS | py 5,823,999 r 5,823,999 rel diff 1.72e-13 |
| component predictions | C cost_band_3 | PASS | py 6,447,024 r 6,447,024 rel diff 1.09e-12 |
| component predictions | C cost_band_4 | PASS | py 4,715,665 r 4,715,665 rel diff 2.12e-13 |
| component predictions | D accidental_damage | PASS | py 1,880,834 r 1,880,834 rel diff 2.68e-09 |
| component predictions | D escape_of_water | PASS | py 3,894,104 r 3,894,104 rel diff 4.62e-10 |
| component predictions | D flood | PASS | py 4,309,152 r 4,309,152 rel diff 1.19e-09 |
| component predictions | D theft | PASS | py 7,126,037 r 7,126,037 rel diff 1.32e-10 |
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
