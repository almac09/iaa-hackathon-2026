# Reconciliation report

Generated 2026-10-23T09:47:21+00:00. 101 checks, 0 failed, 2 warnings (see audit A11).

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
| split test D2 | exposure only | PASS | py 0.03089 r 0.03089 diff 0.00000 |
| split test pred/actual | exposure only | PASS | diff 0.00000 |
| split test D2 | A single Tweedie | PASS | py 0.09336 r 0.09336 diff 0.00000 |
| split test pred/actual | A single Tweedie | PASS | diff 0.00000 |
| split test D2 | B per-peril Tweedie | PASS | py 0.09295 r 0.09295 diff 0.00000 |
| split test pred/actual | B per-peril Tweedie | PASS | diff 0.00000 |
| split test D2 | C per-size-band Tweedie | PASS | py 0.08704 r 0.08704 diff 0.00000 |
| split test pred/actual | C per-size-band Tweedie | PASS | diff 0.00000 |
| split test D2 | D per-peril freq x sev | PASS | py 0.09066 r 0.09066 diff 0.00000 |
| split test pred/actual | D per-peril freq x sev | PASS | diff 0.00000 |
| coefficients | same parameters in every component | PASS | 0 unmatched |
| coefficients | B flood | f_elev_band=q06 | WARN | quasi-separated: py -12.2301 r -12.2301 (relativity ~4.9e-06) |
| coefficients | B flood | f_elev_band=q07 | WARN | quasi-separated: py -12.2344 r -12.2344 (relativity ~4.9e-06) |
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
| component predictions | A total | PASS | py 17,008,206 r 17,008,206 rel diff 1.77e-13 |
| component predictions | B accidental_damage | PASS | py 1,871,200 r 1,871,200 rel diff 1.07e-12 |
| component predictions | B escape_of_water | PASS | py 3,947,962 r 3,947,962 rel diff 2.53e-13 |
| component predictions | B flood | PASS | py 4,033,094 r 4,033,094 rel diff 4.96e-13 |
| component predictions | B theft | PASS | py 7,145,723 r 7,145,723 rel diff 0.00e+00 |
| component predictions | C cost_band_1 | PASS | py 1,163,517 r 1,163,517 rel diff 8.60e-13 |
| component predictions | C cost_band_2 | PASS | py 5,799,688 r 5,799,688 rel diff 3.45e-13 |
| component predictions | C cost_band_3 | PASS | py 6,424,604 r 6,424,604 rel diff 1.09e-12 |
| component predictions | C cost_band_4 | PASS | py 4,717,997 r 4,717,997 rel diff 6.36e-13 |
| component predictions | D accidental_damage | PASS | py 1,873,844 r 1,873,844 rel diff 2.68e-09 |
| component predictions | D escape_of_water | PASS | py 3,873,806 r 3,873,806 rel diff 4.47e-10 |
| component predictions | D flood | PASS | py 4,009,758 r 4,009,758 rel diff 1.23e-09 |
| component predictions | D theft | PASS | py 7,104,954 r 7,104,954 rel diff 1.34e-10 |
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
| vs exploratory | split D2: exposure only | PASS | exploratory 0.0310 pack 0.0309 |
| vs exploratory | split D2: A single Tweedie | PASS | exploratory 0.0933 pack 0.0934 |
| vs exploratory | split D2: B per-peril Tweedie | PASS | exploratory 0.0920 pack 0.0929 |
| vs exploratory | split D2: C per-size-band Tweedie | PASS | exploratory 0.0872 pack 0.0870 |
| vs exploratory | split D2: D per-peril freq x sev | PASS | exploratory 0.0899 pack 0.0907 |
| vs exploratory | one-way gain: f_cs_contacts | PASS | exploratory 0.5774 pack 0.5781 |
| vs exploratory | one-way gain: f_flood_zone | PASS | exploratory 0.0239 pack 0.0239 |
| vs exploratory | one-way gain: f_wp_band | PASS | exploratory 0.0194 pack 0.0195 |
| vs exploratory | one-way gain: f_elev_band | PASS | exploratory 0.0175 pack 0.0186 |
| vs exploratory | one-way gain: f_si_band | PASS | exploratory 0.0164 pack 0.0164 |
| vs exploratory | one-way gain: f_river_band | PASS | exploratory 0.0132 pack 0.0133 |
| vs exploratory | one-way gain: f_tenure | PASS | exploratory 0.0050 pack 0.0050 |
| vs exploratory | one-way gain: f_region | PASS | exploratory 0.0037 pack 0.0037 |
