-- 08 Rating-factor features for the one-way analysis and the GLMs (audit A07, A09; decision D-06).
-- All banding is done here so that Python and R model exactly the same inputs.
-- Missing values become the level 'MISSING' (D-06). Quantile bands use the training rows only,
-- linear interpolation (as pandas.qcut and R quantile type 7) and right-closed intervals; the
-- label 'qNN' counts the inner boundaries strictly below the value.

CREATE OR REPLACE TABLE feature_bounds AS
WITH tr AS (SELECT * FROM contents_model WHERE src = 'train')
SELECT 'sum_insured' AS factor, unnest(quantile_cont(sum_insured, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])) AS bound FROM tr
UNION ALL SELECT 'written_premium', unnest(quantile_cont(written_premium, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])) FROM tr
UNION ALL SELECT 'distance_to_river_km', unnest(quantile_cont(distance_to_river_km, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'elevation_m', unnest(quantile_cont(elevation_m, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'hail_days', unnest(quantile_cont(hail_days, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'wind_exposure_index', unnest(quantile_cont(wind_exposure_index, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'freezing_days', unnest(quantile_cont(freezing_days, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'dry_summer_index', unnest(quantile_cont(dry_summer_index, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'wildfire_index', unnest(quantile_cont(wildfire_index, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'rainfall_mm', unnest(quantile_cont(rainfall_mm, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr
UNION ALL SELECT 'mean_temperature_c', unnest(quantile_cont(mean_temperature_c, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])) FROM tr;

CREATE OR REPLACE MACRO qband(factor_name, x) AS
    CASE WHEN x IS NULL THEN 'MISSING'
         ELSE 'q' || lpad(CAST((SELECT count(*) FROM feature_bounds b WHERE b.factor = factor_name AND b.bound < x) AS VARCHAR), 2, '0')
    END;

CREATE OR REPLACE MACRO lvl(x) AS coalesce(CAST(x AS VARCHAR), 'MISSING');

CREATE OR REPLACE TABLE contents_features AS
WITH si_stats AS (
    SELECT avg(ln(sum_insured)) AS mu, stddev_samp(ln(sum_insured)) AS sd FROM contents_model WHERE src = 'train'
)
SELECT row_id, src, yr, exposure,
       claim_cost, cost_theft, cost_eow, cost_ad, cost_flood, n_theft, n_eow, n_ad, n_flood,
       cost_band_1, cost_band_2, cost_band_3, cost_band_4, n_claims,
       -- policy
       CASE WHEN policyholder_age <= 25 THEN 'a1 <=25' WHEN policyholder_age <= 30 THEN 'a2 26-30'
            WHEN policyholder_age <= 35 THEN 'a3 31-35' WHEN policyholder_age <= 40 THEN 'a4 36-40'
            WHEN policyholder_age <= 50 THEN 'a5 41-50' WHEN policyholder_age <= 60 THEN 'a6 51-60'
            WHEN policyholder_age <= 70 THEN 'a7 61-70' WHEN policyholder_age <= 80 THEN 'a8 71-80'
            WHEN policyholder_age IS NULL THEN 'MISSING' ELSE 'a9 81+' END AS f_age_band,
       qband('sum_insured', sum_insured) AS f_si_band,
       lvl(tenure) AS f_tenure,
       lvl(security_level) AS f_security,
       lvl(CAST(high_value_items AS INTEGER)) AS f_hvi,
       lvl(accidental_damage_cover) AS f_ad_cover,
       lvl(works_from_home) AS f_wfh,
       lvl(lodgers) AS f_lodgers,
       lvl(bicycle_cover) AS f_bike,
       lvl(voluntary_excess) AS f_vol_xs,
       lvl(channel) AS f_channel,
       lvl(payment_method) AS f_payment,
       lvl(least(missed_payments_12m, 3)) AS f_missed_pay,
       lvl(new_business) AS f_new_business,
       CASE WHEN renewal_count <= 0 THEN 'r1 0' WHEN renewal_count <= 1 THEN 'r2 1' WHEN renewal_count <= 3 THEN 'r3 2-3'
            WHEN renewal_count <= 6 THEN 'r4 4-6' WHEN renewal_count <= 10 THEN 'r5 7-10'
            WHEN renewal_count IS NULL THEN 'MISSING' ELSE 'r6 11+' END AS f_renewals,
       CASE WHEN days_quote_to_inception <= 0 THEN 'd1 0' WHEN days_quote_to_inception <= 3 THEN 'd2 1-3'
            WHEN days_quote_to_inception <= 7 THEN 'd3 4-7' WHEN days_quote_to_inception <= 14 THEN 'd4 8-14'
            WHEN days_quote_to_inception <= 21 THEN 'd5 15-21'
            WHEN days_quote_to_inception IS NULL THEN 'MISSING' ELSE 'd6 22+' END AS f_quote_days,
       lvl(discount_applied) AS f_discount,
       lvl(least(customer_service_contacts, 4)) AS f_cs_contacts,
       qband('written_premium', written_premium) AS f_wp_band,
       -- customer
       lvl(occupation) AS f_occupation,
       lvl(marital_status) AS f_marital,
       lvl(least(household_size, 6)) AS f_household,
       lvl(credit_band) AS f_credit,
       lvl(least(prior_claims_5y, 3)) AS f_prior_claims,
       lvl(least(licence_points, 6)) AS f_licence_pts,
       CASE WHEN years_at_address <= 1 THEN 'y1 <=1' WHEN years_at_address <= 3 THEN 'y2 2-3' WHEN years_at_address <= 5 THEN 'y3 4-5'
            WHEN years_at_address <= 10 THEN 'y4 6-10' WHEN years_at_address <= 20 THEN 'y5 11-20'
            WHEN years_at_address IS NULL THEN 'MISSING' ELSE 'y6 21+' END AS f_yrs_at_addr,
       lvl(customer_since_year) AS f_cust_since,
       lvl(email_domain) AS f_email,
       lvl(marketing_consent) AS f_mkt_consent,
       lvl(document_preference) AS f_doc_pref,
       lvl(previous_insurer) AS f_prev_insurer,
       lvl(campaign_code) AS f_campaign,
       lvl(branch_id) AS f_branch,
       lvl(marketing_segment) AS f_segment,
       -- address
       lvl(region) AS f_region,
       lvl(urban_density) AS f_urban,
       lvl(CAST(flood_zone AS INTEGER)) AS f_flood_zone,
       qband('distance_to_river_km', distance_to_river_km) AS f_river_band,
       qband('elevation_m', elevation_m) AS f_elev_band,
       lvl(soil_type) AS f_soil,
       CASE WHEN year_built <= 1900 THEN 'b1 <=1900' WHEN year_built <= 1945 THEN 'b2 1901-1945'
            WHEN year_built <= 1970 THEN 'b3 1946-1970' WHEN year_built <= 1990 THEN 'b4 1971-1990'
            WHEN year_built <= 2005 THEN 'b5 1991-2005'
            WHEN year_built IS NULL THEN 'MISSING' ELSE 'b6 2006+' END AS f_built,
       lvl(property_type) AS f_prop_type,
       lvl(roof_type) AS f_roof,
       lvl(basement) AS f_basement,
       lvl(trees_within_5m) AS f_trees,
       lvl(listed_building) AS f_listed,
       lvl(least(bedrooms, 6)) AS f_bedrooms,
       lvl(energy_rating) AS f_energy,
       lvl(solar_panels) AS f_solar,
       -- weather (year of the row)
       qband('hail_days', hail_days) AS f_hail_band,
       qband('wind_exposure_index', wind_exposure_index) AS f_wind_band,
       qband('freezing_days', freezing_days) AS f_freeze_band,
       qband('dry_summer_index', dry_summer_index) AS f_dry_band,
       qband('wildfire_index', wildfire_index) AS f_wildfire_band,
       qband('rainfall_mm', rainfall_mm) AS f_rain_band,
       qband('mean_temperature_c', mean_temperature_c) AS f_temp_band,
       -- numeric
       (ln(sum_insured) - s.mu) / s.sd AS z_log_si
FROM contents_model, si_stats s;

CREATE OR REPLACE TABLE ev_08_feature_bounds AS
SELECT factor, string_agg(CAST(round(bound, 6) AS VARCHAR), ' | ' ORDER BY bound) AS inner_bounds
FROM feature_bounds GROUP BY factor;
