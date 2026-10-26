-- 02 Data-quality checks across all lines (audit A03-A05). Read-only: nothing is fixed here.
-- Databricks SQL note: datediff('day', a, b) is datediff(b, a) in Spark SQL.

CREATE OR REPLACE VIEW all_policy_rows AS
SELECT 'motor' AS line, 'train' AS src, row_id, policy_id, customer_id, period_start, period_end, exposure, written_premium FROM policies_motor
UNION ALL SELECT 'buildings', 'train', row_id, policy_id, customer_id, period_start, period_end, exposure, written_premium FROM policies_buildings
UNION ALL SELECT 'contents', 'train', row_id, policy_id, customer_id, period_start, period_end, exposure, written_premium FROM policies_contents
UNION ALL SELECT 'motor', 'test', row_id, policy_id, customer_id, period_start, period_end, exposure, written_premium FROM test_motor
UNION ALL SELECT 'buildings', 'test', row_id, policy_id, customer_id, period_start, period_end, exposure, written_premium FROM test_buildings
UNION ALL SELECT 'contents', 'test', row_id, policy_id, customer_id, period_start, period_end, exposure, written_premium FROM test_contents;

-- A12: written_premium is the annual premium, not the premium for the row's period. If it were
-- pro-rata it would scale with time on cover; it does not, so earned premium = written x exposure.
CREATE OR REPLACE TABLE ev_02_premium_basis AS
WITH r AS (
    SELECT line, src, written_premium,
           (datediff('day', period_start, period_end) + 1) / getvariable('exposure_basis_days') AS e
    FROM all_policy_rows
)
SELECT line, src,
       CASE WHEN e >= 0.99 THEN '1 full year' WHEN e >= 0.5 THEN '2 half to full year'
            WHEN e >= 0.25 THEN '3 quarter to half' ELSE '4 under a quarter' END AS time_on_cover,
       count(*) AS n_rows,
       round(avg(written_premium), 2) AS mean_written_premium,
       round(avg(e), 4) AS mean_exposure,
       round(corr(written_premium, e), 4) AS corr_premium_exposure
FROM r
GROUP BY ALL;

-- A04: exposure should be in years. Classify values that are not consistent with the period dates.
CREATE OR REPLACE TABLE ev_02_exposure_units AS
WITH r AS (
    SELECT line, src, year(period_start) AS yr, exposure,
           datediff('day', period_start, period_end) + 1 AS days
    FROM all_policy_rows
)
SELECT line, src, yr,
       count(*) AS n_rows,
       count(*) FILTER (WHERE exposure > 1.0001) AS n_above_1,
       count(*) FILTER (WHERE exposure > 1.0001 AND abs(exposure - days) < 1.01) AS n_recorded_in_days,
       count(*) FILTER (WHERE abs(exposure - days / 360.0) <= 0.0005
                          AND abs(exposure - days / getvariable('exposure_basis_days')) > 0.0005) AS n_360_day_basis,
       count(*) FILTER (WHERE abs(exposure - days / getvariable('exposure_basis_days')) <= 0.0005) AS n_consistent_with_dates
FROM r
GROUP BY ALL;

-- A05: contents sum insured by size band, with premium as a sense check.
CREATE OR REPLACE TABLE ev_02_contents_sum_insured_bands AS
WITH r AS (
    SELECT 'train' AS src, sum_insured, written_premium FROM policies_contents
    UNION ALL SELECT 'test', sum_insured, written_premium FROM test_contents
)
SELECT src,
       CASE WHEN sum_insured < 200000 THEN '1 <200k'
            WHEN sum_insured < 1000000 THEN '2 200k-1m'
            WHEN sum_insured < 10000000 THEN '3 1m-10m'
            ELSE '4 10m+' END AS si_band,
       count(*) AS n_rows,
       round(avg(written_premium), 2) AS avg_written_premium,
       min(sum_insured) AS min_sum_insured,
       max(sum_insured) AS max_sum_insured
FROM r
GROUP BY ALL;

-- A05 verification: the same policy in a different year, below the threshold.
CREATE OR REPLACE TABLE ev_02_sum_insured_same_policy_ratio AS
WITH p AS (
    SELECT policy_id, year(period_start) AS yr, max(sum_insured) AS si
    FROM policies_contents GROUP BY ALL
),
pairs AS (
    SELECT b.si / o.si AS ratio
    FROM p b JOIN p o ON o.policy_id = b.policy_id AND o.yr <> b.yr
    WHERE b.si >= getvariable('si_threshold') AND o.si < getvariable('si_threshold')
)
SELECT count(*) AS n_pairs,
       round(min(ratio), 1) AS min_ratio,
       round(quantile_cont(ratio, 0.5), 1) AS median_ratio,
       round(max(ratio), 1) AS max_ratio
FROM pairs;

-- Motor only (not used in the contents pipeline): customer IDs that lost leading zeros.
CREATE OR REPLACE TABLE ev_02_motor_customer_id_padding AS
SELECT p.src,
       count(*) AS n_rows,
       count(*) FILTER (WHERE c1.customer_id IS NULL) AS n_unmatched_raw,
       count(*) FILTER (WHERE c2.customer_id IS NULL) AS n_unmatched_after_lpad8
FROM all_policy_rows p
LEFT JOIN customers c1 ON c1.customer_id = p.customer_id
LEFT JOIN customers c2 ON c2.customer_id = lpad(p.customer_id, 8, '0')
WHERE p.line = 'motor'
GROUP BY ALL;

-- Claims to policy rows: match on policy_id with loss_date inside the row's period.
CREATE OR REPLACE TABLE ev_02_claim_matching AS
WITH m AS (
    SELECT c.claim_id, c.line, c.loss_date, count(p.row_id) AS n_match
    FROM claims c
    LEFT JOIN all_policy_rows p
      ON p.src = 'train' AND p.policy_id = c.policy_id AND c.loss_date BETWEEN p.period_start AND p.period_end
    GROUP BY ALL
)
SELECT line,
       count(*) AS n_claims,
       count(*) FILTER (WHERE n_match = 1) AS n_matched_one,
       count(*) FILTER (WHERE n_match > 1) AS n_matched_many,
       count(*) FILTER (WHERE n_match = 0) AS n_unmatched,
       count(*) FILTER (WHERE n_match = 0 AND year(loss_date) < 2021) AS n_unmatched_before_2021
FROM m
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_02_contents_missing_values AS
SELECT 'train' AS src, 'security_level' AS column_name, count(*) FILTER (WHERE security_level IS NULL) AS n_missing, count(*) AS n_rows FROM policies_contents
UNION ALL SELECT 'train', 'high_value_items', count(*) FILTER (WHERE high_value_items IS NULL), count(*) FROM policies_contents
UNION ALL SELECT 'test', 'security_level', count(*) FILTER (WHERE security_level IS NULL), count(*) FROM test_contents
UNION ALL SELECT 'test', 'high_value_items', count(*) FILTER (WHERE high_value_items IS NULL), count(*) FROM test_contents
UNION ALL SELECT 'addresses', 'flood_zone', count(*) FILTER (WHERE flood_zone IS NULL), count(*) FROM addresses
UNION ALL SELECT 'addresses', 'distance_to_river_km', count(*) FILTER (WHERE distance_to_river_km IS NULL), count(*) FROM addresses
UNION ALL SELECT 'addresses', 'elevation_m', count(*) FILTER (WHERE elevation_m IS NULL), count(*) FROM addresses;
