-- 09 Buildings and motor rows, and the claim-level target for all three lines (audit A15; decisions D-01 to D-03).
-- Same rules as sql/03 for contents: exposure recomputed from the period dates (D-01), raw values kept beside
-- the fixed ones, claims matched to the row whose period contains the loss date, repudiated claims excluded,
-- net cost = incurred (already net of excess) less recoveries (D-03). Runs after sql/03.

CREATE OR REPLACE MACRO fixed_exposure(raw, s, e) AS
    CASE WHEN getvariable('d01_recompute_exposure')
         THEN (datediff('day', s, e) + 1) / getvariable('exposure_basis_days') ELSE raw END;

-- Buildings sum insured: values of 52m and above are 1,000 times too large (A15). The threshold is 10m, not
-- the contents 1m, because buildings has a genuine 1.06m risk and nothing between 2m and 36m.
CREATE OR REPLACE TABLE buildings_rows AS
WITH stacked AS (
    SELECT 'train' AS src, * FROM policies_buildings
    UNION ALL BY NAME
    SELECT 'test' AS src, * FROM test_buildings
)
SELECT * EXCLUDE (exposure, sum_insured, construction_type),
       year(period_start) AS yr,
       exposure AS exposure_raw,
       fixed_exposure(exposure, period_start, period_end) AS exposure,
       sum_insured AS sum_insured_raw,
       CASE WHEN getvariable('d02_rescale_sum_insured') AND sum_insured >= getvariable('bld_si_threshold')
            THEN sum_insured / getvariable('si_divisor') ELSE sum_insured END AS sum_insured,
       construction_type AS construction_type_raw,
       -- D-24 (A24): timber is coded T in 2021-22 and W from 2023 (every 2025 row is W), a recode rather than a
       -- different risk; left apart, the code stands in for the later years.
       CASE WHEN getvariable('d24_merge_timber') AND lower(trim(construction_type)) IN ('t', 'w') THEN 'timber'
            ELSE lower(trim(construction_type)) END AS construction_type
FROM stacked;

-- Motor customer_id is stored without leading zeros on 10% of rows (98,110 training rows; A18), so they found no
-- customer record and no cross-line history. Padded to the 8 characters of the customers table.
CREATE OR REPLACE TABLE motor_rows AS
WITH stacked AS (
    SELECT 'train' AS src, * FROM policies_motor
    UNION ALL BY NAME
    SELECT 'test' AS src, * FROM test_motor
)
SELECT * EXCLUDE (exposure, customer_id),
       lpad(customer_id, 8, '0') AS customer_id,
       customer_id AS customer_id_raw,
       year(period_start) AS yr,
       exposure AS exposure_raw,
       fixed_exposure(exposure, period_start, period_end) AS exposure
FROM stacked;

CREATE OR REPLACE TABLE line_rows AS
SELECT 'contents' AS line, src, row_id, policy_id, period_start, period_end, yr, exposure FROM contents_rows
UNION ALL SELECT 'buildings', src, row_id, policy_id, period_start, period_end, yr, exposure FROM buildings_rows
UNION ALL SELECT 'motor', src, row_id, policy_id, period_start, period_end, yr, exposure FROM motor_rows;

CREATE OR REPLACE TABLE line_claims AS
SELECT r.line, r.row_id, r.yr, c.claim_id, c.peril, c.status, c.loss_date, c.report_date,
       c.incurred_to_date - c.recoveries AS net_cost
FROM claims_clean c
JOIN line_rows r
  ON r.src = 'train' AND r.line = c.line AND r.policy_id = c.policy_id
 AND c.loss_date BETWEEN r.period_start AND r.period_end
WHERE c.status <> 'repudiated';

-- Evidence.
CREATE OR REPLACE TABLE ev_09_fixes AS
SELECT 'buildings' AS line, src,
       count(*) FILTER (WHERE abs(exposure_raw - exposure) > 0.0005) AS n_exposure_changed,
       count(*) FILTER (WHERE sum_insured_raw <> sum_insured) AS n_sum_insured_rescaled,
       max(sum_insured) AS max_sum_insured_after,
       count(*) FILTER (WHERE construction_type_raw <> construction_type) AS n_construction_type_recased,
       count(*) FILTER (WHERE construction_type = 'timber') AS n_timber_merged
FROM buildings_rows GROUP BY ALL
UNION ALL
SELECT 'motor', src, count(*) FILTER (WHERE abs(exposure_raw - exposure) > 0.0005), 0, NULL, 0, 0
FROM motor_rows GROUP BY ALL;

CREATE OR REPLACE TABLE ev_09_motor_customer_padding AS
SELECT src, count(*) AS n_rows, count(*) FILTER (WHERE customer_id_raw <> customer_id) AS n_padded,
       count(*) FILTER (WHERE customer_id NOT IN (SELECT customer_id FROM customers)) AS n_without_customer_after
FROM motor_rows GROUP BY ALL;

CREATE OR REPLACE TABLE ev_09_claim_reconciliation AS
WITH all_c AS (SELECT * FROM claims_clean)
SELECT a.line,
       (SELECT count(*) FROM claims r WHERE r.line = a.line) - count(*) AS n_duplicates_removed,
       count(*) AS n_claims_after_dedupe,
       count(*) FILTER (WHERE a.status = 'repudiated') AS n_repudiated,
       count(*) FILTER (WHERE a.status <> 'repudiated') - (SELECT count(*) FROM line_claims l WHERE l.line = a.line) AS n_outside_any_period,
       (SELECT count(*) FROM line_claims l WHERE l.line = a.line) AS n_in_target,
       round((SELECT sum(net_cost) FROM line_claims l WHERE l.line = a.line), 2) AS net_cost_in_target
FROM all_c a GROUP BY a.line;

CREATE OR REPLACE TABLE ev_09_contents_claims_match AS
SELECT (SELECT count(*) FROM contents_claims) AS n_sql03,
       (SELECT count(*) FROM line_claims WHERE line = 'contents') AS n_sql09,
       (SELECT count(*) FROM contents_claims c JOIN line_claims l USING (claim_id)
         WHERE l.line = 'contents' AND l.row_id = c.row_id AND l.net_cost = c.net_cost) AS n_identical;
