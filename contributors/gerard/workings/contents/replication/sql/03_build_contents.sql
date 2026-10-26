-- 03 Build the contents modelling table (audit A04-A06; decisions D-01, D-02, D-03).
-- One row per policy period, train and test stacked. Raw values are always kept beside the
-- cleaned ones (exposure_raw, sum_insured_raw) so every fix can be compared row by row.

-- Address corrections (A28; decisions D-26, D-27), found in the final review (A27). Every later join uses
-- addresses_clean; sql/01-02 still profile the raw table, and the raw values are kept beside the corrected ones.
-- D-26: region R11's elevation is recorded in feet (3.25-3.30 times every other region at every percentile; the
--       others cap at 60.0 m and R11 at 196.9 = 60 x 3.28084), so it is converted to metres.
-- D-27: year_built = 1900 is a placeholder on about 19,000 addresses (10.8 times any neighbouring year), whose
--       ratings and claims look like houses of every age; it is replaced by the typical year built of addresses
--       with the same energy rating.
CREATE OR REPLACE VIEW addresses_clean AS
WITH built AS (
    SELECT energy_rating, round(avg(year_built)) AS year_built_typical
    FROM addresses WHERE year_built <> 1900 GROUP BY energy_rating
)
SELECT a.* REPLACE (
           CASE WHEN getvariable('d26_r11_elevation_feet') AND a.region = 'R11' THEN round(a.elevation_m / 3.28084, 2)
                ELSE a.elevation_m END AS elevation_m,
           CASE WHEN getvariable('d27_year_built_placeholder') AND a.year_built = 1900 THEN CAST(b.year_built_typical AS BIGINT)
                ELSE a.year_built END AS year_built),
       a.elevation_m AS elevation_m_raw, a.year_built AS year_built_raw
FROM addresses a
LEFT JOIN built b ON b.energy_rating IS NOT DISTINCT FROM a.energy_rating;

CREATE OR REPLACE TABLE contents_rows AS
WITH stacked AS (
    SELECT 'train' AS src, * EXCLUDE (renewal_decision), renewal_decision FROM policies_contents
    UNION ALL BY NAME
    SELECT 'test' AS src, * FROM test_contents
)
SELECT src, row_id, policy_id, customer_id, address_id, period_start, period_end,
       year(period_start) AS yr,
       exposure AS exposure_raw,
       CASE WHEN getvariable('d01_recompute_exposure')
            THEN (datediff('day', period_start, period_end) + 1) / getvariable('exposure_basis_days')
            ELSE exposure END AS exposure,
       sum_insured AS sum_insured_raw,
       CASE WHEN getvariable('d02_rescale_sum_insured') AND sum_insured >= getvariable('si_threshold')
            THEN sum_insured / getvariable('si_divisor')
            ELSE sum_insured END AS sum_insured,
       policyholder_age, tenure, security_level, high_value_items, accidental_damage_cover,
       works_from_home, lodgers, bicycle_cover, voluntary_excess, compulsory_excess, channel,
       payment_method, missed_payments_12m, new_business, renewal_count, days_quote_to_inception,
       discount_applied, written_premium, customer_service_contacts, renewal_decision
FROM stacked;

-- Earned premium (A12): written_premium is annual, so the premium earned by the row is written x exposure.
ALTER TABLE contents_rows ADD COLUMN earned_premium DOUBLE;
UPDATE contents_rows SET earned_premium = written_premium * exposure;

-- Duplicate claims (D-19, audit A18): 604 claims, in all three lines, repeat another claim on every field except
-- claim_id (policy, peril, loss and report dates, status and amounts). One copy of each is kept. Every later step
-- reads claims_clean; sql/01 and sql/02 profile the raw table.
CREATE OR REPLACE TABLE claims_clean AS
SELECT * FROM claims
QUALIFY NOT getvariable('d19_dedupe_claims') OR row_number() OVER (
    PARTITION BY policy_id, line, peril, loss_date, report_date, status, paid_to_date, incurred_to_date, recoveries
    ORDER BY claim_id) = 1;

-- Claim-level target (D-03): non-repudiated claims matched to the policy row whose period contains
-- the loss date. Net cost = incurred (already net of excess) less recoveries.
CREATE OR REPLACE TABLE contents_claims AS
SELECT r.row_id, c.claim_id, c.peril, c.status, c.loss_date, c.report_date,
       c.incurred_to_date, c.recoveries,
       c.incurred_to_date - c.recoveries AS net_cost
FROM claims_clean c
JOIN contents_rows r
  ON r.src = 'train' AND r.policy_id = c.policy_id AND c.loss_date BETWEEN r.period_start AND r.period_end
WHERE c.line = 'contents' AND c.status <> 'repudiated';

CREATE OR REPLACE TABLE contents_model AS
WITH agg AS (
    SELECT row_id,
           count(*) AS n_claims,
           sum(net_cost) AS claim_cost,
           sum(net_cost) FILTER (WHERE peril = 'theft') AS cost_theft,
           sum(net_cost) FILTER (WHERE peril = 'escape_of_water') AS cost_eow,
           sum(net_cost) FILTER (WHERE peril = 'accidental_damage') AS cost_ad,
           sum(net_cost) FILTER (WHERE peril = 'flood') AS cost_flood,
           count(*) FILTER (WHERE peril = 'theft') AS n_theft,
           count(*) FILTER (WHERE peril = 'escape_of_water') AS n_eow,
           count(*) FILTER (WHERE peril = 'accidental_damage') AS n_ad,
           count(*) FILTER (WHERE peril = 'flood') AS n_flood,
           sum(net_cost) FILTER (WHERE net_cost < getvariable('band_1')) AS cost_band_1,
           sum(net_cost) FILTER (WHERE net_cost >= getvariable('band_1') AND net_cost < getvariable('band_2')) AS cost_band_2,
           sum(net_cost) FILTER (WHERE net_cost >= getvariable('band_2') AND net_cost < getvariable('band_3')) AS cost_band_3,
           sum(net_cost) FILTER (WHERE net_cost >= getvariable('band_3')) AS cost_band_4,
           count(*) FILTER (WHERE status = 'open') AS n_open
    FROM contents_claims
    GROUP BY row_id
)
SELECT r.*,
       coalesce(a.n_claims, 0) AS n_claims, coalesce(a.claim_cost, 0) AS claim_cost,
       coalesce(a.cost_theft, 0) AS cost_theft, coalesce(a.cost_eow, 0) AS cost_eow,
       coalesce(a.cost_ad, 0) AS cost_ad, coalesce(a.cost_flood, 0) AS cost_flood,
       coalesce(a.n_theft, 0) AS n_theft, coalesce(a.n_eow, 0) AS n_eow,
       coalesce(a.n_ad, 0) AS n_ad, coalesce(a.n_flood, 0) AS n_flood,
       coalesce(a.cost_band_1, 0) AS cost_band_1, coalesce(a.cost_band_2, 0) AS cost_band_2,
       coalesce(a.cost_band_3, 0) AS cost_band_3, coalesce(a.cost_band_4, 0) AS cost_band_4,
       coalesce(a.n_open, 0) AS n_open,
       cu.* EXCLUDE (customer_id),
       ad.* EXCLUDE (address_id),
       w.* EXCLUDE (address_id, year)
FROM contents_rows r
LEFT JOIN agg a USING (row_id)
LEFT JOIN customers cu USING (customer_id)
LEFT JOIN addresses_clean ad USING (address_id)
LEFT JOIN weather w ON w.address_id = r.address_id AND w.year = r.yr;

-- Evidence: before and after for each fix, and the claim reconciliation.
CREATE OR REPLACE TABLE ev_03_exposure_before_after AS
SELECT src,
       count(*) FILTER (WHERE exposure_raw > 1.0001) AS rows_above_1_before,
       count(*) FILTER (WHERE exposure > 1.0001) AS rows_above_1_after,
       round(sum(exposure_raw), 1) AS total_exposure_before,
       round(sum(exposure), 1) AS total_exposure_after,
       max(exposure_raw) AS max_before,
       round(max(exposure), 4) AS max_after
FROM contents_rows GROUP BY src;

CREATE OR REPLACE TABLE ev_03_sum_insured_before_after AS
SELECT src,
       count(*) FILTER (WHERE sum_insured_raw >= getvariable('si_threshold')) AS rows_at_or_above_threshold_before,
       count(*) FILTER (WHERE sum_insured >= getvariable('si_threshold')) AS rows_at_or_above_threshold_after,
       round(avg(sum_insured_raw), 0) AS mean_before,
       round(avg(sum_insured), 0) AS mean_after,
       max(sum_insured_raw) AS max_before,
       max(sum_insured) AS max_after
FROM contents_rows GROUP BY src;

CREATE OR REPLACE TABLE ev_03_claim_reconciliation AS
WITH raw_c AS (SELECT * FROM claims WHERE line = 'contents'),
     all_c AS (SELECT * FROM claims_clean WHERE line = 'contents')
SELECT 1 AS step, 'all contents claims' AS description, count(*) AS n_claims, round(sum(incurred_to_date - recoveries), 2) AS net_cost FROM raw_c
UNION ALL SELECT 2, 'less duplicates', (SELECT count(*) FROM all_c) - count(*),
          round((SELECT sum(incurred_to_date - recoveries) FROM all_c) - sum(incurred_to_date - recoveries), 2) FROM raw_c
UNION ALL SELECT 3, 'less repudiated', -count(*), -round(sum(incurred_to_date - recoveries), 2) FROM all_c WHERE status = 'repudiated'
UNION ALL SELECT 4, 'less outside any policy period', -(SELECT count(*) FROM all_c WHERE status <> 'repudiated') + (SELECT count(*) FROM contents_claims),
          -round((SELECT sum(incurred_to_date - recoveries) FROM all_c WHERE status <> 'repudiated') - (SELECT sum(net_cost) FROM contents_claims), 2)
UNION ALL SELECT 5, 'claims in target', count(*), round(sum(net_cost), 2) FROM contents_claims
UNION ALL SELECT 6, 'check: target total on rows', sum(n_claims), round(sum(claim_cost), 2) FROM contents_model;

CREATE OR REPLACE TABLE ev_03_join_completeness AS
SELECT src, count(*) AS n_rows,
       count(*) FILTER (WHERE date_of_birth IS NULL) AS n_no_customer,
       count(*) FILTER (WHERE region IS NULL) AS n_no_address,
       count(*) FILTER (WHERE rainfall_mm IS NULL) AS n_no_weather
FROM contents_model GROUP BY src;

CREATE OR REPLACE TABLE evr_03_exposure_changed_rows AS
SELECT row_id, yr, period_start, period_end, exposure_raw AS exposure_before, round(exposure, 6) AS exposure_after
FROM contents_rows WHERE abs(exposure_raw - exposure) > 0.0005;

CREATE OR REPLACE TABLE evr_03_sum_insured_changed_rows AS
SELECT src, row_id, policy_id, yr, sum_insured_raw AS sum_insured_before, sum_insured AS sum_insured_after, written_premium
FROM contents_rows WHERE sum_insured_raw <> sum_insured;
