-- 11 Modelling tables for the three-line XGBoost run (audit A15; decisions D-10 to D-12). Runs after sql/10.
-- claim_cost is the reported target (as in sql/03); claim_cost_dev is developed (D-10, D-11) and capped per
-- claim at large_claim_cap (D-12). The capped excess is spread back over the line as a loading in Python.

CREATE OR REPLACE TABLE line_target AS
SELECT line, row_id,
       count(*) AS n_claims,
       sum(net_cost) AS claim_cost,
       sum(CASE WHEN getvariable('d12_cap_large_claims') THEN least(net_cost_dev, getvariable('large_claim_cap'))
                ELSE net_cost_dev END) AS claim_cost_dev,
       sum(CASE WHEN getvariable('d12_cap_large_claims') THEN greatest(net_cost_dev - getvariable('large_claim_cap'), 0)
                ELSE 0 END) AS excess_dev
FROM line_claims_dev
GROUP BY ALL;

CREATE OR REPLACE TABLE buildings_model AS
SELECT r.*,
       coalesce(t.n_claims, 0) AS n_claims, coalesce(t.claim_cost, 0) AS claim_cost,
       coalesce(t.claim_cost_dev, 0) AS claim_cost_dev, coalesce(t.excess_dev, 0) AS excess_dev,
       cu.* EXCLUDE (customer_id), ad.* EXCLUDE (address_id), w.* EXCLUDE (address_id, year),
       -- Peril interactions (A28): clay soil in a dry summer drives subsidence; a high or unknown flood zone with
       -- heavy rain drives flood. Used only where config.json lists them.
       coalesce((ad.soil_type = 'clay')::INT * w.dry_summer_index, 0) AS x_clay_dry,
       coalesce((coalesce(ad.flood_zone, 99) >= 3)::INT * w.rainfall_mm / 1000, 0) AS x_fz3_rain
FROM buildings_rows r
LEFT JOIN line_target t ON t.line = 'buildings' AND t.row_id = r.row_id
LEFT JOIN customers cu USING (customer_id)
LEFT JOIN addresses_clean ad USING (address_id)
LEFT JOIN weather w ON w.address_id = r.address_id AND w.year = r.yr;

CREATE OR REPLACE TABLE motor_model AS
SELECT r.*,
       coalesce(t.n_claims, 0) AS n_claims, coalesce(t.claim_cost, 0) AS claim_cost,
       coalesce(t.claim_cost_dev, 0) AS claim_cost_dev, coalesce(t.excess_dev, 0) AS excess_dev,
       cu.* EXCLUDE (customer_id), ad.* EXCLUDE (address_id), w.* EXCLUDE (address_id, year)
FROM motor_rows r
LEFT JOIN line_target t ON t.line = 'motor' AND t.row_id = r.row_id
LEFT JOIN customers cu USING (customer_id)
LEFT JOIN addresses_clean ad USING (address_id)
LEFT JOIN weather w ON w.address_id = r.address_id AND w.year = r.yr;

-- Contents keeps the banded features of sql/08 (A13) and gains the developed target.
CREATE OR REPLACE TABLE contents_model_dev AS
SELECT f.*, coalesce(t.claim_cost_dev, 0) AS claim_cost_dev, coalesce(t.excess_dev, 0) AS excess_dev
FROM contents_features f
LEFT JOIN line_target t ON t.line = 'contents' AND t.row_id = f.row_id;

CREATE OR REPLACE TABLE ev_11_target_by_line_year AS
WITH m AS (
    SELECT 'buildings' AS line, src, yr, exposure, claim_cost, claim_cost_dev, excess_dev FROM buildings_model
    UNION ALL SELECT 'motor', src, yr, exposure, claim_cost, claim_cost_dev, excess_dev FROM motor_model
    UNION ALL SELECT 'contents', src, yr, exposure, claim_cost, claim_cost_dev, excess_dev FROM contents_model_dev
)
SELECT line, yr, count(*) AS n_rows, round(sum(exposure), 1) AS exposure,
       round(sum(claim_cost), 0) AS reported_cost, round(sum(claim_cost_dev), 0) AS developed_capped_cost,
       round(sum(excess_dev), 0) AS excess_over_cap,
       round(sum(claim_cost) / sum(exposure), 2) AS reported_loss_cost,
       round((sum(claim_cost_dev) + sum(excess_dev)) / sum(exposure), 2) AS developed_loss_cost
FROM m WHERE src = 'train'
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_11_join_completeness AS
SELECT 'buildings' AS line, src, count(*) AS n_rows, count(*) FILTER (WHERE date_of_birth IS NULL) AS n_no_customer,
       count(*) FILTER (WHERE region IS NULL) AS n_no_address, count(*) FILTER (WHERE rainfall_mm IS NULL) AS n_no_weather
FROM buildings_model GROUP BY ALL
UNION ALL
SELECT 'motor', src, count(*), count(*) FILTER (WHERE date_of_birth IS NULL), count(*) FILTER (WHERE region IS NULL),
       count(*) FILTER (WHERE rainfall_mm IS NULL)
FROM motor_model GROUP BY ALL;
