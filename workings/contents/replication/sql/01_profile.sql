-- 01 Profile: row counts, key uniqueness and a claims overview (audit A03).
-- Dialect: DuckDB 1.5. The runner registers one view per source table (claims, policies_contents, ...).
-- Tables named ev_* are aggregate evidence; evr_* are row-level evidence (kept out of Git).

CREATE OR REPLACE TABLE ev_01_row_counts AS
SELECT 'addresses' AS table_name, count(*) AS n_rows FROM addresses
UNION ALL SELECT 'claims', count(*) FROM claims
UNION ALL SELECT 'customers', count(*) FROM customers
UNION ALL SELECT 'policies_buildings', count(*) FROM policies_buildings
UNION ALL SELECT 'policies_contents', count(*) FROM policies_contents
UNION ALL SELECT 'policies_motor', count(*) FROM policies_motor
UNION ALL SELECT 'test_buildings', count(*) FROM test_buildings
UNION ALL SELECT 'test_contents', count(*) FROM test_contents
UNION ALL SELECT 'test_motor', count(*) FROM test_motor
UNION ALL SELECT 'weather', count(*) FROM weather;

CREATE OR REPLACE TABLE ev_01_key_uniqueness AS
SELECT 'claims.claim_id' AS key_name, count(*) AS n_rows, count(DISTINCT claim_id) AS n_distinct FROM claims
UNION ALL SELECT 'customers.customer_id', count(*), count(DISTINCT customer_id) FROM customers
UNION ALL SELECT 'addresses.address_id', count(*), count(DISTINCT address_id) FROM addresses
UNION ALL SELECT 'weather.address_id+year', count(*), count(DISTINCT (address_id, year)) FROM weather
UNION ALL SELECT 'policies_contents.row_id', count(*), count(DISTINCT row_id) FROM policies_contents
UNION ALL SELECT 'test_contents.row_id', count(*), count(DISTINCT row_id) FROM test_contents;

CREATE OR REPLACE TABLE ev_01_claims_by_line_year AS
SELECT line,
       year(loss_date) AS loss_year,
       count(*) AS n_claims,
       count(*) FILTER (WHERE status = 'open') AS n_open,
       count(*) FILTER (WHERE status = 'repudiated') AS n_repudiated,
       round(sum(incurred_to_date) FILTER (WHERE status <> 'repudiated'), 2) AS incurred,
       round(sum(recoveries), 2) AS recoveries
FROM claims
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_01_claims_status_checks AS
SELECT status,
       count(*) AS n_claims,
       count(*) FILTER (WHERE incurred_to_date < paid_to_date) AS n_incurred_below_paid,
       count(*) FILTER (WHERE incurred_to_date = 0) AS n_zero_incurred,
       count(*) FILTER (WHERE recoveries > 0) AS n_with_recoveries,
       count(*) FILTER (WHERE report_date < loss_date) AS n_reported_before_loss
FROM claims
GROUP BY ALL;

CREATE OR REPLACE TABLE evr_01_largest_claims AS
SELECT claim_id, line, peril, loss_date, status,
       round(paid_to_date, 2) AS paid_to_date, round(incurred_to_date, 2) AS incurred_to_date
FROM claims
ORDER BY incurred_to_date DESC
LIMIT 12;
