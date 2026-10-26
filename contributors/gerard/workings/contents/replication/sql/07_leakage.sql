-- 07 Leakage check on customer_service_contacts (audit A08; decision D-04).
-- If contacts are driven by claims in training but not in test, the field leaks the target.

CREATE OR REPLACE TABLE ev_07_contacts_train_vs_test AS
WITH t AS (
    SELECT src, least(customer_service_contacts, 4) AS contacts_capped_4, count(*) AS n_rows
    FROM contents_model GROUP BY ALL
)
SELECT src, contacts_capped_4, n_rows,
       round(100.0 * n_rows / sum(n_rows) OVER (PARTITION BY src), 3) AS pct_rows
FROM t;

CREATE OR REPLACE TABLE ev_07_contacts_by_claim_status AS
SELECT 'train with claim' AS grp, round(avg(customer_service_contacts), 4) AS mean_contacts, count(*) AS n_rows
FROM contents_model WHERE src = 'train' AND n_claims > 0
UNION ALL
SELECT 'train without claim', round(avg(customer_service_contacts), 4), count(*)
FROM contents_model WHERE src = 'train' AND n_claims = 0
UNION ALL
SELECT 'test', round(avg(customer_service_contacts), 4), count(*)
FROM contents_model WHERE src = 'test';

CREATE OR REPLACE TABLE ev_07_contacts_oneway AS
SELECT least(customer_service_contacts, 4) AS contacts_capped_4,
       round(sum(exposure), 1) AS exposure,
       sum(n_claims) AS n_claims,
       round(1000 * sum(n_claims) / sum(exposure), 2) AS freq_per_1000
FROM contents_model WHERE src = 'train'
GROUP BY ALL;
