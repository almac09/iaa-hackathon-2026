-- Build the contents modelling table: one row per policy period, cleaned, with claims and joined attributes.
CREATE OR REPLACE MACRO clean_si(si) AS CASE WHEN si >= 1e6 THEN si / 1000 ELSE si END;

CREATE OR REPLACE TABLE clm AS
SELECT p.row_id, c.claim_id, c.peril, c.status, c.loss_date, c.report_date,
       c.incurred_to_date - c.recoveries AS net_inc
FROM 'data/claims.parquet' c
JOIN 'data/policies_contents.parquet' p
  ON c.policy_id = p.policy_id AND c.loss_date BETWEEN p.period_start AND p.period_end
WHERE c.line = 'contents' AND c.status <> 'repudiated';

CREATE OR REPLACE TABLE agg AS
SELECT row_id, count(*) n_claims, sum(net_inc) AS claim_cost,
       sum(net_inc) FILTER (WHERE peril='theft') cost_theft,
       sum(net_inc) FILTER (WHERE peril='escape_of_water') cost_eow,
       sum(net_inc) FILTER (WHERE peril='accidental_damage') cost_ad,
       sum(net_inc) FILTER (WHERE peril='flood') cost_flood,
       count(*) FILTER (WHERE status='open') n_open
FROM clm GROUP BY row_id;

CREATE OR REPLACE TABLE base AS
SELECT 'train' AS src, p.* EXCLUDE (sum_insured, exposure, renewal_decision), p.renewal_decision,
       p.exposure AS exposure_raw, (p.period_end - p.period_start + 1) / 365.0 AS exposure,
       p.sum_insured AS sum_insured_raw, clean_si(p.sum_insured) AS sum_insured,
       year(p.period_start) AS yr
FROM 'data/policies_contents.parquet' p
UNION ALL BY NAME
SELECT 'test' AS src, t.* EXCLUDE (sum_insured, exposure),
       t.exposure AS exposure_raw, (t.period_end - t.period_start + 1) / 365.0 AS exposure,
       t.sum_insured AS sum_insured_raw, clean_si(t.sum_insured) AS sum_insured,
       year(t.period_start) AS yr
FROM 'data/test_contents.parquet' t;

COPY (
SELECT b.*,
       coalesce(a.n_claims,0) n_claims, coalesce(a.claim_cost,0) AS claim_cost,
       coalesce(a.cost_theft,0) cost_theft, coalesce(a.cost_eow,0) cost_eow,
       coalesce(a.cost_ad,0) cost_ad, coalesce(a.cost_flood,0) cost_flood, coalesce(a.n_open,0) n_open,
       cu.* EXCLUDE (customer_id), ad.* EXCLUDE (address_id), w.* EXCLUDE (address_id, year)
FROM base b
LEFT JOIN agg a USING (row_id)
LEFT JOIN 'data/customers.parquet' cu USING (customer_id)
LEFT JOIN 'data/addresses.parquet' ad USING (address_id)
LEFT JOIN 'data/weather.parquet' w ON w.address_id = b.address_id AND w.year = b.yr
) TO 'work/contents.parquet' (FORMAT parquet);
