-- 05 Contents claim-size distribution (audit A09, first part).

CREATE OR REPLACE TABLE ev_05_severity_by_peril AS
WITH c AS (SELECT 'ALL' AS peril, net_cost FROM contents_claims UNION ALL SELECT peril, net_cost FROM contents_claims)
SELECT peril,
       count(*) AS n_claims,
       round(avg(net_cost), 0) AS mean,
       round(stddev_samp(net_cost) / avg(net_cost), 3) AS cv,
       round(quantile_cont(net_cost, 0.10), 0) AS p10,
       round(quantile_cont(net_cost, 0.50), 0) AS p50,
       round(quantile_cont(net_cost, 0.90), 0) AS p90,
       round(quantile_cont(net_cost, 0.99), 0) AS p99,
       round(max(net_cost), 0) AS max,
       round(skewness(ln(net_cost)), 3) AS skew_of_log
FROM c
WHERE net_cost > 0
GROUP BY peril;

CREATE OR REPLACE TABLE ev_05_cost_concentration AS
WITH r AS (SELECT net_cost, row_number() OVER (ORDER BY net_cost DESC, claim_id) AS rk, count(*) OVER () AS n FROM contents_claims)
SELECT round(100 * sum(net_cost) FILTER (WHERE rk <= 0.01 * n) / sum(net_cost), 2) AS top_1pct_share,
       round(100 * sum(net_cost) FILTER (WHERE rk <= 0.05 * n) / sum(net_cost), 2) AS top_5pct_share,
       round(100 * sum(net_cost) FILTER (WHERE rk <= 0.20 * n) / sum(net_cost), 2) AS top_20pct_share
FROM r;

CREATE OR REPLACE TABLE ev_05_size_band_mix AS
WITH c AS (
    SELECT *, CASE WHEN net_cost < getvariable('band_1') THEN 'band_1'
                   WHEN net_cost < getvariable('band_2') THEN 'band_2'
                   WHEN net_cost < getvariable('band_3') THEN 'band_3'
                   ELSE 'band_4' END AS size_band
    FROM contents_claims
)
SELECT size_band,
       count(*) AS n_claims,
       round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_claims,
       round(100 * sum(net_cost) / sum(sum(net_cost)) OVER (), 2) AS pct_cost,
       round(100.0 * count(*) FILTER (WHERE peril = 'accidental_damage') / count(*), 2) AS pct_accidental_damage,
       round(100.0 * count(*) FILTER (WHERE peril = 'theft') / count(*), 2) AS pct_theft,
       round(100.0 * count(*) FILTER (WHERE peril = 'escape_of_water') / count(*), 2) AS pct_escape_of_water,
       round(100.0 * count(*) FILTER (WHERE peril = 'flood') / count(*), 2) AS pct_flood
FROM c
GROUP BY size_band;

CREATE OR REPLACE TABLE ev_05_claims_above_sum_insured AS
SELECT count(*) FILTER (WHERE c.net_cost > m.sum_insured) AS n_above_sum_insured,
       round(max(c.net_cost / m.sum_insured), 4) AS max_cost_to_sum_insured
FROM contents_claims c JOIN contents_model m USING (row_id);
