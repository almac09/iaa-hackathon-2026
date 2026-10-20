-- 04 Claim development and experience by year (audit A03, A06; decision D-03).
-- Claims are valued at the valuation date. Late-reported (IBNR) claims are estimated from the
-- report-lag distribution of loss years old enough to be fully observed.

-- Reporting completeness by line, from loss years up to lag_cdf_max_loss_year.
CREATE OR REPLACE TABLE ev_04_report_lag_by_line AS
SELECT line,
       round(avg((datediff('day', loss_date, report_date) <= 30)::INT), 4) AS share_within_30d,
       round(avg((datediff('day', loss_date, report_date) <= 90)::INT), 4) AS share_within_90d,
       round(avg((datediff('day', loss_date, report_date) <= 182)::INT), 4) AS share_within_182d,
       round(avg((datediff('day', loss_date, report_date) <= 365)::INT), 4) AS share_within_365d
FROM claims
WHERE status <> 'repudiated' AND year(loss_date) BETWEEN 2021 AND getvariable('lag_cdf_max_loss_year')
GROUP BY line;

-- Contents: empirical CDF of report lag, then expected unreported count for 2024 losses.
CREATE OR REPLACE TABLE contents_lag_cdf AS
SELECT k AS lag_days, avg((datediff('day', loss_date, report_date) <= k)::INT) AS completeness
FROM claims, range(0, 800) AS t(k)
WHERE line = 'contents' AND status <> 'repudiated'
  AND year(loss_date) BETWEEN 2021 AND getvariable('lag_cdf_max_loss_year')
GROUP BY k;

CREATE OR REPLACE TABLE ev_04_contents_ibnr_2024 AS
WITH c AS (
    SELECT peril, loss_date FROM claims
    WHERE line = 'contents' AND status <> 'repudiated'
      AND year(loss_date) = year(getvariable('valuation_date'))
)
SELECT coalesce(c.peril, 'ALL') AS peril,
       count(*) AS n_reported,
       round(sum(1 / f.completeness) - count(*), 1) AS n_ibnr_estimate,
       round(100 * (sum(1 / f.completeness) / count(*) - 1), 2) AS ibnr_pct
FROM c
JOIN contents_lag_cdf f ON f.lag_days = datediff('day', c.loss_date, getvariable('valuation_date'))
GROUP BY ROLLUP (c.peril);

CREATE OR REPLACE TABLE ev_04_contents_open_vs_closed AS
SELECT peril, year(loss_date) AS loss_year,
       count(*) AS n_claims,
       round(avg((status = 'open')::INT), 4) AS open_share,
       round(avg(incurred_to_date), 0) AS mean_incurred,
       round(avg(incurred_to_date) FILTER (WHERE status = 'closed'), 0) AS mean_incurred_closed,
       round(avg(incurred_to_date) FILTER (WHERE status = 'open'), 0) AS mean_incurred_open
FROM claims
WHERE line = 'contents' AND status <> 'repudiated' AND year(loss_date) >= 2021
GROUP BY ALL;

-- All lines, for the team: earned premium = annual written premium x exposure (A12).
CREATE OR REPLACE TABLE ev_04_loss_ratio_by_line_year AS
WITH r AS (
    SELECT line, row_id, policy_id, period_start, period_end, written_premium,
           written_premium * (datediff('day', period_start, period_end) + 1) / getvariable('exposure_basis_days') AS earned_premium
    FROM all_policy_rows WHERE src = 'train'
),
c AS (
    SELECT r.row_id, sum(cl.incurred_to_date - cl.recoveries) AS net_cost
    FROM claims cl JOIN r ON r.policy_id = cl.policy_id AND cl.loss_date BETWEEN r.period_start AND r.period_end
    WHERE cl.status <> 'repudiated'
    GROUP BY r.row_id
)
SELECT r.line, year(r.period_start) AS yr,
       round(sum(r.written_premium), 0) AS written_premium_sum,
       round(sum(r.earned_premium), 0) AS earned_premium,
       round(sum(c.net_cost), 0) AS net_incurred,
       round(sum(c.net_cost) / sum(r.earned_premium), 4) AS loss_ratio,
       round(sum(c.net_cost) / sum(r.written_premium), 4) AS loss_ratio_on_written_superseded
FROM r LEFT JOIN c USING (row_id)
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_04_contents_experience_by_year AS
SELECT yr,
       round(sum(exposure), 1) AS exposure,
       sum(n_claims) AS n_claims,
       round(1000 * sum(n_claims) / sum(exposure), 2) AS freq_per_1000,
       round(sum(claim_cost) / sum(n_claims), 0) AS mean_severity,
       round(sum(claim_cost) / sum(exposure), 2) AS loss_cost,
       round(sum(earned_premium), 0) AS earned_premium,
       round(sum(claim_cost) / sum(earned_premium), 4) AS loss_ratio,
       round(sum(claim_cost) / sum(written_premium), 4) AS loss_ratio_on_written_superseded
FROM contents_model
WHERE src = 'train'
GROUP BY yr;
