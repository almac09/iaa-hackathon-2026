-- 10 Development to ultimate for all three lines (audit A15; decisions D-10, D-11). Runs after sql/09.
--
-- IBNR (D-10): a cost-weighted report-lag distribution F by line and peril, estimated from every loss year
-- with the right-truncation product-limit estimator: a claim is only seen if its lag is at most its age at
-- the valuation date, so F(l) = prod over k > l of (1 - d_k / n_k), where d_k is the cost reported at lag k
-- and n_k the cost with lag <= k among claims old enough to have shown lag k. Each reported claim is grossed
-- up by 1 / F(age of its loss date), averaged by loss quarter (see below). The count-weighted version is
-- reported for comparison with sql/04.
--
-- IBNER (D-11): the data is a single snapshot, so movement on reported claims cannot be measured directly.
-- For motor bodily injury, the only peril whose like-for-like claim size falls as its open share rises, each
-- loss year is lifted to the 2021 level of claims reported within 90 days, with no inflation (a lower bound;
-- inflation is left to the trend term). All other perils keep their case estimates.

CREATE OR REPLACE TABLE dev_claims AS
SELECT line, peril, row_id, claim_id, loss_date, status, net_cost,
       datediff('day', loss_date, report_date) AS lag,
       datediff('day', loss_date, getvariable('valuation_date')) AS age
FROM line_claims;

CREATE OR REPLACE TABLE dev_pattern AS
WITH events AS (
    SELECT line, peril, lag AS k, 1.0 AS d_cnt, net_cost AS d_cost, 1.0 AS add_cnt, net_cost AS add_cost FROM dev_claims
    UNION ALL
    SELECT line, peril, age + 1, 0.0, 0.0, -1.0, -net_cost FROM dev_claims
),
by_k AS (
    SELECT line, peril, k, sum(d_cnt) AS d_cnt, sum(d_cost) AS d_cost, sum(add_cnt) AS add_cnt, sum(add_cost) AS add_cost
    FROM events GROUP BY ALL
),
risk AS (
    SELECT *, sum(add_cnt) OVER w AS n_cnt, sum(add_cost) OVER w AS n_cost
    FROM by_k
    WINDOW w AS (PARTITION BY line, peril ORDER BY k ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT line, peril, k, d_cnt, n_cnt, d_cost, n_cost,
       -- log of (1 - d/n) for lag k. d = n only at the shortest lag in the data (1 day: nothing is reported
       -- on the loss date), where F is zero below it; every reported claim is older than that, so it is skipped.
       CASE WHEN d_cnt < n_cnt THEN ln(1 - d_cnt / n_cnt) ELSE 0 END AS log_s_cnt,
       CASE WHEN d_cost < n_cost THEN ln(1 - d_cost / n_cost) ELSE 0 END AS log_s_cost
FROM risk
WHERE d_cnt > 0;

-- 1 / F(age) for every distinct (line, peril, age) that has a reported claim.
CREATE OR REPLACE TABLE dev_factor_by_age AS
SELECT a.line, a.peril, a.age,
       exp(-coalesce(sum(p.log_s_cnt), 0)) AS f_cnt,
       exp(-coalesce(sum(p.log_s_cost), 0)) AS f_cost
FROM (SELECT DISTINCT line, peril, age FROM dev_claims) a
LEFT JOIN dev_pattern p ON p.line = a.line AND p.peril = a.peril AND p.k > a.age
GROUP BY ALL;

CREATE OR REPLACE TABLE ibner_factor AS
WITH cohort AS (
    SELECT line, peril, year(loss_date) AS ay, avg(net_cost) AS mean_net, avg((status = 'open')::INT) AS open_share, count(*) AS n
    FROM dev_claims
    WHERE lag <= 90 AND loss_date <= make_date(year(loss_date), 9, 30)
    GROUP BY ALL
)
SELECT c.line, c.peril, c.ay, c.n, c.mean_net, c.open_share,
       greatest(b.mean_net / c.mean_net, 1.0) AS suggested,
       CASE WHEN getvariable('d11_ibner_motor_bi') AND c.line = 'motor' AND c.peril = 'bodily_injury'
            THEN greatest(b.mean_net / c.mean_net, 1.0) ELSE 1.0 END AS applied
FROM cohort c
JOIN cohort b ON b.line = c.line AND b.peril = c.peril AND b.ay = 2021;

-- The IBNR factor is applied as the average for the line, peril and loss quarter, not claim by claim: per claim,
-- a slow-reporting claim seen a few days after its loss would carry the whole quarter's IBNR (x60 for motor
-- bodily injury at 3 days), creating spurious large row targets. Totals are the same either way.
CREATE OR REPLACE TABLE ibnr_factor_by_quarter AS
SELECT c.line, c.peril, year(c.loss_date) AS ay, quarter(c.loss_date) AS aq,
       sum(c.net_cost * f.f_cost) / sum(c.net_cost) AS f_cost
FROM dev_claims c JOIN dev_factor_by_age f USING (line, peril, age)
GROUP BY ALL;

CREATE OR REPLACE TABLE line_claims_dev AS
SELECT c.*,
       CASE WHEN getvariable('d10_develop_ibnr') THEN q.f_cost ELSE 1.0 END * coalesce(i.applied, 1.0) AS dev_factor,
       c.net_cost * CASE WHEN getvariable('d10_develop_ibnr') THEN q.f_cost ELSE 1.0 END * coalesce(i.applied, 1.0) AS net_cost_dev
FROM dev_claims c
JOIN ibnr_factor_by_quarter q ON q.line = c.line AND q.peril = c.peril AND q.ay = year(c.loss_date) AND q.aq = quarter(c.loss_date)
LEFT JOIN ibner_factor i ON i.line = c.line AND i.peril = c.peril AND i.ay = year(c.loss_date);

-- Evidence: reported vs developed by line, peril and loss year, with the count-based IBNR for comparison.
CREATE OR REPLACE TABLE ev_10_development AS
SELECT c.line, c.peril, year(c.loss_date) AS loss_year, count(*) AS n_reported,
       round(sum(f.f_cnt) - count(*), 1) AS n_ibnr_count_basis,
       round(sum(c.net_cost), 0) AS reported_cost,
       round(sum(c.net_cost * f.f_cost), 0) AS cost_after_ibnr,
       round(sum(c.net_cost_dev), 0) AS cost_developed,
       round(sum(c.net_cost_dev) / sum(c.net_cost), 4) AS dev_factor
FROM line_claims_dev c JOIN dev_factor_by_age f USING (line, peril, age)
GROUP BY GROUPING SETS ((c.line, c.peril, year(c.loss_date)), (c.line, year(c.loss_date)));

CREATE OR REPLACE TABLE ev_10_ibner_check AS
SELECT line, peril, ay, n, round(mean_net, 0) AS mean_net_reported_within_90d, round(open_share, 4) AS open_share,
       round(suggested, 4) AS suggested_vs_2021, round(applied, 4) AS applied
FROM ibner_factor;

CREATE OR REPLACE TABLE ev_10_report_lag_pattern AS
SELECT p.line, p.peril, m.max_lag_days,
       round(exp(coalesce(sum(p.log_s_cost) FILTER (WHERE p.k > 30), 0)), 4) AS cost_share_reported_by_30d,
       round(exp(coalesce(sum(p.log_s_cost) FILTER (WHERE p.k > 90), 0)), 4) AS cost_share_reported_by_90d,
       round(exp(coalesce(sum(p.log_s_cost) FILTER (WHERE p.k > 365), 0)), 4) AS cost_share_reported_by_365d
FROM dev_pattern p
JOIN (SELECT line, peril, max(lag) AS max_lag_days FROM dev_claims GROUP BY ALL) m USING (line, peril)
GROUP BY ALL;
