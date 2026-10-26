-- 12 Claim-history features for every row of every line (audit A16; decision D-14). Runs after sql/09.
-- A row may only see what was known when it was priced: claims with loss AND report date before the row's
-- period start, and exposure from rows that had ended by then. Groups:
--   own    the same line; home lines match customer and address, motor matches customer
--   home   the other home line (buildings for contents and the reverse), matching customer and address
--   cross  motor for home rows and both home lines for motor rows, matching customer
-- Counts: all prior claims (n), those in the year before the period (n1y), and those of hist_large_claim or more
-- (nlarge; A10 found a large claim last year predicts more than a small one). expo = prior exposure in the group.

CREATE OR REPLACE TABLE hist_rows AS
SELECT 'contents' AS line, src, row_id, customer_id, address_id, period_start, period_end, exposure FROM contents_rows
UNION ALL SELECT 'buildings', src, row_id, customer_id, address_id, period_start, period_end, exposure FROM buildings_rows
UNION ALL SELECT 'motor', src, row_id, customer_id, address_id, period_start, period_end, exposure FROM motor_rows;

CREATE OR REPLACE TABLE hist_claims AS
SELECT c.line, r.customer_id, r.address_id, c.loss_date, c.report_date, c.net_cost
FROM line_claims c JOIN hist_rows r USING (line, row_id);

CREATE OR REPLACE TABLE hist_groups AS
SELECT * FROM (VALUES
    ('contents', 'own', 'contents', true), ('contents', 'home', 'buildings', true), ('contents', 'cross', 'motor', false),
    ('buildings', 'own', 'buildings', true), ('buildings', 'home', 'contents', true), ('buildings', 'cross', 'motor', false),
    ('motor', 'own', 'motor', false), ('motor', 'cross', 'contents', false), ('motor', 'cross', 'buildings', false)
) AS t(line, grp, src_line, match_address);

-- Matches are found with equality joins on line and customer only, then filtered on address and dates, so DuckDB
-- can hash-join; a single outer join carrying the date ranges took over ten minutes.
CREATE OR REPLACE TABLE hist_frame AS
SELECT r.line, r.row_id, r.customer_id, r.address_id, r.period_start, g.grp, g.src_line, g.match_address
FROM hist_rows r JOIN hist_groups g ON g.line = r.line;

CREATE OR REPLACE TABLE hist_claim_feats AS
WITH m AS (
    SELECT f.line, f.row_id, f.grp, c.loss_date, c.net_cost, f.period_start
    FROM hist_frame f
    JOIN hist_claims c ON c.line = f.src_line AND c.customer_id = f.customer_id
    WHERE (NOT f.match_address OR c.address_id = f.address_id)
      AND c.loss_date < f.period_start AND c.report_date < f.period_start
)
SELECT line, row_id, grp,
       count(*) AS n,
       count(*) FILTER (WHERE loss_date >= period_start - INTERVAL 365 DAY) AS n1y,
       count(*) FILTER (WHERE net_cost >= getvariable('hist_large_claim')) AS nlarge
FROM m GROUP BY ALL;

CREATE OR REPLACE TABLE hist_expo_feats AS
WITH m AS (
    SELECT f.line, f.row_id, f.grp, p.exposure
    FROM hist_frame f
    JOIN hist_rows p ON p.line = f.src_line AND p.customer_id = f.customer_id
    WHERE p.src = 'train' AND (NOT f.match_address OR p.address_id = f.address_id) AND p.period_end < f.period_start
)
SELECT line, row_id, grp, sum(exposure) AS expo FROM m GROUP BY ALL;

CREATE OR REPLACE TABLE claim_history AS
SELECT f.line, f.row_id,
       max(f.n) FILTER (WHERE f.grp = 'own') AS h_own_n,
       max(f.n1y) FILTER (WHERE f.grp = 'own') AS h_own_n1y,
       max(f.nlarge) FILTER (WHERE f.grp = 'own') AS h_own_nlarge,
       max(f.expo) FILTER (WHERE f.grp = 'own') AS h_own_expo,
       max(f.n) FILTER (WHERE f.grp = 'home') AS h_home_n,
       max(f.n1y) FILTER (WHERE f.grp = 'home') AS h_home_n1y,
       max(f.expo) FILTER (WHERE f.grp = 'home') AS h_home_expo,
       max(f.n) FILTER (WHERE f.grp = 'cross') AS h_cross_n,
       max(f.expo) FILTER (WHERE f.grp = 'cross') AS h_cross_expo
FROM (
    SELECT h.line, h.row_id, h.grp, coalesce(c.n, 0) AS n, coalesce(c.n1y, 0) AS n1y, coalesce(c.nlarge, 0) AS nlarge,
           coalesce(x.expo, 0) AS expo
    FROM (SELECT DISTINCT line, row_id, grp FROM hist_frame) h
    LEFT JOIN hist_claim_feats c USING (line, row_id, grp)
    LEFT JOIN hist_expo_feats x USING (line, row_id, grp)
) f
GROUP BY ALL;

-- Evidence: claim frequency by own-line prior claim count, on training rows with some prior exposure, and how
-- much history the 2025 rows carry compared with training.
CREATE OR REPLACE TABLE ev_12_history_frequency AS
SELECT h.line, least(h.h_own_n, 3) AS prior_own_claims, count(*) AS n_rows, round(sum(r.exposure), 1) AS exposure,
       round(1000 * sum(coalesce(t.n_claims, 0)) / sum(r.exposure), 2) AS freq_per_1000
FROM claim_history h
JOIN hist_rows r USING (line, row_id)
LEFT JOIN line_target t USING (line, row_id)
WHERE r.src = 'train' AND h.h_own_expo > 0
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_12_history_coverage AS
SELECT h.line, r.src, count(*) AS n_rows,
       round(avg(h.h_own_expo), 3) AS mean_prior_own_exposure,
       round(avg((h.h_own_expo > 0)::INT), 4) AS share_with_history,
       round(avg((h.h_own_n > 0)::INT), 4) AS share_with_prior_claim
FROM claim_history h JOIN hist_rows r USING (line, row_id)
GROUP BY ALL;
