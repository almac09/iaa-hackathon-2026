-- 06 Year-to-year claim persistence per contents policy (audit A10).
-- Policy-years with at least min_exposure_persistence of exposure, so that part-years do not
-- dilute the comparison. Expected claims are measured two ways (D-07): exposure x flood-zone rate,
-- and earned premium x portfolio rate. The second controls for everything in the current premium.

CREATE OR REPLACE TABLE contents_policy_years AS
SELECT policy_id, yr,
       sum(exposure) AS exposure, sum(earned_premium) AS earned_premium,
       sum(n_claims) AS n_claims, sum(claim_cost) AS claim_cost,
       any_value(coalesce(CAST(flood_zone AS VARCHAR), 'MISSING')) AS flood_zone
FROM contents_model
WHERE src = 'train'
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_06_transition_matrix AS
SELECT least(a.n_claims, 2) AS claims_this_year,
       count(*) AS n_pairs,
       round(avg((b.n_claims = 0)::INT), 4) AS p_next_0,
       round(avg((b.n_claims = 1)::INT), 4) AS p_next_1,
       round(avg((b.n_claims >= 2)::INT), 4) AS p_next_2_plus
FROM contents_policy_years a
JOIN contents_policy_years b ON b.policy_id = a.policy_id AND b.yr = a.yr + 1
WHERE a.exposure >= getvariable('min_exposure_persistence') AND b.exposure >= getvariable('min_exposure_persistence')
GROUP BY ALL;

CREATE OR REPLACE TABLE ev_06_next_year_by_state AS
WITH t AS (
    SELECT b.exposure AS e_next, b.earned_premium AS ep_next, b.n_claims AS n_next, b.claim_cost AS c_next, a.flood_zone,
           CASE WHEN a.n_claims = 0 THEN '0 none'
                WHEN a.claim_cost < getvariable('band_1') THEN '1 small'
                WHEN a.claim_cost < getvariable('band_2') THEN '2 mid'
                ELSE '3 large' END AS state
    FROM contents_policy_years a
    JOIN contents_policy_years b ON b.policy_id = a.policy_id AND b.yr = a.yr + 1
    WHERE a.exposure >= getvariable('min_exposure_persistence') AND b.exposure >= getvariable('min_exposure_persistence')
),
z AS (SELECT flood_zone, sum(c_next) / sum(e_next) AS rate FROM t GROUP BY flood_zone)
SELECT state,
       count(*) AS n_pairs,
       round(1000 * sum(n_next) / sum(e_next), 2) AS freq_next_per_1000,
       round(sum(c_next) / sum(e_next), 2) AS loss_cost_next,
       round(sum(c_next) / sum(e_next * z.rate), 3) AS ae_cost_within_flood_zone,
       round(sum(c_next) / (sum(ep_next) * (SELECT sum(c_next) / sum(ep_next) FROM t)), 3) AS ae_cost_vs_earned_premium
FROM t JOIN z USING (flood_zone)
GROUP BY state;

CREATE OR REPLACE TABLE ev_06_two_year_history AS
WITH t AS (
    SELECT a.n_claims > 0 AS claim_two_years_ago, b.n_claims > 0 AS claim_last_year,
           c.n_claims AS n_next, c.claim_cost AS c_next, c.exposure AS e_next, c.earned_premium AS ep_next, c.flood_zone
    FROM contents_policy_years a
    JOIN contents_policy_years b ON b.policy_id = a.policy_id AND b.yr = a.yr + 1
    JOIN contents_policy_years c ON c.policy_id = a.policy_id AND c.yr = a.yr + 2
    WHERE a.exposure >= getvariable('min_exposure_persistence')
      AND b.exposure >= getvariable('min_exposure_persistence')
      AND c.exposure >= getvariable('min_exposure_persistence')
),
z AS (SELECT flood_zone, sum(n_next) / sum(e_next) AS rate FROM t GROUP BY flood_zone)
SELECT claim_two_years_ago, claim_last_year,
       count(*) AS n_policies,
       round(1000 * sum(n_next) / sum(e_next), 2) AS freq_next_per_1000,
       round(sum(n_next) / sum(e_next * z.rate), 3) AS ae_freq_within_flood_zone,
       round(sum(n_next) / (sum(ep_next) * (SELECT sum(n_next) / sum(ep_next) FROM t)), 3) AS ae_freq_vs_earned_premium,
       round(sum(c_next) / (sum(ep_next) * (SELECT sum(c_next) / sum(ep_next) FROM t)), 3) AS ae_cost_vs_earned_premium
FROM t JOIN z USING (flood_zone)
GROUP BY ALL;
