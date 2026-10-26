-- 13 Expected claims over each row's history (audit A16; decision D-14). Run by python/run_glm_lines.py once per
-- fitting window, after it has registered freq_pred(line, row_id, mu): the expected claim count of every row from
-- a frequency model without history. Sums mu over the same prior rows, matched the same way, as the exposure in
-- sql/12, so e_own sits beside h_own_n for the credibility factor (k + n) / (k + E).

CREATE OR REPLACE TABLE history_expected AS
WITH m AS (
    SELECT f.line, f.row_id, f.grp, q.mu
    FROM hist_frame f
    JOIN hist_rows p ON p.line = f.src_line AND p.customer_id = f.customer_id
    JOIN freq_pred q ON q.line = p.line AND q.row_id = p.row_id
    WHERE p.src = 'train' AND (NOT f.match_address OR p.address_id = f.address_id) AND p.period_end < f.period_start
),
s AS (SELECT line, row_id, grp, sum(mu) AS e FROM m GROUP BY ALL)
SELECT h.line, h.row_id,
       coalesce(max(s.e) FILTER (WHERE h.grp = 'own'), 0) AS e_own,
       coalesce(max(s.e) FILTER (WHERE h.grp = 'home'), 0) AS e_home,
       coalesce(max(s.e) FILTER (WHERE h.grp = 'cross'), 0) AS e_cross
FROM (SELECT DISTINCT line, row_id, grp FROM hist_frame) h
LEFT JOIN s USING (line, row_id, grp)
GROUP BY ALL;
