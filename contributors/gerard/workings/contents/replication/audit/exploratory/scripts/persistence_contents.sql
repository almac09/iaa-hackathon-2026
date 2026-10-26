-- Year-to-year claim persistence per contents policy (A10). Policy-years with exposure >= 0.5 only.
CREATE TABLE py AS SELECT policy_id, yr, sum(exposure) e, sum(n_claims) n, sum(claim_cost) c, any_value(flood_zone) fz
FROM 'work/contents.parquet' WHERE src='train' GROUP BY ALL;
CREATE TABLE t3 AS SELECT a.n>0 claim_tm1, b.n>0 claim_t, c.n n2, c.e e2, c.fz FROM py a
  JOIN py b ON b.policy_id=a.policy_id AND b.yr=a.yr+1 JOIN py c ON c.policy_id=a.policy_id AND c.yr=a.yr+2
  WHERE a.e>=0.5 AND b.e>=0.5 AND c.e>=0.5;
COPY (SELECT least(p.n,2) from_claims, round(avg((q.n=0)::int),4) to_0, round(avg((q.n=1)::int),4) to_1,
        round(avg((q.n>=2)::int),4) to_2plus, count(*) pairs
      FROM py p JOIN py q ON q.policy_id=p.policy_id AND q.yr=p.yr+1 WHERE p.e>=0.5 AND q.e>=0.5 GROUP BY 1 ORDER BY 1)
  TO 'audit/evidence/A10_transition_matrix.csv' (HEADER);
COPY (WITH z AS (SELECT fz, sum(n2)/sum(e2) r FROM t3 GROUP BY fz)
      SELECT claim_tm1, claim_t, count(*) n_policies, round(1000*sum(n2)/sum(e2),1) freq_next_per_1000,
             round(sum(n2)/sum(e2*r),2) ae_within_flood_zone
      FROM t3 JOIN z USING (fz) GROUP BY ALL ORDER BY 2,1)
  TO 'audit/evidence/A10_two_year_history.csv' (HEADER);
