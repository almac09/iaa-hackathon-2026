"""Scratch (A33): what the identifiers say about channel, and the 2025 emerging transfer block.

    python transfer_block.py <pack>/audit/evidence/emerging
Reads the competition data in <pack>/data and writes two aggregate files; no row-level data is written.
  ev_a33_policy_numbers.csv  one line per check of the policy numbers; each test counts each policy once
  ev_a33_transfer_block.csv  new business by market, line and year, with 2025 split into the 2-15 January block and
                             the rest, and the 2025 renewals
Exposure is recomputed from the period dates (D-01)."""
import math
import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, spearmanr

PACK = Path(__file__).resolve().parents[3]
MARKETS = {"developed": PACK / "data", "emerging": PACK / "data" / "emerging"}
LINES = ["buildings", "contents", "motor"]
WINDOW = ("2025-01-02", "2025-01-15")  # every start date in the emerging January block
OUT = Path(sys.argv[1])

con = duckdb.connect()
con.sql("CREATE TABLE r AS " + " UNION ALL ".join(
    f"SELECT '{m}' AS market, '{line}' AS line, '{src}' AS src, CAST(substr(row_id, 2) AS BIGINT) AS row_no, "
    f"policy_id, lpad(customer_id, 8, '0') AS customer_id, address_id, period_start, "
    f"(datediff('day', period_start, period_end) + 1) / 365 AS exposure, new_business, channel "
    f"FROM '{d / f'{src}_{line}.parquet'}'"
    for m, d in MARKETS.items() for line in LINES for src in ("policies", "test")))
# One row per policy: its first segment's start, its channel and customer (neither changes within a policy, checked
# below) and its number.
con.sql("""CREATE TABLE p AS SELECT market, line, policy_id, left(policy_id, 1) AS letter,
           CAST(substr(policy_id, 2) AS BIGINT) AS num, min(period_start) AS first_start, any_value(channel) AS channel,
           count(DISTINCT channel) AS n_channels, any_value(customer_id) AS customer_id,
           count(DISTINCT customer_id) AS n_customers FROM r GROUP BY ALL""")


def chi2(tab):
    stat, p, _, _ = chi2_contingency(tab)
    return p, math.sqrt(stat / (tab.to_numpy().sum() * (min(tab.shape) - 1)))


rows = []
add = lambda market, check, value, detail="": rows.append(  # noqa: E731
    {"market": market, "check": check, "value": value, "detail": detail})
letters = con.sql("SELECT line, list(DISTINCT letter) AS l FROM p GROUP BY line ORDER BY line").fetchall()
add("both", "letter by line", "; ".join(f"{line} {'/'.join(x)}" for line, x in letters), "the only letter on each line")
for m in MARKETS:
    pol = con.sql(f"SELECT * FROM p WHERE market = '{m}'").df()
    add(m, "policies", len(pol))
    add(m, "policies whose channel changes", int((pol.n_channels > 1).sum()))
    add(m, "policies whose customer changes", int((pol.n_customers > 1).sum()))
    p, v = chi2(pd.crosstab(pol.letter, pol.channel))
    add(m, "letter x channel", f"p {p:.3f}, V {v:.4f}", "chi-square, one row per policy")
    for ch, g in pol.groupby("channel"):
        add(m, f"number quartiles, {ch} (millions)", " / ".join(f"{g.num.quantile(q) / 1e6:.1f}" for q in (.25, .5, .75)))
    digits = pol.policy_id.str[1:]
    tests = [chi2(pd.crosstab(digits.str[i], pol.channel)) for i in range(8)]
    add(m, "digit position x channel, 8 tests", f"min p {min(t[0] for t in tests):.3f}, max V {max(t[1] for t in tests):.4f}",
        "chi-square per digit position, one row per policy; no correction for the 8 tests")
    rho, p = spearmanr(pol.num, pol.first_start.astype("int64"))
    add(m, "number vs first start date", f"Spearman {rho:.4f}, p {p:.2f}")

emg = con.sql("""SELECT num, first_start >= DATE '2025-01-01' AS new_2025 FROM p WHERE market = 'emerging'""").df()
ks = ks_2samp(emg.num[emg.new_2025], emg.num[~emg.new_2025])
add("emerging", "2025-only policies vs earlier, number distribution", f"KS D {ks.statistic:.4f}, p {ks.pvalue:.2f}")
keep = con.sql("""SELECT count(*) FILTER (WHERE policy_id IN (SELECT policy_id FROM r WHERE market = 'emerging'
                   AND src = 'policies')), count(*) FROM r WHERE market = 'emerging' AND src = 'test' AND NOT new_business""").fetchone()
add("emerging", "2025 renewal rows keeping a 2023-24 number", f"{keep[0]} of {keep[1]}")
reuse = con.sql("""SELECT count(*) FILTER (WHERE policy_id IN (SELECT policy_id FROM r q WHERE q.market = t.market
                   AND q.line = t.line AND q.src = 'policies')), count(*) FROM r t WHERE src = 'test'
                   AND new_business""").fetchone()
add("both", "2025 new-business rows reusing a 2021-24 number", f"{reuse[0]} of {reuse[1]}")
flag = con.sql("""SELECT count(*) FROM r JOIN p USING (market, line, policy_id) WHERE r.src = 'test'
                  AND r.new_business <> (p.first_start >= DATE '2025-01-01')""").fetchone()[0]
add("both", "2025 rows whose new-business flag disagrees with a 2025 first start", flag,
    "so a policy first seen in 2025 is new business, and every earlier policy is a renewal")
# Shared numbers across market x line groups, against independent uniform draws from 0-99,999,999.
sizes = [n / 1e8 for (n,) in con.sql("SELECT count(*) FROM p GROUP BY market, line").fetchall()]
none = math.prod(1 - s for s in sizes)
one = sum(s * none / (1 - s) for s in sizes)
expected = 1e8 * (1 - none - one)
shared = con.sql("SELECT count(*) FROM (SELECT num FROM p GROUP BY num HAVING count(*) > 1)").fetchone()[0]
linked = con.sql("""WITH s AS (SELECT DISTINCT market, line, policy_id, num, r.customer_id, address_id FROM r
                      JOIN p USING (market, line, policy_id) WHERE num IN (SELECT num FROM p GROUP BY num HAVING count(*) > 1))
                    SELECT count(DISTINCT (a.num, a.line, b.line)) FROM s a JOIN s b ON a.num = b.num AND a.market = b.market
                    AND a.line < b.line AND (a.customer_id = b.customer_id OR a.address_id = b.address_id)""").fetchone()[0]
add("both", "numbers shared across market x line", f"{shared} observed, {expected:.0f} expected",
    f"expected under independent uniform draws; same-market pairs sharing a customer or an address: {linked}")
both = con.sql("""SELECT count(*) FROM p a JOIN p b ON a.policy_id = b.policy_id AND a.line = b.line
                  AND a.market = 'developed' AND b.market = 'emerging'""").fetchone()[0]
both_expected = sum(d * e / 1e8 for d, e in con.sql("""SELECT count(*) FILTER (WHERE market = 'developed'),
                    count(*) FILTER (WHERE market = 'emerging') FROM p GROUP BY line""").fetchall())
add("both", "policy_ids present in both markets (same line)", f"{both} observed, {both_expected:.0f} expected",
    "so a join across markets needs the market in the key")

pd.DataFrame(rows).to_csv(OUT / "ev_a33_policy_numbers.csv", index=False)

# New business by year (2023 and 2024 from the training files) and the 2025 book: renewals, the 2-15 January block
# and the other new business. A segment's year is its policies' first-start year (2025 for the test file).
con.sql(f"""CREATE TABLE seg AS SELECT r.*, p.first_start, CASE
              WHEN r.src = 'policies' THEN 'new business ' || year(p.first_start)
              WHEN p.first_start < DATE '2025-01-01' THEN '2025 renewals'
              WHEN p.first_start BETWEEN DATE '{WINDOW[0]}' AND DATE '{WINDOW[1]}' THEN '2025 new business, 2-15 January'
              ELSE '2025 new business, other dates' END AS segment,
            CASE WHEN r.src = 'policies' THEN year(p.first_start) ELSE 2025 END AS seg_year
            FROM r JOIN p USING (market, line, policy_id)
            WHERE r.src = 'test' OR (r.new_business AND year(p.first_start) IN (2023, 2024))""")
con.sql("CREATE TABLE cu AS " + " UNION ALL ".join(
    f"SELECT '{m}' AS market, lpad(customer_id, 8, '0') AS customer_id, customer_since_year "
    f"FROM '{d / 'customers.parquet'}'" for m, d in MARKETS.items()))
# The block also sits as one contiguous run in each emerging test file's row order (a generation artefact, not a
# feature): the longest run of consecutive rows in the window segment.
con.sql("""CREATE TABLE run AS SELECT market, line, max(n) AS longest_row_run FROM (
             SELECT market, line, grp, count(*) AS n FROM (
               SELECT market, line, segment, row_no - row_number() OVER (PARTITION BY market, line, segment
                      ORDER BY row_no) AS grp FROM seg WHERE src = 'test') WHERE segment LIKE '%2-15 January'
             GROUP BY ALL) GROUP BY ALL""")
block = con.sql("""WITH sp AS (SELECT market, line, segment, seg_year, policy_id, any_value(channel) AS channel,
                                 any_value(first_start) AS first_start, any_value(customer_id) AS customer_id
                               FROM seg GROUP BY ALL),
                     cf AS (SELECT market, customer_id, min(year(first_start)) AS first_year FROM p GROUP BY ALL),
                     pol AS (SELECT sp.market, sp.line, sp.segment, count(*) AS policies,
                               round(avg((channel = 'broker')::INT), 3) AS broker,
                               round(avg((channel = 'comparison_site')::INT), 3) AS comparison_site,
                               round(avg((channel = 'direct')::INT), 3) AS direct,
                               min(first_start) AS earliest_first_start, max(first_start) AS latest_first_start,
                               median(dayofyear(first_start)) AS median_start_day,
                               count(DISTINCT sp.customer_id) AS customers,
                               round(count(DISTINCT sp.customer_id) FILTER (WHERE cu.customer_since_year = sp.seg_year)
                                     / count(DISTINCT sp.customer_id), 3) AS customers_since_segment_year,
                               count(DISTINCT sp.customer_id) FILTER (WHERE cf.first_year < sp.seg_year)
                                 AS customers_with_earlier_policy
                             FROM sp LEFT JOIN cu USING (market, customer_id) LEFT JOIN cf USING (market, customer_id)
                             GROUP BY ALL),
                     rw AS (SELECT market, line, segment, count(*) AS rows, round(sum(exposure), 1) AS exposure
                            FROM seg GROUP BY ALL)
                   SELECT pol.market, pol.line, pol.segment, policies, rows, exposure, broker, comparison_site, direct,
                     earliest_first_start, latest_first_start, median_start_day, customers,
                     customers_since_segment_year, customers_with_earlier_policy,
                     CASE WHEN pol.segment LIKE '%2-15 January' THEN longest_row_run END AS longest_row_run
                   FROM pol JOIN rw USING (market, line, segment) LEFT JOIN run USING (market, line)
                   ORDER BY ALL""").df()
block.to_csv(OUT / "ev_a33_transfer_block.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(block.to_string(index=False))
