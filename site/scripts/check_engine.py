"""Check the browser engine (site/glm.js) against an independent Python scorer.

AI-assisted (Databricks Genie Code), issue #182.

The Python scorer reads Gerard's terms.csv directly and looks splines up in
the `spline value` grid. The JS engine uses model.json and the cubic pieces.
So the two share no code and no intermediate file. The checks are:

1. every test case in site/data/test_cases.json, plus 2,000 random policies
   per line drawn on the model grid: final, as-is and each component cost agree
   to a relative 1e-9;
2. every spline grid point: JS cubic piece = grid contribution to 1e-9;
3. a hand-worked case: the contents policy with every factor at its base level.

Usage:  python site/scripts/check_engine.py [--model site/data/model.json]
Needs Node 18+ on PATH. Exits 1 on any failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "contributors/gerard/workings/contents/replication/export/excel"
TOL = 1e-9


def load_terms():
    with (SRC / "terms.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    T = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["line"] in ("contents", "motor", "buildings"):  # buildings: large-claim GLM only
            T[(r["line"], r["component"])][r["term"]].append(r)
    return T


def py_component(trs_by_term, p):
    eta = math.log(p.get("exposure", 1))
    for term, trs in trs_by_term.items():
        kinds = {r["kind"] for r in trs}
        if "intercept" in kinds or "scale" in kinds:
            eta += float(trs[0]["contribution"])
        elif "trend" in kinds:
            eta += float(next(r for r in trs if r["level"].startswith("log contribution"))["contribution"])
        elif "offset" in kinds:
            pass
        elif kinds == {"categorical"}:
            lv = {r["level"]: float(r["contribution"]) for r in trs}
            eta += lv.get(str(p[term]), 0.0)
        elif kinds == {"band"} and p.get(term) is None:
            eta += next((float(r["contribution"]) for r in trs if r["level"] == "MISSING"), 0.0)
        elif kinds == {"band"}:
            x = float(p[term])
            eta += next(float(r["contribution"]) for r in trs if r["level"] != "MISSING"
                        and float(r["band_lower"]) < x <= float(r["band_upper"]))
        elif "spline value" in kinds and p.get(term) is None:
            eta += float(next(r for r in trs if r["kind"] == "spline missing")["contribution"])
        elif "spline value" in kinds:
            grid = {round(float(r["value"]), 6): float(r["contribution"]) for r in trs if r["kind"] == "spline value"}
            eta += grid[round(float(p[term]), 6)]
        elif term == "x_clay_dry":  # sql/11: coalesce((soil_type = 'clay') x dry_summer_index, 0)
            d = p.get("dry_summer_index")
            eta += float(trs[0]["contribution"]) * (float(d) if p.get("soil_type") == "clay" and d is not None else 0.0)
        elif term == "x_fz3_rain":  # sql/11: coalesce((coalesce(flood_zone, 99) >= 3) x rainfall_mm / 1000, 0)
            fz, rain = p.get("flood_zone"), p.get("rainfall_mm")
            high = fz is None or float(fz) >= 3
            eta += float(trs[0]["contribution"]) * (float(rain) / 1000 if high and rain is not None else 0.0)
        elif term == "log_si":
            si = float(p["sum_insured"])
            si = si / 1000 if si >= 1e6 else si
            eta += float(trs[0]["contribution"]) * math.log(max(si, 1))
        else:
            raise ValueError(term)
    return math.exp(eta)


def py_policy(T, scalars, line, p):
    comps = {c: py_component(T[(line, c)], p) for (l, c) in T if l == line}
    as_is = sum(comps.values())
    seg = f"{line}_{'nb' if p.get('new_business') in (True, 'true') else 'ren'}"
    s1 = scalars[(seg, "step1")]
    final = as_is * s1
    if (seg, "step2") in scalars:
        final = final * scalars[(seg, "step2")] / s1
    return comps, as_is, final


def load_scalars():
    with (SRC / "scalars.csv").open(newline="", encoding="utf-8") as f:
        out = {}
        for r in csv.DictReader(f):
            if r["item"].startswith("level step 1"):
                out[(r["segment"], "step1")] = float(r["value"])
            elif r["item"].startswith("level step 2"):
                out[(r["segment"], "step2")] = float(r["value"])
        return out


def random_policies(model, line, n, rng):
    L = model["lines"][line]
    out = []
    for _ in range(n):
        p = {"exposure": rng.choice([1, 0.5, 0.25, 0.9178082191780822]), "new_business": rng.random() < 0.3}
        for k, inp in L["inputs"].items():
            if inp["type"] == "select":
                p[k] = rng.choice(inp["options"])["value"]
            elif inp.get("allow_missing") and rng.random() < 0.15:
                p[k] = None  # "Unknown" in the calculator
            else:
                steps = int(round((inp["max"] - inp["min"]) / inp["step"]))
                p[k] = round(inp["min"] + rng.randint(0, steps) * inp["step"], 6)
        out.append(p)
    return out


NODE = r"""
import { readFileSync } from "node:fs";
import { scorePolicy, splineValue, officePremium } from "%s";
const model = JSON.parse(readFileSync(process.argv[1], "utf8"));
const jobs = JSON.parse(readFileSync(0, "utf8"));
const scores = jobs.policies.map(([line, p]) => {
  const r = scorePolicy(model, line, p);
  return { final: r.final, as_is: r.as_is, comps: Object.fromEntries(Object.entries(r.components).map(([k, c]) => [k, c.cost])) };
});
const splines = jobs.splines.map(([line, comp, term, x]) => {
  const t = model.lines[line].components[comp].terms.find((t) => t.term === term);
  return splineValue(t, x);
});
const office = (jobs.office || []).map(([line, channel, p]) => {
  const annual = scorePolicy(model, line, { ...p, exposure: 1 }).final;
  return officePremium(annual, line, channel, jobs.rates);
});
process.stdout.write(JSON.stringify({ scores, splines, office }));
"""


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=REPO / "site/data/model.json")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ref", type=Path, default=REPO / "site/data/reference_policies.json")
    a = ap.parse_args()
    model = json.loads(a.model.read_text())
    cases = json.loads((REPO / "site/data/test_cases.json").read_text())
    T, scalars, rng = load_terms(), load_scalars(), random.Random(20261029)

    policies = [(line, c["policy"]) for line in ("contents", "motor") for c in cases[line]]
    ref_path = a.ref  # built by build_reference.py
    ref, rates, office_jobs = [], {}, []
    if ref_path.exists():
        refj = json.loads(ref_path.read_text())
        ref, rates = refj["policies"], refj["expenses"]["rates_emerging"]
        policies += [(c["line"], c["policy"]) for c in ref if c["line"] in model["lines"]]
        office_jobs = [(c["line"], c["channel"], c["policy"]) for c in ref if c["priced"]]
        print(f"reference policies: {sum(c['priced'] for c in ref)} priced of {len(ref)}")
    n_cases = len(policies)
    for line in ("contents", "motor", "buildings"):
        if line in model["lines"]:
            policies += [(line, p) for p in random_policies(model, line, a.n, rng)]
    splines, expect = [], []
    for (line, comp), by_term in T.items():
        for term, trs in by_term.items():
            for r in trs:
                if r["kind"] == "spline value":
                    splines.append((line, comp, term, float(r["value"])))
                    expect.append(float(r["contribution"]))

    js = NODE % (REPO / "site/glm.js").as_uri()
    res = subprocess.run(["node", "--input-type=module", "-e", js, str(a.model)],
                         input=json.dumps({"policies": policies, "splines": splines, "office": office_jobs, "rates": rates}),
                         capture_output=True, text=True)
    if res.returncode:
        print(res.stderr)
        sys.exit(1)
    out = json.loads(res.stdout)

    worst, fails = 0.0, 0
    for i, ((line, p), s) in enumerate(zip(policies, out["scores"])):
        comps, as_is, final = py_policy(T, scalars, line, p)
        diffs = [rel(s["final"], final), rel(s["as_is"], as_is)] + [rel(s["comps"][k], v) for k, v in comps.items()]
        worst = max(worst, *diffs)
        if max(diffs) > TOL:
            fails += 1
            print("FAIL", line, p, s, final)
        if i < n_cases:
            print(f"case {line:8s} {i:2d}  JS {s['final']:12.6f}  Python {final:12.6f}  as-is {as_is:10.4f}")
    sw = max(abs(x - y) for x, y in zip(out["splines"], expect))
    print(f"policies checked: {len(policies)}, worst relative difference {worst:.2e}, failures {fails}")
    print(f"spline grid points checked: {len(expect)}, worst absolute difference (log) {sw:.2e}")

    # hand-worked: contents, every select at base, numbers as given
    base = {"exposure": 1, "new_business": False}
    for k, inp in model["lines"]["contents"]["inputs"].items():
        base[k] = inp.get("default")
    base.update(elevation_m=50.0, distance_to_river_km=10.0, sum_insured=30000)
    by = T[("contents", "small")]
    g = lambda t, v: next(float(r["contribution"]) for r in by[t] if r["kind"] == "spline value" and abs(float(r["value"]) - v) < 1e-9)
    hand = math.exp(-3.2610405653146253 + 0.10357138890542077 + g("elevation_m", 50.0) + g("distance_to_river_km", 10.0)
                    + float(by["log_si"][0]["contribution"]) * math.log(30000)) * 0.8657493863699818
    _, _, py = py_policy(T, scalars, "contents", base)
    print(f"hand-worked base contents: hand {hand:.6f}  Python {py:.6f}")

    # office premium: JS officePremium (on the JS annual risk) against Gerard's python/office_premium.build_up
    # (on the Python risk for the period, which it annualises itself)
    office_fail = 0
    if office_jobs:
        sys.path.insert(0, str(REPO / "contributors/gerard/workings/contents/replication/python"))
        import numpy as np
        import office_premium as op
        risk = [py_policy(T, scalars, l, p)[2] for l, _, p in office_jobs]
        g = op.build_up(np.array(risk), np.array([p["exposure"] for _, _, p in office_jobs]),
                        np.array([l for l, _, _ in office_jobs]), np.array([c for _, c, _ in office_jobs]), rates)
        for (l, c, p), js_o, (_, row) in zip(office_jobs, out["office"], g.iterrows()):
            d = max(rel(js_o["office_premium"], row.office_premium), rel(js_o["risk_premium"], row.risk_premium_annual))
            office_fail += d > TOL
            print(f"office {l:8s} {c:15s} risk {row.risk_premium_annual:10.4f}  JS office {js_o['office_premium']:10.4f}  "
                  f"Gerard {row.office_premium:10.4f}")
        print(f"office premiums checked: {len(office_jobs)}, failures {office_fail}")
    ok = fails == 0 and sw < TOL and rel(py, hand) < TOL and office_fail == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
