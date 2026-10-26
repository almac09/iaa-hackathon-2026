"""Scratch (A31): turn the exploration, confirmation and review outputs into aggregate evidence files in the pack.

    python build_evidence.py <pack>/audit/evidence/emerging
Every text field is scrubbed of local paths. No row-level data is written."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
PATHS = [(re.compile(r"/private/tmp/claude-\d+/[^/\s]+/[0-9a-f-]+/scratchpad/a31"), "<a31 scratch>"),
         (re.compile(r"<home>/\s]+/iaa-hackathon-2026/workings/contents/replication"), "<pack>"),
         (re.compile(r"<home>/\s]+"), "<home>")]


def scrub(v):
    if isinstance(v, str):
        for pat, rep in PATHS:
            v = pat.sub(rep, v)
    return v


def frame(rows):
    return pd.DataFrame(rows).map(scrub)


def load_json(path):
    t = path.read_text()
    return json.loads(t[t.index("[") if t.lstrip()[0] != "{" else t.index("{"):])


wf = json.loads((HERE / "workflow_result.json").read_text())
grouping = json.loads(Path(sys.argv[2]).read_text())["result"]["result"] if len(sys.argv) > 2 else None
b1 = {r["key"]: r for r in load_json(HERE / "confirm_round1.json")}
r2 = load_json(HERE / "confirm_r2.json")
bt = json.loads((HERE / "r2_backtest.json").read_text())

rows = []
explored = [v["result"] for v in wf["variations"]] + ([grouping] if grouping else [])
for r in explored:
    b = b1[r["key"]]
    rows.append({"round": 1, "key": r["key"], "configurations_scored_on_A": r["configurations_scored_on_A2024"],
                 "final_configuration": r["final_configuration"], "A_gain": r["gain_vs_c0"],
                 "A_ci_low_95": r["gain_ci_low_95"], "A_ci_high_95": r["gain_ci_high_95"],
                 "A_per_line": r["per_line"], "B_gain": b["gain"], "B_p_one_sided": b["p_one_sided"],
                 "B_holm_threshold": b["holm_threshold"], "B_passes_holm": b["passes_holm"],
                 "B_per_line_gain": "; ".join(f"{k} {v['d2_lm'] - v['c0_d2_lm']:+.5f}" for k, v in b["per_line"].items()),
                 "summary": r["summary"]})
for key in ("R1 multiple imputation", "R2 vehicle age minus 2", "R3 calibration power"):
    a = next(x for x in r2 if x["half"] == "A2024" and x["key"] == key)
    b = next(x for x in r2 if x["half"] == "B2024" and x["key"] == key)
    rows.append({"round": 2, "key": key, "configurations_scored_on_A": 1, "final_configuration": "preregistered",
                 "A_gain": a["gain"], "A_ci_low_95": a["ci_low_95"], "A_ci_high_95": a["ci_high_95"],
                 "A_per_line": "; ".join(f"{k} {v['d2_lm'] - v['c0_d2_lm']:+.5f}" for k, v in a["per_line"].items()),
                 "B_gain": b["gain"], "B_p_one_sided": b["p_one_sided"], "B_holm_threshold": b["holm_threshold"],
                 "B_passes_holm": b["passes_holm"],
                 "B_per_line_gain": "; ".join(f"{k} {v['d2_lm'] - v['c0_d2_lm']:+.5f}" for k, v in b["per_line"].items()),
                 "summary": ""})
rows.append({"round": 2, "key": "R4 customer_since_year minus 1 (developed buildings backtest)",
             "configurations_scored_on_A": 0, "final_configuration": "preregistered", "A_gain": None,
             "B_gain": bt["gain"], "B_p_one_sided": bt["p_one_sided"], "B_passes_holm": False,
             "summary": f"Inert by construction: the buildings trees treat every customer_since_year from 2020 on alike, so no "
                        f"new-business prediction changes. Backtest D2 {bt['d2_raw']:.5f} against {bt['d2_csy_minus_1']:.5f}, "
                        f"interval [{bt['ci_low_95']:+.6f}, {bt['ci_high_95']:+.6f}]."})
frame(rows).to_csv(OUT / "ev_a31_variations.csv", index=False)

data = [{"audit": d["key"][:2] if d["key"].startswith("d") else d["key"].split()[0], **f}
        for d in wf["data"] for f in d["findings"]]
frame(data).to_csv(OUT / "ev_a31_data_suitability.csv", index=False)

review = json.loads((HERE / "review_result.json").read_text())
frame([{"reviewer": r["key"], **d} for r in review for d in r["defects"]]).to_csv(OUT / "ev_a31_review.csv", index=False)
print("written:", sorted(p.name for p in OUT.glob("ev_a31_*")))
