# Home contents analysis: replication pack

IAA AI-Assisted Hackathon 2026, Part 1. This pack reproduces every step of the home contents analysis from raw competition data, in SQL, Python and R, and leaves a full audit trail.

- **Owner:** Gerard Murphy (contents line)
- **AI assistance:** Claude Code (Claude Opus 5.5)
- **Status:** analysis and model comparison. No submission model yet. Decisions D-01 to D-06 are **not yet signed off** (see `audit/DECISIONS.md`).

## What it does

| Step | Audit entry | Where |
|---|---|---|
| Profile all tables, check keys | A03 | `sql/01_profile.sql` |
| Data-quality checks (exposure units, sum insured, ID padding, claim matching, missing values) | A03–A05 | `sql/02_quality_checks.sql` |
| Build the contents table: fix exposure (D-01) and sum insured (D-02), build the target (D-03), join customer, address and weather data | A04–A06 | `sql/03_build_contents.sql` |
| Report lags, 2024 IBNR estimate, open vs closed severity, experience and loss ratios on earned premium (all lines) | A06, A12 | `sql/04_development.sql` |
| Claim-size distribution by peril, concentration, size-band mix | A09 | `sql/05_severity.sql` |
| Year-to-year claim persistence (the Markov chain question), against exposure-based and earned-premium-based expected claims | A10, A12 | `sql/06_persistence.sql` |
| Leakage check on customer service contacts (D-04) | A08 | `sql/07_leakage.sql` |
| Rating-factor banding shared by Python and R (D-06) | A07, A09 | `sql/08_features.sql` |
| One-way analysis of 56 factors, scored out of time | A07 | `python/run_all.py`, `R/run_all.R` |
| Single vs per-peril vs per-size-band vs frequency × severity GLMs | A09 | `python/run_all.py`, `R/run_all.R` |
| Python vs R reconciliation, and reproduction of the exploratory numbers | A11 | `reconcile/reconcile.py` |
| XGBoost (Tweedie, exposure offset) on banded and raw features, against GLM A with a paired bootstrap. Python only | A13 | `python/run_xgb.py` |
| All three lines: buildings and motor data layer, development to ultimate (IBNR, motor bodily injury IBNER), large-claim cap, XGBoost per line with a backfitted year trend, 2024 backtest, combined 2025 submission | A15 | `sql/09`–`11`, `python/run_lines.py` |
| Split models: contents GLM, buildings XGBoost (depth 3) and motor GLM for small claims, large claims by GLM with a generalised Pareto tail, claim history as a credibility factor, buildings–contents correlation check, combined 2025 submission | A16 | `sql/12`–`13`, `python/run_glm_lines.py` |

**How the three languages fit together.** The SQL is the single source of truth for data preparation. Python and R both run the same SQL files through DuckDB, so their cleaned data is identical by construction (the reconciliation checks this byte for byte). The modelling is written twice, independently: Python uses scikit-learn's Newton solver and R uses its own penalised IRLS written in base R. Both use the same objective, so the results should agree to within the tolerances set in `config.json`.

**Parameters and decisions** are all in `config.json`. Each decision has an `apply` switch, and the SQL reads the switches through `getvariable()`. To see a decision's effect, turn it off and re-run.

## How to run

You need Python 3.11, R 4.5, the Databricks CLI (logged in) and about 8 GB of free memory. Run every command from this folder (`workings/contents/replication` in the team repo).

First, create and activate a virtual environment (once per machine).

macOS / Linux (zsh). On macOS the command is `python3`, because plain `python` does not exist until the environment is active:
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
Windows PowerShell:
```bash
py -3.11 -m venv .venv
```
```bash
.venv\Scripts\Activate.ps1
```
With the environment active, `python` works on both systems, so the remaining commands are the same everywhere. Reactivate the environment in each new terminal.
```bash
pip install -r requirements.txt
```
```bash
Rscript R/packages.R
```
```bash
python python/fetch_data.py
```
```bash
python python/run_all.py
```
```bash
Rscript R/run_all.R
```
```bash
python reconcile/reconcile.py
```
```bash
python python/run_xgb.py
```
```bash
python python/run_lines.py
```
```bash
python python/run_glm_lines.py
```
```bash
python -m pytest -q tests
```
```bash
Rscript -e "testthat::test_dir('tests/testthat')"
```

`fetch_data.py` exports the parquet files from the team's Databricks Git folder into `data/` and stops if any file's SHA-256 differs from `audit/expected_inputs.sha256`. Use `--from-dir <folder>` to copy from an existing local copy instead. The runners check the hashes again before doing anything.

**Expected result:** `reconcile.py` ends with `0 failed`. The Python run takes about 1–2 minutes. The R run takes a few minutes. `run_xgb.py` is independent of the reconciliation: it writes to `audit/evidence/xgboost/` and has no R twin, because an R xgboost run would not agree closely enough to reconcile. It is checked against GLM A on the same split instead. `run_lines.py` covers all three lines despite the folder's name. It runs `sql/09`–`11` after the shared SQL, writes to `audit/evidence/lines/`, and writes the combined submission to `work/submissions/`. It needs `work/sample_submission.csv` (from the team repo's `data/`).

## The audit trail

| File | What it is | Written by |
|---|---|---|
| `audit/AUDIT_LOG.md` | A narrative entry for every step (A01 onwards): method, before and after, verification, status | Claude, reviewed by the owner |
| `audit/DECISIONS.md` | Each judgement call: AI suggestion, human decision, rationale, sign-off | Owner signs |
| `audit/run_log.jsonl` | One JSON record per run: run ID, UTC times, Git commit, config hash, decision switches and sign-off state, input and SQL hashes, output hashes, package versions, headline metrics | Generated automatically |
| `audit/expected_inputs.sha256` | Pinned hashes of the competition files | Generated once (A03) |
| `audit/evidence/python/`, `audit/evidence/r/` | Every evidence table from each runner (`ev_*` aggregate, `evr_*` row-level) | Generated |
| `audit/evidence/reconciliation_report.md` | Every Python-vs-R and exploratory-vs-pack check, with pass or fail | Generated |
| `audit/exploratory/` | The original exploratory scripts, outputs, hashes and the failed first GLM run, kept as found | Archived |

The run log deliberately records no host names, user names or absolute paths.

## Data handling

The competition data is synthetic, but the rules (section 5b) forbid redistributing it. `data/`, `work/` and all row-level evidence (`evr_*`) are in `.gitignore`. Before pushing this pack to any remote, check that the remote is private.

## Corrections

- **Loss ratios (A12):** `written_premium` is an annual premium, so earned premium = written premium × exposure. Loss ratios quoted before A12 (on written premium) were understated. The pack reports both, with the old basis marked `_superseded`.

## Known differences from the exploratory session

The pack reproduces the exploratory numbers within ±0.002 D² (checked by `reconcile.py`). The remaining differences are deliberate:

- **Persistence (A10):** the exploratory query joined flood zones on `USING (fz)`, which silently dropped policies with a missing flood zone. The pack keeps them as `MISSING`. Two-year A/E moves from 1.67/1.50/2.16 to 1.669/1.517/2.098.
- **GLM solver tolerance** is tightened from 1e-4 to 1e-8 so that Python and R converge to the same optimum. Model D moves from 0.0899 to 0.0904.
- **Band labels** are now defined once in SQL. The partition of rows is the same; only the labels and reference levels differ.
- **The exposure changed-rows list** now includes rows on a 360-day basis as well as rows recorded in days.

## Limitations

- The 2024 target is not developed to ultimate. The ~3% IBNR uplift is estimated but not applied (D-03).
- The persistence A/E controls for flood zone only. Experience rating has not yet been measured against the full GLM.
- The split-test D² differences between models A, B and D have not been tested for significance.
- Databricks serverless notebooks may not support R (this workspace showed no classic clusters). The R runner has been tested locally only.
