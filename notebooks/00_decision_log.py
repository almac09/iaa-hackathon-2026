# Databricks notebook source
# MAGIC %md
# MAGIC # 📋 AI-Assisted Decision Log — IAA Hackathon 2026
# MAGIC
# MAGIC This notebook documents every key decision made during the hackathon,
# MAGIC showing how AI (Databricks Genie) assisted the work while the human
# MAGIC (Alan McDonagh) guided strategy, architecture, and final choices.
# MAGIC
# MAGIC Each entry records: what AI suggested → what the human decided → why.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Decision 1: Project Structure
# MAGIC - **Date**: 2026-10-19
# MAGIC - **AI suggested**: Git Folders with a 5-notebook pipeline (data loading → EDA → feature engineering → model training → submission)
# MAGIC - **Human decided**: Adopted the structure with addition of this decision log and a shared `src/utils.py` module
# MAGIC - **Rationale**: Clean separation of concerns; decision log proves AI-human collaboration
# MAGIC
# MAGIC ## Decision 2: Workflow Tracking
# MAGIC - **Date**: 2026-10-19
# MAGIC - **AI suggested**: Use notebook version history + Git commits for audit trail
# MAGIC - **Human decided**: Extended to include GitHub Issues as task tracker, with AI interaction logs in each issue
# MAGIC - **Rationale**: GitHub Issues provide a public, linkable record that judges can review; shows project management skills
# MAGIC
# MAGIC ## Decision 3: GitHub Connector Setup
# MAGIC - **Date**: 2026-10-19
# MAGIC - **AI suggested**: Use Databricks GitHub MCP connector to automate issue creation and file commits
# MAGIC - **Human decided**: Debugged and installed the Databricks GitHub Connector app to grant write access
# MAGIC - **Rationale**: Demonstrates troubleshooting skills and understanding of GitHub App permissions model
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision 4: Build raw tables from the repo's parquet files
# MAGIC - **Date**: 2026-10-21
# MAGIC - **AI suggested** (Claude): Rebuild `workspace.iaa` from `data/*.parquet` in the repo, copied into the Volume `workspace.iaa.raw`, instead of using manually uploaded copies. A check of the uploads against fingerprints of the files was proposed as the alternative.
# MAGIC - **Human decided**: Rebuild from the repo. The manual uploads were replaced; Delta history keeps the old versions.
# MAGIC - **Rationale**: One reproducible source anyone can rerun. Parquet keeps the column types, so the uploads don't need checking.
# MAGIC - **Evidence**: `notebooks/00_load_raw_from_git`; `01a_row_count_check` 11/11 PASS, logged in `dq_row_count_log`.
# MAGIC
# MAGIC ## Decision 5: Data checks fail the run and leave a log
# MAGIC - **Date**: 2026-10-21
# MAGIC - **AI suggested** (Claude): Replace eleven ad hoc count queries with one check that compares tables with the Kaggle delivery, logs every run, and stops on failure. Add a schema check that compares each table with its source file and requires IDs to be text and dates to be dates.
# MAGIC - **Human decided**: Adopted. A one-table `DESCRIBE` was challenged as not enough for #47, and the check now covers every table.
# MAGIC - **Rationale**: A failure stops the pipeline instead of relying on someone reading the output. The log is dated evidence for the judges.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision 6: Correct the data through an audited rule layer, never by editing raw tables
# MAGIC - **Date**: 2026-10-22
# MAGIC - **Human proposed** (Cian): Keep lists of the values being changed in separate tables, then join them on, taking the corrected value where one exists and the raw value otherwise.
# MAGIC - **AI suggested** (Claude): Build on that with a rule registry (`dq_rules`: rule, evidence, issue, approver) and three generated tables: changed cells (`dq_changes`), removed rows (`dq_exclusions`) and suspect rows kept unchanged (`dq_flags`). Add a reconciliation proving the `_clean` tables differ from raw only where the registry says so.
# MAGIC - **Human decided**: Adopted. Each rule deletes and re-inserts its own rows, so the notebook can be rerun safely.
# MAGIC - **Rationale**: Every changed value traces back to a rule, its evidence, a GitHub issue and an approver. Raw data is never touched, and Delta history keeps earlier versions.
# MAGIC - **Evidence**: `notebooks/01b_corrections`. The reconciliation passes for all ten `_clean` tables: changed cells equal `dq_changes` exactly, rows equal raw minus exclusions, and weather is identical. See also `docs/DATA_CONTRACT.md` and PR #122.
# MAGIC
# MAGIC ## Decision 7: Correction rules R1 to R5 and flag R6
# MAGIC - **Date**: 2026-10-22
# MAGIC - **AI suggested** (Claude): The rules below, each backed by evidence from profiling the raw tables.
# MAGIC - **Human decided**: Each rule approved individually after reviewing queries that show the raw problem (`approved_by` = Cian in `dq_rules`).
# MAGIC
# MAGIC | Rule | Change | Rows | Issue |
# MAGIC |---|---|---:|---|
# MAGIC | R1 | Motor `customer_id` left-padded to 8 characters (0% joined to customers before, 100% after) | 123,664 (98,110 training + 25,554 test) | #51 |
# MAGIC | R2 | `exposure` recomputed as (period_end − period_start + 1) / 365 | 14,164 (motor 5,703, buildings 4,168, contents 4,293) | #52 |
# MAGIC | R3 | `sum_insured` divided by 1,000 where above 100× the training median | 3,527 (buildings 1,395 + 370 test, contents 1,402 + 360 test) | #53 |
# MAGIC | R4 | `construction_type` mapped to brick / stone / timber / non_standard (timber coded T to 2022, W from 2023) | 705,672 (560,124 training + 145,548 test) | #54 |
# MAGIC | R5 | Duplicate claims removed (copies with an 11-character `claim_id`) | 604 (claims 163,150 → 162,546) | #55 |
# MAGIC | R6 | Flagged, not changed: loss date before the policy's first row | 413 (motor 296, buildings 67, contents 50) | #56 |
# MAGIC
# MAGIC - **Rationale**: Correct only what is clearly an error with one right answer. Flag what is suspect but plausible.
# MAGIC - **Consistency with the other pipelines**: Bridget's v3 and Gerard's pack prepare their data separately and apply equivalent rules. On sum insured, Gerard's D-02 (divide by 1,000 where 1m or more) catches the same 3,527 rows plus one buildings test row, B02835094. That row has a sum insured of 1.061m on a new 2025 policy, with a premium above 87% of test rows. D-02 reduces it to 1,061; R3 leaves it unchanged. Gerard also found two address problems that are not in the `_clean` tables: region R11 elevation recorded in feet (D-26) and a year_built = 1900 placeholder on 19,014 addresses (D-27).
# MAGIC
# MAGIC ## Decision 8: Predictions proportional to exposure, with no €0.01 floor for short policies
# MAGIC - **Date**: 2026-10-22
# MAGIC - **AI suggested** (Claude): Drop the proposed rule that set policies under 14 days to €0.01. Short rows claim at close to the normal annual rate (motor 0.117 claims a year against 0.136 for full-year rows). Predicting €0.01 gives a Tweedie deviance 20× worse in motor, 4× in buildings and 10× in contents than annual rate × exposure. Use exposure as an offset, and floor only to avoid exact zeros.
# MAGIC - **Human decided**: Agreed by the team on #92. Bridget confirmed she uses this approach, and v3 (decision 16) and Gerard's pack (D-07) both use a log-exposure offset.
# MAGIC - **Rationale**: A zero prediction scores infinitely badly. Short rows are about 2% of test rows but only about 0.05% of test exposure.
# MAGIC - **Evidence**: #92.
# MAGIC
# MAGIC ## Decision 9: Columns kept out of the models
# MAGIC - **Date**: 2026-10-22
# MAGIC - **AI suggested** (Claude): Exclude `customer_service_contacts` (about 0.4 a year plus about 2.5 per claim in training, while test rows look like no-claim rows), `renewal_decision` (training only) and `compulsory_excess` (constant within each line). Use `written_premium` only as a benchmark, because it is the incumbent's price.
# MAGIC - **Human decided**: Adopted across the team. On #92, Bridget noted that no premium measure is used in any model: a pure risk premium should not copy the previous pricing. The same exclusions are in v3 (R5, R7) and Gerard's pack (D-04, D-05).
# MAGIC - **Rationale**: Contacts and renewal decision leak the outcome, written premium would copy the current rates, and compulsory excess carries no information.
# MAGIC
# MAGIC ## Decision 10: 2025 benchmark and submission check
# MAGIC - **Date**: 2026-10-22
# MAGIC - **AI suggested** (Claude): Benchmark each line's 2025 total cost and check every submission against it. Take cost per policy-year for 2021–2023 as reported at 31 Dec 2024 (2024 is left out as the least developed). Project each year to 2025 at no trend (low), the 2021–23 trend (central) and that trend plus 2 points (high), average the three years and multiply by 2025 test exposure.
# MAGIC - **Human decided**: Adopted, with automatic PASS/FAIL views so nothing has to be checked by eye.
# MAGIC - **Ranges (low – central – high)**: motor €115.1m – €122.5m – €129.9m; buildings €61.1m – €72.9m – €77.1m; contents €16.4m – €19.2m – €20.3m.
# MAGIC - **Evidence**: `notebooks/02a_benchmark_2025` and #91. Every check passes. The sample submission is flagged below range, as expected.
# MAGIC
# MAGIC ## Decision 11: The 11.9m motor bodily-injury claim
# MAGIC - **Date**: recorded 2026-10-25
# MAGIC - **AI suggested** (Claude): Review claim CL30973963 before Part 1 closed (#60). It is an open 2023 bodily-injury claim with €11.89m incurred and €0.74m paid, more than 10× the next largest claim (€1.12m).
# MAGIC - **Human decided**: Not reviewed in Week 1 for lack of time, so it stayed unchanged in `claims_clean` by default. **This was not an active decision.** Separately, Gerard's final candidate capped developed claims at 1m each and spread the excess over motor as a ×1.031 loading (D-12, added during his run and awaiting sign-off).
# MAGIC - **Next**: Decide in Part 2 (#60). It matters for any severity or large-loss assumption carried over from the developed market.
# MAGIC
# MAGIC ## Decision 12: Week 1 final submission
# MAGIC - **Date**: 2026-10-23
# MAGIC - **Human decided**: [CONFIRM: which file was selected as the final Kaggle submission, and its score]
# MAGIC - **Where the reasoning sits**: This log covers the data layer. Modelling and development decisions are in `notebooks_v3/04 Decision Log` (Bridget), `workings/chainladder` (Alan) and `workings/contents/replication/audit/DECISIONS.md` (Gerard, D-01 to D-30, on `contents/gerard-final-review`).