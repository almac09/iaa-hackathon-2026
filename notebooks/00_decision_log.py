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