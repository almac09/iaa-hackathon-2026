# 📋 Project Management Guide — IAA Hackathon 2026

Welcome to the team! This document explains how we manage our hackathon project using **GitHub Issues**, a **Kanban board**, and an **AI-assisted decision log**. Read this before your first contribution.

---

## 🎯 Our Approach: AI-Assisted, Human-Led

This project uses **Databricks Genie** (an AI assistant) to accelerate development, while the **team makes all strategic decisions**. Every AI interaction is logged so we can demonstrate the collaboration model to hackathon judges.

**The AI does**: code generation, boilerplate setup, suggesting approaches, Databricks platform guidance.
**The team does**: architecture decisions, domain knowledge, feature selection, model evaluation, project management.

---

## 🗂️ How We Use GitHub for Project Management

### 1. Issues = Tasks

Every piece of work is tracked as a **GitHub Issue**. Each issue follows a standard format:

```
🤖 AI Interaction Log        ← What AI suggested vs what we decided
📋 Tasks                     ← Checklist of subtasks
📝 Notes                     ← Links to relevant notebooks/code
```

**Issue labels** tell you what type of work it is:

| Label | Meaning |
|---|---|
| `ai-assisted` | AI helped draft or suggest the approach |
| `setup` | Project infrastructure and configuration |
| `data` | Data loading, ingestion, storage |
| `analysis` | EDA, data exploration, visualisation |
| `feature-engineering` | Feature creation and preprocessing |
| `modelling` | Model training, evaluation, tuning |
| `submission` | Kaggle submission generation |
| `process` | Workflow, documentation, project management |

### 2. Project Board = Kanban

We use a **GitHub Project Board** to visualise progress. Find it under the **Projects** tab on the repo.

**Columns:**

| Column | What goes here |
|---|---|
| **To Do** | Work that hasn't started yet |
| **In Progress** | Someone is actively working on this — assign yourself! |
| **In Review** | Work is done but needs a teammate to check it |
| **Done** | Completed and merged/closed |

**How to use the board:**

1. **Pick a task**: Go to the board, find an issue in "To Do" that you want to work on
2. **Assign yourself**: Open the issue → click "Assignees" → add yourself
3. **Move to In Progress**: Drag the card to "In Progress"
4. **Do the work**: Open the corresponding notebook in Databricks, make your changes
5. **Commit & push**: In your Databricks Git Folder, commit with a meaningful message referencing the issue (e.g., `Completed EDA — closes #7`)
6. **Move to Done**: Once committed, drag the card to "Done" (or close the issue and it moves automatically)

### 3. Branches (Optional but Recommended)

If multiple people are working at the same time:

1. Create a branch for your work: `feature/eda` or `feature/model-training`
2. Work in your branch in Databricks (Git Folder → switch branch)
3. When done, create a **Pull Request** to merge into `main`
4. A teammate reviews and approves
5. Merge the PR — the issue auto-closes if you wrote `closes #N` in the PR description

For the hackathon, working directly on `main` is also fine if you coordinate who's editing which notebook.

---

## 📓 Notebooks — Who Works on What

| Notebook | Purpose | Status |
|---|---|---|
| `00_decision_log.py` | AI vs human decision audit trail | 🟢 Everyone updates this |
| `01_data_loading.py` | Load Kaggle data into Databricks | ⬜ Needs: Kaggle dataset chosen |
| `02_eda.py` | Exploratory Data Analysis | ⬜ Needs: Data loaded first |
| `03_feature_engineering.py` | Feature transforms & train/val split | ⬜ Needs: EDA complete |
| `04_model_training.py` | MLflow-tracked model training + Hyperopt | ⬜ Needs: Features ready |
| `05_submission.py` | Generate Kaggle submission file | ⬜ Needs: Best model selected |

> ⚠️ **Notebooks are sequential** — each one depends on the previous. Coordinate on the board so you don't get ahead of each other.

---

## ✍️ How to Log AI Interactions

Whenever you use Databricks Genie (or any AI tool) to help with your work, log it in **two places**:

### In the GitHub Issue
Add a comment to the relevant issue:

```markdown
### 🤖 AI Interaction
**What I asked**: "Help me handle missing values in the age column"
**What AI suggested**: Median imputation for numeric, mode for categorical
**What I decided**: Used median for age, but dropped rows where target was missing
**Why**: Domain knowledge — missing targets can't be imputed reliably
```

### In the Decision Log Notebook
Add a new section to `notebooks/00_decision_log.py`:

```python
# MAGIC ## Decision N: [Short Title]
# MAGIC - **Date**: YYYY-MM-DD
# MAGIC - **AI suggested**: _what the AI recommended_
# MAGIC - **Human decided**: _what you actually did_
# MAGIC - **Rationale**: _why you made that choice_
```

---

## 🚀 Quick Start for New Team Members

1. **Get access**: Make sure you have access to the Databricks workspace and this GitHub repo
2. **Clone the repo in Databricks**: Workspace → Git Folders → Create Git Folder → paste the repo URL
3. **Check the board**: Go to the Projects tab → "project management" board → see what's in "To Do"
4. **Pick a task**: Assign yourself to an issue and move it to "In Progress"
5. **Open the notebook**: In your Databricks Git Folder, open the relevant notebook
6. **Do the work**: Run cells, modify code, add your analysis
7. **Commit**: Git icon in Databricks → write a commit message → Commit & Push
8. **Update the board**: Move your issue to "Done" or close it

---

## 📅 Key Dates & Milestones

| Milestone | Target | Status |
|---|---|---|
| Project setup & repo structure | Day 1 | ✅ Done |
| Data loaded & initial EDA | Day 1-2 | ⬜ |
| Feature engineering complete | Day 2-3 | ⬜ |
| Baseline model trained | Day 3 | ⬜ |
| Hyperparameter tuning done | Day 3-4 | ⬜ |
| Final submission to Kaggle | Day 4 | ⬜ |
| Presentation prep | Day 4-5 | ⬜ |

---

## ❓ Questions?

- **About the project**: Check the [README](../README.md)
- **About a specific task**: Read the GitHub Issue — it has the AI interaction log and context
- **About Databricks**: Ask Databricks Genie in the workspace — and log the interaction!
- **About Git/GitHub**: Ask a teammate or check [GitHub Docs](https://docs.github.com)
