# 🏆 IAA Hackathon 2026

**AI-Assisted Machine Learning on Databricks**

> This project demonstrates an AI-human collaboration workflow for a Kaggle competition, built entirely on Databricks. Every decision is logged, every AI interaction is tracked, and the full evolution of the project is visible through GitHub Issues, commit history, and an in-notebook decision log.

## 🤖 AI-Human Collaboration Model

| Layer | What It Shows |
|---|---|
| **GitHub Issues** | Full conversation trail — what AI suggested vs what the human decided |
| **Git Commit History** | How the code evolved over time |
| **Decision Log** (`notebooks/00_decision_log.py`) | Reasoning at each fork in the road |
| **Notebook Version History** | Granular iteration snapshots in Databricks |
| **GitHub Project Board** | Kanban-style task tracking across all phases |

**The human** (Alan McDonagh) drives architecture, makes judgment calls, applies domain knowledge, and manages the project.
**The AI** (Databricks Genie) generates code templates, suggests approaches, automates repetitive setup, and provides Databricks platform guidance.

## 📁 Project Structure

```
iaa-hackathon-2026/
├── notebooks/
│   ├── 00_decision_log.py      # AI vs human decision audit trail
│   ├── 01_data_loading.py      # Load Kaggle data into Databricks
│   ├── 02_eda.py               # Exploratory Data Analysis
│   ├── 03_feature_engineering.py # Feature transforms & train/val split
│   ├── 04_model_training.py    # MLflow-tracked model training + Hyperopt
│   └── 05_submission.py        # Generate Kaggle submission file
├── src/
│   └── utils.py                # Shared helper functions
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- A Kaggle account and competition dataset
- This repo cloned as a Git Folder in Databricks

### Setup
1. Clone this repo in Databricks: **Workspace → Git Folders → Create Git Folder**
2. Upload your Kaggle dataset to a Unity Catalog Volume
3. Update the `VOLUME_PATH` in `notebooks/01_data_loading.py`
4. Run notebooks in order: `01` → `02` → `03` → `04` → `05`

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Databricks Notebooks** | Development environment |
| **Unity Catalog** | Data governance & storage |
| **MLflow** | Experiment tracking & model registry |
| **Hyperopt** | Automated hyperparameter tuning |
| **Databricks Genie** | AI coding assistant |
| **GitHub** | Version control, issues, project management |

## 📊 Project Phases

See the [Issues tab](../../issues) and [Project Board](../../projects) for detailed task tracking with AI interaction logs.

1. **Setup** — Repo, Git Folders, project scaffold ✅
2. **Data Loading** — Kaggle dataset → Unity Catalog
3. **EDA** — Explore, visualise, understand the data
4. **Feature Engineering** — Transform raw data into model-ready features
5. **Model Training** — Baseline + iteration with MLflow tracking
6. **Hyperparameter Tuning** — Hyperopt optimisation
7. **Submission** — Generate predictions and submit to Kaggle

## 📝 License

This project was created for the IAA Hackathon 2026.
