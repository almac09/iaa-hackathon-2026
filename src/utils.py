"""
Shared utility functions for the IAA Hackathon 2026 project.

This module contains reusable helper functions used across multiple notebooks.
AI-assisted: Databricks Genie provided the initial template.
Human decision: Functions added/modified based on project needs.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def load_kaggle_data(spark, volume_path: str, filename: str) -> pd.DataFrame:
    """Load a CSV file from a Unity Catalog Volume and return as pandas DataFrame."""
    df = spark.read.csv(
        f"{volume_path}/{filename}",
        header=True,
        inferSchema=True
    )
    return df.toPandas()


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': percent.round(2)
    })
    return result[result['missing_count'] > 0].sort_values('missing_count', ascending=False)


def log_decision(decision_num: int, date: str, ai_suggestion: str, human_decision: str, rationale: str):
    """Print a formatted decision log entry (for use in notebooks)."""
    print(f"""\n{'='*60}
## Decision {decision_num} ({date})
- AI suggested: {ai_suggestion}
- Human decided: {human_decision}
- Rationale: {rationale}
{'='*60}""")
