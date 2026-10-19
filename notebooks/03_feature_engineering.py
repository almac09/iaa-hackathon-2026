# Databricks notebook source
# MAGIC %md
# MAGIC # ⚙️ Step 3: Feature Engineering
# MAGIC 
# MAGIC Transform raw data into model-ready features.
# MAGIC 
# MAGIC **AI-assisted**: Genie suggested encoding strategies and common transforms.
# MAGIC **Human decision**: Which features to create based on domain knowledge and EDA findings.

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# TODO: Load your data
# pdf = spark.read.table("<catalog>.<schema>.train").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle Missing Values

# COMMAND ----------

# TODO: Fill or drop missing values based on EDA findings
# pdf['column_name'].fillna(pdf['column_name'].median(), inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encode Categorical Variables

# COMMAND ----------

# TODO: Identify and encode categorical columns
# categorical_cols = pdf.select_dtypes(include='object').columns
# for col in categorical_cols:
#     le = LabelEncoder()
#     pdf[col] = le.fit_transform(pdf[col].astype(str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create New Features

# COMMAND ----------

# TODO: Add domain-specific feature engineering
# Example: interaction terms, ratios, binning, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Validation Split

# COMMAND ----------

# TODO: Replace 'target' with your actual target column
# X = pdf.drop(columns=['target'])
# y = pdf['target']
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"Train: {X_train.shape}, Validation: {X_val.shape}")
