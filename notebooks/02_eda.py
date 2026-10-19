# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 Step 2: Exploratory Data Analysis (EDA)
# MAGIC 
# MAGIC Understand the data before building models.
# MAGIC 
# MAGIC **AI-assisted**: Genie provided EDA template and suggested visualisations.
# MAGIC **Human decision**: Which features to investigate, domain-specific insights.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# TODO: Update with your actual table or file path
# train_df = spark.read.table("<catalog>.<schema>.train")
pdf = train_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Statistics

# COMMAND ----------

pdf.describe()

# COMMAND ----------

pdf.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Values

# COMMAND ----------

missing = pdf.isnull().sum().sort_values(ascending=False)
missing[missing > 0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Distribution

# COMMAND ----------

# TODO: Replace 'target' with your actual target column
# pdf['target'].value_counts().plot(kind='bar', title='Target Distribution')
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation Heatmap

# COMMAND ----------

numeric_cols = pdf.select_dtypes(include='number')
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlations')
plt.tight_layout()
plt.show()
