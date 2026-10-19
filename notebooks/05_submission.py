# Databricks notebook source
# MAGIC %md
# MAGIC # 📤 Step 5: Generate Submission
# MAGIC 
# MAGIC Load the best model, predict on test data, and create the Kaggle submission file.
# MAGIC 
# MAGIC **AI-assisted**: Genie provided submission template.
# MAGIC **Human decision**: Final model selection, post-processing, threshold tuning.

# COMMAND ----------

import mlflow
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Best Model from MLflow

# COMMAND ----------

# TODO: Replace with your best run_id from the MLflow experiment UI
# best_run_id = "<your_best_run_id>"
# best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Test Data

# COMMAND ----------

# TODO: Load test data and apply same feature engineering as training
# test_pdf = spark.read.table("<catalog>.<schema>.test").toPandas()
# Apply same preprocessing steps from notebook 03

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Predictions

# COMMAND ----------

# predictions = best_model.predict(test_pdf)
# print(f"Generated {len(predictions)} predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Submission File

# COMMAND ----------

# TODO: Adjust column names to match Kaggle's expected format
# submission = pd.DataFrame({
#     "Id": test_pdf["id"],
#     "Target": predictions
# })
# 
# # Save to Volume for download
# submission.to_csv("/Volumes/<catalog>/<schema>/<volume>/submission.csv", index=False)
# print(f"Submission saved! Shape: {submission.shape}")
# display(submission.head())
