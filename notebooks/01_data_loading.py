# Databricks notebook source
# MAGIC %md
# MAGIC # 📥 Step 1: Data Loading
# MAGIC 
# MAGIC Load the Kaggle competition dataset into Databricks.
# MAGIC 
# MAGIC **AI-assisted**: Databricks Genie provided the loading template.
# MAGIC **Human decision**: Chose file format, schema validation approach.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option A: Load from Unity Catalog Volume
# MAGIC Upload your Kaggle CSV to a Volume first, then read it here.

# COMMAND ----------

# TODO: Update the path to your actual Kaggle dataset
VOLUME_PATH = "/Volumes/<your_catalog>/<your_schema>/<your_volume>"

train_df = spark.read.csv(
    f"{VOLUME_PATH}/train.csv",
    header=True,
    inferSchema=True
)

test_df = spark.read.csv(
    f"{VOLUME_PATH}/test.csv",
    header=True,
    inferSchema=True
)

print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Schema Check

# COMMAND ----------

train_df.printSchema()

# COMMAND ----------

display(train_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save as Delta Table (optional but recommended)
# MAGIC Persisting as a Delta table makes downstream notebooks faster.

# COMMAND ----------

# TODO: Update catalog/schema names
# train_df.write.mode("overwrite").saveAsTable("<catalog>.<schema>.train")
# test_df.write.mode("overwrite").saveAsTable("<catalog>.<schema>.test")
