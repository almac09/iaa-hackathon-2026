# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 Step 4: Model Training with MLflow
# MAGIC 
# MAGIC Train models and track experiments using MLflow.
# MAGIC 
# MAGIC **AI-assisted**: Genie provided MLflow integration template and Hyperopt setup.
# MAGIC **Human decision**: Model selection, hyperparameter ranges, evaluation criteria.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up MLflow Experiment

# COMMAND ----------

mlflow.set_experiment("/Users/alan.mcdonagh@rgare.com/iaa-hackathon-2026")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Model

# COMMAND ----------

# TODO: Load X_train, X_val, y_train, y_val from feature engineering
# Or run notebook 03 first: %run ./03_feature_engineering

with mlflow.start_run(run_name="baseline-random-forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print(classification_report(y_val, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning with Hyperopt

# COMMAND ----------

def objective(params):
    with mlflow.start_run(nested=True):
        clf = GradientBoostingClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            random_state=42
        )
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, clf.predict(X_val))
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        return {"loss": -accuracy, "status": STATUS_OK}

search_space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300]),
    "max_depth": hp.choice("max_depth", [3, 5, 7, 9]),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.3))
}

with mlflow.start_run(run_name="hyperopt-gbm-tuning"):
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=Trials()
    )
    print(f"Best params: {best}")
