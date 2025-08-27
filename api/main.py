"""
FastAPI inference service for Titanic (Spark pipeline + MLflow registry).

Key additions (Task 5):
- /reload-model endpoint to hot-reload the MODEL_NAME:MODEL_STAGE from MLflow Model Registry
  without restarting the server.
- Model loading refactored into a function (load_registry_model) used on startup and on reload.

Notes:
- Preprocessing pipeline (StringIndexer/OneHot/Assembler) is FIT ON TRAIN DATA AT STARTUP
  so the 'features' vector layout exactly matches training.
- Keep MODEL_NAME/MODEL_STAGE configurable via environment variables.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import yaml
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# --------------------
# Config
# --------------------
# Allow overriding via env vars for flexibility in CI/CD
MODEL_NAME = os.environ.get("MODEL_NAME", "titanic_spark_lr")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")  # e.g., "Production" or "Staging"

with open("conf/train.yaml", "r") as f:
    CFG = yaml.safe_load(f)
DATASET_PATH = CFG["paths"]["dataset"]  # e.g. data/processed/dataset.parquet

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Titanic Spark Inference API")

class Passenger(BaseModel):
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    Sex: str
    Embarked: str
    Pclass: int

# --------------------
# Spark + MLflow setup
# --------------------
# Tracking URI can be local file store, local server, or remote server
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))

spark = (
    SparkSession.builder
    .appName("titanic_api")
    .master(CFG["spark"]["master"])
    .config("spark.sql.shuffle.partitions", CFG["spark"]["shuffle_partitions"])
    .getOrCreate()
)

# 1) Fit the SAME preprocessing on the full training dataset once at startup
#    This preserves column order / index mappings used during training.
train_df = spark.read.parquet(DATASET_PATH)

cat_cols = []
if "Sex" in train_df.columns:      cat_cols.append("Sex")
if "Embarked" in train_df.columns: cat_cols.append("Embarked")
if "Pclass" in train_df.columns:   cat_cols.append("Pclass")

num_cols = []
if "Age" in train_df.columns:  num_cols.append("Age")
if "Fare" in train_df.columns: num_cols.append("Fare")

# derive engineered features exactly as in training helper
if "SibSp" in train_df.columns and "Parch" in train_df.columns:
    train_df = train_df.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
    train_df = train_df.withColumn("IsAlone", F.when((F.col("SibSp")+F.col("Parch"))==0,1).otherwise(0))
    num_cols += ["FamilySize", "IsAlone"]

stages = []
if cat_cols:
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    enc = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_cols],
        outputCols=[f"{c}_ohe" for c in cat_cols]
    )
    stages += indexers + [enc]
    feat_inputs = num_cols + [f"{c}_ohe" for c in cat_cols]
else:
    feat_inputs = num_cols

assembler = VectorAssembler(inputCols=feat_inputs, outputCol="features", handleInvalid="keep")
stages.append(assembler)

# PRE-FIT transformer (trained on the full TRAIN dataset)
# Crucial so that 'features' vector size/order matches what the model expects.
preproc_pipeline = Pipeline(stages=stages).fit(train_df)

# --------------------
# MLflow Model Registry loader (+ global state)
# --------------------
# Keep a global 'model' reference that can be swapped by /reload-model
model = None

def load_registry_model() -> None:
    """
    Load the latest model for MODEL_NAME at MODEL_STAGE from MLflow Model Registry.
    This function can be called at startup and at runtime (via /reload-model) to hot-reload.
    """
    global model
    # Ensure tracking URI is set (re-set is cheap and safe)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"  # e.g., models:/titanic_spark_lr/Production
    # Use mlflow.spark.load_model because the model was logged with Spark flavor
    model = mlflow.spark.load_model(model_uri=model_uri)

# Load at process start
@app.on_event("startup")
def _startup():
    load_registry_model()

# --------------------
# Routes
# --------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }

@app.post("/reload-model")
def reload_model():
    """
    Hot-reload the current MODEL_NAME:MODEL_STAGE from MLflow Model Registry.

    Typical use:
    - After retraining and promoting a new version to Production/Staging,
      call this endpoint to make the API serve the new version immediately.
    """
    try:
        load_registry_model()
        return {"status": "ok", "message": f"Reloaded {MODEL_NAME}:{MODEL_STAGE}"}
    except Exception as e:
        # Surface concise error; full stack appears in server logs
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(p: Passenger):
    """
    Predict endpoint for a single passenger record.
    - Builds a single-row Spark DataFrame with RAW columns
    - Applies the PRE-FIT preprocessing pipeline (fitted on training data)
    - Transforms to 'features' vector
    - Calls the Spark model's .transform() to get the prediction
    """
    try:
        # Build a single-row Spark DF with the RAW columns
        row = {
            "Age": p.Age,
            "Fare": p.Fare,
            "SibSp": p.SibSp,
            "Parch": p.Parch,
            "Sex": p.Sex,
            "Embarked": p.Embarked,
            "Pclass": p.Pclass,
        }
        sdf = spark.createDataFrame([row])

        # Derive engineered features exactly as for training dataset
        if "SibSp" in sdf.columns and "Parch" in sdf.columns:
            sdf = sdf.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
            sdf = sdf.withColumn("IsAlone", F.when((F.col("SibSp")+F.col("Parch"))==0,1).otherwise(0))

        # Use the PRE-FIT transformer (trained on the full dataset) so vector size/order matches training
        sdf2 = preproc_pipeline.transform(sdf)

        # Spark model inference (expects 'features' column)
        pred = model.transform(sdf2).select("prediction").collect()[0][0]
        return {"prediction": int(pred)}
    except Exception as e:
        # Surface concise error to client and log full stack in server logs
        raise HTTPException(status_code=500, detail=str(e))
