import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import yaml

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# --------------------
# Config
# --------------------
MODEL_NAME = "titanic_spark_lr"

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
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))

spark = (
    SparkSession.builder
    .appName("titanic_api")
    .master(CFG["spark"]["master"])
    .config("spark.sql.shuffle.partitions", CFG["spark"]["shuffle_partitions"])
    .getOrCreate()
)

# 1) Fit the SAME preprocessing on the full training dataset once at startup
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

preproc_pipeline = Pipeline(stages=stages).fit(train_df)  # <— crucial: fit on TRAIN data!

# 2) Load the Production model from MLflow Model Registry
#    This model expects a 'features' vector with the SAME layout learned above.
model = mlflow.spark.load_model(model_uri=f"models:/{MODEL_NAME}/Production")

# --------------------
# Routes
# --------------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(p: Passenger):
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

        # derive engineered features exactly as for training dataset
        if "SibSp" in sdf.columns and "Parch" in sdf.columns:
            sdf = sdf.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
            sdf = sdf.withColumn("IsAlone", F.when((F.col("SibSp")+F.col("Parch"))==0,1).otherwise(0))

        # Use the PRE-FIT transformer (trained on the full dataset) so vector size/order matches training
        sdf2 = preproc_pipeline.transform(sdf)

        pred = model.transform(sdf2).select("prediction").collect()[0][0]
        return {"prediction": int(pred)}
    except Exception as e:
        # surface concise error to client and log full stack in server logs
        raise HTTPException(status_code=500, detail=str(e))
