from fastapi import FastAPI
from pydantic import BaseModel
import os
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.linalg import DenseVector, SparseVector

app = FastAPI(title="Titanic Classifier API (Spark, Linux)")

def build_spark():
    """Create a local Spark session for inference"""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("titanic-serving")
        .getOrCreate()
    )

spark = None
model = None

@app.on_event("startup")
def _load_model():
    """Load the trained Spark PipelineModel once at startup"""
    global spark, model
    spark = build_spark()
    model_dir = os.path.join("models", "best")
    model = PipelineModel.load(model_dir)

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Passenger):
    df = spark.createDataFrame([item.dict()])
    row = (
        model.transform(df)
             .select("prediction", "probability")
             .head()
    )

    pred = int(row["prediction"])

    prob = row["probability"]
    # Convert Spark vector -> Python list
    if isinstance(prob, (DenseVector, SparseVector)):
        proba = prob.toArray().tolist()
    else:
        try:
            proba = list(prob)
        except Exception:
            proba = [float(prob[0]), float(prob[1])]

    return {"prediction": pred, "probability": [float(p) for p in proba]}
