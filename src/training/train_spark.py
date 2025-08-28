import argparse, json, os, time, yaml
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import mlflow
import mlflow.spark

from src.utils.logger import get_logger


# ----------------------------
# Spark session builder
# ----------------------------
def build_spark(app_name: str, master: str, shuffle_partitions: int):
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .getOrCreate()
    )


# ----------------------------
# Telemetry row for runs.csv
# ----------------------------
def _collect_spark_metrics_row(spark, P, fit_time_sec: float, acc: float,
                               feature_count: int, dataset_rows: int) -> dict:
    sc = spark.sparkContext
    conf = dict(sc.getConf().getAll())
    # utcnow is fine for a simple timestamp
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "experiment": P["mlflow"].get("experiment", "default"),
        "model_type": P.get("model", {}).get("type", "logreg"),
        "fit_time_sec": round(float(fit_time_sec), 4),
        "accuracy": round(float(acc), 6),
        "shuffle_partitions": int(P["spark"].get("shuffle_partitions", 4)),
        "defaultParallelism": int(sc.defaultParallelism),
        "app_name": P["spark"].get("app_name", "train"),
        "driver_memory": conf.get("spark.driver.memory", ""),
        "executor_memory": conf.get("spark.executor.memory", ""),
        "master": P["spark"].get("master", ""),
        "dataset_rows": int(dataset_rows),
        "feature_count": int(feature_count),
    }


# ----------------------------
# Optional guard: ensure label/features if missing (fallback path)
# ----------------------------
def ensure_features_and_label(df):
    """
    If dataset already has appropriate columns per below, this is a no-op.
    Otherwise, minimally creates:
      - label (from Survived)
      - features (VectorAssembler of numeric + OHE categorical)
    """
    if "label" not in df.columns and "Survived" in df.columns:
        df = df.withColumnRenamed("Survived", "label")

    # numeric
    num_cols = []
    if "Age" in df.columns:  num_cols.append("Age")
    if "Fare" in df.columns: num_cols.append("Fare")

    # engineered family features if present
    if "SibSp" in df.columns and "Parch" in df.columns:
        df = df.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
        df = df.withColumn("IsAlone", F.when((F.col("SibSp") + F.col("Parch")) == 0, 1).otherwise(0))
        if "FamilySize" not in num_cols: num_cols += ["FamilySize", "IsAlone"]

    # categorical
    cat_cols = []
    if "Sex" in df.columns:      cat_cols.append("Sex")
    if "Embarked" in df.columns: cat_cols.append("Embarked")
    if "Pclass" in df.columns:   cat_cols.append("Pclass")  # treat as categorical

    stages = []
    if cat_cols:
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
        enc = OneHotEncoder(
            inputCols=[f"{c}_idx" for c in cat_cols],
            outputCols=[f"{c}_ohe" for c in cat_cols],
        )
        stages += indexers + [enc]
        feat_inputs = num_cols + [f"{c}_ohe" for c in cat_cols]
    else:
        feat_inputs = num_cols

    assembler = VectorAssembler(inputCols=feat_inputs, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    pre = Pipeline(stages=stages).fit(df)
    out = pre.transform(df)

    keep = ["features"] + (["label"] if "label" in out.columns else [])
    if not keep:
        return df  # no change possible
    return out.select(*keep)


# ----------------------------
# Main
# ----------------------------
def main(params_path: str):
    logger = get_logger("train")

    with open(params_path, "r") as f:
        P = yaml.safe_load(f)

    # MLflow setup (ENV → YAML → fallback)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", P.get("mlflow", {}).get("tracking_uri", "file:./mlruns_clean"))
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", P.get("mlflow", {}).get("registry_uri", tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(P.get("mlflow", {}).get("experiment", "default"))

    # helpful one-time print
    try:
        from mlflow.tracking import MlflowClient
        _client = MlflowClient()
        print(f"[trainer] tracking_uri={mlflow.get_tracking_uri()} registry_uri={_client._registry_uri}")
    except Exception:
        print(f"[trainer] tracking_uri={mlflow.get_tracking_uri()}")

    # Spark
    spark = build_spark(
        P["spark"].get("app_name", "train"),
        P["spark"].get("master", "local[*]"),
        int(P["spark"].get("shuffle_partitions", 4)),
    )

    # Data
    data_path = P.get("paths", {}).get("dataset", os.path.join("data", "processed", "dataset.parquet"))
    if not os.path.exists(data_path):
        logger.error(f"Processed dataset missing at {data_path}. Run preprocessing first.")
        raise SystemExit(2)

    df_raw = spark.read.parquet(data_path)

    # Build training pipeline like before (explicit columns)
    cat_cols = [c for c in df_raw.columns if c in ["Sex", "Embarked", "Pclass"]]
    num_cols = [c for c in df_raw.columns if c in ["Pclass", "Age", "SibSp", "Parch", "Fare"]]  # Pclass both places OK

    stages = []
    if cat_cols:
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
        enc = OneHotEncoder(
            inputCols=[f"{c}_idx" for c in cat_cols],
            outputCols=[f"{c}_ohe" for c in cat_cols],
        )
        stages += indexers + [enc]

    feat_inputs = [c for c in ["Pclass", "Age", "SibSp", "Parch", "Fare"] if c in df_raw.columns] + \
                  ([f"{c}_ohe" for c in cat_cols] if cat_cols else [])
    assembler = VectorAssembler(inputCols=feat_inputs, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    # Label handling
    target = P.get("preprocess", {}).get("target", "Survived")
    if target in df_raw.columns and target != "label":
        df = df_raw.withColumnRenamed(target, "label")
    else:
        df = ensure_features_and_label(df_raw)  # fallback if needed

    # Split
    test_size = float(P["train"].get("test_size", 0.2))
    seed = int(P["train"].get("seed", 42))
    train_df, test_df = df.randomSplit([1.0 - test_size, test_size], seed=seed)

    # Choose model
    mcfg = P.get("model", {})
    mtype = mcfg.get("type", "logreg")  # "logreg" or "random_forest"

    if mtype == "random_forest":
        ntrees = int(mcfg.get("rf_n_estimators", 100))
        clf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=ntrees)
    else:
        max_iter = int(mcfg.get("maxIter", 50))
        clf = LogisticRegression(labelCol="label", featuresCol="features", maxIter=max_iter)

    stages_with_model = stages + [clf]
    pipe = Pipeline(stages=stages_with_model)

    # Optional CV (for logreg only); controlled by YAML key: model.cv_enabled
    cv_enabled = bool(mcfg.get("cv_enabled", False)) and mtype == "logreg"
    if cv_enabled:
        lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=int(mcfg.get("maxIter", 50)))
        grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, mcfg.get("regParam_grid", [0.0, 0.01, 0.1]))
            .addGrid(lr.elasticNetParam, mcfg.get("elasticNet_grid", [0.0, 0.5]))
            .build()
        )
        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"
        )
        cv = CrossValidator(
            estimator=Pipeline(stages=stages + [lr]),
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=int(mcfg.get("cv_folds", 3)),
            parallelism=int(mcfg.get("parallelism", 2)),
        )

    # ----------------------------
    # Train + log to MLflow
    # ----------------------------
    with mlflow.start_run(run_name=P.get("mlflow", {}).get("experiment", "run")):
        # Spark conf snapshot
        conf_dict = {k: v for k, v in spark.sparkContext.getConf().getAll()}
        mlflow.log_dict(conf_dict, "artifacts/spark_conf.json")

        # Fit
        t0 = time.time()
        if cv_enabled:
            model = cv.fit(train_df)
            best_stage = model.bestModel.stages[-1]  # LR stage
        else:
            model = pipe.fit(train_df)
            best_stage = model.stages[-1] if hasattr(model, "stages") else None
        fit_time_sec = time.time() - t0
        mlflow.log_metric("fit_time_sec", float(fit_time_sec))

        # Evaluate accuracy
        preds = model.transform(test_df)
        acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        acc = acc_eval.evaluate(preds)
        mlflow.log_metric("accuracy", float
