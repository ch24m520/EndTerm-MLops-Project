import argparse, json, os, yaml
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark          # for logging Spark models to MLflow
import time                  # for fit_time_sec
from src.utils.logger import get_logger


def build_spark(app_name: str, master: str, shuffle_partitions: int):
    return (SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .getOrCreate())


def main(params_path: str):
    logger = get_logger("train")
    with open(params_path, "r") as f:
        P = yaml.safe_load(f)

    # ----------------------------
    # MLflow setup (ENV → YAML → fallback)
    # ----------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", P["mlflow"].get("tracking_uri", "file:./mlruns_clean"))
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", P["mlflow"].get("registry_uri", tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(P["mlflow"].get("experiment", "default"))
    # Debug: print effective URIs once per run
    try:
        from mlflow.tracking import MlflowClient
        _client = MlflowClient()
        print(f"[trainer] tracking_uri={mlflow.get_tracking_uri()} registry_uri={_client._registry_uri}")
    except Exception:
        print(f"[trainer] tracking_uri={mlflow.get_tracking_uri()}")

    # ----------------------------
    # Spark setup (from spark.*)
    # ----------------------------
    spark = build_spark(
        P["spark"].get("app_name", "train"),
        P["spark"].get("master", "local[*]"),
        int(P["spark"].get("shuffle_partitions", 4))
    )

    # ----------------------------
    # Data path (from paths.dataset)
    # ----------------------------
    data_path = P.get("paths", {}).get("dataset", os.path.join("data", "processed", "dataset.parquet"))
    if not os.path.exists(data_path):
        logger.error(f"Processed dataset missing at {data_path}. Run preprocessing first.")
        raise SystemExit(2)

    df = spark.read.parquet(data_path)

    # Target column (fallback to 'Survived' if not specified)
    target = P.get("preprocess", {}).get("target", "Survived")

    # ----------------------------
    # Train/test split (from train.*)
    # ----------------------------
    test_size = float(P["train"].get("test_size", 0.2))
    seed = int(P["train"].get("seed", 42))
    train_df, test_df = df.randomSplit([1.0 - test_size, test_size], seed=seed)

    # ----------------------------
    # Feature engineering
    # ----------------------------
    cat_cols = [c for c in df.columns if c in ["Sex", "Embarked"]]
    num_cols = [c for c in df.columns if c in ["Pclass", "Age", "SibSp", "Parch", "Fare"]]

    stages = []
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))

    features = num_cols + [f"{c}_idx" for c in cat_cols]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    stages.append(assembler)

    # ----------------------------
    # Model selection (from model.*)
    # ----------------------------
    mcfg = P.get("model", {})
    mtype = mcfg.get("type", "logreg")    # "logreg" (default) or "random_forest"

    if mtype == "random_forest":
        ntrees = int(mcfg.get("rf_n_estimators", 100))
        clf = RandomForestClassifier(labelCol=target, featuresCol="features", numTrees=ntrees)
    else:
        # default: logistic regression
        max_iter = int(mcfg.get("maxIter", 50))
        clf = LogisticRegression(labelCol=target, featuresCol="features", maxIter=max_iter)

    stages.append(clf)
    pipe = Pipeline(stages=stages)

    # ----------------------------
    # Training + MLflow logging
    # ----------------------------
    with mlflow.start_run(run_name=P["mlflow"].get("experiment", "run")):
        # Log Spark conf snapshot for reproducibility
        conf_dict = {k: v for k, v in spark.sparkContext.getConf().getAll()}
        mlflow.log_dict(conf_dict, "artifacts/spark_conf.json")

        # Time the fit()
        t0 = time.time()
        model = pipe.fit(train_df)
        fit_time_sec = time.time() - t0
        mlflow.log_metric("fit_time_sec", float(fit_time_sec))

        # Evaluate
        preds = model.transform(test_df)
        evalr = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")
        acc = evalr.evaluate(preds)

        # Log params/metrics (keep names stable)
        mlflow.log_param("model_type", mtype)
        if mtype == "logreg":
            mlflow.log_param("maxIter", int(mcfg.get("maxIter", 50)))
        else:
            mlflow.log_param("rf_n_estimators", int(mcfg.get("rf_n_estimators", 100)))
        mlflow.log_metric("accuracy", float(acc))

        # Log Spark model artifact (for registry / loading via models:/)
        mlflow.spark.log_model(spark_model=model, artifact_path="model")

        # Auto-register & promote if a model_name is provided in YAML
        model_name = P["mlflow"].get("model_name")
        if model_name:
            from mlflow.tracking import MlflowClient
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            try:
                mv = mlflow.register_model(model_uri=model_uri, name=model_name)
                MlflowClient().transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Production", archive_existing_versions=True
                )
                logger.info(f"[trainer] Registered '{model_name}' v{mv.version} -> Production")
            except Exception as e:
                logger.warning(f"[trainer] Model registry step skipped/failed: {e}")

        # Keep original local artifacts
        out_dir = os.path.join("models", "best")
        os.makedirs("models", exist_ok=True)
        model.write().overwrite().save(out_dir)
        with open(os.path.join("models", "metrics.json"), "w") as f:
            json.dump({"accuracy": float(acc), "fit_time_sec": float(fit_time_sec)}, f, indent=2)

        logger.info(f"Accuracy: {acc:.4f} | Fit time: {fit_time_sec:.2f}s | Model saved to {out_dir}")

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)
