import os, json, argparse, yaml
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow, mlflow.spark

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main(cfg_path):
    C = load_yaml(cfg_path)
    ds = C["paths"]["dataset"]
    out_dir = C["paths"]["reports_dir"]
    mlruns_dir = C["paths"]["mlruns_dir"]
    ensure_dir(out_dir)

    # Spark
    spark = (
        SparkSession.builder
        .appName(C["spark"]["app_name"])
        .master(C["spark"]["master"])
        .config("spark.sql.shuffle.partitions", C["spark"]["shuffle_partitions"])
        .getOrCreate()
    )
# Data
    df = spark.read.parquet(ds)  # expects columns: features, label
    train, test = df.randomSplit(
        [1.0 - C["train"]["test_size"], C["train"]["test_size"]],
        seed=C["train"]["seed"]
    )
# Model & CV
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=C["model"]["maxIter"])
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, C["model"]["regParam_grid"])
        .addGrid(lr.elasticNetParam, C["model"]["elasticNet_grid"])
        .build()
    )
evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"
    )
cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=C["model"]["cv_folds"],
        parallelism=C["model"]["parallelism"],
    )
# MLflow (local file store)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    mlflow.set_experiment(C["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        model = cv.fit(train)
        best = model.bestModel
        test_pred = model.transform(test)

        # Metrics
        auc = evaluator.evaluate(test_pred)
        acc = (
            test_pred.withColumn("correct", (F.col("prediction") == F.col("label")).cast("int"))
            .agg(F.avg("correct").alias("acc"))
            .collect()[0]["acc"]
        )
# Confusion matrix (2x2)
        cm = test_pred.groupBy("label", "prediction").count().toPandas()
        cm_path = os.path.join(out_dir, "confusion_matrix.csv")
        cm.to_csv(cm_path, index=False)

        # Coefficients (for LR)
        coef_path = os.path.join(out_dir, "coefficients.txt")
        with open(coef_path, "w") as f:
            f.write(str(best.coefficients))

        # Simple metrics JSON (also DVC-tracked)
        metrics = {"test_auc": float(auc), "test_acc": float(acc)}
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # MLflow logging
        mlflow.log_params({
            "maxIter": best._java_obj.getMaxIter(),
            "regParam": best._java_obj.getRegParam(),
            "elasticNetParam": best._java_obj.getElasticNetParam(),
            "cv_folds": C["model"]["cv_folds"],
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(coef_path)

        # Log full CV model (pipeline) as artifact
        mlflow.spark.log_model(model, artifact_path="model")

        # Save run_id
        with open(os.path.join(out_dir, "run_id.txt"), "w") as f:
            f.write(run.info.run_id)

    spark.stop()
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to train.yaml")
    args = ap.parse_args()
    main(args.params)
