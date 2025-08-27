import os, json, argparse, yaml
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import mlflow, mlflow.spark

def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def ensure_features_and_label(df):
    # If label not present but Survived exists, rename
    if "label" not in df.columns and "Survived" in df.columns:
        df = df.withColumnRenamed("Survived", "label")

    # If features already exists, nothing to do
    if "features" in df.columns:
        return df

    # Build minimal Titanic features if possible
    num_cols = []
    if "Age" in df.columns:        num_cols.append("Age")
    if "Fare" in df.columns:       num_cols.append("Fare")

    if "SibSp" in df.columns and "Parch" in df.columns:
        df = df.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
        df = df.withColumn("IsAlone", F.when((F.col("SibSp")+F.col("Parch"))==0,1).otherwise(0))
        num_cols += ["FamilySize", "IsAlone"]

    cat_cols = []
    if "Sex" in df.columns:      cat_cols.append("Sex")
    if "Embarked" in df.columns: cat_cols.append("Embarked")
    if "Pclass" in df.columns:   cat_cols.append("Pclass")  # treat as categorical

    stages = []
    if cat_cols:
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
        enc = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                            outputCols=[f"{c}_ohe" for c in cat_cols])
        stages += indexers + [enc]
        feat_inputs = num_cols + [f"{c}_ohe" for c in cat_cols]
    else:
        feat_inputs = num_cols

    assembler = VectorAssembler(inputCols=feat_inputs, outputCol="features")
    stages.append(assembler)

    pipeline = Pipeline(stages=stages).fit(df)
    df2 = pipeline.transform(df)
    # Keep label if present, otherwise training will fail with a clear error
    keep = ["features"] + (["label"] if "label" in df2.columns else [])
    return df2.select(*keep)

def main(cfg_path):
    C = load_yaml(cfg_path)
    ds = C["paths"]["dataset"]
    out_dir = C["paths"]["reports_dir"]
    mlruns_dir = C["paths"]["mlruns_dir"]
    ensure_dir(out_dir)

    spark = (
        SparkSession.builder
        .appName(C["spark"]["app_name"])
        .master(C["spark"]["master"])
        .config("spark.sql.shuffle.partitions", C["spark"]["shuffle_partitions"])
        .getOrCreate()
    )

    df = spark.read.parquet(ds)
    df = ensure_features_and_label(df)

    # safety: ensure both required cols exist
    for col in ["features", "label"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing after preprocessing fallback. "
                             f"Available: {df.columns}")

    train, test = df.randomSplit(
        [1.0 - C["train"]["test_size"], C["train"]["test_size"]],
        seed=C["train"]["seed"]
    )

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=C["model"]["maxIter"])
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, C["model"]["regParam_grid"])
        .addGrid(lr.elasticNetParam, C["model"]["elasticNet_grid"])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                              labelCol="label", metricName="areaUnderROC")
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=C["model"]["cv_folds"],
                        parallelism=C["model"]["parallelism"])

    # MLflow local file store
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    mlflow.set_experiment(C["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        model = cv.fit(train)
        best = model.bestModel
        test_pred = model.transform(test)

        auc = evaluator.evaluate(test_pred)
        acc = (
            test_pred.withColumn("correct", (F.col("prediction") == F.col("label")).cast("int"))
            .agg(F.avg("correct").alias("acc"))
            .collect()[0]["acc"]
        )

        # Save artifacts
        cm = test_pred.groupBy("label","prediction").count().toPandas()
        cm_path = os.path.join(out_dir, "confusion_matrix.csv")
        cm.to_csv(cm_path, index=False)

        coef_path = os.path.join(out_dir, "coefficients.txt")
        with open(coef_path, "w") as f:
            f.write(str(best.coefficients))

        metrics = {"test_auc": float(auc), "test_acc": float(acc)}
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_params({
            "maxIter": best._java_obj.getMaxIter(),
            "regParam": best._java_obj.getRegParam(),
            "elasticNetParam": best._java_obj.getElasticNetParam(),
            "cv_folds": C["model"]["cv_folds"],
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(coef_path)
        mlflow.spark.log_model(model, artifact_path="model")

        with open(os.path.join(out_dir, "run_id.txt"), "w") as f:
            f.write(run.info.run_id)

    spark.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="Path to train.yaml")
    args = parser.parse_args()
    main(args.params)
