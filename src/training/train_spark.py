import argparse, json, os, yaml
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
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

    mlflow.set_tracking_uri(P['experiment']['tracking_uri'])
    mlflow.set_registry_uri(P['experiment']['registry_uri'])

    spark = build_spark(P['spark']['app_name'], P['spark']['master'], P['spark']['shuffle_partitions'])

    data_path = os.path.join('data','processed','dataset.parquet')
    if not os.path.exists(data_path):
        logger.error("Processed dataset missing. Run preprocessing first.")
        raise SystemExit(2)

    df = spark.read.parquet(data_path)
    target = P['preprocess']['target']

    # Split
    train_df, test_df = df.randomSplit(
        [1.0 - P['preprocess']['test_size'], P['preprocess']['test_size']],
        seed=P['preprocess']['random_state']
    )

    # Features
    cat_cols = [c for c in df.columns if c in ['Sex','Embarked']]
    num_cols = [c for c in df.columns if c in ['Pclass','Age','SibSp','Parch','Fare']]

    stages = []
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid='keep'))

    features = num_cols + [f"{c}_idx" for c in cat_cols]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    stages.append(assembler)

    algo = P['model']['algo']
    if algo == 'logistic_regression':
        clf = LogisticRegression(labelCol=target, featuresCol='features', maxIter=int(P['model']['max_iter']))
    else:
        clf = RandomForestClassifier(labelCol=target, featuresCol='features', numTrees=int(P['model']['rf_n_estimators']))

    stages.append(clf)
    pipe = Pipeline(stages=stages)

    with mlflow.start_run(run_name=P['experiment']['name']):
        model = pipe.fit(train_df)
        preds = model.transform(test_df)

        evalr = MulticlassClassificationEvaluator(labelCol=target, predictionCol='prediction', metricName='accuracy')
        acc = evalr.evaluate(preds)

        # Log to MLflow
        mlflow.log_param('algo', algo)
        mlflow.log_param('max_iter', P['model']['max_iter'])
        mlflow.log_param('rf_n_estimators', P['model']['rf_n_estimators'])
        mlflow.log_metric('accuracy', float(acc))

        # Save model & metrics
        out_dir = os.path.join('models','best')
        model.write().overwrite().save(out_dir)
        os.makedirs('models', exist_ok=True)
        with open(os.path.join('models','metrics.json'), 'w') as f:
            json.dump({'accuracy': float(acc)}, f, indent=2)

        logger.info(f"Accuracy: {acc:.4f} | Model saved to {out_dir}")

    spark.stop()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    args = ap.parse_args()
    main(args.params)