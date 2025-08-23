import argparse, yaml, os
from pyspark.sql import SparkSession
from src.utils.logger import get_logger

def build_spark(app_name: str, master: str, shuffle_partitions: int):
    return (SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .getOrCreate())

def main(params_path: str):
    logger = get_logger("preprocess")
    with open(params_path, "r") as f:
        P = yaml.safe_load(f)

    spark = build_spark(P['spark']['app_name'], P['spark']['master'], P['spark']['shuffle_partitions'])

    raw_dir = "data/raw"
    proc_dir = "data/processed"
    os.makedirs(proc_dir, exist_ok=True)

    # Expect Kaggle Titanic train.csv renamed to titanic.csv
    raw_csv = os.path.join(raw_dir, "titanic.csv")
    if not os.path.exists(raw_csv):
        logger.error(f"Expected {raw_csv}. Please put Titanic CSV there.")
        raise SystemExit(2)

    df = spark.read.csv(raw_csv, header=True, inferSchema=True)

    # Keep minimal columns (adjust if your CSV headers differ)
    keep_cols = [c for c in df.columns if c in ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    df = df.select(*keep_cols)

    # Basic NA handling
    for c in ['Age','Fare']:
        df = df.fillna({c: 0.0})
    df = df.fillna({'Embarked': 'S'})

    # Save as parquet for speed
    out_path = os.path.join(proc_dir, 'dataset.parquet')
    df.write.mode('overwrite').parquet(out_path)
    logger.info(f"Preprocessing complete: {out_path}")

    spark.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    main(args.params)