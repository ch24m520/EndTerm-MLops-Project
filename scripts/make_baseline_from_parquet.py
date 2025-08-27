#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a baseline CSV for drift detection from the Spark training Parquet defined in conf/train.yaml.
Writes: data/processed/train_baseline.csv
"""
import yaml
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

CFG = yaml.safe_load(open("conf/train.yaml", "r"))
DATASET_PATH = CFG["paths"]["dataset"]  # e.g. data/processed/dataset.parquet

spark = (
    SparkSession.builder
    .appName("make_baseline_csv")
    .master(CFG["spark"]["master"])
    .config("spark.sql.shuffle.partitions", CFG["spark"]["shuffle_partitions"])
    .getOrCreate()
)

df = spark.read.parquet(DATASET_PATH)

# Keep only columns needed for drift + id/target (id/target will be ignored via --id-cols)
cols = [c for c in ["PassengerId","Survived","Age","Fare","SibSp","Parch","Sex","Embarked","Pclass"] if c in df.columns]
df = df.select(*cols)

# Optional: basic cleaning to avoid NaNs for PSI (you can adjust as per your training prep)
# Fill numeric nulls with median, categorical with 'Unknown'
num_cols = [c for c in ["Age","Fare","SibSp","Parch","Pclass"] if c in df.columns]
cat_cols = [c for c in ["Sex","Embarked"] if c in df.columns]

for c in num_cols:
    median = df.approxQuantile(c, [0.5], 0.01)[0] if c in df.columns else None
    if median is not None:
        df = df.na.fill({c: float(median)})

fill_map = {c: "Unknown" for c in cat_cols}
if fill_map:
    df = df.na.fill(fill_map)

# Write to a single CSV file
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = "data/processed/_baseline_tmp"

df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir)

# Move the single part file to final path
import shutil, glob
part = glob.glob(f"{tmp_dir}/part-*.csv")[0]
shutil.move(part, "data/processed/train_baseline.csv")
shutil.rmtree(tmp_dir)

print("Wrote data/processed/train_baseline.csv")
