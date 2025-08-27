#!/usr/bin/env bash
# Quick environment setup for EndTerm-MLops-Project
# Usage: source ./set_env.sh

# Use clean, file-based MLflow store
export MLFLOW_TRACKING_URI="file:./mlruns_clean"
export MLFLOW_REGISTRY_URI="file:./mlruns_clean"

# Model registry identifiers used by API
export MODEL_NAME="titanic_spark_lr"
export MODEL_STAGE="Production"

echo "[ok] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "[ok] MLFLOW_REGISTRY_URI=$MLFLOW_REGISTRY_URI"
echo "[ok] MODEL_NAME=$MODEL_NAME"
echo "[ok] MODEL_STAGE=$MODEL_STAGE"
