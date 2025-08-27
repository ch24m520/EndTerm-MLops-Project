import os, mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "titanic_spark_lr"
RUN_ID_FILE = "reports/training/run_id.txt"
OUT_DIR = "reports/registry"

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    run_id = open(RUN_ID_FILE).read().strip()
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True
    )

    # ensure DVC output exists
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "production.txt"), "w") as f:
        f.write(f"model={MODEL_NAME}\nversion={mv.version}\nrun_id={run_id}\n")

    print(f"Registered {MODEL_NAME} v{mv.version} → Production")

if __name__ == "__main__":
    main()

