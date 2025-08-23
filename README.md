# End-Term MLOps Project — Titanic (Spark + FastAPI)

## Prerequisites
- WSL (Ubuntu), Python 3.12
- Java 17 (OpenJDK, already installed)
- Create and activate venv:  
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
# Preprocess
python -m src.data_preprocessing.preprocess_spark --params params.yaml

# Train
python -m src.training.train_spark --params params.yaml

uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Swagger docs: http://127.0.0.1:8000/docs

Healthcheck: http://127.0.0.1:8000/health

Test the API
python src/api/test_api.py

MLflow UI
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000


Open http://127.0.0.1:5000
