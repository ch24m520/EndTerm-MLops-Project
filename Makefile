Y=python

preprocess:
	$(PY) -m src.data_preprocessing.preprocess_spark --params params.yaml

train:
	$(PY) -m src.training.train_spark --params params.yaml

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test-api:
	$(PY) src/api/test_api.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
