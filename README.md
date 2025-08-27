# EndTerm MLOps Project – Titanic Survival Prediction (Spark + MLflow + FastAPI)

This project demonstrates an **MLOps pipeline** using **Apache Spark (PySpark MLlib)** for training, **MLflow** for experiment tracking & model registry, **DVC** for data & pipeline versioning, and **FastAPI** for model serving.

---

## Project Structure
EndTerm-MLops-Project/
├── api/ # FastAPI service for model inference
│ └── main.py
├── conf/ # YAML config files
│ ├── preprocess.yaml
│ └── train.yaml
├── data/ # Raw & processed data
│ ├── raw.dvc
│ └── processed/
├── docker/ # (Optional) Docker setup
├── models/ # Stored/exported models
├── reports/ # Metrics, confusion matrix, run_id.txt
├── src/ # Source code
│ ├── data_preprocessing/
│ │ └── preprocess_spark.py
│ ├── training/
│ │ └── train_spark_mllib.py
│ └── registry/
│ └── promote_to_production.py
├── dvc.yaml # Pipeline definition
├── params.yaml # Global parameters
├── requirements.txt # Python dependencies
└── README.md


---

### Setup

1. Clone the repository
git clone https://github.com/<your-username>/EndTerm-MLops-Project.git
cd EndTerm-MLops-Project

2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows (PowerShell)

3. Install dependencies
pip install -r requirements.txt


## Running the Pipeline with DVC
1. Reproduce the full pipeline:
    dvc repro

This will:
    Preprocess raw Titanic dataset
    Train a Logistic Regression model with cross-validation
    Log metrics and artifacts to MLflow
    Register the trained model in MLflow Model Registry

2. Force re-run all stages:
    dvc repro --force


## MLflow Tracking & Model Registry
Start MLflow Tracking Server:
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 -p 5001

Open MLflow UI at: http://127.0.0.1:5001
Experiment name: titanic_spark
Registered model: titanic_spark_lr


## Model Serving with FastAPI

1. Start FastAPI server:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
    uvicorn api.main:app --reload --port 8000

2. Test the API:
    python - <<'PY'
    import requests
    sample = {
    "Age": 29,
    "Fare": 35.0,
    "SibSp": 1,
    "Parch": 0,
    "Sex": "male",
    "Embarked": "S",
    "Pclass": 3
    }
r = requests.post("http://127.0.0.1:8000/predict", json=sample)
print("STATUS:", r.status_code)
print("RESPONSE:", r.json())
PY


Example Output:

{"prediction": 0}


## Dockerization

You can containerize the project (training + serving) using the docker/ setup provided.

Build the image:
    docker build -t titanic-mlops .
Run container:
    docker run -p 8000:8000 titanic-mlops



## Following Steps can also be done :

Add CI/CD integration (GitHub Actions)
Deploy API on cloud (AWS/GCP/Azure)
Automate retraining with updated data


##Author: 
    Bhavesh Pant (ch24m520)
