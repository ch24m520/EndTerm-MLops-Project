import requests

def call_sample():
    url = "http://127.0.0.1:8000/predict"
    sample = {
        "Pclass": 3, "Sex": "male", "Age": 22,
        "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"
    }
    resp = requests.post(url, json=sample)
    print("Status:", resp.status_code)
    print("Response:", resp.json())

if __name__ == "__main__":
    call_sample()
