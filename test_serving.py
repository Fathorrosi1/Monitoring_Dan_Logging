import requests
import json
import pandas as pd

MODEL_URL = 'http://localhost:5001/invocations'

# Load sample dari dataset
df = pd.read_csv('namadataset_preprocessing/telco_churn_preprocessing.csv')
sample = df.drop(columns=['Churn']).iloc[:3]  # ambil 3 baris pertama

# Convert ke format MLflow
payload = {
    "dataframe_split": {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist()
    }
}

# Kirim request
response = requests.post(
    MODEL_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print("Status:", response.status_code)
print("Response:", response.json())