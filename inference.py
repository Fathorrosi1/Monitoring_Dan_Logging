# inference.py
import requests
import json
import pandas as pd

MODEL_URL = 'http://localhost:5001/invocations'

def load_sample():
    """Load sample data untuk inference."""
    df = pd.read_csv('namadataset_preprocessing/telco_churn_preprocessing.csv')
    sample = df.drop(columns=['Churn']).iloc[:5]
    return sample

def predict(sample):
    """Kirim request ke model."""
    payload = {
        "dataframe_split": {
            "columns": sample.columns.tolist(),
            "data": sample.values.tolist()
        }
    }
    
    response = requests.post(
        MODEL_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    return response

if __name__ == '__main__':
    print("Loading sample data...")
    sample = load_sample()
    print(f"Sample shape: {sample.shape}")
    
    print("\nMengirim request ke model...")
    try:
        response = predict(sample)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan model serving jalan di port 5001.")