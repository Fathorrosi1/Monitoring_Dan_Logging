# prometheus_exporter.py

import os
import time
import json
from collections import deque

import pandas as pd
import psutil
import requests

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram
)

# KONFIGURASI
MODEL_URL = os.getenv(
    "MODEL_URL",
    "http://localhost:5001/invocations"
)

DATA_PATH = os.getenv(
    "DATA_PATH",
    "namadataset_preprocessing/telco_churn_preprocessing.csv"
)

EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
MODEL_VERSION_NAME = os.getenv(
    "MODEL_VERSION",
    "telco-churn-model:latest"
)

# Set CAST_BOOL_TO_INT=1 jika MLflow menolak nilai true/false (error 400)
CAST_BOOL_TO_INT = os.getenv("CAST_BOOL_TO_INT", "0") == "1"

REQUEST_TIMEOUT = 10
ROLLING_WINDOW_SECONDS = 60

# Pemetaan label kelas agar dashboard lebih terbaca.
# Sesuaikan jika model mengeluarkan label yang berbeda.
CLASS_LABELS = {
    "0": "no_churn",
    "1": "churn",
}

# 1. DEFINISI METRIK PROMETHEUS
REQUEST_COUNT = Counter(
    "model_request",
    "Jumlah request inference berdasarkan HTTP status",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "model_request_latency_seconds",
    "Durasi request inference aktual dalam detik",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
)

PREDICTION_COUNT = Counter(
    "model_prediction",
    "Jumlah prediksi berdasarkan kelas hasil model",
    ["prediction_class"]
)

SUCCESS_COUNT = Counter(
    "model_success",
    "Jumlah request inference yang berhasil"
)

ERROR_COUNT = Counter(
    "model_error",
    "Jumlah request inference yang gagal"
)

CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "Penggunaan CPU aktual seluruh sistem dalam persen"
)

MEMORY_USAGE = Gauge(
    "system_memory_usage_percent",
    "Penggunaan memori aktual seluruh sistem dalam persen"
)

ERROR_RATE = Gauge(
    "model_error_rate",
    "Persentase request inference yang gagal"
)

THROUGHPUT = Gauge(
    "model_throughput_rps",
    "Throughput request inference selama 60 detik terakhir"
)

INPUT_DATA_SIZE = Gauge(
    "model_input_data_size_bytes",
    "Ukuran payload input inference aktual dalam byte"
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Informasi versi model yang dikonfigurasi",
    ["version"]
)

SUCCESS_RATIO = Gauge(
    "model_success_ratio",
    "Rasio request inference yang berhasil"
)

ACTIVE_REQUESTS = Gauge(
    "model_active_requests",
    "Jumlah request inference yang sedang diproses exporter"
)

INPUT_FEATURE_COUNT = Gauge(
    "model_input_feature_count",
    "Jumlah fitur yang dikirim ke model"
)

# [PERBAIKAN 2] Gauge status model: 1 = hidup, 0 = mati/gagal
MODEL_UP = Gauge(
    "model_up",
    "1 jika request inference terakhir berhasil, 0 jika gagal"
)

# 2. MEMUAT DATA INPUT SESUAI SKEMA MODEL
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}. "
        "Jalankan script dari direktori proyek yang benar "
        "atau atur environment variable DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

if "Churn" not in df.columns:
    raise ValueError(
        "Kolom target 'Churn' tidak ditemukan pada dataset."
    )

# Jangan kirim target sebagai input model.
X = df.drop(columns=["Churn"])

if X.empty:
    raise ValueError("Dataset tidak memiliki kolom fitur.")

if X.isnull().any().any():
    raise ValueError(
        "Fitur masih mengandung nilai kosong. "
        "Gunakan dataset preprocessing yang sama dengan training."
    )

# Opsional: ubah kolom boolean menjadi 0/1
if CAST_BOOL_TO_INT:
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

# Kirim satu baris setiap request, bergantian.
# Ini memakai data nyata, bukan membuat nilai fitur acak.
row_index = 0

INPUT_FEATURE_COUNT.set(len(X.columns))
MODEL_VERSION.labels(version=MODEL_VERSION_NAME).set(1)
MODEL_UP.set(0)  # nilai awal sebelum request pertama

# Statistik request aktual
recent_requests = deque()
total_requests = 0
successful_requests = 0
failed_requests = 0


# 3. UPDATE METRIK SISTEM DAN STATISTIK REQUEST
def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)


def update_request_metrics():
    now = time.monotonic()

    while (
        recent_requests
        and now - recent_requests[0] > ROLLING_WINDOW_SECONDS
    ):
        recent_requests.popleft()

    THROUGHPUT.set(
        len(recent_requests) / ROLLING_WINDOW_SECONDS
    )

    if total_requests > 0:
        ERROR_RATE.set(failed_requests / total_requests * 100)
        SUCCESS_RATIO.set(successful_requests / total_requests)
    else:
        ERROR_RATE.set(0)
        SUCCESS_RATIO.set(0)


def normalize_class(value):
    """[PERBAIKAN 5] Ubah 0/1 (int, float, atau string) menjadi label terbaca."""
    text = str(value).strip()

    # Tangani nilai seperti "1.0" -> "1"
    try:
        as_float = float(text)
        if as_float.is_integer():
            text = str(int(as_float))
    except ValueError:
        pass

    return CLASS_LABELS.get(text, text)


def parse_prediction(response_json):
    """Ambil prediksi dari format respons MLflow."""
    if isinstance(response_json, dict):
        if "predictions" not in response_json:
            raise ValueError(
                "Respons model tidak memiliki key 'predictions': "
                f"{response_json}"
            )
        predictions = response_json["predictions"]
    else:
        predictions = response_json

    if not isinstance(predictions, list) or not predictions:
        raise ValueError(
            f"Format prediksi tidak dikenali: {predictions}"
        )

    prediction = predictions[0]

    # Mendukung respons dengan satu prediksi berbentuk dictionary.
    if isinstance(prediction, dict):
        if len(prediction) == 1:
            prediction = next(iter(prediction.values()))
        else:
            raise ValueError(
                f"Format prediksi tidak dikenali: {prediction}"
            )

    return normalize_class(prediction)


# 4. INFERENCE NYATA KE MODEL DOCKER
def run_inference():
    global row_index
    global total_requests
    global successful_requests
    global failed_requests

    endpoint = "/invocations"
    method = "POST"

    # Ambil satu baris dari dataset preprocessing.
    row = X.iloc[[row_index % len(X)]]
    row_index += 1

    # Format dataframe_split didukung oleh MLflow Model Server.
    dataframe_split = json.loads(row.to_json(orient="split"))
    payload = {"dataframe_split": dataframe_split}

    body = json.dumps(payload).encode("utf-8")
    INPUT_DATA_SIZE.set(len(body))

    start = time.perf_counter()
    ACTIVE_REQUESTS.inc()

    # [PERBAIKAN 1] Default bukan "500" karena belum tentu ada respons HTTP.
    # Jika koneksi gagal (container mati/timeout), label ini yang dipakai.
    status = "connection_error"

    try:
        response = requests.post(
            MODEL_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )

        # Respons HTTP diterima, pakai status code yang sebenarnya.
        status = str(response.status_code)
        response.raise_for_status()

        response_json = response.json()
        prediction_class = parse_prediction(response_json)

        # Hanya menghitung prediksi yang benar-benar dikembalikan model.
        PREDICTION_COUNT.labels(
            prediction_class=prediction_class
        ).inc()

        SUCCESS_COUNT.inc()
        successful_requests += 1
        MODEL_UP.set(1)

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"status={status} "
            f"prediction={prediction_class}"
        )

    except (requests.RequestException, ValueError) as exc:
        # [PERBAIKAN 1] Status 200 tapi isi respons tidak valid
        # tidak boleh dicatat sebagai sukses.
        if status == "200":
            status = "invalid_response"

        ERROR_COUNT.inc()
        failed_requests += 1
        MODEL_UP.set(0)

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"status={status} "
            f"Inference gagal: {exc}"
        )

    finally:
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint=endpoint
        ).observe(elapsed)

        ACTIVE_REQUESTS.dec()

        total_requests += 1
        recent_requests.append(time.monotonic())

        update_request_metrics()
        update_system_metrics()


# 5. JALANKAN EXPORTER
if __name__ == "__main__":
    # Inisialisasi pengukuran CPU.
    psutil.cpu_percent(interval=None)

    start_http_server(EXPORTER_PORT)

    print(
        f"Prometheus exporter berjalan di "
        f"http://localhost:{EXPORTER_PORT}/metrics"
    )
    print(f"Model endpoint: {MODEL_URL}")
    print(f"Dataset input: {DATA_PATH}")

    while True:
        run_inference()
        time.sleep(1)