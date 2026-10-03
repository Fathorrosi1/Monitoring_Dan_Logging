# prometheus_exporter.py
import time
import random
import requests
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Definisi Metrik Prometheus (10+ metrik)
REQUEST_COUNT = Counter(
    'model_request_total',
    'Total request ke model',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Latency request ke model',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_COUNT = Counter(
    'model_prediction_total',
    'Total prediksi yang dilakukan',
    ['prediction_class']
)

CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Penggunaan CPU (%)'
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'Penggunaan memory (%)'
)

ERROR_RATE = Gauge(
    'model_error_rate',
    'Error rate model (%)'
)

THROUGHPUT = Gauge(
    'model_throughput_rps',
    'Throughput model (request/detik)'
)

INPUT_DATA_SIZE = Gauge(
    'model_input_data_size',
    'Ukuran input data (byte)'
)

MODEL_VERSION = Gauge(
    'model_version_info',
    'Informasi versi model',
    ['version']
)

SUCCESS_RATIO = Gauge(
    'model_success_ratio',
    'Rasio request sukses'
)

ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Jumlah request aktif'
)

# Simulasi Traffic
MODEL_URL = 'http://localhost:5001/invocations'

def simulate_request():
    """Simulasi request & track metrik."""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        # Simulasi response (tanpa hit model real, biar tidak error)
        time.sleep(random.uniform(0.01, 0.5))
        status = random.choices([200, 400, 500], weights=[0.9, 0.07, 0.03])[0]
        prediction_class = random.choices(['churn', 'no_churn'], weights=[0.3, 0.7])[0]
        
        # Update metrik
        REQUEST_COUNT.labels(method='POST', endpoint='/invocations', status=status).inc()
        PREDICTION_COUNT.labels(prediction_class=prediction_class).inc()
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/invocations').observe(latency)
        
        CPU_USAGE.set(random.uniform(20, 80))
        MEMORY_USAGE.set(random.uniform(30, 70))
        ERROR_RATE.set(random.uniform(0, 5))
        THROUGHPUT.set(random.uniform(10, 100))
        INPUT_DATA_SIZE.set(random.uniform(500, 2000))
        SUCCESS_RATIO.set(random.uniform(0.9, 1.0))
        MODEL_VERSION.labels(version='v1.0').set(1)
        
        print(f"[{time.strftime('%H:%M:%S')}] status={status} "
              f"latency={latency:.3f}s pred={prediction_class}")
        
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/invocations', status=500).inc()
        print(f"Error: {e}")
    finally:
        ACTIVE_REQUESTS.dec()

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter jalan di http://localhost:8000/metrics")
    
    while True:
        simulate_request()
        time.sleep(1)