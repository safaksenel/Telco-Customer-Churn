"""
==============================================================================
config.py — Merkezi Konfigürasyon Dosyası
==============================================================================
Tüm sabit değerler (dosya yolları, hiperparametreler, random seed vb.)
burada tanımlanır. Hiçbir değer kodun içine gömülmez.
"""

import os

# ─── Dosya Yolları ───────────────────────────────────────────────────────────
RAW_DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_DATA_PATH = "cleaned_telco_churn.csv"
MODEL_PATH = "model.joblib"
RESULTS_PATH = "model_results.csv"
STATIC_DIR = "static"

# ─── Model Eğitim Parametreleri ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 6          # RandomizedSearchCV iterasyon sayısı

# ─── Feature Engineering Sabitleri ───────────────────────────────────────────
TENURE_BINS = [0, 12, 24, 48, 60, float("inf")]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61+"]

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# ─── Görselleştirme ──────────────────────────────────────────────────────────
PLOT_COLORS = ["#3498db", "#e74c3c"]
PLOT_PALETTE = {"No": "#3498db", "Yes": "#e74c3c"}
PLOT_DPI = 120

# ─── API ─────────────────────────────────────────────────────────────────────
API_TITLE = "Telco Customer Churn Prediction API"
API_VERSION = "3.0"
