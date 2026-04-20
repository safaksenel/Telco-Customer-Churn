"""
==============================================================================
app.py — FastAPI REST API Servisi (Musteri Kaybi Tahmini)
==============================================================================
Egitilmis en iyi modeli (model.joblib) HTTP servisi olarak sunar.

Endpoint'ler:
  GET  /            -> Web arayuzu (static/index.html)
  POST /predict     -> Tekli musteri tahmini
  POST /predict/batch -> Toplu (batch) musteri tahmini
  GET  /model/info  -> Yuklu model hakkinda bilgi
  GET  /health      -> API saglik kontrolu
  GET  /docs        -> Swagger otomatik dokumantasyon

Kullanim:
    uvicorn app:app --reload
"""

import os
import datetime
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging

from config import (
    MODEL_PATH, STATIC_DIR, API_TITLE, API_VERSION,
    TENURE_BINS, TENURE_LABELS, SERVICE_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- FastAPI ---
app = FastAPI(
    title=API_TITLE,
    description="Musteri kayip riskini tahmin eden uctan uca ML modeli.",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Yukleme ---
model_pipeline = None
model_load_time = None


@app.on_event("startup")
def load_model():
    global model_pipeline, model_load_time
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            model_load_time = datetime.datetime.now().isoformat()
            logging.info(f"Model basariyla yuklendi: {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Model yuklenirken hata: {e}")
    else:
        logging.warning(
            "Model dosyasi bulunamadi! Once asagidaki komutlari calistirin:\n"
            "  python 01_data_preprocessing.py\n"
            "  python 02_model_training.py"
        )


# --- Veri Semasi ---
class CustomerData(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.0)
    TotalCharges: float = Field(..., ge=0, example=840.0)


# --- Feature Engineering (Egitim ile ayni) ---
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    01_data_preprocessing.py ile birebir ayni feature engineering adimlarini
    uygular. Pipeline uyumu icin kritiktir.
    """
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )

    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    existing = [c for c in SERVICE_COLUMNS if c in df.columns]
    df["total_services"] = df[existing].apply(
        lambda row: sum(1 for v in row if v in ("Yes", "Fiber optic", "DSL")),
        axis=1,
    )

    df["has_support"] = (
        ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")).astype(int)
    )

    df["is_autopay"] = df["PaymentMethod"].apply(
        lambda x: 1 if "automatic" in str(x).lower() else 0
    )

    return df


def predict_single(data: dict) -> dict:
    """Tek bir musteri icin tahmin yapar."""
    input_df = pd.DataFrame([data])
    input_df["TotalCharges"] = pd.to_numeric(
        input_df["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    input_df = apply_feature_engineering(input_df)

    prediction = model_pipeline.predict(input_df)[0]
    if hasattr(model_pipeline, "predict_proba"):
        probability = float(model_pipeline.predict_proba(input_df)[0][1])
    else:
        probability = 1.0 if prediction == 1 else 0.0

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
    }


# --- Statik Dosyalar ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/model/info")
def model_info():
    """Yuklu model hakkinda bilgi verir."""
    if model_pipeline is None:
        return {"error": "Model yuklenmemis."}

    steps = [step[0] for step in model_pipeline.steps]
    classifier_name = type(model_pipeline.named_steps["classifier"]).__name__

    return {
        "model_file": MODEL_PATH,
        "loaded_at": model_load_time,
        "pipeline_steps": steps,
        "classifier": classifier_name,
        "api_version": API_VERSION,
    }


# --- Tekli Tahmin ---
@app.post("/predict")
def predict_churn(data: CustomerData):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model henuz yuklenmedi. Lutfen modeli egitin.",
        )
    try:
        result = predict_single(data.dict())
        return result
    except Exception as e:
        logging.error(f"Tahmin hatasi: {e}")
        raise HTTPException(status_code=500, detail=f"Tahmin basarisiz: {e}")


# --- Toplu Tahmin (Batch) ---
@app.post("/predict/batch")
def predict_batch(customers: List[CustomerData]):
    """Birden fazla musteri icin toplu tahmin yapar."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model henuz yuklenmedi.",
        )
    try:
        predictions = []
        for i, customer in enumerate(customers):
            result = predict_single(customer.dict())
            result["customer_index"] = i
            predictions.append(result)
        return {
            "total_predictions": len(predictions),
            "results": predictions,
        }
    except Exception as e:
        logging.error(f"Batch tahmin hatasi: {e}")
        raise HTTPException(status_code=500, detail=f"Batch tahmin basarisiz: {e}")
