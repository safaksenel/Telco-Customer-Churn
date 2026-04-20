from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Telco Customer Churn API")

# Modeli Yükleme
MODEL_PATH = "model.joblib"
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("Model başarıyla yüklendi.")
    else:
        print("Uyarı: Model dosyası bulunamadı. Lütfen önce train.py'yi çalıştırın.")

# Veri Modeli
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Arayüz için static klasörü
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model yüklenmemiş.")
    
    # Gelen veriyi DataFrame'e çevir
    input_data = pd.DataFrame([data.dict()])
    
    try:
        # Tahmin yap
        prediction = model_pipeline.predict(input_data)[0]
        probability = model_pipeline.predict_proba(input_data)[0][1]
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
