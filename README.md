# Telco Musteri Kaybi (Churn) Tahmini ve REST API

*Veri Bilimi Challenge - Uctan Uca Profesyonel Cozum*

Bu proje, Telekom sektorundeki musteri kayiplarini (churn) tespit etmeyi amaclayan uctan uca (End-to-End) bir Makine Ogrenmesi ekosistemidir.

---

## Proje Mimarisi

```
Telco-Customer-Churn/
|-- config.py                         # Merkezi konfigurasyon (dosya yollari, sabitler)
|-- 00_EDA_and_Presentation.ipynb     # Kesifsel Veri Analizi ve Gorsellestirme
|-- 01_data_preprocessing.py          # Veri Temizleme ve Feature Engineering
|-- 02_model_training.py              # Coklu Model Egitimi + Hiperparametre Optimizasyonu
|-- app.py                            # FastAPI REST API Servisi
|-- static/
|   |-- index.html                    # Premium Dark Mode Web Arayuzu
|   |-- confusion_matrix.png          # Confusion Matrix gorseli
|   |-- roc_curves.png                # ROC Curve karsilastirma gorseli
|   |-- feature_importance.png        # Feature Importance gorseli
|   |-- 01_churn_distribution.png     # EDA gorselleri...
|-- requirements.txt
|-- Dockerfile
|-- README.md
```

---

## Uygulanan Veri Bilimi Teknikleri

### Veri Temizleme (Data Cleaning)
- Missing value handling (TotalCharges bosluk temizligi)
- Duplicate removal (22 tekrarli satir silindi)
- Data type correction (string -> float donusumleri)
- Outlier detection ve raporlama (IQR yontemi)

### Feature Engineering
- **tenure_group**: Binning ile sure kategorileri
- **avg_monthly_charge**: Turetilmis ortalama ucret
- **total_services**: Toplam aktif hizmet sayisi
- **has_support**: Destek hizmeti flag'i
- **is_autopay**: Otomatik odeme flag'i

### Veri On Isleme (Pipeline)
- StandardScaler (sayisal degiskenler)
- OneHotEncoder (kategorik degiskenler)
- SMOTE (sinif dengeleme)

### Kesifsel Veri Analizi (EDA)
- Hedef degisken dagilimi (Pie + Bar)
- Sozlesme tipi analizi (Countplot + Stacked Bar)
- Tenure yogunluk analizi (KDE + Boxplot)
- Ucret dagilim analizi (KDE)
- Korelasyon matrisi (Heatmap)
- 6 kategorik degisken churn etki analizi
- Chi-Square bagimsizlik testi (istatistiksel anlamlilik)

### Model Egitimi ve Karsilastirma
10 farkli model + Voting Classifier test edildi:

| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Voting Classifier** | **0.78** | **0.57** | **0.72** | **0.64** | **0.8422** |
| 2 | Gradient Boosting | 0.78 | 0.57 | 0.72 | 0.64 | 0.8417 |
| 3 | CatBoost | 0.79 | 0.58 | 0.70 | 0.63 | 0.8413 |
| 4 | XGBoost | 0.78 | 0.56 | 0.72 | 0.63 | 0.8411 |
| 5 | LightGBM | 0.79 | 0.61 | 0.60 | 0.60 | 0.8406 |
| 6 | AdaBoost | 0.78 | 0.57 | 0.68 | 0.62 | 0.8403 |
| 7 | Logistic Regression | 0.75 | 0.51 | 0.77 | 0.61 | 0.8394 |
| 8 | Random Forest | 0.78 | 0.58 | 0.68 | 0.62 | 0.8390 |
| 9 | SVC | 0.72 | 0.48 | 0.81 | 0.60 | 0.8367 |
| 10 | KNN | 0.70 | 0.45 | 0.69 | 0.55 | 0.7526 |
| 11 | Decision Tree | 0.72 | 0.47 | 0.49 | 0.48 | 0.6781 |

### Gorsellestirmeler (Otomatik Uretilen)
- **Confusion Matrix**: Modelin hatalarini gorsel olarak gosterir
- **ROC Curve**: 10 modelin AUC karsilastirmasi tek grafikte
- **Feature Importance**: En etkili 15 oznitelik siralamasini gosterir

### Model Optimizasyonu
- RandomizedSearchCV ile hiperparametre aramasi
- StratifiedKFold (k=5) cross-validation
- SMOTE ile class imbalance cozumu
- Ensemble learning (Soft Voting - En iyi 3 model)

---

## Kurulum ve Calistirma

### 1. Sanal Ortam ve Bagimliliklar
```bash
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

### 2. Veri On Isleme
```bash
python 01_data_preprocessing.py
```

### 3. Model Egitimi
```bash
python 02_model_training.py
```

### 4. API Baslatma
```bash
uvicorn app:app --reload
```
Tarayicinizda `http://127.0.0.1:8000` adresine gidebilirsiniz.

### 5. EDA Raporu
```bash
jupyter notebook 00_EDA_and_Presentation.ipynb
```

---

## API Endpoint'leri

| Method | Endpoint | Aciklama |
| :--- | :--- | :--- |
| GET | `/` | Premium Dark Mode Web Arayuzu |
| POST | `/predict` | Tekli musteri churn tahmini |
| POST | `/predict/batch` | Toplu musteri tahmini (JSON listesi) |
| GET | `/model/info` | Yuklu model bilgisi |
| GET | `/health` | API saglik kontrolu |
| GET | `/docs` | Swagger otomatik dokumantasyon |

### Ornek Istek
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0, "TotalCharges": 840.0
  }'
```

---

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Teknolojiler
- **Dil:** Python 3.11
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn
- **API:** FastAPI, Uvicorn, Pydantic
- **Gorsellestirme:** Matplotlib, Seaborn
- **Istatistik:** SciPy (Chi-Square testi)
- **Container:** Docker