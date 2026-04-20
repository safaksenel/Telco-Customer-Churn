import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def main():
    print("Veri yükleniyor...")
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(file_path):
        print(f"Hata: {file_path} bulunamadı!")
        return

    df = pd.read_csv(file_path)

    # 1. Veri Temizleme
    print("Veri temizleniyor...")
    # customerID model için gereksiz
    df.drop('customerID', axis=1, inplace=True)

    # TotalCharges içindeki boşlukları (space) temizle ve sayısal tipe dönüştür
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    
    # NaN değerleri o sütunun ortalaması veya medyanı ile doldur
    df['TotalCharges'] = df['TotalCharges'].fillna(0) # Yeni müşteriler için 0 diyebiliriz

    # 2. X ve y ayırımı
    X = df.drop('Churn', axis=1)
    # Churn sütununu 1 ve 0'a çevirelim (Yes=1, No=0)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # Kategorik ve Sayısal sütunları belirleme
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # 3. Ön İşleme Pipeline'ı
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Model Pipeline'ı
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Eğitim ve Test Setlerine Ayırma
    print("Model eğitiliyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli Eğit
    model_pipeline.fit(X_train, y_train)

    # 6. Değerlendirme
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    # 7. Modeli Kaydet
    model_filename = 'model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"Model başarıyla '{model_filename}' olarak kaydedildi.")

if __name__ == "__main__":
    main()
