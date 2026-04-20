"""
==============================================================================
01_data_preprocessing.py — Veri Temizleme ve Feature Engineering Modulu
==============================================================================
Bu modul ham veri setini alir ve asagidaki Data Engineering adimlarini uygular:
  1. Eksik Veri Tespiti ve Doldurma (Missing Value Handling)
  2. Veri Tipi Duzeltmeleri (Data Type Correction)
  3. Tekrarli Kayit Kontrolu (Duplicate Removal)
  4. Outlier Analizi ve Raporlama (Outlier Detection — IQR Method)
  5. Feature Engineering (Yeni Oznitelik Turetme)
  6. Temizlenmis verinin CSV olarak kaydedilmesi

Kullanim:
    python 01_data_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
import logging
from config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, TENURE_BINS,
    TENURE_LABELS, SERVICE_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- 1. Veri Yukleme ---
def load_raw_data(path: str) -> pd.DataFrame:
    """Ham CSV dosyasini okur ve ilk bilgileri loglar."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ham veri dosyasi bulunamadi: {path}")
    df = pd.read_csv(path)
    logging.info(f"Ham veri yuklendi -- Satir: {df.shape[0]:,}, Sutun: {df.shape[1]}")
    return df


# --- 2. Veri Temizleme ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kapsamli veri temizligi:
      - customerID dusurulur (modele katkisi yok, gurultu kaynagi).
      - TotalCharges sutunundaki bosluklar NaN'a cevrilip 0 ile doldurulur.
      - Tekrarli satirlar kontrol edilir ve silinir.
    """
    df = df.copy()

    # customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
        logging.info("'customerID' sutunu dusuruldu (noise reduction).")

    # TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].replace(" ", np.nan), errors="coerce"
        )
        n_null = df["TotalCharges"].isnull().sum()
        if n_null > 0:
            logging.info(
                f"TotalCharges: {n_null} eksik deger tespit edildi -> 0 ile dolduruldu "
                f"(tenure=0 olan yeni aboneler)."
            )
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Tekrarli Satirlar
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df.drop_duplicates(inplace=True)
        logging.info(f"{n_dup} tekrarli satir silindi.")
    else:
        logging.info("Tekrarli satir bulunamadi -- veri temiz.")

    return df


# --- 3. Outlier (Aykiri Deger) Analizi ---
def report_outliers(df: pd.DataFrame) -> None:
    """
    Sayisal sutunlarda IQR (Interquartile Range) yontemiyle
    aykiri degerleri tespit eder ve raporlar.
    """
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        logging.info(
            f"Outlier Analizi -- {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f} "
            f"-> Aykiri Deger Sayisi: {n_outlier}"
        )


# --- 4. Feature Engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain bilgisine dayali yeni oznitelikler turetir:
      - tenure_group        : Abonelik suresini kategorize eder
      - avg_monthly_charge  : Toplam Ucret / Abonelik Suresi
      - total_services      : Toplam aktif servis sayisi
      - has_support         : Online guvenlik veya Teknik destek var mi?
      - is_autopay          : Otomatik odeme yapiyor mu?
    """
    df = df.copy()

    # Tenure Gruplama (Binning)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )

    # Ortalama Aylik Harcama
    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Toplam Aktif Hizmet Sayisi
    existing = [c for c in SERVICE_COLUMNS if c in df.columns]
    df["total_services"] = df[existing].apply(
        lambda row: sum(
            1 for v in row if v in ("Yes", "Fiber optic", "DSL")
        ),
        axis=1,
    )

    # Destek hizmeti var mi?
    df["has_support"] = (
        ((df.get("OnlineSecurity") == "Yes") | (df.get("TechSupport") == "Yes"))
        .astype(int)
    )

    # Otomatik odeme yapiyor mu?
    if "PaymentMethod" in df.columns:
        df["is_autopay"] = df["PaymentMethod"].apply(
            lambda x: 1 if "automatic" in str(x).lower() else 0
        )

    logging.info(
        f"Feature Engineering tamamlandi -- 5 yeni oznitelik turetildi. "
        f"Yeni boyut: {df.shape}"
    )
    return df


# --- 5. Ana Akis ---
def main() -> None:
    logging.info("=" * 60)
    logging.info("01_data_preprocessing -- Veri On Isleme Basladi")
    logging.info("=" * 60)

    df = load_raw_data(RAW_DATA_PATH)
    df = clean_data(df)
    report_outliers(df)
    df = engineer_features(df)

    df.to_csv(CLEAN_DATA_PATH, index=False)
    logging.info(f"Temizlenmis veri kaydedildi -> {CLEAN_DATA_PATH}")
    logging.info(f"Final boyut: {df.shape[0]:,} satir x {df.shape[1]} sutun")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
