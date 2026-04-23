"""
==============================================================================
02_model_training.py — Coklu Model Egitimi ve Hiperparametre Optimizasyonu
==============================================================================
Bu modul temizlenmis veriyi alir ve su adimlari uygular:
  1. Encode ve Scaling (OneHotEncoder + StandardScaler)
  2. Veri Dengeleme (SMOTE)
  3. 10 Farkli Model Egitimi + Hiperparametre Optimizasyonu (RandomizedSearchCV)
  4. Voting Classifier (En iyi 3 modelin Ensemble'i)
  5. Kapsamli Degerlendirme (Accuracy, F1, Recall, Precision, ROC-AUC)
  6. Confusion Matrix + ROC Curve + Feature Importance Gorselleri
  7. En iyi modelin model.joblib olarak kaydedilmesi

Kullanim:
    python 02_model_training.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib
matplotlib.use("Agg")  # GUI olmadan gorsel uretmek icin
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    recall_score, precision_score, roc_auc_score,
    confusion_matrix, roc_curve,
)

# Modeller
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    CLEAN_DATA_PATH, MODEL_PATH, RESULTS_PATH, STATIC_DIR,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, N_ITER_SEARCH,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

sns.set_theme(style="whitegrid")


# --- 1. Veri Yukleme ---
def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} bulunamadi! Once '01_data_preprocessing.py' calistirin."
        )
    df = pd.read_csv(path)
    logging.info(f"Temizlenmis veri yuklendi -- {df.shape}")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop("Churn", axis=1)
    return X, y


# --- 2. Preprocessor ---
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        exclude=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    logging.info(f"Sayisal oznitelikler ({len(numeric_features)}): {numeric_features}")
    logging.info(f"Kategorik oznitelikler ({len(categorical_features)}): {categorical_features}")

    num_pipe = SkPipeline([("scaler", StandardScaler())])
    cat_pipe = SkPipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


# --- 3. Model Tanimlari ---
def get_models_and_params() -> dict:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"classifier__C": [0.01, 0.1, 1, 10], "classifier__penalty": ["l2"]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [3, 5, 7, 11], "classifier__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"classifier__max_depth": [3, 5, 10, None], "classifier__min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"classifier__n_estimators": [100, 200, 300], "classifier__max_depth": [10, 20, None]},
        ),
        "SVC": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["rbf", "linear"]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=RANDOM_STATE),
            {"classifier__n_estimators": [50, 100, 200], "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 5, 7],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__num_leaves": [31, 50, 70],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, allow_writing_files=False),
            {
                "classifier__iterations": [100, 200, 300],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__depth": [4, 6, 8],
            },
        ),
    }


# --- 4. Gorsellestime Fonksiyonlari ---
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Confusion Matrix gorseli olusturur ve static/ klasorune kaydeder."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Kalan (0)", "Kayip (1)"],
        yticklabels=["Kalan (0)", "Kayip (1)"],
    )
    ax.set_xlabel("Tahmin Edilen", fontsize=12)
    ax.set_ylabel("Gercek Deger", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()
    logging.info("Confusion Matrix gorseli kaydedildi.")


def plot_roc_curves(roc_data: dict, best_name: str) -> None:
    """Tum modellerin ROC egrilerini tek grafikte cizer."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, (fpr, tpr, auc_val) in roc_data.items():
        lw = 3 if name == best_name else 1.2
        alpha = 1.0 if name == best_name else 0.6
        ax.plot(fpr, tpr, lw=lw, alpha=alpha, label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Karsilastirmasi", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "roc_curves.png"), dpi=120)
    plt.close()
    logging.info("ROC Curve gorseli kaydedildi.")


def plot_feature_importance(pipeline, preprocessor, X: pd.DataFrame, model_name: str) -> None:
    """En iyi modelin feature importance gradigini cizer."""
    classifier = pipeline.named_steps["classifier"]

    # Voting Classifier ise ilk estimator'un importance'ini al
    if isinstance(classifier, VotingClassifier):
        for name, est in classifier.estimators:
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                model_name = f"Voting ({name})"
                break
        else:
            logging.info("Voting Classifier icinde feature_importances_ bulunamadi.")
            return
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logging.info(f"{model_name} feature importance desteklemiyor.")
        return

    # Feature isimlerini al
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out())
    except Exception:
        cat_names = []

    num_names = preprocessor.transformers_[0][2]
    if isinstance(num_names, list):
        feature_names = num_names + cat_names
    else:
        feature_names = list(num_names) + cat_names

    # Uzunluk uyumsuzlugu kontrolu
    if len(feature_names) != len(importances):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="#3498db", edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top 15 Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    logging.info("Feature Importance gorseli kaydedildi.")


# --- 5. Ana Egitim Dongusu ---
def main() -> None:
    start = time.time()
    logging.info("=" * 65)
    logging.info("02_model_training -- Model Egitim Sureci Basladi")
    logging.info("=" * 65)

    X, y = load_data(CLEAN_DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Train: {X_train.shape[0]:,} satir | Test: {X_test.shape[0]:,} satir")

    preprocessor = build_preprocessor(X)
    models = get_models_and_params()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    trained = {}
    roc_data = {}

    for name, (model, params) in models.items():
        logging.info(f">> Egitiliyor: {name}")

        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", model),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            n_iter=min(N_ITER_SEARCH, len(params)),
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            error_score="raise",
        )

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            logging.warning(f"  X {name} basarisiz: {e}")
            continue

        best = search.best_estimator_
        trained[name] = best

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        # ROC Curve verisi
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr, auc)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "Best_Params": str(search.best_params_),
        })

        logging.info(
            f"  OK {name} -- AUC: {auc:.4f} | F1: {f1:.4f} | "
            f"Recall: {rec:.4f} | Accuracy: {acc:.4f}"
        )

    # --- Sonuc Tablosu ---
    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    logging.info(f"\nModel sonuclari kaydedildi -> {RESULTS_PATH}")

    print("\n" + "=" * 80)
    print("MODEL KARSILASTIRMA TABLOSU (ROC-AUC'ye Gore Sirali)")
    print("=" * 80)
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]].to_string(index=False))
    print("=" * 80)

    # --- Voting Classifier ---
    top3_names = results_df.head(3)["Model"].tolist()
    logging.info(f"\nVoting Classifier kuruluyor -- En iyi 3: {top3_names}")

    voting_estimators = []
    for n in top3_names:
        clf = trained[n].named_steps["classifier"]
        voting_estimators.append((n, clf))

    voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft")
    voting_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", voting_clf),
    ])

    try:
        voting_pipe.fit(X_train, y_train)
        y_pred_v = voting_pipe.predict(X_test)
        y_prob_v = voting_pipe.predict_proba(X_test)[:, 1]
        auc_v = roc_auc_score(y_test, y_prob_v)
        f1_v = f1_score(y_test, y_pred_v)
        logging.info(f"  Voting Classifier -- AUC: {auc_v:.4f} | F1: {f1_v:.4f}")
        trained["Voting Classifier"] = voting_pipe

        fpr_v, tpr_v, _ = roc_curve(y_test, y_prob_v)
        roc_data["Voting Classifier"] = (fpr_v, tpr_v, auc_v)

        results.append({
            "Model": "Voting Classifier",
            "Accuracy": round(accuracy_score(y_test, y_pred_v), 4),
            "Precision": round(precision_score(y_test, y_pred_v), 4),
            "Recall": round(recall_score(y_test, y_pred_v), 4),
            "F1_Score": round(f1_v, 4),
            "ROC_AUC": round(auc_v, 4),
            "Best_Params": f"Ensemble of {top3_names}",
        })
    except Exception as e:
        logging.warning(f"Voting Classifier basarisiz: {e}")

    # --- Final Secim ---
    all_results = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    all_results.to_csv(RESULTS_PATH, index=False)

    best_name = all_results.iloc[0]["Model"]
    best_auc = all_results.iloc[0]["ROC_AUC"]
    best_pipeline = trained[best_name]

    print(f"\n*** SAMPIYON MODEL: {best_name}  (ROC-AUC: {best_auc}) ***")

    y_final = best_pipeline.predict(X_test)
    print("\n--- Siniflandirma Raporu ---")
    print(classification_report(y_test, y_final, target_names=["Kalan (0)", "Kayip (1)"]))

    # --- Gorseller ---
    os.makedirs(STATIC_DIR, exist_ok=True)
    plot_confusion_matrix(y_test, y_final, best_name)
    plot_roc_curves(roc_data, best_name)

    # Feature importance icin preprocessor'u fit etmemiz lazim
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    plot_feature_importance(best_pipeline, preprocessor_fitted, X, best_name)

    # Model Kaydet
    joblib.dump(best_pipeline, MODEL_PATH)
    logging.info(f"En iyi model kaydedildi -> {MODEL_PATH}")

    elapsed = time.time() - start
    logging.info(f"Toplam egitim suresi: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika)")
    logging.info("=" * 65)

if __name__ == "__main__":
    main()
