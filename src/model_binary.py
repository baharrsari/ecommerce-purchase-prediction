"""Binary satın alma sınıflandırıcısı.”

Oturum seviyesindeki veri seti üzerinde bir lojistik regresyon (baseline) ve bir
XGBoost (ana model) eğitilir. Veri setinde yaklaşık %3.4 pozitif sınıf dengesizliği
vardır (satın alma yapanlar azdır). Bu dengesizlik:

Baseline modelde class_weight='balanced' ile,
XGBoost modelinde ise scale_pos_weight ile

dengelenir.

Ana metrikler olarak F1 skoru ve ROC-AUC kullanılır;
ek olarak precision (kesinlik), recall (duyarlılık) ve confusion matrix raporlanır.

Accuracy (doğruluk) özellikle kullanılmaz, çünkü veri dengesiz olduğu için “her zaman 0 tahmin eden” basit bir
model bile yaklaşık %96.6 doğruluk elde edebilir ve bu yanıltıcı olur.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .feature_engineering import FEATURE_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class BinaryResult:
    name: str
    metrics: dict[str, Any]
    model: Any


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Phase-2 parquet dosyasından özellikleri (features) ve ikili hedef değişkeni (target) yükler.
"""
    df = pd.read_parquet(path)
    X = df[FEATURE_COLS].astype("float32")
    y = df["purchased"].astype("int8")
    return X, y


def split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hedef değişkeni koruyarak veriyi %80 eğitim / %20 test olacak şekilde böler."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Veriler ölçeklendirilerek (standardize edilerek) Logistic Regression modeli eğitilir ve sınıflar
        dengesizse bunu dengelemek için class weight (balanced) kullanılır."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", #<---
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_primary(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """XGBoost modeli eğitilir ve sınıflar dengesizse bunu düzeltmek için pozitif sınıfa ağırlık 
        (scale_pos_weight) eğitim verisindeki oranlara göre ayarlanır."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1) #29

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw, #<---
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Phase-3 performans metrikleri, 0.5 eşik değeri (threshold) kullanılarak hesaplanır."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "n_test": int(len(y_test)),
        "n_test_positive": int(y_test.sum()),
    }


def run(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
) -> dict[str, dict[str, Any]]:
    """Phase 3 sürecini baştan sona çalıştırır:

Veriyi yükler
Eğitim/test olarak böler
Hem Logistic Regression hem XGBoost modellerini eğitir
Performanslarını değerlendirir ve karşılaştırır
Sonuçları kaydeder

Ama sadece ana model olan XGBoost kaydedilir (deployment için).
Logistic Regression ise sadece kıyaslama amacıyla kullanılır.
    """
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split(X, y)
    print(
        f"[split] train: {len(y_train):,}  test: {len(y_test):,}  "
        f"train_pos_rate: {y_train.mean():.4f}"
    )

    baseline = train_baseline(X_train, y_train)
    print("[base ] logistic regression fit")
    baseline_metrics = evaluate(baseline, X_test, y_test)

    primary = train_primary(X_train, y_train)
    print("[prim ] xgboost fit")
    primary_metrics = evaluate(primary, X_test, y_test)

    all_metrics = {
        "binary": {
            "baseline_logreg": baseline_metrics,
            "primary_xgboost": primary_metrics,
        }
    }

    _merge_metrics_json(metrics_path, all_metrics)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(primary, model_path)

    return all_metrics


def _merge_metrics_json(path: Path, new_metrics: dict[str, Any]) -> None:
    """Hesaplanan metrikler dosyaya kaydedilir ve eğer dosyada zaten başka bilgiler varsa,
        üst seviye anahtarlarla birleştirilir (üzerine yazmadan eklenir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(new_metrics)
    path.write_text(json.dumps(existing, indent=2))
