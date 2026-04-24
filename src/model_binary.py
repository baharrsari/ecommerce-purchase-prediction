"""Binary purchase classifier.

Trains a Logistic Regression baseline and an XGBoost primary model on the
session-level dataset. Both handle the ~3.4% positive-class imbalance: the
baseline via ``class_weight='balanced'`` and XGBoost via ``scale_pos_weight``.

Metrics reported follow ``CLAUDE.md``: F1 and ROC-AUC as primary, with
precision/recall/confusion matrix as supporting. Accuracy is deliberately
omitted — a trivial "always 0" predictor would score ~96.6%.
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
    """Load features and binary target from the Phase-2 parquet."""
    df = pd.read_parquet(path)
    X = df[FEATURE_COLS].astype("float32")
    y = df["purchased"].astype("int8")
    return X, y


def split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split on the target."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Logistic Regression with standardized inputs and balanced class weights."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
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
    """XGBoost with ``scale_pos_weight`` set from the train-set ratio."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute the Phase-3 metric suite at the default 0.5 threshold."""
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
    """End-to-end Phase 3: load -> split -> train both -> evaluate -> persist.

    Both models are trained, but only the primary (XGBoost) is saved to
    ``model_path`` as the deployment artifact. The baseline is compared
    side-by-side in the metrics JSON.
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
    """Persist metrics, merging with any existing top-level keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(new_metrics)
    path.write_text(json.dumps(existing, indent=2))
