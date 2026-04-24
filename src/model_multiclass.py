"""Multi-class category classifier.

Predicts the session's ``main_cat`` ∈ {accessories, apparel, appliances,
furniture, stationery} from the 14 session-level features. Operates on
``sessions_cat.parquet`` — the subset of sessions with a recoverable category
(~131K rows).

Class balancing:

* **Baseline** (Logistic Regression, multinomial) uses ``class_weight='balanced'``.
* **Primary** (XGBoost ``multi:softprob``) has no native ``class_weight`` in
  multi-class mode, so we pass ``sample_weight`` computed by sklearn's
  ``compute_sample_weight('balanced', y)`` — this is the standard equivalent.

Metrics follow ``CLAUDE.md``: Macro F1 (primary), Weighted F1, and a
labels-sorted 5×5 confusion matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .feature_engineering import FEATURE_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and the string-valued multi-class target."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["main_cat"])
    X = df[FEATURE_COLS].astype("float32")
    y = df["main_cat"].astype("string")
    return X, y


def split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split on the class label."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Multinomial Logistic Regression with balanced class weights."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_primary(
    X_train: pd.DataFrame, y_train_enc: np.ndarray, n_classes: int
) -> XGBClassifier:
    """XGBoost ``multi:softprob`` with balanced sample weights."""
    sample_weight = compute_sample_weight("balanced", y_train_enc)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train_enc, sample_weight=sample_weight)
    return clf


def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    labels: list[str],
    encoded: bool = False,
) -> dict[str, Any]:
    """Compute macro/weighted F1 plus per-class report and confusion matrix.

    Args:
        labels: Class names in the canonical reporting order (used for cm and
            classification_report). For the encoded XGBoost case these must
            match the LabelEncoder's ``classes_``.
        encoded: If True, treat ``model`` as producing integer predictions and
            ``y_test`` as an integer array (XGBoost path). Otherwise treat both
            as string labels (LogReg pipeline path).
    """
    pred = model.predict(X_test)
    if encoded:
        pred_labels = np.asarray(labels)[pred]
        true_labels = np.asarray(labels)[np.asarray(y_test)]
    else:
        pred_labels = pred
        true_labels = np.asarray(y_test)

    macro = f1_score(true_labels, pred_labels, average="macro", labels=labels)
    weighted = f1_score(true_labels, pred_labels, average="weighted", labels=labels)
    per_class = f1_score(true_labels, pred_labels, average=None, labels=labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    report = classification_report(
        true_labels, pred_labels, labels=labels, zero_division=0, output_dict=True
    )

    return {
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class_f1": {lbl: float(v) for lbl, v in zip(labels, per_class)},
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "n_test": int(len(true_labels)),
    }


def run(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
) -> dict[str, dict[str, Any]]:
    """End-to-end Phase 4: load -> split -> train both -> evaluate -> persist."""
    X, y = load_dataset(data_path)
    print(f"[load ] sessions: {len(X):,}  classes: {y.nunique()}")
    print(f"[load ] class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"[split] train: {len(y_train):,}  test: {len(y_test):,}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    labels = list(le.classes_)
    print(f"[split] labels (alpha): {labels}")

    baseline = train_baseline(X_train, y_train)
    print("[base ] logistic regression fit")
    baseline_metrics = evaluate(baseline, X_test, y_test, labels, encoded=False)

    primary = train_primary(X_train, y_train_enc, n_classes=len(labels))
    print("[prim ] xgboost fit")
    primary_metrics = evaluate(primary, X_test, y_test_enc, labels, encoded=True)

    all_metrics = {
        "multiclass": {
            "labels": labels,
            "baseline_logreg": baseline_metrics,
            "primary_xgboost": primary_metrics,
        }
    }

    _merge_metrics_json(metrics_path, all_metrics)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": primary, "label_encoder": le}, model_path)

    return all_metrics


def _merge_metrics_json(path: Path, new_metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(new_metrics)
    path.write_text(json.dumps(existing, indent=2))
