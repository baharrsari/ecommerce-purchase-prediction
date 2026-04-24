"""Phase 5 — evaluation plots.

Generates publication-ready figures for the IEEE report:

* Binary model: ROC curve, Precision-Recall curve, confusion-matrix heatmap,
  XGBoost feature importance.
* Multi-class model: raw and row-normalized confusion-matrix heatmaps,
  XGBoost feature importance.

Trained models are reloaded from disk and evaluated on the same test split
used during training (same ``random_state=42``, same stratification), so the
figures stay consistent with ``results/metrics.json``.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless — write PNGs, don't pop windows.

import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .feature_engineering import FEATURE_COLS
from .model_binary import (
    load_dataset as load_binary_dataset,
    split as split_binary,
)
from .model_multiclass import (
    load_dataset as load_multi_dataset,
    split as split_multi,
)

_DEFAULT_DPI = 150


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Binary Purchase Classifier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)
    return float(roc_auc)


def plot_pr(y_true: np.ndarray, y_proba: np.ndarray, out_path: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"XGBoost (AP = {ap:.3f})", linewidth=2)
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Base rate = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Binary Purchase Classifier")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)
    return float(ap)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    normalize: bool = False,
    cmap: str = "Blues",
) -> None:
    mat = cm.astype("float64")
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 1.0)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = mat.max() / 2.0 if mat.max() > 0 else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j] if normalize else int(cm[i, j])
            ax.text(
                j, i, format(value, fmt),
                ha="center", va="center",
                color="white" if mat[i, j] > thresh else "black",
                fontsize=9,
            )
    _save(fig, out_path)


def plot_feature_importance(
    model,
    feature_names: list[str],
    out_path: Path,
    title: str,
    top_n: int | None = None,
) -> None:
    importance = np.asarray(model.feature_importances_)
    order = np.argsort(importance)
    if top_n is not None:
        order = order[-top_n:]

    fig, ax = plt.subplots(figsize=(7, max(4, len(order) * 0.35)))
    ax.barh(
        [feature_names[i] for i in order],
        importance[order],
        color="steelblue",
    )
    ax.set_xlabel("Gain importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, out_path)


def evaluate_binary(
    data_path: Path,
    model_path: Path,
    figures_dir: Path,
) -> dict[str, float]:
    X, y = load_binary_dataset(data_path)
    _, X_test, _, y_test = split_binary(X, y)

    model = joblib.load(model_path)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    y_np = y_test.to_numpy()
    roc_auc = plot_roc(y_np, proba, figures_dir / "binary_roc.png")
    ap = plot_pr(y_np, proba, figures_dir / "binary_pr.png")

    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(
        cm,
        labels=["no_purchase", "purchase"],
        out_path=figures_dir / "binary_confusion.png",
        title="Binary Confusion Matrix (threshold=0.5)",
    )
    plot_confusion_matrix(
        cm,
        labels=["no_purchase", "purchase"],
        out_path=figures_dir / "binary_confusion_normalized.png",
        title="Binary Confusion Matrix - row-normalized",
        normalize=True,
    )

    plot_feature_importance(
        model, FEATURE_COLS,
        out_path=figures_dir / "binary_importance.png",
        title="XGBoost Feature Importance - Binary",
    )
    return {"roc_auc": roc_auc, "average_precision": ap}


def evaluate_multiclass(
    data_path: Path,
    model_path: Path,
    figures_dir: Path,
) -> None:
    X, y = load_multi_dataset(data_path)
    _, X_test, _, y_test = split_multi(X, y)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    le = artifact["label_encoder"]
    labels = list(le.classes_)

    y_test_enc = le.transform(y_test)
    pred_enc = model.predict(X_test)
    cm = confusion_matrix(y_test_enc, pred_enc, labels=list(range(len(labels))))

    plot_confusion_matrix(
        cm, labels=labels,
        out_path=figures_dir / "multiclass_confusion.png",
        title="Multi-class Confusion Matrix - counts",
    )
    plot_confusion_matrix(
        cm, labels=labels,
        out_path=figures_dir / "multiclass_confusion_normalized.png",
        title="Multi-class Confusion Matrix - row-normalized",
        normalize=True,
    )
    plot_feature_importance(
        model, FEATURE_COLS,
        out_path=figures_dir / "multiclass_importance.png",
        title="XGBoost Feature Importance - Multi-class",
    )
