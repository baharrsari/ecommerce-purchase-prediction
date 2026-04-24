"""Train and evaluate the binary purchase classifier."""
from __future__ import annotations

import time
from pathlib import Path

from src.model_binary import run


def _fmt(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    return (
        f"f1={metrics['f1']:.4f}  "
        f"roc_auc={metrics['roc_auc']:.4f}  "
        f"precision={metrics['precision']:.4f}  "
        f"recall={metrics['recall']:.4f}  "
        f"cm(tn,fp,fn,tp)=({cm['tn']},{cm['fp']},{cm['fn']},{cm['tp']})"
    )


def main() -> None:
    data_path = Path("data/processed/sessions_all.parquet")
    model_path = Path("models/binary_model.pkl")
    metrics_path = Path("results/metrics.json")

    t0 = time.perf_counter()
    metrics = run(data_path, model_path, metrics_path)
    t1 = time.perf_counter()

    print("[eval ] baseline :", _fmt(metrics["binary"]["baseline_logreg"]))
    print("[eval ] primary  :", _fmt(metrics["binary"]["primary_xgboost"]))
    print(f"[save ] model -> {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[save ] metrics -> {metrics_path}")
    print(f"[done ] total: {t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
