"""Multi-class kategori sınıflandırıcısını eğit ve değerlendir.."""
from __future__ import annotations

import time
from pathlib import Path

from src.model_multiclass import run


def _fmt_cm(cm: dict) -> str:
    labels = cm["labels"]
    matrix = cm["matrix"]
    width = max(len(l) for l in labels) + 2
    num_w = max(6, max(len(str(v)) for row in matrix for v in row) + 1)
    header = " " * width + "".join(f"{l:>{num_w}}" for l in labels)
    lines = [header]
    for lbl, row in zip(labels, matrix):
        lines.append(f"{lbl:<{width}}" + "".join(f"{v:>{num_w}}" for v in row))
    return "\n".join(lines)


def _fmt_summary(m: dict) -> str:
    per = "  ".join(f"{k[:4]}={v:.3f}" for k, v in m["per_class_f1"].items())
    return (
        f"macro_f1={m['macro_f1']:.4f}  "
        f"weighted_f1={m['weighted_f1']:.4f}  |  per-class: {per}"
    )


def main() -> None:
    data_path = Path("data/processed/sessions_cat.parquet")
    model_path = Path("models/multiclass_model.pkl")
    metrics_path = Path("results/metrics.json")

    t0 = time.perf_counter()
    metrics = run(data_path, model_path, metrics_path)
    t1 = time.perf_counter()

    mc = metrics["multiclass"]
    print("[eval ] baseline :", _fmt_summary(mc["baseline_logreg"]))
    print("[eval ] primary  :", _fmt_summary(mc["primary_xgboost"]))
    print("[eval ] primary confusion matrix (rows=true, cols=pred):")
    print(_fmt_cm(mc["primary_xgboost"]["confusion_matrix"]))
    print(f"[save ] model   -> {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[save ] metrics -> {metrics_path}")
    print(f"[done ] total: {t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
