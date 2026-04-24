"""Generate Phase 5 evaluation plots for both models."""
from __future__ import annotations

import time
from pathlib import Path

from src.evaluate import evaluate_binary, evaluate_multiclass


def main() -> None:
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    bin_metrics = evaluate_binary(
        data_path=Path("data/processed/sessions_all.parquet"),
        model_path=Path("models/binary_model.pkl"),
        figures_dir=figures_dir,
    )
    t1 = time.perf_counter()
    print(f"[bin  ] ROC_AUC={bin_metrics['roc_auc']:.4f}  "
          f"AP={bin_metrics['average_precision']:.4f}  ({t1 - t0:.1f}s)")

    evaluate_multiclass(
        data_path=Path("data/processed/sessions_cat.parquet"),
        model_path=Path("models/multiclass_model.pkl"),
        figures_dir=figures_dir,
    )
    t2 = time.perf_counter()
    print(f"[multi] plots saved  ({t2 - t1:.1f}s)")

    print("[save ] figures under", figures_dir)
    for p in sorted(figures_dir.glob("*.png")):
        print(f"        - {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"[done ] total: {t2 - t0:.1f}s")


if __name__ == "__main__":
    main()
