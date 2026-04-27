"""Aşama 1 (Phase 1) parquet dosyalarını yükle, türetilmiş özellikleri ekle sonra da modeli 
eğitmek için kullanılacak son veri setlerini kaydet."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.feature_engineering import FEATURE_COLS, add_features


def main() -> None:
    in_dir = Path("data/processed")
    all_in = in_dir / "sessions_all.parquet"
    cat_in = in_dir / "sessions_cat.parquet"

    t0 = time.perf_counter()
    sessions_all = pd.read_parquet(all_in)
    sessions_cat = pd.read_parquet(cat_in)
    t1 = time.perf_counter()
    print(f"[load ] all: {len(sessions_all):,}  cat: {len(sessions_cat):,}  ({t1 - t0:.1f}s)")

    sessions_all = add_features(sessions_all)
    sessions_cat = add_features(sessions_cat)
    t2 = time.perf_counter()
    print(f"[feat ] added ratios + NaN fills  ({t2 - t1:.1f}s)")

    missing = [c for c in FEATURE_COLS if c not in sessions_all.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    nan_counts = sessions_all[FEATURE_COLS].isna().sum()
    if nan_counts.any():
        print("[check] NaNs per feature:")
        print(nan_counts[nan_counts > 0])
    else:
        print("[check] no NaNs in feature columns: OK")

    print("[check] feature summary (sessions_all):")
    print(sessions_all[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]])

    sessions_all.to_parquet(all_in, index=False)
    sessions_cat.to_parquet(cat_in, index=False)
    t3 = time.perf_counter()
    print(f"[save ] {all_in} ({all_in.stat().st_size / 1e6:.1f} MB)")
    print(f"[save ] {cat_in} ({cat_in.stat().st_size / 1e6:.1f} MB)")
    print(f"[done ] total: {t3 - t0:.1f}s")


if __name__ == "__main__":
    main()
