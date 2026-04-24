"""Run Phase 1 end-to-end: raw CSVs -> session parquet files."""
from __future__ import annotations

import time
from pathlib import Path

from src.preprocessing import build_sessions, save_processed


def main() -> None:
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")

    t0 = time.perf_counter()
    sessions = build_sessions(raw_dir)
    t1 = time.perf_counter()

    print(f"[build] sessions rows: {len(sessions):,}  ({t1 - t0:.1f}s)")
    print(f"[build] purchase rate: {sessions['purchased'].mean():.4f}")
    print(f"[build] sessions with main_cat: {sessions['main_cat'].notna().sum():,}")
    print("[build] main_cat distribution:")
    print(sessions["main_cat"].value_counts(dropna=False))

    all_path, cat_path = save_processed(sessions, out_dir)
    t2 = time.perf_counter()

    print(f"[save ] {all_path} ({all_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[save ] {cat_path} ({cat_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[done ] total: {t2 - t0:.1f}s")


if __name__ == "__main__":
    main()
