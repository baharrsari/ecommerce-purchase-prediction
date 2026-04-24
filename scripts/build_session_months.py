"""Assign each user_session to its earliest month.

Raw CSVs are scanned column-by-column (only ``user_session``) so memory cost
is modest. A session appearing in two months (e.g. Dec 31 -> Jan 1) is
assigned to the earlier month, which is what we want for temporal validation.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data_loader import MONTHLY_FILES

RAW = Path("data/raw")
OUT = Path("data/processed/session_months.parquet")


def main() -> None:
    frames: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for fname in MONTHLY_FILES:
        month = fname.replace(".csv", "")
        print(f"  reading {fname} ...")
        col = pd.read_csv(
            RAW / fname,
            usecols=["user_session"],
            dtype={"user_session": "string"},
        )["user_session"].dropna().drop_duplicates()
        frames.append(pd.DataFrame({"user_session": col.values, "month": month}))

    combined = pd.concat(frames, ignore_index=True)
    order = {fname.replace(".csv", ""): i for i, fname in enumerate(MONTHLY_FILES)}
    combined["order"] = combined["month"].map(order).astype("int8")
    combined = (
        combined.sort_values("order", kind="stable")
        .drop_duplicates("user_session", keep="first")[["user_session", "month"]]
        .reset_index(drop=True)
    )

    print(f"\n  total unique sessions: {len(combined):,}")
    print(combined["month"].value_counts().sort_index())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(f"\n  saved -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
