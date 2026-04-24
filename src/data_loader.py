"""Memory-efficient loading of the raw monthly e-commerce CSV files.

The raw dataset contains ~16M events across four months. To stay within a
reasonable memory footprint we:

* read only the columns needed for modeling,
* use ``category`` dtype for low-cardinality string columns,
* parse ``event_time`` eagerly so downstream code can rely on datetime ops,
* expose an iterator so a caller can process one month at a time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

RAW_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

RAW_DTYPES = {
    "event_type": "category",
    "product_id": "int64",
    "category_id": "int64",
    "category_code": "string",
    "brand": "string",
    "price": "float32",
    "user_id": "int64",
    "user_session": "string",
}

MONTHLY_FILES = [
    "2019-Oct.csv",
    "2019-Dec.csv",
    "2020-Jan.csv",
    "2020-Feb.csv",
]


def load_month(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a single monthly CSV with memory-efficient dtypes.

    Args:
        path: Path to a monthly CSV file in ``data/raw/``.
        nrows: Optional row limit for smoke tests.

    Returns:
        DataFrame with ``event_time`` parsed as datetime.
    """
    df = pd.read_csv(
        path,
        usecols=RAW_COLUMNS,
        dtype=RAW_DTYPES,
        parse_dates=["event_time"],
        nrows=nrows,
    )
    return df


def iter_months(raw_dir: Path, nrows: int | None = None) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(month_name, dataframe)`` pairs for each monthly CSV.

    Processes one file at a time so the caller can aggregate to session level
    and discard the raw events before loading the next month.

    Args:
        raw_dir: Directory holding the monthly CSVs.
        nrows: Optional row limit per file for smoke tests.
    """
    for fname in MONTHLY_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing raw file: {fpath}")
        yield fname, load_month(fpath, nrows=nrows)


def load_all(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load and concatenate every monthly CSV.

    Only suitable for exploratory analysis or smoke tests: the full 4-month
    concat is memory-heavy. For the modeling pipeline, use :func:`iter_months`
    and aggregate per month.
    """
    frames = [df for _, df in iter_months(raw_dir, nrows=nrows)]
    return pd.concat(frames, ignore_index=True, copy=False)
