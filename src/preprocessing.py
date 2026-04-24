"""Event-level → session-level aggregation.

Two datasets are produced:

* ``sessions_all`` — one row per ``user_session`` with the binary ``purchased``
  target. Used by Model 1.
* ``sessions_cat`` — the subset of ``sessions_all`` where a ``main_cat`` could
  be recovered via the category-id → category-code mapping. Used by Model 2.

Category recovery is the critical step: raw ``category_code`` is ~98.3% NULL
while ``category_id`` is almost always present, so we forward-fill the code
from any row where both were observed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import MONTHLY_FILES, iter_months

# Classes with <5 occurrences are dropped (see CLAUDE.md).
DROP_CATEGORIES = {"sport"}


def build_category_map(df: pd.DataFrame) -> dict[int, str]:
    """Construct a ``category_id -> category_code`` mapping.

    A small number of ``category_id`` values appear with more than one
    ``category_code``; we keep the first observed code, which is stable in
    practice for this dataset.
    """
    non_null = df.dropna(subset=["category_code"])[["category_id", "category_code"]]
    non_null = non_null.drop_duplicates(subset=["category_id"])
    return dict(zip(non_null["category_id"], non_null["category_code"]))


def fill_main_cat(df: pd.DataFrame, cat_map: dict[int, str]) -> pd.DataFrame:
    """Fill ``main_cat`` on an event-level frame using the id→code map."""
    filled = df["category_id"].map(cat_map)
    df = df.assign(
        category_filled=filled,
        main_cat=filled.str.split(".").str[0],
    )
    return df


def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event rows to one row per ``user_session``.

    Features are computed over **non-purchase** events only. Including purchase
    events in the feature aggregations leaks the target: ``n_events`` would
    strictly exceed ``n_view + n_cart + n_remove`` for purchased sessions,
    price/brand/product stats would absorb purchase-event values, and
    ``session_duration_sec`` would stretch to ``ts_max`` of the purchase. The
    target ``purchased`` and session-level ``main_cat`` are still computed
    over all events.
    """
    df = df.assign(
        is_view=(df["event_type"] == "view").astype("int32"),
        is_cart=(df["event_type"] == "cart").astype("int32"),
        is_remove=(df["event_type"] == "remove_from_cart").astype("int32"),
        is_purchase=(df["event_type"] == "purchase").astype("int32"),
        has_brand_flag=df["brand"].notna().astype("int32"),
    )

    pre = df[df["event_type"] != "purchase"]
    pre_grouped = pre.groupby("user_session", sort=False, observed=True)

    features = pre_grouped.agg(
        user_id=("user_id", "first"),
        n_view=("is_view", "sum"),
        n_cart=("is_cart", "sum"),
        n_remove=("is_remove", "sum"),
        n_events=("event_type", "size"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        price_std=("price", "std"),
        unique_products=("product_id", "nunique"),
        unique_categories=("category_id", "nunique"),
        has_brand=("has_brand_flag", "max"),
        ts_min=("event_time", "min"),
        ts_max=("event_time", "max"),
    )

    all_grouped = df.groupby("user_session", sort=False, observed=True)
    targets = all_grouped.agg(
        purchased=("is_purchase", "max"),
        user_id_fallback=("user_id", "first"),
    )
    main_cat_mode = all_grouped["main_cat"].agg(
        lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else np.nan
    )

    # Outer join preserves purchase-only sessions (all features -> 0).
    sessions = features.join(targets, how="outer")
    sessions["main_cat"] = main_cat_mode
    sessions["user_id"] = sessions["user_id"].fillna(sessions["user_id_fallback"])
    sessions = sessions.drop(columns=["user_id_fallback"])

    sessions["purchased"] = (sessions["purchased"] > 0).astype("int8")
    sessions["session_duration_sec"] = (
        (sessions["ts_max"] - sessions["ts_min"]).dt.total_seconds()
    ).astype("float32")
    sessions = sessions.drop(columns=["ts_min", "ts_max"])

    int_cols = [
        "n_view", "n_cart", "n_remove", "n_events",
        "unique_products", "unique_categories",
    ]
    for col in int_cols:
        sessions[col] = sessions[col].fillna(0).astype("int32")
    sessions["has_brand"] = sessions["has_brand"].fillna(0).astype("int8")
    sessions["user_id"] = sessions["user_id"].astype("int64")

    float_cols = ["avg_price", "max_price", "min_price", "price_std", "session_duration_sec"]
    for col in float_cols:
        sessions[col] = sessions[col].fillna(0.0).astype("float32")

    return sessions.reset_index()


def process_month(df: pd.DataFrame, cat_map: dict[int, str]) -> pd.DataFrame:
    """Event-level frame for one month → session-level frame."""
    df = fill_main_cat(df, cat_map)
    return aggregate_sessions(df)


def build_sessions(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """Build the full session-level dataset across every monthly file.

    Strategy:
      1. First pass: scan each month to collect ``category_id -> category_code``
         pairs so the map is complete before aggregation.
      2. Second pass: aggregate each month to sessions using the global map,
         then concatenate the (much smaller) session frames.

    A session that spans a month boundary is rare given the dataset's session
    granularity, but if it occurs the two halves are merged in
    :func:`_merge_cross_month_sessions`.
    """
    cat_map: dict[int, str] = {}
    for fname in MONTHLY_FILES:
        slim = pd.read_csv(
            raw_dir / fname,
            usecols=["category_id", "category_code"],
            dtype={"category_id": "int64", "category_code": "string"},
            nrows=nrows,
        )
        cat_map.update(build_category_map(slim))
        del slim

    session_frames: list[pd.DataFrame] = []
    for _, df in iter_months(raw_dir, nrows=nrows):
        session_frames.append(process_month(df, cat_map))
        del df

    sessions = pd.concat(session_frames, ignore_index=True, copy=False)
    sessions = _merge_cross_month_sessions(sessions)
    sessions = sessions[~sessions["main_cat"].isin(DROP_CATEGORIES)]
    return sessions


def _merge_cross_month_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate ``user_session`` rows that fell across months.

    Counts are summed, price stats recomputed conservatively, and the session
    is marked purchased if either fragment was.
    """
    dup_mask = sessions["user_session"].duplicated(keep=False)
    if not dup_mask.any():
        return sessions

    clean = sessions[~dup_mask]
    dupes = sessions[dup_mask]

    grouped = dupes.groupby("user_session", sort=False, observed=True)
    merged = grouped.agg(
        user_id=("user_id", "first"),
        n_view=("n_view", "sum"),
        n_cart=("n_cart", "sum"),
        n_remove=("n_remove", "sum"),
        n_events=("n_events", "sum"),
        avg_price=("avg_price", "mean"),
        max_price=("max_price", "max"),
        min_price=("min_price", "min"),
        price_std=("price_std", "mean"),
        unique_products=("unique_products", "sum"),
        unique_categories=("unique_categories", "sum"),
        has_brand=("has_brand", "max"),
        main_cat=("main_cat", lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else np.nan),
        purchased=("purchased", "max"),
        session_duration_sec=("session_duration_sec", "sum"),
    ).reset_index()

    return pd.concat([clean, merged], ignore_index=True, copy=False)


def save_processed(
    sessions: pd.DataFrame,
    out_dir: Path,
    all_name: str = "sessions_all.parquet",
    cat_name: str = "sessions_cat.parquet",
) -> tuple[Path, Path]:
    """Persist the two modeling datasets to Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / all_name
    cat_path = out_dir / cat_name

    sessions.to_parquet(all_path, index=False)
    sessions.dropna(subset=["main_cat"]).to_parquet(cat_path, index=False)

    return all_path, cat_path
