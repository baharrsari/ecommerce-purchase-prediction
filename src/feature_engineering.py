"""Session-level feature derivation.

The raw aggregates produced in :mod:`preprocessing` already cover 12 of the 14
features listed in ``CLAUDE.md``. This module adds the two ratio features and
handles NaNs that arise when a session has only a single event (``price_std``)
or missing prices.

Ratio semantics:

* ``cart_to_view_ratio`` — ``n_cart / n_view``. A session with zero views is
  treated as 0: no browsing means the ratio is undefined, and 0 encodes
  "no view-to-cart conversion signal" which is the behaviorally neutral
  assumption.
* ``remove_to_cart_ratio`` — ``n_remove / max(n_cart, 1)``. Using ``max(·, 1)``
  as the denominator means a session with no cart activity and no removals
  gets 0, while a session with removals but no carts (rare) collapses to
  ``n_remove`` — still monotonic in hesitation signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "n_view",
    "n_cart",
    "n_remove",
    "n_events",
    "cart_to_view_ratio",
    "remove_to_cart_ratio",
    "avg_price",
    "max_price",
    "min_price",
    "price_std",
    "unique_products",
    "unique_categories",
    "has_brand",
    "session_duration_sec",
]


def add_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``sessions`` augmented with derived features.

    The input is expected to be the output of
    :func:`preprocessing.aggregate_sessions` (or its cross-month merged
    equivalent). Non-feature columns (``user_session``, ``user_id``,
    ``main_cat``, ``purchased``) are preserved.
    """
    df = sessions.copy()

    n_view = df["n_view"].to_numpy()
    n_cart = df["n_cart"].to_numpy()
    n_remove = df["n_remove"].to_numpy()

    df["cart_to_view_ratio"] = np.where(
        n_view > 0, n_cart / np.maximum(n_view, 1), 0.0
    ).astype("float32")

    df["remove_to_cart_ratio"] = (
        n_remove / np.maximum(n_cart, 1)
    ).astype("float32")

    # price_std is NaN for single-event sessions; 0 encodes "no variance".
    df["price_std"] = df["price_std"].fillna(0.0).astype("float32")

    # All-NaN price columns (no event had a price) -> 0. Rare but defensive.
    for col in ("avg_price", "max_price", "min_price"):
        df[col] = df[col].fillna(0.0).astype("float32")

    # Guard session_duration_sec against NaN (single-event sessions produce 0
    # naturally, but floats can carry NaN from upstream merges).
    df["session_duration_sec"] = df["session_duration_sec"].fillna(0.0).astype("float32")

    return df


def select_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Return only the 14 modeling feature columns in canonical order."""
    return sessions[FEATURE_COLS]
