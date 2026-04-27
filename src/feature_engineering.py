"""
Oturum davranışlarını daha iyi yakalamak için iki oran özellik ekleniyor ve özel durumlarda veri bozulmadan anlamlı 
şekilde 0 veya uygun değerlerle temsil ediliyor.

cart_to_view_ratio — n_cart / n_view
Bir oturumda hiç görüntüleme yoksa 0 olarak kabul edilir. Çünkü hiç gezinme olmaması,
sepete ekleme dönüşüm sinyali olmadığını ifade eder ve nötr (tarafsız) bir davranış olarak 0 ile temsil edilir.


remove_to_cart_ratio — n_remove / max(n_cart, 1)
Paydada max(·, 1) kullanılır. Bu sayede hiç sepete ekleme ve hiç çıkarma olmayan oturumlar 0 olur.
Sepete ekleme olmadan çıkarma olması (nadir bir durum) ise doğrudan n_remove değerine indirgenir ve yine “kararsızlık/tereddüt”
sinyali açısından monoton (artış yönlü) kalır.

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
    """sessions verisinin bir kopyasını döndürür ve buna türetilmiş (derived) özellikler ekler.

Girdi, aggregate_sessions (veya aylar arası birleştirilmiş hali) çıktısıdır
user_session, user_id, main_cat, purchased sütunları aynen korunur
Diğer veriler bozulmadan sadece yeni feature’lar eklenir
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
    """Sadece modelde kullanılacak 14 özellik sütununu alır ve bunları standart (önceden belirlenmiş) sırada döndürür."""
    return sessions[FEATURE_COLS]
