"""Olay (event) seviyesinden oturum (session) seviyesine toplulaştırma.”

İki veri seti üretilir:

sessions_all — Her user_session için tek satır içerir ve ikili (binary) purchased hedefini barındırır. Model 1 için kullanılır.
sessions_cat — sessions_all içinden, kategori-id → kategori-kodu eşlemesi ile main_cat değeri geri çıkarılabilen oturumların alt
kümesidir. Model 2 için kullanılır.

Kategori geri kazanımı (category recovery) kritik bir adımdır:
Ham veride category_code alanı yaklaşık %98.3 oranında boş (NULL) iken, category_id neredeyse her zaman mevcuttur. Bu yüzden,
her ikisinin de birlikte gözlemlendiği satırlardan category_code bilgisi ileri doldurma (forward-fill) yöntemiyle diğer eksik
satırlara aktarılır.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import MONTHLY_FILES, iter_months

# Classes with <5 occurrences are dropped (see CLAUDE.md).
DROP_CATEGORIES = {"sport"}


def build_category_map(df: pd.DataFrame) -> dict[int, str]:
    """Her category_id için tek bir category_code seçiliyor.
        Birden fazla varsa, ilk görülen kullanılıyor.
    """
    non_null = df.dropna(subset=["category_code"])[["category_id", "category_code"]]
    non_null = non_null.drop_duplicates(subset=["category_id"])
    return dict(zip(non_null["category_id"], non_null["category_code"]))


def fill_main_cat(df: pd.DataFrame, cat_map: dict[int, str]) -> pd.DataFrame:
    """Her olay (event) satırında main_cat alanını, id→code eşlemesini kullanarak doldur.
"""
    filled = df["category_id"].map(cat_map)
    df = df.assign(
        category_filled=filled,
        main_cat=filled.str.split(".").str[0],
    )
    return df


def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Her **user_session** için event satırlarını tek bir satırda topla.

Özellikler (**features**) yalnızca **satın alma (purchase) olmayan event’ler** üzerinden hesaplanır.
Çünkü purchase event’leri dahil etmek hedef bilgiyi sızdırır (data leakage):

* **n_events**, satın alınan oturumlarda **n_view + n_cart + n_remove** toplamından büyük olur
* Fiyat/marka/ürün istatistikleri purchase değerlerini de içerir
* **session_duration_sec**, satın alma zamanına kadar uzar

Ama şu ikisi **tüm event’ler** kullanılarak hesaplanır:

* **purchased** (hedef değişken)
* Oturum seviyesindeki **main_cat**

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
    """Kısaca:

Tüm aylık verilerden **oturum (session) bazlı tek bir veri seti oluşturuluyor**.

* Önce tüm aylardan **id → kod eşleşmeleri** toplanıyor
* Sonra her ay **oturum bazına indiriliyor ve birleştiriliyor**
* Eğer bir oturum iki aya bölünmüşse, **sonradan birleştiriliyor**


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
    """Kısaca:

Aylara bölünmüş aynı **user_session** kayıtları **tek satırda birleştirilir**.

* Sayılar (count) **toplanır**
* Fiyat istatistikleri **yeniden ve temkinli şekilde hesaplanır**
* Parçalardan biri bile satın aldıysa, oturum **“purchased” olarak işaretlenir**

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
