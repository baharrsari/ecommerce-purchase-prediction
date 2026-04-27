"""Ham aylık e-ticaret CSV dosyalarının bellek açısından verimli şekilde yüklenmesi.”

Ham veri seti dört ayda yaklaşık 16 milyon olay içerir. Makul bir bellek kullanımı sağlamak için:

Modelleme için gerekli olan sütunları okuruz.
Az sayıda farklı değere sahip metin sütunları için category veri tipini kullanırız.
event_time alanını baştan tarih formatına çeviririz, böylece sonraki işlemler bunu kolayca kullanabilir.
Veriyi tek seferde değil, ay ay işleyebilmek için bir iterator (yineleyici) sunarız.

Kısaca: Büyük veriyi belleği yormadan, parça parça ve verimli şekilde yüklüyoruz.
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
    """Tek bir aylık CSV dosyasını bellek dostu (optimize edilmiş veri tipleriyle) yükler.
        Gerekirse sadece ilk n satırı yükleyerek hızlı test yapılabilir (nrows)
        event_time sütunu otomatik olarak datetime formatına çevrilir
        Sonuç olarak temizlenmiş bir DataFrame döner
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
    """Her aylık CSV dosyasını tek tek işler ve her biri için:
    (ay_adı, veri_seti) şeklinde çıktı üretir.
    Dosyaları teker teker yükler (bellek tasarrufu için)
    Her ay işlendiğinde sonuç döndürülür, sonra o veri atılır
    Böylece bir sonraki ay yüklenebilir
    nrows varsa sadece ilk N satır alınır (test amaçlı)
    """
    for fname in MONTHLY_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing raw file: {fpath}")
        yield fname, load_month(fpath, nrows=nrows)


def load_all(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """Tüm aylık CSV dosyalarını yükleyip tek bir büyük DataFrame halinde birleştirir.
    Sadece keşif analizi veya küçük testler için uygundur (bellek tüketimi yüksek)
    Modelleme için önerilmez
    Modelleme aşamasında bunun yerine iter_months kullanılıp her ay ayrı ayrı işlenmelidir
    """
    frames = [df for _, df in iter_months(raw_dir, nrows=nrows)]
    return pd.concat(frames, ignore_index=True, copy=False)
