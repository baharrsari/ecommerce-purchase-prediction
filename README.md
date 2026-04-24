# E-Ticaret Satın Alma Tahmini ve Kategori Sınıflandırması

İki aşamalı bir makine öğrenmesi pipeline'ı: bir e-ticaret oturumunun (i) satın alma ile sonuçlanıp sonuçlanmayacağını ve (ii) sonuçlanacaksa hangi ürün kategorisiyle ilişkili olduğunu tahmin eder.

**Tek satır özet**: Session-level davranışsal feature'lardan iki aşamalı satın alma ve kategori tahmini.

---

## Proje Özeti

- **Problem 1 (Binary)**: Oturum satın alma ile sonuçlanacak mı? (~%3.3 pozitif sınıf — ciddi imbalance)
- **Problem 2 (Multi-class)**: Hangi ana kategori? (`appliances`, `furniture`, `apparel`, `stationery`, `accessories`)
- **Veri**: Ekim 2019 – Şubat 2020, ~16M event → **3.6M oturum**
- **Yaklaşım**: Event-level → session-level aggregation, 14 davranışsal feature, XGBoost primary + LogReg baseline, `class_weight='balanced'` / `scale_pos_weight` ile imbalance yönetimi
- **Doğrulama**: 5-fold cross-validation + temporal split (Train: Oct+Dec+Jan, Test: Feb)

---

## Sonuçlar

| Görev | Model | Primary Metrik | Random Split | Temporal Split |
|---|---|---|---:|---:|
| Binary (purchase) | XGBoost | ROC-AUC | **0.941** | 0.939 |
| Binary (purchase) | XGBoost | F1 | 0.297 | 0.283 |
| Multi-class (category) | XGBoost | Macro F1 | **0.855** | 0.819 |
| Multi-class (category) | XGBoost | Weighted F1 | 0.878 | 0.844 |

5-fold CV standart sapmaları: binary F1 ±0.001, multi Macro F1 ±0.001 → modeller stabil.

Detaylı rapor: [`paper/draft.md`](paper/draft.md). Tüm metrikler: [`results/metrics.json`](results/metrics.json). Figürler: `results/figures/`.

---

## Dataset

[Kaggle — mkechinov/ecommerce-events-history-in-cosmetics-shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop)

4 aylık CSV dosyasını indirip `data/raw/` altına yerleştirin:

```
data/raw/
├── 2019-Oct.csv
├── 2019-Dec.csv
├── 2020-Jan.csv
└── 2020-Feb.csv
```

> Not: Adıyla "cosmetics" geçse de veri setinde aslında `appliances`, `furniture`, `apparel`, `stationery`, `accessories` kategorileri yer alır.

---

### Notebook'lar

4 sunum notebook'u (`notebooks/` altında), çıktılar önceden embed edilmiş:

1. `01_exploration.ipynb` — Veri keşfi + feature dağılımları
2. `02_binary_model.ipynb` — Binary classifier training + eval
3. `03_multiclass_model.ipynb` — Multi-class classifier training + eval
4. `04_temporal_validation.ipynb` — Random vs temporal split karşılaştırması

---

## Klasör Yapısı

```
ecommerce-purchase-prediction/
├── README.md
├── CLAUDE.md               # Proje spec'i (design decisions, constraints)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                # (gitignored) Kaggle CSV'leri
│   └── processed/          # (gitignored) Parquet çıktıları
│
├── src/                    # Yeniden kullanılabilir modüller
│   ├── data_loader.py          # Memory-efficient CSV loader
│   ├── preprocessing.py        # Event → session aggregation + leak-free feature scope
│   ├── feature_engineering.py  # Ratio features + NaN handling
│   ├── model_binary.py         # Binary training + evaluation
│   ├── model_multiclass.py     # Multi-class training + evaluation
│   └── evaluate.py             # Plotting utilities
│
├── scripts/                # Orkestrasyon script'leri
│   ├── run_phase1.py ... run_phase5.py
│   ├── build_session_months.py
│   ├── verify.py               # 19-check verification suite
│   ├── build_notebooks.py      # 01-03 notebook generator
│   ├── build_notebook_04.py    # 04 notebook generator
│   └── paper_stats.py
│
├── notebooks/              # Inline-output presentation notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_binary_model.ipynb
│   ├── 03_multiclass_model.ipynb
│   └── 04_temporal_validation.ipynb
│
├── models/                 # (gitignored) .pkl model dosyaları
│
├── results/
│   ├── metrics.json            # Tüm metrikler (random + temporal)
│   └── figures/                # PNG figürler (paper'da kullanılan)
│       └── temporal/
│
└── paper/
    └── draft.md            # IEEE-style paper draft (TR)
```

---

## Önemli Tasarım Kararları

- **Event → session aggregation** zorunlu: raw event-level feature'lar kullanıcı niyetini temsil etmez.
- **Target leakage düzeltmesi**: Tüm feature'lar **sadece non-purchase event'ler** üzerinden hesaplanır. Başlangıçta `n_events` tüm event'leri sayıyordu; bu, purchase sinyalini deterministik olarak sızdırıyordu (ilk deneyde LogReg F1=0.9999 sonucu bu sızıntıdan). Detaylar: paper §III.C.
- **SMOTE kullanılmadı**: Data hacmi yeterli, sentetik örnekler test-sızıntı riski taşıyor, ağırlıklandırma daha şeffaf.
- **Accuracy raporlanmaz**: %3.3 pozitif sınıfta "hep 0 tahmin et" modeli %96.7 accuracy alır ama değersizdir.

---

## Yazar

Bahar Sarımehmetoğlu
