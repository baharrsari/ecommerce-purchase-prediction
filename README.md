# 🛒 E-Commerce Purchase Prediction and Category Classification

A two-stage machine learning pipeline that predicts (i) whether an e-commerce session will result in a purchase, and (ii) if so, which product category it will be associated with.

**One-line summary**: Two-stage purchase and category prediction from session-level behavioral features.

---

## 📋 Project Overview

- **Problem 1 (Binary)**: Will the session end in a purchase? (~3.3% positive class — severe imbalance)
- **Problem 2 (Multi-class)**: Which main category? (`appliances`, `furniture`, `apparel`, `stationery`, `accessories`)
- **Data**: October 2019 – February 2020, ~16M events → **3.6M sessions**
- **Approach**: Event-level → session-level aggregation, 14 behavioral features, XGBoost primary + LogReg baseline, imbalance handling via `class_weight='balanced'` / `scale_pos_weight`
- **Validation**: 5-fold cross-validation + temporal split (Train: Oct+Dec+Jan, Test: Feb)

---

## 📊 Results

| Task | Model | Primary Metric | Random Split | Temporal Split |
|---|---|---|---:|---:|
| Binary (purchase) | XGBoost | ROC-AUC | **0.941** | 0.939 |
| Binary (purchase) | XGBoost | F1 | 0.297 | 0.283 |
| Multi-class (category) | XGBoost | Macro F1 | **0.855** | 0.819 |
| Multi-class (category) | XGBoost | Weighted F1 | 0.878 | 0.844 |

5-fold CV standard deviations: binary F1 ±0.001, multi Macro F1 ±0.001 → models are stable.

Detailed report: [`paper/draft.md`](paper/draft.md). All metrics: [`results/metrics.json`](results/metrics.json). Figures: `results/figures/`.

---

## 📦 Dataset

[Kaggle — mkechinov/ecommerce-events-history-in-cosmetics-shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop)

Download the 4 monthly CSV files and place them under `data/raw/`:

```
data/raw/
├── 2019-Oct.csv
├── 2019-Dec.csv
├── 2020-Jan.csv
└── 2020-Feb.csv
```

> Note: Although the name says "cosmetics", the dataset actually contains the categories `appliances`, `furniture`, `apparel`, `stationery`, and `accessories`.

---

## 🚀 How to Run

**Requirements**: Python 3.10+.

```bash
# Dependencies
pip install -r requirements.txt

# Phase 1 — Raw CSV → session-level parquet (~15 min)
python -m scripts.run_phase1

# Phase 2 — Feature engineering (~3 sec)
python -m scripts.run_phase2

# Phase 3 — Binary classifier (~30 sec)
python -m scripts.run_phase3

# Phase 4 — Multi-class classifier (~50 sec)
python -m scripts.run_phase4

# Phase 5 — Evaluation plots (~15 sec)
python -m scripts.run_phase5

# (Optional) Side table for temporal validation
python -m scripts.build_session_months

# (Optional) Comprehensive verification (leakage + CV + edge cases, ~14 min)
python -X utf8 -m scripts.verify
```

### 📓 Notebooks

4 presentation notebooks (under `notebooks/`), with outputs pre-embedded:

1. `01_exploration.ipynb` — Data exploration + feature distributions
2. `02_binary_model.ipynb` — Binary classifier training + evaluation
3. `03_multiclass_model.ipynb` — Multi-class classifier training + evaluation
4. `04_temporal_validation.ipynb` — Random vs temporal split comparison

---

## 📁 Folder Structure

```
ecommerce-purchase-prediction/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                # (gitignored) Kaggle CSV files
│   └── processed/          # (gitignored) Parquet outputs
│
├── src/                    # Reusable modules
│   ├── data_loader.py          # Memory-efficient CSV loader
│   ├── preprocessing.py        # Event → session aggregation + leak-free feature scope
│   ├── feature_engineering.py  # Ratio features + NaN handling
│   ├── model_binary.py         # Binary training + evaluation
│   ├── model_multiclass.py     # Multi-class training + evaluation
│   └── evaluate.py             # Plotting utilities
│
├── scripts/                # Orchestration scripts
│   ├── run_phase1.py ... run_phase5.py
│   ├── build_session_months.py
│   └── verify.py               # 19-check verification suite
│
├── notebooks/              # Inline-output presentation notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_binary_model.ipynb
│   ├── 03_multiclass_model.ipynb
│   └── 04_temporal_validation.ipynb
│
├── models/                 # (gitignored) .pkl model files
│
├── results/
│   ├── metrics.json            # All metrics (random + temporal)
│   └── figures/                # PNG figures (used in the paper)
│       └── temporal/
│
├── paper/
│   └── draft.md            # IEEE-style paper draft
│
└── bahar_sunum_eticaret.pptx   # Presentation file
```

---

## 🧠 Key Design Decisions

- **Event → session aggregation** is mandatory: raw event-level features do not represent user intent.
- **Target leakage fix**: All features are computed **only from non-purchase events**. Initially, `n_events` was counting all events, which deterministically leaked the purchase signal (the LogReg F1=0.9999 result in the first experiment was caused by this leak). Details: paper §III.C.
- **SMOTE was not used**: Data volume is sufficient, synthetic samples carry test-leakage risk, and weighting is more transparent.
- **Accuracy is not reported**: With a 3.3% positive class, an "always predict 0" model gets 96.7% accuracy but is worthless.

---

## 📦 Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
joblib>=1.3
pyarrow>=14.0
jupyter
```

---

## 👤 Author

Bahar Sarımehmetoğlu · [sarimehmetoglubahar@gmail.com](mailto:sarimehmetoglubahar@gmail.com)
