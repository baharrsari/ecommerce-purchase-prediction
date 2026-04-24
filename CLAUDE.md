# Project: E-Commerce Purchase Behavior & Category Prediction

## 🎯 Project Overview

Two-stage machine learning project analyzing e-commerce user session behavior to predict:

1. **Binary Classification**: Will the user make a purchase? (Yes/No)
2. **Multi-class Classification**: If yes, from which product category?

**Academic Context**: This project will be presented in a scientific-paper format (IEEE) to a jury of domain experts. All design decisions must be defensible with academic rigor.

**One-line summary**:
> Predicting whether an e-commerce user will complete a purchase and, if so, from which product category, based on session-level behavioral features.

---

## 📊 Dataset

**Source**: Kaggle — `mkechinov/ecommerce-events-history-in-cosmetics-shop`

> ⚠️ Despite its name, the dataset is NOT cosmetics-only. Actual categories are: `appliances`, `furniture`, `apparel`, `stationery`, `accessories`.

**Files** (4 monthly CSVs, place under `data/raw/`):

- `2019-Oct.csv`
- `2019-Dec.csv`
- `2020-Jan.csv`
- `2020-Feb.csv`

### Raw Schema (event-level)

| Column | Type | Notes |
|---|---|---|
| `event_time` | datetime | Event timestamp |
| `event_type` | categorical | `view` / `cart` / `remove_from_cart` / `purchase` |
| `product_id` | int | Product identifier |
| `category_id` | int | Category identifier (numeric hash) |
| `category_code` | string | Hierarchical code (e.g., `appliances.environment.vacuum`) — **~98.3% NULL** in raw |
| `brand` | string | Brand name — many NULLs |
| `price` | float | Product price |
| `user_id` | int | User identifier |
| `user_session` | string | Session identifier (grouping key) |

### Key Statistics (across all 4 months)

| Metric | Value |
|---|---|
| Total events | ~16M |
| Total unique sessions | ~3.6M |
| Session-level purchase rate | **~3.4%** (severe imbalance) |
| Sessions with non-null category | ~131K |
| Purchase sessions with category | ~9,100 |

### Main-Category Distribution (session-level, after filling)

| Category | Share |
|---|---|
| appliances | ~42% |
| furniture | ~18% |
| apparel | ~17% |
| stationery | ~14% |
| accessories | ~9% |

> `sport` appears <5 times in total — **drop it.**

---

## 🔑 Key Design Decisions

### 1. Event → Session Aggregation (MANDATORY)

Every row in the final modeling dataset must represent **one `user_session`**, not one event. Use `groupby('user_session')` with appropriate aggregations.

### 2. Category Filling

Raw `category_code` is ~98.3% NULL, but `category_id` is almost always present. Build a `category_id → category_code` mapping from the non-null rows, then forward-fill. This is critical — without this step, multi-class data drops to ~1.7% usable.

```python
mapping = df.dropna(subset=['category_code'])[['category_id', 'category_code']].drop_duplicates()
cat_map = dict(zip(mapping['category_id'], mapping['category_code']))
df['category_filled'] = df['category_id'].map(cat_map)
df['main_cat'] = df['category_filled'].str.split('.').str[0]
```

### 3. Target Variables

- **Binary target** (`purchased`): `1` if ANY event in the session is `purchase`, else `0`.
- **Multi-class target** (`main_cat`): Mode of `main_cat` within the session.

### 4. Subset Selection

- **Model 1 (Binary)** uses **ALL sessions** (~3.6M). No category requirement.
- **Model 2 (Multi-class)** uses **only sessions with a non-null `main_cat`** (~131K).

### 5. Class Imbalance Strategy

- **Both models** → `class_weight='balanced'`.
- **SMOTE is NOT used**: data volume is sufficient; simpler pipeline; avoids risk of test-set leakage.

### 6. Evaluation Metrics

- **Binary** → primary: **F1 + ROC-AUC**. Also report Precision, Recall, Confusion Matrix. **Never report Accuracy alone.**
- **Multi-class** → primary: **Macro F1**. Also Weighted F1 and a 5×5 Confusion Matrix.

---

## 🔧 Feature Engineering

Session-level features to derive from raw events:

| Feature | How | Signal |
|---|---|---|
| `n_view` | count of `view` events | Browsing intensity |
| `n_cart` | count of `cart` events | Purchase intent |
| `n_remove` | count of `remove_from_cart` | Hesitation |
| `n_events` | total events | Activity volume |
| `cart_to_view_ratio` | `n_cart / n_view` | Interest conversion |
| `remove_to_cart_ratio` | `n_remove / max(n_cart, 1)` | Abandonment |
| `avg_price` | mean of `price` | Budget proxy |
| `max_price` | max of `price` | Ceiling interest |
| `min_price` | min of `price` | Floor interest |
| `price_std` | std of `price` | Price variance |
| `unique_products` | `nunique(product_id)` | Exploration breadth |
| `unique_categories` | `nunique(category_id)` | Category spread |
| `has_brand` | any non-null `brand` | Brand awareness |
| `session_duration_sec` | max(ts) − min(ts) | Session length |

Handle divisions-by-zero defensively (use `np.where` or add `+1` in denominator).

---

## 🤖 Models

### Model 1 — Binary Classifier

- **Primary**: `XGBClassifier` with `scale_pos_weight` OR `RandomForestClassifier` with `class_weight='balanced'`.
- **Baseline**: `LogisticRegression` with `class_weight='balanced'`.
- Output: probability + thresholded label.

### Model 2 — Multi-class Classifier

- **Primary**: `RandomForestClassifier` or `XGBClassifier` (multi:softprob) with `class_weight='balanced'`.
- Output: predicted class + class probabilities.

### Train/Test Split

- **Stratified 80/20 split** on the target variable.
- `random_state=42` everywhere for reproducibility.
- Any resampling is applied to **TRAIN ONLY**, never to test.

---

## 📁 Project Structure

```
project/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── 2019-Oct.csv
│   │   ├── 2019-Dec.csv
│   │   ├── 2020-Jan.csv
│   │   └── 2020-Feb.csv
│   └── processed/
│       ├── sessions_all.parquet     # For binary model
│       └── sessions_cat.parquet     # For multi-class model
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Load & concatenate monthly CSVs (memory-aware)
│   ├── preprocessing.py             # Event → session aggregation + category filling
│   ├── feature_engineering.py       # Derive session-level features
│   ├── model_binary.py              # Binary training & evaluation
│   ├── model_multiclass.py          # Multi-class training & evaluation
│   └── evaluate.py                  # Metrics, plots, confusion matrices
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_binary_model.ipynb
│   └── 03_multiclass_model.ipynb
├── models/
│   ├── binary_model.pkl
│   └── multiclass_model.pkl
└── results/
    ├── figures/
    └── metrics.json
```

---

## 🛠️ Development Guidelines

### Code Style

- **Python**: 3.10+
- **Type hints** on public function signatures.
- **Docstrings**: Google style, concise.
- **Randomness**: always `random_state=42`.

### Memory Management

Raw files are ~4M rows each. Apply these precautions:

- Read only the columns you need (`usecols=...`).
- Cast `event_type` to `category` dtype.
- Process files **one at a time**, aggregate to session level, then concat the results (NOT the raw events).
- Persist processed datasets as **Parquet**, not CSV.

### Reproducibility

- Save fitted models with `joblib`.
- Save hyperparameters and final metrics as JSON under `results/`.

### Smoke-Test First

Always test the pipeline on **one month** or a **100K-row sample** before running on the full 4-month data.

---

## 📋 Current Status

- [x] **Phase 0**: Dataset selection and validation
- [ ] **Phase 1**: Data loading & session aggregation
- [ ] **Phase 2**: Feature engineering
- [ ] **Phase 3**: Binary classification model
- [ ] **Phase 4**: Multi-class classification model
- [ ] **Phase 5**: Evaluation & visualization
- [ ] **Phase 6**: IEEE-format academic report

---

## 🚫 Out of Scope (for now)

- Recommendation system
- Deep learning models
- Extensive model comparison / ablation studies
- Large-scale hyperparameter tuning
- Deployment / real-time inference

---

## 💡 Academic Defense Points

Be prepared to justify the following before the jury:

1. **Why session-level, not event-level?**
   Predictions must be actionable at a user-intent level; events alone are too granular to represent intent.

2. **Why two models instead of a single multi-output classifier?**
   The two tasks operate on different data scopes (all sessions vs. only categorized sessions). Separate models yield cleaner interpretation and task-appropriate metrics.

3. **Why `class_weight='balanced'` over SMOTE?**
   Data volume is abundant; synthetic samples introduce noise and add pipeline risk (e.g., test-set leakage). The simpler approach is sufficient here.

4. **Why these specific features?**
   Each feature encodes a distinct behavioral signal — intent (`n_cart`), hesitation (`n_remove`), budget (`avg_price`), exploration (`unique_products`), etc.

5. **Why Macro F1 for multi-class?**
   Class frequencies vary (9% — 42%). Macro F1 treats every category as equally important, which matches the real-world goal of predicting minority categories well.

6. **Why not Accuracy?**
   Under 3.4% positive class, a trivial "always predict 0" classifier scores 96.6% accuracy while being useless. F1/ROC-AUC reflect true performance.

---

## 📦 Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
pyarrow>=14.0
jupyter
```

---

## 🎬 Recommended Execution Order (for Claude Code)

Follow this sequence when building the project:

1. **Scaffold** the folder structure above; create empty `__init__.py` and stubs.
2. **`src/data_loader.py`** — load CSVs one-by-one with memory-efficient dtypes.
3. **`src/preprocessing.py`** — build `category_id → category_code` map, fill main_cat, aggregate events → sessions.
4. **`src/feature_engineering.py`** — produce the 14 session-level features listed above.
5. **Save** aggregated datasets to `data/processed/` as Parquet.
6. **`src/model_binary.py`** — train, evaluate, and save binary model.
7. **`src/model_multiclass.py`** — same for multi-class.
8. **`src/evaluate.py`** — generate all plots and dump metrics to JSON.
9. **Sanity check** on a held-out sample session; print predicted probability + category.

> **Always prototype on 1 month first, confirm correctness, then scale to 4 months.**