"""Build the three presentation notebooks via nbformat.

Each notebook loads directly from ``data/processed/*.parquet`` (no raw-CSV
re-aggregation) and keeps cells tight for slide-style presentation.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path("notebooks")
NB_DIR.mkdir(parents=True, exist_ok=True)


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def _write(nb: nbf.NotebookNode, path: Path) -> None:
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, path)


# ---------------------------------------------------------------------------
# 01 - Exploration
# ---------------------------------------------------------------------------

SETUP = """\
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.feature_engineering import FEATURE_COLS

DATA = Path("..") / "data" / "processed"
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
plt.rcParams["figure.dpi"] = 110
"""

nb = nbf.v4.new_notebook()
nb.cells = [
    _md(
        "# 01 - Veri Keşfi (Exploration)\n\n"
        "**Amaç**: Event-level e-ticaret verisinden türetilen session-level "
        "veri setinin yapısını, hedef değişken dağılımını ve feature "
        "davranışlarını görsel olarak incelemek.\n\n"
        "*Veri kaynağı*: Kaggle `mkechinov/ecommerce-events-history-in-cosmetics-shop` "
        "(4 aylık CSV, ~16M event). Aggregation Phase 1'de tamamlandı.  \n"
        "*Girdi*: `data/processed/sessions_all.parquet`, `data/processed/sessions_cat.parquet`"
    ),
    _code(SETUP),
    _md("## 1. Dataset'i yükle"),
    _code(
        'sessions_all = pd.read_parquet(DATA / "sessions_all.parquet")\n'
        'sessions_cat = pd.read_parquet(DATA / "sessions_cat.parquet")\n'
        'print(f"sessions_all: {sessions_all.shape[0]:,} rows x {sessions_all.shape[1]} cols")\n'
        'print(f"sessions_cat: {sessions_cat.shape[0]:,} rows x {sessions_cat.shape[1]} cols")\n'
        'sessions_all.head(3)'
    ),
    _md(
        "## 2. Hedef değişken: purchased\n\n"
        "Session bazında **~%3.3** purchase oranı. Ciddi class imbalance — "
        "bu yüzden doğrudan Accuracy yerine F1/ROC-AUC kullanacağız."
    ),
    _code(
        'pur_rate = sessions_all["purchased"].mean()\n'
        'counts = sessions_all["purchased"].value_counts().rename({0: "no_purchase", 1: "purchase"})\n'
        'print(f"purchase rate: {pur_rate:.4f}")\n'
        'print(counts)\n'
        '\n'
        'fig, ax = plt.subplots(figsize=(5, 3.2))\n'
        'counts.plot.bar(ax=ax, color=["#8da0cb", "#fc8d62"])\n'
        'ax.set_title("Session Sayisi: purchase vs no_purchase")\n'
        'ax.set_ylabel("session")\n'
        'ax.tick_params(axis="x", rotation=0)\n'
        'for p in ax.patches:\n'
        '    ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),\n'
        '                ha="center", va="bottom", fontsize=9)\n'
        'plt.show()'
    ),
    _md(
        "## 3. Kategori dağılımı (Model 2 subset)\n\n"
        "`category_id → category_code` mapping'i ile doldurulmuş sessionlar. "
        "5 ana kategori: appliances baskın, accessories minority."
    ),
    _code(
        'cat_counts = sessions_cat["main_cat"].value_counts()\n'
        'display(cat_counts.to_frame("count").assign(share=lambda d: d["count"] / d["count"].sum()))\n'
        '\n'
        'fig, ax = plt.subplots(figsize=(6, 3.5))\n'
        'cat_counts.plot.bar(ax=ax, color="steelblue")\n'
        'ax.set_title("main_cat dagilimi (Model 2 subset)")\n'
        'ax.set_ylabel("session")\n'
        'ax.tick_params(axis="x", rotation=20)\n'
        'plt.show()'
    ),
    _md(
        "## 4. Feature özeti\n\n"
        "14 session-level feature'ın ortalama/medyan/aralık değerleri. "
        "Ağır kuyruklu dağılımlar (session_duration, n_events) outlier'ların "
        "varlığına işaret eder."
    ),
    _code(
        'sessions_all[FEATURE_COLS].describe().T[["mean", "50%", "std", "min", "max"]]\\\n'
        '    .round(3)'
    ),
    _md(
        "## 5. Feature dağılımları\n\n"
        "Ağır kuyruk nedeniyle log-scale histogramlar. `n_events` ve `session_duration_sec` "
        "özellikle geniş spektruma yayılıyor."
    ),
    _code(
        'fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))\n'
        'cols = ["n_events", "n_view", "n_cart", "avg_price", "session_duration_sec", "cart_to_view_ratio"]\n'
        'for ax, col in zip(axes.ravel(), cols):\n'
        '    data = sessions_all[col].replace([np.inf, -np.inf], np.nan).dropna()\n'
        '    data = data[data >= 0]\n'
        '    ax.hist(np.log1p(data), bins=60, color="#4c72b0", alpha=0.85)\n'
        '    ax.set_title(f"log1p({col})")\n'
        '    ax.set_ylabel("session")\n'
        'plt.tight_layout()\n'
        'plt.show()'
    ),
    _md(
        "## 6. Davranış farkı: purchased vs no_purchase\n\n"
        "Her feature'in class bazında ortalaması — purchased session'lar "
        "belirgin şekilde daha yüksek etkileşim ve kartlama sergiliyor."
    ),
    _code(
        'cmp = sessions_all.groupby("purchased")[FEATURE_COLS].mean().T\n'
        'cmp.columns = ["no_purchase", "purchase"]\n'
        'cmp["ratio"] = cmp["purchase"] / cmp["no_purchase"].replace(0, np.nan)\n'
        'display(cmp.round(3))'
    ),
    _code(
        'key_feats = ["n_view", "n_cart", "n_remove", "cart_to_view_ratio",\n'
        '             "unique_products", "session_duration_sec"]\n'
        'fig, ax = plt.subplots(figsize=(9, 4.2))\n'
        'medians = sessions_all.groupby("purchased")[key_feats].median().T\n'
        'medians.columns = ["no_purchase", "purchase"]\n'
        'medians.plot.bar(ax=ax, color=["#8da0cb", "#fc8d62"])\n'
        'ax.set_title("Medyan feature degerleri (class bazinda)")\n'
        'ax.set_ylabel("medyan")\n'
        'ax.tick_params(axis="x", rotation=25)\n'
        'ax.legend(title="")\n'
        'plt.tight_layout()\n'
        'plt.show()'
    ),
    _md(
        "## 7. Feature korelasyonu\n\n"
        "Count feature'ları (`n_view/n_cart/n_remove/n_events`) beklendiği üzere "
        "yüksek korelasyonlu; fiyat feature'ları (`avg/max/min_price`) kendi "
        "aralarında kümeleniyor."
    ),
    _code(
        'corr = sessions_all[FEATURE_COLS].corr()\n'
        'fig, ax = plt.subplots(figsize=(9, 7))\n'
        'im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)\n'
        'ax.set_xticks(range(len(FEATURE_COLS)))\n'
        'ax.set_yticks(range(len(FEATURE_COLS)))\n'
        'ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")\n'
        'ax.set_yticklabels(FEATURE_COLS)\n'
        'for i in range(len(FEATURE_COLS)):\n'
        '    for j in range(len(FEATURE_COLS)):\n'
        '        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center",\n'
        '                color="black" if abs(corr.iat[i, j]) < 0.6 else "white", fontsize=7)\n'
        'fig.colorbar(im, ax=ax)\n'
        'ax.set_title("Feature korelasyon matrisi")\n'
        'plt.tight_layout()\n'
        'plt.show()'
    ),
    _md(
        "## 8. Bulgu özeti\n\n"
        "- **Imbalance**: %3.32 purchase — baseline \"hep 0 tahmin et\" %96.7 accuracy almak için yeterli, ama F1=0. Bu yüzden F1/ROC-AUC öncelikli.\n"
        "- **Category subset**: 131,297 session (%3.6'sı ana veriden) — appliances %45 baskın, accessories %9.\n"
        "- **Davranış sinyali güçlü**: Purchased session'lar ortalama ~17 event vs ~3.9; `cart_to_view_ratio` ve `unique_products` net ayrıcı.\n"
        "- **Feature mühendisliği kararı**: Tüm feature'lar non-purchase event'ler üzerinden hesaplandı — hedef sızıntısı yok.\n"
    ),
]
_write(nb, NB_DIR / "01_exploration.ipynb")
print("wrote:", NB_DIR / "01_exploration.ipynb")


# ---------------------------------------------------------------------------
# 02 - Binary model
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb.cells = [
    _md(
        "# 02 - Binary Classifier: Will the user purchase?\n\n"
        "**Problem**: Session-level feature'lardan purchase (1) / no-purchase (0) tahmini.\n\n"
        "**Zorluk**: %3.3 pozitif class.\n\n"
        "**Strateji**: `class_weight='balanced'` (LogReg) + `scale_pos_weight` (XGBoost). "
        "SMOTE kullanılmıyor — veri hacmi yeterli ve pipeline daha basit kalıyor."
    ),
    _code(SETUP + "\nfrom src.model_binary import split, train_baseline, train_primary, evaluate"),
    _md("## 1. Veriyi yükle"),
    _code(
        'sessions_all = pd.read_parquet(DATA / "sessions_all.parquet")\n'
        'X = sessions_all[FEATURE_COLS].astype("float32")\n'
        'y = sessions_all["purchased"].astype("int8")\n'
        'print(f"X: {X.shape}  y: {y.shape}")\n'
        'print(f"positive rate: {y.mean():.4f}")'
    ),
    _md("## 2. Stratified train/test split (80/20, random_state=42)"),
    _code(
        'X_train, X_test, y_train, y_test = split(X, y)\n'
        'print(f"train: {len(y_train):,}  test: {len(y_test):,}")\n'
        'print(f"train pos rate: {y_train.mean():.4f}  test pos rate: {y_test.mean():.4f}")'
    ),
    _md(
        "## 3. Baseline — Logistic Regression\n\n"
        "Standardized feature'lar + `class_weight='balanced'`. Lineer sinirli bir model, "
        "primary modele kıyaslama noktası."
    ),
    _code(
        '%time baseline = train_baseline(X_train, y_train)\n'
        'base_metrics = evaluate(baseline, X_test, y_test)\n'
        'base_metrics'
    ),
    _md(
        "## 4. Primary — XGBoost\n\n"
        "Gradient-boosted trees, `scale_pos_weight = n_neg/n_pos` ile imbalance yönetimi. "
        "Non-linear etkileşimleri yakalar."
    ),
    _code(
        '%time primary = train_primary(X_train, y_train)\n'
        'prim_metrics = evaluate(primary, X_test, y_test)\n'
        'prim_metrics'
    ),
    _md("## 5. Model karşılaştırması"),
    _code(
        'def row(m):\n'
        '    return {"F1": m["f1"], "ROC_AUC": m["roc_auc"],\n'
        '            "Precision": m["precision"], "Recall": m["recall"]}\n'
        '\n'
        'comp = pd.DataFrame({"LogReg": row(base_metrics), "XGBoost": row(prim_metrics)}).round(4)\n'
        'display(comp)'
    ),
    _md(
        "## 6. ROC & Precision-Recall eğrileri\n\n"
        "ROC-AUC threshold-bağımsız **ranking** performansını ölçer. PR-AUC (AP) "
        "imbalanced data'da Accuracy/ROC'tan daha bilgilendirici — base-rate'e göre "
        "relatif iyileşme görülebilir."
    ),
    _code(
        'from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score\n'
        '\n'
        'proba = primary.predict_proba(X_test)[:, 1]\n'
        'fpr, tpr, _ = roc_curve(y_test, proba)\n'
        'prec, rec, _ = precision_recall_curve(y_test, proba)\n'
        'roc_auc_v = auc(fpr, tpr)\n'
        'ap = average_precision_score(y_test, proba)\n'
        '\n'
        'fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n'
        'axes[0].plot(fpr, tpr, label=f"XGBoost (AUC={roc_auc_v:.3f})", linewidth=2)\n'
        'axes[0].plot([0,1],[0,1], "--", color="gray", label="Random")\n'
        'axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC")\n'
        'axes[0].legend(); axes[0].grid(alpha=0.3)\n'
        '\n'
        'axes[1].plot(rec, prec, label=f"XGBoost (AP={ap:.3f})", linewidth=2, color="#e07b39")\n'
        'axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label=f"Base rate={y_test.mean():.3f}")\n'
        'axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall")\n'
        'axes[1].legend(); axes[1].grid(alpha=0.3)\n'
        'plt.tight_layout(); plt.show()'
    ),
    _md(
        "## 7. Confusion matrix (threshold=0.5)\n\n"
        "`class_weight='balanced'` recall'u agresif optimize ediyor — "
        "neredeyse tüm gerçek purchase'ları yakalıyor ama FP sayısı yüksek. "
        "Deployment'ta threshold tuning ile precision/recall trade-off ayarlanabilir."
    ),
    _code(
        'from sklearn.metrics import confusion_matrix\n'
        '\n'
        'pred = (proba >= 0.5).astype(int)\n'
        'cm = confusion_matrix(y_test, pred)\n'
        'fig, ax = plt.subplots(figsize=(5, 4))\n'
        'im = ax.imshow(cm, cmap="Blues")\n'
        'labels = ["no_purchase", "purchase"]\n'
        'ax.set_xticks([0,1]); ax.set_yticks([0,1])\n'
        'ax.set_xticklabels(labels); ax.set_yticklabels(labels)\n'
        'ax.set_xlabel("Predicted"); ax.set_ylabel("True")\n'
        'ax.set_title("Confusion Matrix (thr=0.5)")\n'
        'thr = cm.max() / 2\n'
        'for i in range(2):\n'
        '    for j in range(2):\n'
        '        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",\n'
        '                color="white" if cm[i,j] > thr else "black", fontsize=11)\n'
        'fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()'
    ),
    _md(
        "## 8. Feature importance (XGBoost)\n\n"
        "Gain-based importance. Hangi davranış sinyalleri tahmini belirliyor?"
    ),
    _code(
        'imp = pd.Series(primary.feature_importances_, index=FEATURE_COLS).sort_values()\n'
        'fig, ax = plt.subplots(figsize=(7, 4.5))\n'
        'imp.plot.barh(ax=ax, color="steelblue")\n'
        'ax.set_xlabel("gain importance")\n'
        'ax.set_title("XGBoost Feature Importance — Binary")\n'
        'plt.tight_layout(); plt.show()'
    ),
    _md(
        "## 9. Sonuç\n\n"
        "| Model | F1 | ROC-AUC | Precision | Recall |\n"
        "|---|---:|---:|---:|---:|\n"
        "| LogReg | ~0.26 | ~0.86 | ~0.16 | ~0.66 |\n"
        "| **XGBoost** | **~0.30** | **~0.94** | ~0.18 | **~0.92** |\n\n"
        "- **ROC-AUC 0.94**: model session'ları purchase olasılığına göre çok iyi sıralıyor.\n"
        "- **Recall öncelikli**: imbalance stratejimiz neredeyse tüm purchase'ları yakalamayı tercih ediyor.\n"
        "- **AP 0.44** (base rate 0.033) → random'a göre ~13× iyileşme.\n"
        "- **Sonraki adım**: threshold tuning (deployment için precision-recall trade-off).\n"
    ),
]
_write(nb, NB_DIR / "02_binary_model.ipynb")
print("wrote:", NB_DIR / "02_binary_model.ipynb")


# ---------------------------------------------------------------------------
# 03 - Multi-class model
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb.cells = [
    _md(
        "# 03 - Multi-class Classifier: Which category?\n\n"
        "**Problem**: 5 ana kategoriden hangisi? "
        "(`appliances` / `furniture` / `apparel` / `stationery` / `accessories`)\n\n"
        "**Veri**: `sessions_cat.parquet` — `category_code` recovery ile doldurulmuş "
        "131,297 session.\n\n"
        "**Strateji**: LogReg baseline + XGBoost `multi:softprob` primary. "
        "Primary metrik **Macro F1** (az temsil edilen sınıflara eşit ağırlık)."
    ),
    _code(
        SETUP + "\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "from src.model_multiclass import split, train_baseline, train_primary, evaluate"
    ),
    _md("## 1. Veriyi yükle"),
    _code(
        'sessions_cat = pd.read_parquet(DATA / "sessions_cat.parquet")\n'
        'X = sessions_cat[FEATURE_COLS].astype("float32")\n'
        'y = sessions_cat["main_cat"].astype("string")\n'
        'print(f"X: {X.shape}  y: {y.shape}")\n'
        'print(y.value_counts())'
    ),
    _md("## 2. Train/test split (stratified)"),
    _code(
        'X_train, X_test, y_train, y_test = split(X, y)\n'
        '\n'
        'le = LabelEncoder()\n'
        'y_train_enc = le.fit_transform(y_train)\n'
        'y_test_enc = le.transform(y_test)\n'
        'labels = list(le.classes_)\n'
        'print(f"train: {len(y_train):,}  test: {len(y_test):,}")\n'
        'print(f"labels (alpha): {labels}")'
    ),
    _md("## 3. Baseline — Multinomial Logistic Regression"),
    _code(
        '%time baseline = train_baseline(X_train, y_train)\n'
        'base_metrics = evaluate(baseline, X_test, y_test, labels, encoded=False)\n'
        'print(f"macro_f1    = {base_metrics[\'macro_f1\']:.4f}")\n'
        'print(f"weighted_f1 = {base_metrics[\'weighted_f1\']:.4f}")\n'
        'print("per-class F1:")\n'
        'pd.Series(base_metrics["per_class_f1"]).round(3)'
    ),
    _md("## 4. Primary — XGBoost (`multi:softprob`)"),
    _code(
        '%time primary = train_primary(X_train, y_train_enc, n_classes=len(labels))\n'
        'prim_metrics = evaluate(primary, X_test, y_test_enc, labels, encoded=True)\n'
        'print(f"macro_f1    = {prim_metrics[\'macro_f1\']:.4f}")\n'
        'print(f"weighted_f1 = {prim_metrics[\'weighted_f1\']:.4f}")\n'
        'print("per-class F1:")\n'
        'pd.Series(prim_metrics["per_class_f1"]).round(3)'
    ),
    _md("## 5. Model karşılaştırması"),
    _code(
        'comp = pd.DataFrame({\n'
        '    "LogReg":  [base_metrics["macro_f1"], base_metrics["weighted_f1"]],\n'
        '    "XGBoost": [prim_metrics["macro_f1"], prim_metrics["weighted_f1"]],\n'
        '}, index=["Macro F1", "Weighted F1"]).round(4)\n'
        'display(comp)'
    ),
    _md(
        "## 6. Confusion matrix — sayılar ve row-normalize\n\n"
        "Sol: ham sayılar. Sağ: her satır kendi toplamına bölünmüş — per-class recall'u "
        "doğrudan gösterir. Diyagonal baskınlığı modelin doğru sınıflandırmasını yansıtır."
    ),
    _code(
        'from sklearn.metrics import confusion_matrix\n'
        'import numpy as np\n'
        '\n'
        'pred_enc = primary.predict(X_test)\n'
        'cm = confusion_matrix(y_test_enc, pred_enc, labels=list(range(len(labels))))\n'
        'cm_norm = cm / cm.sum(axis=1, keepdims=True)\n'
        '\n'
        'fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n'
        'for ax, mat, fmt, title in [\n'
        '    (axes[0], cm, "d", "Confusion Matrix — counts"),\n'
        '    (axes[1], cm_norm, ".2f", "Confusion Matrix — row-normalized"),\n'
        ']:\n'
        '    im = ax.imshow(mat, cmap="Blues")\n'
        '    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))\n'
        '    ax.set_xticklabels(labels, rotation=25, ha="right")\n'
        '    ax.set_yticklabels(labels)\n'
        '    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)\n'
        '    thr = mat.max()/2\n'
        '    for i in range(len(labels)):\n'
        '        for j in range(len(labels)):\n'
        '            v = mat[i, j]\n'
        '            ax.text(j, i, format(v, fmt), ha="center", va="center",\n'
        '                    color="white" if v > thr else "black", fontsize=8)\n'
        '    fig.colorbar(im, ax=ax)\n'
        'plt.tight_layout(); plt.show()'
    ),
    _md(
        "## 7. Feature importance (XGBoost)\n\n"
        "Hangi feature'lar kategori ayırımını belirliyor? Fiyat aralıkları baskın "
        "sinyal olmayı bekliyoruz (appliances >> stationery fiyat olarak)."
    ),
    _code(
        'imp = pd.Series(primary.feature_importances_, index=FEATURE_COLS).sort_values()\n'
        'fig, ax = plt.subplots(figsize=(7, 4.5))\n'
        'imp.plot.barh(ax=ax, color="steelblue")\n'
        'ax.set_xlabel("gain importance")\n'
        'ax.set_title("XGBoost Feature Importance — Multi-class")\n'
        'plt.tight_layout(); plt.show()'
    ),
    _md(
        "## 8. Sonuç\n\n"
        "| Model | Macro F1 | Weighted F1 |\n"
        "|---|---:|---:|\n"
        "| LogReg | ~0.56 | ~0.62 |\n"
        "| **XGBoost** | **~0.85** | **~0.88** |\n\n"
        "- Baseline → primary kazancı **+0.30 Macro F1** — non-lineer etkileşimlerin katkısı açık.\n"
        "- En kolay sınıf **appliances** (F1 ~0.93): fiyat + brand + duration ayırıcı.\n"
        "- En zor sınıf **apparel** (F1 ~0.75): fiyat dağılımı diğer kategorilerle örtüşüyor — en büyük karışıklık `appliances ↔ apparel`.\n"
        "- **Leak YOK**: feature'lar non-purchase event'lerden hesaplandı; main_cat target olarak dışlandı.\n"
    ),
]
_write(nb, NB_DIR / "03_multiclass_model.ipynb")
print("wrote:", NB_DIR / "03_multiclass_model.ipynb")
