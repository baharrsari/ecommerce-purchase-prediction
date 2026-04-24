"""Build notebooks/04_temporal_validation.ipynb.

Keeps cells tight and presentation-ready. Reuses training/eval helpers from
``src/`` and merges new metrics into ``results/metrics.json`` under the
``temporal_split`` key — existing ``binary`` / ``multiclass`` keys are
preserved.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/04_temporal_validation.ipynb")


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


SETUP = """\
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import FEATURE_COLS
from src.model_binary import train_primary as train_bin, evaluate as eval_bin
from src.model_multiclass import train_primary as train_mc, evaluate as eval_mc

DATA = Path("..") / "data" / "processed"
FIG_DIR = Path("..") / "results" / "figures" / "temporal"
METRICS_PATH = Path("..") / "results" / "metrics.json"
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 30)
plt.rcParams["figure.dpi"] = 110
"""


nb = nbf.v4.new_notebook()
nb.cells = [
    _md(
        "# 04 - Temporal Train-Test Split Validation\n\n"
        "## Neden temporal split?\n\n"
        "Phase 3-4'te kullandığımız **random stratified** split her session'ı bağımsız/özdeş dağılımdan "
        "(i.i.d.) gibi muamele ediyor — aynı haftanın bir session'ı train'de, diğeri test'te olabiliyor. "
        "Ama gerçek deployment'ta model **geçmişten gelecek'i** tahmin eder. Eğer veride zamansal "
        "yapı varsa (ki genelde vardır), random split **optimistic** bir tablo çizer.\n\n"
        "Bu veri seti için spesifik riskler:\n\n"
        "- **Mevsimsellik**: Aralık (yılbaşı alışveriş), Ocak (indirimler), Şubat (normal) farklı "
        "satın-alma davranışı gösterebilir.\n"
        "- **Ürün kataloğu kayması**: Şubat'ta yeni ürünler / yeni fiyatlar / kaybolan markalar.\n"
        "- **Kullanıcı davranış drift'i**: Platform-level değişimler (UI, öneri sistemi) ile ratio'lar "
        "kayabilir.\n"
        "- **Outlier bot aktivitesi**: Aya göre dalgalanabilir.\n\n"
        "## Split tasarımı\n\n"
        "| Bölüm | Aylar | Senaryo |\n"
        "|---|---|---|\n"
        "| **Train** | 2019-Oct + 2019-Dec + 2020-Jan | Önceki 3 ayın geçmişi |\n"
        "| **Test**  | 2020-Feb | Deploy gününden itibaren 1 ay |\n\n"
        "Bu, Şubat başında sistemi production'a alma senaryosunu simüle eder.\n\n"
        "> **Not**: `results/metrics.json`'daki mevcut `binary` ve `multiclass` key'leri silinmiyor; "
        "yeni ölçümler `temporal_split` altına ekleniyor."
    ),
    _code(SETUP),
    _md("## 1. Veri yükle, ay etiketini birleştir\n\n"
        "`session_months.parquet`: raw CSV'leri tek-kolon tarayarak her session'ın ilk göründüğü "
        "ayı kaydeden side-table. Cross-month session'lar (Dec 31 -> Jan 1) erken aya atanır."),
    _code(
        'sessions_all = pd.read_parquet(DATA / "sessions_all.parquet")\n'
        'sessions_cat = pd.read_parquet(DATA / "sessions_cat.parquet")\n'
        'months_map   = pd.read_parquet(DATA / "session_months.parquet")\n'
        '\n'
        'sessions_all = sessions_all.merge(months_map, on="user_session", how="left")\n'
        'sessions_cat = sessions_cat.merge(months_map, on="user_session", how="left")\n'
        '\n'
        'print(f"sessions_all: {len(sessions_all):,}  (missing month: {sessions_all[\'month\'].isna().sum()})")\n'
        'print(f"sessions_cat: {len(sessions_cat):,}  (missing month: {sessions_cat[\'month\'].isna().sum()})")\n'
        '\n'
        'print("\\nay dagilimi (sessions_all):")\n'
        'print(sessions_all["month"].value_counts().sort_index())\n'
        'print("\\nay dagilimi (sessions_cat):")\n'
        'print(sessions_cat["month"].value_counts().sort_index())'
    ),
    _md("## 2. Temporal split\n\nTrain = ilk 3 ay. Test = son ay. Stratifikasyon yok — "
        "doğal zamansal sıraya sadık kalıyoruz."),
    _code(
        'TRAIN_MONTHS = {"2019-Oct", "2019-Dec", "2020-Jan"}\n'
        'TEST_MONTHS  = {"2020-Feb"}\n'
        '\n'
        'def tsplit(df):\n'
        '    tr = df[df["month"].isin(TRAIN_MONTHS)].copy()\n'
        '    te = df[df["month"].isin(TEST_MONTHS)].copy()\n'
        '    return tr, te\n'
        '\n'
        'bin_tr, bin_te = tsplit(sessions_all)\n'
        'mc_tr,  mc_te  = tsplit(sessions_cat)\n'
        '\n'
        'print("[binary]")\n'
        'print(f"  train: {len(bin_tr):,}  purchase rate {bin_tr[\'purchased\'].mean():.4f}")\n'
        'print(f"  test : {len(bin_te):,}  purchase rate {bin_te[\'purchased\'].mean():.4f}")\n'
        'print()\n'
        'print("[multi] train class distribution:")\n'
        'print(mc_tr["main_cat"].value_counts(normalize=True).round(3))\n'
        'print("\\n[multi] test class distribution:")\n'
        'print(mc_te["main_cat"].value_counts(normalize=True).round(3))'
    ),
    _md("## 3. Binary classifier — temporal\n\n"
        "Phase 3 ile aynı XGBoost konfigurasyonu (`train_primary`). Sadece split farkı."),
    _code(
        'X_tr = bin_tr[FEATURE_COLS].astype("float32"); y_tr = bin_tr["purchased"].astype("int8")\n'
        'X_te = bin_te[FEATURE_COLS].astype("float32"); y_te = bin_te["purchased"].astype("int8")\n'
        '\n'
        '%time model_bin = train_bin(X_tr, y_tr)\n'
        'bin_m = eval_bin(model_bin, X_te, y_te)\n'
        'bin_m'
    ),
    _md("### ROC + Precision-Recall eğrileri"),
    _code(
        'proba = model_bin.predict_proba(X_te)[:, 1]\n'
        'fpr, tpr, _ = roc_curve(y_te, proba)\n'
        'prec, rec, _ = precision_recall_curve(y_te, proba)\n'
        'ap = average_precision_score(y_te, proba)\n'
        'bin_m["average_precision"] = float(ap)\n'
        '\n'
        'fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n'
        'axes[0].plot(fpr, tpr, linewidth=2, label=f"XGB (AUC={bin_m[\'roc_auc\']:.3f})")\n'
        'axes[0].plot([0,1],[0,1], "--", color="gray", label="Random")\n'
        'axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("Temporal — ROC")\n'
        'axes[0].legend(); axes[0].grid(alpha=0.3)\n'
        '\n'
        'axes[1].plot(rec, prec, linewidth=2, color="#e07b39", label=f"XGB (AP={ap:.3f})")\n'
        'axes[1].axhline(y_te.mean(), linestyle="--", color="gray", label=f"Base={y_te.mean():.3f}")\n'
        'axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Temporal — PR")\n'
        'axes[1].legend(); axes[1].grid(alpha=0.3)\n'
        'plt.tight_layout()\n'
        'plt.savefig(FIG_DIR / "binary_roc_pr.png", dpi=150, bbox_inches="tight")\n'
        'plt.show()'
    ),
    _md("### Confusion matrix"),
    _code(
        'pred = (proba >= 0.5).astype(int)\n'
        'cm = confusion_matrix(y_te, pred)\n'
        'fig, ax = plt.subplots(figsize=(5, 4))\n'
        'im = ax.imshow(cm, cmap="Blues")\n'
        'ax.set_xticks([0,1]); ax.set_yticks([0,1])\n'
        'ax.set_xticklabels(["no_purchase","purchase"]); ax.set_yticklabels(["no_purchase","purchase"])\n'
        'ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Temporal — Binary CM (thr=0.5)")\n'
        'thr = cm.max()/2\n'
        'for i in range(2):\n'
        '    for j in range(2):\n'
        '        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",\n'
        '                color="white" if cm[i,j] > thr else "black", fontsize=11)\n'
        'fig.colorbar(im, ax=ax)\n'
        'plt.savefig(FIG_DIR / "binary_confusion.png", dpi=150, bbox_inches="tight")\n'
        'plt.show()'
    ),
    _md("## 4. Multi-class classifier — temporal"),
    _code(
        'X_tr = mc_tr[FEATURE_COLS].astype("float32"); y_tr_raw = mc_tr["main_cat"].astype("string")\n'
        'X_te = mc_te[FEATURE_COLS].astype("float32"); y_te_raw = mc_te["main_cat"].astype("string")\n'
        '\n'
        'le = LabelEncoder()\n'
        'y_tr = le.fit_transform(y_tr_raw)\n'
        'labels = list(le.classes_)\n'
        '# Test setinde train\'de görünmeyen bir sınıf çıkarsa hata almamak için filtrele\n'
        'mask = y_te_raw.isin(set(labels)).values\n'
        'X_te = X_te.iloc[mask]; y_te_raw = y_te_raw[mask]\n'
        'y_te = le.transform(y_te_raw)\n'
        '\n'
        'print(f"train: {len(y_tr):,}  test: {len(y_te):,}  labels: {labels}")\n'
        '\n'
        '%time model_mc = train_mc(X_tr, y_tr, n_classes=len(labels))\n'
        'mc_m = eval_mc(model_mc, X_te, y_te, labels, encoded=True)\n'
        'print(f"macro_f1    = {mc_m[\'macro_f1\']:.4f}")\n'
        'print(f"weighted_f1 = {mc_m[\'weighted_f1\']:.4f}")\n'
        'pd.Series(mc_m["per_class_f1"]).round(3)'
    ),
    _md("### Confusion matrix (row-normalized)"),
    _code(
        'pred_mc = model_mc.predict(X_te)\n'
        'cm_mc = confusion_matrix(y_te, pred_mc, labels=list(range(len(labels))))\n'
        'cm_norm = cm_mc / cm_mc.sum(axis=1, keepdims=True)\n'
        '\n'
        'fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n'
        'for ax, mat, fmt, title in [\n'
        '    (axes[0], cm_mc,   "d",   "Temporal Multi — counts"),\n'
        '    (axes[1], cm_norm, ".2f", "Temporal Multi — row-normalized"),\n'
        ']:\n'
        '    im = ax.imshow(mat, cmap="Blues")\n'
        '    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))\n'
        '    ax.set_xticklabels(labels, rotation=25, ha="right"); ax.set_yticklabels(labels)\n'
        '    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)\n'
        '    thr = mat.max()/2\n'
        '    for i in range(len(labels)):\n'
        '        for j in range(len(labels)):\n'
        '            v = mat[i, j]\n'
        '            ax.text(j, i, format(v, fmt), ha="center", va="center",\n'
        '                    color="white" if v > thr else "black", fontsize=8)\n'
        '    fig.colorbar(im, ax=ax)\n'
        'plt.tight_layout()\n'
        'plt.savefig(FIG_DIR / "multiclass_confusion.png", dpi=150, bbox_inches="tight")\n'
        'plt.show()'
    ),
    _md("## 5. Random vs Temporal karşılaştırma\n\n"
        "Eski `results/metrics.json`'dan random split sonuçlarını okuyup yan yana koyuyoruz. "
        "Temporal split'in performans kaybı **distribution shift'in büyüklüğünü** ölçer."),
    _code(
        'with open(METRICS_PATH) as f:\n'
        '    old = json.load(f)\n'
        '\n'
        'rb = old["binary"]["primary_xgboost"]\n'
        'rm = old["multiclass"]["primary_xgboost"]\n'
        '\n'
        'bin_cmp = pd.DataFrame({\n'
        '    "Random":   {"F1": rb["f1"], "ROC_AUC": rb["roc_auc"],\n'
        '                 "Precision": rb["precision"], "Recall": rb["recall"]},\n'
        '    "Temporal": {"F1": bin_m["f1"], "ROC_AUC": bin_m["roc_auc"],\n'
        '                 "Precision": bin_m["precision"], "Recall": bin_m["recall"]},\n'
        '}).round(4)\n'
        'bin_cmp["delta"] = (bin_cmp["Temporal"] - bin_cmp["Random"]).round(4)\n'
        'print("BINARY:")\n'
        'display(bin_cmp)\n'
        '\n'
        'mc_cmp = pd.DataFrame({\n'
        '    "Random":   {"Macro_F1": rm["macro_f1"], "Weighted_F1": rm["weighted_f1"]},\n'
        '    "Temporal": {"Macro_F1": mc_m["macro_f1"], "Weighted_F1": mc_m["weighted_f1"]},\n'
        '}).round(4)\n'
        'mc_cmp["delta"] = (mc_cmp["Temporal"] - mc_cmp["Random"]).round(4)\n'
        'print("\\nMULTI-CLASS:")\n'
        'display(mc_cmp)'
    ),
    _md("### Per-class Macro F1 karşılaştırması"),
    _code(
        'per_cmp = pd.DataFrame({\n'
        '    "Random":   rm["per_class_f1"],\n'
        '    "Temporal": mc_m["per_class_f1"],\n'
        '}).round(3)\n'
        'per_cmp["delta"] = (per_cmp["Temporal"] - per_cmp["Random"]).round(3)\n'
        'display(per_cmp)'
    ),
    _md("## 6. Metrikleri `results/metrics.json` içine birleştir\n\n"
        "Eski key'ler (`binary`, `multiclass`) korunuyor; yeni sonuçlar `temporal_split` altına "
        "ekleniyor."),
    _code(
        'payload = {\n'
        '    "temporal_split": {\n'
        '        "train_months": sorted(TRAIN_MONTHS),\n'
        '        "test_months":  sorted(TEST_MONTHS),\n'
        '        "binary": {"primary_xgboost": bin_m},\n'
        '        "multiclass": {"primary_xgboost": mc_m, "labels": labels},\n'
        '    }\n'
        '}\n'
        '\n'
        'with open(METRICS_PATH) as f:\n'
        '    existing = json.load(f)\n'
        'existing.update(payload)\n'
        'with open(METRICS_PATH, "w") as f:\n'
        '    json.dump(existing, f, indent=2)\n'
        '\n'
        'print(f"saved -> {METRICS_PATH}")\n'
        'print("top-level keys:", list(existing.keys()))'
    ),
    _md(
        "## 7. Yorum (paper narrative)\n\n"
        "**Temporal split, random split'e kıyasla kaybı ölçer.** Negatif `delta` değeri model'in "
        "zamansal şifta duyarlı olduğunu; pozitif veya ~0 değer test ayı için veri dağılımının "
        "train'e yeterince benzediğini gösterir.\n\n"
        "Beklentiler:\n"
        "- **Binary ROC-AUC**: tree-based model'ler ranking için robust — genelde küçük kayıp (<0.02).\n"
        "- **Binary F1 / threshold-bağımlı metrikler**: threshold 0.5'te kayabilir — prior'ın "
        "(purchase rate) aylar arası farkı bunu direkt etkiliyor.\n"
        "- **Multi-class Macro F1**: kategori dağılımı veya fiyat aralıkları kayarsa sınıfsal "
        "ayırma zayıflar.\n\n"
        "Academic defense argümanı: \"Model training pipeline temporal split altında da "
        "büyük bir kayıp sergilemiyorsa, veride güçlü zamansal drift olmadığı kanıtlanır ve "
        "random i.i.d. varsayımı bu veri setinde makul\". Aksi durumda (büyük delta) temporal "
        "split sonuçlarını primary olarak raporlamak gerekir.\n"
    ),
]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, NB_PATH)
print("wrote:", NB_PATH)
