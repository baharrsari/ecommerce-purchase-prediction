"""Comprehensive verification of the Phase 1-4 pipeline.

Five independent checks, each prints its own PASS/FAIL lines:

1. Target leakage scan  — features cannot deterministically predict target.
2. Sanity checks        — row counts, rates, invariants match CLAUDE.md.
3. Spot check (n=5)     — manually re-aggregate random sessions from raw CSVs
                          and compare to stored parquet values.
4. 5-fold CV stability  — stratified folds for both binary and multi-class.
5. Edge cases           — pure-purchase sessions, NaN handling, etc.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import MONTHLY_FILES, load_month
from src.feature_engineering import FEATURE_COLS
from src.model_binary import train_baseline as train_binary_baseline
from src.model_binary import train_primary as train_binary_primary
from src.model_multiclass import train_baseline as train_multi_baseline
from src.model_multiclass import train_primary as train_multi_primary

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
RANDOM_STATE = 42

PASS = "PASS"
FAIL = "FAIL"


def _line(char: str = "-", n: int = 78) -> None:
    print(char * n)


def _hdr(title: str) -> None:
    _line("=")
    print(title)
    _line("=")


# ---------------------------------------------------------------------------
# 1. Leakage scan
# ---------------------------------------------------------------------------
def check_leakage(sessions_all: pd.DataFrame, sessions_cat: pd.DataFrame) -> list[tuple[str, str]]:
    _hdr("1. TARGET LEAKAGE SCAN")
    results: list[tuple[str, str]] = []

    print("\n[binary] invariant: n_events == n_view + n_cart + n_remove")
    left = sessions_all["n_events"].to_numpy()
    right = (sessions_all["n_view"] + sessions_all["n_cart"] + sessions_all["n_remove"]).to_numpy()
    eq_all = bool((left == right).all())
    print(f"  equal for all {len(sessions_all):,} sessions: {eq_all}")
    results.append(("invariant: n_events=n_view+n_cart+n_remove", PASS if eq_all else FAIL))

    print("\n[binary] per-feature univariate AUC w.r.t. purchased")
    y = sessions_all["purchased"].to_numpy()
    print(f"  {'feature':<24}{'AUC':>8}{'|corr|':>10}")
    aucs = []
    for f in FEATURE_COLS:
        x = sessions_all[f].to_numpy()
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            auc = float("nan")
        corr = abs(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
        aucs.append(auc)
        print(f"  {f:<24}{auc:>8.4f}{corr:>10.4f}")
    max_auc = max([a for a in aucs if not np.isnan(a)])
    ok = max_auc < 0.99
    print(f"  max univariate AUC = {max_auc:.4f}  (expect < 0.99)")
    results.append(("no single feature perfectly separates target", PASS if ok else FAIL))

    print("\n[multi] features derived from non-purchase events only")
    # Sanity: unique_categories <= unique_products (one product has exactly one category)
    good = bool((sessions_all["unique_categories"] <= sessions_all["unique_products"]
                 + (sessions_all["n_events"] == 0).astype(int)).all())
    print(f"  unique_categories <= unique_products: {good}")
    results.append(("unique_categories <= unique_products", PASS if good else FAIL))

    print("\n[multi] per-feature univariate macro-F1 proxy (no perfect class predictor)")
    # For each feature, bin into deciles and check if any single bin maps to exactly one class
    y_m = sessions_cat["main_cat"].astype("string").to_numpy()
    susp = []
    for f in FEATURE_COLS:
        x = sessions_cat[f].to_numpy()
        bins = pd.qcut(pd.Series(x), q=10, duplicates="drop")
        tbl = pd.crosstab(bins, y_m, normalize="index")
        max_cell = float(tbl.values.max())
        if max_cell > 0.95:
            susp.append((f, max_cell))
    if susp:
        for f, v in susp:
            print(f"  {f}: max decile purity = {v:.3f}")
    else:
        print("  no feature has any decile with >95% class purity")
    ok = len(susp) == 0
    results.append(("no feature has >95% class-purity decile", PASS if ok else FAIL))

    return results


# ---------------------------------------------------------------------------
# 2. Sanity checks
# ---------------------------------------------------------------------------
def check_sanity(sessions_all: pd.DataFrame, sessions_cat: pd.DataFrame) -> list[tuple[str, str]]:
    _hdr("2. SANITY CHECKS")
    results: list[tuple[str, str]] = []

    n_all = len(sessions_all)
    print(f"\n  total sessions: {n_all:,}  (expect ~3.6M)")
    results.append(("~3.6M sessions in sessions_all", PASS if 3_400_000 <= n_all <= 3_700_000 else FAIL))

    pr = sessions_all["purchased"].mean()
    print(f"  purchase rate:  {pr:.4f}  (expect ~0.033)")
    results.append(("purchase rate in [0.030, 0.036]", PASS if 0.030 <= pr <= 0.036 else FAIL))

    n_cat = len(sessions_cat)
    print(f"  sessions_cat:   {n_cat:,}  (expect ~131K)")
    results.append(("sessions_cat ~131K", PASS if 125_000 <= n_cat <= 135_000 else FAIL))

    dist = sessions_cat["main_cat"].value_counts(normalize=True).round(3)
    print(f"\n  main_cat shares:")
    for k, v in dist.items():
        print(f"    {k:<14}{v:.3f}")
    share_ok = (
        0.40 <= dist.get("appliances", 0) <= 0.50
        and 0.08 <= dist.get("accessories", 0) <= 0.11
    )
    results.append(("category shares within ±5% of CLAUDE.md", PASS if share_ok else FAIL))

    print(f"\n  no NaN in feature columns:")
    nans = sessions_all[FEATURE_COLS].isna().sum().sum()
    print(f"    sessions_all: {nans}")
    nans_c = sessions_cat[FEATURE_COLS].isna().sum().sum()
    print(f"    sessions_cat: {nans_c}")
    results.append(("feature columns NaN-free", PASS if nans == 0 and nans_c == 0 else FAIL))

    neg_counts = (sessions_all[["n_view", "n_cart", "n_remove", "n_events"]] < 0).any(axis=None)
    print(f"\n  no negative counts: {not neg_counts}")
    results.append(("counts are non-negative", PASS if not neg_counts else FAIL))

    uniq = sessions_all["user_session"].is_unique
    print(f"  user_session unique in sessions_all: {uniq}")
    results.append(("user_session unique in sessions_all", PASS if uniq else FAIL))

    uniq_c = sessions_cat["user_session"].is_unique
    print(f"  user_session unique in sessions_cat: {uniq_c}")
    results.append(("user_session unique in sessions_cat", PASS if uniq_c else FAIL))

    cat_subset = sessions_cat["user_session"].isin(set(sessions_all["user_session"]))
    all_in = bool(cat_subset.all())
    print(f"  sessions_cat is subset of sessions_all: {all_in}")
    results.append(("sessions_cat subset of sessions_all", PASS if all_in else FAIL))

    return results


# ---------------------------------------------------------------------------
# 3. Spot check — manual aggregation on 5 random sessions
# ---------------------------------------------------------------------------
def _manual_aggregate(rows: pd.DataFrame) -> dict[str, float]:
    """Compute features the same way preprocessing.aggregate_sessions does,
    on a single session's raw events.
    """
    pre = rows[rows["event_type"] != "purchase"]
    n_view = int((pre["event_type"] == "view").sum())
    n_cart = int((pre["event_type"] == "cart").sum())
    n_remove = int((pre["event_type"] == "remove_from_cart").sum())
    n_events = int(len(pre))
    purchased = int((rows["event_type"] == "purchase").any())
    if n_events == 0:
        return dict(
            n_view=0, n_cart=0, n_remove=0, n_events=0,
            avg_price=0.0, max_price=0.0, min_price=0.0, price_std=0.0,
            unique_products=0, unique_categories=0, has_brand=0,
            session_duration_sec=0.0, purchased=purchased,
        )
    prices = pre["price"].astype("float64")
    return dict(
        n_view=n_view,
        n_cart=n_cart,
        n_remove=n_remove,
        n_events=n_events,
        avg_price=float(prices.mean()),
        max_price=float(prices.max()),
        min_price=float(prices.min()),
        price_std=float(prices.std(ddof=1)) if n_events > 1 else 0.0,
        unique_products=int(pre["product_id"].nunique()),
        unique_categories=int(pre["category_id"].nunique()),
        has_brand=int(pre["brand"].notna().any()),
        session_duration_sec=float(
            (pre["event_time"].max() - pre["event_time"].min()).total_seconds()
        ),
        purchased=purchased,
    )


def spot_check(sessions_all: pd.DataFrame, n: int = 5) -> list[tuple[str, str]]:
    _hdr("3. SPOT CHECK: manual re-aggregation vs parquet (5 random sessions)")
    rng = np.random.default_rng(RANDOM_STATE)

    # Strategic sample: 2 purchase, 2 non-purchase, 1 edge (n_events=0 or very large)
    pur = sessions_all[sessions_all["purchased"] == 1].sample(2, random_state=42)
    npur = sessions_all[sessions_all["purchased"] == 0].sample(2, random_state=42)
    edge_pool = sessions_all[sessions_all["n_events"] == 0]
    if len(edge_pool) > 0:
        edge = edge_pool.sample(1, random_state=42)
    else:
        edge = sessions_all.sample(1, random_state=43)
    sample = pd.concat([pur, npur, edge], ignore_index=True)
    target_ids = set(sample["user_session"].tolist())
    print(f"  sampled {len(target_ids)} sessions; scanning raw CSVs...")

    # Scan all months, keep only rows in target set
    collected: list[pd.DataFrame] = []
    for fname in MONTHLY_FILES:
        print(f"    reading {fname} ...")
        df = load_month(RAW / fname)
        hit = df[df["user_session"].isin(target_ids)]
        if len(hit):
            collected.append(hit)
        del df
    raw = pd.concat(collected, ignore_index=True)
    print(f"  collected {len(raw):,} raw events across {raw['user_session'].nunique()} sessions")
    # main_cat not needed for the feature-level comparison below, so skip fill_main_cat

    # For each sampled session, manual agg vs parquet row
    mismatches = 0
    tol = 1e-3
    for _, parq_row in sample.iterrows():
        sid = parq_row["user_session"]
        srows = raw[raw["user_session"] == sid].sort_values("event_time")
        computed = _manual_aggregate(srows)

        print(f"\n  session: {sid}")
        print(f"    raw events: {len(srows)}  (types: {srows['event_type'].value_counts().to_dict()})")
        print(f"    {'feature':<24}{'manual':>14}{'parquet':>14}{'ok':>6}")
        row_ok = True
        for k, v_m in computed.items():
            v_p = float(parq_row[k]) if k != "purchased" else int(parq_row[k])
            if isinstance(v_m, int) or k == "purchased":
                ok = int(v_m) == int(v_p)
            else:
                ok = abs(v_m - v_p) <= tol or (
                    abs(v_p) > 0 and abs(v_m - v_p) / abs(v_p) <= 0.01
                )
            flag = "OK" if ok else "MISS"
            if not ok:
                row_ok = False
            print(f"    {k:<24}{v_m:>14.4f}{v_p:>14.4f}{flag:>6}")
        if not row_ok:
            mismatches += 1

    result = PASS if mismatches == 0 else FAIL
    print(f"\n  mismatched sessions: {mismatches} / {len(sample)}  -> {result}")
    return [("spot-check 5 sessions vs manual aggregation", result)]


# ---------------------------------------------------------------------------
# 4. 5-fold CV
# ---------------------------------------------------------------------------
def cv_binary(sessions_all: pd.DataFrame, k: int = 5) -> list[tuple[str, str]]:
    _hdr(f"4a. {k}-FOLD CV — BINARY CLASSIFIER (XGBoost)")
    X = sessions_all[FEATURE_COLS].astype("float32")
    y = sessions_all["purchased"].astype("int8")
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    f1s, aucs = [], []
    for i, (tr, te) in enumerate(kf.split(X, y), 1):
        t0 = time.perf_counter()
        model = train_binary_primary(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        f1 = f1_score(y.iloc[te], pred)
        auc = roc_auc_score(y.iloc[te], proba)
        f1s.append(f1); aucs.append(auc)
        print(f"  fold {i}/{k}: f1={f1:.4f}  roc_auc={auc:.4f}  ({time.perf_counter() - t0:.1f}s)")

    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
    a_mean, a_std = float(np.mean(aucs)), float(np.std(aucs))
    print(f"\n  mean +/- std: f1={f1_mean:.4f} +/- {f1_std:.4f}  roc_auc={a_mean:.4f} +/- {a_std:.4f}")
    ok = f1_std < 0.02 and a_std < 0.01
    return [(f"binary CV stability (std_f1<0.02, std_auc<0.01)", PASS if ok else FAIL)]


def cv_multi(sessions_cat: pd.DataFrame, k: int = 5) -> list[tuple[str, str]]:
    _hdr(f"4b. {k}-FOLD CV — MULTI-CLASS CLASSIFIER (XGBoost)")
    X = sessions_cat[FEATURE_COLS].astype("float32")
    y_raw = sessions_cat["main_cat"].astype("string")
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    macros, weighteds = [], []
    for i, (tr, te) in enumerate(kf.split(X, y), 1):
        t0 = time.perf_counter()
        model = train_multi_primary(X.iloc[tr], y[tr], n_classes=n_classes)
        pred = model.predict(X.iloc[te])
        mf = f1_score(y[te], pred, average="macro")
        wf = f1_score(y[te], pred, average="weighted")
        macros.append(mf); weighteds.append(wf)
        print(f"  fold {i}/{k}: macro_f1={mf:.4f}  weighted_f1={wf:.4f}  ({time.perf_counter() - t0:.1f}s)")

    m_mean, m_std = float(np.mean(macros)), float(np.std(macros))
    w_mean, w_std = float(np.mean(weighteds)), float(np.std(weighteds))
    print(f"\n  mean +/- std: macro_f1={m_mean:.4f} +/- {m_std:.4f}  weighted_f1={w_mean:.4f} +/- {w_std:.4f}")
    ok = m_std < 0.02 and w_std < 0.02
    return [(f"multi CV stability (std_macro<0.02, std_weighted<0.02)", PASS if ok else FAIL)]


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------
def edge_cases(sessions_all: pd.DataFrame, sessions_cat: pd.DataFrame) -> list[tuple[str, str]]:
    _hdr("5. EDGE CASES")
    results: list[tuple[str, str]] = []

    # Pure-purchase sessions (no view/cart/remove)
    pure = sessions_all[sessions_all["n_events"] == 0]
    print(f"\n  pure-purchase sessions (n_events=0): {len(pure):,}")
    if len(pure) > 0:
        pur_pct = pure["purchased"].mean()
        all_zero_feat = (pure[[c for c in FEATURE_COLS if c != "has_brand"]] == 0).all(axis=None)
        print(f"    purchase rate among them: {pur_pct:.4f}")
        print(f"    all features zero:        {all_zero_feat}")
        results.append(("pure-purchase sessions handled", PASS if pur_pct > 0.5 else FAIL))
    else:
        print("    none present (OK)")
        results.append(("pure-purchase sessions handled", PASS))

    # Single-event sessions
    single = sessions_all[sessions_all["n_events"] == 1]
    print(f"\n  single non-purchase event sessions (n_events=1): {len(single):,}  ({len(single)/len(sessions_all):.2%})")
    ps = single["price_std"]
    ok = bool((ps == 0).all())
    print(f"    price_std == 0 for all of them: {ok}")
    results.append(("single-event sessions have price_std=0", PASS if ok else FAIL))

    # Extreme outliers (possible bots)
    p99_events = sessions_all["n_events"].quantile(0.999)
    bots = sessions_all[sessions_all["n_events"] > p99_events]
    print(f"\n  p99.9 n_events cut: {p99_events:.0f}")
    print(f"  sessions above cut: {len(bots):,}  (extreme outliers / bots)")
    print(f"    among those, purchase rate: {bots['purchased'].mean():.4f}  (vs global {sessions_all['purchased'].mean():.4f})")

    # Negative prices
    neg = (sessions_all["avg_price"] < 0).sum()
    print(f"\n  sessions with negative avg_price: {neg:,}  (data-quality note — refund/error records)")

    # Purchased sessions that wouldn't be in sessions_cat
    p_no_cat = sessions_all[(sessions_all["purchased"] == 1) & (~sessions_all["user_session"].isin(set(sessions_cat["user_session"])))]
    print(f"\n  purchase sessions without main_cat: {len(p_no_cat):,}  (unmappable category_id)")

    # Dtypes check
    print(f"\n  key dtypes:")
    for c in ["purchased", "n_events", "avg_price", "user_session"]:
        print(f"    {c:<16}{str(sessions_all[c].dtype)}")

    # class_weight coverage
    y_m = sessions_cat["main_cat"].astype("string").to_numpy()
    sw = compute_sample_weight("balanced", y_m)
    w_ratio = sw.max() / sw.min()
    print(f"\n  balanced sample_weight ratio (max/min): {w_ratio:.2f}")
    results.append(("sample_weight ratio sensible", PASS if 3 <= w_ratio <= 10 else FAIL))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    print(f"Loading parquets from {PROCESSED} ...")
    sessions_all = pd.read_parquet(PROCESSED / "sessions_all.parquet")
    sessions_cat = pd.read_parquet(PROCESSED / "sessions_cat.parquet")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    all_results: list[tuple[str, str]] = []
    all_results += check_leakage(sessions_all, sessions_cat)
    all_results += check_sanity(sessions_all, sessions_cat)
    all_results += spot_check(sessions_all)
    all_results += cv_binary(sessions_all)
    all_results += cv_multi(sessions_cat)
    all_results += edge_cases(sessions_all, sessions_cat)

    _hdr("FINAL SUMMARY")
    width = max(len(name) for name, _ in all_results) + 2
    n_pass = sum(1 for _, r in all_results if r == PASS)
    for name, result in all_results:
        print(f"  [{result}] {name}")
    print(f"\n  {n_pass}/{len(all_results)} checks PASSED")
    print(f"\n  total verification time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
