"""Validate, train and predict freight rates.

Usage (from the repository root):
    python src/train.py

Outputs:
    validation_predictions.csv            12,000 rows: load_id,predicted_rate
    data/december_chart_inputs.csv        predicted_rate column filled (31 rows)
    reports/metrics.json                  all validation results
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, FEATURES, FeatureBuilder, clean, flag_label_outliers, load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SEED = 42

HALF_LIFE_DAYS = 30   # recency weighting: a load 30 days older counts half as much
RIDGE_ALPHA = 1.0
ONE_HOT = ["lane", "equipment", "pickup", "delivery", "dow"]
NUMERIC = ["log_distance", "log_distance_sq", "log_weight", "weight_missing", "log_market_index",
           "market_7d", "market_14d", "market_28d", "quote_signal"]


# ----------------------------------------------------------------------------- models
def _linear_frame(X: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["lane"] = (raw["pickup"] + "|" + raw["delivery"] + "|" + raw["equipment"]).to_numpy()
    X["dow"] = X["dayofweek"].astype(str)
    X["log_distance_sq"] = X["log_distance"] ** 2
    for c in ["equipment", "pickup", "delivery"]:
        X[c] = X[c].astype(str)
    return X


def recency_weights(train: pd.DataFrame) -> np.ndarray:
    age_days = (train["date"].max() - train["date"]).dt.days
    return (0.5 ** (age_days / HALF_LIFE_DAYS)).to_numpy()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, market_by_date=None, model: str = "ridge") -> np.ndarray:
    """Fit on cleaned `train`, return predicted posted_rate for `test`.

    The model predicts log(rate per mile); we convert back with exp(.) * distance.
    `market_by_date` is the daily mean market_index for the prediction period. It is a
    feature column (provided in validation.csv too), not a label, so this is not leakage.
    """
    if market_by_date is None and "market_index" in test:
        market_by_date = test.groupby("date")["market_index"].mean()
    fb = FeatureBuilder().fit(train)
    X_tr, X_te = fb.transform(train), fb.transform(test, market_by_date)
    y = np.log(train["posted_rate"] / train["distance"])
    w = recency_weights(train)

    if model == "ridge":
        pre = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT),
            ("scale", StandardScaler(), NUMERIC),
        ])
        est = make_pipeline(pre, Ridge(alpha=RIDGE_ALPHA))
        est.fit(_linear_frame(X_tr, train), y, ridge__sample_weight=w)
        log_rpm = est.predict(_linear_frame(X_te, test))
    elif model == "hgb":
        est = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=SEED,
        )
        est.fit(X_tr, y, sample_weight=w)
        log_rpm = est.predict(X_te)
    else:
        raise ValueError(model)
    return np.exp(log_rpm) * test["distance"].to_numpy()


def baseline_lane_median(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Simple benchmark: median rate-per-mile of the lane x equipment, times distance."""
    rpm = train["posted_rate"] / train["distance"]
    lane = rpm.groupby([train["pickup"], train["delivery"], train["equipment"]]).median()
    eq = rpm.groupby(train["equipment"]).median()
    key = pd.MultiIndex.from_frame(test[["pickup", "delivery", "equipment"]])
    r = pd.Series(lane.reindex(key).to_numpy(), index=test.index).fillna(test["equipment"].map(eq))
    return (r * test["distance"]).to_numpy()


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": round(float(np.mean(np.abs(err))), 2),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "MAPE_%": round(float(np.mean(np.abs(err) / y_true) * 100), 3),
        "MedianAPE_%": round(float(np.median(np.abs(err) / y_true) * 100), 3),
    }


# ----------------------------------------------------------------------------- main
def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    dev = clean(load(DATA / "train_test.csv"))
    outlier = flag_label_outliers(dev)
    print(f"Development rows: {len(dev):,} | label outliers excluded from training/scoring: {outlier.sum():,}")
    results: dict = {"outliers_removed": int(outlier.sum())}

    # 1) Main hold-out: train Jan-Aug, test Sep-Oct (same 2-month horizon as Nov-Dec validation).
    cut = pd.Timestamp("2025-09-01")
    tr = dev[(dev["date"] < cut) & ~outlier]
    te_all = dev[dev["date"] >= cut]
    clean_mask = (~outlier.loc[te_all.index]).to_numpy()
    md = te_all.groupby("date")["market_index"].mean()
    pred_ridge = fit_predict(tr, te_all, md, "ridge")
    pred_hgb = fit_predict(tr, te_all, md, "hgb")
    pred_base = baseline_lane_median(tr, te_all)
    y = te_all["posted_rate"].to_numpy()
    results["holdout_sep_oct"] = {
        "train_rows": int(len(tr)), "test_rows_clean": int(clean_mask.sum()), "test_rows_all": int(len(te_all)),
        "ridge_clean": metrics(y[clean_mask], pred_ridge[clean_mask]),
        "ridge_all_labels": metrics(y, pred_ridge),
        "hgb_clean": metrics(y[clean_mask], pred_hgb[clean_mask]),
        "baseline_lane_median_clean": metrics(y[clean_mask], pred_base[clean_mask]),
    }
    hold = te_all.assign(pred=pred_ridge)[clean_mask]
    hold["ape"] = (hold["pred"] - hold["posted_rate"]).abs() / hold["posted_rate"] * 100
    results["holdout_by_equipment_MAPE_%"] = hold.groupby("equipment")["ape"].mean().round(3).to_dict()
    hold[["date", "equipment", "distance", "posted_rate", "pred"]].to_csv(REPORTS / "holdout_predictions.csv", index=False)

    # 2) Rolling-origin backtest: expanding training window, predict the next 2 months.
    folds = []
    for start_month in [6, 7, 8, 9]:
        start = pd.Timestamp(2025, start_month, 1)
        end = start + pd.offsets.MonthBegin(2)
        tr_f = dev[(dev["date"] < start) & ~outlier]
        te_f = dev[(dev["date"] >= start) & (dev["date"] < end) & ~outlier]
        row = {"train_until": str((start - pd.Timedelta(days=1)).date()),
               "test_period": f"{start.date()} to {(end - pd.Timedelta(days=1)).date()}"}
        for name in ["ridge", "hgb"]:
            row[f"{name}_MAPE_%"] = metrics(te_f["posted_rate"], fit_predict(tr_f, te_f, model=name))["MAPE_%"]
        row["baseline_MAPE_%"] = metrics(te_f["posted_rate"], baseline_lane_median(tr_f, te_f))["MAPE_%"]
        folds.append(row)
        print(f"  fold {row['test_period']}: ridge {row['ridge_MAPE_%']:.2f}% | "
              f"hgb {row['hgb_MAPE_%']:.2f}% | baseline {row['baseline_MAPE_%']:.2f}%")
    results["rolling_backtest"] = folds

    # 3) Final model on ALL clean development data -> validation.csv + December chart.
    full = dev[~outlier]
    val = clean(load(DATA / "validation.csv"))
    val_market_daily = val.groupby("date")["market_index"].mean()
    val["predicted_rate"] = fit_predict(full, val, val_market_daily, "ridge")

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    cols = list(dec.columns)
    dec["predicted_rate"] = fit_predict(full, clean(dec.drop(columns="predicted_rate")), val_market_daily, "ridge").round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec[cols].to_csv(DATA / "december_chart_inputs.csv", index=False)

    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    h = results["holdout_sep_oct"]
    print("Hold-out Sep-Oct  ridge   :", h["ridge_clean"])
    print("Hold-out Sep-Oct  hgb     :", h["hgb_clean"])
    print("Hold-out Sep-Oct  baseline:", h["baseline_lane_median_clean"])
    print("Wrote validation_predictions.csv and data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
