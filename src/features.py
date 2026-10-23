"""Data cleaning and feature engineering for freight rate prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["equipment", "pickup", "delivery", "pickup_region", "delivery_region"]
NUMERIC = [
    "log_distance", "distance", "weight", "weight_missing", "market_index",
    "market_index_missing", "quote_signal", "dayofweek", "is_weekend",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "lat_delta", "lon_delta", "lane_log_rpm", "lane_count",
    "log_market_index", "market_7d", "market_14d", "market_28d", "log_weight",
]
FEATURES = CATEGORICAL + NUMERIC

# Outlier band for rate-per-mile relative to the equipment/month median.
# Values outside this band form separate clusters (roughly 0.2-0.4x and 2.5-6x),
# which look like data-entry / unit errors rather than real market prices.
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.8


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues that apply to every split (train, validation, chart)."""
    df = df.copy()
    # Negative weights look like a sign error (same magnitude range as positive ones).
    if "weight" in df:
        df["weight"] = df["weight"].abs()
    return df


def flag_label_outliers(df: pd.DataFrame) -> pd.Series:
    """True for training rows whose posted rate is implausible (likely bad labels)."""
    rpm = df["posted_rate"] / df["distance"]
    ref = rpm.groupby([df["equipment"], df["date"].dt.month]).transform("median")
    ratio = rpm / ref
    return (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)


def _region(lon: pd.Series) -> pd.Series:
    return pd.cut(lon, bins=[-180, -105, -90, -80, 0], labels=["West", "Central", "East", "Atlantic"]).astype(str)


class FeatureBuilder:
    """Learns statistics on training data only, then applies them to any split."""

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.weight_median_ = train.groupby("equipment")["weight"].median()
        self.market_daily_ = train.groupby("date")["market_index"].mean()
        self.market_median_ = train["market_index"].median()
        # Smoothed lane-level log rate-per-mile (target encoding, fitted on train only).
        log_rpm = np.log(train["posted_rate"] / train["distance"])
        grp = log_rpm.groupby([train["pickup"], train["delivery"], train["equipment"]])
        stats = grp.agg(["mean", "count"])
        prior = log_rpm.groupby(train["equipment"]).mean()
        k = 10.0  # smoothing strength
        eq = stats.index.get_level_values(2)
        stats["smoothed"] = (stats["mean"] * stats["count"] + prior.reindex(eq).to_numpy() * k) / (stats["count"] + k)
        self.lane_stats_ = stats
        self.prior_ = prior
        # City coordinates (each city has exactly one lat/lon in the data).
        pick = train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        dele = train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        self.coords_ = pd.concat([pick, dele]).drop_duplicates("city").set_index("city")
        self.quote_median_ = train["quote_signal"].median()
        tmp = train.assign(pickup_region=_region(train["pickup_lon"]), delivery_region=_region(train["delivery_lon"]))
        self.categories_ = {c: sorted(tmp[c].astype(str).unique()) for c in CATEGORICAL}
        return self

    def transform(self, df: pd.DataFrame, market_by_date: pd.Series | None = None) -> pd.DataFrame:
        out = df.copy()
        # Rows without coordinates (e.g. the December chart inputs) get them from the city lookup.
        for side in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                col = f"{side}_{axis}"
                looked_up = out[side].map(self.coords_[axis])
                out[col] = out[col].fillna(looked_up) if col in out else looked_up
        out["weight_missing"] = out["weight"].isna().astype(int)
        out["weight"] = out["weight"].fillna(out["equipment"].map(self.weight_median_))

        if "market_index" not in out:
            out["market_index"] = np.nan
        out["market_index_missing"] = out["market_index"].isna().astype(int)
        daily = self.market_daily_ if market_by_date is None else pd.concat([self.market_daily_, market_by_date])
        daily = daily[~daily.index.duplicated(keep="last")]
        out["market_index"] = out["market_index"].fillna(out["date"].map(daily)).fillna(self.market_median_)

        if "quote_signal" not in out:
            out["quote_signal"] = np.nan
        out["quote_signal"] = out["quote_signal"].fillna(self.quote_median_)

        # Rolling averages of the daily market level: rates react to the market with some smoothing.
        full_daily = daily.sort_index().asfreq("D").interpolate(limit_direction="both")
        for w in (7, 14, 28):
            roll = np.log(full_daily).rolling(w, min_periods=1).mean()
            out[f"market_{w}d"] = out["date"].map(roll).fillna(np.log(self.market_median_))
        out["log_market_index"] = np.log(out["market_index"])
        out["log_weight"] = np.log1p(out["weight"])
        out["log_distance"] = np.log(out["distance"])
        out["dayofweek"] = out["date"].dt.dayofweek
        out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
        out["lat_delta"] = out["delivery_lat"] - out["pickup_lat"]
        out["lon_delta"] = out["delivery_lon"] - out["pickup_lon"]
        out["pickup_region"] = _region(out["pickup_lon"])
        out["delivery_region"] = _region(out["delivery_lon"])

        key = pd.MultiIndex.from_frame(out[["pickup", "delivery", "equipment"]])
        lane = self.lane_stats_.reindex(key)
        out["lane_log_rpm"] = lane["smoothed"].to_numpy()
        out["lane_log_rpm"] = out["lane_log_rpm"].fillna(out["equipment"].map(self.prior_))
        out["lane_count"] = lane["count"].fillna(0).to_numpy()

        for col in CATEGORICAL:
            out[col] = pd.Categorical(out[col].astype(str), categories=self.categories_[col])
        return out[FEATURES]
