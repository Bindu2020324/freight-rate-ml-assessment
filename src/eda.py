"""Exploratory data analysis: prints data-quality checks and saves figures to reports/figures.

Usage (from the repository root):
    python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import flag_label_outliers, load

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
COLOR, ACCENT = "#064A56", "#D2691E"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E8EA", linewidth=0.8)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    dev = load(ROOT / "data" / "train_test.csv")
    val = load(ROOT / "data" / "validation.csv")
    dev["rpm"] = dev["posted_rate"] / dev["distance"]

    print("=== Shape & date ranges ===")
    print(f"train_test: {dev.shape}, {dev['date'].min().date()} -> {dev['date'].max().date()}")
    print(f"validation: {val.shape}, {val['date'].min().date()} -> {val['date'].max().date()}")
    print("\n=== Missing values ===")
    print(dev.isna().sum()[dev.isna().sum() > 0])
    print("\n=== Data-quality checks ===")
    print("negative weights (train / validation):", int((dev["weight"] < 0).sum()), "/", int((val["weight"] < 0).sum()))
    print("duplicate load_id:", int(dev["load_id"].duplicated().sum()))
    outliers = flag_label_outliers(dev)
    print("rate-per-mile outliers:", int(outliers.sum()), f"({outliers.mean():.2%})")
    lanes_train = set(zip(dev["pickup"], dev["delivery"]))
    unseen = np.mean([lane not in lanes_train for lane in zip(val["pickup"], val["delivery"])])
    print(f"validation loads on lanes never seen in training: {unseen:.1%}")
    print("\n=== Correlation with posted_rate ===")
    print(dev[["distance", "weight", "market_index", "quote_signal", "posted_rate"]].corr()["posted_rate"].round(3))

    # 1. Rate vs distance
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ok = dev[~outliers].sample(6000, random_state=0)
    bad = dev[outliers]
    ax.scatter(ok["distance"], ok["posted_rate"], s=4, alpha=0.35, color=COLOR, label="normal loads")
    ax.scatter(bad["distance"], bad["posted_rate"], s=8, alpha=0.8, color=ACCENT, label="flagged outliers")
    ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Posted rate ($)"); ax.legend(frameon=False)
    style(ax, "Rate is driven mainly by distance; outliers sit off the main band")
    fig.tight_layout(); fig.savefig(FIG / "rate_vs_distance.png"); plt.close(fig)

    # 2. Rate-per-mile ratio histogram (shows the outlier clusters)
    ref = dev["rpm"].groupby([dev["equipment"], dev["date"].dt.month]).transform("median")
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=160)
    ax.hist(np.log10(dev["rpm"] / ref), bins=120, color=COLOR)
    ax.set_yscale("log"); ax.set_xlabel("log10(rate per mile / equipment-month median)"); ax.set_ylabel("loads (log scale)")
    for x in (np.log10(0.6), np.log10(1.8)):
        ax.axvline(x, color=ACCENT, linestyle="--")
    style(ax, "Rate-per-mile: main peak plus two separate error clusters")
    fig.tight_layout(); fig.savefig(FIG / "rpm_outliers.png"); plt.close(fig)

    # 3. Market index vs rate level over time
    clean = dev[~outliers].copy()
    clean["log_rpm"] = np.log(clean["rpm"])
    clean["lane_adj"] = clean["log_rpm"] - clean.groupby(["pickup", "delivery", "equipment"])["log_rpm"].transform("mean")
    weekly = clean.set_index("date").resample("W")[["lane_adj", "market_index"]].mean()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    ax.plot(weekly.index, (np.exp(weekly["lane_adj"]) - 1) * 100, color=COLOR, lw=2, label="lane-adjusted rate (% vs lane avg)")
    ax.set_ylabel("% vs lane average")
    ax2 = ax.twinx(); ax2.plot(weekly.index, weekly["market_index"], color=ACCENT, lw=2, label="market_index")
    ax2.set_ylabel("market_index"); ax2.spines[["top"]].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines(); ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")
    style(ax, "Market-wide price level moves with the market index")
    fig.tight_layout(); fig.savefig(FIG / "market_index_vs_rate.png"); plt.close(fig)

    # 4. Rate per mile by equipment and by day of week
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160)
    eq = clean.groupby("equipment")["rpm"].median().sort_values()
    axes[0].bar(eq.index, eq.values, color=COLOR); axes[0].set_ylabel("median $ per mile")
    style(axes[0], "By equipment")
    dow = clean.groupby(clean["date"].dt.dayofweek)["rpm"].median()
    axes[1].plot(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, marker="o", color=COLOR, lw=2)
    style(axes[1], "By day of week")
    fig.tight_layout(); fig.savefig(FIG / "rpm_by_equipment_dow.png"); plt.close(fig)

    # 5. Hold-out: predicted vs actual (needs src/train.py to have run first)
    hold_path = ROOT / "reports" / "holdout_predictions.csv"
    if hold_path.exists():
        h = pd.read_csv(hold_path)
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=160)
        ax.scatter(h["posted_rate"], h["pred"], s=3, alpha=0.3, color=COLOR)
        lim = [0, max(h["posted_rate"].max(), h["pred"].max())]
        ax.plot(lim, lim, color=ACCENT, lw=1.5)
        ax.set_xlabel("Actual posted rate ($)"); ax.set_ylabel("Predicted rate ($)")
        style(ax, "Hold-out (Sep-Oct 2025): predicted vs actual")
        fig.tight_layout(); fig.savefig(FIG / "holdout_pred_vs_actual.png"); plt.close(fig)
    print(f"\nFigures saved to {FIG}")


if __name__ == "__main__":
    main()
