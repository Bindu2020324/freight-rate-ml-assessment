# Freight Rate Prediction

Predicts the posted rate (USD) of a truckload from lane, distance, equipment, weight, date and
market signals. Development data covers Jan–Oct 2025; the 12,000 loads to predict are Nov–Dec 2025.

## Results (time-based hold-out: train Jan–Aug, test Sep–Oct 2025)

| Model | MAE ($) | MAPE | Median APE |
|---|---|---|---|
| **Ridge on log rate-per-mile, recency-weighted (final)** | **77.3** | **3.25%** | **3.04%** |
| Gradient boosting (same features) | 103.0 | 4.44% | 4.34% |
| Baseline: lane median $/mile × distance | 88.2 | 3.74% | 3.08% |

Rolling 2-month backtest (4 folds, Jun → Oct): final model MAPE 2.1% – 3.5%, beating both
alternatives in every fold. Full numbers: `reports/metrics.json`.

## Project structure

```
data/                      input CSVs (train_test, validation, template, december_chart_inputs)
src/features.py            cleaning, outlier flagging, feature engineering (fit on train only)
src/train.py               validation (hold-out + rolling backtest), final training, predictions
src/eda.py                 data-quality checks and figures -> reports/figures/
score.py                   provided scorer (validates outputs, draws December chart)
validation_predictions.csv final submission file (load_id,predicted_rate)
reports/                   metrics.json, figures, report
```

## How to run

```bash
python -m pip install -r requirements.txt
python src/train.py      # validation + final predictions (~1 min)
python src/eda.py        # data checks + figures
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` writes `scorer_results/candidate_december.png`.

## Approach in brief

1. **Cleaning** – negative weights are sign errors (made positive); missing weight → equipment median
   plus a missing flag; missing market_index → that day's average; 677 loads (1.4%) whose $/mile is
   < 0.6× or > 1.8× the equipment-month median are removed from training (they form separate
   clusters that look like entry/unit errors).
2. **Target** – `log(posted_rate / distance)`. Removes the dominant distance effect and makes errors
   relative, which suits prices that span $200 – $7,000.
3. **Features** – lane / pickup / delivery / equipment (one-hot), log distance, log weight,
   day of week, market_index and its 7/14/28-day rolling averages, quote_signal.
4. **Split** – strictly by time (no random split), because the task is predicting future months.
5. **Model** – Ridge regression with sample weights that halve every 30 days, so the price level
   reflects the most recent market.
