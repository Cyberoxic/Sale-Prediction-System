# Rossmann Store Sales Prediction

A sales prediction system built using **Linear Regression** and **Random Forest** to forecast daily store sales, trained on the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales) from Kaggle.

## Overview

This project predicts daily sales for Rossmann stores using historical sales data, store metadata, and promotional/seasonal information. Two models are trained and compared — Linear Regression (interpretable baseline) and Random Forest (ensemble, handles non-linear relationships).

## Tech Stack

- Python
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Google Colab

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset from Kaggle:
- `train.csv` — daily sales, promo, and holiday data per store (compressed as `train.csv.gz` in this repo)
- `store.csv` — store metadata (type, assortment, competition distance, Promo2 details)

## Pipeline

1. **Merge** `train.csv` with `store.csv` on `Store` id
2. **Clean** — filter out closed stores (`Open == 0`), fill structurally-missing store metadata (e.g. `CompetitionDistance`, `Promo2SinceWeek`) rather than dropping rows
3. **Feature engineering** — extract `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear` from `Date`; one-hot encode categorical columns (`StoreType`, `Assortment`, `StateHoliday`, `PromoInterval`)
4. **Chronological train/test split** (80/20 by date, not random) to reflect real forecasting conditions
5. **Train** Linear Regression and Random Forest on identical train/test splits
6. **Evaluate** using MAE, MSE, R²

## How I Improved the Model

This project didn't start in its current state — it went through several rounds of debugging and refinement, each fixing a real weakness:

1. **Started simple.** The first version trained directly on `train.csv` alone, dropped the `Date` column entirely, and used a random 80/20 train/test split.

2. **Brought in more signal.** Realized the Rossmann dataset ships a second file, `store.csv`, with store-level metadata (type, assortment, competition distance, promo history) that wasn't being used at all. Merged it in on `Store` id — this unlocked features the model had no access to before.

3. **Stopped throwing away the date.** Instead of dropping `Date`, extracted `Year`, `Month`, `Day`, `DayOfWeek`, and `WeekOfYear` from it, since retail sales are strongly seasonal (weekday vs weekend, holidays, time of year).

4. **Cleaned the data properly.** Switched from a blanket `dropna()` (which would've silently deleted most rows once `store.csv` was merged, since most stores don't run `Promo2`) to targeted `fillna()` — filling missing values with medians or "not applicable" defaults depending on what the missingness actually meant.

5. **Removed noisy rows.** Filtered out closed stores (`Open == 0`), since these rows have `Sales == 0` by definition and were just dragging down the error metrics without adding predictive value.

6. **Fixed the evaluation methodology.** Switched from a random train/test split to a **chronological** split — since this is time-series data, a random split lets "future" days leak into training, giving an artificially rosy performance estimate.

7. **Found and fixed real data leakage.** An early version included `Customers` as a feature, which pushed R² up to 0.96. Investigated why the score looked "too good," and realized `Customers` isn't information you'd actually have in advance when forecasting future sales — it's essentially a proxy for `Sales` itself. Removed it, and the honest R² dropped to a more realistic ~0.88 for Random Forest.

8. **Added interpretability.** Added a Random Forest feature importance chart to show *what* the model is actually relying on, not just how accurate it is.

### Outcome

The end-to-end result is a model that's honest about its own limitations rather than one that looks artificially perfect. Random Forest explains **~88.6%** of the variance in daily sales with an average error of **~698 units** — using only information that would genuinely be available ahead of a forecast (no future data, no post-hoc customer counts).

### What this exercise reinforced

Beyond the model itself, this project was a hands-on lesson in **data leakage** — how a single innocuous-looking feature can quietly inflate performance metrics and hide the true difficulty of a problem. Catching it, understanding *why* it was wrong, and correcting it was a more valuable takeaway than the final R² number itself.

## Results

| Model | MAE | MSE | R² Score |
|---|---|---|---|
| Linear Regression | 1958.98 | 7,271,356.71 | 0.225 |
| Random Forest | 698.03 | 1,072,378.77 | 0.886 |

Random Forest substantially outperforms Linear Regression, capturing non-linear interactions between promotions, store type, competition, and seasonality that a linear model can't.

**Top features (Random Forest):** `CompetitionDistance`, `StoreType`, `Promo`, `Store`, `Assortment` — full breakdown in the notebook.

## How to Run

1. Clone the repo
2. `train.csv.gz` and `store.csv` are already included — unzip `train.csv.gz` first (`gunzip train.csv.gz`), or load it directly with `pd.read_csv("train.csv.gz", compression="gzip")`
3. Install dependencies: `pip install -r requirements.txt`
4. Open the notebook in Google Colab or Jupyter
5. Run all cells top to bottom (**Runtime → Restart session → Run all** in Colab)

## Possible Next Steps

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for Random Forest
- Lag features (e.g. previous week's average sales per store)
- Cross-validation instead of a single train/test split
- Save a trained model artifact (joblib/pickle) for reuse outside the notebook

## Author

[@cyberoxic](https://github.com/cyberoxic)
