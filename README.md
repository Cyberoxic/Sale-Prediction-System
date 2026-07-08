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
- `train.csv` — daily sales, promo, and holiday data per store
- `store.csv` — store metadata (type, assortment, competition distance, Promo2 details)

## Pipeline

1. **Merge** `train.csv` with `store.csv` on `Store` id
2. **Clean** — filter out closed stores (`Open == 0`), fill structurally-missing store metadata (e.g. `CompetitionDistance`, `Promo2SinceWeek`) rather than dropping rows
3. **Feature engineering** — extract `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear` from `Date`; one-hot encode categorical columns (`StoreType`, `Assortment`, `StateHoliday`, `PromoInterval`)
4. **Chronological train/test split** (80/20 by date, not random) to reflect real forecasting conditions
5. **Train** Linear Regression and Random Forest on identical train/test splits
6. **Evaluate** using MAE, MSE, R²

## Data Leakage Note

An earlier version of this model included `Customers` as a feature, which produced an inflated R² of 0.97. This was **data leakage** — `Customers` isn't known ahead of time when forecasting future sales, and it's highly correlated with `Sales` almost by definition. It was removed to reflect a realistic forecasting scenario. The results below are the corrected, leakage-free numbers.

## Results

| Model | MAE | MSE | R² Score |
|---|---|---|---|
| Linear Regression | 2177.52 | 7,860,171.40 | 0.178 |
| Random Forest | 878.17 | 1,545,643.46 | 0.838 |

Random Forest substantially outperforms Linear Regression, capturing non-linear interactions between promotions, store type, competition, and seasonality that a linear model can't.

**Top features (Random Forest):** `CompetitionDistance`, `StoreType`, `Promo`, `Store`, `Assortment` — full breakdown in the notebook.

## How to Run

1. Clone the repo
2. Download `train.csv` and `store.csv` from Kaggle and place them in the project directory
3. Open the notebook in Google Colab or Jupyter
4. Run all cells top to bottom (**Runtime -> Restart session -> Run all** in Colab)

## Project Updates

Changelog of improvements made after the initial version:

- **Merged `store.csv`** into the training data (StoreType, Assortment, CompetitionDistance, Promo2 details) — previously the model only used `train.csv`
- **Fixed missing values properly** — replaced blanket `dropna()` with targeted `fillna()` for structurally-missing store metadata (e.g. stores with no Promo2 or no recorded competitor), instead of discarding data
- **Filtered closed stores** (`Open == 0`) since these rows have `Sales == 0` and add noise, not signal
- **Added date-based feature engineering** — extracted `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear` from `Date` instead of dropping the column entirely
- **Fixed `StateHoliday` dtype** — cast to string before one-hot encoding to avoid duplicate/broken dummy columns
- **Switched to a chronological train/test split** (80/20 by date) instead of a random split, since random splitting on time-series data leaks future information into training
- **Added Random Forest feature importance** analysis and chart
- **Identified and fixed data leakage** — `Customers` was included as a feature and inflated R² to 0.96; removed it since customer count isn't known ahead of time when forecasting, and it's essentially a proxy for sales

## Possible Next Steps

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for Random Forest
- Lag features (e.g. previous week's average sales per store)
- Cross-validation instead of a single train/test split

## Author

[@cyberoxic](https://github.com/cyberoxic)
