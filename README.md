# Hourly Bike Rental Prediction

Can we predict how many bikes get rented each hour from the weather and the time of day?

This project uses the Capital Bikeshare data for Washington, D.C. (2011 to 2012) to forecast hourly rental counts. It walks through cleaning, feature engineering, comparing eight regression models, and checking how well the best one holds up when it has to predict the future rather than shuffled data.

## Data

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), hourly file: 17,379 rows, 17 columns, no missing values. Features cover season, month, hour, weekday, holiday, weather situation, temperature, feels like temperature, humidity, and wind speed. The target is `cnt`, total rentals in the hour.

The CSV is not committed. Download the dataset zip from UCI and put `hour.csv` next to the notebook.

## Approach

1. Exploratory analysis of usage by hour, weekday versus weekend, season, and weather.
2. Kept the 505 IQR outliers, since they are real peak demand hours.
3. Dropped the leakage columns (`casual` and `registered` add up to the target), the row index, and the redundant `atemp` and `windspeed` columns, then one hot encoded the categorical features.
4. Compared Linear Regression, Ridge, Huber, ElasticNet, Decision Tree, Gradient Boosting, Random Forest, and Extra Trees.
5. Evaluated the chosen Random Forest with `TimeSeriesSplit` cross validation and a chronological 80/20 holdout, then tuned it with a grid search.
6. Tested a one hour lag feature to give the model a sense of recent demand.

## Results

On a random train/test split the tree ensembles look excellent:

| Model | R² | RMSE |
|---|---:|---:|
| Extra Trees | 0.93 | 47.7 |
| Random Forest | 0.92 | 51.0 |
| Decision Tree | 0.85 | 71.0 |
| Gradient Boosting | 0.80 | 81.2 |
| Linear Regression | 0.68 | 103.2 |

A random split lets the model train on hours on either side of each test hour, so those numbers are optimistic. When the Random Forest has to predict later dates from earlier ones, performance drops:

| Evaluation | R² | RMSE | MAE |
|---|---:|---:|---:|
| 5 fold time series CV | 0.69 | 94.5 | |
| Chronological 80/20 holdout, tuned | 0.83 | 89.9 | 61.4 |

The model is accurate up to roughly 400 rentals an hour and consistently underpredicts the busiest peaks. The learning curve shows it fits the training data almost perfectly while validation scores stay much lower, so the main limit is generalizing across time rather than model capacity.

## Next steps

* Longer range lags (24 hour, 7 day) and rolling averages
* Gradient boosted models such as XGBoost or LightGBM
* Evaluating peak hours separately, since that is where the error concentrates

## Running it

```bash
pip install -r requirements.txt
jupyter notebook bike_rental_prediction.ipynb
```
