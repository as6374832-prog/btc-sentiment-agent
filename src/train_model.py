
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the combined dataset
df = pd.read_csv("data/btc_training_data.csv")

# Features available at prediction time
features = [
    "Close",
    "Volume",
    "tweet_count",
    "avg_sentiment",
    "likes_total",
    "reposts_total",
    "label_negative",
    "label_neutral",
    "label_positive",
    "return_1d",
    "return_3d",
]

target = "target_next_day"

# Convert features and target to numeric, then remove incomplete rows
for column in features + [target]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [target]).sort_values("day")
df = df.reset_index(drop=True)

if len(df) < 30:
    raise ValueError("Not enough complete rows to train and test.")

# Chronological split: first 80% train, final 20% test
split = int(len(df) * 0.8)
X_train = df[features].iloc[:split]
X_test = df[features].iloc[split:]
y_train = df[target].iloc[:split]
y_test = df[target].iloc[split:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Compare with Linear Regression
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
# Test price-only features against all features
price_features = ["Close", "Volume", "return_1d", "return_3d"]

price_only_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

price_only_model.fit(X_train[price_features], y_train)
price_only_predictions = price_only_model.predict(X_test[price_features])

price_only_mae = mean_absolute_error(y_test, price_only_predictions)
price_only_rmse = np.sqrt(
    mean_squared_error(y_test, price_only_predictions)
)

print("\nPRICE-ONLY LINEAR REGRESSION")
print(f"MAE:  ${price_only_mae:,.2f}")
print(f"RMSE: ${price_only_rmse:,.2f}")

print("\nLINEAR REGRESSION COMPARISON")
print(f"Linear Regression MAE:  ${linear_mae:,.2f}")
print(f"Linear Regression RMSE: ${linear_rmse:,.2f}")
# Simple baseline: predict the next-day price using today's close


mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
baseline_predictions = X_test["Close"].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))

print("\nBASELINE COMPARISON")
print(f"Random Forest MAE:  ${mae:,.2f}")
print(f"Baseline MAE:      ${baseline_mae:,.2f}")
print(f"Random Forest RMSE: ${rmse:,.2f}")
print(f"Baseline RMSE:     ${baseline_rmse:,.2f}")
r2 = r2_score(y_test, predictions)

print("BTC NEXT-DAY PRICE MODEL")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")

joblib.dump(
    {"model": model, "features": features},
    "btc_price_model.joblib",
)
print("\nSaved model: btc_price_model.joblib")
