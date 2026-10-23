
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the combined historical dataset
df = pd.read_csv("data/btc_training_data.csv")
df = df.sort_values("day").reset_index(drop=True)

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

# Ensure model inputs are numeric and valid
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [target]).reset_index(drop=True)

# Expanding-window walk-forward evaluation
initial_train_size = int(len(df) * 0.6)
test_block_size = 30

all_actual = []
all_model_predictions = []
all_baseline_predictions = []
all_dates = []
all_current_closes = []

for test_start in range(initial_train_size, len(df), test_block_size):
    test_end = min(test_start + test_block_size, len(df))

    train = df.iloc[:test_start]
    test = df.iloc[test_start:test_end]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[features], train[target])

    predictions = model.predict(test[features])

    # Baseline: tomorrow's close equals today's close
    baseline = test["Close"].to_numpy()

    all_actual.extend(test[target].to_numpy())
    all_model_predictions.extend(predictions)
    all_baseline_predictions.extend(baseline)
    all_dates.extend(test["day"].astype(str).tolist())
    all_current_closes.extend(test["Close"].tolist())  

actual = np.asarray(all_actual)
model_pred = np.asarray(all_model_predictions)
baseline_pred = np.asarray(all_baseline_predictions)

model_mae = mean_absolute_error(actual, model_pred)
model_rmse = np.sqrt(mean_squared_error(actual, model_pred))

baseline_mae = mean_absolute_error(actual, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_pred))

print("WALK-FORWARD VALIDATION")
print("Total test predictions:", len(actual))

print("\nRandom Forest:")
print(f"MAE:  ${model_mae:,.2f}")
print(f"RMSE: ${model_rmse:,.2f}")

print("\nNaive baseline (next close = current close):")
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")

print("\nModel beats baseline on MAE:", model_mae < baseline_mae)
print("Model beats baseline on RMSE:", model_rmse < baseline_rmse)
results = pd.DataFrame({
    "date": all_dates,
    "current_close": all_current_closes,
    "actual_close": all_actual,
    "model_prediction": all_model_predictions,
    "baseline_prediction": all_baseline_predictions,
})

results.to_csv("data/walk_forward_predictions.csv", index=False)
print("\nPredictions with dates and prices saved successfully.")