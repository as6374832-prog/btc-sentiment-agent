
import pandas as pd
import numpy as np

# Load historical daily BTC prices and daily tweet sentiment
prices = pd.read_csv("data/btc_historical_prices.csv")
sentiment = pd.read_csv("data/btc_daily_sentiment.csv")

# Normalize dates to UTC calendar days
prices["timestamp"] = pd.to_datetime(
    prices["timestamp"], utc=True, errors="coerce"
)
sentiment["day"] = pd.to_datetime(
    sentiment["day"], utc=True, errors="coerce"
)

prices = prices.dropna(subset=["timestamp", "Close"]).copy()
prices["day"] = prices["timestamp"].dt.floor("D")
sentiment["day"] = pd.to_datetime(
    sentiment["day"], utc=True, errors="coerce"
).dt.floor("D")
sentiment = sentiment.dropna(subset=["day"])

# Ensure one price record per calendar day
prices = (
    prices.sort_values("timestamp")
    .drop_duplicates("day", keep="last")
    .sort_values("day")
)

# Ensure one sentiment record per day
sentiment = sentiment.drop_duplicates("day", keep="last")

# Today's price and sentiment are the prediction-time features
data = prices.merge(sentiment, on="day", how="inner")
data = data.sort_values("day").reset_index(drop=True)

# Join tomorrow's actual close by its calendar date.
# This prevents a missing date from silently shifting the target.
future_prices = prices[["day", "Close"]].copy()
future_prices["day"] = future_prices["day"] - pd.Timedelta(days=1)
future_prices = future_prices.rename(
    columns={"Close": "target_next_day"}
)

data = data.merge(future_prices, on="day", how="left")
data["return_1d"] = data["Close"].pct_change()
data["return_3d"] = data["Close"].pct_change(3)

# Remove rows without required values, including the final target
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(
    subset=["Close", "target_next_day", "return_1d", "return_3d"]
).reset_index(drop=True)

data.to_csv("data/btc_training_data.csv", index=False)

print("Saved: data/btc_training_data.csv")
print("Rows:", len(data))
print("Date range:", data["day"].min(), "to", data["day"].max())
print("\nAlignment check:")
print(
    data[["day", "Close", "avg_sentiment", "target_next_day"]]
    .head(5)
    .to_string(index=False)
)


