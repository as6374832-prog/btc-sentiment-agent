
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project folders
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Load historical Bitcoin prices and daily sentiment
prices = pd.read_csv(DATA / "btc_historical_prices.csv")
sentiment = pd.read_csv(DATA / "btc_daily_sentiment.csv")

# Normalize dates before merging
prices["Date"] = pd.to_datetime(prices["timestamp"], utc=True).dt.date
sentiment["date"] = pd.to_datetime(
    sentiment["day"], utc=True
).dt.date

# Join on matching calendar dates
combined = prices.merge(
    sentiment,
    left_on="Date",
    right_on="date",
    how="inner",
)

combined = combined.sort_values("Date").dropna(
    subset=["Close", "avg_sentiment"]
)

if combined.empty:
    raise ValueError("No matching dates found between price and sentiment data.")

# Plot price and sentiment on separate vertical scales
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(
    combined["Date"],
    combined["Close"],
    label="BTC closing price (USD)",
)
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC closing price (USD)")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    combined["Date"],
    combined["avg_sentiment"],
    label="Average daily sentiment",
    linestyle="--",
)
ax2.set_ylabel("Average sentiment score")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("Bitcoin Price vs Daily Tweet Sentiment (Historical Dataset)")
fig.autofmt_xdate()
plt.tight_layout()

output_path = DATA / "btc_price_vs_sentiment.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Chart saved to: {output_path}")
print(f"Matching daily records: {len(combined)}")
print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")


# Sentiment versus next-day Bitcoin price change
combined["next_day_return_pct"] = (
    combined["Close"].shift(-1) / combined["Close"] - 1
) * 100

combined["sentiment_group"] = pd.cut(
    combined["avg_sentiment"],
    bins=[-float("inf"), 0.4, 0.6, float("inf")],
    labels=["Negative", "Neutral", "Positive"],
)

combined["return_group"] = pd.cut(
    combined["next_day_return_pct"],
    bins=[-float("inf"), -1, 1, float("inf")],
    labels=["Down >1%", "Within ±1%", "Up >1%"],
)

matrix = pd.crosstab(
    combined["sentiment_group"],
    combined["return_group"],
    normalize="index",
) * 100

print("\nSentiment vs next-day BTC price change (% within each sentiment group):")
print(matrix.round(1))

matrix.to_csv(DATA / "sentiment_price_matrix.csv")
print("\nMatrix saved to data/sentiment_price_matrix.csv")
plt.show()
