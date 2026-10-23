import yfinance as yf
import pandas as pd

prices = yf.download(
    "BTC-USD",
    start="2022-01-01",
    end="2023-06-24",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if prices.empty:
    raise RuntimeError("No historical price data downloaded.")

if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

prices = prices.reset_index()
prices = prices.rename(columns={"Date": "timestamp"})
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

prices.to_csv("data/btc_historical_prices.csv", index=False)

print("Saved btc_historical_prices.csv")
print("Rows:", len(prices))
print("Date range:", prices["timestamp"].min(), "to", prices["timestamp"].max())
print(prices.head(3))
