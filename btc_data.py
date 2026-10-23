
import yfinance as yf
import pandas as pd

print("Downloading Bitcoin price data...")

btc = yf.download(
    "BTC-USD",
    period="180d",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if btc.empty:
    print("No data received. Check your internet connection.")
else:
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc.index = pd.to_datetime(btc.index, utc=True)
    btc.index.name = "timestamp"

    btc = btc[["Open", "High", "Low", "Close", "Volume"]]
    btc = btc.dropna()

    btc.to_csv("btc_prices.csv")

    print("\nDownload successful!")
    print("Dataset shape:", btc.shape)
    print("\nFirst five rows:")
    print(btc.head())
    print("\nLatest five rows:")
    print(btc.tail())
    print("\nSaved to btc_prices.csv")
