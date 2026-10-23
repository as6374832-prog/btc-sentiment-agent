
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "walk_forward_predictions.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").dropna(
    subset=["date", "current_close", "actual_close", "model_prediction"]
)

# Predict direction using only information available at the time:
# predicted next close versus today's close.
df["signal"] = np.where(
    df["model_prediction"] > df["current_close"], 1, 0
)

# Actual next-day return, aligned to the current-day signal.
df["actual_return"] = (
    df["actual_close"] / df["current_close"] - 1
)

# Long when the model predicts an increase; otherwise stay in cash.
df["strategy_return"] = df["signal"] * df["actual_return"]

# Buy-and-hold benchmark over the same test observations.
df["buy_hold_return"] = df["actual_return"]

def calculate_metrics(returns):
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return {"Sharpe": np.nan, "Sortino": np.nan, "Max Drawdown": np.nan}

    # Daily annualization convention; assumes daily observations.
    volatility = returns.std(ddof=1)
    sharpe = (
        np.sqrt(365) * returns.mean() / volatility
        if pd.notna(volatility) and volatility > 0
        else np.nan
    )

    downside = returns[returns < 0]
    downside_deviation = np.sqrt((downside ** 2).mean())
    sortino = (
        np.sqrt(365) * returns.mean() / downside_deviation
        if downside_deviation > 0
        else np.nan
    )

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": drawdown.min(),
    }

strategy = calculate_metrics(df["strategy_return"])
benchmark = calculate_metrics(df["buy_hold_return"])

print("OUT-OF-SAMPLE STRATEGY RISK METRICS")
print(f"Test observations: {len(df)}")

print("\nModel strategy:")
for metric, value in strategy.items():
    if metric == "Max Drawdown":
        print(f"{metric}: {value:.2%}")
    else:
        print(f"{metric}: {value:.3f}")

print("\nBuy-and-hold benchmark:")
for metric, value in benchmark.items():
    if metric == "Max Drawdown":
        print(f"{metric}: {value:.2%}")
    else:
        print(f"{metric}: {value:.3f}")

df.to_csv(DATA / "strategy_returns.csv", index=False)
print("\nSaved: data/strategy_returns.csv")
