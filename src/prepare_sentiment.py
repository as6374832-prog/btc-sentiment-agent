import pandas as pd

df = pd.read_csv("data/tweets.csv")

df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df["sentiment_score"] = pd.to_numeric(
    df["sentiment_score"], errors="coerce"
)

for col in ["like_count", "retweet_count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df.dropna(subset=["date", "sentiment_score"])
df["day"] = df["date"].dt.floor("D")

daily = df.groupby("day").agg(
    tweet_count=("text", "count"),
    avg_sentiment=("sentiment_score", "mean"),
    likes_total=("like_count", "sum"),
    reposts_total=("retweet_count", "sum"),
).reset_index()

if "sentiment_label" in df.columns:
    labels = pd.crosstab(df["day"], df["sentiment_label"])
    labels.columns = [
        "label_" + str(c).lower().replace(" ", "_")
        for c in labels.columns
    ]
    daily = daily.merge(labels.reset_index(), on="day", how="left")

daily.to_csv("data/btc_daily_sentiment.csv", index=False)

print("Saved: btc_daily_sentiment.csv")
print("Daily records:", len(daily))
print(daily.head())
print("\nDate range:", daily["day"].min(), "to", daily["day"].max())
