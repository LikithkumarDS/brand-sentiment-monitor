from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------
# 1. SAMPLE DATA GENERATOR (Simulating 7 days of mentions)
# ---------------------------------------------------------
def generate_sample_data():
  np.random.seed(42)
  base_time = datetime.now() - timedelta(days=7)
  data = []

  brands = ["TargetBrand", "CompetitorA", "CompetitorB"]

  positive_templates = [
      "Loving the new product from {}! Great experience.",
      "Incredible customer support from {}, five stars!",
      "Super satisfied with {}, works flawlessly.",
  ]
  neutral_templates = [
      "Just saw an ad from {}.",
      "Ordered a package from {} yesterday.",
      "Reading the latest announcement from {}.",
  ]
  negative_templates = [
      "Terrible outage on {}, my account is locked!",
      "{} customer service was completely rude and useless.",
      "Another bug in {}... cancelling my subscription.",
  ]

  # Generate 7 days of normal background activity
  for hour in range(24 * 7):
    timestamp = base_time + timedelta(hours=hour)
    for brand in brands:
      # Base volume
      n_posts = random.randint(3, 8)
      for _ in range(n_posts):
        # 50% positive, 35% neutral, 15% negative normally
        p = random.random()
        if p < 0.50:
          text = random.choice(positive_templates).format(brand)
        elif p < 0.85:
          text = random.choice(neutral_templates).format(brand)
        else:
          text = random.choice(negative_templates).format(brand)

        data.append({
            "timestamp": timestamp,
            "brand": brand,
            "text": text,
            "reach": random.randint(50, 1000),
        })

    # SIMULATE A CRISIS SPIKE on Day 6 for TargetBrand (hours 140 to 144)
    if 140 <= hour <= 144:
      for _ in range(25):  # Sudden surge of complaints
        data.append({
            "timestamp": timestamp,
            "brand": "TargetBrand",
            "text": (
                "Critical security flaw exposed in TargetBrand! Fix this now!"
            ),
            "reach": random.randint(500, 5000),
        })

  return pd.DataFrame(data)


# ---------------------------------------------------------
# 2. VADER SENTIMENT SCORING FUNCTION
# ---------------------------------------------------------
def score_sentiment(text: str):
  compound = analyzer.polarity_scores(text)["compound"]
  if compound >= 0.10:
    label = "positive"
  elif compound <= -0.10:
    label = "negative"
  else:
    label = "neutral"
  return pd.Series([compound, label], index=["sentiment_score", "sentiment"])


# ---------------------------------------------------------
# 3. SHARE OF VOICE (SOV)
# ---------------------------------------------------------
def calculate_and_plot_sov(df: pd.DataFrame):
  sov_counts = df["brand"].value_counts()
  sov_percentage = (sov_counts / len(df)) * 100

  print("\n=== SHARE OF VOICE (SOV) ===")
  for brand, pct in sov_percentage.items():
    print(f"{brand}: {pct:.1f}% ({sov_counts[brand]} mentions)")

  # Plot Donut Chart
  plt.figure(figsize=(6, 6))
  plt.pie(
      sov_percentage,
      labels=sov_percentage.index,
      autopct="%1.1f%%",
      startangle=90,
      wedgeprops=dict(width=0.4, edgecolor="w"),
  )
  plt.title("Share of Voice (7-Day Overview)")
  plt.tight_layout()
  plt.savefig("sov_chart.png")
  print("Saved: sov_chart.png")


# ---------------------------------------------------------
# 4. SENTIMENT TRENDS (Net Sentiment Score over Time)
# ---------------------------------------------------------
def calculate_and_plot_sentiment_trends(df: pd.DataFrame, target_brand: str):
  target_df = (
      df[df["brand"] == target_brand].copy().set_index("timestamp").sort_index()
  )

  # Resample by 4-hour intervals
  grouped = (
      target_df.groupby(
          [pd.Grouper(freq="4h"), "sentiment"], observed=False
      )["text"]
      .count()
      .unstack(fill_value=0)
  )

  # Ensure columns exist
  for col in ["positive", "neutral", "negative"]:
    if col not in grouped.columns:
      grouped[col] = 0

  # Calculate Net Sentiment Score (NSS = (Pos - Neg) / Total * 100)
  total = grouped["positive"] + grouped["neutral"] + grouped["negative"]
  grouped["total"] = total
  grouped["NSS"] = np.where(
      total > 0, ((grouped["positive"] - grouped["negative"]) / total) * 100, 0
  )

  # Plot Dual-Axis Trend Chart
  fig, ax1 = plt.subplots(figsize=(10, 5))

  # Left Axis: Volume Stack
  ax1.bar(
      grouped.index,
      grouped["positive"],
      label="Positive",
      color="green",
      alpha=0.6,
      width=0.1,
  )
  ax1.bar(
      grouped.index,
      grouped["neutral"],
      bottom=grouped["positive"],
      label="Neutral",
      color="gray",
      alpha=0.4,
      width=0.1,
  )
  ax1.bar(
      grouped.index,
      grouped["negative"],
      bottom=grouped["positive"] + grouped["neutral"],
      label="Negative",
      color="red",
      alpha=0.6,
      width=0.1,
  )
  ax1.set_ylabel("Mention Volume")
  ax1.set_xlabel("Time")

  # Right Axis: Net Sentiment Score (NSS)
  ax2 = ax1.twinx()
  ax2.plot(
      grouped.index,
      grouped["NSS"],
      color="blue",
      linewidth=2,
      label="Net Sentiment Score (NSS)",
  )
  ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
  ax2.set_ylabel("Net Sentiment Score (-100 to +100)")
  ax2.set_ylim(-105, 105)

  plt.title(f"{target_brand} - Sentiment Trend & Volume Over Time")
  ax1.legend(loc="upper left")
  ax2.legend(loc="upper right")
  plt.tight_layout()
  plt.savefig("sentiment_trend.png")
  print("Saved: sentiment_trend.png")

  return grouped


# ---------------------------------------------------------
# 5. CRISIS ALERT ENGINE (Z-Score on Negative Mentions)
# ---------------------------------------------------------
def detect_crisis_alerts(
    grouped_df: pd.DataFrame, target_brand: str, z_threshold: float = 3.0
):
  # Calculate rolling baseline (last 12 intervals = 48 hours)
  neg_series = grouped_df["negative"]
  rolling_mean = (
      neg_series.shift(1).rolling(window=12, min_periods=4).mean().fillna(1.0)
  )
  rolling_std = (
      neg_series.shift(1).rolling(window=12, min_periods=4).std().fillna(1.0)
  )
  rolling_std = rolling_std.replace(0, 1.0)  # Avoid division by zero

  z_scores = (neg_series - rolling_mean) / rolling_std

  alerts = []
  for ts, z in z_scores.items():
    neg_vol = grouped_df.loc[ts, "negative"]
    nss = grouped_df.loc[ts, "NSS"]

    # Condition: Z-Score exceeds threshold AND Net Sentiment is negative
    if z >= z_threshold and nss < -20 and neg_vol >= 10:
      alerts.append({
          "timestamp": ts,
          "z_score": round(z, 2),
          "negative_mentions": int(neg_vol),
          "nss": round(nss, 1),
      })

  print(f"\n=== CRISIS ALERT EVALUATION ({target_brand}) ===")
  if alerts:
    print(f"⚠️  {len(alerts)} CRISIS SPIKE(S) DETECTED:")
    for a in alerts:
      print(
          f"  - [{a['timestamp']}] Z-Score: {a['z_score']} | "
          f"Negative Volume: {a['negative_mentions']} | NSS: {a['nss']}"
      )
  else:
    print("Normal baseline maintained. No crises detected.")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
  print("Generating dataset...")
  df = generate_sample_data()

  print("Scoring sentiment with VADER...")
  sentiment_results = df["text"].apply(score_sentiment)
  df[["sentiment_score", "sentiment"]] = sentiment_results

  # 1. Share of Voice
  calculate_and_plot_sov(df)

  # 2. Sentiment Trends
  grouped_trends = calculate_and_plot_sentiment_trends(
      df, target_brand="TargetBrand"
  )

  # 3. Crisis Alerts
  detect_crisis_alerts(
      grouped_trends, target_brand="TargetBrand", z_threshold=3.0
  )