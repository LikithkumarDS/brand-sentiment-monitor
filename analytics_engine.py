import numpy as np
import pandas as pd


def compute_share_of_voice(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates both unweighted volume SOV and reach-weighted SOV."""
  total_mentions = len(df)
  total_reach = df["reach"].sum()

  sov = (
      df.groupby("brand")
      .agg(
          mentions=("text", "count"),
          total_reach=("reach", "sum"),
          pos_count=("sentiment", lambda s: (s == "positive").sum()),
          neg_count=("sentiment", lambda s: (s == "negative").sum()),
      )
      .reset_index()
  )

  sov["volume_sov_pct"] = (sov["mentions"] / total_mentions) * 100
  sov["reach_sov_pct"] = (sov["total_reach"] / total_reach) * 100
  # Net Sentiment Score: (Pos - Neg) / Total * 100
  sov["nss"] = (
      (sov["pos_count"] - sov["neg_count"]) / sov["mentions"]
  ) * 100
  return sov


def detect_crisis_spikes(
    df: pd.DataFrame,
    target_brand: str,
    window_hours: str = "4h",
    z_threshold: float = 3.0,
):
  """Evaluates negative mention volume against rolling baseline."""
  brand_df = (
      df[df["brand"] == target_brand].set_index("timestamp").sort_index()
  )

  # Resample into time windows
  grouped = (
      brand_df.groupby([pd.Grouper(freq=window_hours), "sentiment"])[
          "text"
      ]
      .count()
      .unstack(fill_value=0)
  )

  for col in ["positive", "neutral", "negative"]:
    if col not in grouped.columns:
      grouped[col] = 0

  total = grouped["positive"] + grouped["neutral"] + grouped["negative"]
  grouped["nss"] = np.where(
      total > 0, ((grouped["positive"] - grouped["negative"]) / total) * 100, 0
  )

  # Statistical anomaly detection: Rolling Z-Score
  neg = grouped["negative"]
  rolling_mean = neg.shift(1).rolling(12, min_periods=3).mean().fillna(1.0)
  rolling_std = (
      neg.shift(1).rolling(12, min_periods=3).std().fillna(1.0).replace(0, 1.0)
  )

  grouped["z_score"] = (neg - rolling_mean) / rolling_std
  grouped["is_crisis"] = (
      (grouped["z_score"] >= z_threshold)
      & (grouped["nss"] < -20)
      & (grouped["negative"] >= 5)
  )

  return grouped