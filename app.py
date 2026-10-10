from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Brand Sentiment & Reputation Monitor", layout="wide"
)

st.title("🛡️ Brand Sentiment & Reputation Monitor")
st.caption(
    "Data Science Core: VADER/BERT NLP | Data Analytics Core: Share of Voice &"
    " Live PR Crisis Alerts"
)

analyzer = SentimentIntensityAnalyzer()


# ---------------- DATA GENERATION / INGESTION ----------------
@st.cache_data
def load_data():
  np.random.seed(42)
  base_time = datetime.now() - timedelta(days=7)
  data = []
  brands = ["TargetBrand", "CompetitorA", "CompetitorB"]

  pos_examples = ["Love the product! Outstanding quality.", "Best tool ever!"]
  neu_examples = [
      "Package arrived today.",
      "Reading the latest patch release notes.",
  ]
  neg_examples = [
      "Broken authentication system, completely locked out!",
      "Customer support is unresponsive and terrible.",
  ]

  for hour in range(24 * 7):
    ts = base_time + timedelta(hours=hour)
    for b in brands:
      for _ in range(random.randint(2, 6)):
        p = random.random()
        txt = (
            random.choice(pos_examples)
            if p < 0.5
            else (random.choice(neu_examples) if p < 0.8 else random.choice(neg_examples))
        )
        data.append({
            "timestamp": ts,
            "brand": b,
            "text": txt,
            "reach": random.randint(100, 2000),
        })

    # Crisis burst for TargetBrand at Day 6
    if 135 <= hour <= 138:
      for _ in range(20):
        data.append({
            "timestamp": ts,
            "brand": "TargetBrand",
            "text": "Critical data vulnerability reported! Immediate boycott!",
            "reach": random.randint(1000, 8000),
        })

  df = pd.DataFrame(data)

  # VADER Scoring
  def get_vader(t):
    comp = analyzer.polarity_scores(t)["compound"]
    label = (
        "positive" if comp >= 0.10 else ("negative" if comp <= -0.10 else "neutral")
    )
    return comp, label

  scores = df["text"].apply(get_vader)
  df["sentiment_score"] = [s[0] for s in scores]
  df["sentiment"] = [s[1] for s in scores]
  return df


df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Navigation & Filters")
selected_brand = st.sidebar.selectbox(
    "Target Brand", options=df["brand"].unique(), index=0
)
z_thresh = st.sidebar.slider("Crisis Alert Z-Score Threshold", 2.0, 5.0, 3.0, 0.5)

# ---------------- CRISIS ALERT BANNER ----------------
from analytics_engine import compute_share_of_voice, detect_crisis_spikes

trends_df = detect_crisis_spikes(
    df, target_brand=selected_brand, z_threshold=z_thresh
)
active_crises = trends_df[trends_df["is_crisis"]]

if not active_crises.empty:
  latest_crisis = active_crises.iloc[-1]
  st.error(
      f"🚨 **LIVE PR CRISIS ALERT DETECTED FOR {selected_brand.upper()}**\n\n"
      f"- **Anomaly Time:** {active_crises.index[-1].strftime('%Y-%m-%d %H:%M')}\n"
      f"- **Z-Score Spike:** {latest_crisis['z_score']:.2f}σ above normal"
      f" baseline\n- **Negative Volume:** {int(latest_crisis['negative'])}"
      f" mentions in window\n- **Net Sentiment Score (NSS):**"
      f" {latest_crisis['nss']:.1f}"
  )
else:
  st.success(
      f"✅ **Normal Status**: No active PR crisis spikes detected for"
      f" {selected_brand}."
  )

st.divider()

# ---------------- SECTION 1: SHARE OF VOICE (SOV) ----------------
st.subheader("1. Share of Voice (SoV) Analytics")
sov_df = compute_share_of_voice(df)

col1, col2 = st.columns([1, 1])

with col1:
  fig_sov = px.pie(
      sov_df,
      names="brand",
      values="volume_sov_pct",
      hole=0.4,
      title="Mention Volume Share of Voice (%)",
  )
  st.plotly_chart(fig_sov, use_container_width=True)

with col2:
  fig_reach = px.bar(
      sov_df,
      x="brand",
      y=["volume_sov_pct", "reach_sov_pct"],
      barmode="group",
      title="Volume SoV vs. Reach-Weighted SoV (%)",
      labels={"value": "Percentage (%)", "variable": "Metric"},
  )
  st.plotly_chart(fig_reach, use_container_width=True)

# ---------------- SECTION 2: SENTIMENT TRENDS & CRISIS ----------------
st.subheader(f"2. Sentiment Trends & Anomaly Timeline ({selected_brand})")

fig_trends = go.Figure()
fig_trends.add_trace(
    go.Bar(
        x=trends_df.index,
        y=trends_df["negative"],
        name="Negative Volume",
        marker_color="crimson",
    )
)
fig_trends.add_trace(
    go.Bar(
        x=trends_df.index,
        y=trends_df["positive"],
        name="Positive Volume",
        marker_color="mediumseagreen",
    )
)
fig_trends.add_trace(
    go.Scatter(
        x=trends_df.index,
        y=trends_df["nss"],
        name="Net Sentiment Score (NSS)",
        yaxis="y2",
        line=dict(color="royalblue", width=2),
    )
)

fig_trends.update_layout(
    barmode="stack",
    yaxis=dict(title="Mention Count"),
    yaxis2=dict(
        title="Net Sentiment (-100 to +100)",
        overlaying="y",
        side="right",
        range=[-105, 105],
    ),
    legend=dict(x=0, y=1.1, orientation="h"),
)
st.plotly_chart(fig_trends, use_container_width=True)

# ---------------- SECTION 3: RAW DATA & AUDIT LOG ----------------
with st.expander("🔍 View Raw Mentions & Sentiment Breakdown"):
  st.dataframe(
      df[df["brand"] == selected_brand][
          ["timestamp", "text", "sentiment", "sentiment_score", "reach"]
      ].tail(50),
      use_container_width=True,
  )

  from datetime import datetime, timedelta
import email.utils
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from analytics_engine import compute_share_of_voice, detect_crisis_spikes
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Brand Sentiment & Reputation Monitor", layout="wide"
)

st.title("🛡️ Brand Sentiment & Reputation Monitor")
st.caption(
    "Data Science Core: VADER NLP | Data Analytics Core: Share of Voice & PR"
    " Crisis Alerts"
)

analyzer = SentimentIntensityAnalyzer()


def score_text(text: str):
  compound = analyzer.polarity_scores(text)["compound"]
  if compound >= 0.10:
    return compound, "positive"
  elif compound <= -0.10:
    return compound, "negative"
  return compound, "neutral"


# ---------------- LIVE NEWS FETCHING (REAL WEB DATA) ----------------
def fetch_live_news_for_brand(brand: str, max_items: int = 40):
  encoded_query = urllib.parse.quote(brand.strip())
  url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
  req = urllib.request.Request(
      url,
      headers={
          "User-Agent": (
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
          )
      },
  )

  articles = []
  try:
    with urllib.request.urlopen(req, timeout=12) as response:
      xml_data = response.read()
    root = ET.fromstring(xml_data)
    for item in root.findall(".//item")[:max_items]:
      title = item.find("title").text if item.find("title") is not None else ""
      link = item.find("link").text if item.find("link") is not None else ""
      pub_date_str = (
          item.find("pubDate").text
          if item.find("pubDate") is not None
          else None
      )

      if pub_date_str:
        try:
          dt = email.utils.parsedate_to_datetime(pub_date_str).replace(
              tzinfo=None
          )
        except Exception:
          dt = datetime.utcnow()
      else:
        dt = datetime.utcnow()

      source_elem = item.find("source")
      source_name = source_elem.text if source_elem is not None else "News"

      comp, label = score_text(title)
      articles.append({
          "timestamp": dt,
          "brand": brand.strip(),
          "text": title,
          "source": source_name,
          "link": link,
          "reach": random.randint(1000, 20000),
          "sentiment_score": comp,
          "sentiment": label,
      })
  except Exception as e:
    st.error(f"Error fetching news for '{brand}': {e}")
  return articles


# ---------------- SIMULATED BENCHMARK GENERATOR ----------------
def generate_demo_data():
  np.random.seed(42)
  base_time = datetime.now() - timedelta(days=7)
  data = []
  brands = ["TargetBrand", "CompetitorA", "CompetitorB"]

  pos_examples = ["Love the product! Outstanding quality.", "Best tool ever!"]
  neu_examples = [
      "Package arrived today.",
      "Reading the latest patch release notes.",
  ]
  neg_examples = [
      "Broken authentication system, completely locked out!",
      "Customer support is unresponsive and terrible.",
  ]

  for hour in range(24 * 7):
    ts = base_time + timedelta(hours=hour)
    for b in brands:
      for _ in range(random.randint(2, 5)):
        p = random.random()
        txt = (
            random.choice(pos_examples)
            if p < 0.5
            else (
                random.choice(neu_examples)
                if p < 0.8
                else random.choice(neg_examples)
            )
        )
        comp, label = score_text(txt)
        data.append({
            "timestamp": ts,
            "brand": b,
            "text": txt,
            "source": "Social Media",
            "link": "#",
            "reach": random.randint(100, 2000),
            "sentiment_score": comp,
            "sentiment": label,
        })

    # Crisis burst for TargetBrand at Day 6
    if 135 <= hour <= 138:
      for _ in range(18):
        txt = (
            "CRITICAL SECURITY VULNERABILITY! Immediate boycott of TargetBrand!"
        )
        comp, label = score_text(txt)
        data.append({
            "timestamp": ts,
            "brand": "TargetBrand",
            "text": txt,
            "source": "Social Media",
            "link": "#",
            "reach": random.randint(1000, 8000),
            "sentiment_score": comp,
            "sentiment": label,
        })
  return pd.DataFrame(data)


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙️ Data Source & Settings")
mode = st.sidebar.radio(
    "Select Mode",
    ["🌐 Live Real-World News", "🧪 Simulated Benchmark (with Crisis)"],
)

if mode == "🌐 Live Real-World News":
  st.sidebar.markdown("### Search Any Brand")
  target_brand_input = st.sidebar.text_input("Target Brand", value="Tesla")
  competitors_input = st.sidebar.text_input(
      "Competitors (comma-separated)", value="BYD, Rivian"
  )
  z_thresh = st.sidebar.slider(
      "Crisis Alert Sensitivity (Z-Score)", 1.5, 4.0, 2.5, 0.5
  )

  all_brands = [target_brand_input.strip()] + [
      c.strip() for c in competitors_input.split(",") if c.strip()
  ]

  with st.spinner("Fetching live articles and calculating sentiment..."):
    all_articles = []
    for b in all_brands:
      all_articles.extend(fetch_live_news_for_brand(b))
    df = pd.DataFrame(all_articles)

  selected_brand = target_brand_input.strip()

else:
  df = generate_demo_data()
  selected_brand = st.sidebar.selectbox(
      "Target Brand", options=df["brand"].unique(), index=0
  )
  z_thresh = st.sidebar.slider(
      "Crisis Alert Sensitivity (Z-Score)", 2.0, 5.0, 3.0, 0.5
  )

# ---------------- VALIDATION ----------------
if df.empty or selected_brand not in df["brand"].values:
  st.warning(
      f"No articles found for '{selected_brand}'. Please check the spelling or"
      " try another brand name."
  )
  st.stop()

# ---------------- CRISIS ALERT BANNER ----------------
trends_df = detect_crisis_spikes(
    df, target_brand=selected_brand, z_threshold=z_thresh
)
active_crises = (
    trends_df[trends_df["is_crisis"]]
    if not trends_df.empty and "is_crisis" in trends_df.columns
    else pd.DataFrame()
)

if not active_crises.empty:
  latest_crisis = active_crises.iloc[-1]
  st.error(
      f"🚨 **PR CRISIS SPIKE DETECTED FOR {selected_brand.upper()}**\n\n"
      f"- **Anomaly Time Window:**"
      f" {active_crises.index[-1].strftime('%Y-%m-%d %H:%M')}\n- **Z-Score"
      f" Anomaly:** {latest_crisis['z_score']:.2f}σ above normal baseline\n-"
      f" **Negative Volume:** {int(latest_crisis['negative'])} articles\n-"
      f" **Net Sentiment Score (NSS):** {latest_crisis['nss']:.1f}"
  )
else:
  st.success(
      f"✅ **Normal Status**: No abnormal negative sentiment spike detected for"
      f" {selected_brand}."
  )

st.divider()

# ---------------- SECTION 1: SHARE OF VOICE ----------------
st.subheader("1. Share of Voice (SoV) Tracking")
sov_df = compute_share_of_voice(df)

col1, col2 = st.columns([1, 1])
with col1:
  fig_sov = px.pie(
      sov_df,
      names="brand",
      values="volume_sov_pct",
      hole=0.4,
      title="Article Volume Share of Voice (%)",
  )
  st.plotly_chart(fig_sov, use_container_width=True)

with col2:
  fig_reach = px.bar(
      sov_df,
      x="brand",
      y=["volume_sov_pct", "reach_sov_pct"],
      barmode="group",
      title="Volume SoV vs. Reach-Weighted SoV (%)",
      labels={"value": "Percentage (%)", "variable": "Metric"},
  )
  st.plotly_chart(fig_reach, use_container_width=True)

# ---------------- SECTION 2: SENTIMENT TRENDS ----------------
st.subheader(f"2. Sentiment Trend & Volume Over Time ({selected_brand})")

if not trends_df.empty:
  fig_trends = go.Figure()
  fig_trends.add_trace(
      go.Bar(
          x=trends_df.index,
          y=trends_df["negative"],
          name="Negative Volume",
          marker_color="crimson",
      )
  )
  fig_trends.add_trace(
      go.Bar(
          x=trends_df.index,
          y=trends_df["positive"],
          name="Positive Volume",
          marker_color="mediumseagreen",
      )
  )
  fig_trends.add_trace(
      go.Scatter(
          x=trends_df.index,
          y=trends_df["nss"],
          name="Net Sentiment Score (NSS)",
          yaxis="y2",
          line=dict(color="royalblue", width=2),
      )
  )

  fig_trends.update_layout(
      barmode="stack",
      yaxis=dict(title="Mention / Article Count"),
      yaxis2=dict(
          title="Net Sentiment (-100 to +100)",
          overlaying="y",
          side="right",
          range=[-105, 105],
      ),
      legend=dict(x=0, y=1.1, orientation="h"),
  )
  st.plotly_chart(fig_trends, use_container_width=True)

# ---------------- SECTION 3: RAW ARTICLES EXPLORER ----------------
with st.expander(f"📰 View Analyzed Mentions for {selected_brand}"):
  brand_articles = df[df["brand"] == selected_brand].sort_values(
      by="timestamp", ascending=False
  )
  st.dataframe(
      brand_articles[
          ["timestamp", "text", "sentiment", "sentiment_score", "source"]
      ],
      use_container_width=True,
  )