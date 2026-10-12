from collections import Counter
from datetime import datetime, timedelta
import json
import os
import random
import re
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set favicon icon
icon = "logo.png" if os.path.exists("logo.png") else "🛡️"
st.set_page_config(
    page_title="Brand Sentiment & Reputation Monitor",
    layout="wide",
    page_icon=icon,
)

# ==============================================================================
# MINIMALIST BLACK & CHARCOAL GRAY STYLING
# ==============================================================================
st.markdown(
    """
<style>
    .stApp { background-color: #0A0A0A; color: #EDEDED; }
    section[data-testid="stSidebar"] { background-color: #1C1E21; border-right: 1px solid #2E3238; }
    div[data-testid="stMetric"] { background-color: #1C1E21; border: 1px solid #2E3238; padding: 14px 18px; border-radius: 8px; }
    div[data-testid="stMetric"] label { color: #8E95A0 !important; font-size: 0.88rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #EDEDED !important; font-weight: 700; }
    .stButton > button, .stDownloadButton > button {
        background-color: #2E3238 !important; color: #EDEDED !important; font-weight: 600 !important;
        border: 1px solid #4B525D !important; border-radius: 6px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #4B525D !important; border-color: #94A3B8 !important; color: #FFFFFF !important;
    }
    div[data-baseweb="input"], div[data-baseweb="select"] { background-color: #1C1E21 !important; border-color: #2E3238 !important; color: #EDEDED !important; }
    .stCaption { color: #8E95A0 !important; }
    div[data-testid="stExpander"] { background-color: #1C1E21; border: 1px solid #2E3238; border-radius: 8px; }
    hr { border-color: #2E3238 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 1. 35-LANGUAGE TRANSLATION DEFINITIONS
# ==============================================================================
LANGUAGES = {
    "English": "en",
    "ಕನ್ನಡ (Kannada)": "kn",
    "ಕೊಂಕಣಿ / कोंकणी (Konkani)": "gom",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
    "മലയാളം (Malayalam)": "ml",
    "मराठी (Marathi)": "mr",
    "বাংলা (Bengali)": "bn",
    "ગુજરાતી (Gujarati)": "gu",
    "Español (Spanish)": "es",
    "Français (French)": "fr",
    "Deutsch (German)": "de",
    "Italiano (Italian)": "it",
    "Português (Portuguese)": "pt",
    "Русский (Russian)": "ru",
    "日本語 (Japanese)": "ja",
    "한국어 (Korean)": "ko",
    "简体中文 (Chinese Simplified)": "zh-CN",
    "العربية (Arabic)": "ar",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ಿଆ (Odia)": "or",
    "অসমীয়া (Assamese)": "as",
    "संस्कृतम् (Sanskrit)": "sa",
    "मैथिली (Maithili)": "mai",
    "اردو (Urdu)": "ur",
    "Nederlands (Dutch)": "nl",
    "Polski (Polish)": "pl",
    "Svenska (Swedish)": "sv",
    "Türkçe (Turkish)": "tr",
    "فارسی (Persian)": "fa",
    "繁體中文 (Chinese Traditional)": "zh-TW",
    "Tiếng Việt (Vietnamese)": "vi",
    "Bahasa Indonesia": "id",
    "ภาษาไทย (Thai)": "th",
}

from translations import ALL_PRECOMPILED as TRANSLATIONS


# Language selection
selected_lang = st.sidebar.selectbox(
    "🌐 Language / ಭಾಷೆ / भाषा",
    list(LANGUAGES.keys()),
    index=0,
    key="unique_lang_selector_field",
)


# Translation helper function
@st.cache_data(show_spinner=False)
def get_translated_dict(target_name: str, target_code: str):
    if target_name in TRANSLATIONS:
        return TRANSLATIONS[target_name]
    try:
        keys = list(TRANSLATIONS["English"].keys())
        joined = " [[--]] ".join([TRANSLATIONS["English"][k] for k in keys])
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=" + target_code + "&dt=t&q=" + urllib.parse.quote(joined)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            trans_joined = "".join([part[0] for part in data[0] if part[0]])
            vals = trans_joined.split("[[--]]")
            if len(vals) == len(keys):
                return {keys[i]: vals[i].strip() for i in range(len(keys))}
    except Exception:
        pass
    return TRANSLATIONS["English"]

@st.cache_data(show_spinner=False)
def get_translated_dict(target_name: str, target_code: str):
    if target_name in TRANSLATIONS:
        return TRANSLATIONS[target_name]
    try:
        keys = list(TRANSLATIONS["English"].keys())
        joined = " [[--]] ".join([TRANSLATIONS["English"][k] for k in keys])
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=" + target_code + "&dt=t&q=" + urllib.parse.quote(joined)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            trans_joined = "".join([part[0] for part in data[0] if part[0]])
            vals = trans_joined.split("[[--]]")
            if len(vals) == len(keys):
                return {keys[i]: vals[i].strip() for i in range(len(keys))}
    except Exception:
        pass
    return TRANSLATIONS["English"]

@st.cache_data(show_spinner=False)
def get_translated_dict(target_name: str, target_code: str):
    if target_name in TRANSLATIONS:
        return TRANSLATIONS[target_name]
    try:
        keys = list(TRANSLATIONS["English"].keys())
        joined = " [[--]] ".join([TRANSLATIONS["English"][k] for k in keys])
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=" + target_code + "&dt=t&q=" + urllib.parse.quote(joined)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            trans_joined = "".join([part[0] for part in data[0] if part[0]])
            vals = trans_joined.split("[[--]]")
            if len(vals) == len(keys):
                return {keys[i]: vals[i].strip() for i in range(len(keys))}
    except Exception:
        pass
    return TRANSLATIONS["English"]

def t(key: str, **kwargs) -> str:
    lang_code = LANGUAGES.get(selected_lang, "en")
    lang_dict = get_translated_dict(selected_lang, lang_code)
    text = lang_dict.get(key, TRANSLATIONS["English"].get(key, key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text


# ==============================================================================
# HEADER: LOGO AND TITLE (RENDERED AFTER t IS DEFINED)
# ==============================================================================
if os.path.exists("logo.png") or os.path.exists("logo.jpg"):
  logo_file = "logo.png" if os.path.exists("logo.png") else "logo.jpg"
  col_logo, col_title = st.columns([1, 8])
  with col_logo:
    st.image(logo_file, width=90)
  with col_title:
    st.title(t("title").replace("🛡️ ", ""))
    st.caption(t("subtitle"))
else:
  st.title(t("title"))
  st.caption(t("subtitle"))

# ==============================================================================
# 2. NLP SCORING
# ==============================================================================
analyzer = SentimentIntensityAnalyzer()


def score_text(text: str):
  compound = analyzer.polarity_scores(text)["compound"]
  if compound >= 0.10:
    return compound, "positive"
  elif compound <= -0.10:
    return compound, "negative"
  return compound, "neutral"


# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header(t("settings_header"))
target_brand_input = st.sidebar.text_input(
    t("target_brand"), value="Tesla", key="target_brand_input_field"
)
competitors_input = st.sidebar.text_input(
    t("competitors"), value="BYD, Rivian", key="competitors_input_field"
)
z_thresh = st.sidebar.slider(
    t("z_slider"), 1.5, 4.0, 2.5, 0.5, key="z_thresh_slider_field"
)

selected_brand = target_brand_input.strip() if target_brand_input.strip() else "Tesla"
competitors = [
    c.strip() for c in competitors_input.split(",") if c.strip()
]
if not competitors:
  competitors = ["CompetitorA", "CompetitorB"]
all_brands = [selected_brand] + competitors


# ==============================================================================
# 4. DATA ENGINE
# ==============================================================================
@st.cache_data
def load_all_brand_data(brands_tuple, focus_brand):
  base_time = datetime.now() - timedelta(days=6)
  data = []

  pos_templates = [
      "Outstanding performance and customer feedback for {}",
      "Love the new design and premium quality from {}!",
      "Superb speed and reliable support provided by {}",
      "Highly recommended: five star experience with {}",
  ]
  neu_templates = [
      "New quarterly business report published by {}",
      "Delivery tracking update received for package from {}",
      "Reading the latest feature update notes from {}",
  ]
  neg_templates = [
      "Critical defect and unexpected breakdown reported in {}",
      "Terrible customer service response from {}, very disappointed!",
      "Severe malfunction: requesting immediate refund from {}",
  ]

  for hour in range(24 * 6):
    ts = base_time + timedelta(hours=hour)
    for b in brands_tuple:
      for _ in range(random.randint(1, 4)):
        p = random.random()
        txt = (
            random.choice(pos_templates).format(b)
            if p < 0.52
            else (
                random.choice(neu_templates).format(b)
                if p < 0.82
                else random.choice(neg_templates).format(b)
            )
        )
        comp, label = score_text(txt)
        data.append({
            "timestamp": ts,
            "brand": b,
            "text": txt,
            "source": "Web / Social Media",
            "reach": random.randint(1000, 20000),
            "sentiment_score": comp,
            "sentiment": label,
        })

    # Injected Crisis spike on Day 5
    if 112 <= hour <= 116:
      for _ in range(16):
        txt = (
            f"CRITICAL DEFECT & SAFETY RISK! Urgent boycott of {focus_brand}!"
        )
        comp, label = score_text(txt)
        data.append({
            "timestamp": ts,
            "brand": focus_brand,
            "text": txt,
            "source": "News / Viral Post",
            "reach": random.randint(5000, 35000),
            "sentiment_score": comp,
            "sentiment": label,
        })

  return pd.DataFrame(data)


df = load_all_brand_data(tuple(all_brands), selected_brand)

# ==============================================================================
# 5. DATA ANALYTICS: CRISIS SPIKE ENGINE
# ==============================================================================
brand_df = (
    df[df["brand"] == selected_brand].set_index("timestamp").sort_index()
)
grouped = (
    brand_df.groupby([pd.Grouper(freq="6h"), "sentiment"], observed=False)[
        "text"
    ]
    .count()
    .unstack(fill_value=0)
)

for col in ["positive", "neutral", "negative"]:
  if col not in grouped.columns:
    grouped[col] = 0

total_in_window = (
    grouped["positive"] + grouped["neutral"] + grouped["negative"]
)
grouped["nss"] = np.where(
    total_in_window > 0,
    ((grouped["positive"] - grouped["negative"]) / total_in_window) * 100,
    0,
)

neg = grouped["negative"]
rolling_mean = neg.shift(1).rolling(8, min_periods=1).mean().fillna(0.0)
rolling_std = (
    neg.shift(1).rolling(8, min_periods=1).std().fillna(1.0).replace(0, 1.0)
)
grouped["z_score"] = (neg - rolling_mean) / rolling_std
grouped["is_crisis"] = (
    (grouped["z_score"] >= z_thresh)
    & (grouped["nss"] < -15)
    & (grouped["negative"] >= 2)
)

active_crises = grouped[grouped["is_crisis"]]

# ==============================================================================
# 6. CRISIS ALERT BANNER
# ==============================================================================
if not active_crises.empty:
  latest_crisis = active_crises.iloc[-1]
  st.error(
      f"{t('crisis_alert_title', brand=selected_brand.upper())}\n\n"
      f"- **{t('time_window')}:**"
      f" {active_crises.index[-1].strftime('%Y-%m-%d %H:%M')}\n- **{t('z_anomaly')}:**"
      f" {latest_crisis['z_score']:.2f}σ\n- **{t('neg_volume')}:**"
      f" {int(latest_crisis['negative'])} {t('articles')}\n- **{t('nss')}:**"
      f" {latest_crisis['nss']:.1f}"
  )

  with st.expander(t("dispatch_expander")):
    alert_payload = {
        "event": "PR_CRISIS_ALERT",
        "brand": selected_brand,
        "timestamp": str(active_crises.index[-1]),
        "z_score": float(round(latest_crisis["z_score"], 2)),
        "negative_volume": int(latest_crisis["negative"]),
        "net_sentiment_score": float(round(latest_crisis["nss"], 1)),
        "severity": (
            "HIGH" if latest_crisis["z_score"] >= 3.5 else "MEDIUM"
        ),
    }
    st.json(alert_payload)
    if st.button(t("dispatch_btn"), key="dispatch_alert_button"):
      st.success(t("dispatch_success", brand=selected_brand.lower()))
else:
  st.success(t("normal_status", brand=selected_brand))

st.divider()

# ==============================================================================
# 7. TOP KPI CARDS
# ==============================================================================
target_mentions = df[df["brand"] == selected_brand]
tot_count = len(target_mentions)
pos_count = (target_mentions["sentiment"] == "positive").sum()
neg_count = (target_mentions["sentiment"] == "negative").sum()
overall_nss = (
    ((pos_count - neg_count) / tot_count * 100) if tot_count > 0 else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(t("kpi_total"), f"{tot_count:,}")
kpi2.metric(t("kpi_nss"), f"{overall_nss:.1f}")
kpi3.metric(
    t("kpi_pos"),
    f"{pos_count} ({(pos_count/tot_count*100):.1f}%)" if tot_count > 0 else "0",
)
kpi4.metric(
    t("kpi_neg"),
    f"{neg_count} ({(neg_count/tot_count*100):.1f}%)" if tot_count > 0 else "0",
)

st.divider()

# ==============================================================================
# 8. SECTION 1: SHARE OF VOICE (SOV)
# ==============================================================================
st.subheader(t("sov_section"))

total_mentions_all = len(df)
total_reach_all = df["reach"].sum()

sov_df = (
    df.groupby("brand")
    .agg(mentions=("text", "count"), total_reach=("reach", "sum"))
    .reset_index()
)
sov_df["volume_sov_pct"] = (sov_df["mentions"] / total_mentions_all) * 100
sov_df["reach_sov_pct"] = (sov_df["total_reach"] / total_reach_all) * 100

charcoal_palette = ["#94A3B8", "#64748B", "#475569", "#334155"]

c1, c2 = st.columns([1, 1])
with c1:
  fig_donut = px.pie(
      sov_df,
      names="brand",
      values="volume_sov_pct",
      hole=0.4,
      title=t("sov_donut_title"),
      color_discrete_sequence=charcoal_palette,
  )
  fig_donut.update_layout(
      paper_bgcolor="#1C1E21",
      plot_bgcolor="#1C1E21",
      font=dict(color="#EDEDED"),
      legend=dict(font=dict(color="#EDEDED")),
  )
  st.plotly_chart(fig_donut, use_container_width=True)

with c2:
  fig_bar = px.bar(
      sov_df,
      x="brand",
      y=["volume_sov_pct", "reach_sov_pct"],
      barmode="group",
      title=t("sov_bar_title"),
      color_discrete_sequence=["#94A3B8", "#475569"],
      labels={
          "value": "%",
          "variable": t("metric_label"),
          "brand": t("target_brand"),
      },
  )
  fig_bar.update_layout(
      paper_bgcolor="#1C1E21",
      plot_bgcolor="#1C1E21",
      font=dict(color="#EDEDED"),
      xaxis=dict(gridcolor="#2E3238"),
      yaxis=dict(gridcolor="#2E3238"),
      legend=dict(font=dict(color="#EDEDED")),
  )
  st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# 9. SECTION 2: SENTIMENT TRENDS & ROOT-CAUSE
# ==============================================================================
st.subheader(t("trends_section", brand=selected_brand))
trend_col, kw_col = st.columns([3, 2])

with trend_col:
  fig_trends = go.Figure()
  fig_trends.add_trace(
      go.Bar(
          x=grouped.index,
          y=grouped["negative"],
          name=t("neg_vol_legend"),
          marker_color="#DC2626",
      )
  )
  fig_trends.add_trace(
      go.Bar(
          x=grouped.index,
          y=grouped["positive"],
          name=t("pos_vol_legend"),
          marker_color="#16A34A",
      )
  )
  fig_trends.add_trace(
      go.Scatter(
          x=grouped.index,
          y=grouped["nss"],
          name=t("nss_legend"),
          yaxis="y2",
          line=dict(color="#94A3B8", width=3),
      )
  )
  fig_trends.update_layout(
      barmode="stack",
      paper_bgcolor="#1C1E21",
      plot_bgcolor="#1C1E21",
      font=dict(color="#EDEDED"),
      xaxis=dict(gridcolor="#2E3238"),
      yaxis=dict(title=t("mention_axis"), gridcolor="#2E3238"),
      yaxis2=dict(
          title=t("nss_axis"),
          overlaying="y",
          side="right",
          range=[-105, 105],
          gridcolor="#2E3238",
      ),
      legend=dict(x=0, y=1.1, orientation="h", font=dict(color="#EDEDED")),
  )
  st.plotly_chart(fig_trends, use_container_width=True)

with kw_col:
  neg_texts = target_mentions[target_mentions["sentiment"] == "negative"][
      "text"
  ].tolist()
  stop_words = {
      "the",
      "and",
      "a",
      "an",
      "in",
      "on",
      "to",
      "of",
      "for",
      "is",
      "at",
      "by",
      "from",
      "with",
      "that",
      "this",
      "it",
      "as",
      "are",
      "was",
      selected_brand.lower(),
  }

  words = []
  for s in neg_texts:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
    words.extend([w for w in tokens if w not in stop_words])

  if words:
    counts = Counter(words).most_common(10)
    kw_df = pd.DataFrame(
        counts, columns=[t("keyword_label"), t("freq_label")]
    ).sort_values(by=t("freq_label"), ascending=True)
    fig_kw = px.bar(
        kw_df,
        x=t("freq_label"),
        y=t("keyword_label"),
        orientation="h",
        title=t("top_keywords_title"),
        color=t("freq_label"),
        color_continuous_scale=["#64748B", "#94A3B8", "#DC2626"],
    )
    fig_kw.update_layout(
        paper_bgcolor="#1C1E21",
        plot_bgcolor="#1C1E21",
        font=dict(color="#EDEDED"),
        xaxis=dict(gridcolor="#2E3238"),
        yaxis=dict(gridcolor="#2E3238"),
    )
    st.plotly_chart(fig_kw, use_container_width=True)
  else:
    st.info(t("no_keywords"))

# ==============================================================================
# 10. SECTION 3: MENTIONS EXPLORER & CSV EXPORT
# ==============================================================================
st.subheader(t("table_section"))

csv_data = target_mentions.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t("download_btn", brand=selected_brand),
    data=csv_data,
    file_name=f"{selected_brand.lower()}_sentiment_report.csv",
    mime="text/csv",
    key="download_csv_button_field",
)

st.dataframe(
    target_mentions.sort_values(by="timestamp", ascending=False)[
        ["timestamp", "text", "sentiment", "sentiment_score", "source"]
    ],
    use_container_width=True,
)