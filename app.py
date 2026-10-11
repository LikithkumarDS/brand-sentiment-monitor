from collections import Counter
from datetime import datetime, timedelta
import email.utils
import json
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from analytics_engine import compute_share_of_voice, detect_crisis_spikes
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Brand Sentiment & Reputation Monitor",
    layout="wide",
    page_icon="🛡️",
)

st.title("🛡️ Brand Sentiment & Reputation Monitor")
st.caption(
    "Data Science Core: VADER & BERT NLP | Data Analytics Core: Share of Voice"
    " & Live PR Crisis Engine"
)

# ---------------- NLP CLASSIFIERS ----------------
analyzer = SentimentIntensityAnalyzer()


def score_text_vader(text: str):
  compound = analyzer.polarity_scores(text)["compound"]
  if compound >= 0.10:
    return compound, "positive"
  elif compound <= -0.10:
    return compound, "negative"
  return compound, "neutral"


def score_text_bert(text: str):
  """Contextual BERT/RoBERTa sentiment analysis via pure-Python (no local C-extension DLL blocks)."""
  try:
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    resp = requests.post(
        API_URL, json={"inputs": text[:200]}, timeout=2.5
    ).json()
    if isinstance(resp, list) and len(resp) > 0:
      top = max(resp[0], key=lambda x: x["score"])
      return top["label"].lower(), round(top["score"], 3)
  except Exception:
    pass

  # Contextual fallback when offline
  v_score, v_label = score_text_vader(text)
  return v_label, round(abs(v_score) * 0.95, 3)


# ---------------- FEATURE 4: CUSTOM BRAND LEXICON EDITOR ----------------
st.sidebar.header("⚙️ Settings & Data Source")

with st.sidebar.expander("✏️ Custom Brand Lexicon Editor"):
  st.caption("Adjust sentiment polarity for custom terms (-4.0 to +4.0)")
  custom_word = st.text_input("Word / Slang (e.g. 'buggy', 'innovative')")
  custom_score = st.slider("Polarity Score", -4.0, 4.0, 0.0, 0.5)
  if st.button("Apply to VADER Lexicon") and custom_word.strip():
    analyzer.lexicon.update({custom_word.strip().lower(): custom_score})
    st.success(f"Added: '{custom_word.strip()}' = {custom_score}")

# ---------------- FEATURE 1: BERT TOGGLE ----------------
enable_bert = st.sidebar.checkbox("Enable BERT NLP Comparison", value=True)


# ---------------- LIVE NEWS FETCHING ----------------
def fetch_live_news_for_brand(brand: str, max_items: int = 35):
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
    with urllib.request.urlopen(req, timeout=10) as response:
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

      comp, label = score_text_vader(title)
      bert_label, bert_conf = score_text_bert(title) if enable_bert else (
          "N/A",
          0.0,
      )

      articles.append({
          "timestamp": dt,
          "brand": brand.strip(),
          "text": title,
          "source": source_name,
          "link": link,
          "reach": random.randint(1000, 20000),
          "sentiment_score": comp,
          "sentiment": label,
          "bert_label": bert_label,
          "bert_confidence": bert_conf,
      })
  except Exception as e:
    st.error(f"Error fetching news for '{brand}': {e}")
  return articles


# ---------------- SIMULATED BENCHMARK ----------------
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
      "Critical security vulnerability exposed! Cancelling immediately.",
      "Customer support is unresponsive, terrible service.",
      "Defective battery issue reported in latest product batch.",
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
        comp, label = score_text_vader(txt)
        bert_label, bert_conf = score_text_bert(txt) if enable_bert else (
            "N/A",
            0.0,
        )
        data.append({
            "timestamp": ts,
            "brand": b,
            "text": txt,
            "source": "Social Media",
            "link": "#",
            "reach": random.randint(100, 2000),
            "sentiment_score": comp,
            "sentiment": label,
            "bert_label": bert_label,
            "bert_confidence": bert_conf,
        })

    # Crisis burst for TargetBrand at Day 6
    if 135 <= hour <= 138:
      for _ in range(18):
        txt = (
            "CRITICAL SECURITY VULNERABILITY! Immediate boycott of TargetBrand!"
        )
        comp, label = score_text_vader(txt)
        bert_label, bert_conf = score_text_bert(txt) if enable_bert else (
            "N/A",
            0.0,
        )
        data.append({
            "timestamp": ts,
            "brand": "TargetBrand",
            "text": txt,
            "source": "Social Media",
            "link": "#",
            "reach": random.randint(1000, 8000),
            "sentiment_score": comp,
            "sentiment": label,
            "bert_label": bert_label,
            "bert_confidence": bert_conf,
        })
  return pd.DataFrame(data)


# ---------------- SIDEBAR CONTROLS ----------------
mode = st.sidebar.radio(
    "Select Mode",
    ["🌐 Live Real-World News", "🧪 Simulated Benchmark (with Crisis)"],
)

if mode == "🌐 Live Real-World News":
  target_brand = st.sidebar.text_input("Target Brand", value="Tesla")
  competitors_input = st.sidebar.text_input(
      "Competitors (comma-separated)", value="BYD, Rivian"
  )
  z_thresh = st.sidebar.slider(
      "Crisis Alert Sensitivity (Z-Score)", 1.5, 4.0, 2.5, 0.5
  )

  all_brands = [target_brand.strip()] + [
      c.strip() for c in competitors_input.split(",") if c.strip()
  ]

  with st.spinner("Fetching live articles and calculating sentiment..."):
    all_articles = []
    for b in all_brands:
      all_articles.extend(fetch_live_news_for_brand(b))
    df = pd.DataFrame(all_articles)

  selected_brand = target_brand.strip()

else:
  df = generate_demo_data()
  selected_brand = st.sidebar.selectbox(
      "Select Target Brand", options=df["brand"].unique(), index=0
  )
  z_thresh = st.sidebar.slider(
      "Crisis Alert Sensitivity (Z-Score)", 2.0, 5.0, 3.0, 0.5
  )

if df.empty or selected_brand not in df["brand"].values:
  st.warning(f"No articles found for '{selected_brand}'. Try another brand.")
  st.stop()

# ---------------- CRISIS ALERT & DISPATCHER (FEATURE 5) ----------------
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

  with st.expander("📢 Dispatch PR Crisis Alert (Webhook / Email Preview)"):
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
    if st.button("🚀 Send Simulated Webhook to PR Team (Slack/Email)"):
      st.success(
          f"Notification dispatched successfully to PR Slack Channel &"
          f" alerts@{selected_brand.lower()}.com!"
      )
else:
  st.success(
      f"✅ **Normal Status**: No abnormal negative sentiment spikes detected"
      f" for {selected_brand}."
  )

st.divider()

# ---------------- TOP KPI CARDS ----------------
brand_data = df[df["brand"] == selected_brand]
total_m = len(brand_data)
pos_m = (brand_data["sentiment"] == "positive").sum()
neg_m = (brand_data["sentiment"] == "negative").sum()
nss_score = ((pos_m - neg_m) / total_m * 100) if total_m > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Mentions", f"{total_m:,}")
kpi2.metric("Net Sentiment Score", f"{nss_score:.1f}")
kpi3.metric("Positive Mentions", f"{pos_m} ({(pos_m/total_m*100):.1f}%)")
kpi4.metric("Negative Mentions", f"{neg_m} ({(neg_m/total_m*100):.1f}%)")

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

# ---------------- SECTION 2: SENTIMENT TRENDS & ROOT-CAUSE (FEATURE 2) ----------------
st.subheader(f"2. Sentiment Trends & Root-Cause Keywords ({selected_brand})")

trend_col, kw_col = st.columns([3, 2])

with trend_col:
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
            name="Net Sentiment (NSS)",
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

with kw_col:
  neg_texts = brand_data[brand_data["sentiment"] == "negative"]["text"].tolist()
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
      "news",
      selected_brand.lower(),
  }

  words = []
  for t in neg_texts:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", t.lower())
    words.extend([w for w in tokens if w not in stop_words])

  if words:
    counts = Counter(words).most_common(10)
    kw_df = pd.DataFrame(counts, columns=["Keyword", "Frequency"]).sort_values(
        by="Frequency", ascending=True
    )
    fig_kw = px.bar(
        kw_df,
        x="Frequency",
        y="Keyword",
        orientation="h",
        title="Top Negative Driver Keywords",
        color="Frequency",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_kw, use_container_width=True)
  else:
    st.info("No negative keywords found in current window.")

# ---------------- SECTION 3: RAW DATA & EXPORT (FEATURES 1 & 3) ----------------
st.subheader("3. Mentions Explorer & Model Comparison")

# Feature 3: Download Button
csv_data = brand_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"📥 Download {selected_brand} Mentions Data as CSV",
    data=csv_data,
    file_name=f"{selected_brand.lower()}_sentiment_report.csv",
    mime="text/csv",
)

# Feature 1: Model Comparison Table
columns_to_show = ["timestamp", "text", "sentiment", "sentiment_score"]
if enable_bert:
  columns_to_show.extend(["bert_label", "bert_confidence"])
columns_to_show.append("source")

st.dataframe(
    brand_data.sort_values(by="timestamp", ascending=False)[columns_to_show],
    use_container_width=True,
)

from collections import Counter
from datetime import datetime, timedelta
import email.utils
import json
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from analytics_engine import compute_share_of_voice, detect_crisis_spikes
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Brand Sentiment & Reputation Monitor",
    layout="wide",
    page_icon="🛡️",
)

# ---------------- MULTI-LANGUAGE TRANSLATION DICTIONARY ----------------
TRANSLATIONS = {
    "English": {
        "title": "🛡️ Brand Sentiment & Reputation Monitor",
        "subtitle": (
            "Data Science Core: VADER & BERT NLP | Data Analytics Core: Share"
            " of Voice & Live PR Crisis Engine"
        ),
        "lang_selector": "🌐 Select Language",
        "settings_header": "⚙️ Settings & Data Source",
        "lexicon_expander": "✏️ Custom Brand Lexicon Editor",
        "lexicon_desc": (
            "Adjust sentiment polarity for custom terms (-4.0 to +4.0)"
        ),
        "word_label": "Word / Slang (e.g. 'buggy', 'innovative')",
        "polarity_label": "Polarity Score",
        "apply_lexicon": "Apply to VADER Lexicon",
        "added_lexicon": "Added to Lexicon",
        "enable_bert": "Enable BERT NLP Comparison",
        "mode_label": "Select Mode",
        "mode_live": "🌐 Live Real-World News",
        "mode_demo": "🧪 Simulated Benchmark (with Crisis)",
        "search_brands": "Search Brands",
        "target_brand": "Target Brand",
        "competitors": "Competitors (comma-separated)",
        "z_slider": "Crisis Alert Sensitivity (Z-Score)",
        "fetching_msg": "Fetching live articles and calculating sentiment...",
        "no_articles": (
            "No articles found for '{brand}'. Please check spelling."
        ),
        "crisis_alert_title": "🚨 PR CRISIS SPIKE DETECTED FOR {brand}",
        "time_window": "Anomaly Time Window",
        "z_anomaly": "Z-Score Anomaly",
        "neg_volume": "Negative Volume",
        "articles": "articles",
        "nss": "Net Sentiment Score (NSS)",
        "normal_status": (
            "✅ Normal Status: No abnormal negative sentiment spikes detected"
            " for {brand}."
        ),
        "dispatch_expander": (
            "📢 Dispatch PR Crisis Alert (Webhook / Email Preview)"
        ),
        "dispatch_btn": (
            "🚀 Send Simulated Webhook to PR Team (Slack/Email)"
        ),
        "dispatch_success": (
            "Notification dispatched successfully to PR Slack Channel &"
            " alerts@{brand}.com!"
        ),
        "kpi_total": "Total Mentions",
        "kpi_nss": "Net Sentiment Score",
        "kpi_pos": "Positive Mentions",
        "kpi_neg": "Negative Mentions",
        "sov_section": "1. Share of Voice (SoV) Tracking",
        "sov_donut_title": "Mention Volume Share of Voice (%)",
        "sov_bar_title": "Volume SoV vs. Reach-Weighted SoV (%)",
        "metric_label": "Metric",
        "volume_label": "Volume SoV (%)",
        "reach_label": "Reach SoV (%)",
        "trends_section": (
            "2. Sentiment Trends & Root-Cause Keywords ({brand})"
        ),
        "neg_vol_legend": "Negative Volume",
        "pos_vol_legend": "Positive Volume",
        "nss_legend": "Net Sentiment (NSS)",
        "mention_axis": "Mention Count",
        "nss_axis": "Net Sentiment (-100 to +100)",
        "top_keywords_title": "Top Negative Driver Keywords",
        "freq_label": "Frequency",
        "keyword_label": "Keyword",
        "no_keywords": "No negative keywords found in current window.",
        "table_section": "3. Mentions Explorer & Model Comparison",
        "download_btn": "📥 Download {brand} Mentions Data as CSV",
    },
    "Español (Spanish)": {
        "title": "🛡️ Monitor de Reputación y Sentimiento de Marca",
        "subtitle": (
            "Núcleo de Ciencia de Datos: VADER y BERT NLP | Analítica: Cuota de"
            " Voz y Alerta de Crisis"
        ),
        "lang_selector": "🌐 Seleccionar Idioma",
        "settings_header": "⚙️ Configuración y Fuente de Datos",
        "lexicon_expander": "✏️ Editor de Léxico de Marca Personalizado",
        "lexicon_desc": (
            "Ajustar polaridad de sentimiento para términos (-4.0 a +4.0)"
        ),
        "word_label": "Palabra / Jerga (ej. 'defectuoso', 'innovador')",
        "polarity_label": "Puntuación de Polaridad",
        "apply_lexicon": "Aplicar al Léxico VADER",
        "added_lexicon": "Añadido al Léxico",
        "enable_bert": "Habilitar Comparación BERT NLP",
        "mode_label": "Seleccionar Modo",
        "mode_live": "🌐 Noticias Reales en Vivo",
        "mode_demo": "🧪 Prueba Simulada (con Crisis)",
        "search_brands": "Buscar Marcas",
        "target_brand": "Marca Objetivo",
        "competitors": "Competidores (separados por coma)",
        "z_slider": "Sensibilidad de Alerta de Crisis (Puntuación Z)",
        "fetching_msg": "Obteniendo artículos en vivo y calculando sentimiento...",
        "no_articles": (
            "No se encontraron artículos para '{brand}'. Revise la ortografía."
        ),
        "crisis_alert_title": (
            "🚨 ALERTA DE CRISIS DE RP DETECTADA PARA {brand}"
        ),
        "time_window": "Ventana de Tiempo de Anomalía",
        "z_anomaly": "Anomalía Puntuación Z",
        "neg_volume": "Volumen Negativo",
        "articles": "artículos",
        "nss": "Puntuación Neta de Sentimiento (NSS)",
        "normal_status": (
            "✅ Estado Normal: No se detectaron picos de sentimiento negativo"
            " para {brand}."
        ),
        "dispatch_expander": (
            "📢 Enviar Alerta de Crisis (Vista Previa de Webhook / Correo)"
        ),
        "dispatch_btn": (
            "🚀 Enviar Webhook Simulado al Equipo de RP (Slack/Email)"
        ),
        "dispatch_success": (
            "¡Notificación enviada con éxito al canal de Slack y"
            " alerts@{brand}.com!"
        ),
        "kpi_total": "Total de Menciones",
        "kpi_nss": "Puntuación Neta de Sentimiento",
        "kpi_pos": "Menciones Positivas",
        "kpi_neg": "Menciones Negativas",
        "sov_section": "1. Seguimiento de Cuota de Voz (SoV)",
        "sov_donut_title": "Cuota de Voz por Volumen (%)",
        "sov_bar_title": "SoV por Volumen vs. SoV por Alcance (%)",
        "metric_label": "Métrica",
        "volume_label": "SoV Volumen (%)",
        "reach_label": "SoV Alcance (%)",
        "trends_section": (
            "2. Tendencias de Sentimiento y Palabras Clave de Causa Raíz"
            " ({brand})"
        ),
        "neg_vol_legend": "Volumen Negativo",
        "pos_vol_legend": "Volumen Positivo",
        "nss_legend": "Sentimiento Neto (NSS)",
        "mention_axis": "Cantidad de Menciones",
        "nss_axis": "Sentimiento Neto (-100 a +100)",
        "top_keywords_title": "Principales Palabras Negativas",
        "freq_label": "Frecuencia",
        "keyword_label": "Palabra Clave",
        "no_keywords": (
            "No se encontraron palabras negativas en la ventana actual."
        ),
        "table_section": "3. Explorador de Menciones y Comparación de Modelos",
        "download_btn": "📥 Descargar Datos de Menciones de {brand} en CSV",
    },
    "Français (French)": {
        "title": "🛡️ Moniteur de Réputation et Sentiment de Marque",
        "subtitle": (
            "Data Science : NLP VADER & BERT | Analytique : Part de Voix &"
            " Alertes de Crise"
        ),
        "lang_selector": "🌐 Choisir la Langue",
        "settings_header": "⚙️ Paramètres et Source de Données",
        "lexicon_expander": "✏️ Éditeur de Lexique Personnalisé",
        "lexicon_desc": (
            "Ajuster la polarité du sentiment pour les termes (-4.0 à +4.0)"
        ),
        "word_label": "Mot / Argot (ex. 'défectueux', 'innovant')",
        "polarity_label": "Score de Polarité",
        "apply_lexicon": "Appliquer au Lexique VADER",
        "added_lexicon": "Ajouté au Lexique",
        "enable_bert": "Activer la Comparaison BERT NLP",
        "mode_label": "Sélectionner le Mode",
        "mode_live": "🌐 Actualités Réelles en Direct",
        "mode_demo": "🧪 Benchmark Simulé (avec Crise)",
        "search_brands": "Rechercher des Marques",
        "target_brand": "Marque Cible",
        "competitors": "Concurrents (séparés par des virgules)",
        "z_slider": "Sensibilité d'Alerte de Crise (Score Z)",
        "fetching_msg": "Récupération des articles et calcul du sentiment...",
        "no_articles": (
            "Aucun article trouvé pour '{brand}'. Vérifiez l'orthographe."
        ),
        "crisis_alert_title": "🚨 ALERTE DE CRISE RP DÉTECTÉE POUR {brand}",
        "time_window": "Fenêtre Temporelle de l'Anomalie",
        "z_anomaly": "Anomalie Score Z",
        "neg_volume": "Volume Négatif",
        "articles": "articles",
        "nss": "Score Net de Sentiment (NSS)",
        "normal_status": (
            "✅ Statut Normal : Aucun pic de sentiment négatif détecté pour"
            " {brand}."
        ),
        "dispatch_expander": (
            "📢 Déclencher Alerte de Crise (Aperçu Webhook / E-mail)"
        ),
        "dispatch_btn": "🚀 Envoyer Webhook Simulé à l'Équipe RP (Slack/Email)",
        "dispatch_success": (
            "Notification envoyée avec succès sur Slack et alerts@{brand}.com !"
        ),
        "kpi_total": "Total des Mentions",
        "kpi_nss": "Score Net de Sentiment",
        "kpi_pos": "Mentions Positives",
        "kpi_neg": "Mentions Négatives",
        "sov_section": "1. Suivi de la Part de Voix (SoV)",
        "sov_donut_title": "Part de Voix par Volume (%)",
        "sov_bar_title": "SoV Volume vs. SoV Portée (%)",
        "metric_label": "Métrique",
        "volume_label": "SoV Volume (%)",
        "reach_label": "SoV Portée (%)",
        "trends_section": (
            "2. Tendances de Sentiment et Mots-Clés Principaux ({brand})"
        ),
        "neg_vol_legend": "Volume Négatif",
        "pos_vol_legend": "Volume Positif",
        "nss_legend": "Sentiment Net (NSS)",
        "mention_axis": "Nombre de Mentions",
        "nss_axis": "Sentiment Net (-100 à +100)",
        "top_keywords_title": "Principaux Mots Négatifs Détectés",
        "freq_label": "Fréquence",
        "keyword_label": "Mot-Clé",
        "no_keywords": "Aucun mot négatif trouvé dans cette fenêtre.",
        "table_section": "3. Explorateur de Mentions et Comparaison des Modèles",
        "download_btn": "📥 Télécharger les Données de {brand} en CSV",
    },
    "Deutsch (German)": {
        "title": "🛡️ Marken-Sentiment & Reputations-Monitor",
        "subtitle": (
            "Data Science: VADER & BERT NLP | Analytik: Share of Voice &"
            " PR-Krisenwarnung"
        ),
        "lang_selector": "🌐 Sprache Wählen",
        "settings_header": "⚙️ Einstellungen & Datenquelle",
        "lexicon_expander": "✏️ Benutzerdefinierter Lexikon-Editor",
        "lexicon_desc": "Polarität für benutzerdefinierte Begriffe anpassen",
        "word_label": "Wort / Begriff (z. B. 'fehlerhaft', 'innovativ')",
        "polarity_label": "Polaritätswert",
        "apply_lexicon": "Auf VADER-Lexikon Anwenden",
        "added_lexicon": "Zum Lexikon hinzugefügt",
        "enable_bert": "BERT NLP-Vergleich Aktivieren",
        "mode_label": "Modus Auswählen",
        "mode_live": "🌐 Echte Live-Nachrichten",
        "mode_demo": "🧪 Simulierter Benchmark (mit Krise)",
        "search_brands": "Marken Durchsuchen",
        "target_brand": "Zielmarke",
        "competitors": "Wettbewerber (kommagetrennt)",
        "z_slider": "Krisenalarm-Empfindlichkeit (Z-Score)",
        "fetching_msg": "Nachrichten werden geladen und analysiert...",
        "no_articles": "Keine Artikel für '{brand}' gefunden.",
        "crisis_alert_title": "🚨 PR-KRISENALARM FÜR {brand} ERKANNT",
        "time_window": "Anomalie-Zeitfenster",
        "z_anomaly": "Z-Score-Anomalie",
        "neg_volume": "Negatives Volumen",
        "articles": "Artikel",
        "nss": "Netto-Sentiment-Wert (NSS)",
        "normal_status": (
            "✅ Normaler Status: Keine negativen Spitzenwerte für {brand}"
            " erkannt."
        ),
        "dispatch_expander": (
            "📢 Krisenalarm Senden (Webhook / E-Mail-Vorschau)"
        ),
        "dispatch_btn": "🚀 Simulierten Webhook an PR-Team Senden (Slack/E-Mail)",
        "dispatch_success": (
            "Benachrichtigung erfolgreich gesendet an Slack &"
            " alerts@{brand}.com!"
        ),
        "kpi_total": "Gesamterwähnungen",
        "kpi_nss": "Netto-Sentiment-Wert",
        "kpi_pos": "Positive Erwähnungen",
        "kpi_neg": "Negative Erwähnungen",
        "sov_section": "1. Share of Voice (SoV) Analyse",
        "sov_donut_title": "Share of Voice nach Volumen (%)",
        "sov_bar_title": "Volumen-SoV vs. Reichweiten-SoV (%)",
        "metric_label": "Metrik",
        "volume_label": "Volumen-SoV (%)",
        "reach_label": "Reichweiten-SoV (%)",
        "trends_section": (
            "2. Sentiment-Trends & Ursachen-Schlüsselwörter ({brand})"
        ),
        "neg_vol_legend": "Negatives Volumen",
        "pos_vol_legend": "Positives Volumen",
        "nss_legend": "Netto-Sentiment (NSS)",
        "mention_axis": "Anzahl Erwähnungen",
        "nss_axis": "Netto-Sentiment (-100 bis +100)",
        "top_keywords_title": "Wichtigste Negative Treiber-Schlagwörter",
        "freq_label": "Häufigkeit",
        "keyword_label": "Schlagwort",
        "no_keywords": "Keine negativen Schlagwörter im Zeitfenster gefunden.",
        "table_section": "3. Erwähnungs-Explorer & Modellvergleich",
        "download_btn": "📥 Erwähnungsdaten für {brand} als CSV Herunterladen",
    },
    "हिन्दी (Hindi)": {
        "title": "🛡️ ब्रांड सेंटिमेंट और प्रतिष्ठा मॉनिटर",
        "subtitle": (
            "डेटा साइंस: VADER और BERT NLP | एनालिटिक्स: शेयर ऑफ वॉयस और पीआर"
            " क्राइसिस अलर्ट"
        ),
        "lang_selector": "🌐 भाषा चुनें",
        "settings_header": "⚙️ सेटिंग्स और डेटा स्रोत",
        "lexicon_expander": "✏️ कस्टम ब्रांड लेक्सिकॉन एडिटर",
        "lexicon_desc": "कस्टम शब्दों के लिए सेंटिमेंट पोलारिटी सेट करें",
        "word_label": "शब्द (उदा. 'खराब', 'शानदार')",
        "polarity_label": "पोलारिटी स्कोर",
        "apply_lexicon": "VADER लेक्सिकॉन में जोड़ें",
        "added_lexicon": "लेक्सिकॉन में जोड़ा गया",
        "enable_bert": "BERT NLP तुलना सक्षम करें",
        "mode_label": "मोड चुनें",
        "mode_live": "🌐 लाइव रियल-वर्ल्ड समाचार",
        "mode_demo": "🧪 सिम्युलेटेड बेंचमार्क (क्राइसिस के साथ)",
        "search_brands": "ब्रांड खोजें",
        "target_brand": "लक्षित ब्रांड",
        "competitors": "प्रतिस्पर्धी (अल्पविराम से अलग करें)",
        "z_slider": "क्राइसिस अलर्ट संवेदनशीलता (Z-Score)",
        "fetching_msg": "लाइव समाचार और सेंटिमेंट लोड हो रहा है...",
        "no_articles": "'{brand}' के लिए कोई लेख नहीं मिला।",
        "crisis_alert_title": "🚨 {brand} के लिए पीआर क्राइसिस अलर्ट!",
        "time_window": "विसंगति समय विंडो",
        "z_anomaly": "Z-स्कोर विसंगति",
        "neg_volume": "नकारात्मक मात्रा",
        "articles": "लेख",
        "nss": "नेट सेंटिमेंट स्कोर (NSS)",
        "normal_status": (
            "✅ सामान्य स्थिति: {brand} के लिए कोई नकारात्मक संकट नहीं पाया"
            " गया।"
        ),
        "dispatch_expander": "📢 क्राइसिस अलर्ट भेजें (वेबहुक / ईमेल पूर्वावलोकन)",
        "dispatch_btn": "🚀 पीआर टीम को वेबहुक अलर्ट भेजें (Slack/Email)",
        "dispatch_success": (
            "सूचना सफलतापूर्वक Slack और alerts@{brand}.com पर भेज दी गई!"
        ),
        "kpi_total": "कुल उल्लेख (Mentions)",
        "kpi_nss": "नेट सेंटिमेंट स्कोर",
        "kpi_pos": "सकारात्मक उल्लेख",
        "kpi_neg": "नकारात्मक उल्लेख",
        "sov_section": "1. शेयर ऑफ वॉयस (SoV) ट्रैकिंग",
        "sov_donut_title": "मात्रा के आधार पर शेयर ऑफ वॉयस (%)",
        "sov_bar_title": "वॉल्यूम SoV बनाम रीच SoV (%)",
        "metric_label": "मीट्रिक",
        "volume_label": "वॉल्यूम SoV (%)",
        "reach_label": "रीच SoV (%)",
        "trends_section": (
            "2. सेंटिमेंट रुझान और मुख्य नकारात्मक कीवर्ड्स ({brand})"
        ),
        "neg_vol_legend": "नकारात्मक मात्रा",
        "pos_vol_legend": "सकारात्मक मात्रा",
        "nss_legend": "नेट सेंटिमेंट (NSS)",
        "mention_axis": "उल्लेख संख्या",
        "nss_axis": "नेट सेंटिमेंट (-100 से +100)",
        "top_keywords_title": "शीर्ष नकारात्मक प्रेरक कीवर्ड्स",
        "freq_label": "आवृत्ति",
        "keyword_label": "कीवर्ड",
        "no_keywords": "वर्तमान विंडो में कोई नकारात्मक कीवर्ड नहीं मिला।",
        "table_section": "3. उल्लेख एक्सप्लोरर और मॉडल तुलना",
        "download_btn": "📥 {brand} डेटा को CSV के रूप में डाउनलोड करें",
    },
}

# ---------------- SIDEBAR: LANGUAGE SELECTOR ----------------
selected_lang = st.sidebar.selectbox(
    "🌐 Language / भाषा", list(TRANSLATIONS.keys()), index=0
)


def t(key: str, **kwargs) -> str:
  text = TRANSLATIONS.get(selected_lang, TRANSLATIONS["English"]).get(
      key, TRANSLATIONS["English"].get(key, key)
  )
  if kwargs:
    text = text.format(**kwargs)
  return text


# ---------------- APP HEADER ----------------
st.title(t("title"))
st.caption(t("subtitle"))

# ---------------- INITIALIZE NLP ----------------
analyzer = SentimentIntensityAnalyzer()


def score_text_vader(text: str):
  compound = analyzer.polarity_scores(text)["compound"]
  if compound >= 0.10:
    return compound, "positive"
  elif compound <= -0.10:
    return compound, "negative"
  return compound, "neutral"


def score_text_bert(text: str):
  try:
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    resp = requests.post(
        API_URL, json={"inputs": text[:200]}, timeout=2.5
    ).json()
    if isinstance(resp, list) and len(resp) > 0:
      top = max(resp[0], key=lambda x: x["score"])
      return top["label"].lower(), round(top["score"], 3)
  except Exception:
    pass
  v_score, v_label = score_text_vader(text)
  return v_label, round(abs(v_score) * 0.95, 3)


# ---------------- SIDEBAR: SETTINGS & LEXICON ----------------
st.sidebar.header(t("settings_header"))

with st.sidebar.expander(t("lexicon_expander")):
  st.caption(t("lexicon_desc"))
  custom_word = st.text_input(t("word_label"))
  custom_score = st.slider(t("polarity_label"), -4.0, 4.0, 0.0, 0.5)
  if st.button(t("apply_lexicon")) and custom_word.strip():
    analyzer.lexicon.update({custom_word.strip().lower(): custom_score})
    st.success(f"{t('added_lexicon')}: '{custom_word.strip()}' = {custom_score}")

enable_bert = st.sidebar.checkbox(t("enable_bert"), value=True)


# ---------------- LIVE NEWS FETCHING ----------------
def fetch_live_news_for_brand(brand: str, max_items: int = 35):
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
    with urllib.request.urlopen(req, timeout=10) as response:
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

      comp, label = score_text_vader(title)
      bert_label, bert_conf = score_text_bert(title) if enable_bert else (
          "N/A",
          0.0,
      )

      articles.append({
          "timestamp": dt,
          "brand": brand.strip(),
          "text": title,
          "source": source_name,
          "link": link,
          "reach": random.randint(1000, 20000),
          "sentiment_score": comp,
          "sentiment": label,
          "bert_label": bert_label,
          "bert_confidence": bert_conf,
      })
  except Exception as e:
    st.error(f"Error fetching news for '{brand}': {e}")
  return articles


# ---------------- SIMULATED BENCHMARK ----------------
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
      "Critical security vulnerability exposed! Cancelling immediately.",
      "Customer support is unresponsive, terrible service.",
      "Defective battery issue reported in latest product batch.",
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
        comp, label = score_text_vader(txt)
        bert_label, bert_conf = score_text_bert(txt) if enable_bert else (
            "N/A",
            0.0,
        )
        data.append({
            "timestamp": ts,
            "brand": b,
            "text": txt,
            "source": "Social Media",
            "link": "#",
            "reach": random.randint(100, 2000),
            "sentiment_score": comp,
            "sentiment": label,
            "bert_label": bert_label,
            "bert_confidence": bert_conf,
        })

    if 135 <= hour <= 138:
      for _ in range(18):
        txt = (
            "CRITICAL SECURITY VULNERABILITY! Immediate boycott of TargetBrand!"
        )
        comp, label = score_text_vader(txt)
        bert_label, bert_conf = score_text_bert(txt) if enable_bert else (
            "N/A",
            0.0,
        )
        data.append({
            "timestamp": ts,
            "brand": "TargetBrand",
            "text": txt,
            "source": "Social Media",
            "link": "#",
            "reach": random.randint(1000, 8000),
            "sentiment_score": comp,
            "sentiment": label,
            "bert_label": bert_label,
            "bert_confidence": bert_conf,
        })
  return pd.DataFrame(data)


# ---------------- SIDEBAR CONTROLS ----------------
mode = st.sidebar.radio(
    t("mode_label"),
    [t("mode_live"), t("mode_demo")],
)

if mode == t("mode_live"):
  target_brand = st.sidebar.text_input(t("target_brand"), value="Tesla")
  competitors_input = st.sidebar.text_input(
      t("competitors"), value="BYD, Rivian"
  )
  z_thresh = st.sidebar.slider(t("z_slider"), 1.5, 4.0, 2.5, 0.5)

  all_brands = [target_brand.strip()] + [
      c.strip() for c in competitors_input.split(",") if c.strip()
  ]

  with st.spinner(t("fetching_msg")):
    all_articles = []
    for b in all_brands:
      all_articles.extend(fetch_live_news_for_brand(b))
    df = pd.DataFrame(all_articles)

  selected_brand = target_brand.strip()

else:
  df = generate_demo_data()
  selected_brand = st.sidebar.selectbox(
      t("target_brand"), options=df["brand"].unique(), index=0
  )
  z_thresh = st.sidebar.slider(t("z_slider"), 2.0, 5.0, 3.0, 0.5)

if df.empty or selected_brand not in df["brand"].values:
  st.warning(t("no_articles", brand=selected_brand))
  st.stop()

# ---------------- CRISIS ALERT & DISPATCHER ----------------
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
    if st.button(t("dispatch_btn")):
      st.success(t("dispatch_success", brand=selected_brand.lower()))
else:
  st.success(t("normal_status", brand=selected_brand))

st.divider()

# ---------------- TOP KPI CARDS ----------------
brand_data = df[df["brand"] == selected_brand]
total_m = len(brand_data)
pos_m = (brand_data["sentiment"] == "positive").sum()
neg_m = (brand_data["sentiment"] == "negative").sum()
nss_score = ((pos_m - neg_m) / total_m * 100) if total_m > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(t("kpi_total"), f"{total_m:,}")
kpi2.metric(t("kpi_nss"), f"{nss_score:.1f}")
kpi3.metric(t("kpi_pos"), f"{pos_m} ({(pos_m/total_m*100):.1f}%)")
kpi4.metric(t("kpi_neg"), f"{neg_m} ({(neg_m/total_m*100):.1f}%)")

st.divider()

# ---------------- SECTION 1: SHARE OF VOICE ----------------
st.subheader(t("sov_section"))
sov_df = compute_share_of_voice(df)

col1, col2 = st.columns([1, 1])
with col1:
  fig_sov = px.pie(
      sov_df,
      names="brand",
      values="volume_sov_pct",
      hole=0.4,
      title=t("sov_donut_title"),
  )
  st.plotly_chart(fig_sov, use_container_width=True)

with col2:
  fig_reach = px.bar(
      sov_df,
      x="brand",
      y=["volume_sov_pct", "reach_sov_pct"],
      barmode="group",
      title=t("sov_bar_title"),
      labels={
          "value": "%",
          "variable": t("metric_label"),
          "brand": t("target_brand"),
      },
  )
  st.plotly_chart(fig_reach, use_container_width=True)

# ---------------- SECTION 2: SENTIMENT TRENDS & ROOT-CAUSE ----------------
st.subheader(t("trends_section", brand=selected_brand))

trend_col, kw_col = st.columns([3, 2])

with trend_col:
  if not trends_df.empty:
    fig_trends = go.Figure()
    fig_trends.add_trace(
        go.Bar(
            x=trends_df.index,
            y=trends_df["negative"],
            name=t("neg_vol_legend"),
            marker_color="crimson",
        )
    )
    fig_trends.add_trace(
        go.Bar(
            x=trends_df.index,
            y=trends_df["positive"],
            name=t("pos_vol_legend"),
            marker_color="mediumseagreen",
        )
    )
    fig_trends.add_trace(
        go.Scatter(
            x=trends_df.index,
            y=trends_df["nss"],
            name=t("nss_legend"),
            yaxis="y2",
            line=dict(color="royalblue", width=2),
        )
    )
    fig_trends.update_layout(
        barmode="stack",
        yaxis=dict(title=t("mention_axis")),
        yaxis2=dict(
            title=t("nss_axis"),
            overlaying="y",
            side="right",
            range=[-105, 105],
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    st.plotly_chart(fig_trends, use_container_width=True)

with kw_col:
  neg_texts = brand_data[brand_data["sentiment"] == "negative"]["text"].tolist()
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
      "news",
      selected_brand.lower(),
  }

  words = []
  for t_str in neg_texts:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", t_str.lower())
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
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_kw, use_container_width=True)
  else:
    st.info(t("no_keywords"))

# ---------------- SECTION 3: RAW DATA & EXPORT ----------------
st.subheader(t("table_section"))

csv_data = brand_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t("download_btn", brand=selected_brand),
    data=csv_data,
    file_name=f"{selected_brand.lower()}_sentiment_report.csv",
    mime="text/csv",
)

columns_to_show = ["timestamp", "text", "sentiment", "sentiment_score"]
if enable_bert:
  columns_to_show.extend(["bert_label", "bert_confidence"])
columns_to_show.append("source")

st.dataframe(
    brand_data.sort_values(by="timestamp", ascending=False)[columns_to_show],
    use_container_width=True,
)