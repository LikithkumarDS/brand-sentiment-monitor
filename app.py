from collections import Counter
from datetime import datetime, timedelta
import json
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

st.set_page_config(
    page_title="Brand Sentiment & Reputation Monitor",
    layout="wide",
    page_icon="🛡️",
)

# ==============================================================================
# 1. 35-LANGUAGE TRANSLATION ENGINE
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

TRANSLATIONS = {
    "English": {
        "title": "🛡️ Brand Sentiment & Reputation Monitor",
        "subtitle": (
            "Data Science Core: VADER & BERT NLP | Data Analytics Core: Share"
            " of Voice & Live PR Crisis Engine"
        ),
        "settings_header": "⚙️ Brand & Language Settings",
        "target_brand": "Target Brand",
        "competitors": "Competitors (comma-separated)",
        "z_slider": "Crisis Alert Sensitivity (Z-Score)",
        "crisis_alert_title": "🚨 PR CRISIS SPIKE DETECTED FOR {brand}",
        "time_window": "Anomaly Time Window",
        "z_anomaly": "Z-Score Anomaly",
        "neg_volume": "Negative Volume",
        "articles": "mentions",
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
        "table_section": "3. Mentions Explorer & Sentiment Scores",
        "download_btn": "📥 Download {brand} Mentions Data as CSV",
    },
    "ಕನ್ನಡ (Kannada)": {
        "title": "🛡️ ಬ್ರ್ಯಾಂಡ್ ಸೆಂಟಿಮೆಂಟ್ ಮತ್ತು ಖ್ಯಾತಿ ಮಾನಿಟರ್",
        "subtitle": (
            "ಡೇಟಾ ಸೈನ್ಸ್: VADER & BERT NLP | ಅನಾಲಿಟಿಕ್ಸ್: ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ & ಪಿಆರ್"
            " ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್"
        ),
        "settings_header": "⚙️ ಬ್ರ್ಯಾಂಡ್ ಮತ್ತು ಭಾಷಾ ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
        "target_brand": "ಗುರಿ ಬ್ರ್ಯಾಂಡ್",
        "competitors": "ಸ್ಪರ್ಧಿಗಳು",
        "z_slider": "ಬಿಕ್ಕಟ್ಟು ಎಚ್ಚರಿಕೆ ಸೂಕ್ಷ್ಮತೆ (Z-Score)",
        "crisis_alert_title": "🚨 {brand} ಗಾಗಿ ಪಿಆರ್ ಬಿಕ್ಕಟ್ಟು ಎಚ್ಚರಿಕೆ!",
        "time_window": "ಅಸಂಗತತೆಯ ಸಮಯ",
        "z_anomaly": "Z-ಸ್ಕೋರ್ ಅಸಂಗತತೆ",
        "neg_volume": "ಋಣಾತ್ಮಕ ಪ್ರಮಾಣ",
        "articles": "ಉಲ್ಲೇಖಗಳು",
        "nss": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್ (NSS)",
        "normal_status": "✅ ಸಾಮಾನ್ಯ ಸ್ಥಿತಿ: ಯಾವುದೇ ನಕಾರಾತ್ಮಕ ಬಿಕ್ಕಟ್ಟು ಕಂಡುಬಂದಿಲ್ಲ.",
        "dispatch_expander": "📢 ಬಿಕ್ಕಟ್ಟು ಎಚ್ಚರಿಕೆ ಕಳುಹಿಸಿ (ವೆಬ್‌ಹುಕ್ / ಇಮೇಲ್)",
        "dispatch_btn": "🚀 ಪಿಆರ್ ತಂಡಕ್ಕೆ ವೆಬ್‌ಹುಕ್ ಎಚ್ಚರಿಕೆ ಕಳುಹಿಸಿ",
        "dispatch_success": "ಎಚ್ಚರಿಕೆಯನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಕಳುಹಿಸಲಾಗಿದೆ!",
        "kpi_total": "ಒಟ್ಟು ಉಲ್ಲೇಖಗಳು",
        "kpi_nss": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್",
        "kpi_pos": "ಧನಾತ್ಮಕ ಉಲ್ಲೇಖಗಳು",
        "kpi_neg": "ಋಣಾತ್ಮಕ ಉಲ್ಲೇಖಗಳು",
        "sov_section": "1. ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ (SoV) ಟ್ರ್ಯಾಕಿಂಗ್",
        "sov_donut_title": "ಪ್ರಮಾಣದ ಪ್ರಕಾರ ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ (%)",
        "sov_bar_title": "ವಾಲ್ಯೂಮ್ SoV vs ರೀಚ್ SoV (%)",
        "metric_label": "ಮೆಟ್ರಿಕ್",
        "trends_section": "2. ಸೆಂಟಿಮೆಂಟ್ ಟ್ರೆಂಡ್‌ಗಳು ({brand})",
        "neg_vol_legend": "ಋಣಾತ್ಮಕ ಪ್ರಮಾಣ",
        "pos_vol_legend": "ಧನಾತ್ಮಕ ಪ್ರಮಾಣ",
        "nss_legend": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ (NSS)",
        "mention_axis": "ಉಲ್ಲೇಖಗಳ ಸಂಖ್ಯೆ",
        "nss_axis": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ (-100 ರಿಂದ +100)",
        "top_keywords_title": "ಮುಖ್ಯ ಋಣಾತ್ಮಕ ಕೀವರ್ಡ್‌ಗಳು",
        "freq_label": "ಆವರ್ತನ",
        "keyword_label": "ಕೀವರ್ಡ್",
        "no_keywords": "ಯಾವುದೇ ಋಣಾತ್ಮಕ ಕೀವರ್ಡ್‌ಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
        "table_section": "3. ಉಲ್ಲೇಖಗಳ ಎಕ್ಸ್‌ಪ್ಲೋರರ್ ಮತ್ತು ವಿವರಗಳು",
        "download_btn": "📥 {brand} ಡೇಟಾವನ್ನು CSV ಆಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
    },
    "ಕೊಂಕಣಿ / कोंकणी (Konkani)": {
        "title": "🛡️ ಬ್ರ್ಯಾಂಡ್ ಸೆಂಟಿಮೆಂಟ್ ಆನಿ ಕೀರ್ತ್ ಮಾನಿಟರ್",
        "subtitle": (
            "ಡೇಟಾ ಸೈನ್ಸ್: VADER & BERT NLP | ಅನಾಲಿಟಿಕ್ಸ್: ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ ಆನಿ ಪಿಆರ್"
            " ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್"
        ),
        "settings_header": "⚙️ ಬ್ರ್ಯಾಂಡ್ ಆನಿ ಭಾಸ್ ಸೆಟ್ಟಿಂಗ್ಸ್",
        "target_brand": "ಮುಖೆಲ್ ಬ್ರ್ಯಾಂಡ್",
        "competitors": "ಸ್ಪರ್ಧಿ ಬ್ರ್ಯಾಂಡಾಂ",
        "z_slider": "ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್ ಸೂಕ್ಷ್ಮತಾ (Z-Score)",
        "crisis_alert_title": "🚨 {brand} ಖಾತಿರ್ ಪಿಆರ್ ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್!",
        "time_window": "ಅಸಂಗತತಾ ವೇಳ್ ವಿಂಡೋ",
        "z_anomaly": "Z-ಸ್ಕೋರ್ ಅಸಂಗತತಾ",
        "neg_volume": "ನಕಾರಾತ್ಮಕ್ ಪ್ರಮಾಣ್",
        "articles": "ಉಲ್ಲೇಖಾಂ",
        "nss": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್ (NSS)",
        "normal_status": "✅ ಸಾಧಾರಣ್ ಸ್ಥಿತಿ: ಕಸಲೆಂಯಿ ನಕಾರಾತ್ಮಕ್ ಸಂಕಷ್ಟ್ ನಾ.",
        "dispatch_expander": "📢 ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್ ಧಾಡಾ (ವೆಬ್‌ಹುಕ್ / ಇಮೇಲ್)",
        "dispatch_btn": "🚀 ಪಿಆರ್ ಟೀಮಾಕ್ ವೆಬ್‌ಹುಕ್ ಅಲರ್ಟ್ ಧಾಡಾ",
        "dispatch_success": "ಅಲರ್ಟ್ ಸಾರ್ಕೆಂ ಧಾಡ್ಲೆಂ!",
        "kpi_total": "ಒಟ್ಟು ಉಲ್ಲೇಖ್ (Mentions)",
        "kpi_nss": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್",
        "kpi_pos": "ಸಕಾರಾತ್ಮಕ್ ಉಲ್ಲೇಖ್",
        "kpi_neg": "ನಕಾರಾತ್ಮಕ್ ಉಲ್ಲೇಖ್",
        "sov_section": "1. ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ (SoV) ಟ್ರ್ಯಾಕಿಂಗ್",
        "sov_donut_title": "ಪ್ರಮಾಣಾ ಪ್ರಕಾರ್ ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ (%)",
        "sov_bar_title": "ವಾಲ್ಯೂಮ್ SoV vs ರೀಚ್ SoV (%)",
        "metric_label": "ಮೆಟ್ರಿಕ್",
        "trends_section": "2. ಸೆಂಟಿಮೆಂಟ್ ಟ್ರೆಂಡ್ ಆನಿ ಮುಖೆಲ್ ಕೀವರ್ಡ್ಸ್ ({brand})",
        "neg_vol_legend": "ನಕಾರಾತ್ಮಕ್ ಪ್ರಮಾಣ್",
        "pos_vol_legend": "ಸಕಾರಾತ್ಮಕ್ ಪ್ರಮಾಣ್",
        "nss_legend": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ (NSS)",
        "mention_axis": "ಉಲ್ಲೇಖಾಂಚಿ ಸಂಖ್ಯಾ",
        "nss_axis": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ (-100 ಥಾವ್ನ್ +100)",
        "top_keywords_title": "ಮುಖೆಲ್ ನಕಾರಾತ್ಮಕ್ ಕೀವರ್ಡ್ಸ್",
        "freq_label": "ಆವರ್ತನ್",
        "keyword_label": "ಕೀವರ್ಡ್",
        "no_keywords": "ಯಾವುದೇ ನಕಾರಾತ್ಮಕ್ ಕೀವರ್ಡ್ ಮೆಳುಂಕ್ ನಾ.",
        "table_section": "3. ಉಲ್ಲೇಖ್ ಎಕ್ಸ್‌ಪ್ಲೋರರ್ ಆನಿ ವಿವರಾಂ",
        "download_btn": "📥 {brand} ಡೇಟಾ CSV ಜಾವ್ನ್ ಡೌನ್‌ಲೋಡ್ ಕರಾ",
    },
    "हिन्दी (Hindi)": {
        "title": "🛡️ ब्रांड सेंटिमेंट और प्रतिष्ठा मॉनिटर",
        "subtitle": (
            "डेटा साइंस: VADER & BERT NLP | एनालिटिक्स: शेयर ऑफ वॉयस और पीआर"
            " क्राइसिस अलर्ट"
        ),
        "settings_header": "⚙️ ब्रांड और भाषा सेटिंग्स",
        "target_brand": "लक्षित ब्रांड",
        "competitors": "प्रतिस्पर्धी (अल्पविराम से अलग करें)",
        "z_slider": "क्राइसिस अलर्ट संवेदनशीलता (Z-Score)",
        "crisis_alert_title": "🚨 {brand} के लिए पीआर क्राइसिस अलर्ट!",
        "time_window": "विसंगति समय",
        "z_anomaly": "Z-स्कोर विसंगति",
        "neg_volume": "नकारात्मक मात्रा",
        "articles": "उल्लेख",
        "nss": "नेट सेंटिमेंट स्कोर (NSS)",
        "normal_status": (
            "✅ सामान्य स्थिति: कोई नकारात्मक संकट नहीं पाया गया।"
        ),
        "dispatch_expander": "📢 क्राइसिस अलर्ट भेजें (वेबहुक / ईमेल)",
        "dispatch_btn": "🚀 पीआर टीम को वेबहुक अलर्ट भेजें",
        "dispatch_success": "सूचना सफलतापूर्वक भेज दी गई!",
        "kpi_total": "कुल उल्लेख (Mentions)",
        "kpi_nss": "नेट सेंटिमेंट स्कोर",
        "kpi_pos": "सकारात्मक उल्लेख",
        "kpi_neg": "नकारात्मक उल्लेख",
        "sov_section": "1. शेयर ऑफ वॉयस (SoV) ट्रैकिंग",
        "sov_donut_title": "मात्रा के आधार पर शेयर ऑफ वॉयस (%)",
        "sov_bar_title": "वॉल्यूम SoV बनाम रीच SoV (%)",
        "metric_label": "मीट्रिक",
        "trends_section": "2. सेंटिमेंट रुझान ({brand})",
        "neg_vol_legend": "नकारात्मक मात्रा",
        "pos_vol_legend": "सकारात्मक मात्रा",
        "nss_legend": "नेट सेंटिमेंट (NSS)",
        "mention_axis": "उल्लेख संख्या",
        "nss_axis": "नेट सेंटिमेंट (-100 से +100)",
        "top_keywords_title": "शीर्ष नकारात्मक प्रेरक कीवर्ड्स",
        "freq_label": "आवृत्ति",
        "keyword_label": "कीवर्ड",
        "no_keywords": "कोई नकारात्मक कीवर्ड नहीं मिला।",
        "table_section": "3. उल्लेख एक्सप्लोरर और विवरण",
        "download_btn": "📥 {brand} डेटा को CSV के रूप में डाउनलोड करें",
    },
    "తెలుగు (Telugu)": {
        "title": "🛡️ బ్రాండ్ సెంటిమెంట్ & రెప్యుటేషన్ మానిటర్",
        "subtitle": (
            "డేటా సైన్స్: VADER & BERT NLP | ఎనలిటిక్స్: షేర్ ఆఫ్ వాయిస్ & సంక్షోభ"
            " హెచ్చరిక"
        ),
        "settings_header": "⚙️ బ్రాండ్ & భాషా సెట్టింగ్‌లు",
        "target_brand": "లక్ష్య బ్రాండ్",
        "competitors": "పోటీదారులు",
        "z_slider": "సంక్షోభ హెచ్చరిక సున్నితత్వం (Z-Score)",
        "crisis_alert_title": "🚨 {brand} కోసం సంక్షోభ హెచ్చరిక!",
        "time_window": "సమయ విండో",
        "z_anomaly": "Z-స్కోర్ అసాధారణత",
        "neg_volume": "ప్రతికూల పరిమాణం",
        "articles": "ప్రస్తావనలు",
        "nss": "నికర సెంటిమెంట్ స్కోర్ (NSS)",
        "normal_status": "✅ సాధారణ స్థితి: ప్రతికూల సంక్షోభం కనుగొనబడలేదు.",
        "dispatch_expander": "📢 హెచ్చరికను పంపండి (Webhook / Email)",
        "dispatch_btn": "🚀 PR బృందానికి హెచ్చరిక పంపండి",
        "dispatch_success": "హెచ్చరిక విజయవంతంగా పంపబడింది!",
        "kpi_total": "మొత్తం ప్రస్తావనలు",
        "kpi_nss": "నికర సెంటిమెంట్ స్కోర్",
        "kpi_pos": "సానుకూల ప్రస్తావనలు",
        "kpi_neg": "ప్రతికూల ప్రస్తావనలు",
        "sov_section": "1. షేర్ ఆఫ్ వాయిస్ (SoV)",
        "sov_donut_title": "వాల్యూమ్ ప్రకారం SoV (%)",
        "sov_bar_title": "వాల్యూమ్ vs రీచ్ SoV (%)",
        "metric_label": "మెట్రిక్",
        "trends_section": "2. సెంటిమెంట్ ట్రెండ్‌లు ({brand})",
        "neg_vol_legend": "ప్రతికూల పరిమాణం",
        "pos_vol_legend": "సానుకూల పరిమాణం",
        "nss_legend": "నికర సెంటిమెంట్",
        "mention_axis": "ప్రస్తావనల సంఖ్య",
        "nss_axis": "సెంటిమెంట్ (-100 to +100)",
        "top_keywords_title": "ముఖ్య ప్రతికూల కీలక పదాలు",
        "freq_label": "ఫ్రీక్వెన్సీ",
        "keyword_label": "కీలక పదం",
        "no_keywords": "ప్రతికూల పదాలు లేవు.",
        "table_section": "3. ప్రస్తావనల పట్టిక & వివరాలు",
        "download_btn": "📥 {brand} CSV ఫైల్‌ను డౌన్‌లోడ్ చేయండి",
    },
    "தமிழ் (Tamil)": {
        "title": "🛡️ பிராண்ட் உணர்வு & நற்பெயர் கண்காணிப்பாளர்",
        "subtitle": (
            "டேட்டா சயின்ஸ்: VADER & BERT NLP | அனலிட்டிக்ஸ்: வாய்ஸ் ஷேர் &"
            " நெருக்கடி எச்சரிக்கை"
        ),
        "settings_header": "⚙️ அமைப்புகள் & மொழி",
        "target_brand": "இலக்கு பிராண்ட்",
        "competitors": "போட்டியாளர்கள்",
        "z_slider": "எச்சரிக்கை உணர்திறன் (Z-Score)",
        "crisis_alert_title": "🚨 {brand} நெருக்கடி எச்சரிக்கை!",
        "time_window": "நேர சாளரம்",
        "z_anomaly": "Z-மதிப்பெண் முரண்பாடு",
        "neg_volume": "எதிர்மறை அளவு",
        "articles": "பதிவுகள்",
        "nss": "நிகர உணர்வு மதிப்பெண்",
        "normal_status": "✅ இயல்பான நிலை: நெருக்கடி கண்டறியப்படவில்லை.",
        "dispatch_expander": "📢 எச்சரிக்கை அனுப்பு (Webhook / Email)",
        "dispatch_btn": "🚀 PR குழுவிற்கு எச்சரிக்கை அனுப்பு",
        "dispatch_success": "எச்சரிக்கை வெற்றிகரமாக அனுப்பப்பட்டது!",
        "kpi_total": "மொத்த பதிவுகள்",
        "kpi_nss": "நிகர உணர்வு",
        "kpi_pos": "நேர்மறை பதிவுகள்",
        "kpi_neg": "எதிர்மறை பதிவுகள்",
        "sov_section": "1. வாய்ஸ் ஷேர் (SoV)",
        "sov_donut_title": "அளவு வாய்ஸ் ஷேர் (%)",
        "sov_bar_title": "அளவு vs சென்றடைவு SoV (%)",
        "metric_label": "அளவீடு",
        "trends_section": "2. போக்குகள் & முக்கிய வார்த்தைகள் ({brand})",
        "neg_vol_legend": "எதிர்மறை அளவு",
        "pos_vol_legend": "நேர்மறை அளவு",
        "nss_legend": "நிகர உணர்வு",
        "mention_axis": "பதிவுகளின் எண்ணிக்கை",
        "nss_axis": "உணர்வு (-100 to +100)",
        "top_keywords_title": "முக்கிய எதிர்மறை வார்த்தைகள்",
        "freq_label": "அதிர்வெண்",
        "keyword_label": "முக்கிய வார்த்தை",
        "no_keywords": "வார்த்தைகள் இல்லை.",
        "table_section": "3. பதிவுகள் பட்டியல்",
        "download_btn": "📥 {brand} CSV தரவிறக்கம்",
    },
    "മലയാളം (Malayalam)": {
        "title": "🛡️ ബ്രാൻഡ് സെന്റിമെന്റ് & പ്രശസ്തി മോണിറ്റർ",
        "subtitle": (
            "ഡാറ്റ സയൻസ്: VADER & BERT NLP | അനലിറ്റിക്സ്: ഷെയർ ഓഫ് വോയ്സ് &"
            " ക്രൈസിസ് അലേർട്ട്"
        ),
        "settings_header": "⚙️ ബ്രാൻഡ് & ഭാഷാ ക്രമീകരണങ്ങൾ",
        "target_brand": "ടാർഗെറ്റ് ബ്രാൻഡ്",
        "competitors": "മത്സരാർത്ഥികൾ",
        "z_slider": "അലേർട്ട് സെൻസിറ്റിവിറ്റി (Z-Score)",
        "crisis_alert_title": "🚨 {brand} പ്രതിസന്ധി മുന്നറിയിപ്പ്!",
        "time_window": "സമയ വിൻഡോ",
        "z_anomaly": "Z-സ്കോർ അപാകത",
        "neg_volume": "നെഗറ്റീവ് വോളിയം",
        "articles": "പരാമർശങ്ങൾ",
        "nss": "നെറ്റ് സെന്റിമെന്റ് സ്കോർ (NSS)",
        "normal_status": "✅ സാധാരണ നില: {brand} പ്രതിസന്ധി കണ്ടെത്തിയില്ല.",
        "dispatch_expander": "📢 അലേർട്ട് അയക്കുക (Webhook / Email)",
        "dispatch_btn": "🚀 അലേർട്ട് അയക്കുക",
        "dispatch_success": "അലേർട്ട് വിജയകരമായി അയച്ചു!",
        "kpi_total": "ആകെ പരാമർശങ്ങൾ",
        "kpi_nss": "നെറ്റ് സെന്റിമെന്റ് സ്കോർ",
        "kpi_pos": "പോസിറ്റീവ് പരാമർശങ്ങൾ",
        "kpi_neg": "നെഗറ്റീവ് പരാമർശങ്ങൾ",
        "sov_section": "1. ഷെയർ ഓഫ് വോയ്സ് (SoV)",
        "sov_donut_title": "വോളിയം SoV (%)",
        "sov_bar_title": "വോളിയം vs റീച്ച് SoV (%)",
        "metric_label": "മെട്രിക്",
        "trends_section": "2. സെന്റിമെന്റ് ട്രെൻഡുകൾ ({brand})",
        "neg_vol_legend": "നെഗറ്റീവ് വോളിയം",
        "pos_vol_legend": "പോസിറ്റീവ് വോളിയം",
        "nss_legend": "നെറ്റ് സെന്റിമെന്റ്",
        "mention_axis": "എണ്ണം",
        "nss_axis": "സെന്റിമെന്റ് (-100 to +100)",
        "top_keywords_title": "പ്രധാന നെഗറ്റീവ് വാക്കുകൾ",
        "freq_label": "ആവൃത്തി",
        "keyword_label": "കീവേഡ്",
        "no_keywords": "വാക്കുകൾ ലഭ്യമല്ല.",
        "table_section": "3. പരാമർശങ്ങളുടെ പട്ടിക",
        "download_btn": "📥 {brand} CSV ഡൗൺലോഡ് ചെയ്യുക",
    },
    "मराठी (Marathi)": {
        "title": "🛡️ ब्रँड सेंटिमेंट आणि प्रतिष्ठा मॉनिटर",
        "subtitle": (
            "डेटा सायन्स: VADER & BERT NLP | विश्लेषण: शेअर ऑफ व्हॉइस आणि संकट"
            " इशारा"
        ),
        "settings_header": "⚙️ ब्रँड आणि भाषा सेटिंग्ज",
        "target_brand": "लक्षित ब्रँड",
        "competitors": "स्पर्धक",
        "z_slider": "संकट इशारा संवेदनशीलता (Z-Score)",
        "crisis_alert_title": "🚨 {brand} साठी संकट इशारा!",
        "time_window": "वेळ विंडो",
        "z_anomaly": "Z-स्कोअर विसंगती",
        "neg_volume": "नकारात्मक संख्या",
        "articles": "उल्लेख",
        "nss": "नेट सेंटिमेंट स्कोअर (NSS)",
        "normal_status": (
            "✅ सामान्य स्थिती: कोणतीही नकारात्मक समस्या आढळली नाही."
        ),
        "dispatch_expander": "📢 इशारा पाठवा (Webhook / Email)",
        "dispatch_btn": "🚀 पीआर टीमला इशारा पाठवा",
        "dispatch_success": "इशारा यशस्वीरित्या पाठवला!",
        "kpi_total": "एकूण उल्लेख",
        "kpi_nss": "नेट सेंटिमेंट स्कोअर",
        "kpi_pos": "सकारात्मक उल्लेख",
        "kpi_neg": "नकारात्मक उल्लेख",
        "sov_section": "1. शेअर ऑफ व्हॉइस (SoV)",
        "sov_donut_title": "व्हॉल्यूम SoV (%)",
        "sov_bar_title": "व्हॉल्यूम वि. रीच SoV (%)",
        "metric_label": "मेट्रिक",
        "trends_section": "2. सेंटिमेंट ट्रेंड्स ({brand})",
        "neg_vol_legend": "नकारात्मक संख्या",
        "pos_vol_legend": "सकारात्मक संख्या",
        "nss_legend": "नेट सेंटिमेंट",
        "mention_axis": "उल्लेख संख्या",
        "nss_axis": "सेंटिमेंट (-100 to +100)",
        "top_keywords_title": "मुख्य नकारात्मक कीवर्ड",
        "freq_label": "वारंवारता",
        "keyword_label": "कीवर्ड",
        "no_keywords": "कोणतेही कीवर्ड आढळले नाहीत.",
        "table_section": "3. उल्लेख तक्ता",
        "download_btn": "📥 {brand} डेटा CSV डाउनलोड करा",
    },
    "বাংলা (Bengali)": {
        "title": "🛡️ ব্র্যান্ড সেন্টিমেন্ট ও খ্যাতি মনিটর",
        "subtitle": (
            "ডেটা সায়েন্স: VADER & BERT NLP | অ্যানালিটিক্স: শেয়ার অফ ভয়েস ও"
            " সংকট সতর্কতা"
        ),
        "settings_header": "⚙️ ব্র্যান্ড ও ভাষা সেটিংস",
        "target_brand": "টার্গেট ব্র্যান্ড",
        "competitors": "প্রতিযোগী",
        "z_slider": "সতর্কতা সংবেদনশীলতা (Z-Score)",
        "crisis_alert_title": "🚨 {brand} এর জন্য পিআর সংকট সতর্কতা!",
        "time_window": "সময় উইন্ডো",
        "z_anomaly": "Z-স্কোর অসঙ্গতি",
        "neg_volume": "নেতিবাচক পরিমাণ",
        "articles": "উল্লেখ",
        "nss": "নেট সেন্টিমেন্ট স্কোর (NSS)",
        "normal_status": "✅ স্বাভাবিক অবস্থা: কোনো সংকট ধরা পড়েনি।",
        "dispatch_expander": "📢 সতর্কতা পাঠান (Webhook / Email)",
        "dispatch_btn": "🚀 পিআর টিমকে সতর্কতা পাঠান",
        "dispatch_success": "সতর্কতা সফলভাবে পাঠানো হয়েছে!",
        "kpi_total": "মোট উল্লেখ",
        "kpi_nss": "নেট সেন্টিমেন্ট স্কোর",
        "kpi_pos": "ইতিবাচক উল্লেখ",
        "kpi_neg": "নেতিবাচক উল্লেখ",
        "sov_section": "1. শেয়ার অফ ভয়েস (SoV)",
        "sov_donut_title": "ভলিউম SoV (%)",
        "sov_bar_title": "ভলিউম বনাম রিচ SoV (%)",
        "metric_label": "মেট্রিক",
        "trends_section": "2. সেন্টিমেন্ট ট্রেন্ড ({brand})",
        "neg_vol_legend": "নেতিবাচক পরিমাণ",
        "pos_vol_legend": "ইতিবাচক পরিমাণ",
        "nss_legend": "নেট সেন্টিমেন্ট",
        "mention_axis": "উল্লেখের সংখ্যা",
        "nss_axis": "সেন্টিমেন্ট (-100 to +100)",
        "top_keywords_title": "শীর্ষ নেতিবাচক কিওয়ার্ড",
        "freq_label": "ফ্রিকোয়েন্সি",
        "keyword_label": "কিওয়ার্ড",
        "no_keywords": "কোনো কিওয়ার্ড পাওয়া যায়নি।",
        "table_section": "3. উল্লেখ তালিকা",
        "download_btn": "📥 {brand} ডেটা CSV ডাউনলোড করুন",
    },
    "ગુજરાતી (Gujarati)": {
        "title": "🛡️ બ્રાન્ડ સેન્ટિમેન્ટ અને પ્રતિષ્ઠા મોનિટર",
        "subtitle": (
            "ડેટા સાયન્સ: VADER & BERT NLP | એનાલિટિક્સ: શેર ઓફ વૉઇસ અને કટોકટી"
            " ચેતવણી"
        ),
        "settings_header": "⚙️ બ્રાન્ડ અને ભાષા સેટિંગ્સ",
        "target_brand": "ટાર્ગેટ બ્રાન્ડ",
        "competitors": "હરીફો",
        "z_slider": "ચેતવણી સંવેદનશીલતા (Z-Score)",
        "crisis_alert_title": "🚨 {brand} માટે કટોકટી ચેતવણી!",
        "time_window": "સમય વિંડો",
        "z_anomaly": "Z-સ્કોર વિસંગતતા",
        "neg_volume": "નકારાત્મક વોલ્યુમ",
        "articles": "ઉલ્લેખ",
        "nss": "નેટ સેન્ટિમેન્ટ સ્કોર (NSS)",
        "normal_status": "✅ સામાન્ય સ્થિતિ: કોઈ કટોકટી મળી નથી.",
        "dispatch_expander": "📢 ચેતવણી મોકલો (Webhook / Email)",
        "dispatch_btn": "🚀 PR ટીમને ચેતવણી મોકલો",
        "dispatch_success": "ચેતવણી સફળતાપૂર્વક મોકલી!",
        "kpi_total": "કુલ ઉલ્લેખ",
        "kpi_nss": "નેટ સેન્ટિમેન્ટ સ્કોર",
        "kpi_pos": "હકારાત્મક ઉલ્લેખ",
        "kpi_neg": "નકારાત્મક ઉલ્લેખ",
        "sov_section": "1. શેર ઓફ વૉઇસ (SoV)",
        "sov_donut_title": "વોલ્યુમ SoV (%)",
        "sov_bar_title": "વોલ્યુમ વિ. રીચ SoV (%)",
        "metric_label": "મેટ્રિક",
        "trends_section": "2. સેન્ટિમેન્ટ ટ્રેન્ડ્સ ({brand})",
        "neg_vol_legend": "નકારાત્મક વોલ્યુમ",
        "pos_vol_legend": "હકારાત્મક વોલ્યુમ",
        "nss_legend": "નેટ સેન્ટિમેન્ટ",
        "mention_axis": "સંખ્યા",
        "nss_axis": "સેન્ટિમેન્ટ (-100 to +100)",
        "top_keywords_title": "મુખ્ય નકારાત્મક કીવર્ડ્સ",
        "freq_label": "આવર્તન",
        "keyword_label": "કીવર્ડ",
        "no_keywords": "કીવર્ડ્સ મળ્યા નથી.",
        "table_section": "3. ઉલ્લેખ કોષ્ટક",
        "download_btn": "📥 {brand} CSV ડાઉનલોડ કરો",
    },
    "Español (Spanish)": {
        "title": "🛡️ Monitor de Reputación y Sentimiento de Marca",
        "subtitle": (
            "Ciencia de Datos: VADER y BERT NLP | Analítica: Cuota de Voz y"
            " Alerta de Crisis"
        ),
        "settings_header": "⚙️ Configuración de Marca e Idioma",
        "target_brand": "Marca Objetivo",
        "competitors": "Competidores",
        "z_slider": "Sensibilidad de Alerta de Crisis (Z-Score)",
        "crisis_alert_title": (
            "🚨 ALERTA DE CRISIS DE RP DETECTADA PARA {brand}"
        ),
        "time_window": "Ventana de Tiempo",
        "z_anomaly": "Anomalía Z-Score",
        "neg_volume": "Volumen Negativo",
        "articles": "menciones",
        "nss": "Puntuación Neta de Sentimiento (NSS)",
        "normal_status": (
            "✅ Estado Normal: No se detectaron picos negativos para {brand}."
        ),
        "dispatch_expander": "📢 Enviar Alerta de Crisis (Webhook / Correo)",
        "dispatch_btn": "🚀 Enviar Webhook al Equipo de RP",
        "dispatch_success": "¡Notificación enviada con éxito!",
        "kpi_total": "Total de Menciones",
        "kpi_nss": "Sentimiento Neto",
        "kpi_pos": "Menciones Positivas",
        "kpi_neg": "Menciones Negativas",
        "sov_section": "1. Cuota de Voz (SoV)",
        "sov_donut_title": "Cuota de Voz por Volumen (%)",
        "sov_bar_title": "SoV Volumen vs. SoV Alcance (%)",
        "metric_label": "Métrica",
        "trends_section": "2. Tendencias de Sentimiento ({brand})",
        "neg_vol_legend": "Volumen Negativo",
        "pos_vol_legend": "Volumen Positivo",
        "nss_legend": "Sentimiento Neto",
        "mention_axis": "Cantidad de Menciones",
        "nss_axis": "Sentimiento Neto (-100 a +100)",
        "top_keywords_title": "Palabras Clave Negativas",
        "freq_label": "Frecuencia",
        "keyword_label": "Palabra Clave",
        "no_keywords": "No se encontraron palabras clave negativas.",
        "table_section": "3. Explorador de Menciones",
        "download_btn": "📥 Descargar Datos de {brand} en CSV",
    },
    "Français (French)": {
        "title": "🛡️ Moniteur de Réputation et Sentiment de Marque",
        "subtitle": (
            "Data Science : NLP VADER & BERT | Analytique : Part de Voix &"
            " Alerte de Crise"
        ),
        "settings_header": "⚙️ Paramètres et Langue",
        "target_brand": "Marque Cible",
        "competitors": "Concurrents",
        "z_slider": "Sensibilité d'Alerte de Crise (Score Z)",
        "crisis_alert_title": "🚨 ALERTE DE CRISE RP DÉTECTÉE POUR {brand}",
        "time_window": "Fenêtre Temporelle",
        "z_anomaly": "Anomalie Score Z",
        "neg_volume": "Volume Négatif",
        "articles": "mentions",
        "nss": "Score Net de Sentiment (NSS)",
        "normal_status": (
            "✅ Statut Normal : Aucun pic négatif détecté pour {brand}."
        ),
        "dispatch_expander": "📢 Déclencher Alerte de Crise (Webhook / Email)",
        "dispatch_btn": "🚀 Envoyer Webhook à l'Équipe RP",
        "dispatch_success": "Notification envoyée avec succès !",
        "kpi_total": "Total des Mentions",
        "kpi_nss": "Sentiment Net",
        "kpi_pos": "Mentions Positives",
        "kpi_neg": "Mentions Négatives",
        "sov_section": "1. Part de Voix (SoV)",
        "sov_donut_title": "Part de Voix par Volume (%)",
        "sov_bar_title": "SoV Volume vs. SoV Portée (%)",
        "metric_label": "Métrique",
        "trends_section": "2. Tendances de Sentiment ({brand})",
        "neg_vol_legend": "Volume Négatif",
        "pos_vol_legend": "Volume Positif",
        "nss_legend": "Sentiment Net",
        "mention_axis": "Nombre de Mentions",
        "nss_axis": "Sentiment Net (-100 à +100)",
        "top_keywords_title": "Mots-Clés Négatifs Principaux",
        "freq_label": "Fréquence",
        "keyword_label": "Mot-Clé",
        "no_keywords": "Aucun mot négatif trouvé.",
        "table_section": "3. Explorateur de Mentions",
        "download_btn": "📥 Télécharger les Données de {brand} en CSV",
    },
    "Deutsch (German)": {
        "title": "🛡️ Marken-Sentiment & Reputations-Monitor",
        "subtitle": (
            "Data Science: VADER & BERT NLP | Analytik: Share of Voice &"
            " PR-Krisenwarnung"
        ),
        "settings_header": "⚙️ Einstellungen & Sprache",
        "target_brand": "Zielmarke",
        "competitors": "Wettbewerber",
        "z_slider": "Krisenalarm-Empfindlichkeit (Z-Score)",
        "crisis_alert_title": "🚨 PR-KRISENALARM FÜR {brand} ERKANNT",
        "time_window": "Zeitfenster",
        "z_anomaly": "Z-Score-Anomalie",
        "neg_volume": "Negatives Volumen",
        "articles": "Erwähnungen",
        "nss": "Netto-Sentiment-Wert (NSS)",
        "normal_status": (
            "✅ Normaler Status: Keine negativen Spitzenwerte für {brand}."
        ),
        "dispatch_expander": "📢 Krisenalarm Senden (Webhook / E-Mail)",
        "dispatch_btn": "🚀 Webhook an PR-Team Senden",
        "dispatch_success": "Benachrichtigung erfolgreich gesendet!",
        "kpi_total": "Gesamterwähnungen",
        "kpi_nss": "Netto-Sentiment",
        "kpi_pos": "Positive Erwähnungen",
        "kpi_neg": "Negative Erwähnungen",
        "sov_section": "1. Share of Voice (SoV)",
        "sov_donut_title": "Share of Voice nach Volumen (%)",
        "sov_bar_title": "Volumen-SoV vs. Reichweiten-SoV (%)",
        "metric_label": "Metrik",
        "trends_section": "2. Sentiment-Trends ({brand})",
        "neg_vol_legend": "Negatives Volumen",
        "pos_vol_legend": "Positives Volumen",
        "nss_legend": "Netto-Sentiment",
        "mention_axis": "Anzahl Erwähnungen",
        "nss_axis": "Netto-Sentiment (-100 bis +100)",
        "top_keywords_title": "Negative Treiber-Schlagwörter",
        "freq_label": "Häufigkeit",
        "keyword_label": "Schlagwort",
        "no_keywords": "Keine negativen Schlagwörter gefunden.",
        "table_section": "3. Erwähnungs-Explorer",
        "download_btn": "📥 Daten für {brand} als CSV Herunterladen",
    },
    "Italiano (Italian)": {
        "title": "🛡️ Monitor di Reputazione e Sentiment del Brand",
        "subtitle": (
            "Data Science: VADER e BERT NLP | Analisi: Quota di Voce e Allerte"
            " di Crisi PR"
        ),
        "settings_header": "⚙️ Impostazioni e Lingua",
        "target_brand": "Brand Obiettivo",
        "competitors": "Concorrenti",
        "z_slider": "Sensibilità Allerta Crisi (Z-Score)",
        "crisis_alert_title": "🚨 ALLERTA CRISI PR RILEVATA PER {brand}",
        "time_window": "Finestra Temporale",
        "z_anomaly": "Anomalia Z-Score",
        "neg_volume": "Volume Negativo",
        "articles": "menzioni",
        "nss": "Punteggio Netto di Sentiment (NSS)",
        "normal_status": (
            "✅ Stato Normale: Nessun picco negativo rilevato per {brand}."
        ),
        "dispatch_expander": "📢 Invia Allerta Crisi (Webhook / Email)",
        "dispatch_btn": "🚀 Invia Webhook al Team PR",
        "dispatch_success": "Notifica inviata con successo!",
        "kpi_total": "Menzioni Totali",
        "kpi_nss": "Sentiment Netto",
        "kpi_pos": "Menzioni Positive",
        "kpi_neg": "Menzioni Negative",
        "sov_section": "1. Quota di Voce (SoV)",
        "sov_donut_title": "Quota di Voce per Volume (%)",
        "sov_bar_title": "SoV Volume vs SoV Portata (%)",
        "metric_label": "Metrica",
        "trends_section": "2. Trend Sentiment ({brand})",
        "neg_vol_legend": "Volume Negativo",
        "pos_vol_legend": "Volume Positivo",
        "nss_legend": "Sentiment Netto",
        "mention_axis": "Numero di Menzioni",
        "nss_axis": "Sentiment Netto (-100 a +100)",
        "top_keywords_title": "Parole Chiave Negative",
        "freq_label": "Frequenza",
        "keyword_label": "Parola Chiave",
        "no_keywords": "Nessuna parola chiave negativa trovata.",
        "table_section": "3. Esploratore Menzioni",
        "download_btn": "📥 Scarica Dati di {brand} in CSV",
    },
    "Português (Portuguese)": {
        "title": "🛡️ Monitor de Reputação e Sentimento da Marca",
        "subtitle": (
            "Ciência de Dados: VADER e BERT NLP | Análise: Share of Voice e"
            " Alerta de Crise"
        ),
        "settings_header": "⚙️ Configurações e Idioma",
        "target_brand": "Marca Alvo",
        "competitors": "Concorrentes",
        "z_slider": "Sensibilidade do Alerta (Z-Score)",
        "crisis_alert_title": "🚨 ALERTA DE CRISE DE RP DETECTADO PARA {brand}",
        "time_window": "Janela de Tempo",
        "z_anomaly": "Anomalia Z-Score",
        "neg_volume": "Volume Negativo",
        "articles": "menções",
        "nss": "Pontuação Líquida de Sentimento (NSS)",
        "normal_status": (
            "✅ Status Normal: Nenhum pico negativo detectado para {brand}."
        ),
        "dispatch_expander": "📢 Disparar Alerta de Crise (Webhook / Email)",
        "dispatch_btn": "🚀 Enviar Webhook para Equipe de RP",
        "dispatch_success": "Notificação enviada com sucesso!",
        "kpi_total": "Total de Menções",
        "kpi_nss": "Sentimento Líquido",
        "kpi_pos": "Menções Positivas",
        "kpi_neg": "Menções Negativas",
        "sov_section": "1. Share of Voice (SoV)",
        "sov_donut_title": "Share of Voice por Volume (%)",
        "sov_bar_title": "SoV Volume vs. SoV Alcance (%)",
        "metric_label": "Métrica",
        "trends_section": "2. Tendências de Sentimento ({brand})",
        "neg_vol_legend": "Volume Negativo",
        "pos_vol_legend": "Volume Positivo",
        "nss_legend": "Sentimento Líquido",
        "mention_axis": "Total de Menções",
        "nss_axis": "Sentimento Líquido (-100 a +100)",
        "top_keywords_title": "Principais Palavras-Chave Negativas",
        "freq_label": "Frequência",
        "keyword_label": "Palavra-Chave",
        "no_keywords": "Nenhuma palavra negativa encontrada.",
        "table_section": "3. Explorador de Menções",
        "download_btn": "📥 Baixar Dados de {brand} em CSV",
    },
    "Русский (Russian)": {
        "title": "🛡️ Мониторинг репутации и тональности бренда",
        "subtitle": (
            "Data Science: VADER и BERT NLP | Аналитика: Доля голоса и"
            " оповещения о кризисах"
        ),
        "settings_header": "⚙️ Настройки бренда и языка",
        "target_brand": "Целевой бренд",
        "competitors": "Конкуренты",
        "z_slider": "Чувствительность к кризисам (Z-Score)",
        "crisis_alert_title": "🚨 ПР-КРИЗИС ОБНАРУЖЕН ДЛЯ {brand}",
        "time_window": "Временное окно",
        "z_anomaly": "Аномалия Z-Score",
        "neg_volume": "Негативный объем",
        "articles": "упоминаний",
        "nss": "Индекс чистой тональности (NSS)",
        "normal_status": (
            "✅ Нормальный статус: Кризисных всплесков для {brand} нет."
        ),
        "dispatch_expander": "📢 Отправить оповещение (Webhook / Email)",
        "dispatch_btn": "🚀 Отправить Webhook команде PR",
        "dispatch_success": "Оповещение успешно отправлено!",
        "kpi_total": "Всего упоминаний",
        "kpi_nss": "Чистая тональность",
        "kpi_pos": "Позитивные упоминания",
        "kpi_neg": "Негативные упоминания",
        "sov_section": "1. Доля голоса (Share of Voice)",
        "sov_donut_title": "Доля голоса по объему (%)",
        "sov_bar_title": "Объем vs Охват SoV (%)",
        "metric_label": "Метрика",
        "trends_section": "2. Тренды тональности ({brand})",
        "neg_vol_legend": "Негативный объем",
        "pos_vol_legend": "Позитивный объем",
        "nss_legend": "Чистая тональность",
        "mention_axis": "Число упоминаний",
        "nss_axis": "Тональность (-100 до +100)",
        "top_keywords_title": "Ключевые негативные слова",
        "freq_label": "Частота",
        "keyword_label": "Ключевое слово",
        "no_keywords": "Негативных слов не найдено.",
        "table_section": "3. Таблица упоминаний",
        "download_btn": "📥 Скачать данные {brand} в CSV",
    },
    "日本語 (Japanese)": {
        "title": "🛡️ ブランド・センチメント＆評判モニター",
        "subtitle": (
            "データサイエンス: VADER & BERT NLP | 分析: シェア・オブ・ボイス &"
            " PR危機アラート"
        ),
        "settings_header": "⚙️ ブランドと言語の設定",
        "target_brand": "対象ブランド",
        "competitors": "競合他社",
        "z_slider": "危機アラート感度 (Zスコア)",
        "crisis_alert_title": "🚨 {brand} のPR危機アラートを検知！",
        "time_window": "検知ウィンドウ",
        "z_anomaly": "Zスコア異常値",
        "neg_volume": "ネガティブ量",
        "articles": "言及数",
        "nss": "純感情スコア (NSS)",
        "normal_status": (
            "✅ 正常ステータス: {brand} に異常なネガティブ急増はありません。"
        ),
        "dispatch_expander": "📢 危機アラートを送信 (Webhook / メール)",
        "dispatch_btn": "🚀 PRチームにアラートを送信",
        "dispatch_success": "通知が正常に送信されました！",
        "kpi_total": "総言及数",
        "kpi_nss": "純感情スコア",
        "kpi_pos": "ポジティブ言及",
        "kpi_neg": "ネガティブ言及",
        "sov_section": "1. シェア・オブ・ボイス (SoV)",
        "sov_donut_title": "言及量シェア (%)",
        "sov_bar_title": "言及量 vs リーチ加重 SoV (%)",
        "metric_label": "指標",
        "trends_section": "2. 感情トレンド ({brand})",
        "neg_vol_legend": "ネガティブ量",
        "pos_vol_legend": "ポジティブ量",
        "nss_legend": "純感情 (NSS)",
        "mention_axis": "言及数",
        "nss_axis": "純感情スコア (-100 〜 +100)",
        "top_keywords_title": "主なネガティブ・キーワード",
        "freq_label": "頻度",
        "keyword_label": "キーワード",
        "no_keywords": "ネガティブキーワードは見つかりませんでした。",
        "table_section": "3. 言及エクスプローラー",
        "download_btn": "📥 {brand} のデータをCSVでダウンロード",
    },
    "한국어 (Korean)": {
        "title": "🛡️ 브랜드 감성 및 평판 모니터",
        "subtitle": (
            "데이터 사이언스: VADER & BERT NLP | 분석: 점유율(SoV) 및 실시간"
            " 위기 알림"
        ),
        "settings_header": "⚙️ 브랜드 및 언어 설정",
        "target_brand": "목표 브랜드",
        "competitors": "경쟁사",
        "z_slider": "위기 경보 민감도 (Z-Score)",
        "crisis_alert_title": "🚨 {brand} 홍보 위기 경보 감지!",
        "time_window": "이상 징후 시간대",
        "z_anomaly": "Z-점수 이상치",
        "neg_volume": "부정적 언급량",
        "articles": "건",
        "nss": "순 감성 지수 (NSS)",
        "normal_status": (
            "✅ 정상 상태: {brand}에 비정상적인 부정 급증이 없습니다."
        ),
        "dispatch_expander": "📢 위기 경보 발송 (웹훅 / 이메일)",
        "dispatch_btn": "🚀 홍보팀에 웹훅 경보 발송",
        "dispatch_success": "알림이 성공적으로 발송되었습니다!",
        "kpi_total": "총 언급량",
        "kpi_nss": "순 감성 지수",
        "kpi_pos": "긍정적 언급",
        "kpi_neg": "부정적 언급",
        "sov_section": "1. 보이스 점유율 (SoV) 추적",
        "sov_donut_title": "언급량 기준 점유율 (%)",
        "sov_bar_title": "언급량 vs 도달 범위 점유율 (%)",
        "metric_label": "지표",
        "trends_section": "2. 감성 추이 및 주요 키워드 ({brand})",
        "neg_vol_legend": "부정적 언급량",
        "pos_vol_legend": "긍정적 언급량",
        "nss_legend": "순 감성 지수",
        "mention_axis": "언급 수",
        "nss_axis": "순 감성 (-100 ~ +100)",
        "top_keywords_title": "주요 부정 키워드",
        "freq_label": "빈도",
        "keyword_label": "키워드",
        "no_keywords": "부정 키워드가 없습니다.",
        "table_section": "3. 언급 탐색기",
        "download_btn": "📥 {brand} 데이터 CSV 다운로드",
    },
    "简体中文 (Chinese Simplified)": {
        "title": "🛡️ 品牌情感与声誉监控器",
        "subtitle": (
            "数据科学核心: VADER与BERT NLP | 数据分析核心: 声量份额与实时公关危机预警"
        ),
        "settings_header": "⚙️ 品牌与语言设置",
        "target_brand": "目标品牌",
        "competitors": "竞争对手",
        "z_slider": "危机警报灵敏度 (Z-Score)",
        "crisis_alert_title": "🚨 监测到 {brand} 公关危机激增！",
        "time_window": "异常时间窗口",
        "z_anomaly": "Z-Score 异常值",
        "neg_volume": "负面提及量",
        "articles": "条提及",
        "nss": "净情感得分 (NSS)",
        "normal_status": "✅ 正常状态: 未检测到 {brand} 异常负面激增。",
        "dispatch_expander": "📢 发送公关危机预警 (Webhook / 邮件)",
        "dispatch_btn": "🚀 发送模拟 Webhook 预警至公关团队",
        "dispatch_success": "通知已成功发送至公关团队！",
        "kpi_total": "总提及量",
        "kpi_nss": "净情感得分",
        "kpi_pos": "正面提及",
        "kpi_neg": "负面提及",
        "sov_section": "1. 声量份额 (SoV) 跟踪",
        "sov_donut_title": "提及声量份额 (%)",
        "sov_bar_title": "声量份额 vs 触达加权份额 (%)",
        "metric_label": "指标",
        "trends_section": "2. 情感趋势与核心关键词 ({brand})",
        "neg_vol_legend": "负面提及量",
        "pos_vol_legend": "正面提及量",
        "nss_legend": "净情感得分",
        "mention_axis": "提及次数",
        "nss_axis": "净情感 (-100 至 +100)",
        "top_keywords_title": "主要负面驱动关键词",
        "freq_label": "频次",
        "keyword_label": "关键词",
        "no_keywords": "未发现负面关键词。",
        "table_section": "3. 提及浏览与详情",
        "download_btn": "📥 下载 {brand} 数据为 CSV",
    },
    "العربية (Arabic)": {
        "title": "🛡️ مراقب سمعة وانطباعات العلامة التجارية",
        "subtitle": (
            "علم البيانات: VADER و BERT NLP | التحليلات: حصة الصوت وتنبيهات"
            " الأزمات"
        ),
        "settings_header": "⚙️ إعدادات العلامة التجارية واللغة",
        "target_brand": "العلامة التجارية المستهدفة",
        "competitors": "المنافسون",
        "z_slider": "حساسية تنبيه الأزمة (Z-Score)",
        "crisis_alert_title": "🚨 تم رصد أزمة علاقات عامة لـ {brand}!",
        "time_window": "نافذة وقت الشذوذ",
        "z_anomaly": "شذوذ Z-Score",
        "neg_volume": "الحجم السلبي",
        "articles": "إشارة",
        "nss": "صافي نقاط الانطباع (NSS)",
        "normal_status": "✅ الوضع طبيعي: لا توجد أزمات لـ {brand}.",
        "dispatch_expander": "📢 إرسال تنبيه الأزمة (معاينة Webhook / البريد)",
        "dispatch_btn": "🚀 إرسال تنبيه Webhook لفريق العلاقات العامة",
        "dispatch_success": "تم إرسال التنبيه بنجاح!",
        "kpi_total": "إجمالي الإشارات",
        "kpi_nss": "صافي نقاط الانطباع",
        "kpi_pos": "إشارات إيجابية",
        "kpi_neg": "إشارات سلبية",
        "sov_section": "1. تتبع حصة الصوت (SoV)",
        "sov_donut_title": "حصة الصوت من حيث الحجم (%)",
        "sov_bar_title": "حصة الصوت الحجمية مقابل الوصول (%)",
        "metric_label": "المقياس",
        "trends_section": (
            "2. اتجاهات الانطباعات والكلمات المفتاحية ({brand})"
        ),
        "neg_vol_legend": "حجم سلبي",
        "pos_vol_legend": "حجم إيجابي",
        "nss_legend": "صافي الانطباع",
        "mention_axis": "عدد الإشارات",
        "nss_axis": "صافي الانطباع (-100 إلى +100)",
        "top_keywords_title": "أبرز الكلمات السلبية",
        "freq_label": "التكرار",
        "keyword_label": "الكلمة المفتاحية",
        "no_keywords": "لا توجد كلمات سلبية.",
        "table_section": "3. جدول الإشارات",
        "download_btn": "📥 تنزيل بيانات {brand} بصيغة CSV",
    },
}

selected_lang = st.sidebar.selectbox(
    "🌐 Language / ಭಾಷೆ / भाषा", list(LANGUAGES.keys()), index=0
)


# ---------------- FAST BATCH AUTO-TRANSLATION FOR OTHER LANGUAGES ----------------
@st.cache_data(show_spinner=False)
def get_translated_dict(target_code: str) -> dict:
  """Translates the complete dictionary in a single fast call for other languages."""
  base_dict = TRANSLATIONS["English"]
  if target_code == "en":
    return base_dict

  keys = list(base_dict.keys())
  safe_lines = [base_dict[k].replace("{brand}", "XYZ_BRAND") for k in keys]
  payload = "\n".join(safe_lines)

  try:
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_code}&dt=t&q={urllib.parse.quote(payload)}"
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=3.5,
    )
    if resp.status_code == 200:
      res_json = resp.json()
      translated_full = "".join([part[0] for part in res_json[0] if part[0]])
      translated_lines = translated_full.split("\n")

      if len(translated_lines) == len(keys):
        return {
            k: translated_lines[i].replace("XYZ_BRAND", "{brand}")
            for i, k in enumerate(keys)
        }
  except Exception:
    pass
  return base_dict


def t(key: str, **kwargs) -> str:
  lang_code = LANGUAGES.get(selected_lang, "en")
  # 1. Use pre-compiled dictionary if available (instant)
  if selected_lang in TRANSLATIONS and key in TRANSLATIONS[selected_lang]:
    text = TRANSLATIONS[selected_lang][key]
  else:
    # 2. Automated batch translation for the remaining languages
    trans_map = get_translated_dict(lang_code)
    text = trans_map.get(key, TRANSLATIONS["English"].get(key, key))

  if kwargs:
    try:
      text = text.format(**kwargs)
    except Exception:
      pass
  return text


st.title(t("title"))
st.caption(t("subtitle"))

# ==============================================================================
# 2. DATA SCIENCE NLP SCORING (VADER)
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
target_brand_input = st.sidebar.text_input(t("target_brand"), value="Tesla")
competitors_input = st.sidebar.text_input(
    t("competitors"), value="BYD, Rivian"
)
z_thresh = st.sidebar.slider(t("z_slider"), 1.5, 4.0, 2.5, 0.5)

selected_brand = target_brand_input.strip() if target_brand_input.strip() else "Tesla"
competitors = [
    c.strip() for c in competitors_input.split(",") if c.strip()
]
if not competitors:
  competitors = ["CompetitorA", "CompetitorB"]
all_brands = [selected_brand] + competitors


# ==============================================================================
# 4. INSTANT RELIABLE DATA GENERATOR (NEVER BLANK)
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

    # Injected Crisis spike for the target brand on Day 5
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
# 5. DATA ANALYTICS CORE: CRISIS SPIKE ENGINE
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

# Rolling Z-Score Anomaly detection
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
      f" {active_crises.index[-1].strftime('%Y-%m-%d %H:%M')}\n"
      f"- **{t('z_anomaly')}:** {latest_crisis['z_score']:.2f}σ\n"
      f"- **{t('neg_volume')}:** {int(latest_crisis['negative'])}"
      f" {t('articles')}\n"
      f"- **{t('nss')}:** {latest_crisis['nss']:.1f}"
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

c1, c2 = st.columns([1, 1])
with c1:
  fig_donut = px.pie(
      sov_df,
      names="brand",
      values="volume_sov_pct",
      hole=0.4,
      title=t("sov_donut_title"),
  )
  st.plotly_chart(fig_donut, use_container_width=True)

with c2:
  fig_bar = px.bar(
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
            marker_color="crimson",
        )
    )
  fig_trends.add_trace(
        go.Bar(
            x=grouped.index,
            y=grouped["positive"],
            name=t("pos_vol_legend"),
            marker_color="mediumseagreen",
        )
    )
  fig_trends.add_trace(
        go.Scatter(
            x=grouped.index,
            y=grouped["nss"],
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
        color_continuous_scale="Reds",
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
)

st.dataframe(
    target_mentions.sort_values(by="timestamp", ascending=False)[
        ["timestamp", "text", "sentiment", "sentiment_score", "source"]
    ],
    use_container_width=True,
)