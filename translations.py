# translations.py - 35-Language Localization Engine with Konkani Support
import json
import urllib.parse
import urllib.request
import streamlit as st

LANGUAGE_CODES = {
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
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ିଆ (Odia)": "or",
    "অসমীয়া (Assamese)": "as",
    "संस्कृतम् (Sanskrit)": "sa",
    "मैथिली (Maithili)": "mai",
    "اردو (Urdu)": "ur",
    "Español (Spanish)": "es",
    "Français (French)": "fr",
    "Deutsch (German)": "de",
    "Italiano (Italian)": "it",
    "Português (Portuguese)": "pt",
    "Русский (Russian)": "ru",
    "Nederlands (Dutch)": "nl",
    "Polski (Polish)": "pl",
    "Svenska (Swedish)": "sv",
    "Türkçe (Turkish)": "tr",
    "العربية (Arabic)": "ar",
    "فارسی (Persian)": "fa",
    "日本語 (Japanese)": "ja",
    "한국어 (Korean)": "ko",
    "简体中文 (Chinese Simplified)": "zh-CN",
    "繁體中文 (Chinese Traditional)": "zh-TW",
    "Tiếng Việt (Vietnamese)": "vi",
    "Bahasa Indonesia": "id",
    "ภาษาไทย (Thai)": "th",
}

# Pre-compiled high-speed dictionaries
TRANSLATIONS = {
    "English": {
        "title": "🛡️ Brand Sentiment & Reputation Monitor",
        "subtitle": (
            "Data Science Core: VADER & BERT NLP | Data Analytics Core: Share"
            " of Voice & Live PR Crisis Engine"
        ),
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
    "ಕೊಂಕಣಿ / कोंकणी (Konkani)": {
        "title": "🛡️ ಬ್ರ್ಯಾಂಡ್ ಸೆಂಟಿಮೆಂಟ್ ಆನಿ ಕೀರ್ತ್ ಮಾನಿಟರ್",
        "subtitle": (
            "ಡೇಟಾ ಸೈನ್ಸ್: VADER & BERT NLP | ಅನಾಲಿಟಿಕ್ಸ್: ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ ಆನಿ ಪಿಆರ್"
            " ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್"
        ),
        "settings_header": "⚙️ ಸೆಟ್ಟಿಂಗ್ಸ್ ಆನಿ ಡಾಟಾ ಸೋರ್ಸ್",
        "lexicon_expander": "✏️ ಕಸ್ಟಮ್ ಲೆಕ್ಸಿಕಾನ್ ಸಂಪಾದಕ್",
        "lexicon_desc": (
            "ಕಸ್ಟಮ್ ಸಬ್ದಾಂಚೆಂ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್ ಬದ್ಲಾ (-4.0 ಥಾವ್ನ್ +4.0)"
        ),
        "word_label": "ಸಬ್ದ್ (ಉದಾ: 'ದೋಶ್', 'ಬರೆಂ')",
        "polarity_label": "ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್",
        "apply_lexicon": "VADER ಲೆಕ್ಸಿಕಾನಾಕ್ ಜೊಡಾ",
        "added_lexicon": "ಲೆಕ್ಸಿಕಾನಾಕ್ ಜೊಡ್ಲೆಂ",
        "enable_bert": "BERT NLP ತುಲನಾ ಸಕ್ರಿಯ್ ಕರಾ",
        "mode_label": "ಮೋಡ್ ವಿಂಚಾ",
        "mode_live": "🌐 ಲೈವ್ ನೈಜ್ ಖಬ್ರೊ",
        "mode_demo": "🧪 ಸಿಮ್ಯುಲೇಟೆಡ್ ಡೆಮೊ (ಕ್ರೈಸಿಸ್ ಸಹಿತ್)",
        "search_brands": "ಬ್ರ್ಯಾಂಡ್ ಸೊದಾ",
        "target_brand": "ಮುಖೆಲ್ ಬ್ರ್ಯಾಂಡ್",
        "competitors": "ಸ್ಪರ್ಧಿ ಬ್ರ್ಯಾಂಡಾಂ (ವಿರಾಮಚಿಹ್ನೆ ಘಾಲಾ)",
        "z_slider": "ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್ ಸೂಕ್ಷ್ಮತಾ (Z-Score)",
        "fetching_msg": "ಖಬ್ರೊ ಆನಿ ಸೆಂಟಿಮೆಂಟ್ ಲೋಡ್ ಜಾತಾಸಾ...",
        "no_articles": "'{brand}' ಖಾತಿರ್ ಖಬ್ರೊ ಮೆಳುಂಕ್ ನಾ.",
        "crisis_alert_title": "🚨 {brand} ಖಾತಿರ್ ಪಿಆರ್ ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್!",
        "time_window": "ಅಸಂಗತತಾ ವೇಳ್ ವಿಂಡೋ",
        "z_anomaly": "Z-ಸ್ಕೋರ್ ಅಸಂಗತತಾ",
        "neg_volume": "ನಕಾರಾತ್ಮಕ್ ಪ್ರಮಾಣ್",
        "articles": "ಖಬ್ರೊ",
        "nss": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್ (NSS)",
        "normal_status": (
            "✅ ಸಾಧಾರಣ್ ಸ್ಥಿತಿ: {brand} ಖಾತಿರ್ ಕಸಲೆಂಯಿ ನಕಾರಾತ್ಮಕ್ ಸಂಕಷ್ಟ್ ನಾ."
        ),
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
        "table_section": "3. ಉಲ್ಲೇಖ್ ಎಕ್ಸ್‌ಪ್ಲೋರರ್ ಆನಿ ಮಾಡೆಲ್ ತುಲನಾ",
        "download_btn": "📥 {brand} ಡೇಟಾ CSV ಜಾವ್ನ್ ಡೌನ್‌ಲೋಡ್ ಕರಾ",
    },
    "ಕನ್ನಡ (Kannada)": {
        "title": "🛡️ ಬ್ರ್ಯಾಂಡ್ ಸೆಂಟಿಮೆಂಟ್ ಮತ್ತು ಖ್ಯಾತಿ ಮಾನಿಟರ್",
        "subtitle": (
            "ಡೇಟಾ ಸೈನ್ಸ್: VADER ಮತ್ತು BERT NLP | ಅನಾಲಿಟಿಕ್ಸ್: ಶೇರ್ ಆಫ್ ವಾಯ್ಸ್ ಮತ್ತು"
            " ಪಿಆರ್ ಕ್ರೈಸಿಸ್ ಅಲರ್ಟ್"
        ),
        "settings_header": "⚙️ ಸೆಟ್ಟಿಂಗ್‌ಗಳು ಮತ್ತು ಡೇಟಾ ಮೂಲ",
        "lexicon_expander": "✏️ ಕಸ್ಟಮ್ ಲೆಕ್ಸಿಕಾನ್ ಎಡಿಟರ್",
        "lexicon_desc": "ಕಸ್ಟಮ್ ಪದಗಳ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್ ಹೊಂದಿಸಿ (-4.0 ರಿಂದ +4.0)",
        "word_label": "ಪದ (ಉದಾ: 'ದೋಷ', 'ಅತ್ಯುತ್ತಮ')",
        "polarity_label": "ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್",
        "apply_lexicon": "VADER ಲೆಕ್ಸಿಕಾನ್‌ಗೆ ಸೇರಿಸಿ",
        "added_lexicon": "ಲೆಕ್ಸಿಕಾನ್‌ಗೆ ಸೇರಿಸಲಾಗಿದೆ",
        "enable_bert": "BERT NLP ಹೋಲಿಕೆಯನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಿ",
        "mode_label": "ಮೋಡ್ ಆಯ್ಕೆಮಾಡಿ",
        "mode_live": "🌐 ಲೈವ್ ನೈಜ ಸುದ್ದಿ",
        "mode_demo": "🧪 ಸಿಮ್ಯುಲೇಟೆಡ್ ಡೆಮೊ (ಬಿಕ್ಕಟ್ಟಿನೊಂದಿಗೆ)",
        "search_brands": "ಬ್ರ್ಯಾಂಡ್‌ಗಳನ್ನು ಹುಡುಕಿ",
        "target_brand": "ಗುರಿ ಬ್ರ್ಯಾಂಡ್",
        "competitors": "ಸ್ಪರ್ಧಿಗಳು (ವಿರಾಮಚಿಹ್ನೆಯಿಂದ ಬೇರ್ಪಡಿಸಿ)",
        "z_slider": "ಬಿಕ್ಕಟ್ಟು ಎಚ್ಚರಿಕೆ ಸೂಕ್ಷ್ಮತೆ (Z-Score)",
        "fetching_msg": "ಲೈವ್ ಲೇಖನಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ...",
        "no_articles": "'{brand}' ಗಾಗಿ ಯಾವುದೇ ಲೇಖನಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
        "crisis_alert_title": "🚨 {brand} ಗಾಗಿ ಪಿಆರ್ ಬಿಕ್ಕಟ್ಟು ಎಚ್ಚರಿಕೆ!",
        "time_window": "ಅಸಂಗತತೆಯ ಸಮಯ",
        "z_anomaly": "Z-ಸ್ಕೋರ್ ಅಸಂಗತತೆ",
        "neg_volume": "ಋಣಾತ್ಮಕ ಪ್ರಮಾಣ",
        "articles": "ಲೇಖನಗಳು",
        "nss": "ನೆಟ್ ಸೆಂಟಿಮೆಂಟ್ ಸ್ಕೋರ್ (NSS)",
        "normal_status": (
            "✅ ಸಾಮಾನ್ಯ ಸ್ಥಿತಿ: {brand} ಗಾಗಿ ಯಾವುದೇ ನಕಾರಾತ್ಮಕ ಬಿಕ್ಕಟ್ಟು ಕಂಡುಬಂದಿಲ್ಲ."
        ),
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
        "table_section": "3. ಉಲ್ಲೇಖಗಳ ಎಕ್ಸ್‌ಪ್ಲೋರರ್ ಮತ್ತು ಮಾದರಿ ಹೋಲಿಕೆ",
        "download_btn": "📥 {brand} ಡೇಟಾವನ್ನು CSV ಆಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
    },
    "हिन्दी (Hindi)": {
        "title": "🛡️ ब्रांड सेंटिमेंट और प्रतिष्ठा मॉनिटर",
        "subtitle": (
            "डेटा साइंस: VADER और BERT NLP | एनालिटिक्स: शेयर ऑफ वॉयस और पीआर"
            " क्राइसिस अलर्ट"
        ),
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
        "fetching_msg": "लाइव समाचार लोड हो रहा है...",
        "no_articles": "'{brand}' के लिए कोई लेख नहीं मिला।",
        "crisis_alert_title": "🚨 {brand} के लिए पीआर क्राइसिस अलर्ट!",
        "time_window": "विसंगति समय",
        "z_anomaly": "Z-स्कोर विसंगति",
        "neg_volume": "नकारात्मक मात्रा",
        "articles": "लेख",
        "nss": "नेट सेंटिमेंट स्कोर (NSS)",
        "normal_status": (
            "✅ सामान्य स्थिति: {brand} के लिए कोई नकारात्मक संकट नहीं पाया"
            " गया।"
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
        "no_keywords": "कोई नकारात्मक कीवर्ड नहीं मिला।",
        "table_section": "3. उल्लेख एक्सप्लोरर और मॉडल तुलना",
        "download_btn": "📥 {brand} डेटा को CSV के रूप में डाउनलोड करें",
    },
}


# ---------------- DYNAMIC AUTO-TRANSLATION ENGINE ----------------
@st.cache_data(show_spinner=False)
def translate_dynamic(text: str, target_lang_code: str) -> str:
  """Translates text with persistent in-memory caching."""
  if target_lang_code == "en" or not text:
    return text
  try:
    encoded = urllib.parse.quote(text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang_code}&dt=t&q={encoded}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=3.0) as resp:
      res_json = json.loads(resp.read().decode("utf-8"))
      return "".join([part[0] for part in res_json[0] if part[0]])
  except Exception:
    return text


def get_translation(key: str, lang_name: str, **kwargs) -> str:
  """Fetches translation from pre-compiled dictionary or dynamic cache."""
  lang_code = LANGUAGE_CODES.get(lang_name, "en")

  if lang_name in TRANSLATIONS and key in TRANSLATIONS[lang_name]:
    text = TRANSLATIONS[lang_name][key]
  else:
    base_text = TRANSLATIONS["English"].get(key, key)
    text = translate_dynamic(base_text, lang_code)

  if kwargs:
    try:
      text = text.format(**kwargs)
    except Exception:
      pass
  return text