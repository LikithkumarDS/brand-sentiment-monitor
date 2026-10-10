from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

def classify_mention(text: str) -> dict:
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    # Custom thresholds for brand monitoring (widened neutral zone)
    if compound >= 0.10:
        sentiment = "positive"
    elif compound <= -0.10:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        "text": text,
        "sentiment": sentiment,
        "compound_score": compound,
        "scores": scores
    }

# Example brand mentions
mentions = [
    "Love the new update from @Brand! Super fast and reliable. 😍",
    "Customer service was completely unhelpful and rude.",
    "Ordered the product yesterday, tracking number received.",
    "The battery life is NOT great... expected better.",
    "This tool is INSANELY GOOD!!!"
]

for mention in mentions:
    result = classify_mention(mention)
    print(f"[{result['sentiment'].upper()}] (score: {result['compound_score']:+.2f}) - {result['text']}")