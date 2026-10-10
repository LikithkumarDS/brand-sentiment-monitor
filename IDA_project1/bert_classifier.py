import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class BertSentimentAnalyzer:

  def __init__(self):
    # Specialized social media sentiment model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    self.pipe = pipeline(
        "sentiment-analysis",
        model=self.model,
        tokenizer=self.tokenizer,
        return_all_scores=False,
    )

  def predict(self, text: str) -> dict:
    result = self.pipe(text[:512])[0]
    # Maps 'positive', 'neutral', 'negative'
    label = result["label"].lower()
    score = result["score"]
    return {"bert_label": label, "bert_confidence": round(score, 3)}