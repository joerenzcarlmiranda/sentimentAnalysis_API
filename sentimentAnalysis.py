from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, pipeline
from deep_translator import GoogleTranslator
import re

app = Flask(__name__)

# Load the DistilBERT model for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Create a sentiment-analysis pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Custom Pangasinan dictionary for common words and phrases with their sentiment values
pangasinan_sentiment_dict = {
    # Positive words/phrases
    "mabli": "positive",
    "maong": "positive",
    "masanten": "positive",
    "maliket": "positive",
    "salamat": "positive",
    "makapaliket": "positive",
    "mabulos": "positive",
    "magayaga": "positive",
    "masanting": "positive",
    "marakep": "positive",
    "maples": "positive",
    "masantos": "positive",

    # Negative words/phrases
    "mauges": "negative",
    "masakit": "negative",
    "onsot": "negative",
    "amta la": "negative",
    "maermen": "negative",
    "mabayag": "negative",
    "anggapo": "negative",
    "mainomay": "negative",
    "makapabwesit": "negative",

    # Neutral words/phrases
    "sankaili": "neutral",
    "onla": "neutral",
    "mansiansia": "neutral",
    "mankakasi": "neutral",
}

# Common Pangasinan expressions with their sentiment values
pangasinan_expressions = {
    "anggapo so nakala": "negative",
    "masakit so ulok": "negative",
    "maong ya agew": "positive",
    "mabayag so pila": "negative",
    "masanten ya bulan": "positive",
    "masakbay ka la": "neutral",
    "maong so ginawam": "positive",
    "maermen ak": "negative",
    "maliket ak": "positive",
    "salamat na dakel": "positive",
    "makapabwesit so office u": "negative",
    "masantos na kabwasan sikayo amin": "positive",
}


def check_pangasinan_sentiment(text):
    """Check for Pangasinan words and expressions to determine sentiment."""
    text_lower = text.lower()

    for expression, sentiment in pangasinan_expressions.items():
        if expression.lower() in text_lower:
            return sentiment, True

    for word, sentiment in pangasinan_sentiment_dict.items():
        if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
            return sentiment, True

    return None, False


def detect_and_translate(text):
    """Translate text to English with special handling for Pangasinan."""
    pangasinan_sentiment, is_pangasinan = check_pangasinan_sentiment(text)

    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text, pangasinan_sentiment, is_pangasinan
    except Exception:
        return text, pangasinan_sentiment, is_pangasinan


def custom_sentiment_logic(translated_text):
    """Apply additional custom sentiment rules to translated text."""
    text_lower = translated_text.lower()

    if any(phrase in text_lower for phrase in ["long queue", "long wait", "waiting for hours"]):
        return "negative"
    if any(phrase in text_lower for phrase in ["excellent service", "wonderful experience"]):
        return "positive"

    return None


def classify_sentiment(translated_text):
    """Classify sentiment using DistilBERT with neutral thresholding."""
    results = sentiment_pipeline(translated_text)[0]

    # Extract probabilities
    positive_score = next(item["score"] for item in results if item["label"] == "POSITIVE")
    negative_score = next(item["score"] for item in results if item["label"] == "NEGATIVE")

    # Neutral threshold
    threshold = 0.6
    if max(positive_score, negative_score) < threshold:
        return "neutral"
    elif positive_score > negative_score:
        return "positive"
    else:
        return "negative"


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get("text", "")

        translated_text, pangasinan_sentiment, is_pangasinan = detect_and_translate(text)

        if pangasinan_sentiment:
            sentiment = pangasinan_sentiment
        else:
            custom_sentiment = custom_sentiment_logic(translated_text)
            if custom_sentiment:
                sentiment = custom_sentiment
            else:
                sentiment = classify_sentiment(translated_text)

        response = {
            "Feedback": text,
            "Sentiment Result": sentiment,
            "translated_text": translated_text,
            "contains_pangasinan": is_pangasinan
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
