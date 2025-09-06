from flask import Flask, request, jsonify
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
from deep_translator import GoogleTranslator
import re

app = Flask(__name__)

# Load the RoBERTa model for sentiment analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Create a sentiment-analysis pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Custom Pangasinan dictionary for common words and phrases with their sentiment values
# This dictionary should be expanded with more Pangasinan words and phrases
pangasinan_sentiment_dict = {
    # Positive words/phrases
    "mabli": "positive",  # dear, precious
    "maong": "positive",  # good
    "masanten": "positive",  # beautiful
    "maliket": "positive",  # happy
    "salamat": "positive",  # thank you
    "makapaliket": "positive",  # pleasing
    "mabulos": "positive",  # nice, pleasant
    "magayaga": "positive",  # joyful
    "masanting":"postive",
    "marakep":"postive",
    "maples":"postive",
    "masantos":"postive",


    # Negative words/phrases
    "mauges": "negative",  # bad
    "masakit": "negative",  # painful
    "onsot": "negative",  # angry
    "amta la": "negative",  # disappointing
    "maermen": "negative",  # sad
    "mabayag": "negative",  # slow, taking long time
    "anggapo": "negative",  # nothing
    "mainomay": "negative",  # difficult
    "makapabwesit":"negative",

    # Neutral words/phrases
    "sankaili": "neutral",  # stranger
    "onla": "neutral",  # come
    "mansiansia": "neutral",  # stay
    "mankakasi": "neutral",  # perhaps
}

# Common Pangasinan expressions with their sentiment values
pangasinan_expressions = {
    "anggapo so nakala": "negative",  # "nothing found/gained"
    "masakit so ulok": "negative",  # "my head hurts"
    "maong ya agew": "positive",  # "good day"
    "mabayag so pila": "negative",  # "long queue/line"
    "masanten ya bulan": "positive",  # "beautiful moon"
    "masakbay ka la": "neutral",  # "you're early"
    "maong so ginawam": "positive",  # "you did well"
    "maermen ak": "negative",  # "I am sad"
    "maliket ak": "positive",  # "I am happy"
    "salamat na dakel": "positive",  # "thank you very much"
    "makapabwesit so office u":"negative",
    "masantos na kabwasan sikayo amin":"Positive"
}


def check_pangasinan_sentiment(text):
    """Check for Pangasinan words and expressions to determine sentiment."""
    text_lower = text.lower()

    # Check for expressions first (longer phrases take precedence)
    for expression, sentiment in pangasinan_expressions.items():
        if expression.lower() in text_lower:
            return sentiment, True

    # Check for individual words
    for word, sentiment in pangasinan_sentiment_dict.items():
        # Use word boundary for more accurate matching
        if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
            return sentiment, True

    return None, False


def detect_and_translate(text):
    """Translate text to English with special handling for Pangasinan."""
    # First check if text contains Pangasinan-specific indicators
    pangasinan_sentiment, is_pangasinan = check_pangasinan_sentiment(text)

    try:
        # Translate text to English
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text, pangasinan_sentiment, is_pangasinan
    except Exception as e:
        # If translation fails, return original text
        return text, pangasinan_sentiment, is_pangasinan


def custom_sentiment_logic(translated_text):
    """Apply additional custom sentiment rules to translated text."""
    text_lower = translated_text.lower()

    # Custom rules for common expressions in English translation
    if any(phrase in text_lower for phrase in ["long queue", "long wait", "waiting for hours"]):
        return "negative"
    if any(phrase in text_lower for phrase in ["excellent service", "wonderful experience"]):
        return "positive"

    return None  # No custom rule applied


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()  # Get the JSON data from the request
        text = data.get("text", "")  # Extract the "text" field

        # First check if the text contains Pangasinan words/phrases
        translated_text, pangasinan_sentiment, is_pangasinan = detect_and_translate(text)

        # If Pangasinan words were detected and sentiment determined, prioritize that
        if pangasinan_sentiment:
            sentiment = pangasinan_sentiment
        else:
            # Apply custom sentiment logic to translated text
            custom_sentiment = custom_sentiment_logic(translated_text)
            if custom_sentiment:
                sentiment = custom_sentiment
            else:
                # Perform sentiment analysis using the model
                result = sentiment_pipeline(translated_text)
                sentiment_label = result[0]['label'].lower()

                # Map the model's labels to sentiment values
                sentiment_map = {
                    "label_0": "negative",
                    "label_1": "neutral",
                    "label_2": "positive",
                }
                sentiment = sentiment_map.get(sentiment_label, "neutral")

        response = {
            "Feedback":text,
            "Sentiment Result": sentiment,
            "translated_text": translated_text,
            "contains_pangasinan": is_pangasinan
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)