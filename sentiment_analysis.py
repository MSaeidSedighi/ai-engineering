import dotenv
import os
from openai import OpenAI
from transformers import pipeline

dotenv.load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

    def analyze_sentiment(self, text):
        labels = ["positive", "negative", "neutral" ]
        result = self.classifier(text, candidate_labels=labels)
        return result      

def main():
    analyzer = SentimentAnalyzer()

    while True:
        user_input = input("Enter text to analyze sentiment (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        sentiment = analyzer.analyze_sentiment(user_input)
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()