import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

class SentimentAnalyzer:
    def __init__(self, system_prompt):
        self.llm = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://api.gapgpt.app/v1"
        )
        self.system_prompt = system_prompt
        self.messages = [{'role': 'system', 'content': self.system_prompt}]

    def analyze_sentiment(self, text):
        self.messages.append({'role': 'user', 'content': text})
        response = self.llm.chat.completions.create(
            model="gpt-5.6-terra",
            messages=self.messages,
            temperature=0.7,
            max_tokens=100
        )
        sentiment = response.choices[0].message.content.strip()
        return sentiment

def main():
    analyzer = SentimentAnalyzer(system_prompt="You are a sentiment analysis model. Classify the sentiment of the given text as positive, negative, or neutral.")

    sentiment = analyzer.analyze_sentiment("I love this product! It's amazing.")

    print(sentiment)

if __name__ == "__main__":
    main()