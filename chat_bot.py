from time import sleep

import dotenv
import os
from openai import OpenAI
import tiktoken

MAX_PROMPT_TOKENS = 100
MAX_OUTPUT_TOKENS = 2000
TEMPERATURE = 0.7
TOKEN_PRINT_INTERVAL = 0.1

dotenv.load_dotenv()

class ChatBot:
    def __init__(self, system_prompt):
        self.llm = OpenAI(
                    api_key=os.getenv("API_KEY"),
                    base_url="https://api.gapgpt.app/v1"
                    )
        self.system_prompt = system_prompt

    def callModel(self, message, memory=[]):
        response = self.llm.chat.completions.create(
            model="gpt-5.6-terra",
            messages=memory + [{"role": "user", "content": message}],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True
        )

        response_text = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    response_text += delta
                    sleep(TOKEN_PRINT_INTERVAL)

            if chunk.usage:
                token_count = chunk.usage.total_tokens

        print()

        return response_text, token_count


    def chat(self):
        encoding = tiktoken.encoding_for_model("gpt-5.6-terra")

        memory = [{'role': 'system', 'content': self.system_prompt}]
        print("Welcome to the chat bot! Type 'q' to terminate the chat.")
        total_tokens = 0
        while True:
            user_input = input("Enter your message: ").strip().lower()

            if len(encoding.encode(user_input)) > MAX_PROMPT_TOKENS:
                print("Your message is too long. Try again with a shorter message.")
                continue

            if user_input == "q":
                print(f"Total tokens used: {total_tokens}")
                print("Terminating the chat bot. Goodbye!")
                break

            bot_response, token_count = self.callModel(user_input, memory)
            # print(bot_response)
            total_tokens += token_count
            memory.append({"role": "user", "content": user_input})
            memory.append({"role": "assistant", "content": bot_response})

def main():
    system_prompt = "You are a helpful assistant. Your name is Abbas. Give short and concise answers. If you don't know the answer, say 'I don't know'."
    chat_bot = ChatBot(system_prompt)
    chat_bot.chat()

if __name__ == "__main__":
    main()

