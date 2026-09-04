import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

llm = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="http://localhost:20128/v1"
)

def bot(message, memory=[]):
    response = llm.chat.completions.create(
        model="ai-engineering",
        messages=memory + [{"role": "user", "content": message}],
        temperature=0.7,
        max_tokens=2000
    )

    try: 
        return response.choices[0].message.content
    except:
        print("bot message failed. retrying...")
        bot(message, memory)

def chat():
    memory = []
    print("Welcome to the chat bot! Type 'q' to terminate the chat.")
    while True:
        user_input = input("Enter your message: ").strip().lower()

        if user_input == "q":
            print("Terminating the chat bot. Goodbye!")
            break

        bot_response = bot(user_input, memory)
        print(bot_response)
        memory.append({"role": "user", "content": user_input})
        memory.append({"role": "assistant", "content": bot_response})

def main():
    chat()

if __name__ == "__main__":
    main()

