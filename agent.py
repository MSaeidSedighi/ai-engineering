import dotenv
import langchain_openai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

dotenv.load_dotenv()

class Agent:
    def __init__(self):
        self.llm = langchain_openai.ChatOpenAI(
                        base_url="https://api.gapgpt.app/v1",
                        model_name="gpt-5.6-terra",
                        api_key=dotenv.get_key(dotenv.find_dotenv(), "API_KEY"),
                        temperature=0.7,
                    )
        self.messages = [
            SystemMessage(content="Give Short Answers."),
            ]

    def chat_loop(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'q':
                print("Exiting chat.")
                break

            self.messages.append(HumanMessage(content=user_input))
            response = self.llm.invoke(self.messages)
            self.messages.append(AIMessage(content=response.content))

            print(f"AI: {response.content}")

def main():
    agent = Agent()
    agent.chat_loop()

if __name__ == "__main__":
    main()