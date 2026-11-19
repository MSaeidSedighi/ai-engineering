import dotenv
import langchain_openai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()

class Agent:
    def __init__(self):
        self.llm = langchain_openai.ChatOpenAI(
                        base_url="https://api.gapgpt.app/v1",
                        model_name="gpt-5.6-terra",
                        api_key=dotenv.get_key(dotenv.find_dotenv(), "API_KEY"),
                        temperature=0.7,
                    )
        self.agent = create_agent(model=self.llm, tools=[], system_prompt="Answer questions concisely.", checkpointer=InMemorySaver())

    def chat_loop(self):
        active_session = 'session_1'
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'q':
                print("Exiting chat.")
                break

            if user_input.lower() == 'ch':
                print("Session switched.")
                active_session = 'session_2' if active_session == 'session_1' else 'session_1'
                continue

            config = {'configurable': {'thread_id': active_session}}
            response = self.agent.invoke({"messages": HumanMessage(user_input)}, config)

            print(response['messages'][-1].content)

def main():
    agent = Agent()
    agent.chat_loop()

if __name__ == "__main__":
    main()
