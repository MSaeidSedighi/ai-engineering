import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

class Agent:
    def __init__(self):
        self.llm = ChatOpenAI(
                        base_url="https://api.gapgpt.app/v1",
                        model_name="gpt-5.6-terra",
                        api_key=dotenv.get_key(dotenv.find_dotenv(), "API_KEY"),
                        temperature=0.7,
                    )

    def chat(self):
        prompt = ChatPromptTemplate.from_messages(
           [
               ('system', 'You are a translator from {origin} to {target}.'),
               ('human', '{input}'),
           ]
        )

        chain = prompt | self.llm

        response = chain.invoke({
            'origin': 'English',
            'target': 'French',
            'input': 'Hello, how are you?'
        })

        print(response)

    



def main():
    agent = Agent()
    agent.chat()

if __name__ == "__main__":
    main()