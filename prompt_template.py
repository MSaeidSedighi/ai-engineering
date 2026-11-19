import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        history = [
            HumanMessage('Hello, how are you?'),
            AIMessage('>>Bonjour, comment allez-vous ?<<'),
            HumanMessage('I love you!'),
            AIMessage('>>Je t\'aime !<<'),
        ]
        prompt = ChatPromptTemplate.from_messages(
           [
               ('system', 'You are a translator from {origin} to {target}.'),
               MessagesPlaceholder(variable_name='history_variable'),
               ('human', '{input}'),
           ]
        )

        chain = prompt | self.llm

        response = chain.invoke({
            'origin': 'English',
            'target': 'French',
            'input': 'I hate you!',
            'history_variable': history
        })

        print(response.content)

    



def main():
    agent = Agent()
    agent.chat()

if __name__ == "__main__":
    main()