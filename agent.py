import dotenv
import langchain_openai
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent

dotenv.load_dotenv()

def get_weather(city):
    '''
    get weather information of a city.
    Args:
        city (str): The name of the city to get weather information for.
    Returns:
        dict: A dictionary containing the weather information for the specified city.
    '''

    return {'city': city, 'temperature': '25°C', 'condition': 'Sunny'}

class Agent:
    def __init__(self):
        self.llm = langchain_openai.ChatOpenAI(
                        base_url="https://api.gapgpt.app/v1",
                        model_name="gpt-5.6-terra",
                        api_key=dotenv.get_key(dotenv.find_dotenv(), "API_KEY"),
                        temperature=0.7,
                    )
        self.messages = [
            SystemMessage("Answer the given questions, and use provided tools whenever needed."),
            ]

        self.agent = create_agent(model=self.llm, tools=[get_weather])

    def chat_loop(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'q':
                print("Exiting chat.")
                break

            self.messages.append(HumanMessage(user_input))
            response = self.agent.invoke({"messages": self.messages})
            self.messages.append(AIMessage(response['messages'][-1].content))

            print(response['messages'][-1].content)

def main():
    agent = Agent()
    agent.chat_loop()

if __name__ == "__main__":
    main()