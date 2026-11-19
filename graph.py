from typing import TypedDict
from langgraph.graph import START, StateGraph
import dotenv

dotenv.load_dotenv()

class Essay(TypedDict):
    topic: str
    content: str | None
    score: float | None

def write_essay(essay: Essay):
    return {"content": f"the essay is about {essay['topic']}"}

def score_essay(essay: Essay):
    return {"score": 10}

builder = StateGraph(Essay)
builder.add_node(write_essay)
builder.add_node(score_essay)

builder.add_edge(START, 'write_essay')
builder.add_edge('write_essay', 'score_essay')

graph = builder.compile()

state = graph.invoke({'topic': 'multi-agent automated penetration testing', 'content': None, 'score': None})
print(state)
