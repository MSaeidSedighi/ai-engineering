from typing import TypedDict
from langgraph.graph import START, StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

class Rate(BaseModel):
    score: int = Field(description="score given to paragraph based on coherence, relativeness to topic, structure and ...")
    feedback: str = Field(description="feedback and comment to improve the paragraph")

class Draft(BaseModel):
    draft: str = Field(description="draft about the topic considering feedback if present")

class Essay(TypedDict):
    topic: str
    draft: str
    score: int
    feedback: str
    revision_count: int

llm = ChatOpenAI(
        base_url="https://api.gapgpt.app/v1",
        model_name="gpt-5.6-terra",
        api_key=dotenv.get_key(dotenv.find_dotenv(), "API_KEY"),
        temperature=0.6,
    )

rater = llm.with_structured_output(Rate)
writer = llm.with_structured_output(Draft)

def write_draft(state: Essay):
    if state['feedback'] != '':
        print(f"\n-----------\ngot score {state['score']} with feedback: {state['feedback']}\n-----------\n")
    prompt = f'''draft this paragraph with the following topic and feedback:
        Topic: {state["topic"]}
        Paragraph so far: {state["draft"]}
        Feedback (can be empty): {state["feedback"]}
        '''
    response = writer.invoke(prompt)

    return {'draft': response.draft, 'revision_count': state['revision_count'] + 1}

def review_draft(state: Essay):
    prompt = f'''Rate this paragraph from 0-10 and give one sentence of feedback. pay attention to the topic make sure paragraph matches it. (try critisizing and do not accept paragraphs easily)
    Topic: {state["topic"]}
    Paragraph: {state["draft"]}'''
    response = rater.invoke(prompt)

    return {'score': response.score, 'feedback': response.feedback}

def should_revise(state: Essay):
    return END if state['score'] > 8 or state['revision_count'] >= 3 else 'write_draft'
        

builder = StateGraph(Essay)

builder.add_node(write_draft)
builder.add_node(review_draft)

builder.add_edge(START, 'write_draft')
builder.add_edge('write_draft', 'review_draft')
builder.add_conditional_edges('review_draft', should_revise)

graph = builder.compile()

result = graph.invoke(
    {
        'topic': 'multi-agent automated penetration testing',
        'draft': '',
        'score': 0,
        'feedback': '',
        'revision_count': 0,
    }
)

print(f"FINAL DRAFT (revisons: {result['revision_count']}) SCORE {result['score']}: \n---------------------\n{result['draft']}")

