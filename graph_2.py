from langgraph.graph import StateGraph, END

def add(state: int):
    return state + 1 

def multiply(state: int):
    return state * 2

def should_continue(state: int):
    return END if state > 10 else 'multiply'

workflow = StateGraph(int)
workflow.add_node(add)
workflow.add_node(multiply)

workflow.add_edge('multiply', 'add')
workflow.add_conditional_edges('add', should_continue)

workflow.set_entry_point('multiply')

app = workflow.compile()

state = app.invoke(1)
print(state)