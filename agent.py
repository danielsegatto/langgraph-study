import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# Define a lógica do Chatbot
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile()

# Bloco opcional para testar via terminal no Codespace
if __name__ == "__main__":
    user_input = "Olá, quem é você?"
    events = app.stream({"messages": [("user", user_input)]})
    for event in events:
        for value in event.values():
            print("IA:", value["messages"][-1].content)