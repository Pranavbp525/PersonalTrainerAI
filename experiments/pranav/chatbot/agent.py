from typing import TypedDict, Annotated
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from config import config
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]  # List of {"role": str, "content": str}
    session_id: str  # To tie to your DB/Redis
    tool_results: dict  # To store tool outputs


@tool
def calculate_expression(expression: str) -> str:
    """
    Evaluates a simple arithmetic expression.
    
    Args:
        expression (str): A string representing an arithmetic expression (e.g., "2 + 3 * 4").
    
    Returns:
        str: The result of the evaluated expression, or an error message.
    """
    try:
        result = eval(expression)
    except Exception as e:
        result = f"Error evaluating expression: {e}"
    return str(result)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))


def chat_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    state["messages"].append({"role": "assistant", "content": response.content})
    return state

def tool_node(state: AgentState):
    last_message = state["messages"][-1]["content"]
    # Simple logic: assume last message is a tool request like "calculate: X"
    if last_message.startswith("calculate:"):
        expression = last_message.replace("calculate:", "").strip()
        result = calculate_expression(expression)
        state["tool_results"]["calculate_expression"] = result
        state["messages"].append({"role": "assistant", "content": result})
    return state

def decision_node(state: AgentState):
    last_message = state["messages"][-1]["content"]
    # Basic routing: if the message starts with "calculate:" use the tool node; otherwise, continue chatting.
    if last_message.startswith("calculate:"):
        return "tool_node"
    return "chat_node"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("decision_node", decision_node)

workflow.set_entry_point("decision_node")
workflow.add_conditional_edges("decision_node", decision_node, {"chat_node": "chat_node", "tool_node": "tool_node"})
workflow.add_edge("chat_node", END)
workflow.add_edge("tool_node", END)

# Compile the graph
agent_app = workflow.compile()