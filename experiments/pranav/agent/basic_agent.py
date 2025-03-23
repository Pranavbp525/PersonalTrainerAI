from typing import Annotated, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
from experiments.pranav.chatbot.agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)

# Load environment variables and initialize services
load_dotenv()

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools([retrieve_from_rag, 
                                 tool_fetch_workouts,
                                 tool_get_workout_count,
                                 tool_fetch_routines,
                                 tool_update_routine,
                                 tool_create_routine,
                                 retrieve_from_rag])

# Create the system prompt with detailed instructions
system_prompt = """You are an expert personal fitness trainer with deep knowledge of exercise science.

Your capabilities:
1. Assessment: You can ask relevant questions to understand the user's fitness level, goals, and constraints
2. Knowledge: You have access to exercise science information through the retrieve_from_rag tool
3. Workout Planning: You can create personalized routines using scientific principles
4. Progress Tracking: You can analyze workout logs from Hevy using the tool_fetch_workouts tool
5. Plan Adjustment: You can modify workouts based on progress and feedback

When working with a new user:
- Ask about their fitness goals, experience level, available equipment, and schedule
- Be conversational but thorough in your assessment
- Use retrieve_from_rag to get scientific information about appropriate exercises
- Create a personalized workout plan
- Use tool_create_routine to save the workout to their Hevy account

For returning users:
- Check their progress using tool_fetch_workouts and tool_get_workout_count
- Analyze their adherence and performance
- Suggest modifications to their routine as needed
- Use tool_update_routine to modify existing routines

Always provide scientific rationale for your recommendations and maintain a supportive, motivational tone.
"""

# Define the main agent node
def fitness_agent(state: AgentState) -> AgentState:
    """The main fitness trainer agent that handles all aspects of the interaction."""
    # Add the system prompt for first-time messages
    if len(state["messages"]) == 1 and isinstance(state["messages"][0], HumanMessage):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
    else:
        messages = state["messages"]
        
    # Invoke the model with tools
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else None
    }

# Define the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("fitness_agent", fitness_agent)
graph.add_node("tools", ToolNode([
    retrieve_from_rag, 
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
]))

# Add edges
graph.add_conditional_edges(
    "fitness_agent",
    tools_condition,
    {
        True: "tools",
        False: END
    }
)
graph.add_edge("tools", "fitness_agent")

# Set entry point
graph.set_entry_point("fitness_agent")

# Compile the graph
app = graph.compile()
