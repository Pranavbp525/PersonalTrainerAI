from typing import Annotated, List, Dict, Any, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from experiments.pranav.chatbot.agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools([retrieve_from_rag, 
                                 tool_fetch_workouts,
                                 tool_get_workout_count,
                                 tool_fetch_routines,
                                 tool_update_routine,
                                 tool_create_routine,
                                 retrieve_from_rag])

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_profile: Optional[dict]
    workout_plan: Optional[dict]
    progress_data: Optional[dict]
    stage: Literal["assessment", "planning", "monitoring"]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Assessment stage - handles user profiling and goal setting
def assessment_stage(state: AgentState) -> AgentState:
    assessment_prompt = """You are a fitness expert in the assessment phase.
    Ask targeted questions about the user's:
    1. Fitness goals (strength, endurance, weight loss, etc.)
    2. Training history and current fitness level
    3. Available equipment and schedule
    4. Any health concerns or limitations
    
    Use the information to build a comprehensive user profile.
    """
    
    messages = state["messages"] + [SystemMessage(content=assessment_prompt)]
    response = llm.invoke(messages)
    
    # Extract profile information from conversation if possible
    # (code for profile extraction would go here)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile", {}),
        "stage": "planning" if state.get("user_profile") else "assessment"
    }

# Planning stage - creates workout routines
def planning_stage(state: AgentState) -> AgentState:
    planning_prompt = """You are a fitness expert in the planning phase.
    Use the user profile and retrieve_from_rag to create a personalized workout plan.
    Include:
    1. Weekly schedule breakdown
    2. Specific exercises with sets, reps, and rest periods
    3. Progression scheme over time
    4. Format the plan for Hevy app integration
    
    Explain the scientific rationale behind each element of the plan.
    """
    
    messages = state["messages"] + [SystemMessage(content=planning_prompt)]
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": {"content": response.content, "created": "now"},
        "stage": "monitoring"
    }

# Monitoring stage - tracks progress and adjusts plans
def monitoring_stage(state: AgentState) -> AgentState:
    monitoring_prompt = """You are a fitness expert in the monitoring phase.
    Analyze the user's workout logs from the Hevy API and:
    1. Track adherence to the workout plan
    2. Identify progress or plateaus
    3. Suggest adjustments to the plan as needed
    4. Provide motivation and science-based advice
    
    Use tool_fetch_workouts to view recent activity.
    """
    
    messages = state["messages"] + [SystemMessage(content=monitoring_prompt)]
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": state.get("workout_plan"),
        "stage": "monitoring"  # Stay in monitoring once we reach it
    }

# Stage router
def stage_router(state: AgentState) -> str:
    return state["stage"]

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("assessment", assessment_stage)
workflow.add_node("planning", planning_stage)
workflow.add_node("monitoring", monitoring_stage)
workflow.add_node("tools", ToolNode([retrieve_from_rag, tool_fetch_workouts]))

# Add edges
workflow.add_conditional_edges(
    "assessment",
    stage_router,
    {
        "planning": "planning"
    }
)
workflow.add_conditional_edges(
    "planning",
    stage_router,
    {
        "monitoring": "monitoring"
    }
)
workflow.add_conditional_edges(
    "monitoring",
    stage_router,
    {
        "planning": "planning",  # Allow returning to planning for major updates
        "end": END  # Allow ending the conversation
    }
)

# Add tool usage edges
for node in ["assessment", "planning", "monitoring"]:
    workflow.add_conditional_edges(
        node,
        tools_condition,
        {
            True: "tools",
            False: node
        }
    )

workflow.add_conditional_edges("tools",
                               lambda state: state['stage'],
                               {
                                   "assessment":"assessment",
                                   "planning":"planning",
                                   "monitoring":"monitoring"
                               })

workflow.set_entry_point("assessment")

stage_based_agent = workflow.compile()
