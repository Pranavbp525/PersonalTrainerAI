import os
from typing import Annotated, List, Dict, Any, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)
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

# Define the agent state with more detailed tracking
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_profile: Optional[dict]
    workout_plan: Optional[dict]
    progress_data: Optional[dict]
    training_history: Optional[List[Dict]]
    analysis_results: Optional[Dict]
    next_action: Literal["orchestrate", "assess", "research", "plan", "analyze", "adjust", "end"]
    context: Dict[str, Any]  # For sharing context between nodes
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Orchestrator - central coordinator that decides workflow
def orchestrator_node(state: AgentState) -> AgentState:
    """Determines which specialized node should handle the current interaction."""
    
    orchestrator_prompt = """You are the coordinator of a fitness training system.
    Based on the conversation history and user needs, determine the next appropriate action:
    
    - "assess": When we need to gather information about the user
    - "research": When we need to retrieve exercise science information
    - "plan": When we need to create a new workout routine
    - "analyze": When we need to analyze workout data and progress
    - "adjust": When we need to modify an existing workout plan
    - "end": When the interaction is complete
    
    Consider the user's profile completeness, workout plan existence, and recent progress.
    """
    
    # Add thinking context for the orchestrator
    context_summary = {
        "has_profile": bool(state.get("user_profile")),
        "has_workout_plan": bool(state.get("workout_plan")),
        "has_progress_data": bool(state.get("progress_data")),
        "training_history_count": len(state.get("training_history", [])),
        "last_action": state.get("next_action", "orchestrate")
    }
    
    messages = [
        SystemMessage(content=orchestrator_prompt),
        HumanMessage(content=f"Context: {context_summary}\n\nDecide the next action based on the conversation history: {state['messages'][-1].content if state['messages'] else 'Initial interaction'}")
    ]
    
    response = llm.invoke(messages)
    
    # Extract the next action from the response
    action_mapping = {
        "assess": "assess",
        "research": "research",
        "plan": "plan", 
        "analyze": "analyze",
        "adjust": "adjust",
        "end": "end"
    }
    
    # Simple parsing (in real system, would be more robust)
    next_action = "orchestrate"  # Default
    for action in action_mapping:
        if action in response.content.lower():
            next_action = action_mapping[action]
            break
    
    # Update context with orchestrator's reasoning
    context = state.get("context", {})
    context["orchestrator_reasoning"] = response.content
    
    return {
        "messages": state["messages"],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": state.get("workout_plan"),
        "progress_data": state.get("progress_data"),
        "training_history": state.get("training_history", []),
        "next_action": next_action,
        "context": context
    }

# Assessment Worker - gathers user information
def assessment_worker(state: AgentState) -> AgentState:
    """Handles user assessment and profile building."""
    
    assessment_prompt = """You are a fitness assessment specialist.
    Ask targeted questions to build a comprehensive user profile.
    Focus on gathering missing information based on what you already know.
    
    Current profile: {{user_profile}}
    
    Areas to assess:
    - Fitness goals and priorities
    - Training experience and current level
    - Available equipment and schedule
    - Physical limitations or health concerns
    - Measurement data (if willing to share)
    
    Be conversational but thorough.
    """
    
    # Customize the prompt with current profile
    filled_prompt = assessment_prompt.replace(
        "{{user_profile}}", 
        str(state.get("user_profile", "No existing profile"))
    )
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # In a real system, we'd have logic to extract profile information
    # and update the user_profile dictionary
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile", {}),
        "workout_plan": state.get("workout_plan"),
        "progress_data": state.get("progress_data"),
        "training_history": state.get("training_history", []),
        "next_action": "orchestrate",  # Return to orchestrator
        "context": state.get("context", {})
    }

# Research Worker - retrieves exercise science information
def research_worker(state: AgentState) -> AgentState:
    """Retrieves relevant exercise science information."""
    
    research_prompt = """You are a fitness research specialist.
    Based on the user's profile and current needs, retrieve relevant exercise science information.
    Use the retrieve_from_rag tool to find scientific information about:
    
    - Optimal training approaches for the user's goals
    - Exercise selection based on equipment and limitations
    - Evidence-based progression schemes
    - Recovery and periodization strategies
    
    Organize your findings in a clear, structured way.
    """
    
    context = state.get("context", {})
    research_needed = context.get("research_topics", ["general training principles"])
    
    # Add research topics to prompt
    research_prompt += f"\n\nSpecific topics to research: {', '.join(research_needed)}"
    
    messages = state["messages"] + [SystemMessage(content=research_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update context with research findings
    context["research_findings"] = response.content
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": state.get("workout_plan"),
        "progress_data": state.get("progress_data"),
        "training_history": state.get("training_history", []),
        "next_action": "orchestrate",
        "context": context
    }

# Planning Worker - creates workout routines
def planning_worker(state: AgentState) -> AgentState:
    """Creates personalized workout plans."""
    
    planning_prompt = """You are a workout programming specialist.
    Create a detailed, personalized workout plan for the user.
    
    User profile: {{user_profile}}
    Research findings: {{research_findings}}
    
    Your plan should include:
    1. Weekly schedule with workout days and focus areas
    2. Detailed workout structure for each session
       - Specific exercises with sets, reps, rest periods
       - Warm-up routines and cool-down stretches
       - Form cues and technique notes
    3. Progression scheme for 4-6 weeks
    4. Structured format ready for Hevy app integration
    
    Explain the scientific rationale behind your choices.
    """
    
    context = state.get("context", {})
    
    # Customize the prompt
    filled_prompt = planning_prompt.replace(
        "{{user_profile}}", 
        str(state.get("user_profile", "No profile available"))
    ).replace(
        "{{research_findings}}",
        context.get("research_findings", "No research findings available")
    )
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Create workout plan structure
    workout_plan = {
        "content": response.content,
        "created_date": "current_date",
        "version": 1.0,
        "hevy_routines": []  # Would contain IDs of created Hevy routines
    }
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": workout_plan,
        "progress_data": state.get("progress_data"),
        "training_history": state.get("training_history", []),
        "next_action": "orchestrate",
        "context": context
    }

# Analysis Worker - analyzes workout data
def analysis_worker(state: AgentState) -> AgentState:
    """Analyzes workout logs and progress."""
    
    analysis_prompt = """You are a fitness data analyst.
    Retrieve and analyze the user's workout logs to assess progress.
    
    Use these tools:
    - tool_fetch_workouts: Get recent workout data
    - tool_get_workout_count: Check overall workout volume
    
    Analyze:
    1. Workout adherence and consistency
    2. Progress in key metrics (weight, reps, sets)
    3. Comparison to the workout plan
    4. Potential plateaus or areas needing attention
    
    Provide objective analysis backed by data.
    """
    
    messages = state["messages"] + [SystemMessage(content=analysis_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Structure the analysis results
    analysis_results = {
        "analysis": response.content,
        "date": "current_date",
        "key_metrics": {},  # Would contain extracted metrics
        "recommendations": []  # Would contain extracted recommendations
    }
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": state.get("workout_plan"),
        "progress_data": state.get("progress_data"),
        "training_history": state.get("training_history", []),
        "analysis_results": analysis_results,
        "next_action": "orchestrate",
        "context": state.get("context", {})
    }

# Adjustment Worker - modifies workout plans
def adjustment_worker(state: AgentState) -> AgentState:
    """Adjusts workout plans based on analysis."""
    
    adjustment_prompt = """You are a workout optimization specialist.
    Adjust the user's workout plan based on progress analysis.
    
    Current plan: {{workout_plan}}
    Analysis results: {{analysis_results}}
    User profile: {{user_profile}}
    
    Determine the appropriate level of adjustment:
    1. Minor: Adjust weights, reps, or sets
    2. Moderate: Substitute exercises or change training variables
    3. Major: Create a new phase or approach
    
    Use tool_update_routine to implement changes in Hevy.
    
    Explain the scientific rationale for each adjustment.
    """
    
    # Customize the prompt
    filled_prompt = adjustment_prompt.replace(
        "{{workout_plan}}",
        str(state.get("workout_plan", "No plan available"))
    ).replace(
        "{{analysis_results}}",
        str(state.get("analysis_results", "No analysis available"))
    ).replace(
        "{{user_profile}}",
        str(state.get("user_profile", "No profile available"))
    )
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update the workout plan with adjustments
    current_plan = state.get("workout_plan", {})
    adjusted_plan = {
        **current_plan,
        "adjusted_content": response.content,
        "adjustment_date": "current_date",
        "version": current_plan.get("version", 1.0) + 0.1
    }
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "session_id": state["session_id"],
        "user_profile": state.get("user_profile"),
        "workout_plan": adjusted_plan,
        "progress_data": state.get("progress_data"),
        "training_history": state.get("training_history", []),
        "next_action": "orchestrate",
        "context": state.get("context", {})
    }

# Action router
def action_router(state: AgentState) -> str:
    return state["next_action"]

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("assess", assessment_worker)
workflow.add_node("research", research_worker)
workflow.add_node("plan", planning_worker)
workflow.add_node("analyze", analysis_worker)
workflow.add_node("adjust", adjustment_worker)
workflow.add_node("tools", ToolNode([
    retrieve_from_rag, 
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine
]))

# Add orchestrator routing
workflow.add_conditional_edges(
    "orchestrator",
    action_router,
    {
        "assess": "assess",
        "research": "research",
        "plan": "plan",
        "analyze": "analyze",
        "adjust": "adjust",
        "orchestrate": "orchestrator",
        "end": END
    }
)

# Add return edges to orchestrator
for node in ["assess", "research", "plan", "analyze", "adjust"]:
    workflow.add_conditional_edges(
        node,
        action_router,
        {
            "orchestrate": "orchestrator",
        }
    )

# Add tool usage for all nodes
for node in ["assess", "research", "plan", "analyze", "adjust"]:
    workflow.add_conditional_edges(
        node,
        tools_condition,
        {
            True: "tools",
            False: node
        }
    )

# Tool return edges
workflow.add_conditional_edges(
    "tools",
    lambda state: state["next_action"],
    {
        "orchestrate": "orchestrator",
        "assess": "assess",
        "research": "research",
        "plan": "plan",
        "analyze": "analyze",
        "adjust": "adjust"
    }
)


workflow.set_entry_point("orchestrator")

orchestrator_worker_agent = workflow.compile()