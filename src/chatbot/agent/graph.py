from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from agent.agent_models import AgentState
from datetime import datetime, timedelta
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.personal_trainer_agent import (coach_agent,
                                    coordinator,
                                    user_modeler,
                                    research_agent,
                                    planning_agent,
                                    adaptation_agent,
                                    progress_analysis_agent,
                                    end_conversation)


from agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)




def coordinator_condition(state: AgentState):
    """Determine the next node based on state conditions."""
    selected_agent = state.get("current_agent", "coordinator")
    
    if selected_agent == "assessment":
        return "end_conversation"  # Route to end node for user input
    elif selected_agent == "end_conversation":
        return "end_conversation"
    else:
        # Route to the selected agent
        return selected_agent
    

# Agent selector function
async def agent_selector(state: AgentState, reasoning_text: str = "") -> str:
    """Determines which agent should handle the current interaction."""

    # For new users, ensure comprehensive assessment
    if state.get("agent_state", {}).get("status") == "complete":
        return "end_conversation"

    if not state.get("user_model") or state.get("user_model", {}).get("assessment_complete") != True:
        return "assessment_agent"
    
    # For users with assessment but no plan, prioritize research and planning
    if state.get("user_model") and not state.get("fitness_plan"):
        if not state.get("working_memory", {}).get("research_findings"):
            return "research_agent"
        else:
            return "planning_agent"
    
    # For users with existing plans, check if analysis is needed
    if state.get("fitness_plan") and state.get("working_memory", {}).get("last_analysis_date"):
        last_analysis = datetime.fromisoformat(state.get("working_memory", {}).get("last_analysis_date"))
        if (datetime.now() - last_analysis).days >= 7:  # Weekly analysis
            return "progress_analysis_agent"
    
    # Use reasoning text to determine appropriate agent
    if reasoning_text:
        if "assessment" in reasoning_text.lower() or "profile" in reasoning_text.lower():
            return "assessment_agent"
        elif "research" in reasoning_text.lower() or "knowledge" in reasoning_text.lower():
            return "research_agent"
        elif "plan" in reasoning_text.lower() or "routine" in reasoning_text.lower():
            return "planning_agent"
        elif "progress" in reasoning_text.lower() or "analyze" in reasoning_text.lower():
            return "progress_analysis_agent"
        elif "adjust" in reasoning_text.lower() or "modify" in reasoning_text.lower():
            return "adaptation_agent"
        elif "motivate" in reasoning_text.lower() or "coach" in reasoning_text.lower():
            return "coach_agent"
    
    # Default to coordinator for general interactions
    return "coordinator"

# State management utilities
async def get_or_create_state(session_id: str) -> AgentState:
    """Retrieve existing state or create a new one."""
    # In a real implementation, this would check a database
    # For now, just create a new state
    return {
        "messages": [],
        "session_id": session_id,
        "memory": {},
        "working_memory": {},
        "user_model": {},
        "fitness_plan": {},
        "progress_data": {},
        "reasoning_trace": [],
        "agent_state": {"status": "new"},
        "current_agent": "coordinator",
        "tool_calls": None,
        "tool_results": None
    }

async def save_state(session_id: str, state: AgentState) -> None:
    """Save state to persistent storage."""
    # In a real implementation, this would save to a database
    # For now, just print a message
    print(f"State saved for session {session_id}")

# Enhanced error handling wrapper
def agent_with_error_handling(agent_func):
    """Wrapper to add error handling to agent functions."""
    async def wrapped_agent(state: AgentState) -> AgentState:
        try:
            return await agent_func(state)
        except Exception as e:
            # Log the error
            print(f"Error in {agent_func.__name__}: {str(e)}")
            
            # Add error message to state
            error_message = f"I encountered an issue while processing your request: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            
            # Update agent state
            state["agent_state"] = state.get("agent_state", {}) | {"error": str(e), "status": "error"}
            
            # Return to coordinator for recovery
            state["current_agent"] = "coordinator"
            
            return state
    
    return wrapped_agent



def build_fitness_trainer_graph():
    """Constructs the simplified multi-agent graph for the fitness trainer."""
    workflow = StateGraph(AgentState)
    checkpointer = MemorySaver()
    
    # Add nodes
    workflow.add_node("coordinator", agent_with_error_handling(coordinator))
    workflow.add_node("user_modeler", agent_with_error_handling(user_modeler))
    workflow.add_node("research_agent", agent_with_error_handling(research_agent))
    workflow.add_node("planning_agent", agent_with_error_handling(planning_agent))
    workflow.add_node("progress_analysis_agent", agent_with_error_handling(progress_analysis_agent))
    workflow.add_node("adaptation_agent", agent_with_error_handling(adaptation_agent))
    workflow.add_node("coach_agent", agent_with_error_handling(coach_agent))
    
    
    # Add tools node
    workflow.add_node("tools", ToolNode([
        retrieve_from_rag,
        tool_fetch_workouts,
        tool_get_workout_count,
        tool_fetch_routines,
        tool_update_routine,
        tool_create_routine
    ]))
    
    # Add the end node for handling user input
    workflow.add_node("end_conversation", agent_with_error_handling(end_conversation))
    
    # Define the edges - route based on coordinator's decision
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_condition,
        {
            "research_agent": "research_agent",
            "planning_agent": "planning_agent",
            "progress_analysis_agent": "progress_analysis_agent",
            "adaptation_agent": "adaptation_agent",
            "coach_agent": "coach_agent",
            "user_modeler": "user_modeler",
            "end_conversation": "end_conversation"
        }
    )
    
    # Connect everything back to coordinator
    workflow.add_edge("research_agent", "coordinator")
    workflow.add_edge("planning_agent", "coordinator")
    workflow.add_edge("progress_analysis_agent", "coordinator")
    workflow.add_edge("adaptation_agent", "coordinator")
    workflow.add_edge("coach_agent", "coordinator")
    workflow.add_edge("user_modeler", "coordinator")
    
    
    # Connect end_conversation to END for final responses
    workflow.add_edge("end_conversation", END)
    
    workflow.set_entry_point("coordinator")
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)


