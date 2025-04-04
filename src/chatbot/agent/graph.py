from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from agent.agent_models import AgentState, DeepFitnessResearchState, StreamlinedRoutineState
from datetime import datetime, timedelta
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent.personal_trainer_agent import (coach_agent,
                                    coordinator,
                                    user_modeler,
                                    research_agent,
                                    planning_agent,
                                    adaptation_agent,
                                    progress_analysis_agent,
                                    end_conversation,
                                    plan_research_steps,
                                    generate_rag_query_v2,
                                    execute_rag_direct,
                                    synthesize_rag_results,
                                    reflect_on_progress_v2,
                                    finalize_research_report,
                                    structured_planning_node,
                                    format_and_lookup_node,
                                    tool_execution_node
                                    )

from typing import Literal


from agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_completion_and_route_v2(state: DeepFitnessResearchState) -> Literal["generate_rag_query", "finalize_research_report"]:
    """
    Checks overall iteration limit and if all sub-questions are done.
    Relies on 'current_sub_question_idx' being correctly updated by reflect_on_progress_v2.
    """
    logger.info("--- Deep Research: Routing (v2) ---")
    iteration = state.get('iteration_count', 0) # Iteration count already updated by generate_query
    max_iters = state.get('max_iterations', 5)

    # Check overall iteration limit FIRST
    if iteration >= max_iters:
        logger.warning(f"Max overall iterations ({max_iters}) reached. Finalizing report.")
        return "finalize_research_report"

    # Check if index (potentially updated by reflection) is now out of bounds
    sub_questions = state.get('sub_questions', [])
    current_idx = state.get('current_sub_question_idx', 0)

    if current_idx >= len(sub_questions):
        logger.info(f"All sub-questions ({len(sub_questions)}) completed based on index. Finalizing report.")
        return "finalize_research_report"
    else:
        # Log which sub-question index we are targeting *next*
        logger.info(f"Routing: Continue research (Targeting SubQ Index: {current_idx}, Iteration: {iteration + 1}).")
        return "generate_rag_query"


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
            return "deep_research"
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
            return "deep_research"
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
        "hevy_payloads": None,
        "progress_data": {},
        "reasoning_trace": [],
        "agent_state": {"status": "new"},
        "current_agent": "coordinator",
        "research_topic": None,
        "user_profile_str": None,
        "final_report": None,
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



def build_deep_research_subgraph():
    """Builds the LangGraph StateGraph for the deep research loop."""
    builder = StateGraph(DeepFitnessResearchState)

    # Add Nodes
    builder.add_node("plan_steps", plan_research_steps)
    builder.add_node("generate_query", generate_rag_query_v2) # Use v2
    builder.add_node("execute_rag", execute_rag_direct)
    builder.add_node("synthesize", synthesize_rag_results)
    builder.add_node("reflect", reflect_on_progress_v2) # Use v2
    builder.add_node("finalize_report", finalize_research_report)

    # Define Edges
    builder.add_edge(START, "plan_steps")
    builder.add_edge("plan_steps", "generate_query") # Start loop after planning
    builder.add_edge("generate_query", "execute_rag")
    builder.add_edge("execute_rag", "synthesize")
    builder.add_edge("synthesize", "reflect")

    # Conditional Routing from Reflection step
    builder.add_conditional_edges(
        "reflect",
        check_completion_and_route_v2, # Use v2 router logic
        {
            "generate_rag_query": "generate_query",      # Loop back to generate query (for same or next subQ)
            "finalize_research_report": "finalize_report" # Finish research
        }
    )

    builder.add_edge("finalize_report", END) # END node of the subgraph

    # Compile the subgraph
    subgraph = builder.compile()
    logger.info("Deep research subgraph compiled successfully.")
    return subgraph

def build_streamlined_routine_graph():
    workflow = StateGraph(StreamlinedRoutineState)

    # Add nodes
    workflow.add_node("planner", structured_planning_node)
    workflow.add_node("format_lookup", format_and_lookup_node)
    workflow.add_node("execute_tool", tool_execution_node)

    # Define edges (Linear Flow)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "format_lookup")
    workflow.add_edge("format_lookup", "execute_tool")
    workflow.add_edge("execute_tool", END) # End after tool execution

    # Compile the graph
    app = workflow.compile()
    logger.info("Streamlined Routine Creation Agent graph compiled.")
    return app


def build_fitness_trainer_graph():
    """Constructs the simplified multi-agent graph for the fitness trainer."""
    workflow = StateGraph(AgentState)
    checkpointer = MemorySaver()

    deep_research_subgraph = build_deep_research_subgraph()
    planning_subgraph = build_streamlined_routine_graph()
    
    # Add nodes
    workflow.add_node("coordinator", agent_with_error_handling(coordinator))
    workflow.add_node("user_modeler", agent_with_error_handling(user_modeler))
    # workflow.add_node("research_agent", agent_with_error_handling(research_agent))
    workflow.add_node("deep_research", deep_research_subgraph)
    # workflow.add_node("planning_agent", agent_with_error_handling(planning_agent))
    workflow.add_node("planning_agent", planning_subgraph)
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
            "deep_research": "deep_research",
            "planning_agent": "planning_agent",
            "progress_analysis_agent": "progress_analysis_agent",
            "adaptation_agent": "adaptation_agent",
            "coach_agent": "coach_agent",
            "user_modeler": "user_modeler",
            "end_conversation": "end_conversation"
        }
    )
    
    # Connect everything back to coordinator
    workflow.add_edge("deep_research", "coordinator")
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


