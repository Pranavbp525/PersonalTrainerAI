from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from agent.agent_models import AgentState, DeepFitnessResearchState, StreamlinedRoutineState, ProgressAnalysisAdaptationStateV2
from datetime import datetime, timedelta
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
import time # Import time
try:
    # Assuming graph.py is at the same level as elk_logging.py
    from elk_logging import get_agent_logger, setup_elk_logging
except ImportError:
    # Fallback if structure is different (e.g., running from parent dir)
    from ..elk_logging import get_agent_logger, setup_elk_logging

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
                                    tool_execution_node,
                                    fetch_logs_node,
                                    process_targets_node,
                                    fetch_all_routines_node,
                                    compile_final_report_node_v2,
                                    identify_target_routines_node
                                    )

from typing import Literal, Union, Dict, Any, Optional
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.runnables import RunnableConfig


from agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)
import logging


# --- Initialize ELK Logger for Graph Building ---
graph_build_log = setup_elk_logging("fitness-chatbot.graph_builder")
# ---

# --- Remove Basic Logging Config ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__) # Remove module-level logger
# ---

# --- Checkpointer Initialization ---
# Initialize MemorySaver (or your chosen checkpointer)
agent_checkpointer = MemorySaver()
graph_build_log.info("Initialized MemorySaver checkpointer.", extra={"checkpointer_type": type(agent_checkpointer).__name__})
# ---

# --- Helper function to get session_id from state (reusable) ---
def _get_session_id_from_state_or_config(
    state_or_config: Union[Dict, RunnableConfig, None],
    default_prefix: str = "unknown"
) -> str:
    """Safely extracts thread_id (session_id) from state or config."""
    session_id = f"{default_prefix}_session_no_config"
    if isinstance(state_or_config, dict):
        # Try state first
        if "configurable" in state_or_config:
            session_id = state_or_config["configurable"].get("thread_id", f"{default_prefix}_session_state")
        # Try config if passed directly
        elif "config" in state_or_config and isinstance(state_or_config["config"], dict) and "configurable" in state_or_config["config"]:
             session_id = state_or_config["config"]["configurable"].get("thread_id", f"{default_prefix}_session_direct_config")
    elif hasattr(state_or_config, 'get') and callable(state_or_config.get): # Check if it behaves like RunnableConfig dict
        configurable = state_or_config.get('configurable', {})
        if isinstance(configurable, dict):
            session_id = configurable.get('thread_id', f"{default_prefix}_session_runnable_config")

    return session_id
# ---


# --- Routing Functions with Logging ---
def check_completion_and_route_v2(state: DeepFitnessResearchState) -> Literal["generate_rag_query", "finalize_research_report"]:
    """
    Checks overall iteration limit and if all sub-questions are done.
    Relies on 'current_sub_question_idx' being correctly updated by reflect_on_progress_v2.
    """
    # --- Logging Setup ---
    session_id = _get_session_id_from_state_or_config(state, "deep_research")
    router_log = get_agent_logger("deep_research.router", session_id)
    router_log.debug("Executing research routing logic (check_completion_and_route_v2).")
    # ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Routing (v2) ---") # Removed old logger
    iteration = state.get('iteration_count', 0)
    max_iters = state.get('max_iterations', 5)

    if iteration >= max_iters:
        # logger.warning(f"Max overall iterations ({max_iters}) reached. Finalizing report.") # Removed old logger
        router_log.warning("Max overall iterations reached. Routing to finalize.", extra={"iteration": iteration, "max_iterations": max_iters}) # Added log
        return "finalize_research_report"

    sub_questions = state.get('sub_questions', [])
    current_idx = state.get('current_sub_question_idx', 0)

    if current_idx >= len(sub_questions):
        # logger.info(f"All sub-questions ({len(sub_questions)}) completed based on index. Finalizing report.") # Removed old logger
        router_log.info("All sub-questions completed based on index. Routing to finalize.", extra={"current_sub_question_idx": current_idx, "num_sub_questions": len(sub_questions)}) # Added log
        return "finalize_research_report"
    else:
        # logger.info(f"Routing: Continue research (Targeting SubQ Index: {current_idx}, Iteration: {iteration + 1}).") # Removed old logger
        router_log.info("Routing to continue research.", extra={"next_sub_question_idx": current_idx, "next_iteration": iteration + 1}) # Added log
        return "generate_rag_query"

def coordinator_condition(state: AgentState) -> str:
    """Determine the next node based on state conditions."""
    # --- Logging Setup ---
    session_id = _get_session_id_from_state_or_config(state, "main_graph")
    router_log = get_agent_logger("main_graph.coordinator_router", session_id)
    # ---

    # --- Original Logic ---
    selected_agent = state.get("current_agent", "coordinator")
    router_log.debug(f"Coordinator condition checking selected agent.", extra={"selected_agent": selected_agent}) # Added log

    if selected_agent == "assessment":
        router_log.info("Routing to end_conversation (assessment selected).") # Added log
        return "end_conversation"
    elif selected_agent == "end_conversation":
         router_log.info("Routing to end_conversation (end_conversation selected).") # Added log
         return "end_conversation"
    else:
        # Check if the selected agent is a valid node name
        # (Add this check if node names might differ from agent names)
        # valid_nodes = ["deep_research", "planning_agent", "progress_analysis_adaptation_agent", "coach_agent", "user_modeler"]
        # if selected_agent not in valid_nodes:
        #     router_log.error(f"Coordinator selected invalid agent/node name: {selected_agent}. Defaulting to coordinator.")
        #     return "coordinator"
        router_log.info(f"Routing to selected agent node: {selected_agent}") # Added log
        return selected_agent


def _check_targets_found(state: ProgressAnalysisAdaptationStateV2) -> str:
    # --- Logging Setup ---
    session_id = _get_session_id_from_state_or_config(state, "progress_adapt")
    router_log = get_agent_logger("progress_adapt.target_check_router", session_id)
    router_log.debug("Executing target check routing logic (_check_targets_found).")
    # ---

    # --- Original Logic ---
    process_error = state.get("process_error")
    if process_error:
        # logger.warning(f"Routing to compile_report due to process error: {state['process_error']}") # Removed old logger
        router_log.warning("Routing to compile_report due to previous process error.", extra={"process_error": process_error}) # Added log
        return "compile_report"

    identified = state.get("identified_targets")
    num_identified = len(identified) if isinstance(identified, list) else 0

    if num_identified > 0:
        # logger.info(f"Targets identified ({len(identified)}). Proceeding to process.") # Removed old logger
        router_log.info(f"Targets identified. Routing to process targets.", extra={"identified_count": num_identified}) # Added log
        return "process_targets"
    else:
        # logger.info("No relevant targets identified for adaptation. Proceeding to compile report.") # Removed old logger
        router_log.info("No targets identified. Routing to compile report.", extra={"identified_count": 0}) # Added log
        return "compile_report"
# --- End Routing Functions ---


# --- State Management Utilities (Logging not typically added here unless debugging state issues) ---
async def get_or_create_state(session_id: str) -> AgentState:
    """Retrieve existing state or create a new one."""
    # --- Logging Setup ---
    # Use a generic logger here as session_id might be new
    state_log = setup_elk_logging("fitness-chatbot.state_manager")
    state_log.debug(f"Called get_or_create_state for session.", extra={"session_id": session_id})
    # ---
    # In a real implementation, check checkpointer/database
    # checkpoint_tuple: Optional[CheckpointTuple] = agent_checkpointer.get(config={"configurable": {"thread_id": session_id}})
    # if checkpoint_tuple:
    #     state_log.info("Retrieved existing state for session.", extra={"session_id": session_id})
    #     return checkpoint_tuple.checkpoint # Return the state dictionary from the checkpoint
    # else:
    #     state_log.info("No existing state found, creating new state for session.", extra={"session_id": session_id})
    #     # Return initial state structure
    # --- For MemorySaver demo (always creates new): ---
    state_log.info("Creating new initial state for session (MemorySaver behavior).", extra={"session_id": session_id})
    return {
        "messages": [],
        "session_id": session_id, # Store session_id within state if needed by nodes
        "memory": {}, "working_memory": {}, "user_model": {},
        "hevy_payloads": None, "hevy_results": None, # Added hevy_results initialization
        "progress_data": {}, "reasoning_trace": [],
        "agent_state": {"status": "new"},
        "current_agent": "coordinator",
        "research_topic": None, "user_profile_str": None, "final_report": None,
        "tool_calls": None, "tool_results": None,
        'user_request_context': None, 'final_report_and_notification': None,
        'cycle_completed_successfully': None, 'processed_results': None,
        "errors": [], # Initialize errors list
        # --- Required for subgraphs ---
        "fetched_routines_list": None, "workout_logs": None, "identified_targets": None, # ProgressAdapt
        "planner_structured_output": None, # StreamlinedRoutine
        "sub_questions": None, "accumulated_findings": None, "iteration_count": None, # DeepResearch
        "current_sub_question_idx": None, "reflections": None, "research_complete": None, # DeepResearch (cont.)
        "queries_this_sub_question": None, "sub_question_complete_flag": None, # DeepResearch (cont.)
        "max_iterations": None, "max_queries_per_sub_question": None, "current_rag_query": None, "rag_results": None # DeepResearch (cont.)
    }

async def save_state(session_id: str, state: AgentState) -> None:
    """Save state to persistent storage."""
    # Logging not typically useful here unless debugging checkpointer issues
    # print(f"State saved for session {session_id}") # Original stub
    pass # Checkpointer handles saving automatically when used in .compile()
# --- End State Management ---

# --- Error Handling Wrapper with Logging ---
# Note: This might interfere with LangGraph's internal error handling. Use with caution.
# Consider letting errors propagate to LangGraph first.
def agent_with_error_handling(agent_func):
    """Wrapper to add error handling and logging to agent functions."""
    # --- Logging Setup ---
    # Get logger name from the function being wrapped
    base_logger_name = f"fitness-chatbot.agent.{agent_func.__name__}"
    # ---
    async def wrapped_agent(state: Union[Dict, AgentState, Any], config: Optional[RunnableConfig] = None) -> Union[Dict, AgentState, Any]:
        session_id = _get_session_id_from_state_or_config(state, agent_func.__name__)
        agent_log = get_agent_logger(base_logger_name, session_id) # Use specific agent logger
        node_start_time = time.time()
        agent_log.debug(f"Entering wrapped agent: {agent_func.__name__}")
        try:
            # Pass config if the original function accepts it
            if "config" in agent_func.__code__.co_varnames:
                 result_state = await agent_func(state, config=config)
            else:
                 result_state = await agent_func(state)
            node_duration = time.time() - node_start_time
            agent_log.debug(f"Exiting wrapped agent successfully.", extra={"duration_seconds": round(node_duration, 2)})
            return result_state
        except Exception as e:
            node_duration = time.time() - node_start_time
            agent_log.error(f"Error occurred in wrapped agent '{agent_func.__name__}'. Returning error state.", exc_info=True, extra={"duration_seconds": round(node_duration, 2)})
            # --- Original Error Handling Logic (modified for safety) ---
            error_message = f"I encountered an issue in '{agent_func.__name__}': {type(e).__name__}" # Avoid exposing full error detail potentially
            # Ensure state is a mutable dictionary before modifying
            if isinstance(state, dict):
                mutable_state = state
            else:
                # If state is not a dict (e.g., Pydantic model), create a dict copy if possible
                # This is a fallback and might not capture all necessary state fields
                agent_log.warning("Attempting to handle error but input state is not a dict. Creating basic error state.")
                mutable_state = {"original_state_type": str(type(state))}

            # Safely append message if 'messages' key exists and is a list
            if "messages" in mutable_state and isinstance(mutable_state["messages"], list):
                 mutable_state["messages"] = mutable_state["messages"] + [AIMessage(content=error_message)]
            else:
                 mutable_state["messages"] = [AIMessage(content=error_message)] # Initialize messages

            # Safely update agent_state
            current_agent_state = mutable_state.get("agent_state", {})
            if isinstance(current_agent_state, dict):
                mutable_state["agent_state"] = current_agent_state | {"error": str(e), "status": "error"}
            else:
                 mutable_state["agent_state"] = {"error": str(e), "status": "error"}

            # Safely update current_agent
            mutable_state["current_agent"] = "coordinator" # Route back for recovery attempt

            # Safely update errors list
            if "errors" in mutable_state and isinstance(mutable_state["errors"], list):
                 mutable_state["errors"] = mutable_state["errors"] + [f"Error in {agent_func.__name__}: {str(e)}"]
            else:
                 mutable_state["errors"] = [f"Error in {agent_func.__name__}: {str(e)}"]

            return mutable_state
    return wrapped_agent
# --- End Error Handling ---


# --- Subgraph Builders with Logging ---
def build_progress_analysis_adaptation_graph_v2(): # Return type hint
    # --- Logging Setup ---
    subgraph_build_log = graph_build_log.add_context(subgraph_name="progress_analysis_adaptation_v2")
    subgraph_build_log.info("Building progress analysis & adaptation subgraph V2...")
    # ---
    workflow = StateGraph(ProgressAnalysisAdaptationStateV2)

    # Add nodes (Names match the functions)
    subgraph_build_log.debug("Adding nodes...")
    workflow.add_node("fetch_routines", fetch_all_routines_node)
    workflow.add_node("fetch_logs", fetch_logs_node)
    workflow.add_node("identify_targets", identify_target_routines_node)
    workflow.add_node("process_targets", process_targets_node)
    workflow.add_node("compile_report", compile_final_report_node_v2)

    # Define edges
    subgraph_build_log.debug("Defining edges...")
    workflow.set_entry_point("fetch_routines")
    workflow.add_edge("fetch_routines", "fetch_logs")
    workflow.add_edge("fetch_logs", "identify_targets")
    workflow.add_conditional_edges(
        "identify_targets",
        _check_targets_found, # Router function
        {
            "process_targets": "process_targets",
            "compile_report": "compile_report"
        }
    )
    workflow.add_edge("process_targets", "compile_report")
    workflow.add_edge("compile_report", END) # Connect final node to subgraph's END

    # logger.info("Progress Analysis & Adaptation subgraph built (V2 - Identification Inside).") # Removed old logger
    compiled_graph = workflow.compile()
    subgraph_build_log.info("Progress analysis & adaptation subgraph V2 compiled successfully.")
    return compiled_graph

def build_deep_research_subgraph(): # Return type hint
    """Builds the LangGraph StateGraph for the deep research loop."""
    # --- Logging Setup ---
    subgraph_build_log = graph_build_log.add_context(subgraph_name="deep_research")
    subgraph_build_log.info("Building deep research subgraph...")
    # ---
    builder = StateGraph(DeepFitnessResearchState)

    # Add Nodes
    subgraph_build_log.debug("Adding nodes...")
    builder.add_node("plan_steps", plan_research_steps)
    builder.add_node("generate_query", generate_rag_query_v2)
    builder.add_node("execute_rag", execute_rag_direct)
    builder.add_node("synthesize", synthesize_rag_results)
    builder.add_node("reflect", reflect_on_progress_v2)
    builder.add_node("finalize_report", finalize_research_report)

    # Define Edges
    subgraph_build_log.debug("Defining edges...")
    # builder.add_edge(START, "plan_steps") # START is implicit entry for subgraphs unless set_entry_point used
    builder.set_entry_point("plan_steps") # Explicitly set entry point
    builder.add_edge("plan_steps", "generate_query")
    builder.add_edge("generate_query", "execute_rag")
    builder.add_edge("execute_rag", "synthesize")
    builder.add_edge("synthesize", "reflect")

    # Conditional Routing from Reflection step
    builder.add_conditional_edges(
        "reflect",
        check_completion_and_route_v2, # Router function
        {
            "generate_rag_query": "generate_query",
            "finalize_research_report": "finalize_report"
        }
    )

    builder.add_edge("finalize_report", END) # Connect final node to subgraph's END

    # Compile the subgraph
    subgraph = builder.compile()
    # logger.info("Deep research subgraph compiled successfully.") # Removed old logger
    subgraph_build_log.info("Deep research subgraph compiled successfully.")
    return subgraph

def build_streamlined_routine_graph(): # Return type hint
    # --- Logging Setup ---
    subgraph_build_log = graph_build_log.add_context(subgraph_name="streamlined_routine")
    subgraph_build_log.info("Building streamlined routine creation subgraph...")
    # ---
    workflow = StateGraph(StreamlinedRoutineState)

    # Add nodes
    subgraph_build_log.debug("Adding nodes...")
    workflow.add_node("planner", structured_planning_node)
    workflow.add_node("format_lookup", format_and_lookup_node)
    workflow.add_node("execute_tool", tool_execution_node)

    # Define edges (Linear Flow)
    subgraph_build_log.debug("Defining edges...")
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "format_lookup")
    workflow.add_edge("format_lookup", "execute_tool")
    workflow.add_edge("execute_tool", END) # End after tool execution

    # Compile the graph
    app = workflow.compile()
    # logger.info("Streamlined Routine Creation Agent graph compiled.") # Removed old logger
    subgraph_build_log.info("Streamlined routine creation subgraph compiled successfully.")
    return app
# --- End Subgraph Builders ---


# --- Main Graph Builder with Logging ---
def build_fitness_trainer_graph(): # Return type hint
    """Constructs the multi-agent graph for the fitness trainer."""
    # --- Logging Setup ---
    main_graph_log = graph_build_log.add_context(graph_name="fitness_trainer_main")
    main_graph_log.info("Building main fitness trainer graph...")
    # ---

    workflow = StateGraph(AgentState)

    # Build Subgraphs first
    main_graph_log.info("Building required subgraphs...")
    deep_research_subgraph = build_deep_research_subgraph()
    planning_subgraph = build_streamlined_routine_graph()
    progress_analysis_adaptation_subgraph = build_progress_analysis_adaptation_graph_v2()
    main_graph_log.info("Subgraphs built.")

    # --- Add Nodes (Using error handling wrapper for non-subgraph nodes) ---
    main_graph_log.debug("Adding main graph nodes...")
    workflow.add_node("coordinator", agent_with_error_handling(coordinator))
    workflow.add_node("user_modeler", agent_with_error_handling(user_modeler))
    workflow.add_node("coach_agent", agent_with_error_handling(coach_agent))
    workflow.add_node("end_conversation", agent_with_error_handling(end_conversation))

    # --- Add Subgraph Nodes ---
    # Names here MUST match the keys used in the coordinator_condition router
    workflow.add_node("deep_research", deep_research_subgraph)
    workflow.add_node("planning_agent", planning_subgraph)
    workflow.add_node("progress_adaptation_agent", progress_analysis_adaptation_subgraph)

    # --- Add Tools Node (Not typically wrapped with error handling) ---
    tools_list = [
        retrieve_from_rag, tool_fetch_workouts, tool_get_workout_count,
        tool_fetch_routines, tool_update_routine, tool_create_routine
    ]
    workflow.add_node("tools", ToolNode(tools_list))
    main_graph_log.debug(f"Added tools node with {len(tools_list)} tools.")


    # --- Define Edges ---
    main_graph_log.debug("Defining main graph edges...")
    # Conditional routing from Coordinator
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_condition, # Router function
        {
            # Map agent names (keys) to actual node names (values)
            "deep_research": "deep_research",
            "planning_agent": "planning_agent",
            "progress_adaptation_agent": "progress_adaptation_agent",
            "coach_agent": "coach_agent",
            "user_modeler": "user_modeler",
            "end_conversation": "end_conversation"
            # Add "tools": "tools" if coordinator can route directly to tools
        }
    )

    # Connect agent/subgraph nodes back to coordinator
    workflow.add_edge("deep_research", "coordinator")
    workflow.add_edge("planning_agent", "coordinator")
    workflow.add_edge("progress_adaptation_agent", "coordinator")
    workflow.add_edge("coach_agent", "coordinator")
    workflow.add_edge("user_modeler", "coordinator")
    # Do NOT connect tools node back automatically, needs specific routing logic usually

    # Connect end_conversation to the graph's END
    workflow.add_edge("end_conversation", END)

    # Set Entry Point
    workflow.set_entry_point("coordinator")
    main_graph_log.debug("Set graph entry point to 'coordinator'.")

    # Compile the graph with the checkpointer
    main_graph_log.info("Compiling main graph with checkpointer...")
    compiled_app = workflow.compile(checkpointer=agent_checkpointer)
    main_graph_log.info("Main fitness trainer graph compiled successfully.")

    return compiled_app