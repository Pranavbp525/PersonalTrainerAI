"""
Enhanced Multi-Agent Architecture for Personal Fitness Trainer AI

This implementation combines the strengths of the multi-agent system with
specialized components for fitness training, RAG integration, and Hevy API usage.
"""

from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
# from langchain_core.pydantic_v1 import BaseModel, Field # Use v1 Pydantic with LangChain/Graph typically
from langchain.output_parsers.pydantic import PydanticOutputParser
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
import json
import os
import time
# import logging
import re
import asyncio
from elk_logging import get_agent_logger
import uuid
from pydantic import BaseModel, Field
from agent.agent_models import (
    SetRoutineCreate, ExerciseRoutineCreate, RoutineCreate, RoutineCreateRequest,
    SetRoutineUpdate, ExerciseRoutineUpdate, RoutineUpdate, RoutineUpdateRequest,
    PlannerExerciseRoutineCreate, PlannerRoutineCreate, PlannerSetRoutineCreate, PlannerOutputContainer, HevyRoutineApiPayload,
    AnalysisFindings, IdentifiedRoutineTarget,
    RoutineAdaptationResult
)
from agent.agent_models import AgentState, UserProfile, DeepFitnessResearchState, StreamlinedRoutineState, ProgressAnalysisAdaptationStateV2
from agent.utils import (extract_adherence_rate,
                   extract_adjustments,
                   extract_approaches,
                   extract_citations,
                   extract_issues,
                   extract_principles,
                   extract_progress_metrics,
                   extract_routine_data,
                   extract_routine_structure,
                   extract_routine_updates,
                   retrieve_data,
                   get_exercise_template_by_title_fuzzy,
                   validate_and_lookup_exercises)
from agent.prompts import (
    get_adaptation_prompt,
    get_analysis_prompt,
    get_coach_prompt,
    get_coordinator_prompt,
    get_memory_consolidation_prompt,
    get_planning_prompt,
    get_research_prompt,
    get_user_modeler_prompt,
    summarize_routine_prompt,
    get_analysis_v2_template,
    get_final_cycle_report_template_v2,
    get_reasoning_generation_template,
    get_routine_identification_prompt,
    get_routine_modification_template_v2,
    get_targeted_rag_query_template,
    get_analysis_v2_prompt,
    get_final_cycle_report_v2_prompt,
    get_reasoning_generation_prompt,
    get_routine_modification_v2_prompt,
    get_summarize_routine_prompt,
    get_targeted_rag_query_prompt,
    _get_prompt_from_langsmith,
    get_finalize_research_report_prompt,
    get_generate_rag_query_v2_prompt,
    generate_rag_query_v2_prompt,
    get_plan_research_steps_prompt,
    get_reflect_on_progress_v2_prompt,
    get_structured_planning_prompt,
    get_synthesize_rag_results_prompt,
)


from agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)



from dotenv import load_dotenv


from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.schema.runnable import RunnableLambda


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

load_dotenv()




# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))

# Configure LLM with tools
llm_with_tools = llm.bind_tools([
    retrieve_from_rag,
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine
])



# User modeler for comprehensive user understanding
async def user_modeler(state: AgentState) -> AgentState:
    """Builds and maintains a comprehensive model of the user."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = state.get("configurable", {}).get("thread_id", "unknown_session")
    agent_log = get_agent_logger("user_modeler", session_id)
    agent_log.info("Entering user_modeler node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info(f"User Modeler : Updating the User Modeller") # Removed old logger
    agent_log.debug("Preparing Pydantic parser and prompt") # Added log
    parser = PydanticOutputParser(pydantic_object=UserProfile)
    format_instructions = parser.get_format_instructions()
    prompt_template = get_user_modeler_prompt()

    current_user_model = state.get("user_model", {})
    recent_exchanges = state.get("working_memory", {}).get("recent_exchanges", [])

    prompt_context = { # Added context for logging
        "has_current_model": bool(current_user_model),
        "num_recent_exchanges": len(recent_exchanges),
    }
    agent_log.debug("Formatting prompt", extra=prompt_context) # Added log
    formatted_prompt = prompt_template.format(
        user_model=json.dumps(current_user_model),
        recent_exchanges=json.dumps(recent_exchanges),
        format_instructions=format_instructions
    )

    messages = [SystemMessage(content=formatted_prompt)]
    response = None # Initialize response
    parsed_model = None # Initialize parsed_model

    # --- Logging around LLM call and Parsing ---
    try:
        agent_log.info("Invoking LLM for user model update")
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke(messages)
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info(f"LLM invocation completed", extra={"duration_seconds": round(llm_duration, 2)})

        agent_log.debug("Attempting to parse LLM response")
        parsing_start_time = time.time()
        # --- Original Logic: Parsing ---
        parsed_model = parser.parse(response.content)
        # --- End Original Logic ---
        parsing_duration = time.time() - parsing_start_time
        agent_log.info("Successfully parsed LLM response into UserProfile model", extra={"duration_seconds": round(parsing_duration, 2)})

    except Exception as e:
        agent_log.error(f"Error during LLM call or parsing in user_modeler", exc_info=True)
        # Original logic didn't explicitly handle errors here, so we just log and continue
        # The rest of the original logic will proceed with potentially None `parsed_model`
    # --- End Logging around LLM call and Parsing ---

    # --- Original Logic: Update user model ---
    user_model = state.get("user_model", {}) # Gets original or {}
    user_model["last_updated"] = datetime.now().isoformat()
    user_model["model_version"] = user_model.get("model_version", 0) + 1

    # --- Logging before update loop ---
    update_count = 0
    updated_fields_list = [] # For logging which fields were updated
    # ---

    # Check if parsing was successful before attempting to update
    if parsed_model: # Check added for safety based on added try/except
        agent_log.debug("Updating user model fields based on parsed response") # Added log
        # Original logic implicitly updates based on parsed_model structure
        if parsed_model.name is not None:
            if user_model.get("name") != parsed_model.name: updated_fields_list.append("name"); update_count += 1
            user_model["name"] = parsed_model.name
        if parsed_model.age is not None:
            if user_model.get("age") != parsed_model.age: updated_fields_list.append("age"); update_count += 1
            user_model["age"] = parsed_model.age
        if parsed_model.gender is not None:
            if user_model.get("gender") != parsed_model.gender: updated_fields_list.append("gender"); update_count += 1
            user_model["gender"] = parsed_model.gender
        if parsed_model.goals is not None:
            if user_model.get("goals") != parsed_model.goals: updated_fields_list.append("goals"); update_count += 1
            user_model["goals"] = parsed_model.goals
        if parsed_model.preferences is not None:
            if user_model.get("preferences") != parsed_model.preferences: updated_fields_list.append("preferences"); update_count += 1
            user_model["preferences"] = parsed_model.preferences
        if parsed_model.constraints is not None:
            if user_model.get("constraints") != parsed_model.constraints: updated_fields_list.append("constraints"); update_count += 1
            user_model["constraints"] = parsed_model.constraints
        if parsed_model.fitness_level is not None:
            if user_model.get("fitness_level") != parsed_model.fitness_level: updated_fields_list.append("fitness_level"); update_count += 1
            user_model["fitness_level"] = parsed_model.fitness_level
        if parsed_model.motivation_factors is not None:
            if user_model.get("motivation_factors") != parsed_model.motivation_factors: updated_fields_list.append("motivation_factors"); update_count += 1
            user_model["motivation_factors"] = parsed_model.motivation_factors
        if parsed_model.learning_style is not None:
            if user_model.get("learning_style") != parsed_model.learning_style: updated_fields_list.append("learning_style"); update_count += 1
            user_model["learning_style"] = parsed_model.learning_style
        if parsed_model.confidence_scores is not None:
            if user_model.get("confidence_scores") != parsed_model.confidence_scores: updated_fields_list.append("confidence_scores"); update_count += 1
            user_model["confidence_scores"] = parsed_model.confidence_scores
        if parsed_model.available_equipment is not None:
            if user_model.get("available_equipment") != parsed_model.available_equipment: updated_fields_list.append("available_equipment"); update_count += 1
            user_model["available_equipment"] = parsed_model.available_equipment
        if parsed_model.training_environment is not None:
            if user_model.get("training_environment") != parsed_model.training_environment: updated_fields_list.append("training_environment"); update_count += 1
            user_model["training_environment"] = parsed_model.training_environment
        if parsed_model.schedule is not None:
            if user_model.get("schedule") != parsed_model.schedule: updated_fields_list.append("schedule"); update_count += 1
            user_model["schedule"] = parsed_model.schedule
        if parsed_model.measurements is not None:
            if user_model.get("measurements") != parsed_model.measurements: updated_fields_list.append("measurements"); update_count += 1
            user_model["measurements"] = parsed_model.measurements
        if parsed_model.height is not None:
            if user_model.get("height") != parsed_model.height: updated_fields_list.append("height"); update_count += 1
            user_model["height"] = parsed_model.height
        if parsed_model.weight is not None:
            if user_model.get("weight") != parsed_model.weight: updated_fields_list.append("weight"); update_count += 1
            user_model["weight"] = parsed_model.weight
        if parsed_model.workout_history is not None:
            # Be careful comparing large histories; maybe just check if it exists
            if user_model.get("workout_history") != parsed_model.workout_history: updated_fields_list.append("workout_history"); update_count += 1
            user_model["workout_history"] = parsed_model.workout_history
        # --- Logging after update loop ---
        agent_log.info(f"User model update attempted.", extra={"fields_updated_count": update_count, "updated_fields": updated_fields_list})
    else:
        # Log that updates were skipped because parsing failed
        agent_log.warning("Skipping user model field updates because LLM parsing failed or yielded no result.")


    # --- Original Logic: Check assessment ---
    required_fields = ["goals", "fitness_level", "available_equipment",
                       "training_environment", "schedule", "constraints"]
    missing_fields = [field for field in required_fields
                      if field not in user_model or not user_model.get(field)] # Uses the updated user_model dict

    user_model["missing_fields"] = missing_fields
    assessment_complete = len(missing_fields) == 0
    user_model["assessment_complete"] = assessment_complete

    # --- Logging assessment result ---
    agent_log.info(f"User model assessment check complete", extra={
        "assessment_complete": assessment_complete,
        "missing_fields": missing_fields,
        "model_version": user_model.get("model_version")
    })
    # ---

    # logger.info(f"User model updated successfully via PydanticOutputParser") # Removed old logger

    # --- Original Logic: Construct updated state ---
    # This uses dict unpacking which creates a new dictionary, implicitly copying state
    # and overwriting 'user_model' and 'current_agent'
    updated_state = {
        **state,
        "user_model": user_model, # Use the user_model dict that was updated in place
        "current_agent": "coordinator"
    }

    # logger.info(f"User Modeler : Updated the User Modeller Successfully") # Removed old logger

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting user_modeler node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return updated_state # Return the originally constructed state


async def coordinator(state: AgentState) -> AgentState:
    """Central coordinator that manages the overall interaction flow, assessment, and memory."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = state.get("configurable", {}).get("thread_id", "unknown_session")
    agent_log = get_agent_logger("coordinator", session_id)
    agent_log.info("Entering coordinator node")
    # Log key input state details (be careful with large objects)
    agent_log.debug("Coordinator input state keys", extra={"state_keys": list(state.keys())})
    # --- End Logging Setup ---

    # --- <<<< CHECK FOR RETURN FROM PROGRESS/ADAPTATION SUBGRAPH >>>> ---
    progress_notification = state.get("final_report_and_notification")
    if progress_notification is not None:
        # logger.info("Coordinator: Detected return from progress_adaptation_subgraph.") # Old logger
        agent_log.info("Detected return from progress_adaptation_subgraph.") # New logger
        working_memory = state.get("working_memory", {})
        processed_results = state.get("processed_results") # Optional details
        success_status = state.get("cycle_completed_successfully") # Optional status

        # Log the event in working memory
        memory_log_key = f"progress_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Added for logging context
        working_memory[memory_log_key] = {
            "status": "Success" if success_status else "Failed/Partial",
            "notification_sent": progress_notification,
        }
        agent_log.info("Logged progress review result in working memory", extra={"memory_key": memory_log_key, "status": "Success" if success_status else "Failed/Partial"}) # Added log

        # Prepare state update (Original Logic)
        next_state = {**state}
        next_state["messages"] = add_messages(state.get("messages", []), [AIMessage(content=f"<user>{progress_notification}</user>")])
        next_state["working_memory"] = working_memory
        next_state["final_report_and_notification"] = None
        next_state["cycle_completed_successfully"] = None
        next_state["processed_results"] = None
        next_state["user_request_context"] = None
        next_state["current_agent"] = "end_conversation"

        # logger.info("Coordinator: Processed progress subgraph return. Routing to end_conversation.") # Old logger
        duration = time.time() - start_time
        agent_log.info("Processed progress subgraph return. Routing to end_conversation.", extra={"next_agent": "end_conversation", "duration_seconds": round(duration, 2)}) # New logger
        return next_state

    # --- Check if returning from Planning Subgraph ---
    returned_hevy_results = state.get("hevy_results")
    if returned_hevy_results is not None:
        # logger.info("Coordinator: Detected return from planning_subgraph (hevy_results is present).") # Old logger
        agent_log.info("Detected return from planning_subgraph (hevy_results is present).") # New logger

        working_memory = state.get("working_memory", {})
        generated_routines_key = f"generated_hevy_routines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        working_memory[generated_routines_key] = returned_hevy_results
        # logger.info(f"Stored generated routines in working_memory under key: {generated_routines_key}") # Old logger
        agent_log.info(f"Stored generated routines in working_memory.", extra={"memory_key": generated_routines_key}) # New logger

        payloads_json = json.dumps(returned_hevy_results, indent=2)
        summary_prompt_filled = summarize_routine_prompt.format(hevy_results_json=payloads_json)
        agent_log.debug("Prepared prompt for routine summarization.") # Added log

        user_facing_summary = "An error occurred while summarizing the generated plan." # Default error message
        try:
            # logger.info("Coordinator: Generating user-facing summary of routines...") # Old logger
            agent_log.info("Invoking LLM for routine summarization.") # New logger
            llm_start_time = time.time()
            # --- Original Logic: LLM Call ---
            summary_response = await llm.ainvoke([SystemMessage(content=summary_prompt_filled)])
            # --- End Original Logic ---
            llm_duration = time.time() - llm_start_time
            agent_log.info("LLM invocation for summary completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

            user_facing_summary = summary_response.content.strip()
            # logger.info("Coordinator: Summary generated successfully.") # Old logger
            agent_log.info("Summary generated successfully.") # New logger

        except Exception as summary_err:
            # logger.error(f"Coordinator: Error generating routine summary: {summary_err}", exc_info=True) # Old logger
            agent_log.error(f"Error generating routine summary", exc_info=True) # New logger

        # Prepare the state update (Original Logic)
        next_state = {**state}
        next_state["messages"] = add_messages(state.get("messages", []), [AIMessage(content=f"{user_facing_summary}")])
        next_state["working_memory"] = working_memory
        next_state["hevy_results"] = None
        next_state["current_agent"] = "end_conversation"

        # logger.info("Coordinator Agent - Output State (After Planning Summary): Routing back to end-conversation to show results to user.") # Old logger
        duration = time.time() - start_time
        agent_log.info("Processed planning subgraph return. Routing to end_conversation.", extra={"next_agent": "end_conversation", "duration_seconds": round(duration, 2)}) # New logger
        return next_state

    # --- Check if returning from Deep Research ---
    final_report = state.get("final_report")
    if final_report is not None:
        # logger.info("Coordinator: Received final report from deep_research subgraph.") # Old logger
        agent_log.info("Detected return from deep_research subgraph (final_report is present).") # New logger
        working_memory = state.get("working_memory", {})
        research_findings = working_memory.get("research_findings", {})
        research_findings["report"] = final_report # Original logic implicitly updates
        research_findings["report_timestamp"] = datetime.now().isoformat()
        working_memory["research_findings"] = research_findings
        working_memory["research_needs"] = None # Clear needs after report received
        agent_log.info("Integrated research report into working memory.") # Added log

        # Prepare state update (Original Logic)
        next_state = {**state}
        ai_message = AIMessage(content="<user>Okay, I've completed the research based on your profile and needs. Now I can create a plan, or would you like to discuss the findings?</user>")
        next_state["messages"] = add_messages(state.get("messages", []), [ai_message])
        next_state["working_memory"] = working_memory
        next_state["final_report"] = None
        next_state["research_topic"] = None
        next_state["user_profile_str"] = None
        next_state["current_agent"] = "planning_agent"

        # logger.info("Coordinator: Integrated report, cleared subgraph keys. Routing to planning_agent.") # Old logger
        duration = time.time() - start_time
        agent_log.info("Processed deep_research subgraph return. Routing to planning_agent.", extra={"next_agent": "planning_agent", "duration_seconds": round(duration, 2)}) # New logger
        return next_state

    # logger.info("Coordinator: Detected nothing. Resuming Normal Flow.") # Old logger
    agent_log.info("No specific subgraph return detected. Proceeding with normal flow.") # New logger

    # =================== MEMORY MANAGEMENT SECTION ===================
    agent_log.debug("Starting memory management section.") # Added log
    recent_exchanges = []
    for msg in state["messages"][-10:]:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(msg, AIMessage):
                user_match = re.search(r'<user>(.*?)</user>', content, re.DOTALL)
                if user_match:
                    content = user_match.group(1).strip()
            recent_exchanges.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": content
            })
    agent_log.debug(f"Processed {len(recent_exchanges)} recent exchanges for working memory.") # Added log

    working_memory = state.get("working_memory", {})
    working_memory["recent_exchanges"] = recent_exchanges
    working_memory["last_updated"] = datetime.now().isoformat()

    memory = state.get("memory", {})
    memory_key = f"interaction_{datetime.now().isoformat()}"
    memory[memory_key] = {
        "agent_states": state.get("agent_state", {}),
        "current_agent": state.get("current_agent", "coordinator"),
        "user_intent": working_memory.get("current_user_intent", "unknown")
    }
    agent_log.debug("Logged current interaction details to main memory.", extra={"memory_key": memory_key}) # Added log

    memory_size = len(memory)
    if memory_size > 0 and memory_size % 10 == 0:
        # logger.info(f"Performing memory consolidation (memory size: {memory_size})") # Old logger
        agent_log.info(f"Triggering memory consolidation.", extra={"memory_size": memory_size}) # New logger
        try:
            memory_prompt = get_memory_consolidation_prompt()
            filled_prompt = memory_prompt.format(
                memory=json.dumps(memory),
                user_model=json.dumps(state.get("user_model", {})),
                working_memory=json.dumps(working_memory)
            )
            agent_log.debug("Invoking LLM for memory consolidation.") # Added log
            llm_start_time = time.time()
            # --- Original Logic: LLM Call ---
            memory_messages = [SystemMessage(content=filled_prompt)]
            memory_response = await llm.ainvoke(memory_messages)
            # --- End Original Logic ---
            llm_duration = time.time() - llm_start_time
            agent_log.info("LLM invocation for memory consolidation completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

            # logger.info(f"Memory consolidation: {memory_response.content}") # Old logger # Log potentially large/sensitive data
            agent_log.info(f"Memory consolidation processing completed.") # New logger (safer log message)

            working_memory["last_consolidation"] = {
                "timestamp": datetime.now().isoformat(),
                "result": memory_response.content # Store original result
            }
            agent_log.info("Stored memory consolidation result in working memory.") # Added log
        except Exception as e:
            # logger.error(f"Memory consolidation error: {str(e)}") # Old logger
            agent_log.error(f"Memory consolidation error", exc_info=True) # New logger
    # =================== END MEMORY MANAGEMENT SECTION ===================

    # Normal flow - use LLM for ALL routing decisions
    agent_log.debug("Preparing coordinator prompt for routing decision.") # Added log
    coordinator_prompt = get_coordinator_prompt()
    filled_prompt = coordinator_prompt.format(
        user_model=json.dumps(state.get("user_model", {})),
        fitness_plan=json.dumps(state.get("hevy_results", {})), # This might be None here in normal flow
        recent_exchanges=json.dumps(working_memory.get("recent_exchanges", [])),
        research_findings=json.dumps(working_memory.get("research_findings", {}))
    )

    agent_log.info("Invoking LLM for coordinator routing decision.") # Added log
    llm_start_time = time.time()
    # --- Original Logic: LLM Call ---
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    # --- End Original Logic ---
    llm_duration = time.time() - llm_start_time
    agent_log.info("LLM invocation for routing completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

    # Process the response (Original Logic)
    content = response.content
    user_response = ""
    internal_reasoning = content
    selected_agent = "coordinator"
    research_needs = None

    agent_log.debug("Parsing LLM response for user message, agent tag, and research needs.") # Added log
    user_match = re.search(r'<user>(.*?)</user>', content, re.DOTALL)
    if user_match:
        user_response = user_match.group(1).strip()
        internal_reasoning = content.replace(f"<user>{user_response}</user>", "").strip()
        agent_log.debug("Extracted user-facing response.") # Added log
        selected_agent = "end_conversation"

    research_needs_match = re.search(r'<research_needs>(.*?)</research_needs>', internal_reasoning, re.DOTALL)
    if research_needs_match:
        research_needs = research_needs_match.group(1).strip()
        internal_reasoning = internal_reasoning.replace(f"<research_needs>{research_needs}</research_needs>", "").strip()
        agent_log.debug(f"Extracted research needs: {research_needs}") # Added log

    agent_tags = {
        "<Assessment>": "assessment", "<Research>": "deep_research", "<Planning>": "planning_agent",
        "<Progress_and_Adaptation>": "progress_adaptation_agent", "<Coach>": "coach_agent",
        "<User Modeler>": "user_modeler", "<Complete>": "end_conversation"
    }
    agent_selected_tag = "None" # For logging
    for tag, agent in agent_tags.items():
        if tag in internal_reasoning:
            selected_agent = agent
            internal_reasoning = internal_reasoning.replace(tag, "").strip()
            agent_selected_tag = tag # For logging
            # logger.info(f'Coordinator agent decided on the agent: {selected_agent}') # Old logger
            break
    agent_log.info(f'Coordinator LLM selected next agent.', extra={"selected_agent": selected_agent, "trigger_tag": agent_selected_tag}) # New logger

    # Update agent state (Original Logic)
    agent_state = state.get("agent_state", {})
    next_state_update = {}
    working_memory = state.get("working_memory", {}) # Re-get potentially updated WM

    # Prepare Subgraph Input (Original Logic + Logging)
    if selected_agent == "deep_research":
        topic = research_needs if research_needs else working_memory.get('research_needs')
        if not topic:
             user_goals = state.get("user_model", {}).get("goals")
             topic = f"Research fitness principles for achieving goals: {', '.join(user_goals)}" if user_goals else "General fitness planning based on user profile."
             # logger.warning(f"No specific research needs found, using derived/default topic: {topic}") # Old logger
             agent_log.warning(f"No specific research needs found, deriving topic.", extra={"derived_topic": topic}) # New logger

        # logger.info(f"Coordinator: Preparing state for deep_research subgraph. Topic: '{topic}'") # Old logger
        agent_log.info(f"Preparing state for deep_research subgraph.", extra={"research_topic": topic}) # New logger
        next_state_update["research_topic"] = topic
        next_state_update["user_profile_str"] = json.dumps(state.get("user_model", {}))
        working_memory["research_needs"] = topic
        next_state_update["final_report"] = None

    elif selected_agent == "planning_agent":
         # logger.info("Coordinator: Preparing state for planning_subgraph.") # Old logger
         agent_log.info("Preparing state for planning_subgraph.") # New logger
         next_state_update["hevy_results"] = None
         next_state_update["research_topic"] = None
         next_state_update["user_profile_str"] = None
         next_state_update["final_report"] = None

    elif selected_agent == "progress_adaptation_agent":
        # logger.info("Coordinator: Preparing state for progress_adaptation_subgraph.") # Old logger
        agent_log.info("Preparing state for progress_adaptation_subgraph.") # New logger
        last_message = state["messages"][-1]
        request_context = None
        context_source = "None" # For logging
        if isinstance(last_message, HumanMessage):
             request_context = last_message.content
             context_source = "Last Human Message"
             agent_log.info(f"Using last human message as user_request_context: {request_context}")
             # logger.debug(f"Using last human message as user_request_context: {request_context}") # Old logger
        elif working_memory.get("system_trigger_review"):
             request_context = working_memory.pop("system_trigger_review")
             context_source = "System Trigger Flag"
             agent_log.info(f"Using system trigger as user_request_context: {request_context}")
             # logger.debug(f"Using system trigger as user_request_context: {request_context}") # Old logger

        agent_log.debug(f"Determined user_request_context for progress subgraph.", extra={"context_source": context_source, "context_value": request_context}) # Added log
        next_state_update["user_request_context"] = request_context
        next_state_update["final_report"] = None
        next_state_update["hevy_results"] = None
        next_state_update["final_report_and_notification"] = None

    else:
        agent_log.debug("Clearing subgraph input/output keys for non-subgraph routing.") # Added log
        next_state_update["research_topic"] = None
        next_state_update["user_profile_str"] = None
        next_state_update["final_report"] = None
        next_state_update["hevy_results"] = None
        next_state_update["user_request_context"] = None
        next_state_update["final_report_and_notification"] = None
        next_state_update["cycle_completed_successfully"] = None
        next_state_update["processed_results"] = None

    # Safety check (Original Logic + Logging)
    if selected_agent != "user_modeler":
        user_model = state.get("user_model", {})
        required_fields = ["goals", "fitness_level", "available_equipment", "training_environment", "schedule"]
        missing_fields = [field for field in required_fields if field not in user_model or not user_model.get(field)]
        if missing_fields and selected_agent != "assessment":
            original_selection = selected_agent # For logging
            selected_agent = "assessment"
            # logger.info(f"Forcing assessment due to missing fields: {missing_fields}") # Old logger
            agent_log.info(f"Overriding routing: Forcing assessment due to missing fields.", extra={"missing_fields": missing_fields, "original_selection": original_selection, "final_selection": selected_agent}) # New logger

    # Update agent state needs_human_input (Original Logic + Logging)
    if selected_agent == "assessment":
        agent_state["needs_human_input"] = True
        agent_log.debug("Setting agent_state.needs_human_input to True (assessment selected).") # Added log
    else:
        agent_state["needs_human_input"] = False
        # agent_log.debug("Setting agent_state.needs_human_input to False.") # Can be noisy, optional

    # Final state update construction (Original Logic)
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=user_response)],
        "current_agent": selected_agent,
        "agent_state": agent_state,
        "memory": memory,
        "working_memory": working_memory | {
            "internal_reasoning": internal_reasoning, # Store reasoning from LLM
            "selected_agent": selected_agent # Store final selection
        },
        # Merge updates using .get() for safety
        "research_topic": next_state_update.get("research_topic"),
        "user_profile_str": next_state_update.get("user_profile_str"),
        "final_report": next_state_update.get("final_report"),
        "hevy_results": next_state_update.get("hevy_results"),
        "user_request_context": next_state_update.get("user_request_context"),
        "final_report_and_notification": next_state_update.get("final_report_and_notification"),
        "cycle_completed_successfully": next_state_update.get("cycle_completed_successfully"),
        "processed_results": next_state_update.get("processed_results"),
    }

    # logger.info(f"Coordinator Agent - Output State: {updated_state}") # Old logger (Potentially huge log)
    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting coordinator node", extra={"final_route_to": selected_agent, "duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return updated_state


# --- Coach agent with ELK Logging Added ---
async def coach_agent(state: AgentState) -> AgentState:
    """Provides motivation, adherence strategies, and behavioral coaching."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = state.get("configurable", {}).get("thread_id", "unknown_session")
    agent_log = get_agent_logger("coach_agent", session_id)
    agent_log.info("Entering coach_agent node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info(f"Coach Agent - Input State: {state}") # Removed old logger
    agent_log.debug("Preparing coach prompt.") # Added log
    coach_prompt = get_coach_prompt()

    # Prepare prompt context for logging (avoid logging full user model unless needed)
    prompt_context = {
        "has_user_model": "user_model" in state and bool(state["user_model"]),
        "has_progress_data": "progress_data" in state and bool(state["progress_data"]),
        "num_recent_exchanges": len(state.get("working_memory", {}).get("recent_exchanges", [])),
    }
    agent_log.debug("Formatting coach prompt", extra=prompt_context) # Added log

    filled_prompt = coach_prompt.format(
        user_profile=json.dumps(state.get("user_model", {})),
        progress_data=json.dumps(state.get("progress_data", {})),
        recent_exchanges=json.dumps(state.get("working_memory", {}).get("recent_exchanges", []))
    )

    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = None # Initialize

    # --- Logging around LLM call ---
    try:
        agent_log.info("Invoking LLM for coaching message.") # Added log
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke(messages)
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation for coaching completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

    except Exception as e:
        agent_log.error("Error during LLM call in coach_agent", exc_info=True)
        # Decide on fallback behavior if LLM fails
        # For now, we'll let it return None response content below
        response_content = "Sorry, I had trouble formulating a coaching message right now."
    else:
        # Get content only if LLM call succeeded
        response_content = response.content if response else "Sorry, couldn't generate a response."

    # --- Original Logic: Construct updated state ---
    # Ensure response_content is used
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=response_content)] # Use response_content
    }

    # logger.info(f"Coach Agent - Output State: {updated_state}") # Removed old logger (too verbose)
    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting coach_agent node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    # --- FIX: Return the updated state ---
    return updated_state
    # The original code had 'return' without a value, which implicitly returns None.
    # It should return the dictionary constructed above.
    # --- End FIX ---

# --- End Conversation node with ELK Logging Added ---
async def end_conversation(state: AgentState) -> AgentState:
    """Marks the conversation as complete in the state."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = state.get("configurable", {}).get("thread_id", "unknown_session")
    agent_log = get_agent_logger("end_conversation", session_id)
    agent_log.info("Entering end_conversation node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logging.info(f"End Conversation. Input State: {state}") # Removed old logging

    # Modify state directly (as done in original code) or create copy
    # Direct modification is fine if that's the pattern used elsewhere in the graph for this node
    current_agent_state = state.get("agent_state", {})
    new_agent_state = current_agent_state | {"status": "complete"}
    state["agent_state"] = new_agent_state
    state["conversation_complete"] = True

    agent_log.info("Marked conversation as complete in state.", extra={"conversation_complete": True, "agent_status": "complete"}) # Added log

    # logging.info(f"End Conversation. Output State: {state}") # Removed old logging (too verbose)
    # --- End Original Logic ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting end_conversation node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    # Return the modified state (as per original logic)
    return state


##################################### Deep Research Agent ######################################
# --- Helper function to get session_id from config ---
def _get_session_id_from_config(config: Optional[RunnableConfig]) -> str:
    """Safely extracts thread_id (session_id) from RunnableConfig."""
    if config and isinstance(config, dict) and "configurable" in config:
        return config["configurable"].get("thread_id", "unknown_subgraph_session")
    # Add handling if config is not a dict but has .get method (older Langchain?)
    # elif hasattr(config, 'get'):
    #     configurable = config.get('configurable', {})
    #     return configurable.get('thread_id', 'unknown_subgraph_session')
    return "unknown_subgraph_session_no_config"
# ---

# --- Plan Research Steps node with ELK Logging ---
async def plan_research_steps(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Uses LLM to break down the research topic into sub-questions.
    Initializes the internal state for the research loop, including limits.
    """
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_config(config)
    agent_log = get_agent_logger("deep_research.plan_steps", session_id)
    agent_log.info("Entering plan_research_steps node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Planning Steps ---") # Removed old logger
    topic = state['research_topic']
    user_profile = state['user_profile_str']
    agent_log.debug("Planning research steps.", extra={"topic": topic, "has_user_profile": bool(user_profile)}) # Added log

    
    prompt = get_plan_research_steps_prompt()

    sub_questions = [] # Initialize
    # --- Logging around LLM call and Parsing ---
    try:
        agent_log.info("Invoking LLM to plan sub-questions.") # Added log
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation for planning completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

        sub_questions_str = response.content.strip()
        # logger.debug(f"LLM response for planning: {sub_questions_str}") # Removed old logger
        agent_log.debug("Attempting to parse LLM response for sub-questions.") # Added log
        # --- Original Logic: Parsing ---
        sub_questions = json.loads(sub_questions_str)
        if not isinstance(sub_questions, list) or not all(isinstance(q, str) for q in sub_questions):
            raise ValueError("LLM did not return a valid JSON list of strings.")
        # --- End Original Logic ---
        # logger.info(f"Planned sub-questions: {sub_questions}") # Removed old logger
        agent_log.info(f"Successfully planned sub-questions.", extra={"num_sub_questions": len(sub_questions)}) # Added log

    except Exception as e:
        # logger.error(f"Error planning research steps: {e}. Using topic as single question.") # Removed old logger
        agent_log.error(f"Error planning research steps, falling back.", exc_info=True) # Added log
        sub_questions = [topic] # Fallback

    # --- Original Logic: Initialize state ---
    run_config = config.get("configurable", {}) if config else {}
    max_iterations = run_config.get("max_iterations", 5)
    max_queries_per_sub_q = run_config.get("max_queries_per_sub_question", 2)
    agent_log.info("Initialized research loop state.", extra={"max_iterations": max_iterations, "max_queries_per_sub_q": max_queries_per_sub_q}) # Added log

    # --- Construct return dictionary (Original Logic) ---
    return_state = {
        "sub_questions": sub_questions,
        "accumulated_findings": f"Initial research topic: {topic}\nUser Profile Summary: {user_profile}\n\nStarting research...\n",
        "iteration_count": 0,
        "current_sub_question_idx": 0,
        "reflections": [],
        "research_complete": False,
        "queries_this_sub_question": 0,
        "sub_question_complete_flag": False,
        "max_iterations": max_iterations,
        "max_queries_per_sub_question": max_queries_per_sub_q
    }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting plan_research_steps node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Generate RAG Query node with ELK Logging ---
async def generate_rag_query_v2(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generates RAG query for the current sub-question and increments counters.
    """
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_config(config)
    agent_log = get_agent_logger("deep_research.generate_query", session_id)
    agent_log.info("Entering generate_rag_query_v2 node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Generating RAG Query (v2) ---") # Removed old logger
    new_iteration_count = state.get('iteration_count', 0) + 1
    queries_count_this_sub_q = state.get('queries_this_sub_question', 0) + 1
    agent_log.debug("Incremented counters.", extra={"new_iteration_count": new_iteration_count, "queries_this_sub_q": queries_count_this_sub_q}) # Added log

    if not state.get('sub_questions'):
        # logger.warning("No sub_questions found, cannot generate query.") # Removed old logger
        agent_log.warning("No sub_questions found, cannot generate query.") # Added log
        return {"current_rag_query": None, "iteration_count": new_iteration_count, "queries_this_sub_question": queries_count_this_sub_q}

    current_idx = state['current_sub_question_idx']
    sub_questions = state.get('sub_questions', []) # Added for safety
    if current_idx >= len(sub_questions):
         # logger.info("All sub-questions processed based on index.") # Removed old logger
         agent_log.warning("All sub-questions processed based on index, cannot generate query.") # Added log
         return {"current_rag_query": None, "research_complete": True, "iteration_count": new_iteration_count, "queries_this_sub_question": queries_count_this_sub_q}

    current_sub_question = sub_questions[current_idx]
    findings = state['accumulated_findings']
    reflections_str = "\n".join(state.get('reflections', []))

    prompt = get_generate_rag_query_v2_prompt()
    query_to_run = None
    # --- Logging around LLM call ---
    try:
        agent_log.info("Invoking LLM to generate RAG query.", extra={"current_sub_question_idx": current_idx}) # Added log
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation for query generation completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

        query = response.content.strip()
        if not query:
            raise ValueError("LLM returned an empty query.")
        # logger.info(f"Generated RAG query: {query}") # Removed old logger
        agent_log.info(f"Successfully generated RAG query.", extra={"query_generated": query}) # Added log
        query_to_run = query
    except Exception as e:
        # logger.error(f"Error generating RAG query: {e}. Using fallback.") # Removed old logger
        agent_log.error(f"Error generating RAG query, using fallback.", exc_info=True) # Added log
        query_to_run = f"Details about {current_sub_question}" # Fallback
    # --- End Logging ---

    # --- Construct return dictionary (Original Logic) ---
    return_state = {
        "current_rag_query": query_to_run,
        "iteration_count": new_iteration_count,
        "queries_this_sub_question": queries_count_this_sub_q
        }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting generate_rag_query_v2 node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Execute RAG node with ELK Logging ---
async def execute_rag_direct(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Directly executes the retrieve_data function using the current_rag_query.
    """
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_config(config)
    agent_log = get_agent_logger("deep_research.execute_rag", session_id)
    agent_log.info("Entering execute_rag_direct node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Executing RAG Query ---") # Removed old logger
    query = state.get('current_rag_query')
    rag_results = None # Initialize

    if not query:
        # logger.warning("No RAG query provided in state. Skipping execution.") # Removed old logger
        agent_log.warning("No RAG query provided in state. Skipping execution.") # Added log
        rag_results = "No query was provided."
    else:
        # --- Logging around RAG call ---
        try:
            # logger.debug(f"Calling retrieve_data with query: {query}") # Removed old logger
            agent_log.info(f"Executing RAG query.", extra={"query": query}) # Added log
            rag_start_time = time.time()
            # --- Original Logic: RAG Call ---
            # Assume retrieve_data is an async function imported correctly
            rag_output = await retrieve_data(query=query)
            # --- End Original Logic ---
            rag_duration = time.time() - rag_start_time
            agent_log.info(f"RAG query execution completed.", extra={"duration_seconds": round(rag_duration, 2)}) # Added log

            # logger.info(f"RAG execution successful. Result length: {len(rag_output)}") # Removed old logger
            if not rag_output or rag_output.strip() == "":
                # logger.warning(f"RAG query '{query}' returned empty results.") # Removed old logger
                agent_log.warning(f"RAG query returned empty results.", extra={"query": query}) # Added log
                rag_results = f"No information found in knowledge base for query: '{query}'"
            else:
                agent_log.info(f"RAG execution successful.", extra={"result_length": len(rag_output), "query": query}) # Added log
                rag_results = rag_output
        except Exception as e:
            # logger.error(f"Error executing RAG query '{query}': {e}", exc_info=True) # Removed old logger
            agent_log.error(f"Error executing RAG query", exc_info=True, extra={"query": query}) # Added log
            rag_results = f"Error retrieving information for query '{query}': {str(e)}"
        # --- End Logging ---

    # --- Construct return dictionary (Original Logic) ---
    return_state = {"rag_results": rag_results}
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting execute_rag_direct node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Synthesize RAG Results node with ELK Logging ---
async def synthesize_rag_results(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Uses LLM to integrate the latest RAG results into the accumulated findings.
    """
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_config(config)
    agent_log = get_agent_logger("deep_research.synthesize", session_id)
    agent_log.info("Entering synthesize_rag_results node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Synthesizing RAG Results ---") # Removed old logger
    rag_results = state.get('rag_results')
    findings = state['accumulated_findings'] # Get existing findings
    query_used = state.get('current_rag_query', 'N/A') # Get query for context
    updated_findings = findings # Default to existing if synthesis skipped

    # Check if synthesis should be skipped
    skip_synthesis = False
    if not rag_results or "No query was provided" in rag_results or "No information found" in rag_results or "Error retrieving information" in rag_results:
        # logger.warning(f"Skipping synthesis due to missing or problematic RAG results: {rag_results}") # Removed old logger
        agent_log.warning(f"Skipping synthesis due to missing or problematic RAG results.", extra={"rag_result_status": rag_results, "query_used": query_used}) # Added log
        error_marker = f"\n\n[Skipped synthesis for query: '{query_used}' due to RAG result: {rag_results}]\n"
        updated_findings = findings + error_marker
        skip_synthesis = True

    if not skip_synthesis:
        current_idx = state['current_sub_question_idx']
        current_sub_question = state['sub_questions'][current_idx] if state.get('sub_questions') and current_idx < len(state['sub_questions']) else "the current topic"

        prompt = get_synthesize_rag_results_prompt()
        # --- Logging around LLM call ---
        try:
            agent_log.info("Invoking LLM to synthesize RAG results.", extra={"current_sub_question_idx": current_idx, "query_used": query_used}) # Added log
            llm_start_time = time.time()
            # --- Original Logic: LLM Call ---
            response = await llm.ainvoke([SystemMessage(content=prompt)])
            # --- End Original Logic ---
            llm_duration = time.time() - llm_start_time
            agent_log.info("LLM invocation for synthesis completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

            synthesized_text = response.content.strip()
            # logger.debug(f"Synthesized findings length: {len(updated_findings)}") # Removed old logger # Used wrong variable
            agent_log.info("Successfully synthesized RAG results.", extra={"synthesized_length": len(synthesized_text)}) # Added log
            query_marker = f"\n\n[Synthesized info based on RAG query: '{query_used}']\n"
            updated_findings = synthesized_text + query_marker
        except Exception as e:
            # logger.error(f"Error synthesizing RAG results: {e}") # Removed old logger
            agent_log.error(f"Error synthesizing RAG results", exc_info=True, extra={"query_used": query_used}) # Added log
            error_marker = f"\n\n[Error synthesizing results for query: '{query_used}'. RAG Results length: {len(rag_results)}. Error: {e}]\n" # Avoid logging full RAG results
            updated_findings = findings + error_marker
        # --- End Logging ---

    # --- Construct return dictionary (Original Logic) ---
    return_state = {"accumulated_findings": updated_findings}
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting synthesize_rag_results node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Reflect on Progress node with ELK Logging ---
async def reflect_on_progress_v2(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Reflects on progress, decides if sub-question is complete (naturally or by force),
    and updates the index and per-question query count accordingly.
    """
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_config(config)
    agent_log = get_agent_logger("deep_research.reflect", session_id)
    agent_log.info("Entering reflect_on_progress_v2 node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Reflecting on Progress (v2) ---") # Removed old logger
    current_idx = state['current_sub_question_idx']
    sub_questions = state.get('sub_questions', [])

    if not sub_questions or current_idx >= len(sub_questions):
        # logger.warning("Cannot reflect, index out of bounds or no sub-questions.") # Removed old logger
        agent_log.warning("Cannot reflect, index out of bounds or no sub-questions.", extra={"current_idx": current_idx, "num_sub_questions": len(sub_questions)}) # Added log
        return {
            "reflections": state.get('reflections', []) + ["Reflection skipped: Index out of bounds."], # Append to existing
            "sub_question_complete_flag": True,
            "current_sub_question_idx": current_idx,
            "queries_this_sub_question": 0
            }

    current_sub_question = sub_questions[current_idx]
    findings = state['accumulated_findings']
    queries_this_sub_q = state.get('queries_this_sub_question', 0)
    max_queries_sub_q = state.get('max_queries_per_sub_question', 2)
    force_next_question = queries_this_sub_q >= max_queries_sub_q
    agent_log.debug("Checking reflection conditions.", extra={ # Added log
        "current_sub_question_idx": current_idx,
        "current_sub_question": current_sub_question,
        "queries_this_sub_q": queries_this_sub_q,
        "max_queries_sub_q": max_queries_sub_q,
        "force_next_question": force_next_question
    })

    prompt = get_reflect_on_progress_v2_prompt()
    reflection_text = "Error during reflection." # Default
    is_complete = True # Default to complete on error
    next_idx = current_idx + 1
    queries_reset = 0
    new_reflections = state.get('reflections', []) # Get existing reflections

    # --- Logging around LLM call ---
    try:
        agent_log.info("Invoking LLM to reflect on progress.") # Added log
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation for reflection completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

        reflection_text = response.content.strip()
        # logger.info(f"Reflection on '{current_sub_question}':\n{reflection_text}") # Removed old logger # Potentially large
        agent_log.info(f"Received reflection on sub-question.", extra={"current_sub_question_idx": current_idx}) # Added log

        natural_completion = "CONCLUSION: SUB_QUESTION_COMPLETE" in reflection_text.upper()
        is_complete = natural_completion or force_next_question

        if force_next_question and not natural_completion:
             # logger.warning(f"Forcing completion of sub-question {current_idx + 1} due to query limit ({max_queries_sub_q}).") # Removed old logger
             agent_log.warning(f"Forcing completion of sub-question due to query limit.", extra={"sub_question_idx": current_idx, "query_limit": max_queries_sub_q}) # Added log
             reflection_text += "\n\n[Note: Sub-question concluded due to query limit.]"

        next_idx = current_idx + 1 if is_complete else current_idx
        queries_reset = 0 if is_complete else queries_this_sub_q

        agent_log.info("Reflection processing complete.", extra={ # Added log
            "natural_completion": natural_completion,
            "force_completion": force_next_question,
            "final_is_complete": is_complete,
            "next_sub_question_idx": next_idx
        })
        new_reflections.append(reflection_text) # Append new reflection

    except Exception as e:
        # logger.error(f"Error reflecting on progress: {e}") # Removed old logger
        agent_log.error(f"Error reflecting on progress, forcing completion.", exc_info=True) # Added log
        # Fallback logic from original code:
        is_complete = True
        next_idx = current_idx + 1
        queries_reset = 0
        new_reflections.append(f"Reflection Error: {e}. Assuming sub-question complete.") # Append error reflection
    # --- End Logging ---

    # --- Construct return dictionary (Original Logic) ---
    return_state = {
        "reflections": new_reflections, # Return updated list
        "sub_question_complete_flag": is_complete,
        "current_sub_question_idx": next_idx,
        "queries_this_sub_question": queries_reset
        }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting reflect_on_progress_v2 node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Finalize Report node with ELK Logging ---
async def finalize_research_report(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Uses LLM to generate the final research report based on all findings.
    """
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_config(config)
    agent_log = get_agent_logger("deep_research.finalize_report", session_id)
    agent_log.info("Entering finalize_research_report node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Deep Research: Finalizing Report ---") # Removed old logger
    topic = state['research_topic']
    findings = state['accumulated_findings']
    reflections = state.get('reflections', [])
    sub_questions = state.get('sub_questions', [])
    agent_log.debug("Preparing final report prompt.", extra={"topic": topic, "num_sub_questions": len(sub_questions), "num_reflections": len(reflections)}) # Added log

    prompt = get_finalize_research_report_prompt()
    final_report = "Error generating final report." # Default
    # --- Logging around LLM call ---
    try:
        agent_log.info("Invoking LLM to generate final research report.") # Added log
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation for final report completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

        final_report = response.content.strip()
        # logger.info("Generated final research report.") # Removed old logger
        agent_log.info("Successfully generated final research report.", extra={"report_length": len(final_report)}) # Added log

    except Exception as e:
        # logger.error(f"Error generating final report: {e}") # Removed old logger
        agent_log.error(f"Error generating final report", exc_info=True) # Added log
        final_report = f"Error generating report: {e}\n\nRaw Findings:\n{findings}" # Keep original fallback
    # --- End Logging ---

    # --- Construct return dictionary (Original Logic) ---
    return_state = {
        "final_report": final_report,
        "research_complete": True
        }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting finalize_research_report node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state



##################################### Routine Planner and Creation Agent ######################################

def _get_session_id_from_state(state: Union[StreamlinedRoutineState, Dict]) -> str:
    """Safely extracts thread_id (session_id) from state's configurable key."""
    if isinstance(state, dict) and "configurable" in state:
        return state["configurable"].get("thread_id", "unknown_planning_session")
    # Add fallback if state isn't a dict or doesn't have configurable
    return "unknown_planning_session_no_config"

async def structured_planning_node(state: StreamlinedRoutineState) -> StreamlinedRoutineState:
    """Generates the workout plan directly as a validated Pydantic object using with_structured_output."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("planning.structured_plan", session_id)
    agent_log.info("Entering structured_planning_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Executing Structured Planning Node (using with_structured_output) ---") # Removed old logger
    user_model_str = json.dumps(state.get("user_model", {}), indent=2)
    research_findings_str = json.dumps(state.get("working_memory", {}).get("research_findings", {}), indent=2)
    errors = state.get("errors", [])
    agent_log.debug("Prepared user model and research findings for prompt.", extra={"has_user_model": bool(user_model_str != "{}"), "has_research": bool(research_findings_str != "{}")}) # Added log

    PLANNING_PROMPT_TEMPLATE = get_structured_planning_prompt()
    prompt = PromptTemplate(
        template=PLANNING_PROMPT_TEMPLATE,
        input_variables=["user_profile", "research_findings"],
    )
    prompt_str = prompt.format(user_profile=user_model_str, research_findings=research_findings_str)
    agent_log.debug("Formatted planning prompt.") # Added log

    current_messages_for_llm = [SystemMessage(content=prompt_str)]
    parsed_output_container = None # Initialize
    return_state = {**state} # Start preparing return state

    # --- Logging around LLM call and Structuring ---
    try:
        agent_log.info("Invoking LLM with structured output.") # Added log
        # Ensure llm is defined and has with_structured_output method
        if not hasattr(llm, 'with_structured_output'):
             raise AttributeError("The 'llm' object does not have the 'with_structured_output' method.")

        structured_llm = llm.with_structured_output(PlannerOutputContainer)

        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        parsed_output_container = await structured_llm.ainvoke(current_messages_for_llm)
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation with structured output completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log

        if isinstance(parsed_output_container, PlannerOutputContainer):
            planner_routines = parsed_output_container.routines
            # logger.info(f"Successfully received structured output with {len(planner_routines)} routine(s) via with_structured_output.") # Removed old logger
            agent_log.info(f"Successfully parsed structured output.", extra={"num_routines": len(planner_routines)}) # Added log

            # --- Original Logic: Update state on success ---
            return_state["planner_structured_output"] = planner_routines
            return_state["errors"] = errors
        else:
            # logger.error(f"llm.with_structured_output returned unexpected type: {type(parsed_output_container)}") # Removed old logger
            agent_log.error(f"LLM with_structured_output returned unexpected type.", extra={"return_type": str(type(parsed_output_container))}) # Added log
            # --- Original Logic: Update state on type error ---
            return_state["planner_structured_output"] = None
            return_state["errors"] = errors + ["Structured Planning Node Failed: Unexpected return type from with_structured_output"]

    except Exception as e:
        # logger.error(f"Error using llm.with_structured_output: {e}", exc_info=True) # Removed old logger
        agent_log.error(f"Error using llm.with_structured_output", exc_info=True) # Added log
        error_detail = str(e)
        if "model_json_schema" in error_detail or "schema_json" in error_detail:
             error_detail += " (Possible Pydantic V1/V2 incompatibility with with_structured_output. Consider migrating models to standard Pydantic V2.)"
             agent_log.warning("Potential Pydantic incompatibility detected with with_structured_output.") # Added log
        # --- Original Logic: Update state on general error ---
        return_state["planner_structured_output"] = None
        return_state["errors"] = errors + [f"Structured Planning Node Failed (with_structured_output): {error_detail}"]
    # --- End Logging ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting structured_planning_node", extra={"duration_seconds": round(duration, 2), "num_errors": len(return_state.get('errors', []))})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Format and Lookup Node with ELK Logging ---
async def format_and_lookup_node(state: StreamlinedRoutineState) -> StreamlinedRoutineState:
    """Looks up exercise IDs and formats routines for the Hevy API tool."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("planning.format_lookup", session_id)
    agent_log.info("Entering format_and_lookup_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Executing Format & Lookup Node ---") # Removed old logger
    planner_routines: Optional[List[PlannerRoutineCreate]] = state.get("planner_structured_output")
    errors = state.get("errors", [])
    hevy_payloads = []
    routine_errors = [] # Local errors for this node
    processed_routine_count = 0
    skipped_exercises_details = [] # Store more details

    if not planner_routines:
        # logger.warning("No structured planner output found to format.") # Removed old logger
        agent_log.warning("No structured planner output found to format. Skipping.") # Added log
        if not state.get("errors"): # Avoid duplicate generic error
             errors.append("Format/Lookup Node Error: No structured plan available from planner.")
        # --- Logging Exit ---
        duration = time.time() - start_time
        agent_log.info(f"Exiting format_and_lookup_node early (no input)", extra={"duration_seconds": round(duration, 2)})
        # --- End Logging Exit ---
        return {**state, "hevy_payloads": [], "errors": errors}

    agent_log.info(f"Processing {len(planner_routines)} routines from planner output.") # Added log

    for routine_idx, planner_routine in enumerate(planner_routines):
        # logger.info(f"Formatting routine: '{planner_routine.title}'") # Removed old logger
        agent_log.info(f"Formatting routine #{routine_idx + 1}: '{planner_routine.title}'") # Added log
        hevy_exercises = []
        superset_mapping = {}
        next_superset_id_numeric = 0
        exercise_skipped_in_this_routine = False

        for exercise_idx, planner_ex in enumerate(planner_routine.exercises):
            exercise_name = planner_ex.exercise_name
            agent_log.debug(f"Processing exercise #{exercise_idx + 1}: '{exercise_name}' in routine '{planner_routine.title}'") # Added log

            template_info = None
            # --- Logging around fuzzy lookup ---
            try:
                lookup_start_time = time.time()
                # --- Original Logic: Fuzzy Lookup ---
                template_info = await get_exercise_template_by_title_fuzzy(exercise_name)
                # --- End Original Logic ---
                lookup_duration = time.time() - lookup_start_time
                agent_log.debug(f"Fuzzy lookup completed.", extra={"duration_seconds": round(lookup_duration, 2), "input_name": exercise_name, "found": bool(template_info)}) # Added log
            except Exception as lookup_err:
                 agent_log.error(f"Error during fuzzy lookup for '{exercise_name}'", exc_info=True) # Added log
                 # template_info remains None
            # --- End Logging ---

            if not template_info or not template_info.get("id"):
                err_msg = f"Exercise '{exercise_name}' in routine '{planner_routine.title}' could not be matched to a Hevy exercise template. Skipping exercise."
                # logger.warning(err_msg) # Removed old logger
                agent_log.warning(err_msg) # Added log
                routine_errors.append(err_msg)
                skipped_exercises_details.append({"input_name": exercise_name, "routine_title": planner_routine.title, "reason": "No match found"})
                exercise_skipped_in_this_routine = True
                continue

            exercise_template_id = template_info["id"]
            matched_title = template_info["title"]
            # logger.debug(f"Matched '{exercise_name}' to '{matched_title}' (ID: {exercise_template_id})") # Removed old logger
            agent_log.info(f"Matched exercise.", extra={"input_name": exercise_name, "matched_title": matched_title, "hevy_id": exercise_template_id}) # Added log

            # --- Handle Sets ---
            hevy_sets = []
            valid_sets_found = False
            for set_idx, planner_set in enumerate(planner_ex.sets):
                 # Original logic for creating hevy_sets
                 weight = planner_set.weight_kg if planner_set.weight_kg is not None else 0
                 reps = planner_set.reps
                 set_type = planner_set.type if planner_set.type is not None else "normal"
                 hevy_sets.append(SetRoutineCreate(
                     type=set_type, weight_kg=weight, reps=reps,
                     distance_meters=planner_set.distance_meters, duration_seconds=planner_set.duration_seconds
                 ))
                 valid_sets_found = True # Mark if at least one set object is created
            agent_log.debug(f"Processed {len(hevy_sets)} sets for exercise '{matched_title}'.") # Added log

            if not valid_sets_found and planner_ex.sets: # Check if list was conceptually defined but yielded no valid sets
                 # logger.warning(f"No valid sets created for exercise '{matched_title}' although sets were defined conceptually. Skipping exercise.") # Removed old logger
                 agent_log.warning(f"No valid sets created for exercise '{matched_title}' although sets were conceptually defined. Skipping exercise.") # Added log
                 skipped_exercises_details.append({"input_name": exercise_name, "routine_title": planner_routine.title, "reason": "No valid sets created"})
                 exercise_skipped_in_this_routine = True
                 continue

            # --- Handle Supersets ---
            hevy_superset_id = None
            conceptual_superset_group = planner_ex.superset_id
            if conceptual_superset_group is not None and str(conceptual_superset_group).strip():
                group_key = str(conceptual_superset_group).strip()
                if group_key not in superset_mapping:
                    superset_mapping[group_key] = str(next_superset_id_numeric)
                    next_superset_id_numeric += 1
                hevy_superset_id = superset_mapping[group_key]
                # logger.debug(f"Assigning Hevy superset_id '{hevy_superset_id}' for planner group '{group_key}'") # Removed old logger
                agent_log.debug(f"Assigned superset ID.", extra={"planner_group": group_key, "hevy_superset_id": hevy_superset_id}) # Added log

            # --- Handle Notes and Rest ---
            exercise_notes = planner_ex.notes or matched_title # Default notes to matched title
            rest_time = planner_ex.rest_seconds if planner_ex.rest_seconds is not None else 60 # Default rest
            agent_log.debug("Handled exercise notes and rest time.", extra={"notes_source": "Planner" if planner_ex.notes else "Default (Matched Title)", "rest_seconds": rest_time}) # Added log

            # --- Create Hevy Exercise Object ---
            hevy_exercises.append(ExerciseRoutineCreate(
                exercise_template_id=exercise_template_id, superset_id=hevy_superset_id,
                rest_seconds=rest_time, notes=exercise_notes, sets=hevy_sets
            ))

        # --- Construct Final Routine Payload ---
        if hevy_exercises:
            final_hevy_routine = RoutineCreate(
                title=planner_routine.title,
                notes=planner_routine.notes or "",
                exercises=hevy_exercises
            )
            # Use model_dump for Pydantic V2
            api_payload = HevyRoutineApiPayload(routine_data=final_hevy_routine).model_dump(exclude_none=True)
            hevy_payloads.append(api_payload)
            processed_routine_count += 1
            # logger.info(f"Successfully formatted routine '{planner_routine.title}' for Hevy API.") # Removed old logger
            agent_log.info(f"Successfully formatted routine '{planner_routine.title}' for Hevy API.", extra={"num_exercises": len(hevy_exercises)}) # Added log
        elif planner_routine.exercises and not exercise_skipped_in_this_routine: # Avoid logging skip if all exercises were individually skipped
             err_msg = f"Routine '{planner_routine.title}' had exercises defined but none resulted in a valid Hevy exercise after formatting. Skipping routine export."
             # logger.warning(err_msg) # Removed old logger
             agent_log.warning(err_msg) # Added log
             routine_errors.append(err_msg)
        elif not planner_routine.exercises:
             agent_log.info(f"Routine '{planner_routine.title}' had no exercises defined by the planner.") # Added log


    final_errors = errors + routine_errors
    agent_log.info(f"Formatting complete.", extra={"routines_processed": processed_routine_count, "payloads_created": len(hevy_payloads), "total_errors": len(final_errors), "skipped_exercise_count": len(skipped_exercises_details)}) # Added log

    # Construct return state (Original Logic)
    return_state = {
        **state,
        "hevy_payloads": hevy_payloads,
        "errors": final_errors
    }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting format_and_lookup_node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Tool Execution Node with ELK Logging ---
async def tool_execution_node(state: StreamlinedRoutineState) -> StreamlinedRoutineState:
    """Executes the tool_create_routine for each formatted payload."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("planning.tool_exec", session_id)
    agent_log.info("Entering tool_execution_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Executing Tool Execution Node ---") # Removed old logger
    payloads = state.get("hevy_payloads", [])
    errors = state.get("errors", [])
    hevy_results = []

    if not payloads:
        # logger.info("No Hevy payloads to execute.") # Removed old logger
        agent_log.info("No Hevy payloads found to execute. Skipping.") # Added log
        # --- Logging Exit ---
        duration = time.time() - start_time
        agent_log.info(f"Exiting tool_execution_node early (no payloads)", extra={"duration_seconds": round(duration, 2)})
        # --- End Logging Exit ---
        return {**state, "hevy_results": []} # Keep original return structure

    # logger.info(f"Attempting to create {len(payloads)} routines in Hevy...") # Removed old logger
    agent_log.info(f"Attempting to create {len(payloads)} routines via Hevy tool.") # Added log

    tasks = []
    # Ensure tool_create_routine is accessible
    if 'tool_create_routine' not in globals() and 'tool_create_routine' not in locals():
         raise NameError("tool_create_routine is not defined or imported in the current scope.")
    hevy_tool = tool_create_routine

    for i, payload in enumerate(payloads):
        routine_title = payload.get("routine_data", {}).get("title", f"Routine {i+1}")
        # logger.info(f"Scheduling tool_create_routine call for: '{routine_title}' using ainvoke") # Removed old logger
        agent_log.debug(f"Scheduling Hevy tool call for routine: '{routine_title}'") # Added log
        # Ensure the tool has an ainvoke method
        if not hasattr(hevy_tool, 'ainvoke'):
             raise AttributeError(f"The tool object for '{routine_title}' does not have an 'ainvoke' method.")
        tasks.append(hevy_tool.ainvoke(input=payload))

    tool_outputs = []
    # --- Logging around asyncio.gather ---
    try:
        agent_log.info(f"Gathering results from {len(tasks)} tool calls.") # Added log
        gather_start_time = time.time()
        # --- Original Logic: Gather ---
        tool_outputs = await asyncio.gather(*tasks, return_exceptions=True)
        # --- End Original Logic ---
        gather_duration = time.time() - gather_start_time
        agent_log.info(f"Tool call gathering complete.", extra={"duration_seconds": round(gather_duration, 2), "num_results": len(tool_outputs)}) # Added log
    except Exception as gather_err:
         agent_log.error("Error during asyncio.gather for tool calls", exc_info=True) # Added log
         # Append a general error and potentially return early or process partial results
         errors.append(f"Tool Execution Error: Failed during asyncio.gather - {str(gather_err)}")
         # Decide how to handle partial results if needed, for now, continue to process any results obtained before error
    # --- End Logging ---

    success_count = 0
    failure_count = 0
    # --- Logging during result processing ---
    for i, result in enumerate(tool_outputs):
        # Ensure payload index exists before accessing
        routine_title = "Unknown Routine"
        if i < len(payloads):
             routine_title = payloads[i].get("routine_data", {}).get("title", f"Routine {i+1}")
        else:
             agent_log.warning(f"Result index {i} out of bounds for payloads list (length {len(payloads)}). Likely due to gather error.")

        agent_log.debug(f"Processing tool result for routine: '{routine_title}'") # Added log
        if isinstance(result, Exception):
            failure_count += 1
            # logger.error(f"Error during tool ainvoke for '{routine_title}': {result}", exc_info=result) # Removed old logger
            agent_log.error(f"Tool execution failed with exception for routine '{routine_title}'", exc_info=result) # Added log
            error_detail = str(result)
            if isinstance(result, NotImplementedError):
                 error_detail += " (Sync invocation attempted on async tool?)"
            elif hasattr(result, 'detail'): # Handle potential FastAPI/HTTP exceptions if tool makes web requests
                 error_detail = f"{getattr(result, 'status_code', 'Unknown Status')}: {result.detail}"

            errors.append(f"Tool Execution Failed for '{routine_title}': {error_detail}")
            hevy_results.append({"error": f"Tool Execution Failed: {error_detail}", "routine_title": routine_title, "status": "failed"})
        elif isinstance(result, dict) and result.get("error"):
            failure_count += 1
            error_msg = f"Hevy API Error for '{routine_title}': {result.get('error')} (Status: {result.get('status_code', 'N/A')})"
            # logger.error(error_msg) # Removed old logger
            agent_log.error(error_msg) # Added log
            errors.append(error_msg)
            if "status" not in result: result["status"] = "failed" # Ensure status field exists
            hevy_results.append(result) # Append the error dict from the tool
        else:
            # Assume success if no exception and no error field in dict (or if result is not a dict, though less common for tools)
            success_count += 1
            # logger.info(f"Tool result for '{routine_title}': {result}") # Removed old logger # Potentially large output
            agent_log.info(f"Tool execution successful for routine: '{routine_title}'", extra={"result_type": str(type(result))}) # Added log
            hevy_results.append(result) # Append successful result

    # logger.info(result_summary) # Removed old logger
    agent_log.info(f"Tool execution processing complete.", extra={"success_count": success_count, "failure_count": failure_count}) # Added log

    # --- Construct return state (Original Logic) ---
    return_state = {
        **state,
        "hevy_results": hevy_results,
        "errors": errors
    }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting tool_execution_node", extra={"duration_seconds": round(duration, 2)})
    # --- End Logging Exit ---

    return return_state

##################################### Progress Analysis and Adaptation Agent ###################################



def _get_session_id_from_state(state: Union[ProgressAnalysisAdaptationStateV2, Dict]) -> str:
    """Safely extracts thread_id (session_id) from state's configurable key."""
    if isinstance(state, dict) and "configurable" in state:
        return state["configurable"].get("thread_id", "unknown_progress_session")
    # Add fallback if state isn't a dict or doesn't have configurable
    return "unknown_progress_session_no_config"
# ---

# --- Fetch All Routines Node with ELK Logging ---
async def fetch_all_routines_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Fetches all available routines from Hevy."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("progress_adapt.fetch_routines", session_id)
    agent_log.info("Entering fetch_all_routines_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Progress Cycle V2: Fetching All Routines ---") # Removed old logger
    all_routines = []
    page = 1
    page_size = 20
    process_error = None # Initialize local error

    # --- Logging around tool call loop ---
    try:
        while True:
            agent_log.debug(f"Attempting to fetch routines page {page}...") # Added log
            tool_start_time = time.time()
            # --- Original Logic: Tool Call ---
            result = await tool_fetch_routines.ainvoke({"page": page, "pageSize": page_size})
            # --- End Original Logic ---
            tool_duration = time.time() - tool_start_time
            agent_log.debug(f"Tool 'tool_fetch_routines' call completed.", extra={"page": page, "duration_seconds": round(tool_duration, 2)}) # Added log

            if isinstance(result, dict) and result.get("routines"):
                fetched_page = result["routines"]
                all_routines.extend(fetched_page)
                agent_log.debug(f"Fetched {len(fetched_page)} routines from page {page}.") # Added log
                if len(fetched_page) < page_size:
                    agent_log.debug("Last page detected.") # Added log
                    break
                page += 1
            elif isinstance(result, dict) and result.get("error"):
                 error_msg = f"Hevy API error fetching routines page {page}: {result.get('error')}"
                 agent_log.error(error_msg) # Added log
                 raise Exception(error_msg) # Propagate to outer try/except
            else:
                 agent_log.debug(f"No routines found on page {page} or unexpected format received, stopping fetch.") # Added log
                 break # Assume end of routines

        # logger.info(f"Successfully fetched {len(all_routines)} routines.") # Removed old logger
        agent_log.info(f"Successfully fetched routines.", extra={"total_routines": len(all_routines), "pages_fetched": page}) # Added log
        return_state = {"fetched_routines_list": all_routines, "process_error": None}

    except Exception as e:
        # logger.error(f"Error fetching routines: {e}", exc_info=True) # Removed old logger
        agent_log.error(f"Error fetching routines", exc_info=True) # Added log
        process_error = f"Failed to fetch routines: {e}"
        return_state = {"fetched_routines_list": None, "process_error": process_error}
    # --- End Logging ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting fetch_all_routines_node", extra={"duration_seconds": round(duration, 2), "success": process_error is None})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Fetch Logs Node with ELK Logging ---
async def fetch_logs_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Fetches workout logs from Hevy."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("progress_adapt.fetch_logs", session_id)
    agent_log.info("Entering fetch_logs_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    if state.get("process_error"):
        agent_log.warning("Skipping fetch_logs_node due to previous error.", extra={"previous_error": state["process_error"]}) # Added log
        return {} # Skip on prior error

    # logger.info("--- Progress Cycle V2: Fetching Logs ---") # Removed old logger
    process_error = None # Initialize local error
    workout_logs_data = None # Initialize

    # --- Logging around tool calls ---
    try:
        # Fetch count
        agent_log.debug("Fetching workout count.") # Added log
        count_start_time = time.time()
        # --- Original Logic: Count Tool Call ---
        workout_count_data = await tool_get_workout_count.ainvoke({})
        # --- End Original Logic ---
        count_duration = time.time() - count_start_time
        total_workouts = workout_count_data.get('count', 'N/A')
        agent_log.debug("Workout count fetch completed.", extra={"duration_seconds": round(count_duration, 2), "total_workouts": total_workouts}) # Added log

        # Fetch logs
        page_size = 30
        # logger.info(f"Fetching latest {page_size} workout logs (total: {total_workouts})...") # Removed old logger
        agent_log.info(f"Fetching latest workout logs.", extra={"page_size": page_size, "total_workouts": total_workouts}) # Added log
        logs_start_time = time.time()
        # --- Original Logic: Logs Tool Call ---
        fetched_logs = await tool_fetch_workouts.ainvoke({"page": 1, "pageSize": page_size})
        # --- End Original Logic ---
        logs_duration = time.time() - logs_start_time
        agent_log.debug("Workout logs fetch completed.", extra={"duration_seconds": round(logs_duration, 2)}) # Added log

        workout_logs_data = fetched_logs.get('workouts', [])

        if not workout_logs_data:
            # logger.warning("No workout logs found.") # Removed old logger
            agent_log.warning("No workout logs found.") # Added log
            workout_logs_data = [] # Ensure it's an empty list
        else:
            # logger.info(f"Successfully fetched {len(workout_logs_data)} logs.") # Removed old logger
            agent_log.info(f"Successfully fetched workout logs.", extra={"logs_fetched_count": len(workout_logs_data)}) # Added log
        return_state = {"workout_logs": workout_logs_data, "process_error": None}

    except Exception as e:
        # logger.error(f"Error fetching workout logs: {e}", exc_info=True) # Removed old logger
        agent_log.error(f"Error fetching workout logs", exc_info=True) # Added log
        process_error = f"Warning: Failed to fetch logs: {e}" # Treat as warning
        return_state = {"workout_logs": None, "process_error": process_error}
    # --- End Logging ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting fetch_logs_node", extra={"duration_seconds": round(duration, 2), "success": process_error is None or "Warning" in process_error})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Identify Target Routines Node with ELK Logging ---
async def identify_target_routines_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Identifies target routines based on logs, request, and available routines."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("progress_adapt.identify_targets", session_id)
    agent_log.info("Entering identify_target_routines_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    process_error = state.get("process_error") # Get existing error status
    if process_error and "Failed to fetch routines" in process_error:
        # logger.error("Halting identification: Failed to fetch routines list.") # Removed old logger
        agent_log.error("Halting identification: Failed to fetch routines list.", extra={"previous_error": process_error}) # Added log
        return {"process_error": process_error} # Propagate fatal error

    # logger.info("--- Progress Cycle V2: Identifying Target Routines ---") # Removed old logger
    routines_list = state.get("fetched_routines_list")
    logs = state.get("workout_logs")
    user_model = state.get("user_model", {})
    user_request = state.get("user_request_context", "")
    identified_targets: List[IdentifiedRoutineTarget] = [] # Initialize

    if not routines_list:
        # logger.warning("No routines available to identify targets from.") # Removed old logger
        agent_log.warning("No routines available to identify targets from. Skipping identification.") # Added log
        return {"identified_targets": [], "process_error": process_error} # Return empty, keep existing error

    # --- Logging around prompt formatting and LLM call ---
    filled_prompt = ""
    output_str = ""
    try:
        id_prompt_template = get_routine_identification_prompt()
        agent_log.debug("Formatting identification prompt.") # Added log
        routines_limit = 10 # Limit context size
        logs_limit = 10
        routines_json = json.dumps(routines_list[:routines_limit], indent=2)
        logs_json = json.dumps(logs[:logs_limit] if logs else [], indent=2)

        filled_prompt = id_prompt_template.format(
            user_profile=json.dumps(user_model, indent=2),
            user_request_context=user_request,
            routines_list_json=routines_json,
            logs_list_json=logs_json
        )
        agent_log.debug("Identification prompt formatted successfully.") # Added log

    except Exception as e:
         # logger.error(f"Error formatting identification prompt: {e}") # Removed old logger
         agent_log.error("Error formatting identification prompt", exc_info=True) # Added log
         process_error = f"Internal error formatting identification prompt: {e}"
         # --- Logging Exit ---
         duration = time.time() - start_time
         agent_log.info(f"Exiting identify_target_routines_node due to formatting error", extra={"duration_seconds": round(duration, 2)})
         # --- End Logging Exit ---
         return {"process_error": process_error}

    try:
        agent_log.info("Invoking LLM for routine identification.") # Added log
        messages = [SystemMessage(content=filled_prompt)]
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke(messages)
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        agent_log.info("LLM invocation for identification completed.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log
        output_str = response.content.strip()

        agent_log.debug("Attempting to parse identification JSON from LLM.") # Added log
        # --- Original Logic: Parsing ---
        identified_targets_raw = json.loads(output_str)
        if not isinstance(identified_targets_raw, list):
            raise ValueError("LLM did not return a JSON list.")
        identified_targets = identified_targets_raw # Assume structure matches IdentifiedRoutineTarget
        # --- End Original Logic ---
        agent_log.debug("Successfully parsed identification JSON.") # Added log

        # logger.info(f"Identified {len(identified_targets)} target routine(s).") # Removed old logger
        valid_targets = [t for t in identified_targets if isinstance(t, dict) and isinstance(t.get("routine_data"), dict) and t["routine_data"].get("id")]
        num_identified = len(identified_targets)
        num_valid = len(valid_targets)
        agent_log.info(f"Routine identification complete.", extra={"identified_count": num_identified, "valid_count": num_valid}) # Added log

        if num_valid != num_identified:
             # logger.warning("Some identified targets had invalid structure and were filtered.") # Removed old logger
             agent_log.warning("Some identified targets had invalid structure and were filtered.", extra={"invalid_count": num_identified - num_valid}) # Added log

        return_state = {"identified_targets": valid_targets, "process_error": process_error} # Keep existing non-fatal error

    except json.JSONDecodeError as e:
         # logger.error(f"Error parsing identification JSON from LLM: {e}\nLLM Response:\n{output_str}", exc_info=True) # Removed old logger
         agent_log.error(f"Error parsing identification JSON from LLM.", exc_info=True, extra={"llm_response_snippet": output_str[:500]}) # Log snippet
         process_error = f"Failed to parse valid JSON targets from LLM: {e}"
         return_state = {"process_error": process_error} # Overwrite process_error as this is fatal for this step
    except Exception as e:
        # logger.error(f"Error during routine identification LLM call: {e}\nLLM Response:\n{output_str if 'output_str' in locals() else 'N/A'}", exc_info=True) # Removed old logger
        agent_log.error(f"Error during routine identification LLM call.", exc_info=True, extra={"llm_response_snippet": output_str[:500] if output_str else 'N/A'}) # Log snippet
        process_error = f"Error during routine identification: {e}"
        return_state = {"process_error": process_error} # Overwrite process_error
    # --- End Logging ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting identify_target_routines_node", extra={"duration_seconds": round(duration, 2), "success": "process_error" not in return_state or return_state["process_error"] is None})
    # --- End Logging Exit ---

    return return_state
# ---

# --- Process Targets Node with ELK Logging ---
async def process_targets_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Iterates through identified targets and attempts adaptation for each."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("progress_adapt.process_targets", session_id)
    agent_log.info("Entering process_targets_node")
    # --- End Logging Setup ---

    # --- Original Logic ---
    process_error = state.get("process_error") # Get existing error
    if process_error:
        agent_log.warning("Skipping process_targets_node due to previous critical error.", extra={"previous_error": process_error}) # Added log
        return {} # Skip if critical error happened before

    # logger.info("--- Progress Cycle V2: Processing Identified Targets ---") # Removed old logger
    targets = state.get("identified_targets", [])
    if not targets:
        # logger.info("No targets to process.") # Removed old logger
        agent_log.info("No targets identified to process.") # Added log
        return {"processed_results": []}

    agent_log.info(f"Starting processing for {len(targets)} identified target(s).") # Added log
    logs = state.get("workout_logs")
    user_model = state.get("user_model", {})
    user_request = state.get("user_request_context", "")
    processed_results: List[RoutineAdaptationResult] = []
    any_target_failed = False # Flag local errors

    # Get Prompt Templates Once
    analysis_prompt_template = get_analysis_v2_prompt()
    rag_query_prompt_template = get_targeted_rag_query_prompt()
    modification_prompt_template = get_routine_modification_v2_prompt()
    reasoning_prompt_template = get_reasoning_generation_prompt()
    analysis_parser = PydanticOutputParser(pydantic_object=AnalysisFindings)
    analysis_format_instructions = analysis_parser.get_format_instructions()

    # Loop Through Each Target
    for target_idx, target in enumerate(targets):
        routine_data = target["routine_data"]
        routine_id = routine_data["id"]
        original_title = routine_data.get("title", "Unknown Title")
        target_log_ctx = {"target_index": target_idx, "routine_id": routine_id, "routine_title": original_title} # Context for logs
        # logger.info(f"--- Processing Target: '{original_title}' (ID: {routine_id}) ---") # Removed old logger
        agent_log.info(f"Processing target #{target_idx + 1}: '{original_title}'", extra=target_log_ctx) # Added log

        current_result = RoutineAdaptationResult(
            routine_id=routine_id, original_title=original_title, status="Skipped (Error)",
            message="Processing started", updated_routine_data=None
        )

        try: # Outer try for unexpected errors in the loop logic for this target
            # 1. Analyze Report for Target
            # logger.debug(f"Analyzing target '{original_title}'...") # Removed old logger
            agent_log.info("Step 1: Analyzing target.", extra=target_log_ctx) # Added log
            analysis_findings: Optional[AnalysisFindings] = None
            analysis_error = None
            try:
                analysis_filled_prompt = analysis_prompt_template.format(
                    user_profile=json.dumps(user_model, indent=2),
                    target_routine_details=json.dumps(routine_data),
                    workout_logs=json.dumps(logs[:20] if logs else []),
                    format_instructions=analysis_format_instructions
                )
                analysis_llm_start = time.time()
                # --- Original Logic: Analysis LLM Call ---
                analysis_response = await llm.ainvoke([SystemMessage(content=analysis_filled_prompt)])
                # --- End Original Logic ---
                analysis_llm_duration = time.time() - analysis_llm_start
                agent_log.debug("Analysis LLM call completed.", extra={**target_log_ctx, "duration_seconds": round(analysis_llm_duration, 2)}) # Added log

                analysis_parse_start = time.time()
                # --- Original Logic: Analysis Parsing ---
                analysis_findings = analysis_parser.parse(analysis_response.content)
                # --- End Original Logic ---
                analysis_parse_duration = time.time() - analysis_parse_start
                # logger.debug(f"Analysis successful for '{original_title}'. Areas found: {len(analysis_findings.areas_for_potential_adjustment)}") # Removed old logger
                # logger.debug(f"Analysis for {original_title} are {analysis_findings}") # Removed old logger # Too verbose
                agent_log.info("Analysis parsing successful.", extra={**target_log_ctx, "adjustment_areas_found": len(analysis_findings.areas_for_potential_adjustment), "duration_seconds": round(analysis_parse_duration, 2)}) # Added log

            except Exception as e:
                analysis_error = f"Failed analysis step: {e}"
                # logger.error(analysis_error, exc_info=True) # Removed old logger
                agent_log.error("Analysis step failed.", exc_info=True, extra=target_log_ctx) # Added log
                current_result["message"] = analysis_error
                any_target_failed = True
                processed_results.append(current_result)
                continue # Skip to next target

            # Check if adjustments needed
            if not analysis_findings or not analysis_findings.areas_for_potential_adjustment:
                # logger.info(f"No adjustment areas identified for '{original_title}'. Skipping further steps.") # Removed old logger
                agent_log.info("No adjustment areas identified. Skipping further steps for this target.", extra=target_log_ctx) # Added log
                current_result["status"] = "Skipped (No Changes)"
                current_result["message"] = "Analysis completed, no modifications needed for this routine."
                processed_results.append(current_result)
                continue # Skip to next target

            # 2. Research Gaps for Target
            # logger.debug(f"Researching gaps for '{original_title}'...") # Removed old logger
            agent_log.info("Step 2: Researching gaps via RAG.", extra=target_log_ctx) # Added log
            adaptation_rag_results = {}
            rag_error = None
            try:
                user_profile_json = json.dumps(user_model, indent=2)
                for area_idx, area in enumerate(analysis_findings.areas_for_potential_adjustment):
                    area_log_ctx = {**target_log_ctx, "adjustment_area_index": area_idx, "adjustment_area": area} # Area context
                    try:
                        # Query Generation
                        agent_log.debug("Generating RAG query for area.", extra=area_log_ctx) # Added log
                        query_gen_prompt = rag_query_prompt_template.format(
                            user_profile=user_profile_json, area_for_adjustment=area,
                            previous_query="N/A", previous_result="N/A"
                        )
                        qgen_llm_start = time.time()
                        # --- Original Logic: Query Gen LLM Call ---
                        query_response = await llm.ainvoke([SystemMessage(content=query_gen_prompt)])
                        # --- End Original Logic ---
                        qgen_llm_duration = time.time() - qgen_llm_start
                        rag_query = query_response.content.strip()
                        agent_log.debug("RAG query generation LLM call complete.", extra={**area_log_ctx, "duration_seconds": round(qgen_llm_duration, 2), "generated_query": rag_query}) # Added log

                        # RAG Execution
                        if rag_query:
                            # logger.debug(f"Querying RAG with generated query for {area}: {query_response}") # Removed old logger # Used response obj
                            agent_log.debug(f"Executing RAG query.", extra={**area_log_ctx, "query": rag_query}) # Added log
                            rag_start_time = time.time()
                            # --- Original Logic: RAG Tool Call ---
                            rag_result = await retrieve_from_rag.ainvoke({"query": rag_query})
                            # --- End Original Logic ---
                            rag_duration = time.time() - rag_start_time
                            # logger.debug(f"Queried results for RAG with generated query for {area} with {query_response}: {rag_result}") # Removed old logger # Verbose
                            agent_log.debug("RAG execution complete.", extra={**area_log_ctx, "duration_seconds": round(rag_duration, 2), "result_length": len(rag_result or "")}) # Added log
                            adaptation_rag_results[area] = rag_result or "No specific info."
                        else:
                             agent_log.warning("LLM generated an empty RAG query for area.", extra=area_log_ctx) # Added log
                             adaptation_rag_results[area] = "Could not generate query."
                    except Exception as inner_e:
                         # logger.warning(f"Failed RAG for area '{area}': {inner_e}") # Removed old logger
                         agent_log.warning(f"Failed RAG step for area.", exc_info=inner_e, extra=area_log_ctx) # Added log
                         adaptation_rag_results[area] = f"Error: {inner_e}"
                         rag_error = f"Partial RAG failure on area '{area}'" # Flag partial failure
                agent_log.info("RAG querying finished for all areas.", extra={**target_log_ctx, "areas_queried": len(analysis_findings.areas_for_potential_adjustment)}) # Added log
            except Exception as e:
                 rag_error = f"Failed RAG step: {e}"
                 # logger.error(rag_error, exc_info=True) # Removed old logger
                 agent_log.error("RAG step failed.", exc_info=True, extra=target_log_ctx) # Added log

            if rag_error and not current_result["message"].startswith("Failed"):
                 current_result["message"] = rag_error # Report partial failure

            # 3. Generate Modifications for Target
            # logger.debug(f"Generating modifications for '{original_title}'...") # Removed old logger
            agent_log.info("Step 3: Generating modifications.", extra=target_log_ctx) # Added log
            proposed_mods_dict: Optional[Dict[str, Any]] = None
            modification_reasoning = "N/A"
            mod_output_str = "N/A"
            generation_error = None
            try:
                mod_filled_prompt = modification_prompt_template.format(
                    user_profile=json.dumps(user_model, indent=2),
                    user_request_context=user_request,
                    analysis_findings=json.dumps(analysis_findings.model_dump()), # Use model_dump for Pydantic V2
                    adaptation_rag_results=json.dumps(adaptation_rag_results),
                    current_routine_json=json.dumps(routine_data, indent=2)
                )
                mod_llm_start = time.time()
                # --- Original Logic: Modification LLM Call ---
                mod_response = await llm.ainvoke([SystemMessage(content=mod_filled_prompt)])
                # --- End Original Logic ---
                mod_llm_duration = time.time() - mod_llm_start
                mod_output_str = mod_response.content.strip()
                agent_log.debug("Modification generation LLM call complete.", extra={**target_log_ctx, "duration_seconds": round(mod_llm_duration, 2)}) # Added log

                # --- Original Logic: Parse Modifications ---
                proposed_mods_dict = json.loads(mod_output_str)
                # logger.debug(f"Proposed Modifications are {proposed_mods_dict}") # Removed old logger # Too verbose
                if not isinstance(proposed_mods_dict, dict) or 'exercises' not in proposed_mods_dict:
                    raise ValueError("LLM output was not a valid routine JSON dict.")
                agent_log.debug("Successfully parsed proposed modifications JSON.", extra=target_log_ctx) # Added log

                # Generate reasoning
                try:
                     agent_log.debug("Generating modification reasoning.", extra=target_log_ctx) # Added log
                     reasoning_filled_prompt = reasoning_prompt_template.format(
                         original_routine_snippet=json.dumps(routine_data.get('exercises', [])[:2], indent=2),
                         modified_routine_snippet=json.dumps(proposed_mods_dict.get('exercises', [])[:2], indent=2),
                         analysis_findings=json.dumps(analysis_findings.model_dump()),
                         adaptation_rag_results=json.dumps(adaptation_rag_results)
                     )
                     reason_llm_start = time.time()
                     # --- Original Logic: Reasoning LLM Call ---
                     reasoning_response = await llm.ainvoke([SystemMessage(content=reasoning_filled_prompt)])
                     # --- End Original Logic ---
                     reason_llm_duration = time.time() - reason_llm_start
                     modification_reasoning = reasoning_response.content.strip()
                     # logger.debug(f"Modification reasoning for proposed modifications are : {modification_reasoning}") # Removed old logger # Too verbose
                     agent_log.debug("Successfully generated modification reasoning.", extra={**target_log_ctx, "duration_seconds": round(reason_llm_duration, 2)}) # Added log
                except Exception as reason_e:
                     # logger.warning(f"Failed to generate reasoning: {reason_e}") # Removed old logger
                     agent_log.warning("Failed to generate modification reasoning.", exc_info=reason_e, extra=target_log_ctx) # Added log
                     modification_reasoning = "(Failed to generate reasoning)"

                # logger.debug(f"Modification generation successful for '{original_title}'.") # Removed old logger

            except Exception as e:
                generation_error = f"Failed modification step: {e}"
                # logger.error(f"{generation_error}\nLLM Response:\n{mod_output_str if 'mod_output_str' in locals() else 'N/A'}", exc_info=True) # Removed old logger
                agent_log.error(f"Modification generation/parsing failed.", exc_info=True, extra={**target_log_ctx, "llm_response_snippet": mod_output_str[:500] if mod_output_str else 'N/A'}) # Added log
                current_result["message"] = generation_error
                any_target_failed = True
                processed_results.append(current_result)
                continue # Skip to next target

            # 4. Validate and Lookup Exercises
            # logger.debug(f"Validating and looking up exercises for '{original_title}'...") # Removed old logger
            agent_log.info("Step 4: Validating and looking up exercises.", extra=target_log_ctx) # Added log
            validated_routine_dict = None
            validation_errors = []
            try:
                validate_start = time.time()
                # --- Original Logic: Validation Call ---
                validated_routine_dict, validation_errors = await validate_and_lookup_exercises(
                    proposed_mods_dict, # Input is the dict from LLM
                    original_title
                )
                # --- End Original Logic ---
                validate_duration = time.time() - validate_start
                agent_log.debug("Exercise validation/lookup complete.", extra={**target_log_ctx, "duration_seconds": round(validate_duration, 2), "validation_error_count": len(validation_errors)}) # Added log

                if validation_errors:
                    errors_string = "; ".join(validation_errors)
                    # logger.warning(f"Validation issues found for routine '{original_title}': {errors_string}") # Removed old logger
                    agent_log.warning(f"Validation issues found.", extra={**target_log_ctx, "validation_errors": errors_string}) # Added log
                    if validated_routine_dict is None: # Check if validation deemed it fatal
                        fatal_error_msg = f"Fatal validation error for '{original_title}', cannot update Hevy. Issues: {errors_string}"
                        # logger.error(fatal_error_msg) # Removed old logger
                        agent_log.error("Fatal validation error, cannot update Hevy.", extra={**target_log_ctx, "validation_errors": errors_string}) # Added log
                        current_result["message"] = fatal_error_msg
                        current_result["status"] = "Skipped (Error)"
                        any_target_failed = True
                        processed_results.append(current_result)
                        continue # Skip Hevy update

                # logger.info(f"Validation successful for '{original_title}'. Proceeding to update Hevy.") # Removed old logger

            except Exception as val_e:
                 agent_log.error("Unexpected error during exercise validation/lookup.", exc_info=val_e, extra=target_log_ctx) # Added log
                 current_result["message"] = f"Unexpected validation error: {val_e}"
                 current_result["status"] = "Skipped (Error)"
                 any_target_failed = True
                 processed_results.append(current_result)
                 continue # Skip Hevy update


            # Log the prepared payload (if validation didn't fail fatally)
            if validated_routine_dict:
                 agent_log.debug(f"Payload prepared for Hevy update.", extra=target_log_ctx) # Added log
                 try:
                     logger.debug(json.dumps(validated_routine_dict, indent=2, default=str)) # Use logger for potentially large output
                 except Exception as log_e:
                     agent_log.error(f"Failed to serialize validated_routine_dict for debug logging: {log_e}", extra=target_log_ctx)


            # 5. Update Hevy for Target
            # logger.debug(f"Updating Hevy for '{original_title}'...") # Removed old logger
            agent_log.info("Step 5: Updating routine in Hevy.", extra=target_log_ctx) # Added log
            hevy_error = None
            updated_data_from_hevy = None
            try:
                tool_start_time = time.time()
                # --- Original Logic: Hevy Update Tool Call ---
                hevy_result = await tool_update_routine.ainvoke({
                    "routine_id": routine_id,
                    "routine_data": validated_routine_dict # Use validated dict
                })
                # --- End Original Logic ---
                tool_duration = time.time() - tool_start_time
                agent_log.debug("Hevy update tool call completed.", extra={**target_log_ctx, "duration_seconds": round(tool_duration, 2)}) # Added log
                # logger.info(f"Hevy API update result (raw type: {type(hevy_result)}): {hevy_result}") # Removed old logger # Too verbose

                # Check result structure (Original logic)
                updated_routine_dict = None
                is_success = False
                if isinstance(hevy_result, dict) and isinstance(hevy_result.get("routine"), list) and len(hevy_result["routine"]) > 0 and isinstance(hevy_result["routine"][0], dict) and hevy_result["routine"][0].get("id"):
                    updated_routine_dict = hevy_result["routine"][0]
                    is_success = True
                elif isinstance(hevy_result, dict) and isinstance(hevy_result.get("routine"), dict) and hevy_result["routine"].get("id"):
                     updated_routine_dict = hevy_result["routine"]
                     is_success = True
                elif isinstance(hevy_result, list) and len(hevy_result) > 0 and isinstance(hevy_result[0], dict) and hevy_result[0].get("id"):
                    updated_routine_dict = hevy_result[0]
                    is_success = True

                if is_success:
                    # logger.info(f"Hevy routine '{original_title}' updated successfully.") # Removed old logger
                    agent_log.info(f"Hevy routine updated successfully.", extra=target_log_ctx) # Added log
                    current_result["status"] = "Success"
                    current_result["message"] = f"Routine '{original_title}' updated successfully. Reasoning: {modification_reasoning}" # Include reasoning
                    current_result["updated_routine_data"] = updated_routine_dict # Store updated data
                    updated_data_from_hevy = updated_routine_dict
                else:
                    hevy_error_detail = json.dumps(hevy_result.get("error", hevy_result) if isinstance(hevy_result, dict) else hevy_result)
                    hevy_error = f"Hevy update failed: {hevy_error_detail}"
                    # logger.error(hevy_error) # Removed old logger
                    agent_log.error("Hevy update failed.", extra={**target_log_ctx, "hevy_error": hevy_error_detail}) # Added log
                    current_result["message"] = hevy_error
                    current_result["status"] = "Skipped (Error)"
                    any_target_failed = True

            except Exception as e:
                hevy_error = f"Internal error during Hevy update: {e}"
                # logger.error(hevy_error, exc_info=True) # Removed old logger
                agent_log.error("Internal error during Hevy update.", exc_info=True, extra=target_log_ctx) # Added log
                current_result["message"] = hevy_error
                current_result["status"] = "Skipped (Error)"
                any_target_failed = True

            # Append result for this target
            processed_results.append(current_result)
            agent_log.info(f"Finished processing target #{target_idx + 1}.", extra={**target_log_ctx, "final_status": current_result["status"]}) # Added log

        except Exception as outer_e:
             # Catch unexpected errors in the loop logic itself
             # logger.error(f"Unexpected error processing target '{original_title}': {outer_e}", exc_info=True) # Removed old logger
             agent_log.error(f"Unexpected error processing target.", exc_info=True, extra=target_log_ctx) # Added log
             current_result["message"] = f"Unexpected error: {outer_e}"
             current_result["status"] = "Skipped (Error)"
             any_target_failed = True
             processed_results.append(current_result)
             continue # Ensure loop continues

    # Return Accumulated Results
    final_state_update = {"processed_results": processed_results}
    if any_target_failed and not state.get("process_error"): # Only set if no prior critical error
         new_process_error = "One or more routines failed during the adaptation process."
         agent_log.warning(new_process_error) # Added log
         final_state_update["process_error"] = new_process_error

    # logger.info(f"Finished processing all targets. Results count: {len(processed_results)}") # Removed old logger
    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting process_targets_node", extra={"duration_seconds": round(duration, 2), "targets_processed": len(processed_results), "any_target_failed": any_target_failed})
    # --- End Logging Exit ---
    return final_state_update
# ---

# --- Compile Final Report Node with ELK Logging ---
async def compile_final_report_node_v2(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Compiles the final user-facing report for potentially multiple routines."""
    # --- Logging Setup ---
    start_time = time.time()
    session_id = _get_session_id_from_state(state)
    agent_log = get_agent_logger("progress_adapt.compile_report", session_id)
    agent_log.info("Entering compile_final_report_node_v2")
    # --- End Logging Setup ---

    # --- Original Logic ---
    # logger.info("--- Progress Cycle V2: Compiling Final Report ---") # Removed old logger
    processed_results = state.get("processed_results", [])
    initial_process_error = state.get("process_error")
    user_model = state.get("user_model", {})
    agent_log.debug("Compiling report inputs.", extra={"num_processed_results": len(processed_results), "has_initial_error": bool(initial_process_error)}) # Added log

    # Determine overall status (Original logic)
    overall_status = "Failed"
    overall_message = initial_process_error or "An unknown error occurred."
    processed_summary_lines = []
    success_count = 0
    skipped_no_changes_count = 0
    failure_count = 0
    cycle_completed_successfully = False # Default

    if not initial_process_error:
        if not state.get("identified_targets") and not processed_results:
            overall_status = "Completed (No Targets)"
            overall_message = "I looked at your routines and recent logs, but didn't identify any specific routines needing adaptation right now based on the analysis."
            cycle_completed_successfully = True
        elif processed_results:
            for res in processed_results:
                status_icon = "✅" if res["status"] == "Success" else ("⚠️" if res["status"] == "Skipped (No Changes)" else "❌")
                processed_summary_lines.append(f"{status_icon} **{res['original_title']}**: {res['message']}")
                if res["status"] == "Success": success_count += 1
                if res["status"] == "Skipped (No Changes)": skipped_no_changes_count += 1
                if res["status"] == "Failed" or res["status"] == "Skipped (Error)": failure_count += 1

            if failure_count == 0 and success_count > 0:
                overall_status = "Success"
                overall_message = "I've finished analyzing your progress and successfully updated the relevant routines!"
            elif failure_count == 0 and success_count == 0 and skipped_no_changes_count > 0:
                overall_status = "Completed (No Changes)"
                overall_message = "I've analyzed your progress for the relevant routines, and everything looks on track - no changes needed this time!"
            elif success_count > 0 and failure_count > 0:
                overall_status = "Partial Success"
                overall_message = f"I analyzed your routines. I was able to update {success_count} routine(s), but encountered issues with {failure_count}."
            else: # All failed or skipped with errors
                overall_status = "Failed"
                overall_message = "I tried to analyze and adapt your routines, but encountered errors during the process."

            cycle_completed_successfully = failure_count == 0
        else:
             overall_message = "An unexpected state occurred after identifying targets."
             cycle_completed_successfully = False
    else:
         overall_status = "Failed"
         overall_message = f"I couldn't complete the progress review due to an initial error: {initial_process_error}"
         cycle_completed_successfully = False

    processed_results_summary = "\n".join(processed_summary_lines) if processed_summary_lines else "No routines were processed."
    agent_log.info("Determined overall cycle status.", extra={ # Added log
        "overall_status": overall_status, "success_count": success_count,
        "failure_count": failure_count, "skipped_no_changes_count": skipped_no_changes_count,
        "cycle_completed_successfully": cycle_completed_successfully
    })

    # Generate the final user message using LLM
    final_message = overall_message # Fallback
    try:
        final_report_prompt_template = get_final_cycle_report_v2_prompt()
        user_name = user_model.get("name", "")
        agent_log.debug("Generating final report message via LLM.") # Added log
        filled_prompt = final_report_prompt_template.format(
             user_name=user_name, processed_results_summary=processed_results_summary,
             overall_status=overall_status, overall_message=overall_message
         )
        llm_start_time = time.time()
        # --- Original Logic: LLM Call ---
        response = await llm.ainvoke([SystemMessage(content=filled_prompt)])
        # --- End Original Logic ---
        llm_duration = time.time() - llm_start_time
        final_message = response.content.strip()
        agent_log.info("Successfully generated final report message.", extra={"duration_seconds": round(llm_duration, 2)}) # Added log
    except Exception as e:
         # logger.error(f"Error generating final report message: {e}", exc_info=True) # Removed old logger
         agent_log.error(f"Error generating final report message via LLM, using fallback.", exc_info=True) # Added log
         final_message = f"Hi {user_model.get('name', '')}, {overall_message}\n\nDetails:\n{processed_results_summary}" # Use constructed fallback

    # logger.info(f"Final Cycle Report V2: Status={overall_status}, Successful={cycle_completed_successfully}") # Removed old logger
    # Construct return state (Original logic)
    return_state = {
        "final_report_and_notification": final_message,
        "cycle_completed_successfully": cycle_completed_successfully,
        "process_error": initial_process_error if initial_process_error else state.get("process_error")
    }
    # ---

    # --- Logging Exit ---
    duration = time.time() - start_time
    agent_log.info(f"Exiting compile_final_report_node_v2", extra={"duration_seconds": round(duration, 2), "final_status": overall_status})
    # --- End Logging Exit ---

    return return_state
# ---