"""
Coordinator Agent

This agent manages the overall interaction flow, assessment, and memory.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages

from ..agent_models import AgentState
from .base_agent import BaseAgent
from elk_logging import get_agent_logger


class CoordinatorAgent(BaseAgent):
    """Agent that manages the overall interaction flow, assessment, and memory."""
    
    @property
    def name(self) -> str:
        return "coordinator"
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state to coordinate the interaction flow.
        
        Args:
            state: The current state
            
        Returns:
            The updated state
        """
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
            agent_log.info("Detected return from progress_adaptation_subgraph.")
            working_memory = state.get("working_memory", {})
            processed_results = state.get("processed_results")
            success_status = state.get("cycle_completed_successfully")

            # Log the event in working memory
            memory_log_key = f"progress_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            working_memory[memory_log_key] = {
                "status": "Success" if success_status else "Failed/Partial",
                "notification_sent": progress_notification,
            }
            agent_log.info("Logged progress review result in working memory", extra={"memory_key": memory_log_key, "status": "Success" if success_status else "Failed/Partial"})

            # Prepare state update (Original Logic)
            next_state = {**state}
            next_state["messages"] = add_messages(state.get("messages", []), [AIMessage(content=f"<user>{progress_notification}</user>")])
            next_state["working_memory"] = working_memory
            next_state["final_report_and_notification"] = None
            next_state["cycle_completed_successfully"] = None
            next_state["processed_results"] = None
            
            # --- Logging Exit ---
            duration = time.time() - start_time
            agent_log.info(f"Exiting coordinator node after handling progress notification", extra={"duration_seconds": round(duration, 2)})
            # --- End Logging Exit ---
            
            return next_state

        # --- <<<< CHECK FOR RETURN FROM DEEP RESEARCH SUBGRAPH >>>> ---
        final_report = state.get("final_report")
        if final_report is not None:
            agent_log.info("Detected return from deep_research subgraph.")
            working_memory = state.get("working_memory", {})
            
            # Log the event in working memory
            memory_log_key = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            working_memory[memory_log_key] = {
                "topic": state.get("research_topic", "Unknown Topic"),
                "report_generated": True,
            }
            agent_log.info("Logged research report in working memory", extra={"memory_key": memory_log_key})
            
            # Prepare state update
            next_state = {**state}
            next_state["messages"] = add_messages(state.get("messages", []), [AIMessage(content=final_report)])
            next_state["working_memory"] = working_memory
            next_state["final_report"] = None
            next_state["research_topic"] = None
            next_state["current_agent"] = "coach_agent"
            
            # --- Logging Exit ---
            duration = time.time() - start_time
            agent_log.info(f"Exiting coordinator node after handling research report", extra={"duration_seconds": round(duration, 2)})
            # --- End Logging Exit ---
            
            return next_state

        # --- <<<< STANDARD COORDINATION LOGIC >>>> ---
        # Determine the next agent based on the current state
        user_model = state.get("user_model", {})
        working_memory = state.get("working_memory", {})
        recent_messages = state.get("messages", [])[-5:] if state.get("messages") else []
        
        # Check if assessment is complete
        assessment_complete = user_model.get("assessment_complete", False)
        
        # Determine the next agent
        next_agent = "coach_agent"  # Default to coach agent
        
        if not assessment_complete:
            # If assessment is not complete, prioritize completing it
            missing_fields = user_model.get("missing_fields", [])
            if missing_fields:
                agent_log.info(f"Assessment incomplete. Missing fields: {missing_fields}")
                next_agent = "coach_agent"  # Coach will handle assessment
        else:
            # Assessment is complete, determine the next agent based on context
            # This is a simplified version - in a real implementation, you would have more sophisticated logic
            agent_log.info("Assessment complete. Determining next agent based on context.")
            
            # Example: Check if there's a research request
            research_requested = any("research" in msg.content.lower() for msg in recent_messages if hasattr(msg, 'content'))
            if research_requested:
                agent_log.info("Research request detected. Routing to research agent.")
                next_agent = "research_agent"
            
            # Example: Check if there's a planning request
            planning_requested = any(("plan" in msg.content.lower() or "routine" in msg.content.lower()) for msg in recent_messages if hasattr(msg, 'content'))
            if planning_requested:
                agent_log.info("Planning request detected. Routing to planning agent.")
                next_agent = "planning_agent"
            
            # Example: Check if there's a progress analysis request
            progress_requested = any(("progress" in msg.content.lower() or "adapt" in msg.content.lower()) for msg in recent_messages if hasattr(msg, 'content'))
            if progress_requested:
                agent_log.info("Progress analysis request detected. Routing to progress analysis agent.")
                next_agent = "progress_analysis_agent"
        
        # Update the state with the next agent
        next_state = {
            **state,
            "current_agent": next_agent
        }
        
        agent_log.info(f"Routing to {next_agent}")
        
        # --- Logging Exit ---
        duration = time.time() - start_time
        agent_log.info(f"Exiting coordinator node", extra={"duration_seconds": round(duration, 2), "next_agent": next_agent})
        # --- End Logging Exit ---
        
        return next_state