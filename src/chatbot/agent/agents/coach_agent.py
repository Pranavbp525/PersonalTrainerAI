"""
Coach Agent

This agent handles direct interactions with the user, providing fitness guidance and advice.
"""

import time
from typing import Dict, Any, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from ..agent_models import AgentState
from ..llm_providers import LLMProvider
from ..prompts import get_coach_prompt
from .base_agent import BaseAgent
from elk_logging import get_agent_logger


class CoachAgent(BaseAgent):
    """Agent that handles direct interactions with the user, providing fitness guidance and advice."""
    
    def __init__(self, llm_provider: LLMProvider, model_name: str = "gpt-4o"):
        """
        Initialize the CoachAgent.
        
        Args:
            llm_provider: The LLM provider to use
            model_name: The name of the model to use
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.llm = llm_provider.get_chat_model(model_name)
    
    @property
    def name(self) -> str:
        return "coach_agent"
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state to provide coaching to the user.
        
        Args:
            state: The current state
            
        Returns:
            The updated state
        """
        # --- Logging Setup ---
        start_time = time.time()
        session_id = state.get("configurable", {}).get("thread_id", "unknown_session")
        agent_log = get_agent_logger("coach_agent", session_id)
        agent_log.info("Entering coach_agent node")
        # --- End Logging Setup ---

        # --- Original Logic ---
        agent_log.debug("Preparing coach prompt")
        
        # Get the user model and messages
        user_model = state.get("user_model", {})
        messages = state.get("messages", [])
        working_memory = state.get("working_memory", {})
        
        # Check if assessment is complete
        assessment_complete = user_model.get("assessment_complete", False)
        missing_fields = user_model.get("missing_fields", [])
        
        # Prepare the system prompt
        prompt_template = get_coach_prompt()
        formatted_prompt = prompt_template.format(
            user_model=str(user_model),
            assessment_complete=assessment_complete,
            missing_fields=missing_fields
        )
        
        # Create the message list for the LLM
        system_message = SystemMessage(content=formatted_prompt)
        message_list = [system_message]
        
        # Add the conversation history
        for message in messages:
            message_list.append(message)
        
        # If there are no user messages yet, add a greeting
        if not any(isinstance(msg, HumanMessage) for msg in messages):
            message_list.append(HumanMessage(content="Hello, I'm looking for fitness advice."))
        
        agent_log.info("Invoking LLM for coach response")
        llm_start_time = time.time()
        
        try:
            # Invoke the LLM
            response = await self.llm_provider.ainvoke(self.llm, message_list)
            
            llm_duration = time.time() - llm_start_time
            agent_log.info("LLM invocation completed", extra={"duration_seconds": round(llm_duration, 2)})
            
            # Process the response
            response_content = response.content
            
            # Update recent exchanges in working memory
            recent_exchanges = working_memory.get("recent_exchanges", [])
            recent_exchanges.append({
                "user": messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "Initial greeting",
                "assistant": response_content
            })
            
            # Keep only the last 10 exchanges
            if len(recent_exchanges) > 10:
                recent_exchanges = recent_exchanges[-10:]
            
            working_memory["recent_exchanges"] = recent_exchanges
            
            # Determine the next agent
            next_agent = "user_modeler"  # Default to user modeler after coach
            
            # Check if we need to route to a different agent based on the response
            if "research" in response_content.lower():
                agent_log.info("Research mentioned in response, considering research agent")
                if "I'll research" in response_content.lower() or "let me research" in response_content.lower():
                    next_agent = "research_agent"
                    agent_log.info("Routing to research agent based on response content")
            
            if "plan" in response_content.lower() or "routine" in response_content.lower():
                agent_log.info("Planning mentioned in response, considering planning agent")
                if "I'll create a plan" in response_content.lower() or "let me design a routine" in response_content.lower():
                    next_agent = "planning_agent"
                    agent_log.info("Routing to planning agent based on response content")
            
            if "progress" in response_content.lower() or "adapt" in response_content.lower():
                agent_log.info("Progress analysis mentioned in response, considering progress analysis agent")
                if "I'll analyze your progress" in response_content.lower() or "let me adapt your routine" in response_content.lower():
                    next_agent = "progress_analysis_agent"
                    agent_log.info("Routing to progress analysis agent based on response content")
            
            # Update the state
            updated_state = {
                **state,
                "messages": add_messages(messages, [AIMessage(content=response_content)]),
                "working_memory": working_memory,
                "current_agent": next_agent
            }
            
        except Exception as e:
            agent_log.error(f"Error during LLM call in coach_agent", exc_info=True)
            
            # Fallback response in case of error
            error_response = "I'm having trouble processing your request right now. Could you please try again or rephrase your question?"
            
            # Update the state with the error response
            updated_state = {
                **state,
                "messages": add_messages(messages, [AIMessage(content=error_response)]),
                "working_memory": working_memory,
                "current_agent": "coordinator"  # Route back to coordinator on error
            }
        
        # --- Logging Exit ---
        duration = time.time() - start_time
        agent_log.info(f"Exiting coach_agent node", extra={"duration_seconds": round(duration, 2), "next_agent": updated_state["current_agent"]})
        # --- End Logging Exit ---
        
        return updated_state