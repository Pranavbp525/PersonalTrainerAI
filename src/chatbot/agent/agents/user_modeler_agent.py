"""
User Modeler Agent

This agent builds and maintains a comprehensive model of the user.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from ..agent_models import AgentState, UserProfile
from ..llm_providers import LLMProvider
from ..prompts import get_user_modeler_prompt
from .base_agent import BaseAgent
from elk_logging import get_agent_logger


class UserModelerAgent(BaseAgent):
    """Agent that builds and maintains a comprehensive model of the user."""
    
    def __init__(self, llm_provider: LLMProvider, model_name: str = "gpt-4o"):
        """
        Initialize the UserModelerAgent.
        
        Args:
            llm_provider: The LLM provider to use
            model_name: The name of the model to use
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.llm = llm_provider.get_chat_model(model_name)
    
    @property
    def name(self) -> str:
        return "user_modeler"
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state to update the user model.
        
        Args:
            state: The current state
            
        Returns:
            The updated state
        """
        # --- Logging Setup ---
        start_time = time.time()
        session_id = state.get("configurable", {}).get("thread_id", "unknown_session")
        agent_log = get_agent_logger("user_modeler", session_id)
        agent_log.info("Entering user_modeler node")
        # --- End Logging Setup ---

        # --- Original Logic ---
        agent_log.debug("Preparing Pydantic parser and prompt")
        parser = PydanticOutputParser(pydantic_object=UserProfile)
        format_instructions = parser.get_format_instructions()
        prompt_template = get_user_modeler_prompt()

        current_user_model = state.get("user_model", {})
        recent_exchanges = state.get("working_memory", {}).get("recent_exchanges", [])

        prompt_context = {
            "has_current_model": bool(current_user_model),
            "num_recent_exchanges": len(recent_exchanges),
        }
        agent_log.debug("Formatting prompt", extra=prompt_context)
        formatted_prompt = prompt_template.format(
            user_model=json.dumps(current_user_model),
            recent_exchanges=json.dumps(recent_exchanges),
            format_instructions=format_instructions
        )

        messages = [SystemMessage(content=formatted_prompt)]
        response = None
        parsed_model = None

        # --- Logging around LLM call and Parsing ---
        try:
            agent_log.info("Invoking LLM for user model update")
            llm_start_time = time.time()
            # --- Original Logic: LLM Call ---
            response = await self.llm_provider.ainvoke(self.llm, messages)
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
        user_model = state.get("user_model", {})
        user_model["last_updated"] = datetime.now().isoformat()
        user_model["model_version"] = user_model.get("model_version", 0) + 1

        # --- Logging before update loop ---
        update_count = 0
        updated_fields_list = []
        # ---

        # Check if parsing was successful before attempting to update
        if parsed_model:
            agent_log.debug("Updating user model fields based on parsed response")
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
                          if field not in user_model or not user_model.get(field)]

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

        # --- Original Logic: Construct updated state ---
        updated_state = {
            **state,
            "user_model": user_model,
            "current_agent": "coordinator"
        }

        # --- Logging Exit ---
        duration = time.time() - start_time
        agent_log.info(f"Exiting user_modeler node", extra={"duration_seconds": round(duration, 2)})
        # --- End Logging Exit ---

        return updated_state