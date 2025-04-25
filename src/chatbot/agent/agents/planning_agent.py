"""
Planning Agent Module

This module implements the Planning Agent for the Personal Trainer AI system.
The Planning Agent is responsible for creating structured workout routines.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .base_agent import BaseAgent
from ..llm_providers import LLMProvider


class PlanningAgent(BaseAgent):
    """
    Planning Agent for creating structured workout routines.
    
    This agent is responsible for:
    1. Creating structured workout plans
    2. Formatting and looking up exercises
    3. Executing tools to create or update routines
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Planning Agent with an LLM provider."""
        super().__init__(llm_provider)
        self.name = "planning_agent"
        self.description = "Creates structured workout routines based on user goals and preferences"
    
    async def process(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Process the current state and create workout routines.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state
        """
        # Determine which planning step to execute based on state
        planning_step = state.get("planning_step", "structured_planning")
        
        if planning_step == "structured_planning":
            return await self.structured_planning(state)
        elif planning_step == "format_and_lookup":
            return await self.format_and_lookup(state)
        elif planning_step == "tool_execution":
            return await self.tool_execution(state)
        else:
            # Default to structured planning
            return await self.structured_planning(state)
    
    async def structured_planning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured workout plan based on user goals and preferences.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with structured workout plan
        """
        user_request = state.get("user_request_context", "")
        user_profile = state.get("user_profile_str", "")
        
        system_prompt = f"""You are a professional fitness trainer creating a structured workout routine.
        
TASK: Create a detailed, structured workout routine based on the user's request and profile.

USER REQUEST: {user_request}

USER PROFILE: {user_profile}

OUTPUT REQUIREMENTS:
1. Create a structured workout routine with the following components:
   - Routine name
   - Description
   - Goals (strength, hypertrophy, endurance, etc.)
   - Frequency (days per week)
   - Duration (weeks)
   - Equipment needed
   - Detailed workout schedule with:
     * Day assignments (e.g., Monday: Upper Body, Tuesday: Lower Body)
     * Exercises for each day with sets, reps, and rest periods
     * Warm-up and cool-down recommendations
2. Ensure the routine is:
   - Appropriate for the user's fitness level
   - Aligned with their goals
   - Realistic given their available equipment and time
   - Progressive (increases in difficulty over time)
3. Format your response as a structured JSON object

Your output should be a comprehensive, well-structured workout routine that the user can immediately implement.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a workout routine based on: {user_request}")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["planner_structured_output"] = response
        updated_state["planning_step"] = "format_and_lookup"
        
        return updated_state
    
    async def format_and_lookup(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the structured plan and look up exercises.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with formatted plan and exercise details
        """
        structured_output = state.get("planner_structured_output", "")
        
        system_prompt = f"""You are a fitness exercise database expert.
        
TASK: Format the structured workout plan and look up detailed information for each exercise.

STRUCTURED WORKOUT PLAN:
{structured_output}

OUTPUT REQUIREMENTS:
1. Parse the structured workout plan
2. For each exercise in the plan:
   - Provide a detailed description
   - List primary and secondary muscles worked
   - Describe proper form and technique
   - Suggest modifications for different fitness levels
   - Note any safety considerations
3. Format the exercises in a consistent structure
4. Ensure all exercise information is accurate and evidence-based

Your output should be a well-formatted workout plan with detailed exercise information that helps the user perform each exercise correctly and safely.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Format the workout plan and provide exercise details")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["formatted_plan"] = response
        updated_state["planning_step"] = "tool_execution"
        
        return updated_state
    
    async def tool_execution(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tools to create or update routines.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with tool execution results
        """
        formatted_plan = state.get("formatted_plan", "")
        
        # In a real implementation, this would call external tools to create or update routines
        # For now, we'll simulate tool execution
        
        # Update state
        updated_state = state.copy()
        updated_state["tool_results"] = {
            "status": "success",
            "message": "Workout routine created successfully",
            "routine_id": "sim_routine_123"
        }
        updated_state["current_agent"] = "coordinator"
        
        # Add AI message to messages
        if "messages" in updated_state:
            updated_state["messages"].append(AIMessage(content=f"I've created a personalized workout routine based on your goals and preferences. The routine includes detailed exercise instructions, proper form guidance, and a progressive structure to help you achieve your fitness goals effectively."))
        
        return updated_state