"""
Adaptation Agent Module

This module implements the Adaptation Agent for the Personal Trainer AI system.
The Adaptation Agent is responsible for adapting workout routines based on user progress.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .base_agent import BaseAgent
from ..llm_providers import LLMProvider


class AdaptationAgent(BaseAgent):
    """
    Adaptation Agent for adapting workout routines based on user progress.
    
    This agent is responsible for:
    1. Analyzing user progress data
    2. Identifying areas for adaptation
    3. Creating adapted workout routines
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Adaptation Agent with an LLM provider."""
        super().__init__(llm_provider)
        self.name = "adaptation_agent"
        self.description = "Adapts workout routines based on user progress"
    
    async def process(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Process the current state and adapt workout routines.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state
        """
        progress_data = state.get("progress_data", {})
        routines = state.get("fetched_routines_list", [])
        user_profile = state.get("user_profile_str", "")
        
        system_prompt = f"""You are a fitness adaptation expert specializing in modifying workout routines.
        
TASK: Analyze the user's progress data and adapt their workout routines accordingly.

USER PROFILE:
{user_profile}

PROGRESS DATA:
{progress_data}

CURRENT ROUTINES:
{routines}

OUTPUT REQUIREMENTS:
1. Analyze the user's progress data to identify:
   - Areas of improvement
   - Plateaus or stagnation
   - Potential overtraining or injury risks
2. For each routine that needs adaptation:
   - Provide specific modifications to exercises, sets, reps, or intensity
   - Explain the rationale behind each adaptation
   - Ensure the adaptations align with the user's goals and current progress
3. Create a progressive plan that challenges the user appropriately
4. Format your response as a structured adaptation plan

Your output should be a comprehensive adaptation plan that helps the user continue making progress toward their fitness goals.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Adapt the user's workout routines based on their progress data")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["adaptation_plan"] = response
        updated_state["current_agent"] = "coordinator"
        
        # Add AI message to messages
        if "messages" in updated_state:
            updated_state["messages"].append(AIMessage(content=f"I've analyzed your progress and created personalized adaptations to your workout routines. These changes are designed to address your current needs and help you continue making progress toward your fitness goals."))
        
        return updated_state