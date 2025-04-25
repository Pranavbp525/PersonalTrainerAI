"""
Progress Analysis Agent Module

This module implements the Progress Analysis Agent for the Personal Trainer AI system.
The Progress Analysis Agent is responsible for analyzing user progress and adapting routines.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .base_agent import BaseAgent
from ..llm_providers import LLMProvider


class ProgressAnalysisAgent(BaseAgent):
    """
    Progress Analysis Agent for analyzing user progress and adapting routines.
    
    This agent is responsible for:
    1. Fetching user routines
    2. Fetching workout logs
    3. Identifying target routines for adaptation
    4. Processing targets and making adaptations
    5. Compiling final reports
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Progress Analysis Agent with an LLM provider."""
        super().__init__(llm_provider)
        self.name = "progress_analysis_agent"
        self.description = "Analyzes user progress and adapts workout routines"
    
    async def process(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Process the current state and analyze user progress.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state
        """
        # Determine which analysis step to execute based on state
        analysis_step = state.get("analysis_step", "fetch_routines")
        
        if analysis_step == "fetch_routines":
            return await self.fetch_all_routines(state)
        elif analysis_step == "fetch_logs":
            return await self.fetch_logs(state)
        elif analysis_step == "identify_targets":
            return await self.identify_target_routines(state)
        elif analysis_step == "process_targets":
            return await self.process_targets(state)
        elif analysis_step == "compile_report":
            return await self.compile_final_report(state)
        else:
            # Default to fetching routines
            return await self.fetch_all_routines(state)
    
    async def fetch_all_routines(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch all user routines.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with user routines
        """
        # In a real implementation, this would call an API to fetch user routines
        # For now, we'll simulate fetching routines
        
        simulated_routines = [
            {
                "id": "routine_1",
                "name": "Full Body Strength",
                "description": "A comprehensive full-body strength training routine",
                "goals": ["strength", "muscle_building"],
                "frequency": 3,
                "duration_weeks": 8,
                "created_at": "2023-01-15",
                "last_modified": "2023-03-10"
            },
            {
                "id": "routine_2",
                "name": "Cardio Endurance",
                "description": "Cardio routine focused on building endurance",
                "goals": ["endurance", "fat_loss"],
                "frequency": 4,
                "duration_weeks": 6,
                "created_at": "2023-02-20",
                "last_modified": "2023-04-05"
            }
        ]
        
        # Update state
        updated_state = state.copy()
        updated_state["fetched_routines_list"] = simulated_routines
        updated_state["analysis_step"] = "fetch_logs"
        
        return updated_state
    
    async def fetch_logs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch user workout logs.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with workout logs
        """
        # In a real implementation, this would call an API to fetch workout logs
        # For now, we'll simulate fetching logs
        
        simulated_logs = [
            {
                "id": "log_1",
                "routine_id": "routine_1",
                "date": "2023-03-01",
                "exercises": [
                    {"name": "Squat", "sets": 3, "reps": 10, "weight": 135, "notes": "Felt strong"},
                    {"name": "Bench Press", "sets": 3, "reps": 8, "weight": 155, "notes": "Struggled on last set"}
                ],
                "duration_minutes": 45,
                "perceived_exertion": 7
            },
            {
                "id": "log_2",
                "routine_id": "routine_1",
                "date": "2023-03-03",
                "exercises": [
                    {"name": "Deadlift", "sets": 3, "reps": 8, "weight": 185, "notes": "Good form"},
                    {"name": "Pull-ups", "sets": 3, "reps": 6, "weight": 0, "notes": "Improved from last time"}
                ],
                "duration_minutes": 50,
                "perceived_exertion": 8
            }
        ]
        
        # Update state
        updated_state = state.copy()
        updated_state["workout_logs"] = simulated_logs
        updated_state["analysis_step"] = "identify_targets"
        
        return updated_state
    
    async def identify_target_routines(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify target routines for adaptation.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with identified target routines
        """
        routines = state.get("fetched_routines_list", [])
        logs = state.get("workout_logs", [])
        user_request = state.get("user_request_context", "")
        
        system_prompt = f"""You are a fitness progress analyst identifying routines that need adaptation.
        
TASK: Analyze the user's routines and workout logs to identify routines that need adaptation.

USER REQUEST: {user_request}

ROUTINES:
{routines}

WORKOUT LOGS:
{logs}

OUTPUT REQUIREMENTS:
1. Analyze the routines and workout logs to identify routines that need adaptation
2. For each identified routine, provide:
   - Routine ID
   - Reason for adaptation (e.g., plateau, progression needed, injury accommodation)
   - Specific areas that need adaptation (e.g., exercise selection, volume, intensity)
   - Priority level (high, medium, low)
3. If no routines need adaptation, explain why
4. Format your response as a structured list

Your output should be a clear identification of routines that need adaptation with detailed reasoning.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Identify routines that need adaptation based on the user's progress")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # For demonstration purposes, let's assume we identified targets
        identified_targets = [
            {
                "routine_id": "routine_1",
                "reason": "Plateau in strength gains",
                "areas_for_adaptation": ["increase weight", "vary rep ranges"],
                "priority": "high"
            }
        ]
        
        # Update state
        updated_state = state.copy()
        updated_state["identified_targets"] = identified_targets
        
        # Determine next step based on whether targets were identified
        if identified_targets:
            updated_state["analysis_step"] = "process_targets"
        else:
            updated_state["analysis_step"] = "compile_report"
        
        return updated_state
    
    async def process_targets(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process target routines and make adaptations.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with processed targets
        """
        targets = state.get("identified_targets", [])
        routines = state.get("fetched_routines_list", [])
        logs = state.get("workout_logs", [])
        
        processed_results = []
        
        for target in targets:
            routine_id = target.get("routine_id")
            
            # Find the routine in the list
            routine = next((r for r in routines if r.get("id") == routine_id), None)
            
            if not routine:
                continue
            
            system_prompt = f"""You are a fitness routine adaptation expert.
            
TASK: Create an adapted version of the routine based on the user's progress and identified adaptation needs.

ROUTINE:
{routine}

ADAPTATION NEEDS:
{target}

WORKOUT LOGS:
{[log for log in logs if log.get("routine_id") == routine_id]}

OUTPUT REQUIREMENTS:
1. Create an adapted version of the routine that addresses the identified needs
2. Provide specific changes to:
   - Exercise selection (add, remove, or substitute exercises)
   - Sets, reps, and weight recommendations
   - Rest periods
   - Frequency or duration
3. Explain the rationale for each adaptation
4. Format your response as a structured routine

Your output should be a complete, adapted routine that addresses the user's progress needs.
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Adapt routine '{routine.get('name')}' based on the identified needs")
            ]
            
            response = await self.llm_provider.generate_response(messages)
            
            processed_results.append({
                "routine_id": routine_id,
                "original_routine": routine,
                "adaptation_needs": target,
                "adapted_routine": response
            })
        
        # Update state
        updated_state = state.copy()
        updated_state["processed_results"] = processed_results
        updated_state["analysis_step"] = "compile_report"
        
        return updated_state
    
    async def compile_final_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile a final report of progress analysis and adaptations.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with final report
        """
        processed_results = state.get("processed_results", [])
        identified_targets = state.get("identified_targets", [])
        
        system_prompt = f"""You are a fitness progress analyst compiling a final report.
        
TASK: Compile a comprehensive report of the progress analysis and routine adaptations.

IDENTIFIED ADAPTATION NEEDS:
{identified_targets}

PROCESSED ADAPTATIONS:
{processed_results}

OUTPUT REQUIREMENTS:
1. Create a comprehensive report that includes:
   - Summary of the progress analysis
   - Overview of identified adaptation needs
   - Detailed description of adaptations made to each routine
   - Recommendations for implementation
   - Expected outcomes from the adaptations
2. Format the report for readability with clear sections and bullet points
3. Use a professional, encouraging tone

Your output should be a complete, well-structured report that provides valuable insights and clear guidance for the user.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Compile a final report of the progress analysis and adaptations")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["final_report"] = response
        updated_state["current_agent"] = "coordinator"
        
        # Add AI message to messages
        if "messages" in updated_state:
            updated_state["messages"].append(AIMessage(content=response))
        
        return updated_state