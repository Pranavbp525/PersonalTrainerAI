"""
Agents Package

This package contains the various agent implementations for the personal trainer AI.
"""

from .base_agent import BaseAgent
from .user_modeler_agent import UserModelerAgent
from .coordinator_agent import CoordinatorAgent
from .coach_agent import CoachAgent
from .research_agent import ResearchAgent
from .planning_agent import PlanningAgent
from .progress_analysis_agent import ProgressAnalysisAgent
from .adaptation_agent import AdaptationAgent
from .agent_factory import AgentFactory

__all__ = [
    "BaseAgent",
    "UserModelerAgent",
    "CoordinatorAgent",
    "CoachAgent",
    "ResearchAgent",
    "PlanningAgent",
    "ProgressAnalysisAgent",
    "AdaptationAgent",
    "AgentFactory",
]