"""
Base Agent

This module defines the base class for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..agent_models import AgentState


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state and return an updated state.
        
        Args:
            state: The current state
            
        Returns:
            The updated state
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the agent.
        
        Returns:
            The agent name
        """
        pass