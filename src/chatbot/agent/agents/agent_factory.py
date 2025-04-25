"""
Agent Factory

This module provides a factory for creating and managing agents.
"""

from typing import Dict, Optional, Type

from ..llm_providers import LLMProvider, LLMProviderFactory
from .base_agent import BaseAgent
from .user_modeler_agent import UserModelerAgent
from .coordinator_agent import CoordinatorAgent
from .coach_agent import CoachAgent


class AgentFactory:
    """Factory for creating and managing agents."""
    
    _agent_classes: Dict[str, Type[BaseAgent]] = {
        "user_modeler": UserModelerAgent,
        "coordinator": CoordinatorAgent,
        "coach_agent": CoachAgent,
    }
    
    _instances: Dict[str, BaseAgent] = {}
    
    @classmethod
    def get_agent(cls, agent_name: str, llm_provider: Optional[LLMProvider] = None, model_name: Optional[str] = None, **kwargs) -> BaseAgent:
        """
        Get an agent instance.
        
        Args:
            agent_name: The name of the agent to get
            llm_provider: The LLM provider to use (if needed)
            model_name: The name of the model to use (if needed)
            **kwargs: Additional parameters for the agent constructor
            
        Returns:
            An agent instance
            
        Raises:
            ValueError: If the agent name is not recognized
        """
        if agent_name not in cls._agent_classes:
            raise ValueError(f"Unknown agent: {agent_name}. Available agents: {list(cls._agent_classes.keys())}")
        
        # Create a new instance if one doesn't exist or if kwargs are provided
        if agent_name not in cls._instances or kwargs:
            agent_class = cls._agent_classes[agent_name]
            
            # Check if the agent requires an LLM provider
            if agent_class in [UserModelerAgent, CoachAgent]:
                if llm_provider is None:
                    # Use OpenAI provider by default
                    llm_provider = LLMProviderFactory.get_provider("openai")
                
                if model_name is None:
                    # Use gpt-4o by default
                    model_name = "gpt-4o"
                
                cls._instances[agent_name] = agent_class(llm_provider, model_name, **kwargs)
            else:
                cls._instances[agent_name] = agent_class(**kwargs)
        
        return cls._instances[agent_name]
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent.
        
        Args:
            name: The name to register the agent under
            agent_class: The agent class to register
            
        Raises:
            TypeError: If the agent class does not inherit from BaseAgent
        """
        if not issubclass(agent_class, BaseAgent):
            raise TypeError("Agent class must inherit from BaseAgent")
        
        cls._agent_classes[name] = agent_class
    
    @classmethod
    def list_agents(cls) -> list:
        """
        List all available agents.
        
        Returns:
            A list of agent names
        """
        return list(cls._agent_classes.keys())