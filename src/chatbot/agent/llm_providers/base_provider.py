"""
Base LLM Provider Interface

This module defines the abstract base class for LLM providers.
All concrete LLM provider implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Get a chat model instance from this provider.
        
        Args:
            model_name: The name of the model to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A LangChain chat model instance
        """
        pass
    
    @abstractmethod
    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """
        Bind tools to a chat model.
        
        Args:
            model: The chat model to bind tools to
            tools: List of tools to bind
            
        Returns:
            A chat model with tools bound
        """
        pass
    
    @abstractmethod
    def with_structured_output(self, model: BaseChatModel, output_schema: type) -> Any:
        """
        Configure a model to return structured output according to a schema.
        
        Args:
            model: The chat model to configure
            output_schema: The Pydantic model or schema to use for structured output
            
        Returns:
            A configured model that returns structured output
        """
        pass
    
    @abstractmethod
    async def ainvoke(self, model: BaseChatModel, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Asynchronously invoke the model with messages.
        
        Args:
            model: The chat model to invoke
            messages: List of messages to send to the model
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The model's response
        """
        pass