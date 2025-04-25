"""
OpenAI LLM Provider

This module implements the LLM provider interface for OpenAI models.
"""

import os
from typing import Any, Dict, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .base_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """LLM provider implementation for OpenAI models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    
    def get_chat_model(self, model_name: str = "gpt-4o", **kwargs) -> BaseChatModel:
        """
        Get an OpenAI chat model instance.
        
        Args:
            model_name: The name of the model to use (default: "gpt-4o")
            **kwargs: Additional parameters for ChatOpenAI
            
        Returns:
            A ChatOpenAI instance
        """
        return ChatOpenAI(model=model_name, api_key=self.api_key, **kwargs)
    
    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """
        Bind tools to an OpenAI chat model.
        
        Args:
            model: The ChatOpenAI model to bind tools to
            tools: List of tools to bind
            
        Returns:
            A ChatOpenAI model with tools bound
        """
        if not isinstance(model, ChatOpenAI):
            raise TypeError("Model must be a ChatOpenAI instance")
        
        return model.bind_tools(tools)
    
    def with_structured_output(self, model: BaseChatModel, output_schema: type) -> Any:
        """
        Configure an OpenAI model to return structured output according to a schema.
        
        Args:
            model: The ChatOpenAI model to configure
            output_schema: The Pydantic model to use for structured output
            
        Returns:
            A configured model that returns structured output
        """
        if not isinstance(model, ChatOpenAI):
            raise TypeError("Model must be a ChatOpenAI instance")
        
        return model.with_structured_output(output_schema)
    
    async def ainvoke(self, model: BaseChatModel, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Asynchronously invoke the OpenAI model with messages.
        
        Args:
            model: The ChatOpenAI model to invoke
            messages: List of messages to send to the model
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The model's response
        """
        if not isinstance(model, ChatOpenAI):
            raise TypeError("Model must be a ChatOpenAI instance")
        
        return await model.ainvoke(messages, **kwargs)