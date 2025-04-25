"""
Google Gemini LLM Provider

This module implements the LLM provider interface for Google Gemini models.
"""

import os
from typing import Any, Dict, List, Optional, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .base_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """LLM provider implementation for Google Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google API key. If not provided, will use GOOGLE_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
    
    def get_chat_model(self, model_name: str = "gemini-pro", **kwargs) -> BaseChatModel:
        """
        Get a Gemini chat model instance.
        
        Args:
            model_name: The name of the model to use (default: "gemini-pro")
            **kwargs: Additional parameters for ChatGoogleGenerativeAI
            
        Returns:
            A ChatGoogleGenerativeAI instance
        """
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=self.api_key, **kwargs)
    
    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """
        Bind tools to a Gemini chat model.
        
        Args:
            model: The ChatGoogleGenerativeAI model to bind tools to
            tools: List of tools to bind
            
        Returns:
            A ChatGoogleGenerativeAI model with tools bound
        """
        if not isinstance(model, ChatGoogleGenerativeAI):
            raise TypeError("Model must be a ChatGoogleGenerativeAI instance")
        
        # Gemini uses a different approach for tool binding
        return model.bind_tools(tools)
    
    def with_structured_output(self, model: BaseChatModel, output_schema: type) -> Any:
        """
        Configure a Gemini model to return structured output according to a schema.
        
        Args:
            model: The ChatGoogleGenerativeAI model to configure
            output_schema: The Pydantic model to use for structured output
            
        Returns:
            A configured model that returns structured output
        """
        if not isinstance(model, ChatGoogleGenerativeAI):
            raise TypeError("Model must be a ChatGoogleGenerativeAI instance")
        
        return model.with_structured_output(output_schema)
    
    async def ainvoke(self, model: BaseChatModel, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Asynchronously invoke the Gemini model with messages.
        
        Args:
            model: The ChatGoogleGenerativeAI model to invoke
            messages: List of messages to send to the model
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The model's response
        """
        if not isinstance(model, ChatGoogleGenerativeAI):
            raise TypeError("Model must be a ChatGoogleGenerativeAI instance")
        
        return await model.ainvoke(messages, **kwargs)