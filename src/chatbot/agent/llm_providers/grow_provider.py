"""
Grow LLM Provider

This module implements the LLM provider interface for Grow models.
"""

import os
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .base_provider import LLMProvider


class GrowProvider(LLMProvider):
    """LLM provider implementation for Grow models (via Groq API)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Grow provider.
        
        Args:
            api_key: Groq API key. If not provided, will use GROQ_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided and GROQ_API_KEY environment variable not set")
    
    def get_chat_model(self, model_name: str = "llama3-70b-8192", **kwargs) -> BaseChatModel:
        """
        Get a Grow chat model instance via Groq.
        
        Args:
            model_name: The name of the model to use (default: "llama3-70b-8192")
            **kwargs: Additional parameters for ChatGroq
            
        Returns:
            A ChatGroq instance
        """
        return ChatGroq(model=model_name, api_key=self.api_key, **kwargs)
    
    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """
        Bind tools to a Grow chat model.
        
        Args:
            model: The ChatGroq model to bind tools to
            tools: List of tools to bind
            
        Returns:
            A ChatGroq model with tools bound
        """
        if not isinstance(model, ChatGroq):
            raise TypeError("Model must be a ChatGroq instance")
        
        # Groq may have a different approach for tool binding
        # This is a simplified implementation
        return model.bind_tools(tools)
    
    def with_structured_output(self, model: BaseChatModel, output_schema: type) -> Any:
        """
        Configure a Grow model to return structured output according to a schema.
        
        Args:
            model: The ChatGroq model to configure
            output_schema: The Pydantic model to use for structured output
            
        Returns:
            A configured model that returns structured output
        """
        if not isinstance(model, ChatGroq):
            raise TypeError("Model must be a ChatGroq instance")
        
        # Using the standard approach if supported
        try:
            return model.with_structured_output(output_schema)
        except AttributeError:
            # Fallback to a more generic approach if not natively supported
            from langchain.output_parsers import PydanticOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that outputs in the format requested.\n{format_instructions}"),
                ("human", "{query}")
            ])
            
            chain = prompt | model | parser
            return chain
    
    async def ainvoke(self, model: BaseChatModel, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Asynchronously invoke the Grow model with messages.
        
        Args:
            model: The ChatGroq model to invoke
            messages: List of messages to send to the model
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The model's response
        """
        if not isinstance(model, ChatGroq):
            raise TypeError("Model must be a ChatGroq instance")
        
        return await model.ainvoke(messages, **kwargs)