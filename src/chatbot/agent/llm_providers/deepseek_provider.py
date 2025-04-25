"""
DeepSeek LLM Provider

This module implements the LLM provider interface for DeepSeek models.
"""

import os
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models import DeepSeekChat
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .base_provider import LLMProvider


class DeepSeekProvider(LLMProvider):
    """LLM provider implementation for DeepSeek models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key. If not provided, will use DEEPSEEK_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and DEEPSEEK_API_KEY environment variable not set")
    
    def get_chat_model(self, model_name: str = "deepseek-chat", **kwargs) -> BaseChatModel:
        """
        Get a DeepSeek chat model instance.
        
        Args:
            model_name: The name of the model to use (default: "deepseek-chat")
            **kwargs: Additional parameters for DeepSeekChat
            
        Returns:
            A DeepSeekChat instance
        """
        return DeepSeekChat(model=model_name, api_key=self.api_key, **kwargs)
    
    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """
        Bind tools to a DeepSeek chat model.
        
        Args:
            model: The DeepSeekChat model to bind tools to
            tools: List of tools to bind
            
        Returns:
            A DeepSeekChat model with tools bound
        """
        if not isinstance(model, DeepSeekChat):
            raise TypeError("Model must be a DeepSeekChat instance")
        
        # DeepSeek may have a different approach for tool binding
        # This is a simplified implementation
        return model.bind_tools(tools)
    
    def with_structured_output(self, model: BaseChatModel, output_schema: type) -> Any:
        """
        Configure a DeepSeek model to return structured output according to a schema.
        
        Args:
            model: The DeepSeekChat model to configure
            output_schema: The Pydantic model to use for structured output
            
        Returns:
            A configured model that returns structured output
        """
        if not isinstance(model, DeepSeekChat):
            raise TypeError("Model must be a DeepSeekChat instance")
        
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
        Asynchronously invoke the DeepSeek model with messages.
        
        Args:
            model: The DeepSeekChat model to invoke
            messages: List of messages to send to the model
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The model's response
        """
        if not isinstance(model, DeepSeekChat):
            raise TypeError("Model must be a DeepSeekChat instance")
        
        return await model.ainvoke(messages, **kwargs)