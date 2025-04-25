"""
Ollama LLM Provider

This module implements the LLM provider interface for Ollama models.
"""

import os
from typing import Any, Dict, List, Optional, Union
from langchain_community.llms.ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .base_provider import LLMProvider


class OllamaProvider(LLMProvider):
    """LLM provider implementation for Ollama models."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            base_url: Ollama API base URL. If not provided, will use OLLAMA_BASE_URL environment variable
                     or default to "http://localhost:11434".
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def get_chat_model(self, model_name: str = "llama3", **kwargs) -> BaseChatModel:
        """
        Get an Ollama chat model instance.
        
        Args:
            model_name: The name of the model to use (default: "llama3")
            **kwargs: Additional parameters for ChatOllama
            
        Returns:
            A ChatOllama instance
        """
        return ChatOllama(model=model_name, base_url=self.base_url, **kwargs)
    
    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """
        Bind tools to an Ollama chat model.
        
        Args:
            model: The ChatOllama model to bind tools to
            tools: List of tools to bind
            
        Returns:
            A ChatOllama model with tools bound
        """
        if not isinstance(model, ChatOllama):
            raise TypeError("Model must be a ChatOllama instance")
        
        # Ollama may not support tool binding in the same way as OpenAI
        # This is a simplified implementation
        return model.bind(functions=[tool.dict() for tool in tools])
    
    def with_structured_output(self, model: BaseChatModel, output_schema: type) -> Any:
        """
        Configure an Ollama model to return structured output according to a schema.
        
        Args:
            model: The ChatOllama model to configure
            output_schema: The Pydantic model to use for structured output
            
        Returns:
            A configured model that returns structured output
        """
        if not isinstance(model, ChatOllama):
            raise TypeError("Model must be a ChatOllama instance")
        
        # Ollama may not have native structured output support
        # Using a more generic approach
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
        Asynchronously invoke the Ollama model with messages.
        
        Args:
            model: The ChatOllama model to invoke
            messages: List of messages to send to the model
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The model's response
        """
        if not isinstance(model, ChatOllama):
            raise TypeError("Model must be a ChatOllama instance")
        
        return await model.ainvoke(messages, **kwargs)