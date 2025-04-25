"""
LLM Provider Factory

This module provides a factory for creating and managing LLM providers.
"""

from typing import Dict, Optional, Type

from .base_provider import LLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .deepseek_provider import DeepSeekProvider
from .grow_provider import GrowProvider


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
        "deepseek": DeepSeekProvider,
        "grow": GrowProvider,
    }
    
    _instances: Dict[str, LLMProvider] = {}
    
    @classmethod
    def get_provider(cls, provider_name: str, **kwargs) -> LLMProvider:
        """
        Get an LLM provider instance.
        
        Args:
            provider_name: The name of the provider to get
            **kwargs: Additional parameters for the provider constructor
            
        Returns:
            An LLM provider instance
            
        Raises:
            ValueError: If the provider name is not recognized
        """
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available providers: {list(cls._providers.keys())}")
        
        # Create a new instance if one doesn't exist or if kwargs are provided
        if provider_name not in cls._instances or kwargs:
            provider_class = cls._providers[provider_name]
            cls._instances[provider_name] = provider_class(**kwargs)
        
        return cls._instances[provider_name]
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a new LLM provider.
        
        Args:
            name: The name to register the provider under
            provider_class: The provider class to register
            
        Raises:
            TypeError: If the provider class does not inherit from LLMProvider
        """
        if not issubclass(provider_class, LLMProvider):
            raise TypeError("Provider class must inherit from LLMProvider")
        
        cls._providers[name] = provider_class
    
    @classmethod
    def list_providers(cls) -> list:
        """
        List all available providers.
        
        Returns:
            A list of provider names
        """
        return list(cls._providers.keys())