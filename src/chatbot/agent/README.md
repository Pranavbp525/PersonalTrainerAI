# Personal Trainer AI - Modular Agent Architecture

This directory contains the modular agent architecture for the Personal Trainer AI system. The architecture is designed to be flexible and extensible, allowing for easy integration of different LLM providers and agent implementations.

## Directory Structure

```
agent/
├── agents/                  # Agent implementations
│   ├── base_agent.py        # Base agent interface
│   ├── user_modeler_agent.py # User modeling agent
│   ├── coordinator_agent.py # Coordinator agent
│   ├── coach_agent.py       # Coach agent
│   ├── agent_factory.py     # Factory for creating agents
│   └── __init__.py          # Package initialization
├── llm_providers/           # LLM provider implementations
│   ├── base_provider.py     # Base provider interface
│   ├── openai_provider.py   # OpenAI provider
│   ├── gemini_provider.py   # Google Gemini provider
│   ├── ollama_provider.py   # Ollama provider
│   ├── deepseek_provider.py # DeepSeek provider
│   ├── grow_provider.py     # Grow provider (via Groq)
│   ├── provider_factory.py  # Factory for creating providers
│   └── __init__.py          # Package initialization
├── models/                  # Data models
│   └── __init__.py          # Package initialization
├── utils_module/            # Utility functions
│   └── __init__.py          # Package initialization
├── agent_models.py          # Agent data models
├── graph.py                 # Agent graph definition
├── hevy_api.py              # Hevy API integration
├── llm_tools.py             # LLM tools
├── main_agent.py            # Main agent entry point
├── personal_trainer_agent.py # Legacy agent implementation
├── prompts.py               # Agent prompts
├── utils.py                 # Utility functions
└── __init__.py              # Package initialization
```

## Usage

### Basic Usage

```python
from src.chatbot.agent import process_message

# Process a user message
result = await process_message("Hello, I'm looking for fitness advice.")

# Extract the assistant's response
messages = result.get("messages", [])
if messages:
    assistant_message = messages[-1].content
    print(f"Assistant: {assistant_message}")
```

### Using Different LLM Providers

```python
from src.chatbot.agent import process_message

# Use OpenAI (default)
result_openai = await process_message("Hello", provider_name="openai")

# Use Google Gemini
result_gemini = await process_message("Hello", provider_name="gemini")

# Use Ollama
result_ollama = await process_message("Hello", provider_name="ollama")

# Use DeepSeek
result_deepseek = await process_message("Hello", provider_name="deepseek")

# Use Grow (via Groq)
result_grow = await process_message("Hello", provider_name="grow")
```

### Creating a Custom Agent Graph

```python
from src.chatbot.agent import create_agent_graph

# Create a graph with OpenAI
graph_openai = create_agent_graph(provider_name="openai")

# Create a graph with Gemini
graph_gemini = create_agent_graph(provider_name="gemini")

# Compile the graph
app = graph_openai.compile()

# Process a state
result = await app.ainvoke({
    "messages": [],
    "memory": {},
    "working_memory": {},
    "user_model": {},
    "current_agent": "coordinator",
})
```

## Extending the Architecture

### Adding a New LLM Provider

1. Create a new provider class that inherits from `LLMProvider`
2. Implement the required methods
3. Register the provider with the `LLMProviderFactory`

```python
from src.chatbot.agent.llm_providers import LLMProvider, LLMProviderFactory

class MyCustomProvider(LLMProvider):
    # Implement the required methods
    ...

# Register the provider
LLMProviderFactory.register_provider("my_custom", MyCustomProvider)

# Use the provider
provider = LLMProviderFactory.get_provider("my_custom")
```

### Adding a New Agent

1. Create a new agent class that inherits from `BaseAgent`
2. Implement the required methods
3. Register the agent with the `AgentFactory`

```python
from src.chatbot.agent.agents import BaseAgent, AgentFactory

class MyCustomAgent(BaseAgent):
    # Implement the required methods
    ...

# Register the agent
AgentFactory.register_agent("my_custom", MyCustomAgent)

# Use the agent
agent = AgentFactory.get_agent("my_custom")
```

## API Reference

### LLM Providers

- `LLMProvider`: Base class for LLM providers
- `OpenAIProvider`: Provider for OpenAI models
- `GeminiProvider`: Provider for Google Gemini models
- `OllamaProvider`: Provider for Ollama models
- `DeepSeekProvider`: Provider for DeepSeek models
- `GrowProvider`: Provider for Grow models (via Groq)
- `LLMProviderFactory`: Factory for creating and managing LLM providers

### Agents

- `BaseAgent`: Base class for agents
- `UserModelerAgent`: Agent for modeling the user
- `CoordinatorAgent`: Agent for coordinating the interaction flow
- `CoachAgent`: Agent for providing fitness coaching
- `AgentFactory`: Factory for creating and managing agents

### Main API

- `create_agent_graph`: Create an agent graph with the specified LLM provider
- `process_message`: Process a user message and return the updated state