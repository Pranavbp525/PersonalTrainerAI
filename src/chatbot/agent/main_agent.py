"""
Main Agent Module

This module provides the main entry point for the personal trainer agent,
using the modular architecture with pluggable LLM providers and agent implementations.
"""

import os
from typing import Dict, Any, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .agent_models import AgentState
from .llm_providers import LLMProviderFactory
from .agents import AgentFactory
from .graph import coordinator_condition
from elk_logging import setup_elk_logging

# Initialize logger
main_log = setup_elk_logging("fitness-chatbot.main_agent")


def create_agent_graph(
    provider_name: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> StateGraph:
    """
    Create the main agent graph with the specified LLM provider and model.
    
    Args:
        provider_name: The name of the LLM provider to use
        model_name: The name of the model to use (provider-specific)
        **kwargs: Additional parameters for the LLM provider
        
    Returns:
        A StateGraph instance representing the agent workflow
    """
    main_log.info(f"Creating agent graph with provider: {provider_name}")
    
    # Get the LLM provider
    try:
        llm_provider = LLMProviderFactory.get_provider(provider_name, **kwargs)
        main_log.info(f"Using LLM provider: {provider_name}")
    except ValueError as e:
        main_log.error(f"Error getting LLM provider: {e}")
        raise
    
    # Use provider-specific default model if not specified
    if model_name is None:
        if provider_name == "openai":
            model_name = "gpt-4o"
        elif provider_name == "gemini":
            model_name = "gemini-pro"
        elif provider_name == "ollama":
            model_name = "llama3"
        elif provider_name == "deepseek":
            model_name = "deepseek-chat"
        elif provider_name == "grow":
            model_name = "llama3-70b-8192"
        else:
            model_name = "gpt-4o"  # Default fallback
    
    main_log.info(f"Using model: {model_name}")
    
    # Create the agents
    user_modeler = AgentFactory.get_agent("user_modeler", llm_provider, model_name)
    coordinator = AgentFactory.get_agent("coordinator")
    coach_agent = AgentFactory.get_agent("coach_agent", llm_provider, model_name)
    research_agent = AgentFactory.get_agent("research_agent", llm_provider, model_name)
    planning_agent = AgentFactory.get_agent("planning_agent", llm_provider, model_name)
    progress_analysis_agent = AgentFactory.get_agent("progress_analysis_agent", llm_provider, model_name)
    adaptation_agent = AgentFactory.get_agent("adaptation_agent", llm_provider, model_name)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("user_modeler", user_modeler.process)
    workflow.add_node("coordinator", coordinator.process)
    workflow.add_node("coach_agent", coach_agent.process)
    workflow.add_node("research_agent", research_agent.process)
    workflow.add_node("planning_agent", planning_agent.process)
    workflow.add_node("progress_analysis_agent", progress_analysis_agent.process)
    workflow.add_node("adaptation_agent", adaptation_agent.process)
    workflow.add_node("end_conversation", lambda state: {**state, "current_agent": "end_conversation"})
    
    # Add edges
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_condition,
        {
            "user_modeler": "user_modeler",
            "coach_agent": "coach_agent",
            "research_agent": "research_agent",
            "planning_agent": "planning_agent",
            "progress_analysis_agent": "progress_analysis_agent",
            "adaptation_agent": "adaptation_agent",
            "end_conversation": "end_conversation",
        }
    )
    
    workflow.add_edge("user_modeler", "coordinator")
    workflow.add_edge("coach_agent", "coordinator")
    workflow.add_edge("research_agent", "coordinator")
    workflow.add_edge("planning_agent", "coordinator")
    workflow.add_edge("progress_analysis_agent", "coordinator")
    workflow.add_edge("adaptation_agent", "coordinator")
    workflow.add_edge("end_conversation", END)
    
    # Set the entry point
    workflow.set_entry_point("coordinator")
    
    main_log.info("Agent graph created successfully")
    
    return workflow


async def process_message(
    message: str,
    state: Optional[AgentState] = None,
    provider_name: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a user message and return the updated state.
    
    Args:
        message: The user message to process
        state: The current state (if any)
        provider_name: The name of the LLM provider to use
        model_name: The name of the model to use (provider-specific)
        **kwargs: Additional parameters for the LLM provider
        
    Returns:
        The updated state after processing the message
    """
    from langchain_core.messages import HumanMessage
    from langgraph.graph.message import add_messages
    
    # Create a new state if none is provided
    if state is None:
        state = {
            "messages": [],
            "memory": {},
            "working_memory": {},
            "user_model": {},
            "current_agent": "coordinator",
        }
    
    # Add the user message to the state
    state["messages"] = add_messages(state.get("messages", []), [HumanMessage(content=message)])
    
    # Create the agent graph
    workflow = create_agent_graph(provider_name, model_name, **kwargs)
    
    # Compile the graph
    app = workflow.compile()
    
    # Process the message
    result = await app.ainvoke(state)
    
    return result