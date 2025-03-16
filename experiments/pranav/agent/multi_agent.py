from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import json
import os
import uuid
from llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools([retrieve_from_rag, 
                                 tool_fetch_workouts,
                                 tool_get_workout_count,
                                 tool_fetch_routines,
                                 tool_update_routine,
                                 tool_create_routine,
                                 retrieve_from_rag])

# Define more sophisticated state model
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    memory: Dict[str, Any]  # Long-term memory storage
    working_memory: Dict[str, Any]  # Short-term contextual memory
    user_model: Dict[str, Any]  # Comprehensive user model
    fitness_plan: Dict[str, Any]  # Structured fitness plan
    reasoning_trace: List[Dict[str, Any]]  # Traces of reasoning steps
    agent_state: Dict[str, str]  # Current state of each agent
    current_agent: str  # Currently active agent
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Memory manager for sophisticated state management
def memory_manager(state: AgentState) -> AgentState:
    """Manages long-term and working memory, consolidating information and pruning as needed."""
    
    memory_prompt = """You are the memory manager for a fitness training system.
    Review the conversation history and current agent states to:
    
    1. Identify key information that should be stored in long-term memory
    2. Update the user model with new insights
    3. Consolidate redundant information
    4. Prune outdated or superseded information
    5. Ensure critical context is available in working memory
    
    Current long-term memory: {{memory}}
    Current user model: {{user_model}}
    Current working memory: {{working_memory}}
    
    Return a structured update of what should be stored, updated, or removed.
    """
    
    # Fill template
    filled_prompt = memory_prompt.replace("{{memory}}", str(state.get("memory", {})))
    filled_prompt = filled_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{working_memory}}", str(state.get("working_memory", {})))
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Parse and update memory structures
    # This would include sophisticated logic to maintain memory integrity
    
    # For this example, we'll make a simple update
    memory = state.get("memory", {})
    working_memory = state.get("working_memory", {})
    user_model = state.get("user_model", {})
    
    # Extract recent messages for context
    recent_exchanges = []
    for msg in state["messages"][-10:]:  # Last 10 messages
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            recent_exchanges.append({"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                                    "content": msg.content})
    
    # Update working memory with recent context
    working_memory["recent_exchanges"] = recent_exchanges
    working_memory["last_updated"] = datetime.now().isoformat()
    
    # Track agent interactions in memory
    memory_key = f"interaction_{datetime.now().isoformat()}"
    memory[memory_key] = {
        "agent_states": state.get("agent_state", {}),
        "current_agent": state.get("current_agent", "coordinator"),
        "user_intent": working_memory.get("current_user_intent", "unknown")
    }
    
    return {
        **state,
        "memory": memory,
        "working_memory": working_memory,
        "user_model": user_model
    }

# Reasoning engine for sophisticated decision-making
def reasoning_engine(state: AgentState) -> AgentState:
    """Advanced reasoning to determine the optimal next steps and actions."""
    
    reasoning_prompt = """You are the reasoning engine for a fitness training system.
    Analyze the current situation using sophisticated reasoning to determine:
    
    1. What is the user's current need or intent?
    2. What information do we currently have and what's missing?
    3. What potential approaches could address the user's needs?
    4. What are the tradeoffs between different approaches?
    5. What is the optimal next action given all constraints?
    
    User model: {{user_model}}
    Working memory: {{working_memory}}
    Fitness plan: {{fitness_plan}}
    
    Think step by step and document your reasoning process.
    """
    
    # Fill template
    filled_prompt = reasoning_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{working_memory}}", str(state.get("working_memory", {})))
    filled_prompt = filled_prompt.replace("{{fitness_plan}}", str(state.get("fitness_plan", {})))
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Extract reasoning trace
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append({
        "timestamp": datetime.now().isoformat(),
        "reasoning": response.content,
        "context": {
            "current_agent": state.get("current_agent"),
            "working_memory_snapshot": state.get("working_memory", {})
        }
    })
    
    # Determine next agent based on reasoning
    # Extract key insights to guide agent selection
    agent_selection = "coordinator"  # Default
    
    if "assessment" in response.content.lower() or "profile" in response.content.lower():
        agent_selection = "profiler_agent"
    elif "research" in response.content.lower() or "knowledge" in response.content.lower():
        agent_selection = "research_agent"
    elif "plan" in response.content.lower() or "routine" in response.content.lower():
        agent_selection = "planner_agent"
    elif "progress" in response.content.lower() or "analyze" in response.content.lower():
        agent_selection = "analyst_agent"
    elif "adjust" in response.content.lower() or "modify" in response.content.lower():
        agent_selection = "adaptation_agent"
    elif "motivate" in response.content.lower() or "coach" in response.content.lower():
        agent_selection = "coach_agent"
    
    # Update working memory with reasoning results
    working_memory = state.get("working_memory", {})
    working_memory["reasoning_summary"] = response.content
    working_memory["selected_agent"] = agent_selection
    
    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "current_agent": agent_selection,
        "working_memory": working_memory
    }

# User modeler for comprehensive user understanding
def user_modeler(state: AgentState) -> AgentState:
    """Builds and maintains a comprehensive model of the user."""
    
    modeling_prompt = """You are a user modeling specialist for a fitness training system.
    Analyze all available information about the user to build a comprehensive model:
    
    1. Extract explicit information (stated goals, preferences, constraints)
    2. Infer implicit information (fitness level, motivation factors, learning style)
    3. Identify gaps in our understanding that need to be addressed
    4. Update confidence levels for different aspects of the model
    
    Current user model: {{user_model}}
    Recent exchanges: {{recent_exchanges}}
    
    Return an updated user model with confidence scores for each attribute.
    """
    
    # Fill template
    filled_prompt = modeling_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{recent_exchanges}}", 
                                         str(state.get("working_memory", {}).get("recent_exchanges", [])))
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Parse and update user model
    # In a real system, we'd have sophisticated parsing and model updating
    
    # Simple example update
    user_model = state.get("user_model", {})
    user_model["last_updated"] = datetime.now().isoformat()
    user_model["model_version"] = user_model.get("model_version", 0) + 1
    
    # Extract goals if mentioned in recent messages
    for msg in state["messages"][-5:]:
        if isinstance(msg, HumanMessage) and "goal" in msg.content.lower():
            user_model["has_explicit_goals"] = True
    
    return {
        **state,
        "user_model": user_model
    }

# Agent coordinator for orchestrating the multi-agent system
def coordinator_agent(state: AgentState) -> AgentState:
    """Coordinates the multi-agent system and handles high-level decision making."""
    
    coordinator_prompt = """You are the coordinator for a fitness training multi-agent system.
    
    Current agent states:
    {{agent_states}}
    
    Reasoning summary:
    {{reasoning_summary}}
    
    Selected agent: {{selected_agent}}
    
    Your tasks:
    1. Ensure smooth transitions between agents
    2. Maintain conversation coherence across agent handoffs
    3. Resolve conflicts between agent recommendations
    4. Determine when to invoke specialized agents vs. handling directly
    
    If you're responding to the user directly, maintain a supportive, expert tone.
    """
    
    # Fill template with state information
    filled_prompt = coordinator_prompt.replace("{{agent_states}}", str(state.get("agent_state", {})))
    filled_prompt = filled_prompt.replace("{{reasoning_summary}}", 
                                         str(state.get("working_memory", {}).get("reasoning_summary", "")))
    filled_prompt = filled_prompt.replace("{{selected_agent}}", 
                                         state.get("working_memory", {}).get("selected_agent", "coordinator"))
    
    # If user message is the last one, respond
    last_message_is_user = False
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message_is_user = True
            break
        elif isinstance(msg, AIMessage):
            last_message_is_user = False
            break
    
    if last_message_is_user:
        messages = state["messages"] + [SystemMessage(content=filled_prompt)]
        response = llm_with_tools.invoke(messages)
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "current_agent": "coordinator"
        }
    else:
        # Just update state without responding
        return state

# Specialized profiler agent for in-depth user assessment
def profiler_agent(state: AgentState) -> AgentState:
    """Specialized agent for in-depth user assessment and profiling."""
    
    profiler_prompt = """You are a fitness assessment and profiling specialist.
    
    User model: {{user_model}}
    
    Your specialized capabilities:
    1. Comprehensive fitness assessment methodology
    2. Psychological profiling for motivation and adherence factors
    3. Goal elicitation and refinement techniques
    4. Learning style and communication preference assessment
    
    Ask targeted questions based on gaps in the user model.
    Focus on building a complete picture of the user's needs and constraints.
    """
    
    # Fill template
    filled_prompt = profiler_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    agent_state["profiler_agent"] = "active"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "current_agent": "profiler_agent",
        "agent_state": agent_state
    }

# Research specialist agent for fitness knowledge
def research_agent(state: AgentState) -> AgentState:
    """Specialized agent for retrieving and synthesizing fitness knowledge."""
    
    research_prompt = """You are a fitness research and knowledge specialist.
    
    User model: {{user_model}}
    Working memory: {{working_memory}}
    
    Your specialized capabilities:
    1. Access to evidence-based fitness research
    2. Critical analysis of fitness methodologies
    3. Translation of scientific concepts to practical applications
    4. Personalized recommendations based on research findings
    
    Use the retrieve_from_rag tool to access relevant scientific information.
    Synthesize findings in a clear, actionable manner relevant to the user's needs.
    """
    
    # Fill template
    filled_prompt = research_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{working_memory}}", str(state.get("working_memory", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    agent_state["research_agent"] = "active"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "current_agent": "research_agent",
        "agent_state": agent_state
    }

# Program design specialist agent
def planner_agent(state: AgentState) -> AgentState:
    """Specialized agent for creating sophisticated workout plans."""
    
    planner_prompt = """You are a fitness program design specialist.
    
    User model: {{user_model}}
    Working memory: {{working_memory}}
    
    Your specialized capabilities:
    1. Periodized program design based on sports science principles
    2. Exercise selection optimization for specific goals
    3. Progression modeling for continuous adaptation
    4. Integration of recovery and volume management
    
    Create a scientifically-sound, personalized workout plan.
    Format the plan for implementation in the Hevy app.
    """
    
    # Fill template
    filled_prompt = planner_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{working_memory}}", str(state.get("working_memory", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    agent_state["planner_agent"] = "active"
    
    # Store the workout plan
    fitness_plan = state.get("fitness_plan", {})
    fitness_plan["plan_content"] = response.content
    fitness_plan["created_date"] = datetime.now().isoformat()
    fitness_plan["version"] = fitness_plan.get("version", 0) + 1
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "current_agent": "planner_agent",
        "agent_state": agent_state,
        "fitness_plan": fitness_plan
    }

# Progress analysis specialist agent
def analyst_agent(state: AgentState) -> AgentState:
    """Specialized agent for analyzing workout data and progress."""
    
    analyst_prompt = """You are a fitness data analysis specialist.
    
    User model: {{user_model}}
    Fitness plan: {{fitness_plan}}
    
    Your specialized capabilities:
    1. Workout data analysis with statistical methods
    2. Progress tracking across multiple dimensions
    3. Pattern recognition in training responses
    4. Predictive modeling for future progress
    
    Use tool_fetch_workouts to retrieve workout data.
    Analyze the data for adherence, progress, and areas for improvement.
    Provide data-driven insights and visualizations when appropriate.
    """
    
    # Fill template
    filled_prompt = analyst_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = analyst_prompt.replace("{{fitness_plan}}", str(state.get("fitness_plan", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    agent_state["analyst_agent"] = "active"
    
    # Store analysis in working memory
    working_memory = state.get("working_memory", {})
    working_memory["latest_analysis"] = {
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "current_agent": "analyst_agent",
        "agent_state": agent_state,
        "working_memory": working_memory
    }

# Plan adaptation specialist agent
def adaptation_agent(state: AgentState) -> AgentState:
    """Specialized agent for adapting workout plans based on progress and feedback."""
    
    adaptation_prompt = """You are a workout program adaptation specialist.
    
    User model: {{user_model}}
    Fitness plan: {{fitness_plan}}
    Working memory: {{working_memory}}
    
    Your specialized capabilities:
    1. Intelligent workout modification based on progress data
    2. Adaptation strategies for plateaus and setbacks
    3. Progressive overload implementation
    4. Personalization based on user feedback and preferences
    
    Analyze the latest progress data and user feedback.
    Determine appropriate adaptations to the current fitness plan.
    Use tool_update_routine to implement changes in the Hevy app.
    """
    
    # Fill template
    filled_prompt = adaptation_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = adaptation_prompt.replace("{{fitness_plan}}", str(state.get("fitness_plan", {})))
    filled_prompt = adaptation_prompt.replace("{{working_memory}}", str(state.get("working_memory", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    agent_state["adaptation_agent"] = "active"
    
    # Update fitness plan with adaptations
    fitness_plan = state.get("fitness_plan", {})
    fitness_plan["adaptation_history"] = fitness_plan.get("adaptation_history", [])
    fitness_plan["adaptation_history"].append({
        "date": datetime.now().isoformat(),
        "changes": response.content,
        "version": fitness_plan.get("version", 0) + 0.1
    })
    fitness_plan["version"] = fitness_plan.get("version", 0) + 0.1
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "current_agent": "adaptation_agent",
        "agent_state": agent_state,
        "fitness_plan": fitness_plan
    }

# Motivational coaching specialist agent
def coach_agent(state: AgentState) -> AgentState:
    """Specialized agent for motivational coaching and adherence support."""
    
    coach_prompt = """You are a motivational fitness coach and psychology specialist.
    
    User model: {{user_model}}
    Working memory: {{working_memory}}
    
    Your specialized capabilities:
    1. Behavioral psychology for workout adherence
    2. Motivational interviewing techniques
    3. Obstacle identification and mitigation strategies
    4. Celebration of progress and achievement recognition
    
    Provide motivational support tailored to the user's mindset.
    Use positive reinforcement while maintaining accountability.
    Focus on building intrinsic motivation and self-efficacy.
    """
    
    # Fill template
    filled_prompt = coach_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = coach_prompt.replace("{{working_memory}}", str(state.get("working_memory", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    agent_state["coach_agent"] = "active"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "current_agent": "coach_agent",
        "agent_state": agent_state
    }

# Agent router based on current_agent
def agent_router(state: AgentState) -> str:
    return state["current_agent"]

# Build the multi-agent graph
workflow = StateGraph(AgentState)

# Add nodes for all agents and components
workflow.add_node("memory_manager", memory_manager)
workflow.add_node("reasoning_engine", reasoning_engine)
workflow.add_node("user_modeler", user_modeler)
workflow.add_node("coordinator", coordinator_agent)
workflow.add_node("profiler_agent", profiler_agent)
workflow.add_node("research_agent", research_agent)
workflow.add_node("planner_agent", planner_agent)
workflow.add_node("analyst_agent", analyst_agent)
workflow.add_node("adaptation_agent", adaptation_agent)
workflow.add_node("coach_agent", coach_agent)
workflow.add_node("tools", ToolNode([
    retrieve_from_rag, 
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine
]))

# Define the flow through the system
# 1. Start with the memory manager to update state
workflow.add_edge("memory_manager", "user_modeler")

# 2. Update user model before reasoning
workflow.add_edge("user_modeler", "reasoning_engine")

# 3. Reasoning engine selects the appropriate agent
workflow.add_conditional_edges(
    "reasoning_engine",
    agent_router,
    {
        "coordinator": "coordinator",
        "profiler_agent": "profiler_agent",
        "research_agent": "research_agent",
        "planner_agent": "planner_agent",
        "analyst_agent": "analyst_agent",
        "adaptation_agent": "adaptation_agent",
        "coach_agent": "coach_agent"
    }
)

# 4. All specialized agents return to the coordinator
for agent in ["profiler_agent", "research_agent", "planner_agent", 
              "analyst_agent", "adaptation_agent", "coach_agent"]:
    workflow.add_edge(agent, "coordinator")

# 5. Coordinator completes the interaction or continues
workflow.add_conditional_edges(
    "coordinator",
    lambda state: "end" if state.get("working_memory", {}).get("should_end", False) else "memory_manager",
    {
        "memory_manager": "memory_manager",
        "end": END
    }
)

# 6. Add tool usage for all agents
for agent in ["coordinator", "profiler_agent", "research_agent", "planner_agent", 
              "analyst_agent", "adaptation_agent", "coach_agent"]:
    workflow.add_conditional_edges(
        agent,
        tools_condition,
        {
            True: "tools",
            False: agent  # No-op if no tool usage
        }
    )

# 7. Tool returns to the agent that called it
workflow.add_conditional_edges(
    "tools",
    agent_router,
    {
        "coordinator": "coordinator",
        "profiler_agent": "profiler_agent",
        "research_agent": "research_agent",
        "planner_agent": "planner_agent", 
        "analyst_agent": "analyst_agent",
        "adaptation_agent": "adaptation_agent",
        "coach_agent": "coach_agent"
    }
)

# Set entry point
workflow.set_entry_point("memory_manager")

# Compile the graph
multi_agent = workflow.compile()
