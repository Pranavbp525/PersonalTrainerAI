"""
Enhanced Multi-Agent Architecture for Personal Fitness Trainer AI

This implementation combines the strengths of the multi-agent system with
specialized components for fitness training, RAG integration, and Hevy API usage.
"""

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
from pydantic import BaseModel, Field
from llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools([
    retrieve_from_rag,
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine
])

# Define comprehensive state model
class AgentState(TypedDict):
    # Core state
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    
    # Memory components
    memory: Dict[str, Any]  # Long-term memory storage
    working_memory: Dict[str, Any]  # Short-term contextual memory
    
    # User and fitness data
    user_model: Dict[str, Any]  # Comprehensive user profile
    fitness_plan: Dict[str, Any]  # Structured fitness plan
    progress_data: Dict[str, Any]  # Progress tracking data
    
    # Agent management
    reasoning_trace: List[Dict[str, Any]]  # Traces of reasoning steps
    agent_state: Dict[str, str]  # Current state of each agent
    current_agent: str  # Currently active agent
    
    # Tool interaction
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Helper functions for data extraction
def extract_principles(text: str) -> List[str]:
    """Extract scientific principles from text."""
    # Implementation would parse the text to identify key principles
    # For now, return a placeholder
    return ["principle1", "principle2"]

def extract_approaches(text: str) -> List[Dict]:
    """Extract training approaches from text."""
    # Implementation would parse the text to identify approaches
    # For now, return a placeholder
    return [{"name": "approach1", "description": "description1"}]

def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text."""
    # Implementation would parse the text to identify citations
    # For now, return a placeholder
    return [{"source": "Jeff Nippard", "content": "content1"}]

def extract_routine_data(text: str) -> Dict:
    """Extract structured routine data for Hevy API."""
    # Implementation would parse the text to create a structured routine
    # For now, return a placeholder
    return {
        "name": "Routine",
        "description": "Description",
        "workouts": []
    }

def extract_adherence_rate(text: str) -> float:
    """Extract adherence rate from analysis text."""
    # Implementation would parse the text to identify adherence rate
    # For now, return a placeholder
    return 0.85

def extract_progress_metrics(text: str) -> Dict:
    """Extract progress metrics from analysis text."""
    # Implementation would parse the text to identify progress metrics
    # For now, return a placeholder
    return {"strength": 0.1, "endurance": 0.05}

def extract_issues(text: str) -> List[str]:
    """Extract identified issues from analysis text."""
    # Implementation would parse the text to identify issues
    # For now, return a placeholder
    return ["issue1", "issue2"]

def extract_adjustments(text: str) -> List[Dict]:
    """Extract suggested adjustments from analysis text."""
    # Implementation would parse the text to identify adjustments
    # For now, return a placeholder
    return [{"target": "exercise1", "change": "increase weight"}]

# Memory manager for sophisticated state management
def memory_manager(state: AgentState) -> AgentState:
    """Manages long-term and working memory, consolidating information and pruning as needed."""
    memory_prompt = """You are the memory manager for a fitness training system. Review the conversation history and current agent states to:
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
    response = llm.invoke(messages)
    
    # Update memory structures
    memory = state.get("memory", {})
    working_memory = state.get("working_memory", {})
    user_model = state.get("user_model", {})
    
    # Extract recent messages for context
    recent_exchanges = []
    for msg in state["messages"][-10:]:  # Last 10 messages
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            recent_exchanges.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
    
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
    reasoning_prompt = """You are the reasoning engine for a fitness training system. Analyze the current situation using sophisticated reasoning to determine:
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
    response = llm.invoke(messages)
    
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
    agent_selection = agent_selector(state, response.content)
    
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

# Agent selector function
def agent_selector(state: AgentState, reasoning_text: str = "") -> str:
    """Determines which agent should handle the current interaction."""
    # For new users, ensure comprehensive assessment
    if not state.get("user_model") or state.get("user_model", {}).get("assessment_complete") != True:
        return "assessment_agent"
    
    # For users with assessment but no plan, prioritize research and planning
    if state.get("user_model") and not state.get("fitness_plan"):
        if not state.get("working_memory", {}).get("research_findings"):
            return "research_agent"
        else:
            return "planning_agent"
    
    # For users with existing plans, check if analysis is needed
    if state.get("fitness_plan") and state.get("working_memory", {}).get("last_analysis_date"):
        last_analysis = datetime.fromisoformat(state.get("working_memory", {}).get("last_analysis_date"))
        if (datetime.now() - last_analysis).days >= 7:  # Weekly analysis
            return "progress_analysis_agent"
    
    # Use reasoning text to determine appropriate agent
    if reasoning_text:
        if "assessment" in reasoning_text.lower() or "profile" in reasoning_text.lower():
            return "assessment_agent"
        elif "research" in reasoning_text.lower() or "knowledge" in reasoning_text.lower():
            return "research_agent"
        elif "plan" in reasoning_text.lower() or "routine" in reasoning_text.lower():
            return "planning_agent"
        elif "progress" in reasoning_text.lower() or "analyze" in reasoning_text.lower():
            return "progress_analysis_agent"
        elif "adjust" in reasoning_text.lower() or "modify" in reasoning_text.lower():
            return "adaptation_agent"
        elif "motivate" in reasoning_text.lower() or "coach" in reasoning_text.lower():
            return "coach_agent"
    
    # Default to coordinator for general interactions
    return "coordinator_agent"

# User modeler for comprehensive user understanding
def user_modeler(state: AgentState) -> AgentState:
    """Builds and maintains a comprehensive model of the user."""
    modeling_prompt = """You are a user modeling specialist for a fitness training system. Analyze all available information about the user to build a comprehensive model:
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
    response = llm.invoke(messages)
    
    # Update user model
    user_model = state.get("user_model", {})
    user_model["last_updated"] = datetime.now().isoformat()
    user_model["model_version"] = user_model.get("model_version", 0) + 1
    
    # In a real implementation, we would parse the response to update specific fields
    # For now, we'll just store the raw response
    user_model["latest_analysis"] = response.content
    
    return {
        **state,
        "user_model": user_model
    }

# Coordinator agent for overall orchestration
def coordinator_agent(state: AgentState) -> AgentState:
    """Central coordinator that manages the overall interaction flow."""
    coordinator_prompt = """You are the coordinator for a personal fitness trainer AI. Your role is to:
    1. Understand the user's current needs and context
    2. Determine which specialized agent should handle the interaction
    3. Provide a coherent experience across different agents
    4. Ensure all user needs are addressed appropriately
    
    You have access to these specialized agents:
    - Assessment Agent: Gathers user information and builds profiles
    - Research Agent: Retrieves scientific fitness knowledge
    - Planning Agent: Creates personalized workout routines
    - Progress Analysis Agent: Analyzes workout data and tracks progress
    - Adaptation Agent: Modifies workout plans based on progress and feedback
    - Coach Agent: Provides motivation and adherence strategies
    
    Current user model: {{user_model}}
    Current fitness plan: {{fitness_plan}}
    Recent interactions: {{recent_exchanges}}
    
    Respond to the user and indicate which agent should handle the next step.
    """
    
    # Fill template
    filled_prompt = coordinator_prompt.replace("{{user_model}}", str(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{fitness_plan}}", str(state.get("fitness_plan", {})))
    filled_prompt = filled_prompt.replace("{{recent_exchanges}}", 
                                         str(state.get("working_memory", {}).get("recent_exchanges", [])))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm.invoke(messages)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

# Assessment agent for user profiling
def assessment_agent(state: AgentState) -> AgentState:
    """Handles comprehensive user assessment and profile building."""
    assessment_prompt = """You are a fitness assessment specialist. Your goal is to build a comprehensive user profile by asking targeted questions about:
    1. Fitness goals (strength, hypertrophy, endurance, weight loss, etc.)
    2. Training experience and current fitness level
    3. Available equipment and training environment
    4. Schedule and time availability
    5. Physical limitations, injuries, or health concerns
    6. Dietary preferences and restrictions
    7. Measurement data (if willing to share)
    8. Previous workout history and experiences
    
    Current profile: {{user_profile}}
    
    Be conversational but thorough. Focus on gathering missing information based on what you already know.
    If this is the first interaction, introduce yourself and begin the assessment process.
    """
    
    # Fill template
    filled_prompt = assessment_prompt.replace("{{user_profile}}", str(state.get("user_model", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update user model with assessment information
    # In a real implementation, we would extract specific profile attributes
    user_model = state.get("user_model", {})
    
    # Mark assessment as complete if we have sufficient information
    # This is a simplified check - in reality, we would verify specific fields
    if len(state["messages"]) > 5:  # Arbitrary threshold for demonstration
        user_model["assessment_complete"] = True
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "user_model": user_model
    }

# Research agent for scientific knowledge retrieval
def research_agent(state: AgentState) -> AgentState:
    """Retrieves and synthesizes scientific fitness knowledge from RAG."""
    research_prompt = """You are a fitness research specialist. Based on the user's profile and current needs:
    1. Identify key scientific principles relevant to their goals
    2. Retrieve evidence-based approaches from fitness influencers like Jeff Nippard, Dr. Mike Isratel, and Jeremy Ethier
    3. Synthesize this information into actionable insights
    4. Provide citations to specific sources
    
    Current user profile: {{user_profile}}
    Current research needs: {{research_needs}}
    
    Use the retrieve_from_rag tool to access scientific fitness information.
    """
    
    # Fill template with state data
    filled_prompt = research_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{research_needs}}", 
                                         json.dumps(state.get("working_memory", {}).get("research_needs", [])))
    
    # Use RAG for knowledge retrieval
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update working memory with structured research findings
    working_memory = state.get("working_memory", {})
    working_memory["research_findings"] = {
        "principles": extract_principles(response.content),
        "approaches": extract_approaches(response.content),
        "citations": extract_citations(response.content),
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "working_memory": working_memory
    }

# Planning agent for workout routine creation
def planning_agent(state: AgentState) -> AgentState:
    """Creates personalized workout routines and exports to Hevy."""
    planning_prompt = """You are a workout programming specialist. Create a detailed, personalized workout plan:
    1. Design a structured routine based on scientific principles and user profile
    2. Format the routine specifically for Hevy app integration
    3. Include exercise selection, sets, reps, rest periods, and progression scheme
    4. Provide clear instructions for implementation
    
    User profile: {{user_profile}}
    Research findings: {{research_findings}}
    
    Use the tool_create_routine tool to save the routine to Hevy.
    """
    
    # Fill template with state data
    filled_prompt = planning_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{research_findings}}", 
                                         json.dumps(state.get("working_memory", {}).get("research_findings", {})))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Extract structured routine data for Hevy API
    routine_data = extract_routine_data(response.content)
    
    # In a real implementation, we would use the Hevy API to create the routine
    # For now, we'll simulate the result
    hevy_result = {"routine_id": str(uuid.uuid4())}
    
    # Update fitness plan with both content and Hevy reference
    fitness_plan = state.get("fitness_plan", {})
    fitness_plan["content"] = response.content
    fitness_plan["hevy_routine_id"] = hevy_result.get("routine_id")
    fitness_plan["created_at"] = datetime.now().isoformat()
    fitness_plan["version"] = fitness_plan.get("version", 0) + 1
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "fitness_plan": fitness_plan
    }

# Progress analysis agent for workout log analysis
def progress_analysis_agent(state: AgentState) -> AgentState:
    """Analyzes workout logs to track progress and suggest adjustments."""
    analysis_prompt = """You are a fitness progress analyst. Examine the user's workout logs to:
    1. Track adherence to the planned routine
    2. Identify trends in performance (improvements, plateaus, regressions)
    3. Compare actual progress against expected progress
    4. Suggest specific adjustments to optimize results
    
    User profile: {{user_profile}}
    Current fitness plan: {{fitness_plan}}
    Recent workout logs: {{workout_logs}}
    
    Use the tool_fetch_workouts tool to access workout logs from Hevy.
    """
    
    # In a real implementation, we would fetch workout logs from Hevy API
    # For now, we'll simulate the result
    workout_logs = []
    
    # Fill template with state data
    filled_prompt = analysis_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    filled_prompt = filled_prompt.replace("{{workout_logs}}", json.dumps(workout_logs))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Extract structured analysis results
    analysis_results = {
        "adherence_rate": extract_adherence_rate(response.content),
        "progress_metrics": extract_progress_metrics(response.content),
        "identified_issues": extract_issues(response.content),
        "suggested_adjustments": extract_adjustments(response.content),
        "analysis_date": datetime.now().isoformat()
    }
    
    # Update working memory with analysis date
    working_memory = state.get("working_memory", {})
    working_memory["last_analysis_date"] = datetime.now().isoformat()
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "progress_data": state.get("progress_data", {}) | {"latest_analysis": analysis_results},
        "working_memory": working_memory
    }

# Adaptation agent for routine modification
def adaptation_agent(state: AgentState) -> AgentState:
    """Modifies workout routines based on progress and feedback."""
    adaptation_prompt = """You are a workout adaptation specialist. Based on progress data and user feedback:
    1. Identify specific aspects of the routine that need modification
    2. Apply scientific principles to make appropriate adjustments
    3. Ensure changes align with the user's goals and constraints
    4. Update the routine in Hevy
    
    User profile: {{user_profile}}
    Current fitness plan: {{fitness_plan}}
    Progress data: {{progress_data}}
    Suggested adjustments: {{suggested_adjustments}}
    
    Use the tool_update_routine tool to update the routine in Hevy.
    """
    
    # Fill template with state data
    filled_prompt = adaptation_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    filled_prompt = filled_prompt.replace("{{progress_data}}", json.dumps(state.get("progress_data", {})))
    filled_prompt = filled_prompt.replace("{{suggested_adjustments}}", 
                                         json.dumps(state.get("progress_data", {})
                                                   .get("latest_analysis", {})
                                                   .get("suggested_adjustments", [])))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update fitness plan with modifications
    fitness_plan = state.get("fitness_plan", {})
    fitness_plan["content"] = response.content
    fitness_plan["updated_at"] = datetime.now().isoformat()
    fitness_plan["version"] = fitness_plan.get("version", 0) + 1
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "fitness_plan": fitness_plan
    }

# Coach agent for motivation and adherence
def coach_agent(state: AgentState) -> AgentState:
    """Provides motivation, adherence strategies, and behavioral coaching."""
    coach_prompt = """You are a fitness motivation coach. Your role is to:
    1. Provide encouragement and motivation tailored to the user's profile
    2. Offer strategies to improve adherence and consistency
    3. Address psychological barriers to fitness progress
    4. Celebrate achievements and milestones
    
    User profile: {{user_profile}}
    Progress data: {{progress_data}}
    Recent exchanges: {{recent_exchanges}}
    
    Be supportive, empathetic, and science-based in your approach.
    """
    
    # Fill template with state data
    filled_prompt = coach_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{progress_data}}", json.dumps(state.get("progress_data", {})))
    filled_prompt = filled_prompt.replace("{{recent_exchanges}}", 
                                         json.dumps(state.get("working_memory", {}).get("recent_exchanges", [])))
    
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = llm.invoke(messages)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

# Build the graph
def build_fitness_trainer_graph():
    """Constructs the multi-agent graph for the fitness trainer."""
    workflow = StateGraph(AgentState)
    
    # Add core nodes
    workflow.add_node("coordinator_agent", coordinator_agent)
    workflow.add_node("reasoning_engine", reasoning_engine)
    workflow.add_node("memory_manager", memory_manager)
    workflow.add_node("user_modeler", user_modeler)
    
    # Add specialized agent nodes
    workflow.add_node("assessment_agent", assessment_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("planning_agent", planning_agent)
    workflow.add_node("progress_analysis_agent", progress_analysis_agent)
    workflow.add_node("adaptation_agent", adaptation_agent)
    workflow.add_node("coach_agent", coach_agent)
    
    # Add tools node
    workflow.add_node("tools", ToolNode([
        retrieve_from_rag,
        tool_fetch_workouts,
        tool_get_workout_count,
        tool_fetch_routines,
        tool_update_routine,
        tool_create_routine
    ]))
    
    # Add core flow edges
    workflow.add_edge("coordinator_agent", "reasoning_engine")
    workflow.add_edge("reasoning_engine", "memory_manager")
    workflow.add_edge("memory_manager", "user_modeler")
    
    # Add conditional edges from user_modeler to specialized agents
    workflow.add_conditional_edges(
        "user_modeler",
        lambda state: state.get("current_agent", "coordinator_agent"),
        {
            "coordinator_agent": "coordinator_agent",
            "assessment_agent": "assessment_agent",
            "research_agent": "research_agent",
            "planning_agent": "planning_agent",
            "progress_analysis_agent": "progress_analysis_agent",
            "adaptation_agent": "adaptation_agent",
            "coach_agent": "coach_agent"
        }
    )
    
    # Add tool usage edges for all agent nodes
    for node in [
        "coordinator_agent", "assessment_agent", "research_agent", 
        "planning_agent", "progress_analysis_agent", "adaptation_agent", "coach_agent"
    ]:
        workflow.add_conditional_edges(
            node,
            tools_condition,
            {
                True: "tools",
                False: "coordinator_agent"  # Return to coordinator after agent completes
            }
        )
    
    # Add edge from tools back to the originating agent
    workflow.add_edge("tools", "coordinator_agent")
    
    # Set entry point
    workflow.set_entry_point("coordinator_agent")
    
    return workflow.compile()

# Initialize the agent
fitness_trainer_agent = build_fitness_trainer_graph()

# Example usage
if __name__ == "__main__":
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Hi, I want to start a new workout routine.")],
        "session_id": str(uuid.uuid4()),
        "memory": {},
        "working_memory": {},
        "user_model": {},
        "fitness_plan": {},
        "progress_data": {},
        "reasoning_trace": [],
        "agent_state": {},
        "current_agent": "coordinator_agent"
    }
    
    # Run the agent
    result = fitness_trainer_agent.invoke(initial_state)
    
    # Print the result
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
