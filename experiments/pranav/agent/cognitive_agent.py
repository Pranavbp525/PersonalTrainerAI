from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import uuid
from pydantic import BaseModel, Field
from experiments.pranav.chatbot.agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools([retrieve_from_rag, 
                                 tool_fetch_workouts,
                                 tool_get_workout_count,
                                 tool_fetch_routines,
                                 tool_update_routine,
                                 tool_create_routine,
                                 retrieve_from_rag])


# Advanced state model with cognitive architecture layers
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    episodic_memory: Dict[str, Any]  # Memory of interactions and events
    semantic_memory: Dict[str, Any]  # Conceptual knowledge and facts
    working_memory: Dict[str, Any]  # Active short-term memory
    user_model: Dict[str, Any]  # Comprehensive user model
    fitness_domain: Dict[str, Any]  # Domain-specific knowledge
    metacognition: Dict[str, Any]  # System's awareness of its own state and performance
    current_plan: Dict[str, Any]  # Current workout or interaction plan
    execution_trace: List[Dict]  # Record of actions and outcomes
    reflection_log: List[Dict]  # Self-evaluation and improvement notes
    controller_state: str  # Current state of the hierarchical controller
    error_state: Optional[Dict]  # Current error state, if any
    human_feedback: Optional[Dict]  # Explicit feedback from human user
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Controller states
CONTROLLER_STATES = Literal[
    "perception", "interpretation", "planning", "execution", 
    "monitoring", "reflection", "adaptation", "error_recovery", "end"
]

# Cognitive cycle step 1: Perception
def perception_node(state: AgentState) -> AgentState:
    """Process new inputs and update working memory with relevant perceptions."""
    
    perception_prompt = """You are the perception component of an advanced fitness training system.
    
    Your task is to:
    1. Process all new inputs from the user or environment
    2. Identify salient features and important information
    3. Integrate this information into working memory
    4. Tag information with metadata for retrieval and processing
    
    Recent messages:
    {{recent_messages}}
    
    Current working memory:
    {{working_memory}}
    
    Process new perceptions without interpretation. Focus only on what is directly observed.
    """
    
    # Extract recent messages
    recent_messages = []
    for msg in state["messages"][-5:]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            recent_messages.append({"role": "human" if isinstance(msg, HumanMessage) else "ai", 
                                  "content": msg.content})
    
    # Fill prompt template
    filled_prompt = perception_prompt.replace("{{recent_messages}}", json.dumps(recent_messages, indent=2))
    filled_prompt = filled_prompt.replace("{{working_memory}}", json.dumps(state.get("working_memory", {}), indent=2))
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update working memory with new perceptions
    working_memory = state.get("working_memory", {})
    working_memory["current_perceptions"] = {
        "timestamp": datetime.now().isoformat(),
        "content": response.content
    }
    
    return {
        **state,
        "working_memory": working_memory,
        "controller_state": "interpretation"
    }

# Cognitive cycle step 2: Interpretation
def interpretation_node(state: AgentState) -> AgentState:
    """Interpret perceptions in context of prior knowledge."""
    
    interpretation_prompt = """You are the interpretation component of an advanced fitness training system.
    
    Your task is to:
    1. Analyze current perceptions in the context of prior knowledge
    2. Infer user's intent, emotional state, and fitness needs
    3. Update user model with new interpretations
    4. Identify potential knowledge gaps to address
    
    Current perceptions:
    {{current_perceptions}}
    
    User model:
    {{user_model}}
    
    Semantic memory (knowledge):
    {{semantic_memory}}
    
    Provide a structured interpretation that connects perceptions to existing knowledge.
    """
    
    # Fill prompt template
    filled_prompt = interpretation_prompt.replace(
        "{{current_perceptions}}", 
        json.dumps(state.get("working_memory", {}).get("current_perceptions", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{user_model}}", 
        json.dumps(state.get("user_model", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{semantic_memory}}", 
        json.dumps(state.get("semantic_memory", {}), indent=2)
    )
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update working memory with interpretation
    working_memory = state.get("working_memory", {})
    working_memory["current_interpretation"] = {
        "timestamp": datetime.now().isoformat(),
        "content": response.content
    }
    
    # Update user model based on interpretation
    user_model = state.get("user_model", {})
    
    # In a real system, we'd parse the interpretation to update the user model
    # Here we're just tracking that we did the interpretation
    user_model["last_interpretation"] = datetime.now().isoformat()
    
    return {
        **state,
        "working_memory": working_memory,
        "user_model": user_model,
        "controller_state": "planning"
    }

# Cognitive cycle step 3: Planning
def planning_node(state: AgentState) -> AgentState:
    """Develop multi-level plans based on interpreted information."""
    
    planning_prompt = """You are the planning component of an advanced fitness training system.
    
    Your task is to:
    1. Develop a hierarchical plan to address the user's needs
    2. Create high-level fitness goals broken down into actionable steps
    3. Incorporate contingencies for potential obstacles
    4. Ensure plans align with scientific principles and the user's constraints
    
    Current interpretation:
    {{current_interpretation}}
    
    User model:
    {{user_model}}
    
    Existing plans:
    {{current_plan}}
    
    Develop a structured plan with clear objectives, actions, and success criteria.
    Your plan should include short-term actions and long-term strategy.
    """
    
    # Fill prompt template
    filled_prompt = planning_prompt.replace(
        "{{current_interpretation}}", 
        json.dumps(state.get("working_memory", {}).get("current_interpretation", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{user_model}}", 
        json.dumps(state.get("user_model", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{current_plan}}", 
        json.dumps(state.get("current_plan", {}), indent=2)
    )
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update current plan
    current_plan = state.get("current_plan", {})
    current_plan["plan_content"] = response.content
    current_plan["created_at"] = datetime.now().isoformat()
    current_plan["version"] = current_plan.get("version", 0) + 1
    
    # Update working memory with plan
    working_memory = state.get("working_memory", {})
    working_memory["current_plan_summary"] = {
        "timestamp": datetime.now().isoformat(),
        "content": response.content[:500] + "..." if len(response.content) > 500 else response.content
    }
    
    return {
        **state,
        "working_memory": working_memory,
        "current_plan": current_plan,
        "controller_state": "execution"
    }

# Cognitive cycle step 4: Execution
def execution_node(state: AgentState) -> AgentState:
    """Execute the plan and interact with the user or environment."""
    
    execution_prompt = """You are the execution component of an advanced fitness training system.
    
    Your task is to:
    1. Implement the current plan through direct communication with the user
    2. Use appropriate tools to update workout routines or retrieve information
    3. Maintain a supportive, expert tone aligned with fitness coaching best practices
    4. Document actions taken for monitoring and reflection
    
    Current plan to execute:
    {{current_plan}}
    
    User model:
    {{user_model}}
    
    Respond to the user directly, implementing the current plan in a natural, conversational manner.
    Use scientific knowledge and motivational techniques appropriate to the user's needs.
    """
    
    # Fill prompt template
    filled_prompt = execution_prompt.replace(
        "{{current_plan}}", 
        json.dumps(state.get("current_plan", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{user_model}}", 
        json.dumps(state.get("user_model", {}), indent=2)
    )
    
    # Only respond if the last message is from the user
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
        
        # Record execution in trace
        execution_trace = state.get("execution_trace", [])
        execution_trace.append({
            "timestamp": datetime.now().isoformat(),
            "action": "user_communication",
            "content": response.content,
            "plan_reference": state.get("current_plan", {}).get("version")
        })
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "execution_trace": execution_trace,
            "controller_state": "monitoring"
        }
    else:
        # No response needed, move to monitoring
        return {
            **state,
            "controller_state": "monitoring"
        }

# Cognitive cycle step 5: Monitoring
def monitoring_node(state: AgentState) -> AgentState:
    """Monitor execution and detect deviations or errors."""
    
    monitoring_prompt = """You are the monitoring component of an advanced fitness training system.
    
    Your task is to:
    1. Evaluate the execution against the planned objectives
    2. Detect deviations, errors, or unexpected user responses
    3. Assess user engagement and emotional response
    4. Identify opportunities for adaptation
    
    Current plan:
    {{current_plan}}
    
    Recent execution:
    {{recent_execution}}
    
    User responses:
    {{user_responses}}
    
    Provide a structured assessment of execution quality and any issues detected.
    """
    
    # Extract user responses
    user_responses = []
    for msg in state["messages"][-3:]:
        if isinstance(msg, HumanMessage):
            user_responses.append(msg.content)
    
    # Get recent execution from trace
    recent_execution = state.get("execution_trace", [])[-3:] if state.get("execution_trace") else []
    
    # Fill prompt template
    filled_prompt = monitoring_prompt.replace(
        "{{current_plan}}", 
        json.dumps(state.get("current_plan", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{recent_execution}}", 
        json.dumps(recent_execution, indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{user_responses}}", 
        json.dumps(user_responses, indent=2)
    )
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Determine if there's an error or if we should reflect
    error_detected = "error" in response.content.lower() or "issue" in response.content.lower()
    
    # Update working memory
    working_memory = state.get("working_memory", {})
    working_memory["monitoring_results"] = {
        "timestamp": datetime.now().isoformat(),
        "content": response.content,
        "error_detected": error_detected
    }
    
    # Set next state based on monitoring results
    next_state = "error_recovery" if error_detected else "reflection"
    
    # If error detected, capture error state
    error_state = None
    if error_detected:
        error_state = {
            "timestamp": datetime.now().isoformat(),
            "description": response.content,
            "error_context": {
                "current_plan": state.get("current_plan", {}),
                "recent_messages": [m.content for m in state["messages"][-3:] if isinstance(m, (HumanMessage, AIMessage))]
            }
        }
    
    return {
        **state,
        "working_memory": working_memory,
        "error_state": error_state,
        "controller_state": next_state
    }

# Cognitive cycle step 6: Reflection
def reflection_node(state: AgentState) -> AgentState:
    """Reflect on performance and identify improvements."""
    
    reflection_prompt = """You are the reflection component of an advanced fitness training system.
    
    Your task is to:
    1. Evaluate overall performance and effectiveness
    2. Identify patterns across multiple interactions
    3. Recognize opportunities for knowledge enhancement
    4. Develop specific improvements to agent behavior
    
    Recent monitoring results:
    {{monitoring_results}}
    
    Execution history:
    {{execution_trace}}
    
    Metacognition state:
    {{metacognition}}
    
    Provide thoughtful reflection on system performance and concrete improvement ideas.
    """
    
    # Fill prompt template
    filled_prompt = reflection_prompt.replace(
        "{{monitoring_results}}", 
        json.dumps(state.get("working_memory", {}).get("monitoring_results", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{execution_trace}}", 
        json.dumps(state.get("execution_trace", [])[-5:], indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{metacognition}}", 
        json.dumps(state.get("metacognition", {}), indent=2)
    )
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update reflection log
    reflection_log = state.get("reflection_log", [])
    reflection_log.append({
        "timestamp": datetime.now().isoformat(),
        "content": response.content
    })
    
    # Update metacognition with insights
    metacognition = state.get("metacognition", {})
    metacognition["last_reflection"] = datetime.now().isoformat()
    metacognition["improvement_ideas"] = metacognition.get("improvement_ideas", [])
    metacognition["improvement_ideas"].append({
        "timestamp": datetime.now().isoformat(),
        "content": response.content
    })
    
    # Decide whether to adapt or end
    adapt_needed = "adapt" in response.content.lower() or "improve" in response.content.lower() or "change" in response.content.lower()
    
    return {
        **state,
        "reflection_log": reflection_log,
        "metacognition": metacognition,
        "controller_state": "adaptation" if adapt_needed else "end"
    }

# Cognitive cycle step 7: Adaptation
def adaptation_node(state: AgentState) -> AgentState:
    """Adapt behavior, knowledge, or plans based on reflection."""
    
    adaptation_prompt = """You are the adaptation component of an advanced fitness training system.
    
    Your task is to:
    1. Implement changes based on reflection insights
    2. Update domain knowledge with new information
    3. Adjust the user model to reflect new understanding
    4. Refine plans to better meet objectives
    
    Recent reflection:
    {{recent_reflection}}
    
    Current plan:
    {{current_plan}}
    
    User model:
    {{user_model}}
    
    Specify concrete changes to implement across system components.
    """
    
    # Get most recent reflection
    recent_reflection = state.get("reflection_log", [])[-1] if state.get("reflection_log") else {}
    
    # Fill prompt template
    filled_prompt = adaptation_prompt.replace(
        "{{recent_reflection}}", 
        json.dumps(recent_reflection, indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{current_plan}}", 
        json.dumps(state.get("current_plan", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{user_model}}", 
        json.dumps(state.get("user_model", {}), indent=2)
    )
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Update plan with adaptations
    current_plan = state.get("current_plan", {})
    current_plan["adaptations"] = current_plan.get("adaptations", [])
    current_plan["adaptations"].append({
        "timestamp": datetime.now().isoformat(),
        "content": response.content
    })
    current_plan["version"] = current_plan.get("version", 0) + 0.1
    
    # Update user model based on adaptation insights
    user_model = state.get("user_model", {})
    user_model["last_adaptation"] = datetime.now().isoformat()
    
    return {
        **state,
        "current_plan": current_plan,
        "user_model": user_model,
        "controller_state": "perception"  # Return to perception to start a new cycle
    }

# Error recovery module
def error_recovery_node(state: AgentState) -> AgentState:
    """Handle detected errors with graceful recovery."""
    
    recovery_prompt = """You are the error recovery component of an advanced fitness training system.
    
    Your task is to:
    1. Analyze the detected error and its severity
    2. Develop an appropriate recovery strategy
    3. Implement immediate corrections in user communication
    4. Update the system to prevent similar errors
    
    Error details:
    {{error_state}}
    
    Current plan:
    {{current_plan}}
    
    Recent messages:
    {{recent_messages}}
    
    Develop a recovery plan that addresses the error while maintaining user trust.
    """
    
    # Extract recent messages
    recent_messages = []
    for msg in state["messages"][-5:]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            recent_messages.append({"role": "human" if isinstance(msg, HumanMessage) else "ai", 
                                  "content": msg.content})
    
    # Fill prompt template
    filled_prompt = recovery_prompt.replace(
        "{{error_state}}", 
        json.dumps(state.get("error_state", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{current_plan}}", 
        json.dumps(state.get("current_plan", {}), indent=2)
    )
    filled_prompt = filled_prompt.replace(
        "{{recent_messages}}", 
        json.dumps(recent_messages, indent=2)
    )
    
    messages = [SystemMessage(content=filled_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Implement recovery by directly responding to user
    recovery_message = f"{response.content}"
    
    # Record recovery action
    execution_trace = state.get("execution_trace", [])
    execution_trace.append({
        "timestamp": datetime.now().isoformat(),
        "action": "error_recovery",
        "content": recovery_message,
        "error_reference": state.get("error_state", {}).get("timestamp")
    })
    
    # Update metacognition with error handling
    metacognition = state.get("metacognition", {})
    metacognition["error_history"] = metacognition.get("error_history", [])
    metacognition["error_history"].append({
        "timestamp": datetime.now().isoformat(),
        "error": state.get("error_state", {}),
        "recovery": recovery_message
    })
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=recovery_message)],
        "execution_trace": execution_trace,
        "metacognition": metacognition,
        "error_state": None,  # Clear error state
        "controller_state": "reflection"  # Move to reflection after recovery
    }

# Controller state router
def controller_router(state: AgentState) -> str:
    return state["controller_state"]

# Build the hierarchical cognitive graph
workflow = StateGraph(AgentState)

# Add cognitive architecture nodes
workflow.add_node("perception", perception_node)
workflow.add_node("interpretation", interpretation_node)
workflow.add_node("planning", planning_node)
workflow.add_node("execution", execution_node)
workflow.add_node("monitoring", monitoring_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("adaptation", adaptation_node)
workflow.add_node("error_recovery", error_recovery_node)
workflow.add_node("tools", ToolNode([
    retrieve_from_rag, 
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine
]))

# Connect cognitive cycle
workflow.add_conditional_edges(
    "perception",
    controller_router,
    {
        "interpretation": "interpretation",
        "planning": "planning",
        "execution": "execution",
        "monitoring": "monitoring",
        "reflection": "reflection",
        "adaptation": "adaptation",
        "error_recovery": "error_recovery",
        "end": END
    }
)

workflow.add_conditional_edges(
    "interpretation",
    controller_router,
    {
        "planning": "planning",
        "execution": "execution",
        "monitoring": "monitoring",
        "reflection": "reflection",
        "adaptation": "adaptation",
        "error_recovery": "error_recovery",
        "end": END
    }
)

workflow.add_conditional_edges(
    "planning",
    controller_router,
    {
        "execution": "execution",
        "monitoring": "monitoring",
        "reflection": "reflection",
        "adaptation": "adaptation",
        "error_recovery": "error_recovery",
        "end": END
    }
)

workflow.add_conditional_edges(
    "execution",
    controller_router,
    {
        "monitoring": "monitoring",
        "reflection": "reflection",
        "adaptation": "adaptation",
        "error_recovery": "error_recovery",
        "end": END
    }
)

workflow.add_conditional_edges(
    "monitoring",
    controller_router,
    {
        "reflection": "reflection",
        "adaptation": "adaptation",
        "error_recovery": "error_recovery",
        "end": END
    }
)

workflow.add_conditional_edges(
    "reflection",
    controller_router,
    {
        "adaptation": "adaptation",
        "perception": "perception",
        "end": END
    }
)

workflow.add_conditional_edges(
    "adaptation",
    controller_router,
    {
        "perception": "perception",
        "end": END
    }
)

workflow.add_conditional_edges(
    "error_recovery",
    controller_router,
    {
        "reflection": "reflection",
        "perception": "perception",
        "end": END
    }
)

# Add tool usage for applicable nodes
for node in ["execution"]:
    workflow.add_conditional_edges(
        node,
        tools_condition,
        {
            True: "tools",
            False: node
        }
    )

# Tool returns to the execution node
workflow.add_edge("tools", "execution")

# Set entry point
workflow.set_entry_point("perception")

# Compile the graph
cognitive_agent = workflow.compile()
