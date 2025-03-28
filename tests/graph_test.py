import pytest
import asyncio
from datetime import datetime
from src.chatbot.agent.graph import (
    coordinator_condition, agent_selector, get_or_create_state, save_state, 
    agent_with_error_handling, build_fitness_trainer_graph
)
from src.chatbot.agent.agent_models import AgentState

@pytest.mark.asyncio
async def test_coordinator_condition():
    state = {
        "current_agent": "assessment"
    }
    result = coordinator_condition(state)
    assert result == "end_conversation"

    state["current_agent"] = "end_conversation"
    result = coordinator_condition(state)
    assert result == "end_conversation"

    state["current_agent"] = "other_agent"
    result = coordinator_condition(state)
    assert result == "other_agent"

@pytest.mark.asyncio
async def test_agent_selector():
    state = {
        "agent_state": {"status": "incomplete"},
        "user_model": {"assessment_complete": False},
        "working_memory": {},
        "fitness_plan": None
    }
    result = await agent_selector(state)
    assert result == "assessment_agent"

    state["user_model"]["assessment_complete"] = True
    result = await agent_selector(state)
    assert result == "planning_agent"

    state["fitness_plan"] = {"some_plan": "data"}
    state["working_memory"]["last_analysis_date"] = (datetime.now() - timedelta(days=8)).isoformat()
    result = await agent_selector(state)
    assert result == "progress_analysis_agent"

    result = await agent_selector(state, "I need a new plan")
    assert result == "planning_agent"

@pytest.mark.asyncio
async def test_get_or_create_state():
    session_id = "test_session"
    state = await get_or_create_state(session_id)
    assert state["session_id"] == session_id
    assert state["current_agent"] == "coordinator"

@pytest.mark.asyncio
async def test_save_state(capsys):
    session_id = "test_session"
    state = await get_or_create_state(session_id)
    await save_state(session_id, state)
    captured = capsys.readouterr()
    assert f"State saved for session {session_id}" in captured.out

@pytest.mark.asyncio
async def test_agent_with_error_handling():
    async def test_agent(state):
        if state.get("raise_error"):
            raise ValueError("Test error")
        return state

    wrapped_agent = agent_with_error_handling(test_agent)
    
    state = {"raise_error": False}
    result = await wrapped_agent(state)
    assert result == state

    state = {"raise_error": True}
    result = await wrapped_agent(state)
    assert result["agent_state"]["status"] == "error"
    assert "Test error" in result["agent_state"]["error"]
    assert result["current_agent"] == "coordinator"

def test_build_fitness_trainer_graph():
    workflow = build_fitness_trainer_graph()
    assert workflow.entry_point == "coordinator"
    assert "coordinator" in workflow.nodes
    assert "end_conversation" in workflow.nodes
