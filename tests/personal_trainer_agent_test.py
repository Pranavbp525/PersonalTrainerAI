import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from src.chatbot.agent.personal_trainer_agent import (
    user_modeler, coordinator, research_agent, planning_agent,
    progress_analysis_agent, adaptation_agent, coach_agent, end_conversation
)
from src.chatbot.agent.agent_models import AgentState, UserProfile
from datetime import datetime

@pytest.mark.asyncio
async def test_user_modeler():
    state = {
        "user_model": {},
        "working_memory": {
            "recent_exchanges": []
        }
    }
    mock_response = AsyncMock()
    mock_response.content = '{"name": "John", "age": 30}'
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm.ainvoke", return_value=mock_response):
        updated_state = await user_modeler(state)
        
    assert updated_state["user_model"]["name"] == "John"
    assert updated_state["user_model"]["age"] == 30
    assert updated_state["current_agent"] == "coordinator"

@pytest.mark.asyncio
async def test_coordinator():
    state = {
        "messages": [],
        "user_model": {},
        "fitness_plan": {},
        "working_memory": {}
    }
    mock_response = AsyncMock()
    mock_response.content = "<user>Hello</user>[Research]"
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm.ainvoke", return_value=mock_response):
        updated_state = await coordinator(state)
        
    assert updated_state["messages"][-1].content == "Hello"
    assert updated_state["current_agent"] == "research_agent"

@pytest.mark.asyncio
async def test_research_agent():
    state = {
        "user_model": {"goals": ["strength"]},
        "working_memory": {}
    }
    mock_response = AsyncMock()
    mock_response.content = "Research content"
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm_with_tools.ainvoke", return_value=mock_response):
        updated_state = await research_agent(state)
        
    assert updated_state["messages"][-1].content == "Research content"
    assert "research_findings" in updated_state["working_memory"]

@pytest.mark.asyncio
async def test_planning_agent():
    state = {
        "user_model": {},
        "working_memory": {},
        "messages": []
    }
    mock_response = AsyncMock()
    mock_response.content = "Planning content"
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm.ainvoke", return_value=mock_response):
        updated_state = await planning_agent(state)
        
    assert "Planning content" in updated_state["messages"][-1].content
    assert "fitness_plan" in updated_state

@pytest.mark.asyncio
async def test_progress_analysis_agent():
    state = {
        "user_model": {},
        "fitness_plan": {},
        "messages": [],
        "working_memory": {}
    }
    mock_response = AsyncMock()
    mock_response.content = "Analysis content"
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm_with_tools.ainvoke", return_value=mock_response):
        with patch("src.chatbot.agent.personal_trainer_agent.tool_fetch_workouts", return_value={"logs": []}):
            updated_state = await progress_analysis_agent(state)
            
    assert "Analysis content" in updated_state["messages"][-1].content
    assert "latest_analysis" in updated_state["progress_data"]

@pytest.mark.asyncio
async def test_adaptation_agent():
    state = {
        "user_model": {},
        "fitness_plan": {"hevy_routine_id": "routine123"},
        "progress_data": {
            "latest_analysis": {
                "suggested_adjustments": []
            }
        },
        "messages": []
    }
    mock_response = AsyncMock()
    mock_response.content = "Adaptation content"
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm_with_tools.ainvoke", return_value=mock_response):
        with patch("src.chatbot.agent.personal_trainer_agent.tool_update_routine", return_value={"status": "success"}):
            updated_state = await adaptation_agent(state)
            
    assert "Adaptation content" in updated_state["messages"][-1].content
    assert "fitness_plan" in updated_state

@pytest.mark.asyncio
async def test_coach_agent():
    state = {
        "user_model": {},
        "progress_data": {},
        "working_memory": {"recent_exchanges": []},
        "messages": []
    }
    mock_response = AsyncMock()
    mock_response.content = "Coaching content"
    
    with patch("src.chatbot.agent.personal_trainer_agent.llm.ainvoke", return_value=mock_response):
        updated_state = await coach_agent(state)
        
    assert "Coaching content" in updated_state["messages"][-1].content

@pytest.mark.asyncio
async def test_end_conversation():
    state = {
        "agent_state": {},
        "conversation_complete": False
    }
    updated_state = await end_conversation(state)
    
    assert updated_state["agent_state"]["status"] == "complete"
    assert updated_state["conversation_complete"] == True
