import pytest
from unittest.mock import patch, MagicMock
from src.chatbot.agent.prompts import (
    push_user_modeler_prompt, get_user_modeler_prompt,
    push_memory_consolidation_prompt, get_memory_consolidation_prompt,
    push_coordinator_prompt, get_coordinator_prompt,
    push_research_prompt, get_research_prompt,
    push_planning_prompt, get_planning_prompt,
    push_analysis_prompt, get_analysis_prompt,
    push_adaptation_prompt, get_adaptation_prompt,
    push_coach_prompt, get_coach_prompt,
    tag_prompt, push_all_prompts
)

def test_push_user_modeler_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/user_modeler"
    
    url = push_user_modeler_prompt()
    assert url == "http://example.com/user_modeler"
    mock_client.push_prompt.assert_called_once()

def test_get_user_modeler_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_user_modeler_prompt"
    
    prompt = get_user_modeler_prompt()
    assert prompt == "mock_user_modeler_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-user-modeler")

def test_push_memory_consolidation_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/memory_consolidation"
    
    url = push_memory_consolidation_prompt()
    assert url == "http://example.com/memory_consolidation"
    mock_client.push_prompt.assert_called_once()

def test_get_memory_consolidation_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_memory_consolidation_prompt"
    
    prompt = get_memory_consolidation_prompt()
    assert prompt == "mock_memory_consolidation_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-memory-consolidation")

def test_push_coordinator_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/coordinator"
    
    url = push_coordinator_prompt()
    assert url == "http://example.com/coordinator"
    mock_client.push_prompt.assert_called_once()

def test_get_coordinator_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_coordinator_prompt"
    
    prompt = get_coordinator_prompt()
    assert prompt == "mock_coordinator_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-coordinator")

def test_push_research_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/research"
    
    url = push_research_prompt()
    assert url == "http://example.com/research"
    mock_client.push_prompt.assert_called_once()

def test_get_research_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_research_prompt"
    
    prompt = get_research_prompt()
    assert prompt == "mock_research_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-research")

def test_push_planning_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/planning"
    
    url = push_planning_prompt()
    assert url == "http://example.com/planning"
    mock_client.push_prompt.assert_called_once()

def test_get_planning_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_planning_prompt"
    
    prompt = get_planning_prompt()
    assert prompt == "mock_planning_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-planning")

def test_push_analysis_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/analysis"
    
    url = push_analysis_prompt()
    assert url == "http://example.com/analysis"
    mock_client.push_prompt.assert_called_once()

def test_get_analysis_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_analysis_prompt"
    
    prompt = get_analysis_prompt()
    assert prompt == "mock_analysis_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-analysis")

def test_push_adaptation_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/adaptation"
    
    url = push_adaptation_prompt()
    assert url == "http://example.com/adaptation"
    mock_client.push_prompt.assert_called_once()

def test_get_adaptation_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_adaptation_prompt"
    
    prompt = get_adaptation_prompt()
    assert prompt == "mock_adaptation_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-adaptation")

def test_push_coach_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.push_prompt.return_value = "http://example.com/coach"
    
    url = push_coach_prompt()
    assert url == "http://example.com/coach"
    mock_client.push_prompt.assert_called_once()

def test_get_coach_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    mock_client.pull_prompt.return_value = "mock_coach_prompt"
    
    prompt = get_coach_prompt()
    assert prompt == "mock_coach_prompt"
    mock_client.pull_prompt.assert_called_once_with("fitness-coach")

def test_tag_prompt(mocker):
    mock_client = mocker.patch("src.chatbot.agent.prompts.client")
    
    tag_prompt("coordinator", "commit123", "v1")
    mock_client.tag_prompt.assert_called_once_with("fitness-coordinator", "commit123", "v1")

def test_push_all_prompts(mocker):
    mock_push_user_modeler = mocker.patch("src.chatbot.agent.prompts.push_user_modeler_prompt", return_value="http://example.com/user_modeler")
    mock_push_memory_consolidation = mocker.patch("src.chatbot.agent.prompts.push_memory_consolidation_prompt", return_value="http://example.com/memory_consolidation")
    mock_push_coordinator = mocker.patch("src.chatbot.agent.prompts.push_coordinator_prompt", return_value="http://example.com/coordinator")
    mock_push_research = mocker.patch("src.chatbot.agent.prompts.push_research_prompt", return_value="http://example.com/research")
    mock_push_planning = mocker.patch("src.chatbot.agent.prompts.push_planning_prompt", return_value="http://example.com/planning")
    mock_push_analysis = mocker.patch("src.chatbot.agent.prompts.push_analysis_prompt", return_value="http://example.com/analysis")
    mock_push_adaptation = mocker.patch("src.chatbot.agent.prompts.push_adaptation_prompt", return_value="http://example.com/adaptation")
    mock_push_coach = mocker.patch("src.chatbot.agent.prompts.push_coach_prompt", return_value="http://example.com/coach")
    
    results = push_all_prompts()
    
    assert results["user_modeler"] == "http://example.com/user_modeler"
    assert results["memory_consolidation"] == "http://example.com/memory_consolidation"
    assert results["coordinator"] == "http://example.com/coordinator"
    assert results["research"] == "http://example.com/research"
    assert results["planning"] == "http://example.com/planning"
    assert results["analysis"] == "http://example.com/analysis"
    assert results["adaptation"] == "http://example.com/adaptation"
    assert results["coach"] == "http://example.com/coach"
    
    mock_push_user_modeler.assert_called_once()
    mock_push_memory_consolidation.assert_called_once()
    mock_push_coordinator.assert_called_once()
    mock_push_research.assert_called_once()
    mock_push_planning.assert_called_once()
    mock_push_analysis.assert_called_once()
    mock_push_adaptation.assert_called_once()
    mock_push_coach.assert_called_once()
