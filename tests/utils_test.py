import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.chatbot.agent.utils import (
    extract_principles, extract_approaches, extract_citations,
    extract_routine_data, extract_adherence_rate, extract_progress_metrics,
    extract_issues, extract_adjustments, extract_routine_structure,
    extract_routine_updates, retrieve_data
)

@pytest.mark.asyncio
async def test_extract_principles():
    text = "Sample text with principles."
    mock_result = MagicMock()
    mock_result.principles = ["Principle 1", "Principle 2"]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_principles(text)
        
    assert result == ["Principle 1", "Principle 2"]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_approaches():
    text = "Sample text with approaches."
    mock_result = MagicMock()
    mock_result.approaches = [
        {"name": "Approach 1", "description": "Description 1"},
        {"name": "Approach 2", "description": "Description 2"}
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_approaches(text)
        
    assert result == [
        {"name": "Approach 1", "description": "Description 1"},
        {"name": "Approach 2", "description": "Description 2"}
    ]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_citations():
    text = "Sample text with citations."
    mock_result = MagicMock()
    mock_result.citations = [
        {"source": "Source 1", "content": "Content 1"},
        {"source": "Source 2", "content": "Content 2"}
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_citations(text)
        
    assert result == [
        {"source": "Source 1", "content": "Content 1"},
        {"source": "Source 2", "content": "Content 2"}
    ]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_routine_data():
    text = "Sample text with routine data."
    mock_result = MagicMock()
    mock_result.name = "Routine Name"
    mock_result.description = "Routine Description"
    mock_result.workouts = ["Workout 1", "Workout 2"]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_routine_data(text)
        
    assert result == {
        "name": "Routine Name",
        "description": "Routine Description",
        "workouts": ["Workout 1", "Workout 2"]
    }
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_adherence_rate():
    text = "Sample text with adherence rate."
    mock_result = MagicMock()
    mock_result.rate = 0.85
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_adherence_rate(text)
        
    assert result == 0.85
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_progress_metrics():
    text = "Sample text with progress metrics."
    mock_result = MagicMock()
    mock_result.metrics = {"strength": 10, "endurance": 5}
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_progress_metrics(text)
        
    assert result == {"strength": 10, "endurance": 5}
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_issues():
    text = "Sample text with issues."
    mock_result = MagicMock()
    mock_result.issues = ["Issue 1", "Issue 2"]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_issues(text)
        
    assert result == ["Issue 1", "Issue 2"]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_adjustments():
    text = "Sample text with adjustments."
    mock_result = MagicMock()
    mock_result.adjustments = [
        {"target": "Exercise 1", "change": "Increase weight"},
        {"target": "Exercise 2", "change": "Reduce reps"}
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_adjustments(text)
        
    assert result == [
        {"target": "Exercise 1", "change": "Increase weight"},
        {"target": "Exercise 2", "change": "Reduce reps"}
    ]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_routine_structure():
    text = "Sample text with routine structure."
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "title": "Routine Title",
        "notes": "Routine Notes",
        "exercises": ["Exercise 1", "Exercise 2"]
    }
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_routine_structure(text)
        
    assert result == {
        "title": "Routine Title",
        "notes": "Routine Notes",
        "exercises": ["Exercise 1", "Exercise 2"]
    }
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_routine_updates():
    text = "Sample text with routine updates."
    mock_result = MagicMock()
    mock_result.title = "Updated Routine Title"
    mock_result.notes = "Updated Routine Notes"
    mock_result.exercises = [
        {
            "exercise_name": "Exercise 1",
            "exercise_id": "ID1",
            "exercise_type": "strength",
            "sets": [
                {"type": "normal", "weight": 10, "reps": 5, "duration_seconds": None, "distance_meters": None}
            ],
            "notes": "Exercise Notes"
        }
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_routine_updates(text)
        
    assert result == {
        "title": "Updated Routine Title",
        "notes": "Updated Routine Notes",
        "exercises": [
            {
                "exercise_name": "Exercise 1",
                "exercise_id": "ID1",
                "exercise_type": "strength",
                "sets": [
                    {"type": "normal", "weight": 10, "reps": 5, "duration_seconds": None, "distance_meters": None}
                ],
                "notes": "Exercise Notes"
            }
        ]
    }
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_data(mocker):
    query = "What is progressive overload?"
    dummy_results = {
        "matches": [
            {"metadata": {"text": "Progressive overload is..."}},
            {"metadata": {"text": "It involves..."}}
        ]
    }
    mock_index = mocker.patch("src.chatbot.agent.utils.index")
    mock_index.query.return_value = dummy_results

    with patch("src.chatbot.agent.utils.embeddings.embed_query", return_value=[0.1, 0.2, 0.3]):
        result = await retrieve_data(query)
        
    assert "Progressive overload is..." in result
    assert "It involves..." in result
    mock_index.query.assert_called_once()import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.chatbot.agent.utils import (
    extract_principles, extract_approaches, extract_citations,
    extract_routine_data, extract_adherence_rate, extract_progress_metrics,
    extract_issues, extract_adjustments, extract_routine_structure,
    extract_routine_updates, retrieve_data
)

@pytest.mark.asyncio
async def test_extract_principles():
    text = "Sample text with principles."
    mock_result = MagicMock()
    mock_result.principles = ["Principle 1", "Principle 2"]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_principles(text)
        
    assert result == ["Principle 1", "Principle 2"]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_approaches():
    text = "Sample text with approaches."
    mock_result = MagicMock()
    mock_result.approaches = [
        {"name": "Approach 1", "description": "Description 1"},
        {"name": "Approach 2", "description": "Description 2"}
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_approaches(text)
        
    assert result == [
        {"name": "Approach 1", "description": "Description 1"},
        {"name": "Approach 2", "description": "Description 2"}
    ]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_citations():
    text = "Sample text with citations."
    mock_result = MagicMock()
    mock_result.citations = [
        {"source": "Source 1", "content": "Content 1"},
        {"source": "Source 2", "content": "Content 2"}
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_citations(text)
        
    assert result == [
        {"source": "Source 1", "content": "Content 1"},
        {"source": "Source 2", "content": "Content 2"}
    ]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_routine_data():
    text = "Sample text with routine data."
    mock_result = MagicMock()
    mock_result.name = "Routine Name"
    mock_result.description = "Routine Description"
    mock_result.workouts = ["Workout 1", "Workout 2"]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_routine_data(text)
        
    assert result == {
        "name": "Routine Name",
        "description": "Routine Description",
        "workouts": ["Workout 1", "Workout 2"]
    }
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_adherence_rate():
    text = "Sample text with adherence rate."
    mock_result = MagicMock()
    mock_result.rate = 0.85
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_adherence_rate(text)
        
    assert result == 0.85
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_progress_metrics():
    text = "Sample text with progress metrics."
    mock_result = MagicMock()
    mock_result.metrics = {"strength": 10, "endurance": 5}
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_progress_metrics(text)
        
    assert result == {"strength": 10, "endurance": 5}
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_issues():
    text = "Sample text with issues."
    mock_result = MagicMock()
    mock_result.issues = ["Issue 1", "Issue 2"]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_issues(text)
        
    assert result == ["Issue 1", "Issue 2"]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_adjustments():
    text = "Sample text with adjustments."
    mock_result = MagicMock()
    mock_result.adjustments = [
        {"target": "Exercise 1", "change": "Increase weight"},
        {"target": "Exercise 2", "change": "Reduce reps"}
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_adjustments(text)
        
    assert result == [
        {"target": "Exercise 1", "change": "Increase weight"},
        {"target": "Exercise 2", "change": "Reduce reps"}
    ]
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_routine_structure():
    text = "Sample text with routine structure."
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "title": "Routine Title",
        "notes": "Routine Notes",
        "exercises": ["Exercise 1", "Exercise 2"]
    }
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_routine_structure(text)
        
    assert result == {
        "title": "Routine Title",
        "notes": "Routine Notes",
        "exercises": ["Exercise 1", "Exercise 2"]
    }
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_extract_routine_updates():
    text = "Sample text with routine updates."
    mock_result = MagicMock()
    mock_result.title = "Updated Routine Title"
    mock_result.notes = "Updated Routine Notes"
    mock_result.exercises = [
        {
            "exercise_name": "Exercise 1",
            "exercise_id": "ID1",
            "exercise_type": "strength",
            "sets": [
                {"type": "normal", "weight": 10, "reps": 5, "duration_seconds": None, "distance_meters": None}
            ],
            "notes": "Exercise Notes"
        }
    ]
    
    with patch("src.chatbot.agent.utils.llm.with_structured_output", return_value=mock_result):
        result = extract_routine_updates(text)
        
    assert result == {
        "title": "Updated Routine Title",
        "notes": "Updated Routine Notes",
        "exercises": [
            {
                "exercise_name": "Exercise 1",
                "exercise_id": "ID1",
                "exercise_type": "strength",
                "sets": [
                    {"type": "normal", "weight": 10, "reps": 5, "duration_seconds": None, "distance_meters": None}
                ],
                "notes": "Exercise Notes"
            }
        ]
    }
    mock_result.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_data(mocker):
    query = "What is progressive overload?"
    dummy_results = {
        "matches": [
            {"metadata": {"text": "Progressive overload is..."}},
            {"metadata": {"text": "It involves..."}}
        ]
    }
    mock_index = mocker.patch("src.chatbot.agent.utils.index")
    mock_index.query.return_value = dummy_results

    with patch("src.chatbot.agent.utils.embeddings.embed_query", return_value=[0.1, 0.2, 0.3]):
        result = await retrieve_data(query)
        
    assert "Progressive overload is..." in result
    assert "It involves..." in result
    mock_index.query.assert_called_once()
