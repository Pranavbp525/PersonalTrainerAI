import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi import HTTPException
from src.chatbot.agent.llm_tools import (
    tool_fetch_workouts, tool_get_workout_count, tool_fetch_routines,
    tool_update_routine, tool_create_routine, retrieve_from_rag
)
from src.chatbot.agent.agent_models import RoutineUpdateRequest, RoutineCreateRequest

def test_tool_fetch_workouts():
    with patch("src.chatbot.agent.hevy_api.get_workouts", new_callable=AsyncMock) as mock_get_workouts:
        mock_get_workouts.return_value = {"data": "workouts"}
        result = asyncio.run(tool_fetch_workouts(page=1, page_size=5))
        assert result == {"data": "workouts"}
        mock_get_workouts.assert_awaited_once_with(page=1, pageSize=5)

def test_tool_get_workout_count():
    with patch("src.chatbot.agent.hevy_api.get_workout_count", new_callable=AsyncMock) as mock_get_workout_count:
        mock_get_workout_count.return_value = {"count": 42}
        result = asyncio.run(tool_get_workout_count())
        assert result == {"count": 42}
        mock_get_workout_count.assert_awaited_once()

def test_tool_fetch_routines():
    with patch("src.chatbot.agent.hevy_api.get_routines", new_callable=AsyncMock) as mock_get_routines:
        mock_get_routines.return_value = {"data": "routines"}
        result = asyncio.run(tool_fetch_routines(page=1, page_size=5))
        assert result == {"data": "routines"}
        mock_get_routines.assert_awaited_once_with(page=1, pageSize=5)

def test_tool_update_routine():
    routine_id = "test_routine_id"
    routine_data = {
        "title": "Updated Routine",
        "notes": "Updated notes",
        "exercises": []
    }
    with patch("src.chatbot.agent.hevy_api.update_routine", new_callable=AsyncMock) as mock_update_routine:
        mock_update_routine.return_value = {"id": routine_id}
        result = asyncio.run(tool_update_routine(routine_id, routine_data))
        assert result == {"id": routine_id}
        mock_update_routine.assert_awaited_once()

def test_tool_update_routine_invalid_data():
    routine_id = "test_routine_id"
    routine_data = {
        "title": "Updated Routine",
        "notes": "Updated notes",
        "exercises": "invalid_exercises_data"  # This should cause an error
    }
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(tool_update_routine(routine_id, routine_data))
    assert exc_info.value.status_code == 400
    assert "Invalid routine data" in exc_info.value.detail

def test_tool_create_routine():
    routine_data = {
        "title": "New Routine",
        "folder_id": None,
        "notes": "Routine notes",
        "exercises": []
    }
    with patch("src.chatbot.agent.hevy_api.create_routine", new_callable=AsyncMock) as mock_create_routine:
        mock_create_routine.return_value = {"id": "new_routine_id"}
        result = asyncio.run(tool_create_routine(routine_data))
        assert result == {"id": "new_routine_id"}
        mock_create_routine.assert_awaited_once()

def test_tool_create_routine_invalid_data():
    routine_data = {
        "title": "New Routine",
        "folder_id": None,
        "notes": "Routine notes",
        "exercises": "invalid_exercises_data"  # This should cause an error
    }
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(tool_create_routine(routine_data))
    assert exc_info.value.status_code == 400
    assert "Invalid routine data" in exc_info.value.detail

def test_retrieve_from_rag():
    query = "What is progressive overload?"
    dummy_results = {
        "matches": [
            {"metadata": {"text": "Progressive overload is..."}},
            {"metadata": {"text": "It involves..."}}
        ]
    }
    with patch("pinecone.Pinecone.query", new_callable=AsyncMock) as mock_query, \
         patch("langchain_huggingface.embeddings.HuggingFaceEmbeddings.embed_query", return_value=[0.1, 0.2, 0.3]) as mock_embed_query:
        mock_query.return_value = dummy_results
        result = asyncio.run(retrieve_from_rag(query))
        assert "Progressive overload is..." in result
        mock_embed_query.assert_called_once_with(query)
        mock_query.assert_awaited_once()

def test_retrieve_from_rag_error():
    query = "What is progressive overload?"
    with patch("pinecone.Pinecone.query", new_callable=AsyncMock) as mock_query, \
         patch("langchain_huggingface.embeddings.HuggingFaceEmbeddings.embed_query", return_value=[0.1, 0.2, 0.3]) as mock_embed_query:
        mock_query.side_effect = Exception("Pinecone error")
        result = asyncio.run(retrieve_from_rag(query))
        assert "Error retrieving information" in result
        mock_embed_query.assert_called_once_with(query)
        mock_query.assert_awaited_once()
