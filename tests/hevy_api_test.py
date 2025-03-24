import pytest
from fastapi.testclient import TestClient
from fastapi import status
from src.chatbot.agent.hevy_api import app, get_api_key, API_KEY
from src.chatbot.agent.agent_models import WorkoutUpdateRequest, RoutineUpdateRequest, RoutineCreateRequest
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.fixture
def api_key_header():
    return {"api-key": API_KEY}

# Mock API key validation
@pytest.fixture(autouse=True)
def mock_auth():
    with patch("src.chatbot.agent.hevy_api.get_api_key", return_value=API_KEY):
        yield

@pytest.mark.asyncio
async def test_get_workouts(api_key_header):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": "mock_workouts"}

        response = client.get("/v1/workouts", headers=api_key_header)
        assert response.status_code == 200
        assert response.json() == {"data": "mock_workouts"}

@pytest.mark.asyncio
async def test_get_workout_count(api_key_header):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"count": 42}

        response = client.get("/v1/workouts/count", headers=api_key_header)
        assert response.status_code == 200
        assert response.json() == {"count": 42}

@pytest.mark.asyncio
async def test_get_workout(api_key_header):
    workout_id = "test_workout_id"
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"id": workout_id}

        response = client.get(f"/v1/workouts/{workout_id}", headers=api_key_header)
        assert response.status_code == 200
        assert response.json() == {"id": workout_id}

@pytest.mark.asyncio
async def test_get_routines(api_key_header):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": "mock_routines"}

        response = client.get("/v1/routines", headers=api_key_header)
        assert response.status_code == 200
        assert response.json() == {"data": "mock_routines"}

@pytest.mark.asyncio
async def test_update_workout(api_key_header):
    workout_id = "test_workout_id"
    update_data = WorkoutUpdateRequest(
        workout={
            "title": "Updated Workout",
            "description": "Updated description",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "is_private": False,
            "exercises": []
        }
    )
    with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"id": workout_id}

        response = client.put(f"/v1/workouts/{workout_id}", headers=api_key_header, json=update_data.dict())
        assert response.status_code == 200
        assert response.json() == {"id": workout_id}

@pytest.mark.asyncio
async def test_update_routine(api_key_header):
    routine_id = "test_routine_id"
    update_data = RoutineUpdateRequest(
        routine={
            "title": "Updated Routine",
            "notes": "Updated notes",
            "exercises": []
        }
    )
    with patch("httpx.AsyncClient.put", new_callable=AsyncMock) as mock_put:
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"id": routine_id}

        response = client.put(f"/v1/routines/{routine_id}", headers=api_key_header, json=update_data.dict())
        assert response.status_code == 200
        assert response.json() == {"id": routine_id}

@pytest.mark.asyncio
async def test_create_routine(api_key_header):
    create_data = RoutineCreateRequest(
        routine={
            "title": "New Routine",
            "folder_id": None,
            "notes": "Routine notes",
            "exercises": []
        }
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": "new_routine_id"}

        response = client.post("/v1/routines", headers=api_key_header, json=create_data.dict())
        assert response.status_code == 201
        assert response.json() == {"id": "new_routine_id"}
