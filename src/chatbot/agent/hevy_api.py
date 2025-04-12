import time # Import time for timing
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

import os
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

# Assuming these models are correctly defined in agent_models.py
from agent.agent_models import (
    SetUpdate,
    ExerciseUpdate,
    WorkoutUpdate,
    WorkoutUpdateRequest,
    SetRoutineUpdate,
    RoutineUpdateRequest,
    RoutineCreateRequest
)

# --- ELK Logging Import ---
try:
    # Assuming hevy_api.py is in the same directory level as elk_logging.py
    # Adjust the path '..' if hevy_api.py is inside a subdirectory like 'agent/'
    # from ..elk_logging import setup_elk_logging # If in subdirectory
    from ..elk_logging import setup_elk_logging # If at same level
except ImportError:
    # Fallback for different execution contexts
    print("Could not import elk_logging, using standard print for Hevy API logs.")
    # Define a dummy logger class if elk_logging is unavailable
    class DummyLogger:
        def add_context(self, **kwargs): return self
        def info(self, msg, *args, **kwargs): print(f"INFO: {msg} | Context: {kwargs.get('extra')}")
        def warning(self, msg, *args, **kwargs): print(f"WARN: {msg} | Context: {kwargs.get('extra')}")
        def error(self, msg, *args, **kwargs): print(f"ERROR: {msg} | Context: {kwargs.get('extra')}")
        def debug(self, msg, *args, **kwargs): print(f"DEBUG: {msg} | Context: {kwargs.get('extra')}")
    api_log = DummyLogger()
else:
    api_log = setup_elk_logging("fitness-chatbot.hevy_api")
# --- End ELK Logging Import ---

load_dotenv()

API_KEY = os.getenv("HEVY_API_KEY")
if not API_KEY:
    api_log.error("HEVY_API_KEY environment variable not set!") # Log missing key
    raise Exception("HEVY_API_KEY environment variable not set")
else:
    api_log.info("HEVY_API_KEY loaded.")

HEADERS = {
    "api-key": API_KEY,
    "accept": "application/json",
    "Content-Type": "application/json"
}


# === Workout GET requests ===

async def get_workouts(page: int = 1, pageSize: int = 5):
    """Fetches a page of workout logs from Hevy API."""
    endpoint = "/v1/workouts"
    url = f"https://api.hevyapp.com{endpoint}"
    params = {"page": page, "pageSize": pageSize}
    log_ctx = {"endpoint": endpoint, "method": "GET", "page": page, "page_size": pageSize}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=HEADERS, params=params)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        if status_code != 200:
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]}) # Log error snippet
            raise HTTPException(status_code=status_code, detail="Error fetching workouts")

        result_data = response.json()
        num_workouts = len(result_data.get('workouts', [])) if isinstance(result_data, dict) else 'N/A'
        api_log.info(f"Successfully fetched from {endpoint}", extra={**log_ctx_res, "workouts_fetched": num_workouts})
        return result_data

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}") # 503 for network/request errors
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error fetching workouts: {e}")


async def get_workout_count():
    """Fetches the total workout count from Hevy API."""
    endpoint = "/v1/workouts/count"
    url = f"https://api.hevyapp.com{endpoint}"
    log_ctx = {"endpoint": endpoint, "method": "GET"}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=HEADERS)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        if status_code != 200:
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]})
            raise HTTPException(status_code=status_code, detail="Error fetching workout count")

        result_data = response.json()
        count = result_data.get('count', 'N/A') if isinstance(result_data, dict) else 'N/A'
        api_log.info(f"Successfully fetched from {endpoint}", extra={**log_ctx_res, "workout_count": count})
        return result_data

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}")
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error fetching workout count: {e}")


async def get_workout(workout_id: str):
    """Fetches a specific workout by ID from Hevy API."""
    endpoint = f"/v1/workouts/{workout_id}"
    url = f"https://api.hevyapp.com{endpoint}"
    log_ctx = {"endpoint": endpoint, "method": "GET", "workout_id": workout_id}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=HEADERS)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        if status_code != 200:
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]})
            raise HTTPException(status_code=status_code, detail="Error fetching workout by ID")

        api_log.info(f"Successfully fetched from {endpoint}", extra=log_ctx_res)
        return response.json()

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}")
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error fetching workout by ID: {e}")


# === Routine GET ===

async def get_routines(page: int = 1, pageSize: int = 5):
    """Fetches a page of workout routines from Hevy API."""
    endpoint = "/v1/routines"
    url = f"https://api.hevyapp.com{endpoint}"
    params = {"page": page, "pageSize": pageSize}
    log_ctx = {"endpoint": endpoint, "method": "GET", "page": page, "page_size": pageSize}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=HEADERS, params=params)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        if status_code != 200:
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]})
            raise HTTPException(status_code=status_code, detail="Error fetching routines")

        result_data = response.json()
        num_routines = len(result_data.get('routines', [])) if isinstance(result_data, dict) else 'N/A'
        api_log.info(f"Successfully fetched from {endpoint}", extra={**log_ctx_res, "routines_fetched": num_routines})
        return result_data

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}")
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error fetching routines: {e}")


# === Routine PUT ===

async def update_routine(routine_id: str, update_data: RoutineUpdateRequest):
    """Updates an existing routine via Hevy API."""
    endpoint = f"/v1/routines/{routine_id}"
    url = f"https://api.hevyapp.com{endpoint}"
    log_ctx = {"endpoint": endpoint, "method": "PUT", "routine_id": routine_id}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()
    payload = {} # Initialize payload

    try:
        # --- Original Logic: Encoding ---
        payload = jsonable_encoder(update_data)
        # Log payload size or specific fields if needed for debugging, be cautious with sensitive data
        api_log.debug("Encoded payload for update routine.", extra={**log_ctx, "payload_keys": list(payload.keys())})
        # --- End Original Logic ---

        async with httpx.AsyncClient() as client:
            response = await client.put(url, headers=HEADERS, json=payload)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        if status_code != 200:
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]})
            # Include response text in the exception detail
            raise HTTPException(status_code=status_code, detail=f"Error updating routine: {response.text}")

        api_log.info(f"Successfully executed {endpoint}", extra=log_ctx_res)
        return response.json()

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}")
    except Exception as e:
        duration = time.time() - start_time
        # Catch potential jsonable_encoder errors too
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error updating routine: {e}")


# === Workout PUT ===

async def update_workout(workout_id: str, update_data: WorkoutUpdateRequest):
    """Updates an existing workout via Hevy API."""
    endpoint = f"/v1/workouts/{workout_id}"
    url = f"https://api.hevyapp.com{endpoint}"
    log_ctx = {"endpoint": endpoint, "method": "PUT", "workout_id": workout_id}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()
    payload = {} # Initialize payload

    try:
        # --- Original Logic: Encoding ---
        payload = jsonable_encoder(update_data)
        api_log.debug("Encoded payload for update workout.", extra={**log_ctx, "payload_keys": list(payload.keys())})
        # --- End Original Logic ---

        async with httpx.AsyncClient() as client:
            response = await client.put(url, headers=HEADERS, json=payload)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        if status_code != 200:
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]})
            raise HTTPException(
                status_code=status_code,
                detail=f"Error updating workout: {response.text}"
            )

        api_log.info(f"Successfully executed {endpoint}", extra=log_ctx_res)
        return response.json()

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}")
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error updating workout: {e}")


# === Routine POST ===

async def create_routine(create_data: RoutineCreateRequest):
    """Creates a new routine via Hevy API."""
    endpoint = "/v1/routines"
    url = f"https://api.hevyapp.com{endpoint}"
    # Extract title for logging context if possible, handle potential missing attributes
    routine_title = getattr(getattr(create_data, 'routine', None), 'title', 'Unknown Title')
    log_ctx = {"endpoint": endpoint, "method": "POST", "routine_title": routine_title}
    api_log.info(f"Requesting {endpoint}", extra=log_ctx)
    start_time = time.time()
    payload = {} # Initialize payload

    try:
        # --- Original Logic: Encoding ---
        payload = jsonable_encoder(create_data)
        api_log.debug("Encoded payload for create routine.", extra={**log_ctx, "payload_keys": list(payload.keys())})
        # --- End Original Logic ---

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=HEADERS, json=payload)

        duration = time.time() - start_time
        status_code = response.status_code
        log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2), "status_code": status_code}

        # Check for successful status codes (200 or 201 typically for POST)
        if status_code not in (200, 201):
            api_log.error(f"Error response from {endpoint}", extra={**log_ctx_res, "response_text": response.text[:500]})
            raise HTTPException(status_code=status_code, detail=f"Error creating routine: {response.text}")

        api_log.info(f"Successfully executed {endpoint}", extra=log_ctx_res)
        return response.json()

    except httpx.RequestError as req_err:
        duration = time.time() - start_time
        api_log.error(f"HTTP Request Error calling {endpoint}", exc_info=req_err, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=503, detail=f"Service unavailable: {req_err}")
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"Unexpected error calling {endpoint}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error creating routine: {e}")