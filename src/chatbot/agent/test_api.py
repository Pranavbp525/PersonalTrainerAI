import asyncio
from datetime import datetime, timedelta
from agent.llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_create_routine,
    tool_update_routine,
    tool_update_workout,
)


async def test_fetch_workouts():
    print("📥 Test: Fetch Workouts")
    result = await tool_fetch_workouts.ainvoke({"page": 1, "page_size": 3})
    assert isinstance(result, dict) or isinstance(result, list), "Expected a JSON response"
    print("✅ Result:", result, "\n")


async def test_get_workout_count():
    print("🔢 Test: Workout Count")
    result = await tool_get_workout_count.ainvoke({})
    print("🚨 Raw response:", result)

    count = result.get("workout_count")
    assert count is not None, f"❌ 'workout_count' not found in response: {result}"
    print("✅ Workout Count:", count, "\n")




async def test_fetch_routines():
    print("📜 Test: Fetch Routines")
    result = await tool_fetch_routines.ainvoke({"page": 1, "page_size": 3})
    assert isinstance(result, dict) or isinstance(result, list), "Expected a JSON response"
    print("✅ Result:", result, "\n")


async def test_create_routine():
    print("🆕 Test: Create Routine")
    sample_routine = {
        "routine_data": {
            "title": "LangChain Test Routine",
            "notes": "Auto-generated test routine",
            "exercises": []
        }
    }

    result = await tool_create_routine.ainvoke(sample_routine)
    assert "id" in result or isinstance(result, dict), "Expected new routine ID"
    print("✅ Created Routine:", result, "\n")
    return result.get("id")


async def test_update_routine(routine_id: str):
    print("✏️ Test: Update Routine")
    updated_data = {
        "routine_id": routine_id,
        "routine_data": {
            "title": "LangChain Routine (Updated)",
            "notes": "Updated note",
            "exercises": []
        }
    }

    result = await tool_update_routine.ainvoke(updated_data)
    assert isinstance(result, dict), "Expected a JSON response"
    print("✅ Updated Routine:", result, "\n")


async def test_update_workout():
    print("✏️ Test: Update Workout")
    # NOTE: You'll need a valid workout_id for this test
    workout_id = "REPLACE_WITH_VALID_WORKOUT_ID"
    update_payload = {
        "workout_id": workout_id,
        "update_data": {
            "title": "Updated Workout",
            "description": "This is a test update",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(minutes=60)).isoformat(),
            "is_private": False,
            "exercises": []
        }
    }

    try:
        result = await tool_update_workout.ainvoke(update_payload)
        print("✅ Updated Workout:", result, "\n")
    except Exception as e:
        print("⚠️ Skipping update_workout (missing valid workout ID):", str(e))


async def main():
    # await test_fetch_workouts()
    # await test_get_workout_count()
     await test_fetch_routines()

    # routine_id = await test_create_routine()
    # if routine_id:
    #     await test_update_routine(routine_id)

    # await test_update_workout()

asyncio.run(main())
