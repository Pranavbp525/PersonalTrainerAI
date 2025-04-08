import asyncio
from datetime import datetime, timedelta
from llm_tools import (
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
    result = await tool_fetch_routines.ainvoke({"page": 1, "page_size": 6})
    assert isinstance(result, dict) or isinstance(result, list), "Expected a JSON response"
    print("✅ Result:", result, "\n")


async def test_create_routine():
    print("🆕 Test: Create Routine")
    sample_routine = {
        "routine_data": {
            "title": "Legs and abs test",
            "folder_id": None,
            "notes": "Auto-generated test routine from sample example",
            "exercises": [
                {
                    "exercise_template_id": "0EB695C9",
                    "superset_id": None,
                    "rest_seconds": 60,
                    "notes": "Leg Press Horizontal (Machine)",
                    "sets": [
                        {
                            "type": "warmup",
                            "weight_kg": 54.43114913227677,
                            "reps": 9,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "dropset",
                            "weight_kg": 81.64672369841516,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 90.71858188712795,
                            "reps": 11,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 90.71858188712795,
                            "reps": 10,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 81.64672369841516,
                            "reps": 5,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                },
                {
                    "exercise_template_id": "11A123F3",
                    "superset_id": None,
                    "rest_seconds": 60,
                    "notes": "Seated Leg Curl (Machine)",
                    "sets": [
                        {
                            "type": "normal",
                            "weight_kg": 52.16318458509857,
                            "reps": 13,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 45.35929094356398,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 45.35929094356398,
                            "reps": 11,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 45.35929094356398,
                            "reps": 10,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                },
                {
                    "exercise_template_id": "75A4F6C4",
                    "superset_id": None,
                    "rest_seconds": 60,
                    "notes": "Leg Extension (Machine)",
                    "sets": [
                        {
                            "type": "normal",
                            "weight_kg": 58.96707822663317,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 58.96707822663317,
                            "reps": 13,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 58.96707822663317,
                            "reps": 10,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 52.16318458509857,
                            "reps": 12,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                },
                {
                    "exercise_template_id": "F4B4C6EE",
                    "superset_id": "0",  # Converting the numeric superset_id (0) to string
                    "rest_seconds": 60,
                    "notes": "Hip Abduction (Machine)",
                    "sets": [
                        {
                            "type": "normal",
                            "weight_kg": 40.82336184920758,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 45.35929094356398,
                            "reps": 9,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 40.82336184920758,
                            "reps": 10,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 36.28743275485118,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                },
                {
                    "exercise_template_id": "062AB91A",
                    "superset_id": None,
                    "rest_seconds": 60,
                    "notes": "Seated Calf Raise",
                    "sets": [
                        {
                            "type": "normal",
                            "weight_kg": 22.67964547178199,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 22.67964547178199,
                            "reps": 12,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 22.67964547178199,
                            "reps": 10,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                },
                {
                    "exercise_template_id": "B2398CD1",
                    "superset_id": None,
                    "rest_seconds": 60,
                    "notes": "Decline Crunch (Weighted)",
                    "sets": [
                        {
                            "type": "normal",
                            "weight_kg": 15.87575183024739,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 15.87575183024739,
                            "reps": 10,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 15.87575183024739,
                            "reps": 7,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 15.87575183024739,
                            "reps": 6,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                },
                {
                    "exercise_template_id": "FBB62888",
                    "superset_id": None,
                    "rest_seconds": 60,
                    "notes": "Torso Rotation",
                    "sets": [
                        {
                            "type": "normal",
                            "weight_kg": 52.16318458509857,
                            "reps": 15,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 52.16318458509857,
                            "reps": 12,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 52.16318458509857,
                            "reps": 12,
                            "distance_meters": None,
                            "duration_seconds": None
                        },
                        {
                            "type": "normal",
                            "weight_kg": 52.16318458509857,
                            "reps": 12,
                            "distance_meters": None,
                            "duration_seconds": None
                        }
                    ]
                }
            ]
        }
    }



    result = await tool_create_routine.ainvoke(sample_routine)
    assert "id" in result or isinstance(result, dict), "Expected new routine ID"
    print("✅ Created Routine:", result, "\n")
    return result.get("id")


async def test_update_routine(routine_id: str = "56b56a12-9f6b-4953-9cc9-a3fcf0181018"):
    print("✏️ Test: Update Routine")
    updated_data = {
      "id": "56b56a12-9f6b-4953-9cc9-a3fcf0181018",
      "title": "Back and biceps",
      "folder_id": None,
      "updated_at": "2025-03-29T03:34:49.376Z",
      "created_at": "2025-02-25T07:15:02.118Z",
      "exercises": [
        {
          "index": 0,
          "title": "Lat Pulldown (Cable)",
          "notes": None,
          "exercise_template_id": "6A6C31A5",
          "superset_id": None,
          "sets": [
            {
              "index": 0,
              "type": "normal",
              "weight_kg": 100.0,
              "reps": 15,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 1,
              "type": "normal",
              "weight_kg": 100.0,
              "reps": 15,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 2,
              "type": "normal",
              "weight_kg": 100.0,
              "reps": 15,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 3,
              "type": "normal",
              "weight_kg": 100.0,
              "reps": 15,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            }
          ]
        },
        {
          "index": 1,
          "title": "Seated Cable Row - V Grip (Cable)",
          "notes": None,
          "exercise_template_id": "0393F233",
          "superset_id": None,
          "sets": [
            {
              "index": 0,
              "type": "normal",
              "weight_kg": 54.43114913227677,
              "reps": 10,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 1,
              "type": "normal",
              "weight_kg": 54.43114913227677,
              "reps": 7,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 2,
              "type": "normal",
              "weight_kg": 54.43114913227677,
              "reps": 7,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 3,
              "type": "normal",
              "weight_kg": 45.35929094356398,
              "reps": 9,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            }
          ]
        },
        {
          "index": 2,
          "title": "Preacher Curl (Machine)",
          "notes": None,
          "exercise_template_id": "1E9A6B8E",
          "superset_id": None,
          "sets": [
            {
              "index": 0,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 13,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 1,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 9,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 2,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 6,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 3,
              "type": "normal",
              "weight_kg": 22.67964547178199,
              "reps": 8,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            }
          ]
        },
        {
          "index": 3,
          "title": "Rear Delt Reverse Fly (Machine)",
          "notes": None,
          "exercise_template_id": "D8281C62",
          "superset_id": None,
          "sets": [
            {
              "index": 0,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 10,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 1,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 6,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 2,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 10,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 3,
              "type": "normal",
              "weight_kg": 24.947610018960187,
              "reps": 10,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            }
          ]
        },
        {
          "index": 4,
          "title": "Bicep Curl (Machine)",
          "notes": None,
          "exercise_template_id": "AF328E3D",
          "superset_id": None,
          "sets": [
            {
              "index": 0,
              "type": "normal",
              "weight_kg": 36.28743275485118,
              "reps": 8,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 1,
              "type": "normal",
              "weight_kg": 36.28743275485118,
              "reps": 6,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 2,
              "type": "normal",
              "weight_kg": 31.75150366049478,
              "reps": 7,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            },
            {
              "index": 3,
              "type": "normal",
              "weight_kg": 24.947610018960187,
              "reps": 10,
              "distance_meters": None,
              "duration_seconds": None,
              "custom_metric": None
            }
          ]
        }
      ]
    }

    result = await tool_update_routine.ainvoke({"routine_id": routine_id, "routine_data":updated_data})
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
    # print(routine_id)
    
    # await test_update_routine()

    # await test_update_workout()

asyncio.run(main())