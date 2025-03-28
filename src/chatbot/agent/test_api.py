import asyncio
from agent.llm_tools import tool_fetch_workouts, tool_get_workout_count, tool_create_routine

async def main():
    workouts = await tool_fetch_workouts.ainvoke({"page": 1, "page_size": 3})
    print("Workouts:", workouts)

    count = await tool_get_workout_count.ainvoke({})
    print("Workout Count:", count)

    # sample_routine = {
    #     "name": "LangChain Chest Day",
    #     "days": ["Monday"],
    #     "exercises": []
    # }
    # routine = await tool_create_routine.ainvoke({"routine_data": sample_routine})
    # print("Routine:", routine)

asyncio.run(main())
