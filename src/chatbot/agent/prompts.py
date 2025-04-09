from langsmith import Client
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LangSmith client
client = Client(api_key=os.environ.get('LANGSMITH_API'))

# Define your prompt templates
USER_MODELER_TEMPLATE = """
You are a user modeling specialist for a fitness training system. Analyze all available information about the user to build a comprehensive model:
1. Extract explicit information (stated goals, preferences, constraints)
2. Infer implicit information (fitness level, motivation factors, learning style)
3. Identify gaps in our understanding that need to be addressed
4. Update confidence levels for different aspects of the model

Current user model: {user_model}
Recent exchanges: {recent_exchanges}

Return an updated user model with confidence scores for each attribute in the following JSON format:
{format_instructions}
"""

MEMORY_CONSOLIDATION_TEMPLATE = """You are the memory manager for a fitness training system. Review the conversation history and current agent states to:
1. Identify key information that should be stored in long-term memory
2. Update the user model with new insights
3. Consolidate redundant information
4. Prune outdated or superseded information
5. Ensure critical context is available in working memory

Current long-term memory: {memory}
Current user model: {user_model}
Current working memory: {working_memory}

Return a structured update of what should be stored, updated, or removed.
"""

COORDINATOR_TEMPLATE = """You are the coordinator for a personal fitness trainer AI. Your role is to:
1. Understand the user's current needs and context
2. Determine which specialized agent should handle the interaction
3. Provide a coherent experience across different interactions
4. Ensure all user needs are addressed appropriately
5. Conduct user assessment when needed

You have direct access to these specialized capabilities:
- Research: Retrieve scientific fitness knowledge using the retrieve_from_rag tool
- Planning: Create personalized workout routines
- Progress Analysis and Adaptation: Analyze workout data, track progress and, update the user's workout routines based on the analysis.
- Coach: Provide motivation and adherence strategies
- User Modeler: Updates the user model with new information from the user

IMPORTANT ROUTING INSTRUCTIONS:
- When a user responds to an assessment question, ALWAYS route to <User Modeler> first
- The User Modeler will update the profile and then route back to you for next steps
- If assessment is complete but research_findings is empty, route to <Research>
- If assessment is complete and research_findings exists but seems irrelevant to current user goals, route to <Research>
- Only route to <Planning> when assessment is complete AND relevant research is available
- Only route to <Progress_and_Adaptation> when the most recent message either explictly asks for analysis or modification of routine, or when there's a scheduled System trigger message.

Assessment process:
- If user profile is incomplete, you should ask assessment questions
- Required fields for assessment: goals, fitness_level, available_equipment, training_environment, schedule, constraints

RESPONSE FORMAT:
1. First provide your internal reasoning (not shown to user)
2. If choosing <Research>, include specific research needs in format: <research_needs>specific research topics and information needed based on user profile</research_needs>
3. End your internal reasoning with one of these agent tags:
<Assessment> - If you need to ask an assessment question
<Research> - If research information is not present, then choose this agent to conduct research in order to have enough information to create a routine.
<Planning> - If workout routine creation is needed, choose this agent. Only choose if the user model and the research findings are already present.
<Progress_and_Adaptation> - If progress analysis and/or routine modification is needed based on user feedback, logs, schedule triggers, or explicit requests to review progress or change routines.
<User Modeler> - If the user's message contains information that should update their profile, or if the user's message is a response to an assessment question with information in it, then choose this agent.
<Complete> - If you can directly handle the response
4. Then wrap your user-facing response in <user>...</user> tags

Current user model: {user_model}
Current fitness plan: {fitness_plan}
Recent interactions: {recent_exchanges}
Research findings: {research_findings}
"""

RESEARCH_TEMPLATE = """You are a fitness research specialist. Based on the user's profile and current needs:
1. Identify key scientific principles relevant to their goals
2. Retrieve evidence-based approaches from the rag knowledge base
3. Synthesize this information into actionable insights
4. Provide citations to specific sources

Current user profile: {user_profile}
Current research needs: {research_needs}

Use the retrieve_from_rag tool to access scientific fitness information.
"""

PLANNING_TEMPLATE = """You are a workout programming specialist. Create a detailed, personalized workout plan:
1. Design a structured routine based on scientific principles and user profile
2. Format the routine specifically for Hevy app integration
3. Include exercise selection, sets, reps, rest periods, and progression scheme
4. Provide clear instructions for implementation

User profile: {user_profile}
Research findings: {research_findings}

    
The routine you provided will be converted into pydantic base classes and accessed this way by the user:
exercises = []
for exercise_data in routine_structure.get("exercises", []):
    sets = []
    for set_data in exercise_data.get("sets", []):
        sets.append(SetRoutineCreate(
            type=set_data.get("type", "normal"),
            weight_kg=set_data.get("weight", 0.0),
            reps=set_data.get("reps", 0),
            duration_seconds=set_data.get("duration", None),
            distance_meters=set_data.get("distance", None)
        ))
    
    exercises.append(ExerciseRoutineCreate(
        exercise_template_id=exercise_data.get("exercise_id", ""),
        exercise_name=exercise_data.get("exercise_name", ""),
        exercise_type=exercise_data.get("exercise_type", "strength"),
        sets=sets,
        notes=exercise_data.get("notes", ""),
        rest_seconds=60  # Default rest time
    ))

# Create the full routine object
routine = RoutineCreate(
    title=routine_structure.get("title", "Personalized Routine"),
    notes=routine_structure.get("notes", "AI-generated routine"),
    exercises=exercises
)
So make sure you include all the fields in your response
"""

ANALYSIS_TEMPLATE = """You are a fitness progress analyst. Examine the user's workout logs to:
1. Track adherence to the planned routine
2. Identify trends in performance (improvements, plateaus, regressions)
3. Compare actual progress against expected progress
4. Suggest specific adjustments to optimize results

User profile: {user_profile}
Current fitness plan: {fitness_plan}
Recent workout logs: {workout_logs}

Use the tool_fetch_workouts tool to access workout logs from Hevy.
"""

ADAPTATION_TEMPLATE = """You are a workout adaptation specialist. Based on progress data and user feedback:
1. Identify specific aspects of the routine that need modification
2. Apply scientific principles to make appropriate adjustments
3. Ensure changes align with the user's goals and constraints
4. Update the routine in Hevy

User profile: {user_profile}
Current fitness plan: {fitness_plan}
Progress data: {progress_data}
Suggested adjustments: {suggested_adjustments}

Use the tool_update_routine tool to update the routine in Hevy.
"""

COACH_TEMPLATE = """You are a fitness motivation coach. Your role is to:
1. Provide encouragement and motivation tailored to the user's profile
2. Offer strategies to improve adherence and consistency
3. Address psychological barriers to fitness progress
4. Celebrate achievements and milestones

User profile: {user_profile}
Progress data: {progress_data}
Recent exchanges: {recent_exchanges}

Be supportive, empathetic, and science-based in your approach.
"""


SUMMARIZE_ROUTINE_TEMPLATE = """You are an assistant summarizing a generated workout plan for a user.
The following JSON data represents one or more workout routines intended for the Hevy app.
DO NOT include technical details like 'exercise_template_id' or 'superset_id'.
Focus on presenting the plan clearly: Routine Title, Exercises (by name/notes), Sets (Type, Weight, Reps), and Rest Times.

Hevy Routine Payloads:
{hevy_results_json}

Generate a user-friendly text summary of the workout plan(s) described in the JSON data. If the list is empty, state that no routines were generated.
Start the summary directly. Example: "Okay, here is the workout plan I've created for you:"
"""

ANALYSIS_TEMPLATE_V2 = """You are a fitness progress analyst reviewing a user's workout data from the Hevy app, specifically in the context of ONE potential target routine.
Your goal is to:
1.  Generate a Comprehensive Report Summary: Summarize the user's workout history from the provided logs *as it relates to the exercises in the target routine*. Include adherence (if logs match routine structure), performance trends (volume, weight, reps on matching exercises), consistency, and any notable achievements or plateaus for exercises in *this specific routine*.
2.  Identify Key Observations: List the most important positive and negative trends observed *related to this routine*.
3.  Pinpoint Areas for Adjustment/Research: Based on the report and the user's profile/goals, identify specific, actionable areas where *this specific routine* might be adjusted or where further research is needed.

User Profile:
{user_profile}

Target Routine Details:
{target_routine_details}

Recent Workout Logs (provide context for analysis):
{workout_logs}

{format_instructions} # For AnalysisFindings model

Output ONLY the JSON object conforming to the AnalysisFindings structure.
"""

def get_analysis_v2_template(version=None):
    return PromptTemplate(
        input_variables=["user_profile", "target_routine_details", "workout_logs", "format_instructions"],
        template=ANALYSIS_TEMPLATE_V2
    )

# Use TARGETED_RAG_QUERY_TEMPLATE content from previous messages
TARGETED_RAG_QUERY_TEMPLATE = """Based on the user's profile and a specific area identified for potential adjustment in their fitness plan, generate a concise, targeted query for our fitness science RAG system (`retrieve_from_rag`).

User Profile:
{user_profile}

Area for Adjustment/Research: "{area_for_adjustment}"

Previous RAG Query (if any for this area): {previous_query}
Previous RAG Result (if any for this area): {previous_result}

Generate the *next* best query string to get specific, actionable scientific information related to the adjustment area. Focus on principles, techniques, or evidence. Output ONLY the query string.
"""
def get_targeted_rag_query_template(version=None):
    return PromptTemplate(
        input_variables=["user_profile", "area_for_adjustment", "previous_query", "previous_result"],
        template=TARGETED_RAG_QUERY_TEMPLATE
    )

# Use ROUTINE_MODIFICATION_TEMPLATE_V2 content from previous messages
ROUTINE_MODIFICATION_TEMPLATE_V2 = """You are an expert workout adaptation specialist. You are given the user's current workout routine as a JSON object and findings from their progress analysis and research for *this specific routine*. Your task is to **modify the provided JSON object** to incorporate necessary adjustments based on the findings and return the **entire, updated JSON object**.

**Critical Instructions:**
1.  **Input & Output:** You will receive the `current_routine_json`. Your output MUST be a valid JSON object representing the *complete*, *modified* routine, adhering strictly to the structure of the input JSON.
2.  **Modify, Don't Recreate:** Make targeted changes to the input JSON based on the `analysis_findings` and `adaptation_rag_results`. Do NOT create a new structure from scratch.
3.  **Preserve Metadata:** Ensure all original keys (like `id`, `folder_id`, `updated_at`, `created_at`, `index`, `custom_metric` within sets, etc.) are present in your output JSON, even if their values are unchanged or null. The API requires the full object structure.
4.  **Handling Exercise Changes:**
    *   **Modification:** To change sets, reps, weight, rest, or notes for an *existing* exercise, modify its properties directly within its JSON object. Keep its original `exercise_template_id`.
    *   **Deletion:** To remove an exercise, simply omit its entire JSON object from the `exercises` list in your output.
    *   **Addition:** To add a *new* exercise:
        *   Create a new exercise JSON object in the desired position within the `exercises` list.
        *   Set `"exercise_template_id": null`.
        *   Provide the **full, specific name** of the desired exercise (including equipment, e.g., 'Squat (Barbell)', 'Dumbbell Bench Press') in the `"title"` field.
        *   Fill in the `sets`, `rest_seconds`, `notes`, etc., for the new exercise.
    *   **Replacement:** To replace an *existing* exercise with a different one:
        *   Remove the original exercise's JSON object from the `exercises` list.
        *   Add a *new* exercise JSON object (as described in "Addition" above) in its place, setting `"exercise_template_id": null` and using the new exercise's specific name in the `"title"` field.
    *   **IMPORTANT:** NEVER invent or guess an `exercise_template_id`. Only use `null` for additions/replacements. The system will look up the correct ID based on the `title` you provide.
5.  **Explain Changes:** You will be asked to generate reasoning in a separate step. Focus *only* on outputting the modified JSON here.
6.  **Focus:** Modify weights, reps, sets, notes, add/replace/delete exercises based *only* on the `analysis_findings` and `adaptation_rag_results`. Ensure `rest_seconds` is present.

User Profile:
{user_profile}

Analysis Findings (for this routine):
{analysis_findings} # Example: "User plateaud on Barbell Bench Press. Consider swapping for Dumbbell Bench Press.", "Add Lateral Raises for shoulder width."

Relevant RAG Research Results (for this routine):
{adaptation_rag_results} # Example: "Dumbbell presses allow greater range of motion...", "Lateral raises effectively target medial deltoid..."

Current Routine JSON (Modify this structure):
```json
{current_routine_json}
```
Output only the complete, modified JSON object for the routine. Do not include ```json markdown delimiters or any other text.
"""

def get_routine_modification_template_v2(version=None):
    return PromptTemplate(
    input_variables=["user_profile", "analysis_findings", "adaptation_rag_results", "current_routine_json"],
    template=ROUTINE_MODIFICATION_TEMPLATE_V2
    )

REASONING_GENERATION_TEMPLATE = """Based on the changes made between the original routine and the modified routine, and considering the analysis/research findings, generate a concise user-facing explanation for the modifications made to this specific routine.

Original Routine Snippet (for context):
{original_routine_snippet}

Modified Routine Snippet (for context):
{modified_routine_snippet}

Analysis Findings (for this routine):
{analysis_findings}

Relevant RAG Research Results (for this routine):
{adaptation_rag_results}

Generate the reasoning text only.
"""

def get_reasoning_generation_template(version=None):
    return PromptTemplate(
    input_variables=["original_routine_snippet", "modified_routine_snippet", "analysis_findings", "adaptation_rag_results"],
    template=REASONING_GENERATION_TEMPLATE
    )

FINAL_CYCLE_REPORT_TEMPLATE_V2 = """You are an AI fitness coach summarizing the results of a potentially multi-routine progress analysis and adaptation cycle for the user.

User's Name: {user_name}

Summary of Processed Routines:
{processed_results_summary} # A formatted string summarizing outcome for each routine processed

Overall Cycle Status: {overall_status} # e.g., "Success", "Partial Success", "No Changes Made", "Failed"
Overall Message: {overall_message} # General concluding message based on status

Task: Generate a concise, clear, and encouraging final notification message for the user based on the information above.

Start with a greeting (e.g., "Hi {user_name},").

Use the overall_message as the main body.

Incorporate details from processed_results_summary if appropriate and not redundant with the overall message.

Keep the tone supportive and action-oriented if failures occurred (e.g., "I couldn't update routine X, we can look into that...").

Generate only the user-facing message.
"""
def get_final_cycle_report_template_v2(version=None):
    return PromptTemplate(
    input_variables=["user_name", "processed_results_summary", "overall_status", "overall_message"],
    template=FINAL_CYCLE_REPORT_TEMPLATE_V2
    )

ROUTINE_IDENTIFICATION_PROMPT = """You are an AI assistant analyzing a user's fitness routines and workout logs to identify which routine(s) should be targeted for adaptation.

User Profile:
{user_profile}

User's Explicit Request (if any): "{user_request_context}"

Available Saved Routines (List of JSON objects):

```json
{routines_list_json}
```

Recent Workout Logs (List of JSON objects):

```json
{logs_list_json}
```

Task: Identify the routine(s) from the Available Saved Routines list that are most relevant for adaptation based on the user's request, recent logs, and profile. Consider:

User Request: If the user mentioned a specific routine (e.g., "leg day", "push workout"), prioritize routines matching that description.

Log Matching: Find routines whose exercises strongly overlap with exercises performed in the most recent logs. Look for consistent use patterns.

Primary Routine: Identify if there seems to be a main routine the user follows most often.

Goals: Align the selected routine(s) with the user's stated goals.

Output Format:
Return a JSON list containing the full JSON object of each identified routine from the input Available Saved Routines list. Include a reason_for_selection key within each object in the output list.

Example Output:
```json
[
  {{
    "routine_data": {{ ... full routine object from input list ... }},
    "reason_for_selection": "Matches user request 'leg day' and has high overlap with recent logs."
  }},
  {{
    "routine_data": {{ ... another full routine object ... }},
    "reason_for_selection": "Appears to be the primary strength routine based on log frequency."
  }}
]
```

If no suitable routines are found, return an empty JSON list [].

Output only the JSON list. Do not include ```json markdown delimiters or any other text.
"""

def get_routine_identification_prompt(version=None):
    return PromptTemplate(
    input_variables=["user_profile", "user_request_context", "routines_list_json", "logs_list_json"],
    template=ROUTINE_IDENTIFICATION_PROMPT
    )


# Create the PromptTemplate object (add this with the others)
summarize_routine_prompt = PromptTemplate(
    input_variables=["hevy_results_json"],
    template=SUMMARIZE_ROUTINE_TEMPLATE
)


# Create the PromptTemplate object
user_modeler_prompt = PromptTemplate(
    input_variables=["user_model", "recent_exchanges", "format_instructions"],
    template=USER_MODELER_TEMPLATE
)

memory_consolidation_prompt = PromptTemplate(
    input_variables=["memory", "user_model", "working_memory"],
    template=MEMORY_CONSOLIDATION_TEMPLATE
)

coordinator_prompt = PromptTemplate(
    input_variables=["user_model", "fitness_plan", "recent_exchanges", "research_findings"],
    template=COORDINATOR_TEMPLATE
)

# Fix these templates by changing input_types to a dictionary
research_prompt = PromptTemplate(
    input_variables=["user_profile", "research_needs"],  # Add input_variables
    input_types={"user_profile": dict, "research_needs": list},  # Change to dict
    template=RESEARCH_TEMPLATE
)

planning_prompt = PromptTemplate(
    input_variables=["user_profile", "research_findings"],  # Add input_variables
    input_types={"user_profile": dict, "research_findings": dict},  # Change to dict
    template=PLANNING_TEMPLATE
)

analysis_prompt = PromptTemplate(
    input_variables=["user_profile", "fitness_plan", "workout_logs"],  # Add input_variables
    input_types={"user_profile": dict, "fitness_plan": dict, "workout_logs": list},  # Change to dict
    template=ANALYSIS_TEMPLATE
)

adaptation_prompt = PromptTemplate(
    input_variables=["user_profile", "fitness_plan", "progress_data", "suggested_adjustments"],  # Add input_variables
    input_types={"user_profile": dict, "fitness_plan": dict, "progress_data": dict, "suggested_adjustments": list},  # Change to dict
    template=ADAPTATION_TEMPLATE
)

coach_prompt = PromptTemplate(
    input_variables=["user_profile", "progress_data", "recent_exchanges"],  # Add input_variables
    input_types={"user_profile": dict, "progress_data": dict, "recent_exchanges": list},  # Change to dict
    template=COACH_TEMPLATE
)



import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

    
# ------ User Modeler Prompt Functions ------
def push_user_modeler_prompt():
    """Push the user modeler prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-user-modeler",
            object=user_modeler_prompt,
            description="Prompt for modeling fitness user profiles",
            tags=["fitness", "user-modeling"],
            is_public=False
        )
        logger.info(f"Pushed user modeler prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing user modeler prompt: {e}")
        raise

def get_user_modeler_prompt(version=None):
    """
    Get a specific version of the user modeler prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-user-modeler"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling user modeler prompt: {e}")
        return user_modeler_prompt  # Fallback to local version

# ------ Memory Consolidation Prompt Functions ------
def push_memory_consolidation_prompt():
    """Push the memory consolidation prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-memory-consolidation",
            object=memory_consolidation_prompt,
            description="Prompt for fitness memory consolidation",
            tags=["fitness", "memory-management"]
        )
        logger.info(f"Pushed memory consolidation prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing memory consolidation prompt: {e}")
        raise

def get_memory_consolidation_prompt(version=None):
    """
    Get a specific version of the memory consolidation prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-memory-consolidation"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling memory consolidation prompt: {e}")
        return memory_consolidation_prompt  # Fallback to local version

# ------ Coordinator Prompt Functions ------
def push_coordinator_prompt():
    """Push the coordinator prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-coordinator",
            object=coordinator_prompt,
            description="Prompt for fitness coordination and agent routing",
            tags=["fitness", "coordination", "routing"]
        )
        logger.info(f"Pushed coordinator prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing coordinator prompt: {e}")
        raise

def get_coordinator_prompt(version=None):
    """
    Get a specific version of the coordinator prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-coordinator"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling coordinator prompt: {e}")
        return coordinator_prompt  # Fallback to local version

# ------ Research Prompt Functions ------
def push_research_prompt():
    """Push the research prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-research",
            object=research_prompt,
            description="Prompt for fitness research and knowledge retrieval",
            tags=["fitness", "research", "knowledge"]
        )
        logger.info(f"Pushed research prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing research prompt: {e}")
        raise

def get_research_prompt(version=None):
    """
    Get a specific version of the research prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-research"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling research prompt: {e}")
        return research_prompt  # Fallback to local version

# ------ Planning Prompt Functions ------
def push_planning_prompt():
    """Push the planning prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-planning",
            object=planning_prompt,
            description="Prompt for fitness workout planning",
            tags=["fitness", "planning", "routines"]
        )
        logger.info(f"Pushed planning prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing planning prompt: {e}")
        raise

def get_planning_prompt(version=None):
    """
    Get a specific version of the planning prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-planning"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling planning prompt: {e}")
        return planning_prompt  # Fallback to local version

# ------ Analysis Prompt Functions ------
def push_analysis_prompt():
    """Push the analysis prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-analysis",
            object=analysis_prompt,
            description="Prompt for fitness progress analysis",
            tags=["fitness", "analysis", "progress"]
        )
        logger.info(f"Pushed analysis prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing analysis prompt: {e}")
        raise

def get_analysis_prompt(version=None):
    """
    Get a specific version of the analysis prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-analysis"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling analysis prompt: {e}")
        return analysis_prompt  # Fallback to local version

# ------ Adaptation Prompt Functions ------
def push_adaptation_prompt():
    """Push the adaptation prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-adaptation",
            object=adaptation_prompt,
            description="Prompt for fitness plan adaptation",
            tags=["fitness", "adaptation", "adjustment"]
        )
        logger.info(f"Pushed adaptation prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing adaptation prompt: {e}")
        raise

def get_adaptation_prompt(version=None):
    """
    Get a specific version of the adaptation prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-adaptation"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling adaptation prompt: {e}")
        return adaptation_prompt  # Fallback to local version

# ------ Coach Prompt Functions ------
def push_coach_prompt():
    """Push the coach prompt to LangSmith for versioning"""
    try:
        url = client.push_prompt(
            "fitness-coach",
            object=coach_prompt,
            description="Prompt for fitness coaching and motivation",
            tags=["fitness", "coaching", "motivation"]
        )
        logger.info(f"Pushed coach prompt to: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing coach prompt: {e}")
        raise

def get_coach_prompt(version=None):
    """
    Get a specific version of the coach prompt
    
    Args:
        version: Optional version identifier (commit hash or tag)
    
    Returns:
        The prompt template object
    """
    prompt_id = "fitness-coach"
    if version:
        prompt_id = f"{prompt_id}:{version}"
    try:
        return client.pull_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error pulling coach prompt: {e}")
        return coach_prompt  # Fallback to local version

# ------ General Prompt Utilities ------
def tag_prompt(prompt_name, commit_hash, tag_name):
    """
    Tag a specific version of a prompt
    
    Args:
        prompt_name: Base name of the prompt (e.g., 'coordinator')
        commit_hash: The commit hash to tag
        tag_name: The tag to apply (e.g., 'production', 'v1')
    """
    try:
        client.tag_prompt(f"fitness-{prompt_name}", commit_hash, tag_name)
        logger.info(f"Tagged {prompt_name} prompt version {commit_hash} as '{tag_name}'")
    except Exception as e:
        logger.error(f"Error tagging prompt: {e}")
        raise

def push_all_prompts():
    """Push all prompts to LangSmith for versioning"""
    results = {}
    try:
        
        results["memory_consolidation"] = push_memory_consolidation_prompt()
        results["coordinator"] = push_coordinator_prompt()
        results["research"] = push_research_prompt()
        results["planning"] = push_planning_prompt()
        results["analysis"] = push_analysis_prompt()
        results["adaptation"] = push_adaptation_prompt()
        results["coach"] = push_coach_prompt()
        results["user_modeler"] = push_user_modeler_prompt()
        logger.info("Successfully pushed all prompts to LangSmith")
    except Exception as e:
        logger.error(f"Error pushing all prompts: {e}")
    return results
