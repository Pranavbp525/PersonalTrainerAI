import json
import logging
import os

from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import PromptTemplate # ChatPromptTemplate is not used, removing

# Configure logging
logging.basicConfig(level=logging.INFO) # Set default logging level
logger = logging.getLogger(__name__)

load_dotenv()



# Initialize LangSmith client
try:
    client = Client() # Ensure LANGSMITH_API_KEY is used if that's your env var name
    logger.info("LangSmith client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {e}")
    # Depending on your application's needs, you might want to exit or disable LangSmith features
    client = None # Set client to None if initialization fails

# --- Prompt Template Definitions ---

# 1. User Modeler
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
user_modeler_prompt = PromptTemplate(
    input_variables=["user_model", "recent_exchanges", "format_instructions"],
    template=USER_MODELER_TEMPLATE
)

# 2. Memory Consolidation
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
memory_consolidation_prompt = PromptTemplate(
    input_variables=["memory", "user_model", "working_memory"],
    template=MEMORY_CONSOLIDATION_TEMPLATE
)

# 3. Coordinator
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
coordinator_prompt = PromptTemplate(
    input_variables=["user_model", "fitness_plan", "recent_exchanges", "research_findings"],
    template=COORDINATOR_TEMPLATE
)


# 8. Coach
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
coach_prompt = PromptTemplate(
    input_variables=["user_profile", "progress_data", "recent_exchanges"],
    template=COACH_TEMPLATE
)

# 9. Summarize Routine
SUMMARIZE_ROUTINE_TEMPLATE = """You are an assistant summarizing a generated workout plan for a user.
The following JSON data represents one or more workout routines intended for the Hevy app.
DO NOT include technical details like 'exercise_template_id' or 'superset_id'.
Focus on presenting the plan clearly: Routine Title, Exercises (by name/notes), Sets (Type, Weight, Reps), and Rest Times.

Hevy Routine Payloads:
{hevy_results_json}

Generate a user-friendly text summary of the workout plan(s) described in the JSON data. If the list is empty, state that no routines were generated.
Start the summary directly. Example: "Okay, here is the workout plan I've created for you:"
"""
summarize_routine_prompt = PromptTemplate(
    input_variables=["hevy_results_json"],
    template=SUMMARIZE_ROUTINE_TEMPLATE
)

# 10. Analysis V2
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
analysis_v2_prompt = PromptTemplate(
    input_variables=["user_profile", "target_routine_details", "workout_logs", "format_instructions"],
    template=ANALYSIS_TEMPLATE_V2
)

# 11. Targeted RAG Query
TARGETED_RAG_QUERY_TEMPLATE = """Based on the user's profile and a specific area identified for potential adjustment in their fitness plan, generate a concise, targeted query for our fitness science RAG system (`retrieve_from_rag`).

User Profile:
{user_profile}

Area for Adjustment/Research: "{area_for_adjustment}"

Previous RAG Query (if any for this area): {previous_query}
Previous RAG Result (if any for this area): {previous_result}

Generate the *next* best query string to get specific, actionable scientific information related to the adjustment area. Focus on principles, techniques, or evidence. Output ONLY the query string.
"""
targeted_rag_query_prompt = PromptTemplate(
    input_variables=["user_profile", "area_for_adjustment", "previous_query", "previous_result"],
    template=TARGETED_RAG_QUERY_TEMPLATE
)

# 12. Routine Modification V2
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
        *   Provide the **full, specific name** of the desired exercise (including equipment, e.g., 'Squat (Barbell)', 'Bench Press (Barbell)') in the `"title"` field if applicable.
        *   Fill in the `sets`, `rest_seconds`, `notes`, etc., for the new exercise.
    *   **Replacement:** To replace an *existing* exercise with a different one:
        *   Remove the original exercise's JSON object from the `exercises` list.
        *   Add a *new* exercise JSON object (as described in "Addition" above) in its place, setting `"exercise_template_id": null` and using the new exercise's specific name in the `"title"` field.
    *   **IMPORTANT:** NEVER invent or guess an `exercise_template_id`. Only use `null` for additions/replacements. The system will look up the correct ID based on the `title` you provide.
5.  **Prioritize User Request:** If the `user_request_context` provides a clear, specific, actionable instruction (e.g., "replace exercise X with Y", "remove exercise Z", "make this routine harder", "focus more on chest"), **prioritize fulfilling that request.** Use the `Analysis Findings` and `Relevant RAG Research Results` to inform *how* you implement the request if they are relevant to the request. (e.g., choosing appropriate sets/reps for a replacement) or to suggest *additional* changes if the request doesn't conflict with major issues found in the analysis. If the request is vague (e.g., "make it better"), rely more on the analysis and research.
6.  **Explain Changes:** You will be asked to generate reasoning in a separate step. Focus *only* on outputting the modified JSON here.
7.  **Focus:** Modify weights, reps, sets, notes, add/replace/delete exercises based *only* on the `user_request_context`, `analysis_findings` and `adaptation_rag_results`. Ensure `rest_seconds` is present.

User Profile:
{user_profile}

User's Specific Request Regarding This Adaptation Cycle:
"{user_request_context}"

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
routine_modification_v2_prompt = PromptTemplate(
input_variables=["user_profile", "user_request_context", "analysis_findings", "adaptation_rag_results", "current_routine_json"],
template=ROUTINE_MODIFICATION_TEMPLATE_V2
)

# 13. Reasoning Generation

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
reasoning_generation_prompt = PromptTemplate(
input_variables=["original_routine_snippet", "modified_routine_snippet", "analysis_findings", "adaptation_rag_results"],
template=REASONING_GENERATION_TEMPLATE
)

# 14. Final Cycle Report V2

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
final_cycle_report_v2_prompt = PromptTemplate(
input_variables=["user_name", "processed_results_summary", "overall_status", "overall_message"],
template=FINAL_CYCLE_REPORT_TEMPLATE_V2
)

# 15. Routine Identification

ROUTINE_IDENTIFICATION_PROMPT = """You are an AI assistant analyzing a user's fitness routines and workout logs to identify which routine(s) should be targeted for adaptation.

User Profile:
{user_profile}

User's Explicit Request (if any): "{user_request_context}"

Available Saved Routines (List of JSON objects):

{routines_list_json}


Recent Workout Logs (List of JSON objects):

{logs_list_json}


Task: Identify the routine(s) from the Available Saved Routines list that are most relevant for adaptation based on the user's request, recent logs, and profile. Consider:

User Request: If the user mentioned a specific routine (e.g., "leg day", "push workout"), prioritize routines matching that description.

Log Matching: Find routines whose exercises strongly overlap with exercises performed in the most recent logs. Look for consistent use patterns.

Primary Routine: Identify if there seems to be a main routine the user follows most often.

Goals: Align the selected routine(s) with the user's stated goals.

Output Format:
Return a JSON list containing the full JSON object of each identified routine from the input Available Saved Routines list. Include a reason_for_selection key within each object in the output list.

Example Output:

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


If no suitable routines are found, return an empty JSON list [].

Output only the JSON list. Do not include ```json markdown delimiters or any other text.
"""
routine_identification_prompt = PromptTemplate(
input_variables=["user_profile", "user_request_context", "routines_list_json", "logs_list_json"],
template=ROUTINE_IDENTIFICATION_PROMPT
)


# --- Deep Research Agent Prompts ---
# 16. Plan Research Steps
PLAN_RESEARCH_STEPS_TEMPLATE = """Given the main research topic: '{research_topic}' for a user with this profile:
<user_profile>
{user_profile}
</user_profile>
Break this down into 3-5 specific, actionable sub-questions relevant to fitness science that can likely be answered using our internal knowledge base (RAG system). Focus on aspects like training principles, exercise selection, progression, nutrition timing, recovery, etc., as relevant to the topic.
Output ONLY a JSON list of strings, where each string is a sub-question. Example:
["What are the optimal rep ranges for muscle hypertrophy based on recent studies?", "How does protein timing affect muscle protein synthesis post-workout?", "What are common exercise modifications for individuals with lower back pain?"]
"""
plan_research_steps_prompt = PromptTemplate(
input_variables=["research_topic", "user_profile"],
template=PLAN_RESEARCH_STEPS_TEMPLATE
)

# 17. Generate RAG Query V2
GENERATE_RAG_QUERY_V2_TEMPLATE = """You are a research assistant formulating queries for an internal fitness science knowledge base (RAG system accessed via the retrieve_data function).
Current Research Sub-Question: "{current_sub_question}"
Query attempt number {queries_this_sub_question} for this sub-question.
Accumulated Findings So Far:
<findings>
{accumulated_findings}
</findings>
Previous Reflections on Progress:
<reflections>
{reflections}
</reflections>
Based on the current sub-question and the information gathered or reflected upon so far, formulate the single, most effective query string to retrieve the next piece of relevant scientific information from our fitness RAG system. Be specific and targeted. If previous attempts failed to yield useful info, try a different angle.
Output only the query string itself, without any explanation or preamble.
"""
generate_rag_query_v2_prompt = PromptTemplate(
input_variables=["current_sub_question", "queries_this_sub_question", "accumulated_findings", "reflections"],
template=GENERATE_RAG_QUERY_V2_TEMPLATE
)

#18. Synthesize RAG Results
SYNTHESIZE_RAG_RESULTS_TEMPLATE = """You are a research assistant synthesizing information for a fitness report.
Current Research Sub-Question: "{current_sub_question}"
Existing Accumulated Findings:
<existing_findings>
{accumulated_findings}
</existing_findings>
Newly Retrieved Information from Knowledge Base (RAG):
<new_info>
{rag_results}
</new_info>
Task: Integrate the key points from the "Newly Retrieved Information" into the "Existing Accumulated Findings". Focus only on information directly relevant to the "Current Research Sub-Question". Update the findings concisely and maintain a logical flow. Avoid redundancy. If the new info isn't relevant or adds nothing substantially new, state that briefly within the updated findings.
Output only the complete, updated accumulated findings text. Do not include headers like "Updated Findings".
"""
synthesize_rag_results_prompt = PromptTemplate(
input_variables=["current_sub_question", "accumulated_findings", "rag_results"],
template=SYNTHESIZE_RAG_RESULTS_TEMPLATE
)

#19. Reflect on Progress V2
REFLECT_ON_PROGRESS_V2_TEMPLATE = """You are an expert research assistant evaluating the progress on a specific fitness research sub-question.
Current Sub-Question: "{current_sub_question}"
Number of queries made for this sub-question: {queries_this_sub_question} (Max recommended: {max_queries_per_sub_question})
Accumulated Findings Gathered So Far:
<findings>
{accumulated_findings}
</findings>
Task: Critically evaluate the findings related specifically to the current sub-question.
Assess Sufficiency: Is the sub-question adequately answered based on the findings?
Identify Gaps: What specific, crucial information related to this sub-question is still missing or unclear?
Suggest Next Step: Based on the gaps, should we:
a) Perform another RAG query for this sub-question? (If yes, briefly suggest what to query for).
b) Conclude this sub-question and move to the next?
{force_completion_note}
Format your response clearly, addressing points 1, 2, and 3. Start your response with "CONCLUSION:" followed by either "CONTINUE_SUB_QUESTION" or "SUB_QUESTION_COMPLETE".
Example 1 (Needs More):
CONCLUSION: CONTINUE_SUB_QUESTION
Sufficiency: Partially answered...
Gaps: Need details on...
Next Step: Perform another RAG query focusing on...
Example 2 (Sufficient):
CONCLUSION: SUB_QUESTION_COMPLETE
Sufficiency: Yes, the findings cover the core aspects adequately.
Gaps: Minor details could be explored, but not critical.
Next Step: Conclude this sub-question.
"""
reflect_on_progress_v2_prompt = PromptTemplate(
input_variables=["current_sub_question", "queries_this_sub_question", "max_queries_per_sub_question", "accumulated_findings", "force_completion_note"],
template=REFLECT_ON_PROGRESS_V2_TEMPLATE
)

#20. Finalize Research Report
FINALIZE_RESEARCH_REPORT_TEMPLATE = """You are a research assistant compiling a final report based only on information gathered from our internal fitness science knowledge base.
Main Research Topic: "{research_topic}"
Original Research Plan (Sub-questions):
{sub_questions_json}
Accumulated Findings (Synthesized from RAG results):
<findings>
{accumulated_findings}
</findings>
Reflections During Research:
<reflections>
{reflections_json}
</reflections>
Task: Generate a comprehensive, well-structured research report addressing the main topic.
Use only the information presented in the "Accumulated Findings". Do not add external knowledge.
Structure the report logically, perhaps following the flow of the sub-questions, synthesizing related points.
Incorporate insights or limitations mentioned in the "Reflections" where appropriate (e.g., mention if a topic was concluded due to limits or lack of info).
Ensure the report is clear, concise, and scientifically grounded based only on the provided findings.
Start the report directly. Do not include a preamble like "Here is the final report".
Output the final report text.
"""
finalize_research_report_prompt = PromptTemplate(
input_variables=["research_topic", "sub_questions_json", "accumulated_findings", "reflections_json"],
template=FINALIZE_RESEARCH_REPORT_TEMPLATE
)

# --- Routine Planner Agent Prompts ---
# 21. Structured Planning (Routine Creation)
STRUCTURED_PLANNING_TEMPLATE = """You are an expert personal trainer specializing in evidence-based workout programming.
Based on the user profile and research findings provided below, create a detailed workout plan.
Critical Instructions:
Exercise Names: For exercise_name in each exercise, use the MOST SPECIFIC name possible, including the equipment used (e.g., 'Bench Press (Barbell)', 'Squat (Barbell)', 'Lat Pulldown (Cable)', 'Arnold Press (Dumbbell)'). This is crucial for matching with the exercise database. Do NOT use generic names like 'Bench Press' or 'Row' without specifying equipment.
Completeness: Fill in all required fields based on the requested output structure. Provide reasonable defaults for optional fields if appropriate (e.g., rest times = 60s).
Multiple Routines: If the plan involves multiple workout days (e.g., Push/Pull/Legs), create a separate routine object within the routines list for each day.
User Profile:
{user_profile}
Research Findings:
{research_findings}
Generate the workout plan adhering strictly to the required output structure based on the user profile and research findings.
"""
structured_planning_prompt = PromptTemplate(
input_variables=["user_profile", "research_findings"],
template=STRUCTURED_PLANNING_TEMPLATE
)
#--- LangSmith Push/Pull Functions ---

def _push_prompt_to_langsmith(name, prompt_object, description, tags):
    """Helper function to push a prompt to LangSmith."""
    if not client:
        logger.warning(f"LangSmith client not initialized. Cannot push prompt '{name}'.")
        return None
    try:
        url = client.push_prompt(
        name,
        object=prompt_object,
        description=description,
        tags=["fitness"] + tags, # Add base 'fitness' tag
        is_public=False # Defaulting to private, adjust if needed
        )
        logger.info(f"Pushed '{name}' prompt to LangSmith: {url}")
        return url
    except Exception as e:
        logger.error(f"Error pushing '{name}' prompt to LangSmith: {e}")
        # Decide if you want to raise the exception or just return None
        # raise e
        return None

def _get_prompt_from_langsmith(name, fallback_prompt, version=None):
    """Helper function to pull a prompt from LangSmith with fallback."""
    if not client:
        logger.warning(f"LangSmith client not initialized. Using local fallback for prompt '{name}'.")
        return fallback_prompt

    prompt_id = name
    if version:
        prompt_id = f"{name}:{version}"
    try:
        pulled_prompt = client.pull_prompt(prompt_id)
        logger.info(f"Successfully pulled '{prompt_id}' prompt from LangSmith.")
        return pulled_prompt
    except Exception as e:
        logger.warning(f"Error pulling '{prompt_id}' prompt from LangSmith: {e}. Falling back to local version.")
        return fallback_prompt

# --- Specific Prompt Push/Get Functions ---
# 1. User Modeler

def push_user_modeler_prompt():
    return _push_prompt_to_langsmith(
    "fitness-user-modeler",
    user_modeler_prompt,
    "Prompt for modeling fitness user profiles",
    ["user-modeling", "assessment"]
    )
def get_user_modeler_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-user-modeler", user_modeler_prompt, version)

# 2. Memory Consolidation

def push_memory_consolidation_prompt():
    return _push_prompt_to_langsmith(
    "fitness-memory-consolidation",
    memory_consolidation_prompt,
    "Prompt for fitness memory consolidation",
    ["memory-management", "state"]
    )
def get_memory_consolidation_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-memory-consolidation", memory_consolidation_prompt, version)

#3. Coordinator

def push_coordinator_prompt():
    return _push_prompt_to_langsmith(
    "fitness-coordinator",
    coordinator_prompt,
    "Prompt for fitness coordination and agent routing",
    ["coordination", "routing", "agent"]
    )
def get_coordinator_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-coordinator", coordinator_prompt, version)


#8. Coach

def push_coach_prompt():
    return _push_prompt_to_langsmith(
    "fitness-coach",
    coach_prompt,
    "Prompt for fitness coaching and motivation",
    ["coaching", "motivation", "support"]
    )
def get_coach_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-coach", coach_prompt, version)

#9. Summarize Routine

def push_summarize_routine_prompt():
    return _push_prompt_to_langsmith(
    "fitness-summarize-routine",
    summarize_routine_prompt,
    "Prompt to summarize generated Hevy routines for the user",
    ["summary", "user-facing", "planning"]
    )
def get_summarize_routine_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-summarize-routine", summarize_routine_prompt, version)

#10. Analysis V2

def push_analysis_v2_prompt():
    return _push_prompt_to_langsmith(
    "fitness-analysis-v2",
    analysis_v2_prompt,
    "Prompt for fitness progress analysis (V2 - single routine focus)",
    ["analysis", "progress", "v2", "adaptation-cycle"]
    )
def get_analysis_v2_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-analysis-v2", analysis_v2_prompt, version)

#11. Targeted RAG Query

def push_targeted_rag_query_prompt():
    return _push_prompt_to_langsmith(
    "fitness-targeted-rag-query",
    targeted_rag_query_prompt,
    "Prompt to generate targeted RAG queries for adaptation",
    ["research", "rag", "query-generation", "adaptation-cycle"]
    )
def get_targeted_rag_query_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-targeted-rag-query", targeted_rag_query_prompt, version)

#12. Routine Modification V2

def push_routine_modification_v2_prompt():
    return _push_prompt_to_langsmith(
    "fitness-routine-modification-v2",
    routine_modification_v2_prompt,
    "Prompt to modify a routine JSON based on analysis/research (V2)",
    ["adaptation", "adjustment", "routines", "v2", "adaptation-cycle"]
    )
def get_routine_modification_v2_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-routine-modification-v2", routine_modification_v2_prompt, version)

#13. Reasoning Generation

def push_reasoning_generation_prompt():
    return _push_prompt_to_langsmith(
    "fitness-reasoning-generation",
    reasoning_generation_prompt,
    "Prompt to generate user-facing reasoning for routine modifications",
    ["adaptation", "explanation", "user-facing", "adaptation-cycle"]
    )
def get_reasoning_generation_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-reasoning-generation", reasoning_generation_prompt, version)

#14. Final Cycle Report V2

def push_final_cycle_report_v2_prompt():
    return _push_prompt_to_langsmith(
    "fitness-final-cycle-report-v2",
    final_cycle_report_v2_prompt,
    "Prompt to generate the final user summary for an adaptation cycle (V2)",
    ["summary", "user-facing", "adaptation-cycle", "v2"]
    )
def get_final_cycle_report_v2_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-final-cycle-report-v2", final_cycle_report_v2_prompt, version)

#15. Routine Identification

def push_routine_identification_prompt():
    return _push_prompt_to_langsmith(
    "fitness-routine-identification",
    routine_identification_prompt,
    "Prompt to identify target routines for adaptation based on logs/request",
    ["adaptation", "analysis", "routing", "adaptation-cycle"]
    )
def get_routine_identification_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-routine-identification", routine_identification_prompt, version)

# --- Deep Research Agent Prompts ---
# 16. Plan Research Steps
def push_plan_research_steps_prompt():
    return _push_prompt_to_langsmith(
    "fitness-deep-research-plan-steps",
    plan_research_steps_prompt,
    "Prompt to break down a research topic into sub-questions",
    ["deep-research", "planning", "sub-questions"]
    )
def get_plan_research_steps_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-deep-research-plan-steps", plan_research_steps_prompt, version)

#17. Generate RAG Query V2
def push_generate_rag_query_v2_prompt():
    return _push_prompt_to_langsmith(
    "fitness-deep-research-generate-query",
    generate_rag_query_v2_prompt,
    "Prompt to generate the next RAG query for a sub-question",
    ["deep-research", "rag", "query-generation"]
    )
def get_generate_rag_query_v2_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-deep-research-generate-query", generate_rag_query_v2_prompt, version)

#18. Synthesize RAG Results
def push_synthesize_rag_results_prompt():
    return _push_prompt_to_langsmith(
    "fitness-deep-research-synthesize",
    synthesize_rag_results_prompt,
    "Prompt to integrate RAG results into accumulated findings",
    ["deep-research", "synthesis", "rag"]
    )
def get_synthesize_rag_results_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-deep-research-synthesize", synthesize_rag_results_prompt, version)

#19. Reflect on Progress V2
def push_reflect_on_progress_v2_prompt():
    return _push_prompt_to_langsmith(
    "fitness-deep-research-reflect",
    reflect_on_progress_v2_prompt,
    "Prompt to reflect on progress for a sub-question and decide next step",
    ["deep-research", "reflection", "control-flow"]
    )
def get_reflect_on_progress_v2_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-deep-research-reflect", reflect_on_progress_v2_prompt, version)

#20. Finalize Research Report
def push_finalize_research_report_prompt():
    return _push_prompt_to_langsmith(
    "fitness-deep-research-finalize",
    finalize_research_report_prompt,
    "Prompt to generate the final research report from findings",
    ["deep-research", "reporting", "synthesis"]
    )
def get_finalize_research_report_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-deep-research-finalize", finalize_research_report_prompt, version)

#--- Routine Planner Agent Prompts ---
#21. Structured Planning (Routine Creation)
def push_structured_planning_prompt():
    return _push_prompt_to_langsmith(
    "fitness-structured-planning",
    structured_planning_prompt,
    "Prompt for generating structured workout routines (for with_structured_output)",
    ["planning", "routines", "structured-output", "generation"]
    )
def get_structured_planning_prompt(version=None):
    return _get_prompt_from_langsmith("fitness-structured-planning", structured_planning_prompt, version)

#--- General Prompt Utilities ---

def tag_prompt(prompt_name_suffix, commit_hash, tag_name):
    """
    Tag a specific version of a fitness prompt.

    Args:
        prompt_name_suffix: Base suffix of the prompt (e.g., 'coordinator', 'analysis-v2')
        commit_hash: The commit hash to tag
        tag_name: The tag to apply (e.g., 'production', 'v1.2')
    """
    if not client:
        logger.warning(f"LangSmith client not initialized. Cannot tag prompt 'fitness-{prompt_name_suffix}'.")
        return
    full_prompt_name = f"fitness-{prompt_name_suffix}"
    try:
        client.tag_prompt(full_prompt_name, commit_hash, tag_name)
        logger.info(f"Tagged {full_prompt_name} prompt version {commit_hash} as '{tag_name}'")
    except Exception as e:
        logger.error(f"Error tagging prompt {full_prompt_name}: {e}")
        # raise e # Optional: re-raise the exception


def push_all_prompts():
    """Push all defined fitness prompts to LangSmith for versioning."""
    if not client:
        logger.error("LangSmith client not initialized. Cannot push prompts.")
        return {}

    results = {}
    logger.info("Starting push of all prompts to LangSmith...")

    push_functions = {
        "user_modeler": push_user_modeler_prompt,
        "memory_consolidation": push_memory_consolidation_prompt,
        "coordinator": push_coordinator_prompt,
        "coach": push_coach_prompt,
        "summarize_routine": push_summarize_routine_prompt,
        "analysis_v2": push_analysis_v2_prompt,
        "targeted_rag_query": push_targeted_rag_query_prompt,
        "routine_modification_v2": push_routine_modification_v2_prompt,
        "reasoning_generation": push_reasoning_generation_prompt,
        "final_cycle_report_v2": push_final_cycle_report_v2_prompt,
        "routine_identification": push_routine_identification_prompt,
        # Add new deep research prompts
        "deep_research_plan_steps": push_plan_research_steps_prompt,
        "deep_research_generate_query": push_generate_rag_query_v2_prompt,
        "deep_research_synthesize": push_synthesize_rag_results_prompt,
        "deep_research_reflect": push_reflect_on_progress_v2_prompt,
        "deep_research_finalize": push_finalize_research_report_prompt,
        # Add new planning prompt
        "structured_planning": push_structured_planning_prompt,
    }

    success_count = 0
    fail_count = 0
    for name, push_func in push_functions.items():
        try:
            result_url = push_func()
            results[name] = result_url
            if result_url:
                success_count += 1
            else:
                # Error logged within the push function if client exists
                fail_count += 1
        except Exception as e:
            # Catch potential errors raised by push_func itself
            logger.error(f"Unhandled exception pushing prompt '{name}': {e}")
            results[name] = f"Error: {e}"
            fail_count += 1

    logger.info(f"Finished pushing prompts. Success: {success_count}, Failed: {fail_count}.")
    if fail_count > 0:
        logger.warning("Some prompts failed to push. Check logs for details.")
        return results

