# eval.py

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langsmith import Client, evaluate, traceable, wrappers, schemas # Import schemas
from openai import OpenAI
from pydantic import BaseModel, Field
import logging

# --- Configuration ---
load_dotenv()
os.environ['LANGSMITH_API_KEY'] = os.environ.get('LANGSMITH_API')
os.environ['LANGSMITH_TRACING'] = os.environ.get('LANGSMITH_TRACING')
os.environ['LANGSMITH_PROJECT'] = os.environ.get('LANGSMITH_PROJECT')
LANGSMITH_PROJECT_NAME = os.environ.get("LANGSMITH_PROJECT")

JUDGE_LLM_MODEL = "deepseek-chat" # Or "gpt-4o", etc.


FETCH_RUNS_LIMIT = 100 # Start small to test again, increase later
FEEDBACK_KEY = f"llm_run_quality_score_{datetime.now().strftime('%Y%m%d')}" # Date-encoded key

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Adjust logging for the specific client library if needed
# logging.getLogger("openai").setLevel(logging.WARNING)

# --- Initialize Clients ---
try:
    ls_client = Client()
    logger.info("LangSmith client initialized.")

    # Initialize the correct client for your JUDGE_LLM_MODEL
    
    oai_client = wrappers.wrap_openai(OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"))
    

except Exception as e:
    logger.error(f"Error initializing clients: {e}", exc_info=True)
    exit(1)

if not LANGSMITH_PROJECT_NAME:
    logger.error("LANGSMITH_PROJECT environment variable not set.")
    exit(1)


# --- Helper Functions for LLM Run Structure (extract_llm_prompt, extract_llm_completion) ---
# (Use the versions from the previous response - they worked for dataset creation)
def extract_llm_prompt(inputs: Optional[Dict[str, Any]], run_id: str) -> Optional[str]:
    """Extracts the full prompt string sent to the LLM from an LLM run's inputs."""
    if not inputs or not isinstance(inputs, dict) or "messages" not in inputs:
        logger.debug(f"Run {run_id}: 'messages' field missing or invalid in inputs.")
        return None
    messages_data = inputs["messages"]
    if isinstance(messages_data, list) and len(messages_data) == 1 and isinstance(messages_data[0], list):
        messages = messages_data[0]
    elif isinstance(messages_data, list):
         messages = messages_data
    else:
        logger.warning(f"Run {run_id}: Unexpected structure for 'messages' in inputs: {type(messages_data)}")
        return None
    prompt_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("kwargs", {}).get("content") or msg.get("content")
            role = msg.get("kwargs", {}).get("role") or msg.get("role")
            if not role and isinstance(msg.get("id"), list):
                 if any("SystemMessage" in str(part) for part in msg["id"]): role = "system"
                 elif any("HumanMessage" in str(part) for part in msg["id"]): role = "user"
                 elif any("AIMessage" in str(part) for part in msg["id"]): role = "assistant"
                 else: role = "unknown"
            if content:
                prompt_parts.append(f"<{role or 'message'}>\n{str(content)}\n</{role or 'message'}>")
        elif isinstance(msg, str):
             prompt_parts.append(str(msg))
    if prompt_parts:
         full_prompt = "\n".join(prompt_parts)
         logger.debug(f"Run {run_id}: Extracted LLM prompt successfully.")
         return full_prompt
    else:
         logger.warning(f"Run {run_id}: Could not extract any message content from inputs: {json.dumps(inputs, default=str)[:500]}")
         return None

def extract_llm_completion(outputs: Optional[Dict[str, Any]], run_id: str) -> Optional[str]:
    """Extracts the LLM's generated completion from an LLM run's outputs."""
    if not outputs or not isinstance(outputs, dict) or "generations" not in outputs:
        logger.debug(f"Run {run_id}: 'generations' field missing or invalid in outputs.")
        return None
    generations = outputs["generations"]
    if isinstance(generations, list) and len(generations) >= 1 and isinstance(generations[0], list) and len(generations[0]) >= 1:
        generation_data = generations[0][0]
    elif isinstance(generations, list) and len(generations) >= 1:
        generation_data = generations[0]
    else:
        logger.warning(f"Run {run_id}: Unexpected structure or empty 'generations' in outputs: {type(generations)}")
        return None
    if isinstance(generation_data, dict):
        text_content = generation_data.get("text")
        if text_content is not None:
            logger.debug(f"Run {run_id}: Extracted LLM completion from 'text' field.")
            return str(text_content)
        message_data = generation_data.get("message")
        if isinstance(message_data, dict):
             content = message_data.get("kwargs", {}).get("content") or message_data.get("content")
             if content is not None:
                 logger.debug(f"Run {run_id}: Extracted LLM completion from nested 'message' field.")
                 return str(content)
    logger.warning(f"Run {run_id}: Could not extract text/content from generation data: {json.dumps(generation_data, default=str)[:500]}")
    return None

# --- Fetch LLM Runs ---
def fetch_runs_for_evaluation(project_name: str, limit: int) -> List[schemas.Run]:
    """Fetches LLM runs from a LangSmith project for evaluation."""
    logger.info(f"Fetching up to {limit} LLM runs from project '{project_name}'...")
    try:
        runs_iterator = ls_client.list_runs(
            project_name=project_name,
            run_type="llm",
            error=False,
            limit=limit,
        )
        runs = list(runs_iterator)
        logger.info(f"Fetched {len(runs)} LLM runs.")
        return runs
    except Exception as e:
        logger.error(f"Error fetching runs from LangSmith: {e}", exc_info=True)
        return []


# --- LLM Judge Logic ---
class JudgeLLMRunResult(BaseModel):
    score: int = Field(..., description="The numeric score from 0 to 100.", ge=0, le=100)
    reasoning: str = Field(..., description="Detailed reasoning for the score based on the criteria.")

def get_llm_run_judgement(llm_prompt: str, llm_completion: str, run_id_for_log: str) -> Optional[Dict]:
    """Calls the judge LLM for a single prompt/completion pair."""
    if llm_prompt is None or llm_completion is None:
        logger.warning(f"Skipping judgement for run {run_id_for_log}: Missing prompt or completion.")
        return None

    judge_prompt = f"""You are an expert evaluator assessing the quality of a single Large Language Model (LLM) response within a larger fitness chatbot system.
Evaluate the LLM's completion based on the provided prompt and the following criteria, providing a score from 0 to 100 and detailed reasoning.

**Evaluation Criteria:**

1.  **Instruction Following & Formatting (Weight: 40%):** Did the LLM adhere to the instructions in the prompt? If a specific format (e.g., JSON, specific tags) was requested, did the output match it?
2.  **Relevance (Weight: 30%):** Is the completion directly relevant to the input prompt? Does it address the core task described?
3.  **Coherence & Quality (Weight: 30%):** Is the completion well-written, coherent, and free of obvious grammatical errors or nonsensical statements?

**LLM Input Prompt:**

{llm_prompt}
**LLM's Completion:**

{llm_completion}
**Task:**
Provide a final score (0-100) reflecting an overall assessment based on the weighted criteria. Provide clear reasoning for your score, referencing specific parts of the prompt and completion.

**Output Format:**
Your response MUST be a JSON object containing ONLY the keys "score" (an integer between 0 and 100) and "reasoning" (a string explaining the score). Do not include any other text or markdown formatting. Example: {{"score": 85, "reasoning": "The model followed instructions well..."}}
"""

    try:
        completion = oai_client.chat.completions.create(
            model=JUDGE_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator grading an LLM's response. Respond only in the requested JSON format."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0
        )
        response_content = completion.choices[0].message.content
        cleaned_content = response_content.strip()
        if cleaned_content.startswith("```json"): cleaned_content = cleaned_content[7:].strip()
        if cleaned_content.startswith("```"): cleaned_content = cleaned_content[3:].strip()
        if cleaned_content.endswith("```"): cleaned_content = cleaned_content[:-3].strip()

        try:
            parsed_data = json.loads(cleaned_content)
            # Validate using Pydantic
            validated_result = JudgeLLMRunResult(**parsed_data)
            logger.debug(f"LLM Judge successful for run {run_id_for_log}: Score={validated_result.score}")
            return validated_result.model_dump() # Return dict from validated model
        except (json.JSONDecodeError, TypeError, ValueError) as parse_error: # Catches Pydantic validation errors too
            logger.error(f"Failed to parse/validate LLM Judge response for run {run_id_for_log}. Error: {parse_error}. Cleaned: '{cleaned_content}'. Original: '{response_content}'", exc_info=False) # Keep log concise
            return None
    except Exception as e:
        logger.error(f"Error during LLM Judge API call for run {run_id_for_log}: {e}", exc_info=True)
        return None

# --- Evaluation Loop using create_feedback ---
# --- MODIFIED Evaluation Loop to Accumulate Scores ---
def evaluate_runs_and_get_average(runs: List[schemas.Run], feedback_key: str) -> Optional[float]:
    """
    Iterates through runs, gets judgement, posts feedback, and calculates average score.
    Returns the average score (0-100) or None if no scores were recorded.
    """
    logger.info(f"Starting evaluation for {len(runs)} runs using create_feedback...")
    successful_evals = 0
    failed_evals = 0
    recorded_scores = [] # List to store scores that were successfully posted

    for i, run in enumerate(runs):
        run_id_str = str(run.id)
        logger.info(f"--- Processing run {i+1}/{len(runs)} (ID: {run_id_str}) ---")
        llm_prompt = extract_llm_prompt(run.inputs, run_id_str)
        llm_completion = extract_llm_completion(run.outputs, run_id_str)

        if llm_prompt is None or llm_completion is None:
            logger.warning(f"Skipping feedback for run {run_id_str}: Could not extract prompt/completion.")
            failed_evals += 1
            continue

        judgement = get_llm_run_judgement(llm_prompt, llm_completion, run_id_str)

        if judgement and isinstance(judgement, dict) and "score" in judgement:
            # Store the original 0-100 score before trying to post feedback
            current_score = judgement["score"]
            try:
                ls_client.create_feedback(
                    run_id=run.id,
                    key=feedback_key,
                    score=current_score / 100.0, # Normalize score for LangSmith
                    comment=judgement.get("reasoning", "No reasoning provided."),
                )
                logger.info(f"Posted feedback for run {run_id_str} (Score: {current_score})")
                recorded_scores.append(current_score) # Add score to list *after* successful post
                successful_evals += 1
            except Exception as feedback_err:
                logger.error(f"Failed to post feedback for run {run_id_str}: {feedback_err}", exc_info=True)
                failed_evals += 1 # Count as failed if feedback post fails
        else:
            logger.warning(f"Skipping feedback for run {run_id_str}: Judgement failed or returned invalid format.")
            failed_evals += 1

    logger.info(f"Finished evaluation loop. Successful judgements posted: {successful_evals}, Failures/Skipped: {failed_evals}")

    # Calculate and return average score
    if recorded_scores:
        average_score = sum(recorded_scores) / len(recorded_scores)
        logger.info(f"Average Score ({feedback_key}) across {len(recorded_scores)} evaluated runs: {average_score:.2f}")
        return average_score
    else:
        logger.warning("No scores were successfully recorded. Cannot calculate average.")
        return None

# --- Main Execution Logic ---
def evaluate_agent(feedback_key=FEEDBACK_KEY, langsmith_project_name=LANGSMITH_PROJECT_NAME, fetch_runs_limit=FETCH_RUNS_LIMIT):
    logger.info(f"--- Starting Fitness Chatbot LLM Run Evaluation Script (Key: {feedback_key}) ---")

    # 1. Fetch LLM Runs
    fetched_runs = fetch_runs_for_evaluation(langsmith_project_name, fetch_runs_limit)

    if not fetched_runs:
        logger.error("No LLM runs fetched. Exiting evaluation.")
        exit(1)

    # 2. Run Evaluation and Get Average Score
    average_accuracy = evaluate_runs_and_get_average(fetched_runs, feedback_key)

    # 3. Report Final Result
    if average_accuracy is not None:
         logger.info(f"  Overall Average LLM Run Quality Score: {average_accuracy:.2f} / 100")
    else:
         logger.info("  Could not calculate average score (no successful evaluations).")


    logger.info("--- Fitness Chatbot LLM Run Evaluation Script Finished ---")
    logger.info(f"Check LangSmith project '{langsmith_project_name}' for individual run feedback ({feedback_key}).")

    return average_accuracy

if __name__ == "__main__":
    # Run evaluation and output results as JSON to stdout
    avg = evaluate_agent()
    # Print JSON to stdout for GitHub Actions to capture
    out = {"accuracy": round(avg, 2) if avg is not None else 0.0}
    print(json.dumps(out))
