# src/chatbot/agent_eval/eval.py

import os
import sys
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# --- REMOVED load_dotenv - rely on environment ---
# from dotenv import load_dotenv
from langsmith import Client, evaluate, traceable, wrappers, schemas
from openai import OpenAI
from pydantic import BaseModel, Field
import logging

# Add project root to sys.path to allow importing from src.rag_model
# This assumes the script is run from a context where the project root is accessible
# or that the calling process (like Airflow DAG) has already set up sys.path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up two levels from agent_eval to reach src, then one more to reach project root
    project_root_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
        logging.info(f"[Agent Eval] Added {project_root_path} to sys.path")

    # Now import the tracker
    from src.rag_model.mlflow.mlflow_rag_tracker import MLflowRAGTracker # Use src prefix
    mlflow_imported = True
except ImportError as import_err:
    logging.warning(f"[Agent Eval] Could not import MLflowRAGTracker. MLflow logging will be disabled. Error: {import_err}")
    mlflow_imported = False
    MLflowRAGTracker = None # Define as None if import fails
except Exception as path_err:
     logging.error(f"[Agent Eval] Error setting up sys.path: {path_err}", exc_info=True)
     mlflow_imported = False
     MLflowRAGTracker = None


# --- Configuration ---
# REMOVED load_dotenv() - Rely on environment variables being set externally (e.g., via .env in docker-compose)

# Get required variables from environment
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT_NAME = os.getenv('LANGSMITH_PROJECT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # Used for Judge LLM
# DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') # Uncomment if using Deepseek
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") # Default if not set

# Optional tracing variable
LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING', "true") # Default to true if not set
os.environ['LANGSMITH_TRACING'] = LANGSMITH_TRACING # Ensure it's set for LangSmith client

JUDGE_LLM_MODEL = "gpt-4o-mini" # Or your preferred judge model

FETCH_RUNS_LIMIT = 100 # Number of LangSmith runs to evaluate
FEEDBACK_KEY = f"llm_run_quality_score_{datetime.now().strftime('%Y%m%d')}" # Date-encoded key

# MLflow Configuration
AGENT_MLFLOW_EXPERIMENT_NAME = "Agent Evaluation" # Define experiment name


# --- Logging Setup ---
# Use standard logging config - Airflow handlers will capture it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger hierarchy
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langsmith").setLevel(logging.INFO) # Adjust LangSmith client logging if needed


# --- Validate Essential Configuration ---
if not LANGSMITH_API_KEY:
    logger.error("LANGSMITH_API_KEY environment variable not set.")
    raise ValueError("LANGSMITH_API_KEY environment variable not set.")
if not LANGSMITH_PROJECT_NAME:
    logger.error("LANGSMITH_PROJECT environment variable not set.")
    raise ValueError("LANGSMITH_PROJECT environment variable not set.")
if not OPENAI_API_KEY: # Check for the key needed by the judge
    logger.error(f"{JUDGE_LLM_MODEL} API key (e.g., OPENAI_API_KEY) environment variable not set.")
    raise ValueError(f"{JUDGE_LLM_MODEL} API key environment variable not set.")


# --- Initialize Clients ---
# Wrapped in function for clarity and potential retry logic later
def initialize_clients():
    try:
        ls_client = Client(api_key=LANGSMITH_API_KEY) # Pass key explicitly if needed
        logger.info("LangSmith client initialized.")

        # Initialize the correct client for your JUDGE_LLM_MODEL
        # Ensure the correct API key is used based on JUDGE_LLM_MODEL
        if "gpt-" in JUDGE_LLM_MODEL:
            judge_llm_client = wrappers.wrap_openai(OpenAI(api_key=OPENAI_API_KEY))
        # Example for Deepseek:
        # elif "deepseek-" in JUDGE_LLM_MODEL:
        #     deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        #     if not deepseek_key: raise ValueError("DEEPSEEK_API_KEY not set for judge model")
        #     judge_llm_client = wrappers.wrap_openai(OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"))
        else:
            raise ValueError(f"Unsupported JUDGE_LLM_MODEL specified: {JUDGE_LLM_MODEL}")

        logger.info(f"Judge LLM client initialized for model: {JUDGE_LLM_MODEL}")
        return ls_client, judge_llm_client

    except Exception as e:
        logger.error(f"Error initializing clients: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize clients: {e}") from e

# Initialize globally or within evaluate_agent
ls_client, judge_llm_client = initialize_clients()


# --- Helper Functions (extract_llm_prompt, extract_llm_completion) ---
# Keep these functions as provided in your original code
# ... (Insert your existing extract_llm_prompt function here) ...
def extract_llm_prompt(inputs: Optional[Dict[str, Any]], run_id: str) -> Optional[str]:
    """Extracts the full prompt string sent to the LLM from an LLM run's inputs."""
    if not inputs or not isinstance(inputs, dict) or "messages" not in inputs:
        logger.debug(f"Run {run_id}: 'messages' field missing or invalid in inputs.")
        # Try alternative common input structures
        if isinstance(inputs.get("input"), str): return inputs["input"]
        if isinstance(inputs.get("question"), str): return inputs["question"]
        if isinstance(inputs.get("prompt"), str): return inputs["prompt"]
        return None
    messages_data = inputs["messages"]
    # Handle different potential list structures
    if isinstance(messages_data, list) and len(messages_data) == 1 and isinstance(messages_data[0], list):
        messages = messages_data[0]
    elif isinstance(messages_data, list):
         messages = messages_data
    else:
        logger.warning(f"Run {run_id}: Unexpected structure for 'messages' in inputs: {type(messages_data)}")
        return None

    prompt_parts = []
    for msg in messages:
        content = None
        role = None
        if isinstance(msg, dict):
            # Standard OpenAI format
            content = msg.get("content")
            role = msg.get("role")
            # LangChain format variations
            if not content: content = msg.get("kwargs", {}).get("content")
            if not role: role = msg.get("kwargs", {}).get("role")
            # Heuristic based on ID if role is missing
            if not role and isinstance(msg.get("id"), list):
                 if any("SystemMessage" in str(part) for part in msg["id"]): role = "system"
                 elif any("HumanMessage" in str(part) for part in msg["id"]): role = "user"
                 elif any("AIMessage" in str(part) for part in msg["id"]): role = "assistant"
                 else: role = "unknown"

            if content:
                prompt_parts.append(f"<{role or 'message'}>\n{str(content)}\n</{role or 'message'}>")
        elif isinstance(msg, str): # Handle plain string messages if any
             prompt_parts.append(str(msg))

    if prompt_parts:
         full_prompt = "\n".join(prompt_parts)
         logger.debug(f"Run {run_id}: Extracted LLM prompt successfully.")
         return full_prompt
    else:
         logger.warning(f"Run {run_id}: Could not extract any message content from inputs: {json.dumps(inputs, default=str)[:500]}")
         logger.debug(f"Run {run_id}: Failing input structure: {json.dumps(inputs, default=str)}")
         return None

# ... (Insert your existing extract_llm_completion function here) ...
def extract_llm_completion(outputs: Optional[Dict[str, Any]], run_id: str) -> Optional[str]:
    """Extracts the LLM's generated completion from an LLM run's outputs."""
    if not outputs or not isinstance(outputs, dict):
        logger.debug(f"Run {run_id}: Outputs field missing or invalid.")
        return None

    # Prioritize standard output keys
    if isinstance(outputs.get("output"), str): return outputs["output"]
    if isinstance(outputs.get("answer"), str): return outputs["answer"]
    if isinstance(outputs.get("result"), str): return outputs["result"]

    # Check generations structure
    if "generations" not in outputs:
        logger.debug(f"Run {run_id}: 'generations' field missing in outputs.")
        return None

    generations = outputs["generations"]
    # Handle different list structures for generations
    if isinstance(generations, list) and len(generations) >= 1 and isinstance(generations[0], list) and len(generations[0]) >= 1:
        generation_data = generations[0][0] # Takes the first generation from the first list
    elif isinstance(generations, list) and len(generations) >= 1:
        generation_data = generations[0] # Takes the first generation element
    else:
        logger.warning(f"Run {run_id}: Unexpected structure or empty 'generations' in outputs: {type(generations)}")
        return None

    # Extract text from generation data dictionary
    if isinstance(generation_data, dict):
        # Prioritize 'text' field
        text_content = generation_data.get("text")
        if text_content is not None:
            logger.debug(f"Run {run_id}: Extracted LLM completion from 'text' field.")
            return str(text_content)

        # Check nested 'message' field (common in chat models)
        message_data = generation_data.get("message")
        if isinstance(message_data, dict):
             # Handle OpenAI and potential LangChain message formats
             content = message_data.get("content") or \
                       message_data.get("kwargs", {}).get("content")
             if content is not None:
                 logger.debug(f"Run {run_id}: Extracted LLM completion from nested 'message' field.")
                 return str(content)

    # Fallback if extraction failed
    logger.warning(f"Run {run_id}: Could not extract text/content from generation data: {json.dumps(generation_data, default=str)[:500]}")
    logger.debug(f"Run {run_id}: Failing output structure: {json.dumps(outputs, default=str)}")
    return None

# --- Fetch LLM Runs ---
# Keep this function unchanged
def fetch_runs_for_evaluation(project_name: str, limit: int) -> List[schemas.Run]:
    # ... (Insert your existing fetch_runs_for_evaluation function here) ...
    logger.info(f"Fetching up to {limit} LLM runs from project '{project_name}'...")
    try:
        # Use the globally initialized ls_client
        runs_iterator = ls_client.list_runs(
            project_name=project_name,
            run_type="llm", # Fetch only LLM runs for this evaluator
            error=False, # Exclude runs that errored out
            limit=limit,
            # Optional: Add start_time filter if needed
            # start_time=datetime.now() - timedelta(days=1)
        )
        runs = list(runs_iterator) # Convert iterator to list
        logger.info(f"Fetched {len(runs)} LLM runs.")
        if not runs:
             logger.warning(f"No LLM runs found in project '{project_name}' matching criteria.")
        return runs
    except Exception as e:
        logger.error(f"Error fetching runs from LangSmith project '{project_name}': {e}", exc_info=True)
        return []

# --- LLM Judge Logic ---
# Keep JudgeLLMRunResult class unchanged
class JudgeLLMRunResult(BaseModel):
    # ... (Insert your existing JudgeLLMRunResult class here) ...
    score: int = Field(..., description="The numeric score from 0 to 100.", ge=0, le=100)
    reasoning: str = Field(..., description="Detailed reasoning for the score based on the criteria.")

# Keep get_llm_run_judgement function unchanged (uses global judge_llm_client)
def get_llm_run_judgement(llm_prompt: str, llm_completion: str, run_id_for_log: str) -> Optional[Dict]:
    # ... (Insert your existing get_llm_run_judgement function here) ...
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
        # Use the globally initialized judge_llm_client
        completion = judge_llm_client.chat.completions.create(
            model=JUDGE_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator grading an LLM's response. Respond only in the requested JSON format."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} # Request JSON output if API supports it
        )
        response_content = completion.choices[0].message.content
        cleaned_content = response_content.strip()
        # No need to strip ```json anymore if response_format works

        try:
            parsed_data = json.loads(cleaned_content)
            # Validate using Pydantic
            validated_result = JudgeLLMRunResult(**parsed_data)
            logger.debug(f"LLM Judge successful for run {run_id_for_log}: Score={validated_result.score}")
            return validated_result.model_dump() # Return dict from validated model
        except (json.JSONDecodeError, TypeError, ValueError) as parse_error: # Catches Pydantic validation errors too
            logger.error(f"Failed to parse/validate LLM Judge response for run {run_id_for_log}. Error: {parse_error}. Response: '{cleaned_content}'", exc_info=False)
            return None
    except Exception as e:
        logger.error(f"Error during LLM Judge API call for run {run_id_for_log}: {e}", exc_info=True)
        return None

# --- Evaluation Loop ---
# Keep evaluate_runs_and_get_average function unchanged
def evaluate_runs_and_get_average(runs: List[schemas.Run], feedback_key: str) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    # ... (Insert your existing evaluate_runs_and_get_average function here) ...
    logger.info(f"Starting evaluation for {len(runs)} runs using create_feedback...")
    successful_evals = 0
    failed_evals = 0
    recorded_scores = [] # List to store scores that were successfully posted
    successful_judgements = [] # List to store judgement dicts for MLflow

    for i, run in enumerate(runs):
        run_id_str = str(run.id)
        logger.info(f"--- Processing run {i+1}/{len(runs)} (ID: {run_id_str}) ---")
        llm_prompt = extract_llm_prompt(run.inputs, run_id_str)
        llm_completion = extract_llm_completion(run.outputs, run_id_str)

        if llm_prompt is None or llm_completion is None:
            logger.warning(f"Skipping feedback for run {run_id_str}: Could not extract prompt/completion.")
            failed_evals += 1
            continue

        # Get judgement from the judge LLM
        judgement = get_llm_run_judgement(llm_prompt, llm_completion, run_id_str)

        if judgement and isinstance(judgement, dict) and "score" in judgement:
            current_score = judgement["score"] # Original 0-100 score
            try:
                # Post feedback to LangSmith (normalized score 0-1)
                ls_client.create_feedback(
                    run_id=run.id,
                    key=feedback_key,
                    score=current_score / 100.0,
                    comment=judgement.get("reasoning", "No reasoning provided."),
                    source="llm_judge" # Add source info
                )
                logger.info(f"Posted feedback to LangSmith for run {run_id_str} (Score: {current_score})")
                recorded_scores.append(current_score) # Add original score to list *after* successful post
                # Add the full judgement dict to the list for MLflow artifact
                successful_judgements.append({
                    "run_id": run_id_str,
                    "score": current_score,
                    "reasoning": judgement.get("reasoning", "No reasoning provided."),
                    # Optionally add truncated prompt/completion for context in artifact
                    "prompt_preview": llm_prompt[:300] + "..." if llm_prompt else None,
                    "completion_preview": llm_completion[:300] + "..." if llm_completion else None,
                })
                successful_evals += 1
            except Exception as feedback_err:
                logger.error(f"Failed to post feedback to LangSmith for run {run_id_str}: {feedback_err}", exc_info=True)
                failed_evals += 1 # Count as failed if feedback post fails
        else:
            logger.warning(f"Skipping feedback for run {run_id_str}: Judgement failed or returned invalid format.")
            failed_evals += 1

    logger.info(f"Finished evaluation loop. Successful judgements posted to LangSmith: {successful_evals}, Failures/Skipped: {failed_evals}")

    # Calculate average score
    average_score = None
    if recorded_scores:
        average_score = sum(recorded_scores) / len(recorded_scores)
        logger.info(f"Average Score ({feedback_key}) across {len(recorded_scores)} evaluated runs: {average_score:.2f}")
    else:
        logger.warning("No scores were successfully recorded. Cannot calculate average.")

    # Return both average score and the list of judgements
    return average_score, successful_judgements

# --- Main Execution Logic for Airflow ---
def evaluate_agent(feedback_key=FEEDBACK_KEY, langsmith_project_name=LANGSMITH_PROJECT_NAME, fetch_runs_limit=FETCH_RUNS_LIMIT):
    """
    Main function called by Airflow to evaluate agent runs.
    Handles MLflow logging if available.
    Returns average accuracy or raises Exception on critical failure.
    """
    logger.info(f"--- Starting Fitness Chatbot LLM Run Evaluation Script (Key: {feedback_key}) ---")

    # Initialize MLflow Tracker if imported
    mlflow_tracker = None
    if mlflow_imported and MLflowRAGTracker: # Check class exists
        try:
            # Ensure URI is valid
            if not MLFLOW_TRACKING_URI or not MLFLOW_TRACKING_URI.startswith(("http", "databricks", "file:")):
                 logger.error(f"Invalid MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}. Disabling MLflow.")
            else:
                 mlflow_tracker = MLflowRAGTracker(
                     experiment_name=AGENT_MLFLOW_EXPERIMENT_NAME,
                     tracking_uri=MLFLOW_TRACKING_URI
                 )
                 logger.info(f"MLflow Tracker initialized for experiment '{AGENT_MLFLOW_EXPERIMENT_NAME}' at {MLFLOW_TRACKING_URI}")
        except Exception as tracker_init_err:
            logger.error(f"Failed to initialize MLflow Tracker: {tracker_init_err}", exc_info=True)
            mlflow_tracker = None
    else:
         logger.warning("MLflow logging disabled (import failed or tracker class unavailable).")


    # 1. Fetch LLM Runs
    fetched_runs = fetch_runs_for_evaluation(langsmith_project_name, fetch_runs_limit)

    if not fetched_runs:
        logger.error("No LLM runs fetched from LangSmith. Cannot proceed.")
        # Log failure to MLflow if possible
        if mlflow_tracker:
             try:
                run_name = f"AgentEval_FAILED_FETCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with mlflow_tracker.start_run(run_name=run_name, tags={"evaluation_type": "agent", "status": "failed_fetch"}):
                    mlflow_tracker.log_parameters({
                        "langsmith_project": langsmith_project_name,
                        "fetch_limit": fetch_runs_limit,
                        "error": "No runs fetched from LangSmith"
                    })
             except Exception as log_err:
                 logger.error(f"Failed to log fetch failure to MLflow: {log_err}")
        # Raise exception to fail the Airflow task
        raise RuntimeError("No LLM runs fetched from LangSmith.")

    # 2. Run Evaluation and Get Average Score & Judgements
    average_accuracy, all_judgements = evaluate_runs_and_get_average(fetched_runs, feedback_key)

    # 3. Report Final Result (Console)
    if average_accuracy is not None:
         logger.info(f"  Overall Average LLM Run Quality Score: {average_accuracy:.2f} / 100")
    else:
         logger.info("  Could not calculate average score (no successful evaluations).")


    # 4. Log results to MLflow
    if mlflow_tracker:
        if average_accuracy is not None:
            mlflow_params = {
                "judge_llm_model": JUDGE_LLM_MODEL,
                "langsmith_project": langsmith_project_name,
                "num_runs_fetched": len(fetched_runs),
                "num_runs_evaluated": len(all_judgements),
                "fetch_limit": fetch_runs_limit,
                "langsmith_feedback_key": feedback_key,
                "evaluation_timestamp": datetime.now().isoformat()
            }
            mlflow_metrics = {
                "avg_llm_quality_score": average_accuracy
            }
            try:
                run_name = f"AgentEval_{feedback_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                # Use the dedicated logging method from the tracker
                mlflow_tracker.log_agent_evaluation_results(
                    run_name=run_name,
                    parameters=mlflow_params,
                    metrics=mlflow_metrics,
                    judgements=all_judgements # Pass the list of judgements for artifact logging
                )
                logger.info("Successfully logged agent evaluation results to MLflow.")
            except Exception as mlflow_err:
                logger.error(f"Failed to log agent evaluation results to MLflow: {mlflow_err}", exc_info=True)
                # Don't fail the whole task just because MLflow logging failed
        else:
            logger.warning("Skipping MLflow logging as no average score was calculated.")
            # Log a failed run to MLflow for traceability
            try:
                 run_name = f"AgentEval_FAILED_SCORE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                 with mlflow_tracker.start_run(run_name=run_name, tags={"evaluation_type": "agent", "status": "failed_score"}):
                     mlflow_tracker.log_parameters({
                         "langsmith_project": langsmith_project_name,
                         "fetch_limit": fetch_runs_limit,
                         "num_runs_fetched": len(fetched_runs),
                         "error": "No average score calculated"
                     })
            except Exception as log_err:
                 logger.error(f"Failed to log scoring failure to MLflow: {log_err}")


    logger.info("--- Fitness Chatbot LLM Run Evaluation Script Finished ---")
    logger.info(f"Check LangSmith project '{langsmith_project_name}' for individual run feedback ({feedback_key}).")
    logger.info(f"Check MLflow UI at '{MLFLOW_TRACKING_URI}' for aggregated results in experiment '{AGENT_MLFLOW_EXPERIMENT_NAME}'.")

    # If the process reached here without critical errors (like LangSmith fetch), consider it successful.
    # The average_accuracy being None is logged but doesn't necessarily mean the script failed.
    # Raise error only if average_accuracy is None AND len(all_judgements) == 0? Optional.
    # if average_accuracy is None and not all_judgements:
    #      raise RuntimeError("Agent evaluation completed but failed to generate any valid scores.")

    # No explicit return needed for Airflow success (absence of exception implies success)


# Make sure the script can be run directly if needed
if __name__ == "__main__":
    logger.info("Running agent evaluation script directly...")
    try:
        evaluate_agent()
        logger.info("Direct script execution finished successfully.")
    except Exception as e:
         logger.error(f"Direct script execution failed: {e}", exc_info=True)