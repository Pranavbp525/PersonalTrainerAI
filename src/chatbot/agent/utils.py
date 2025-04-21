import time # Import time for timing
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from agent.agent_models import ( # Assuming these models are correctly defined
    TrainingPrinciples, TrainingApproaches, Citations, BasicRoutine,
    AdherenceRate, ProgressMetrics, IssuesList, AdjustmentsList,
    RoutineCreate, RoutineExtract
)
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from agent.agent_models import AgentState # Assuming used for context if needed
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from rapidfuzz import process, fuzz
import json
# import logging # Keep for type hinting if needed, remove basicConfig

# --- ELK Logging Import ---
try:
    # Assuming utils.py is in the same directory level as elk_logging.py
    # Adjust the path '..' if utils.py is inside a subdirectory like 'agent/'
    # from ..elk_logging import setup_elk_logging # If in subdirectory
    from ..elk_logging import setup_elk_logging # If at same level
except ImportError:
    # Fallback for different execution contexts
    print("Could not import elk_logging, using standard print for utils logs.")
    # Define a dummy logger class if elk_logging is unavailable
    class DummyLogger:
        def add_context(self, **kwargs): return self
        def info(self, msg, *args, **kwargs): print(f"INFO: {msg} | Context: {kwargs.get('extra')}")
        def warning(self, msg, *args, **kwargs): print(f"WARN: {msg} | Context: {kwargs.get('extra')}")
        def error(self, msg, *args, **kwargs): print(f"ERROR: {msg} | Context: {kwargs.get('extra')}")
        def debug(self, msg, *args, **kwargs): print(f"DEBUG: {msg} | Context: {kwargs.get('extra')}")
        def exception(self, msg, *args, **kwargs): print(f"EXCEPTION: {msg} | Context: {kwargs.get('extra')}")
    utils_log = DummyLogger()
else:
    utils_log = setup_elk_logging("fitness-chatbot.utils")
# --- End ELK Logging Import ---

# --- Remove Basic Logging Config ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__) # Remove module-level logger
# ---

load_dotenv()

# --- Initialization with Logging ---
try:
    utils_log.info("Initializing Pinecone in utils...")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "not_found")
    utils_log.info(f"IN UTILS Pinecode api : {pinecone_api_key}")
    if not pinecone_api_key:
        utils_log.error("IN UTILS PINECONE_API_KEY environment variable not set!")
        raise ValueError("IN UTILS PINECONE_API_KEY not found.")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "fitness-chatbot"

    if index_name not in pc.list_indexes().names():
        utils_log.info(f"In utils Pinecone index '{index_name}' not found, creating...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        utils_log.info(f"In utils Pinecone index '{index_name}' created successfully.")
    else:
        utils_log.info(f"In utils Found existing Pinecone index '{index_name}'.")

    index = pc.Index(index_name)
    utils_log.info("Pinecone initialized successfully in utils.")
except Exception as e:
    utils_log.error("Failed to initialize Pinecone in utils", exc_info=True)
    # Depending on severity, you might raise the exception or try to continue without Pinecone
    index = None # Ensure index is None if initialization failed


try:
    utils_log.info("Initializing HuggingFace embeddings model in utils...")
    embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    utils_log.info("Embeddings model initialized successfully in utils.", extra={"model_name": embeddings_model_name})
except Exception as e:
    utils_log.error("Failed to initialize embeddings model in utils", exc_info=True)
    embeddings = None

try:
    utils_log.info("Initializing ChatOpenAI model in utils...")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
         utils_log.error("OPENAI_API_KEY environment variable not set!")
         raise ValueError("OPENAI_API_KEY not found.")
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    utils_log.info("ChatOpenAI model initialized successfully in utils.", extra={"model_name": "gpt-4o"})
except Exception as e:
    utils_log.error("Failed to initialize ChatOpenAI model in utils", exc_info=True)
    llm = None # Ensure llm is None if failed
# --- End Initialization ---

# --- Global Variables for Hevy Data ---
HEVY_EXERCISES_LIST: List[Dict[str, Any]] = []
HEVY_TITLES: List[str] = []
HEVY_TITLE_TO_TEMPLATE_MAP: Dict[str, Dict[str, Any]] = {}
EQUIPMENT_KEYWORDS = [ # Keep original list
    "barbell", "dumbbell", "cable", "machine", "smith", "band",
    "kettlebell", "bodyweight", "weighted", "ez-bar", "trap bar",
    "plate", "landmine", "olympic", "hex bar", "sled", "ball", "trx",
    "assisted", "resistance", "foam roller", "bosu", "ring", "rope",
    "wheel", "bar",
]
# ---

def load_hevy_exercise_data(filepath: str = "hevy_exercises.json"):
    """Loads Hevy exercise data from a JSON file."""
    global HEVY_EXERCISES_LIST, HEVY_TITLES, HEVY_TITLE_TO_TEMPLATE_MAP
    log_ctx = {"function": "load_hevy_exercise_data", "filepath": filepath}
    try:
        if HEVY_EXERCISES_LIST:
            # logger.info("Hevy exercise data already loaded.") # Removed old logger
            utils_log.info("Hevy exercise data already loaded.", extra=log_ctx) # Added log
            return

        # logger.info(f"Loading Hevy exercise data from {filepath}...") # Removed old logger
        utils_log.info(f"Loading Hevy exercise data...", extra=log_ctx) # Added log
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Add basic type check
            if not isinstance(data, list):
                 raise TypeError("Expected JSON file to contain a list of exercises.")
            HEVY_EXERCISES_LIST = data

        valid_exercises = []
        invalid_count = 0
        for exercise in HEVY_EXERCISES_LIST:
             if isinstance(exercise, dict) and "id" in exercise and "title" in exercise:
                  valid_exercises.append(exercise)
             else:
                  # logger.warning(f"Skipping invalid exercise data format: {exercise}") # Removed old logger
                  utils_log.warning("Skipping invalid exercise data format during load.", extra={**log_ctx, "invalid_data_snippet": str(exercise)[:100]}) # Added log
                  invalid_count += 1
        HEVY_EXERCISES_LIST = valid_exercises

        # Ensure title is string before adding to list/map
        HEVY_TITLES = [str(exercise['title']) for exercise in HEVY_EXERCISES_LIST if exercise.get('title')]
        HEVY_TITLE_TO_TEMPLATE_MAP = {
            str(exercise['title']): exercise
            for exercise in HEVY_EXERCISES_LIST if exercise.get('title')
        }
        # logger.info(f"Loaded {len(HEVY_TITLES)} Hevy exercise templates.") # Removed old logger
        utils_log.info(f"Hevy exercise data loaded successfully.", extra={**log_ctx, "loaded_count": len(HEVY_TITLES), "invalid_skipped_count": invalid_count}) # Added log

    except FileNotFoundError:
        # logger.error(f"Hevy exercise data file not found at {filepath}. Fuzzy matching will not work.") # Removed old logger
        utils_log.error(f"Hevy exercise data file not found.", extra=log_ctx) # Added log
        HEVY_EXERCISES_LIST, HEVY_TITLES, HEVY_TITLE_TO_TEMPLATE_MAP = [], [], {}
    except (json.JSONDecodeError, TypeError) as e: # Catch specific load errors
        utils_log.error(f"Error decoding or processing Hevy exercise JSON data.", exc_info=True, extra=log_ctx) # Added log
        HEVY_EXERCISES_LIST, HEVY_TITLES, HEVY_TITLE_TO_TEMPLATE_MAP = [], [], {}
    except Exception as e:
        # logger.error(f"Error loading Hevy exercise data: {e}", exc_info=True) # Removed old logger
        utils_log.error(f"Unexpected error loading Hevy exercise data.", exc_info=True, extra=log_ctx) # Added log
        HEVY_EXERCISES_LIST, HEVY_TITLES, HEVY_TITLE_TO_TEMPLATE_MAP = [], [], {}

# --- Load data on module import ---
load_hevy_exercise_data()
# ---


# --- Extraction Functions with Logging ---
# Helper for structured output calls
async def _run_structured_extraction(llm_instance, output_model, input_text: str, prompt_prefix: str, tool_name: str):
    """Helper to run LLM structured output with logging and error handling."""
    if llm_instance is None:
        utils_log.error(f"LLM not initialized, cannot perform extraction for {tool_name}.")
        raise RuntimeError("LLM not available for extraction.")

    log_ctx = {"tool_name": tool_name, "output_model": output_model.__name__, "input_text_length": len(input_text)}
    utils_log.info(f"Executing structured extraction: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        extraction_chain = llm_instance.with_structured_output(output_model)
        prompt = prompt_prefix + "\n\n" + input_text
        # --- Original Logic: LLM Call ---
        result = await extraction_chain.ainvoke(prompt)
        # --- End Original Logic ---
        duration = time.time() - start_time
        utils_log.info(f"Structured extraction successful: {tool_name}", extra={**log_ctx, "duration_seconds": round(duration, 2)})
        return result
    except Exception as e:
        duration = time.time() - start_time
        utils_log.error(f"Error during structured extraction: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        # Re-raise the exception so the caller knows it failed
        raise e

# Modified extraction functions using the helper
async def extract_principles(text: str) -> List[str]:
    """Extract scientific principles from text."""
    result = await _run_structured_extraction(
        llm, TrainingPrinciples, text,
        "Extract the scientific fitness principles from the following text. Return a list of specific principles mentioned:",
        "extract_principles"
    )
    return result.principles

async def extract_approaches(text: str) -> List[Dict]:
    """Extract training approaches from text."""
    result = await _run_structured_extraction(
        llm, TrainingApproaches, text,
        "Extract the training approaches with their names and descriptions from the following text:",
        "extract_approaches"
    )
    return [{"name": approach.name, "description": approach.description} for approach in result.approaches]

async def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text."""
    result = await _run_structured_extraction(
        llm, Citations, text,
        "Extract the citations and their sources from the following text. For each citation, identify the source (author, influencer, publication) and the main content or claim:",
        "extract_citations"
    )
    return [{"source": citation.source, "content": citation.content} for citation in result.citations]

async def extract_routine_data(text: str) -> Dict:
    """Extract structured routine data for Hevy API."""
    result = await _run_structured_extraction(
        llm, BasicRoutine, text,
        "Extract the basic workout routine information from the following text. Identify the name, description, and list of workout days or sessions:",
        "extract_routine_data"
    )
    return {"name": result.name, "description": result.description, "workouts": result.workouts}

async def extract_adherence_rate(text: str) -> float:
    """Extract adherence rate from analysis text."""
    result = await _run_structured_extraction(
        llm, AdherenceRate, text,
        "Extract the workout adherence rate as a decimal between 0 and 1 from the following analysis text. This should represent the percentage of planned workouts that were completed:",
        "extract_adherence_rate"
    )
    return result.rate

async def extract_progress_metrics(text: str) -> Dict:
    """Extract progress metrics from analysis text."""
    result = await _run_structured_extraction(
        llm, ProgressMetrics, text,
        "Extract the progress metrics with their values from the following analysis text. Identify metrics like strength gains, endurance improvements, weight changes, etc. and their numeric values:",
        "extract_progress_metrics"
    )
    return result.metrics

async def extract_issues(text: str) -> List[str]:
    """Extract identified issues from analysis text."""
    result = await _run_structured_extraction(
        llm, IssuesList, text,
        "Extract the identified issues or problems from the following workout analysis text. List each distinct issue that needs attention:",
        "extract_issues"
    )
    return result.issues

async def extract_adjustments(text: str) -> List[Dict]:
    """Extract suggested adjustments from analysis text."""
    result = await _run_structured_extraction(
        llm, AdjustmentsList, text,
        "Extract the suggested workout adjustments from the following analysis text. For each adjustment, identify the target (exercise, schedule, etc.) and the specific change recommended:",
        "extract_adjustments"
    )
    return [{"target": adj.target, "change": adj.change} for adj in result.adjustments]

async def extract_routine_structure(text: str) -> Dict:
    """Extract detailed routine structure from text for Hevy API."""
    # Note: This uses a direct prompt format in the original code
    if llm is None:
        utils_log.error("LLM not initialized, cannot perform extraction for extract_routine_structure.")
        raise RuntimeError("LLM not available for extraction.")

    tool_name = "extract_routine_structure"
    log_ctx = {"tool_name": tool_name, "output_model": RoutineCreate.__name__, "input_text_length": len(text)}
    utils_log.info(f"Executing structured extraction: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        extraction_chain = llm.with_structured_output(RoutineCreate)
        prompt = """
    Extract a detailed workout routine structure from the following text, suitable for the Hevy API:

    """ + text + """

    Create a structured workout routine with:
    - A title for the routine
    - Overall notes or description
    - A list of exercises, each with:
      - Exercise name
      - Exercise ID (use a placeholder if not specified)
      - Exercise type (strength, cardio, etc)
      - Sets with reps, weight, and type
      - Any specific notes for the exercise
    """
        # --- Original Logic: LLM Call ---
        result = await extraction_chain.ainvoke(prompt)
        # --- End Original Logic ---
        duration = time.time() - start_time
        # Use model_dump for Pydantic V2
        result_dict = result.model_dump()
        utils_log.info(f"Structured extraction successful: {tool_name}", extra={**log_ctx, "duration_seconds": round(duration, 2)})
        return result_dict
    except Exception as e:
        duration = time.time() - start_time
        utils_log.error(f"Error during structured extraction: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise e

async def extract_routine_updates(text: str) -> Dict:
    """Extract routine updates from text for Hevy API."""
    # Note: This uses a direct prompt format and manual dict creation in the original code
    if llm is None:
        utils_log.error("LLM not initialized, cannot perform extraction for extract_routine_updates.")
        raise RuntimeError("LLM not available for extraction.")

    tool_name = "extract_routine_updates"
    log_ctx = {"tool_name": tool_name, "output_model": RoutineExtract.__name__, "input_text_length": len(text)}
    utils_log.info(f"Executing structured extraction: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        extraction_chain = llm.with_structured_output(RoutineExtract)
        prompt = """
    Extract updates to a workout routine from the following text, suitable for the Hevy API:

    """ + text + """

    Create a structured representation of the updated workout routine with:
    - The updated title for the routine
    - Updated overall notes
    - The list of exercises with their updated details, each with:
      - Exercise name
      - Exercise ID (use a placeholder if not specified)
      - Exercise type (strength, cardio, etc)
      - Updated sets with reps, weight, and type
      - Any updated notes for the exercise
    """
        # --- Original Logic: LLM Call ---
        result = await extraction_chain.ainvoke(prompt)
        # --- End Original Logic ---
        duration = time.time() - start_time
        utils_log.info(f"Structured extraction successful: {tool_name}", extra={**log_ctx, "duration_seconds": round(duration, 2)})

        # --- Original Logic: Manual Dict Creation ---
        # This manual creation is kept as it was in the original code
        return_dict = {
            "title": result.title, "notes": result.notes,
            "exercises": [
                {
                    "exercise_name": ex.exercise_name, "exercise_id": ex.exercise_id,
                    "exercise_type": ex.exercise_type, "notes": ex.notes,
                    "sets": [
                        {
                            "type": s.type, "weight": s.weight, "reps": s.reps,
                            "duration_seconds": s.duration_seconds, "distance_meters": s.distance_meters
                        } for s in ex.sets
                    ]
                } for ex in result.exercises
            ]
        }
        # --- End Original Logic ---
        return return_dict
    except Exception as e:
        duration = time.time() - start_time
        utils_log.error(f"Error during structured extraction: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise e
# --- End Extraction Functions ---


# --- RAG Function with Logging ---
async def retrieve_data(query: str) -> str:
    """Retrieves science-based exercise information from Pinecone vector store."""
    # This function is identical to retrieve_from_rag in llm_tools.py
    # Consolidate or ensure logging is consistent. Assuming llm_tools version is primary.
    # For now, adding logging here for completeness if this version is called directly.
    tool_name = "retrieve_data" # Use specific name
    log_ctx = {"tool_name": tool_name, "query": query}
    utils_log.info(f"Executing RAG retrieval: {tool_name}", extra=log_ctx)
    start_time = time.time()

    if embeddings is None or index is None:
        utils_log.error("Embeddings model or Pinecone index not initialized. Cannot perform RAG retrieval.", extra=log_ctx)
        return "Error: RAG components not available."

    try:
        # --- Original Logic ---
        # logger.info(f"RAG query: {query}") # Removed old logger
        utils_log.debug("Generating query embedding.", extra=log_ctx)
        embed_start = time.time()
        query_embedding = embeddings.embed_query(query)
        embed_duration = time.time() - embed_start
        # logger.info(f"Generated query embedding: {query_embedding[:5]}... (truncated)") # Removed old logger
        utils_log.debug("Query embedding generated.", extra={**log_ctx, "embedding_duration_seconds": round(embed_duration, 2)})

        utils_log.debug("Querying Pinecone index.", extra=log_ctx)
        query_start = time.time()
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        query_duration = time.time() - query_start
        num_matches = len(results.get("matches", []))
        # logger.info(f"Pinecone query results: {results}") # Removed old logger # Too verbose
        utils_log.debug("Pinecone query complete.", extra={**log_ctx, "query_duration_seconds": round(query_duration, 2), "matches_found": num_matches})

        retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
        # logger.info(f"Retrieved documents: {retrieved_docs}") # Removed old logger # Too verbose
        result_string = "\n".join(retrieved_docs) # Original just joined, no prefix needed if tool adds it
        # --- End Original Logic ---
        duration = time.time() - start_time
        utils_log.info(f"RAG retrieval successful: {tool_name}", extra={**log_ctx, "duration_seconds": round(duration, 2), "matches_retrieved": num_matches})
        return result_string
    except Exception as e:
        duration = time.time() - start_time
        utils_log.error(f"Error during RAG retrieval: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        return f"Error retrieving information: {str(e)}" # Return error string
# --- End RAG Function ---


# --- Fuzzy Matching Function with Logging ---
async def get_exercise_template_by_title_fuzzy(
    planner_exercise_name: str,
    threshold: int = 80
) -> Optional[Dict[str, Any]]:
    """Finds the best matching Hevy exercise template using fuzzy matching with logging."""
    func_name = "get_exercise_template_by_title_fuzzy"
    log_ctx = {"function": func_name, "input_name": planner_exercise_name, "threshold": threshold}
    utils_log.debug(f"Executing fuzzy match.", extra=log_ctx)
    start_time = time.time()

    if not planner_exercise_name:
        # logger.warning("Received empty exercise name for lookup.") # Removed old logger
        utils_log.warning("Received empty exercise name for lookup.", extra=log_ctx) # Added log
        return None

    if not HEVY_TITLES or not HEVY_TITLE_TO_TEMPLATE_MAP:
        # logger.error("Hevy exercise data not loaded. Cannot perform lookup.") # Removed old logger
        utils_log.error("Hevy exercise data not loaded. Cannot perform lookup.", extra=log_ctx) # Added log
        return None

    processed_name = planner_exercise_name.lower().strip()
    log_ctx["processed_name"] = processed_name # Add processed name to context
    # logger.debug(f"Attempting lookup for processed name: '{processed_name}'") # Removed old logger

    # Exact Match
    for hevy_title in HEVY_TITLES:
        if hevy_title.lower() == processed_name:
            # logger.info(f"Exact match found for '{planner_exercise_name}' -> '{hevy_title}'") # Removed old logger
            duration = time.time() - start_time
            utils_log.info(f"Exact match found.", extra={**log_ctx, "match_type": "exact", "matched_title": hevy_title, "duration_seconds": round(duration, 2)}) # Added log
            return HEVY_TITLE_TO_TEMPLATE_MAP[hevy_title].copy()

    # logger.debug(f"No exact match for '{processed_name}'. Proceeding with fuzzy matching (threshold: {threshold})...") # Removed old logger
    utils_log.debug("No exact match. Proceeding with fuzzy matching...", extra=log_ctx) # Added log

    # Fuzzy Match
    result: Optional[Tuple[str, float, int]] = process.extractOne(
        processed_name, HEVY_TITLES, scorer=fuzz.WRatio, score_cutoff=threshold
    )

    duration = time.time() - start_time # Calculate duration regardless of match outcome
    log_ctx_res = {**log_ctx, "duration_seconds": round(duration, 2)}

    if result:
        matched_title, score, _ = result
        log_ctx_res["fuzzy_match_candidate"] = matched_title
        log_ctx_res["fuzzy_score"] = round(score, 2)
        # logger.info(f"Fuzzy match candidate for '{planner_exercise_name}' -> '{matched_title}' (Score: {score:.2f})") # Removed old logger
        utils_log.debug("Fuzzy match candidate found.", extra=log_ctx_res) # Added log

        # Ambiguity Check
        input_is_generic = not any(keyword in processed_name for keyword in EQUIPMENT_KEYWORDS)
        log_ctx_res["input_is_generic"] = input_is_generic

        if input_is_generic:
            utils_log.debug("Input name is generic, performing ambiguity check.", extra=log_ctx_res) # Added log
            potential_matches = process.extract(
                processed_name, HEVY_TITLES, scorer=fuzz.WRatio,
                limit=3, score_cutoff=threshold * 0.9
            )
            alternative_matches = [m for m in potential_matches if m[0] != matched_title and m[1] > threshold * 0.9]
            log_ctx_res["alternative_matches_count"] = len(alternative_matches)

            if alternative_matches:
                 # logger.warning(...) # Removed old logger
                 utils_log.warning("Ambiguous fuzzy match for generic input. Rejecting match.", extra={**log_ctx_res, "alternative_example": alternative_matches[0][0]}) # Added log
                 return None
            else:
                # logger.debug(...) # Removed old logger
                 utils_log.debug("Input is generic, but no strong alternatives. Accepting match.", extra=log_ctx_res) # Added log

        # Accept match (either non-generic input or generic with no strong alternatives)
        # logger.info(f"Confirmed match for '{planner_exercise_name}' -> '{matched_title}'") # Removed old logger
        utils_log.info("Fuzzy match confirmed.", extra={**log_ctx_res, "match_type": "fuzzy", "matched_title": matched_title}) # Added log
        return HEVY_TITLE_TO_TEMPLATE_MAP[matched_title].copy()

    else:
        # logger.warning(f"No suitable match found for '{planner_exercise_name}' (Threshold: {threshold})") # Removed old logger
        utils_log.warning("No suitable fuzzy match found.", extra=log_ctx_res) # Added log
        return None
# --- End Fuzzy Matching ---


# --- Validation Function with Logging ---
async def validate_and_lookup_exercises(
    proposed_routine_json: Dict[str, Any],
    original_routine_title: str
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Validates exercises in a proposed routine JSON, performs lookups, with logging.
    Returns tuple: (corrected_routine_dict | None, list_of_validation_errors).
    Returns None for dict if validation encounters a fatal error for the routine.
    """
    func_name = "validate_and_lookup_exercises"
    log_ctx = {"function": func_name, "routine_title": original_routine_title}
    utils_log.info(f"Executing exercise validation/lookup.", extra=log_ctx)
    start_time = time.time()
    validation_errors = [] # Initialize local errors list

    if not proposed_routine_json or not isinstance(proposed_routine_json.get("exercises"), list):
        utils_log.error("Invalid routine structure received.", extra=log_ctx) # Added log
        return None, ["Invalid routine structure received from LLM."]

    if not HEVY_EXERCISES_LIST or not HEVY_TITLE_TO_TEMPLATE_MAP:
         error_msg = "Cannot validate exercises: Hevy exercise data not loaded."
         # logger.error(error_msg) # Removed old logger
         utils_log.error(error_msg, extra=log_ctx) # Added log
         return None, [error_msg]

    corrected_exercises = []
    original_exercises = proposed_routine_json.get("exercises", [])
    # logger.debug(f"Validating {len(original_exercises)} exercises proposed for routine '{original_routine_title}'.") # Removed old logger
    utils_log.debug(f"Validating proposed exercises.", extra={**log_ctx, "proposed_exercise_count": len(original_exercises)}) # Added log

    valid_template_ids = {ex['id'] for ex in HEVY_EXERCISES_LIST if isinstance(ex, dict) and 'id' in ex}
    id_to_official_title = {ex['id']: ex['title'] for ex in HEVY_EXERCISES_LIST if isinstance(ex, dict) and 'id' in ex and 'title' in ex}

    for index, proposed_ex in enumerate(original_exercises):
        ex_log_ctx = {**log_ctx, "exercise_index": index} # Context for this specific exercise
        if not isinstance(proposed_ex, dict):
             warn_msg = "Invalid format (not a dictionary)."
             # logger.warning(warn_msg) # Removed old logger
             utils_log.warning(f"Skipping item: {warn_msg}", extra=ex_log_ctx) # Added log
             validation_errors.append(f"Exercise {index}: {warn_msg}")
             continue

        llm_provided_id = proposed_ex.get("exercise_template_id")
        llm_provided_title = str(proposed_ex.get("title", "")).strip()
        ex_log_ctx["llm_provided_title"] = llm_provided_title # Add title to context
        ex_log_ctx["llm_provided_id"] = llm_provided_id

        # Case 1: ID Provided
        if llm_provided_id is not None and str(llm_provided_id).strip() != "":
            current_id_str = str(llm_provided_id).strip()
            ex_log_ctx["processed_id"] = current_id_str

            if current_id_str in valid_template_ids:
                # --- Valid ID ---
                official_title = id_to_official_title.get(current_id_str, llm_provided_title)
                ex_log_ctx["official_title"] = official_title
                if official_title != llm_provided_title:
                     # logger.debug(...) # Removed old logger
                     utils_log.debug("Correcting title based on valid ID.", extra=ex_log_ctx) # Added log
                     proposed_ex["title"] = official_title
                proposed_ex["exercise_template_id"] = current_id_str # Ensure it's set correctly
                corrected_exercises.append(proposed_ex)
                # logger.debug(...) # Removed old logger
                utils_log.debug("Validated using provided valid ID.", extra=ex_log_ctx) # Added log
            else:
                # --- Invalid ID ---
                warn_msg = "Provided exercise_template_id is invalid or not found."
                # logger.warning(warn_msg) # Removed old logger
                utils_log.warning(warn_msg, extra=ex_log_ctx) # Added log
                validation_errors.append(f"Exercise '{llm_provided_title}' ({index}): {warn_msg}")

                # Fallback Logic
                if llm_provided_title:
                    # logger.info(...) # Removed old logger
                    utils_log.info("Attempting fallback fuzzy match for title due to invalid ID.", extra=ex_log_ctx) # Added log
                    # --- Logging around fallback fuzzy match ---
                    fallback_start = time.time()
                    matched_template = await get_exercise_template_by_title_fuzzy(llm_provided_title) # Calls logged function
                    fallback_duration = time.time() - fallback_start
                    ex_log_ctx_fallback = {**ex_log_ctx, "fallback_match_duration": round(fallback_duration,2)}
                    # ---
                    if matched_template and matched_template.get("id"):
                        matched_id = matched_template["id"]
                        matched_title = matched_template["title"]
                        # logger.info(...) # Removed old logger
                        utils_log.info("Fallback successful: Matched to template via fuzzy.", extra={**ex_log_ctx_fallback, "matched_id": matched_id, "matched_title": matched_title}) # Added log
                        proposed_ex["exercise_template_id"] = matched_id
                        proposed_ex["title"] = matched_title
                        corrected_exercises.append(proposed_ex)
                        continue # Success, move to next exercise
                    else:
                        err_msg_detail = f"Fallback failed: Could not find fuzzy match for title '{llm_provided_title}'."
                        # logger.error(err_msg_detail) # Removed old logger
                        utils_log.error(f"Fallback failed: {err_msg_detail} Skipping exercise.", extra=ex_log_ctx_fallback) # Added log
                        validation_errors.append(f"Exercise '{llm_provided_title}' ({index}): {err_msg_detail}")
                        continue # Skip this exercise
                else:
                    err_msg_detail = "Cannot perform fallback lookup: No title provided."
                    # logger.error(err_msg_detail) # Removed old logger
                    utils_log.error(f"Fallback failed: {err_msg_detail} Skipping exercise.", extra=ex_log_ctx) # Added log
                    validation_errors.append(f"Exercise at index {index}: {err_msg_detail}")
                    continue # Skip this exercise

        # Case 2: ID is Null/Empty, Title Provided
        elif llm_provided_title:
            ex_log_ctx["lookup_reason"] = "ID was null/empty"
            # logger.debug(...) # Removed old logger
            utils_log.debug("Attempting fuzzy lookup as ID was null/empty.", extra=ex_log_ctx) # Added log
            # --- Logging around fuzzy match ---
            lookup_start = time.time()
            matched_template = await get_exercise_template_by_title_fuzzy(llm_provided_title) # Calls logged function
            lookup_duration = time.time() - lookup_start
            ex_log_ctx_lookup = {**ex_log_ctx, "lookup_duration": round(lookup_duration, 2)}
            # ---
            if matched_template and matched_template.get("id"):
                matched_id = matched_template["id"]
                matched_title = matched_template["title"]
                # logger.info(...) # Removed old logger
                utils_log.info("Fuzzy match successful for null/empty ID.", extra={**ex_log_ctx_lookup, "matched_id": matched_id, "matched_title": matched_title}) # Added log
                proposed_ex["exercise_template_id"] = matched_id
                proposed_ex["title"] = matched_title
                corrected_exercises.append(proposed_ex)
            else:
                err_msg = f"Failed to find a suitable fuzzy match in Hevy exercise data (ID was null/empty)."
                # logger.error(err_msg) # Removed old logger
                utils_log.error(f"Fuzzy match failed: {err_msg} Skipping exercise.", extra=ex_log_ctx_lookup) # Added log
                validation_errors.append(f"Exercise '{llm_provided_title}' ({index}): {err_msg}")
                continue # Skip this exercise

        # Case 3: ID is Null/Empty AND Title is Empty
        else:
             err_msg = "exercise_template_id is null/empty, and no title was provided."
             # logger.error(err_msg) # Removed old logger
             utils_log.error(f"Cannot process exercise: {err_msg} Skipping.", extra=ex_log_ctx) # Added log
             validation_errors.append(f"Exercise at index {index}: {err_msg}")
             continue # Skip this exercise

    # Final Routine Assembly
    validated_routine_dict = {**proposed_routine_json, "exercises": corrected_exercises}
    final_exercise_count = len(validated_routine_dict.get("exercises", []))

    # Final Check & Logging
    if final_exercise_count == 0 and original_exercises:
         error_msg = "Validation resulted in no valid exercises remaining, although exercises were proposed. Cannot update."
         # logger.error(error_msg) # Removed old logger
         utils_log.error(error_msg, extra=log_ctx) # Added log
         validation_errors.append(error_msg)
         duration = time.time() - start_time
         utils_log.info(f"Exiting validation with fatal error.", extra={**log_ctx, "duration_seconds": round(duration, 2)}) # Added log
         return None, validation_errors
    elif final_exercise_count == 0:
         # logger.info(...) # Removed old logger
         utils_log.info("Routine proposed to have no exercises after validation. This might be intentional.", extra=log_ctx) # Added log

    duration = time.time() - start_time
    # logger.info(...) # Removed old logger
    utils_log.info(f"Exercise validation/lookup complete.", extra={**log_ctx, "final_exercise_count": final_exercise_count, "issue_count": len(validation_errors), "duration_seconds": round(duration, 2)}) # Added log
    return validated_routine_dict, validation_errors
# --- End Validation Function ---