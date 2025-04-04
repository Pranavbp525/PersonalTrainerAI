from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from agent.agent_models import (TrainingPrinciples,
                    TrainingApproaches,
                    Citations,
                    BasicRoutine,
                    AdherenceRate,
                    ProgressMetrics,
                    IssuesList,
                    AdjustmentsList,
                    RoutineCreate,
                    RoutineExtract)
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from agent.agent_models import AgentState
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from rapidfuzz import process, fuzz
import json
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "fitness-chatbot"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Matches embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))


HEVY_EXERCISES_LIST: List[Dict[str, Any]] = []
HEVY_TITLES: List[str] = []
HEVY_TITLE_TO_TEMPLATE_MAP: Dict[str, Dict[str, Any]] = {}
EQUIPMENT_KEYWORDS = [
    "barbell", "dumbbell", "cable", "machine", "smith", "band",
    "kettlebell", "bodyweight", "weighted", "ez-bar", "trap bar",
    "plate", "landmine", "olympic", "hex bar", "sled", "ball", "trx",
    "assisted", "resistance", "foam roller", "bosu", "ring", "rope",
    "wheel", "bar", # Keep 'bar' last as it's less specific
]

def load_hevy_exercise_data(filepath: str = "hevy_exercises.json"):
    """Loads Hevy exercise data from a JSON file."""
    global HEVY_EXERCISES_LIST, HEVY_TITLES, HEVY_TITLE_TO_TEMPLATE_MAP
    try:
        # Check if data is already loaded
        if HEVY_EXERCISES_LIST:
            logger.info("Hevy exercise data already loaded.")
            return

        logger.info(f"Loading Hevy exercise data from {filepath}...")
        # Example: Loading from a JSON file
        with open(filepath, 'r') as f:
            HEVY_EXERCISES_LIST = json.load(f) # Assuming file contains a list of dicts

        # --- Alternative: Loading from DB ---
        # async def load_from_db():
        #     global HEVY_EXERCISES_LIST
        #     # Replace with your actual async DB query
        #     # async with get_db_session() as session:
        #     #     result = await session.execute(select(ExerciseTemplate))
        #     #     templates = result.scalars().all()
        #     #     HEVY_EXERCISES_LIST = [
        #     #         template.__dict__ for template in templates # Adjust conversion as needed
        #     #     ]
        #     pass # Placeholder
        # asyncio.run(load_from_db()) # Or integrate into your app's async startup

        # --- Post-processing for faster lookups ---
        # Ensure all required keys exist
        valid_exercises = []
        for exercise in HEVY_EXERCISES_LIST:
             if isinstance(exercise, dict) and "id" in exercise and "title" in exercise:
                  valid_exercises.append(exercise)
             else:
                  logger.warning(f"Skipping invalid exercise data format: {exercise}")
        HEVY_EXERCISES_LIST = valid_exercises
        
        HEVY_TITLES = [str(exercise['title']) for exercise in HEVY_EXERCISES_LIST if exercise.get('title')]
        # Create a map for quick retrieval after matching the title
        HEVY_TITLE_TO_TEMPLATE_MAP = {
            str(exercise['title']): exercise
            for exercise in HEVY_EXERCISES_LIST if exercise.get('title')
        }
        logger.info(f"Loaded {len(HEVY_TITLES)} Hevy exercise templates.")

    except FileNotFoundError:
        logger.error(f"Hevy exercise data file not found at {filepath}. Fuzzy matching will not work.")
        # Handle this critical error appropriately in your application
        HEVY_EXERCISES_LIST = []
        HEVY_TITLES = []
        HEVY_TITLE_TO_TEMPLATE_MAP = {}
    except Exception as e:
        logger.error(f"Error loading Hevy exercise data: {e}", exc_info=True)
        HEVY_EXERCISES_LIST = []
        HEVY_TITLES = []
        HEVY_TITLE_TO_TEMPLATE_MAP = {}

load_hevy_exercise_data()

def extract_principles(text: str) -> List[str]:
    """Extract scientific principles from text."""
    extraction_chain = llm.with_structured_output(TrainingPrinciples)
    
    result = extraction_chain.invoke(
        "Extract the scientific fitness principles from the following text. Return a list of specific principles mentioned:\n\n" + text
    )
    
    return result.principles

def extract_approaches(text: str) -> List[Dict]:
    """Extract training approaches from text."""
    extraction_chain = llm.with_structured_output(TrainingApproaches)
    
    result = extraction_chain.invoke(
        "Extract the training approaches with their names and descriptions from the following text:\n\n" + text
    )
    
    return [{"name": approach.name, "description": approach.description} 
            for approach in result.approaches]

def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text."""
    extraction_chain = llm.with_structured_output(Citations)
    
    result = extraction_chain.invoke(
        "Extract the citations and their sources from the following text. For each citation, identify the source (author, influencer, publication) and the main content or claim:\n\n" + text
    )
    
    return [{"source": citation.source, "content": citation.content} 
            for citation in result.citations]

def extract_routine_data(text: str) -> Dict:
    """Extract structured routine data for Hevy API."""
    extraction_chain = llm.with_structured_output(BasicRoutine)
    
    result = extraction_chain.invoke(
        "Extract the basic workout routine information from the following text. Identify the name, description, and list of workout days or sessions:\n\n" + text
    )
    
    return {
        "name": result.name,
        "description": result.description,
        "workouts": result.workouts
    }

def extract_adherence_rate(text: str) -> float:
    """Extract adherence rate from analysis text."""
    extraction_chain = llm.with_structured_output(AdherenceRate)
    
    result = extraction_chain.invoke(
        "Extract the workout adherence rate as a decimal between 0 and 1 from the following analysis text. This should represent the percentage of planned workouts that were completed:\n\n" + text
    )
    
    return result.rate

def extract_progress_metrics(text: str) -> Dict:
    """Extract progress metrics from analysis text."""
    extraction_chain = llm.with_structured_output(ProgressMetrics)
    
    result = extraction_chain.invoke(
        "Extract the progress metrics with their values from the following analysis text. Identify metrics like strength gains, endurance improvements, weight changes, etc. and their numeric values:\n\n" + text
    )
    
    return result.metrics

def extract_issues(text: str) -> List[str]:
    """Extract identified issues from analysis text."""
    extraction_chain = llm.with_structured_output(IssuesList)
    
    result = extraction_chain.invoke(
        "Extract the identified issues or problems from the following workout analysis text. List each distinct issue that needs attention:\n\n" + text
    )
    
    return result.issues

def extract_adjustments(text: str) -> List[Dict]:
    """Extract suggested adjustments from analysis text."""
    extraction_chain = llm.with_structured_output(AdjustmentsList)
    
    result = extraction_chain.invoke(
        "Extract the suggested workout adjustments from the following analysis text. For each adjustment, identify the target (exercise, schedule, etc.) and the specific change recommended:\n\n" + text
    )
    
    return [{"target": adj.target, "change": adj.change} 
            for adj in result.adjustments]


def extract_routine_structure(text: str) -> Dict:
    """Extract detailed routine structure from text for Hevy API."""
    extraction_chain = llm.with_structured_output(RoutineCreate)
    
    result = extraction_chain.invoke("""
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
    """)

    
    return result.model_dump()

def extract_routine_updates(text: str) -> Dict:
    """Extract routine updates from text for Hevy API."""
    extraction_chain = llm.with_structured_output(RoutineExtract)
    
    result = extraction_chain.invoke("""
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
    """)
    
    return {
        "title": result.title,
        "notes": result.notes,
        "exercises": [
            {
                "exercise_name": ex.exercise_name,
                "exercise_id": ex.exercise_id,
                "exercise_type": ex.exercise_type,
                "sets": [
                    {
                        "type": s.type, 
                        "weight": s.weight,
                        "reps": s.reps,
                        "duration_seconds": s.duration_seconds,
                        "distance_meters": s.distance_meters
                    } for s in ex.sets
                ],
                "notes": ex.notes
            } for ex in result.exercises
        ]
    }


async def retrieve_data(query: str) -> str:
    
    """Retrieves science-based exercise information from Pinecone vector store."""
    logger.info(f"RAG query: {query}")
    query_embedding = embeddings.embed_query(query)
    logger.info(f"Generated query embedding: {query_embedding[:5]}... (truncated)")
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    logger.info(f"Pinecone query results: {results}")
    retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
    logger.info(f"Retrieved documents: {retrieved_docs}")
    return "\n".join(retrieved_docs)



async def get_exercise_template_by_title_fuzzy(
    planner_exercise_name: str,
    threshold: int = 80 # Default similarity threshold (0-100)
) -> Optional[Dict[str, Any]]:
    """
    Finds the best matching Hevy exercise template for a given name.

    1. Preprocesses the input name.
    2. Attempts an exact (case-insensitive) match against loaded Hevy titles.
    3. If no exact match, performs fuzzy matching using rapidfuzz.
    4. Applies ambiguity checks to avoid incorrect matches for generic terms.
    5. Returns the matched template dictionary (containing at least 'id', 'title')
       or None if no suitable match is found.

    Args:
        planner_exercise_name: The exercise name provided by the planner.
        threshold: The minimum similarity score (0-100) for a fuzzy match.

    Returns:
        A dictionary with the matched Hevy exercise template info, or None.
    """
    if not planner_exercise_name:
        logger.warning("Received empty exercise name for lookup.")
        return None

    if not HEVY_TITLES or not HEVY_TITLE_TO_TEMPLATE_MAP:
        logger.error("Hevy exercise data not loaded. Cannot perform lookup.")
        # Depending on application requirements, you might raise an error here
        return None

    # --- 1. Preprocessing ---
    processed_name = planner_exercise_name.lower().strip()
    # Optional: Replace common variations like 'db' with 'dumbbell'?
    # processed_name = processed_name.replace(" db ", " dumbbell ") # Example

    logger.debug(f"Attempting lookup for processed name: '{processed_name}'")

    # --- 2. Exact Match (Case-Insensitive) ---
    for hevy_title in HEVY_TITLES:
        if hevy_title.lower() == processed_name:
            logger.info(f"Exact match found for '{planner_exercise_name}' -> '{hevy_title}'")
            # Return a copy to prevent modification of the original cache
            return HEVY_TITLE_TO_TEMPLATE_MAP[hevy_title].copy()

    logger.debug(f"No exact match for '{processed_name}'. Proceeding with fuzzy matching (threshold: {threshold})...")

    # --- 3. Fuzzy Match ---
    # Use WRatio for better handling of word order and subset strings
    # extractOne returns tuple: (best_match_string, score, index) or None
    result: Optional[Tuple[str, float, int]] = process.extractOne(
        processed_name,
        HEVY_TITLES,
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )

    if result:
        matched_title, score, _ = result
        logger.info(f"Fuzzy match candidate for '{planner_exercise_name}' -> '{matched_title}' (Score: {score:.2f})")

        # --- 4. Ambiguity Check ---
        # Is the *input* name generic? (Lacks specific equipment keywords)
        input_is_generic = not any(keyword in processed_name for keyword in EQUIPMENT_KEYWORDS)

        # If the input was generic AND we needed fuzzy matching (no exact match found),
        # be extra cautious.
        if input_is_generic:
            # Check if multiple DIFFERENT variations might also match closely.
            # This helps distinguish "Bench Press" (generic) from a slight typo
            # like "Bench Presss (Dumbell)" (specific, typo).
            potential_matches = process.extract(
                processed_name,
                HEVY_TITLES,
                scorer=fuzz.WRatio,
                limit=3, # Check top 3 matches
                score_cutoff=threshold * 0.9 # Slightly lower threshold to catch alternatives
            )

            # Filter out the exact best match we already found
            alternative_matches = [m for m in potential_matches if m[0] != matched_title and m[1] > threshold * 0.9] # Check if other variations also score high

            if len(alternative_matches) > 0:
                 # If other plausible variations exist, it's likely ambiguous
                 logger.warning(
                     f"Ambiguous fuzzy match for generic input '{planner_exercise_name}'. "
                     f"Best match '{matched_title}' ({score:.2f}), but alternatives exist "
                     f"(e.g., '{alternative_matches[0][0]}' at {alternative_matches[0][1]:.2f}). "
                     f"Rejecting match."
                 )
                 return None # Reject due to ambiguity

            # If input is generic, but no other variations score highly, it might be
            # a generic exercise that *exists* in Hevy (like 'Pull Up' or 'Dip').
            # We accept the match but log a note.
            logger.debug(f"Input '{planner_exercise_name}' looks generic, but no strong alternative matches found. Accepting match '{matched_title}'.")


        # If not ambiguous or input was specific, accept the match
        logger.info(f"Confirmed match for '{planner_exercise_name}' -> '{matched_title}'")
        return HEVY_TITLE_TO_TEMPLATE_MAP[matched_title].copy()

    else:
        logger.warning(f"No suitable match found for '{planner_exercise_name}' (Threshold: {threshold})")
        return None
