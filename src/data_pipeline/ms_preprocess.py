# src/data_pipeline/ms_preprocess.py
import json
import unicodedata
import logging
import os
import re # Added for more robust cleaning

# Import GCS utility functions
try:
    from .gcs_utils import read_json_from_gcs, upload_string_to_gcs
except ImportError:
    # Fallback for potential direct execution testing
    from gcs_utils import read_json_from_gcs, upload_string_to_gcs

# ----------------------------------
# Logging Configuration
# ----------------------------------
# Keep existing logging setup - logs to local file & console
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/preprocessing.log")) # Using same log file
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized in ms_preprocess.py")


# ----------------------------------
# GCS Configuration
# ----------------------------------
RAW_BUCKET = "ragllm-454718-raw-data"
PROCESSED_BUCKET = "ragllm-454718-processed-data"

# Define input blob (relative path within RAW_BUCKET)
INPUT_BLOB_NAME = "raw_json_data/ms_data.json" # Correct filename from ms.py output

# Define output blob (relative path within PROCESSED_BUCKET)
OUTPUT_BLOB_NAME = "preprocessed_json_data/ms_data.json"


# ----------------------------------
# Helper Functions
# ----------------------------------
def clean_text(text):
    """
    Removes special characters, excessive spaces,
    and normalizes text to keep only alphanumeric characters,
    basic punctuation, and spaces. (Using consistent cleaning)
    """
    if not isinstance(text, str):
        logger.debug(f"Received non-string input for clean_text: {type(text)}. Returning empty string.")
        return ""
    # Keep only words, digits, spaces, and basic punctuation
    text = re.sub(r"[^\w\s.,!?'\"-]", "", text, flags=re.UNICODE)
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_summary(summary):
    """Formats the summary dictionary into a readable text format, excluding 'Workout PDF'."""
    summary_text = []
    # Ensure summary is a dictionary
    if not isinstance(summary, dict):
        logger.warning(f"Summary data is not a dictionary: {type(summary)}. Cannot format.")
        return ""

    for key, value in summary.items():
        # Ensure key and value are strings before processing
        key_str = str(key)
        value_str = str(value)
        if key_str != "Workout PDF": # Exclude "Workout PDF"
             # Clean the value before adding
            cleaned_value = clean_text(value_str)
            if cleaned_value: # Only add if value is not empty after cleaning
                 summary_text.append(f"{clean_text(key_str)}: {cleaned_value}") # Clean key too

    return "\n".join(summary_text) # Join all summary fields into text

def extract_relevant_workout_data(json_data):
    """
    Extracts relevant data from the M&S raw JSON structure, cleans text fields,
    and formats into a consistent output schema.
    """
    logger.info("Starting M&S workout data extraction and formatting...")

    # Expecting input like: {"metadata": {...}, "workouts": [ workout_dict1, ... ]}
    if not isinstance(json_data, dict) or "workouts" not in json_data:
        logger.error("Input JSON data is not in the expected format (missing 'workouts' list).")
        return None # Indicate failure
    if not isinstance(json_data["workouts"], list):
         logger.error("Input JSON 'workouts' field is not a list.")
         return None # Indicate failure

    workouts = json_data.get("workouts", []) # Extract workouts list
    logger.info(f"Found {len(workouts)} workouts in the input data.")
    extracted_data = []

    for index, workout in enumerate(workouts):
        # Ensure workout is a dictionary before proceeding
        if not isinstance(workout, dict):
            logger.warning(f"Skipping workout #{index+1} as it is not a dictionary ({type(workout)}).")
            continue

        try:
            # Extract required fields, cleaning the text ones
            # Use .get() with defaults for safety
            source = clean_text(workout.get("source_site", "muscleandstrength.com")) # Use source_site if available
            title = clean_text(workout.get("title", "No Title Provided"))
            url = workout.get("url", "No URL Provided")

            # Format summary (handles cleaning internally)
            summary_data = workout.get("summary", {})
            formatted_summary = format_summary(summary_data)

            # Extract and clean workout description
            workout_description = clean_text(workout.get("description", ""))

            # Extract, clean, and combine exercise descriptions
            exercises = workout.get("exercises", [])
            if not isinstance(exercises, list):
                 logger.warning(f"Exercises field is not a list for workout '{title}'. Skipping exercises.")
                 exercises = []

            exercise_texts = []
            for exercise in exercises:
                 if isinstance(exercise, dict):
                      # Clean exercise name and description
                      ex_name = clean_text(exercise.get("exercise", "Unknown Exercise"))
                      ex_desc = clean_text(exercise.get("description", ""))
                      if ex_desc and ex_desc != "No description available." and ex_desc != "No description available (fetch failed).": # Avoid default filler text
                           exercise_texts.append(f"Exercise: {ex_name}\n{ex_desc}")
                 else:
                      logger.warning(f"Skipping invalid exercise entry (not a dict) in workout '{title}'.")

            # Combine summary, main description, and exercise texts
            # Use double newlines to separate sections clearly
            combined_text = f"{formatted_summary}\n\n{workout_description}"
            if exercise_texts:
                combined_text += "\n\n" + "\n\n".join(exercise_texts)

            # Clean the final combined text just in case
            final_cleaned_text = clean_text(combined_text)

            # Append structured workout data only if we have meaningful text
            if final_cleaned_text:
                extracted_data.append({
                    "source": source, # Keep source domain
                    "title": title,
                    "url": url,
                    "text": final_cleaned_text # Use 'text' key for combined content
                })
                logger.info(f"Processed workout {index + 1}/{len(workouts)}: '{title}'")
            else:
                logger.warning(f"Skipped workout {index + 1}/{len(workouts)} ('{title}') due to empty text after processing.")

        except Exception as e:
            # Catch unexpected errors during processing of a single workout
            logger.error(f"Error processing workout {index + 1} ('{workout.get('title', 'N/A')}'): {e}", exc_info=True)
            # Continue to the next workout

    logger.info(f"Workout data extraction completed. Successfully processed {len(extracted_data)} workouts.")
    return extracted_data


# ----------------------------------
# Main Processing Function for Airflow
# ----------------------------------
def run_ms_preprocess_pipeline():
    """
    Reads the raw M&S JSON data from GCS, preprocesses it,
    and saves the cleaned/formatted data back to GCS.
    """
    logger.info("--- Starting Muscle&Strength Preprocessing Pipeline (GCS) ---")
    gcs_input_path = f"gs://{RAW_BUCKET}/{INPUT_BLOB_NAME}"
    gcs_output_path = f"gs://{PROCESSED_BUCKET}/{OUTPUT_BLOB_NAME}"

    logger.info(f"Reading input file: {gcs_input_path}")
    raw_json_data = read_json_from_gcs(RAW_BUCKET, INPUT_BLOB_NAME)

    if raw_json_data is None:
        logger.error(f"Failed to read raw M&S data from GCS at {gcs_input_path}. Cannot preprocess.")
        return False # Indicate failure

    processed_workouts = extract_relevant_workout_data(raw_json_data)

    if processed_workouts is None: # Check if extraction function indicated failure
         logger.error("M&S data extraction failed due to input format errors.")
         return False # Indicate failure
    if not processed_workouts:
        logger.warning("No processable workout data found after extraction. Saving empty list.")
        # Proceed to save an empty list if that's desired downstream

    # Save processed data to GCS Processed Bucket
    logger.info(f"Attempting to save {len(processed_workouts)} processed M&S workouts to {gcs_output_path}")
    try:
        json_output_string = json.dumps(processed_workouts, indent=4, ensure_ascii=False)
        save_success = upload_string_to_gcs(PROCESSED_BUCKET, OUTPUT_BLOB_NAME, json_output_string)

        if save_success:
            logger.info(f"Successfully saved processed M&S data to {gcs_output_path}")
            logger.info("--- Muscle&Strength Preprocessing Pipeline (GCS) finished successfully! ---")
            return True
        else:
            logger.error(f"Failed to save processed M&S data to {gcs_output_path}")
            logger.error("--- Muscle&Strength Preprocessing Pipeline (GCS) failed during saving phase. ---")
            return False
    except Exception as e:
        logger.error(f"Error serializing or saving processed M&S data: {e}", exc_info=True)
        logger.error("--- Muscle&Strength Preprocessing Pipeline (GCS) failed during saving phase. ---")
        return False

# Allow direct execution for testing
if __name__ == "__main__":
    logger.info("Running ms_preprocess.py directly for testing...")
    # Prerequisites for local testing:
    # 1. gcs_utils.py must be in the same directory or accessible via PYTHONPATH.
    # 2. Run 'gcloud auth application-default login' in your terminal.
    # 3. Ensure input file (ms_data.json) exists in gs://ragllm-454718-raw-data/raw_json_data/
    # 4. Ensure you have permissions to write to gs://ragllm-454718-processed-data/preprocessed_json_data/
    run_ms_preprocess_pipeline()