# src/data_pipeline/other_preprocesing.py
import json
import re
import logging
import os

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
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/preprocessing.log")) # Changed filename
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
logger.info("Logging initialized in other_preprocessing.py")


# ----------------------------------
# GCS Configuration
# ----------------------------------
RAW_BUCKET = "ragllm-454718-raw-data"
PROCESSED_BUCKET = "ragllm-454718-processed-data"

# Define input blobs (relative paths within RAW_BUCKET)
INPUT_BLOB_PREFIX = "raw_json_data/"
INPUT_FILENAMES = ["blogs.json", "pdf_data.json"] # Files produced by previous steps

# Define output prefix (relative path within PROCESSED_BUCKET)
OUTPUT_BLOB_PREFIX = "preprocessed_json_data/"


# ----------------------------------
# Helper Functions
# ----------------------------------
def clean_text(text):
    """
    Removes special characters, excessive spaces,
    and normalizes text to keep only alphanumeric characters,
    basic punctuation, and spaces. (Keeping original logic)
    """
    if not isinstance(text, str):
        logger.warning(f"Received non-string input for clean_text: {type(text)}. Returning empty string.")
        return ""
    # Keep only words, digits, spaces, and basic punctuation
    # This might be too aggressive, consider adjusting regex if needed
    text = re.sub(r"[^\w\s.,!?'\"-]", "", text, flags=re.UNICODE) # Added common punctuation, ensure unicode flag
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------
# Main Processing Function for Airflow
# ----------------------------------
def run_other_preprocess_pipeline():
    """
    Reads specified raw JSON files from GCS raw bucket, cleans the 'text' field,
    and saves the processed files to GCS processed bucket.
    """
    logger.info("--- Starting Other JSON Preprocessing Pipeline (GCS) ---")
    overall_success = True # Track if any file fails

    for filename in INPUT_FILENAMES:
        input_blob_name = f"{INPUT_BLOB_PREFIX}{filename}"
        output_blob_name = f"{OUTPUT_BLOB_PREFIX}{filename}"
        gcs_input_path = f"gs://{RAW_BUCKET}/{input_blob_name}"
        gcs_output_path = f"gs://{PROCESSED_BUCKET}/{output_blob_name}"

        logger.info(f"Attempting to process: {gcs_input_path}")

        # Load JSON data from GCS Raw Bucket
        data = read_json_from_gcs(RAW_BUCKET, input_blob_name)

        # Skip if file does not exist in GCS or fails to load/parse
        if data is None:
            logger.warning(f"Skipping {filename}, failed to read or parse from {gcs_input_path}.")
            # Depending on requirements, this might be considered a failure
            # overall_success = False # Uncomment if missing input is critical
            continue

        # Expecting a list of dictionaries
        if not isinstance(data, list):
             logger.warning(f"Skipping {filename}, expected a list but got {type(data)}.")
             continue

        # Clean text in each entry
        entries_processed = 0
        entries_cleaned = 0
        cleaned_data = [] # Store results in a new list
        for entry in data:
            entries_processed += 1
            # *** IMPORTANT: Assuming the field to clean is 'text' ***
            # *** Change 'text' to 'description' if previous steps output that key ***
            if isinstance(entry, dict) and 'text' in entry:
                original_text = entry['text']
                cleaned = clean_text(original_text)
                if cleaned: # Only keep entry if cleaning resulted in non-empty text
                     entry['text'] = cleaned # Update the entry in-place (or create new dict)
                     cleaned_data.append(entry) # Add to the list of cleaned entries
                     entries_cleaned += 1
                else:
                     logger.debug(f"Entry skipped after cleaning resulted in empty text. Original source: {entry.get('source', 'N/A')}")
            else:
                 # Log if entry is not a dict or doesn't have the 'text' field
                 logger.warning(f"Skipping invalid entry #{entries_processed} in {filename}: Does not contain 'text' field or is not a dictionary.")

        if not cleaned_data:
            logger.warning(f"No valid entries found or remaining after cleaning in {filename}. Nothing will be saved for this file.")
            continue # Move to the next file

        # Save cleaned data to GCS Processed Bucket
        logger.info(f"Attempting to save {len(cleaned_data)} cleaned entries from {filename} to {gcs_output_path}")
        try:
            json_output_string = json.dumps(cleaned_data, indent=4, ensure_ascii=False)
            save_success = upload_string_to_gcs(PROCESSED_BUCKET, output_blob_name, json_output_string)
            if save_success:
                logger.info(f"Successfully processed and saved {filename}. Input entries: {len(data)}, Output entries: {len(cleaned_data)}.")
            else:
                logger.error(f"Failed to save cleaned data for {filename} to GCS.")
                overall_success = False # Mark pipeline as failed if any save fails
        except Exception as e:
            logger.error(f"Error serializing or saving cleaned data for {filename}: {e}", exc_info=True)
            overall_success = False # Mark pipeline as failed

    if overall_success:
        logger.info("--- Other JSON Preprocessing Pipeline (GCS) completed successfully! ---")
        print("Preprocessing completed.") # Keep original print statement
        return True
    else:
        logger.error("--- Other JSON Preprocessing Pipeline (GCS) completed with errors during saving. ---")
        return False

# Allow direct execution for testing
if __name__ == "__main__":
    logger.info("Running other_preprocessing.py directly for testing...")
    # Prerequisites for local testing:
    # 1. gcs_utils.py must be in the same directory or accessible via PYTHONPATH.
    # 2. Run 'gcloud auth application-default login' in your terminal.
    # 3. Ensure input files (e.g., blogs.json, pdf_data.json) exist in gs://ragllm-454718-raw-data/raw_json_data/
    # 4. Ensure you have permissions to write to gs://ragllm-454718-processed-data/preprocessed_json_data/
    run_other_preprocess_pipeline()