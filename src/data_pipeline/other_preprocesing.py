import os
import json
import re
import logging

# ----------------------------------
# Constants and Directories
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw_json_data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data"))
LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/preprocessing.log"))

# ----------------------------------
# Logging Configuration
# ----------------------------------
# Configure logging (Logs to console + file)
logger = logging.getLogger(__name__)  # Inherit global logger

if not logger.handlers:
    # Ensure logs are written to 'scraper.log' from pdfs.py
    file_handler = logging.FileHandler("preprocessing.log", mode='a')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

logger.info("Logging initialized in other_preprocessing.py")


# List of files to process
JSON_FILES = ["blogs.json", "pdf_data.json"]

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

# ----------------------------------
# Helper Functions
# ----------------------------------
def clean_text(text):
    """
    Removes special characters, excessive spaces, 
    and normalizes text to keep only alphanumeric characters, 
    basic punctuation, and spaces.
    """
    # Keep only words, digits, spaces, and basic punctuation
    text = re.sub(r"[^\w\s.,!?]", "", text)
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------
# Main Processing
# ----------------------------------

def preprocess_json_other_files():
    """Clean and preprocess text within specified JSON files."""

    logger.info("Starting JSON preprocessing...")
    for file_name in JSON_FILES:
        input_file_path = os.path.join(INPUT_DIR, file_name)
        output_file_path = os.path.join(OUTPUT_DIR, file_name)

        # Skip if file does not exist
        if not os.path.exists(input_file_path):
            logger.warning(f"Skipping {file_name}, file not found.")
            continue

        # Load JSON data
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {file_name}: {e}", exc_info=True)
            continue


        # Clean text in each entry
        entries_cleaned = 0
        for entry in data:
            if 'description' in entry:
                original_text = entry['description']
                entry['description'] = clean_text(original_text)
                entries_cleaned += 1

        # Save cleaned data to the new output directory
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Processed {entries_cleaned} entries in {file_name}. Saved cleaned data to {output_file_path}.")
        except Exception as e:
            logger.error(f"Error saving cleaned data for {file_name}: {e}", exc_info=True)

    logger.info("Preprocessing completed!")
    print("Preprocessing completed.")

