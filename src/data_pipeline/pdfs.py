# src/data_pipeline/pdfs.py
import os
import json
import PyPDF2  # Using PyPDF2 as per your original code
import unicodedata
import logging
import re

# Import GCS utility functions (adjust path if your structure differs slightly)
try:
    from .gcs_utils import list_gcs_files, download_blob_to_temp, upload_string_to_gcs, cleanup_temp_file
except ImportError:
    # Fallback for potential direct execution testing (though less ideal)
    from gcs_utils import list_gcs_files, download_blob_to_temp, upload_string_to_gcs, cleanup_temp_file


# Configure logging (keeping your original setup for now)
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/scraper.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # Also print logs to the console
    ]
)
logger = logging.getLogger(__name__)  # Logger specific to this file
logger.info("Logging initialized in pdfs.py")


# --- Define GCS paths ---
# Ensure these bucket names match the ones you created
SOURCE_BUCKET = "ragllm-454718-raw-data"
# Make sure your source PDFs are inside this folder in the SOURCE_BUCKET
SOURCE_PREFIX = "source_pdf/"
# Bucket where the raw extracted JSON will be saved
OUTPUT_BUCKET = "ragllm-454718-raw-data"
# Path within the OUTPUT_BUCKET for the JSON file
OUTPUT_BLOB_NAME = "raw_json_data/pdf_data.json"


def clean_text(text):
    """Normalize text encoding to remove special characters and excessive whitespace."""
    # Keeping your original cleaning function
    # You might want to enhance this further, e.g., handle ligatures, normalize unicode more aggressively
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace chars with a single space
    # Consider removing specific problematic characters if needed
    # text = re.sub(r'[●"“”]', '', text) # Example: remove specific bullets/quotes
    return text.strip()

def extract_text_from_local_pdf(pdf_path):
    """
    Extracts text from a single PDF file located at the given local path.
    Uses PyPDF2 as per the original script.
    """
    text = ""
    logger.debug(f"Extracting text from local path: {pdf_path}")
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            logger.debug(f"PDF has {num_pages} pages.")
            for i, page in enumerate(reader.pages):
                try:
                    extracted_page_text = page.extract_text()
                    if extracted_page_text:
                        text += extracted_page_text + " " # Add space between pages
                    else:
                        logger.warning(f"No text extracted from page {i+1} of {os.path.basename(pdf_path)}")
                except Exception as page_e:
                     logger.error(f"Error extracting text from page {i+1} of {os.path.basename(pdf_path)}: {page_e}")
        # Clean the combined text from all pages
        cleaned = clean_text(text)
        if not cleaned:
             logger.warning(f"No text content could be extracted or cleaned from: {os.path.basename(pdf_path)}")
             return None
        logger.info(f"Successfully extracted and cleaned text from: {os.path.basename(pdf_path)}")
        return cleaned
    except FileNotFoundError:
        logger.error(f"Temporary PDF file not found: {pdf_path}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error reading or processing local PDF {pdf_path}: {e}", exc_info=True)
        return None

def process_pdfs_from_gcs(limit=None):
    """
    Lists PDFs in GCS, downloads them temporarily, extracts text using PyPDF2,
    cleans the text, and returns a list of dictionaries.
    """
    logger.info(f"Starting PDF extraction process from GCS path gs://{SOURCE_BUCKET}/{SOURCE_PREFIX} (Limit: {limit})...")
    pdf_blobs = list_gcs_files(SOURCE_BUCKET, SOURCE_PREFIX)

    if not pdf_blobs:
        logger.warning(f"No files found in GCS at gs://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
        return []

    # Filter for actual PDF files (basic check), ignore zero-byte files and the prefix "folder" itself
    pdf_files_to_process = [
        b for b in pdf_blobs
        if b.name.lower().endswith(".pdf") and b.size > 0 and b.name != SOURCE_PREFIX
    ]
    logger.info(f"Found {len(pdf_files_to_process)} potential PDF files in GCS.")

    if limit is not None and limit > 0:
        pdf_files_to_process = pdf_files_to_process[:limit]
        logger.info(f"Processing limit applied: Now processing {len(pdf_files_to_process)} PDFs.")

    all_pdfs_data = []
    processed_count = 0
    for blob in pdf_files_to_process:
        gcs_source_path = f"gs://{SOURCE_BUCKET}/{blob.name}"
        logger.info(f"Processing GCS PDF: {gcs_source_path}")
        temp_pdf_path = None # Ensure variable exists for finally block
        try:
            # Download the PDF from GCS to a temporary local file
            temp_pdf_path = download_blob_to_temp(SOURCE_BUCKET, blob.name)

            if temp_pdf_path:
                # Extract text from the downloaded temporary file
                text_content = extract_text_from_local_pdf(temp_pdf_path)
                if text_content:
                    # Use original filename (without extension) as title
                    base_filename = os.path.basename(blob.name)
                    title = os.path.splitext(base_filename)[0]

                    all_pdfs_data.append({
                        "source": gcs_source_path, # Record GCS path as source
                        "title": title,
                        "url": "N/A", # PDF doesn't have a web URL in this context
                        "text": text_content # Use 'text' key for consistency
                    })
                    processed_count += 1
                else:
                    # Logged within extract_text_from_local_pdf if extraction failed
                    logger.warning(f"Skipping {gcs_source_path} due to lack of extracted text.")
            else:
                logger.error(f"Failed to download {gcs_source_path}, skipping.")

        except Exception as e:
            # Catch any unexpected errors during the processing of a single blob
            logger.error(f"Unexpected error processing GCS blob {gcs_source_path}: {e}", exc_info=True)
        finally:
            # IMPORTANT: Clean up the temporary file after processing or if an error occurred
            if temp_pdf_path:
                cleanup_temp_file(temp_pdf_path)

    logger.info(f"PDF extraction completed. Successfully processed text from {processed_count} out of {len(pdf_files_to_process)} targeted PDFs.")
    return all_pdfs_data

def save_data_to_gcs(data):
    """Saves the extracted PDF data list as JSON to the specified GCS location."""
    if not data:
        logger.warning("No extracted PDF data to save to GCS.")
        return True # Consider success if there was nothing to save

    logger.info(f"Attempting to save extracted PDF data to gs://{OUTPUT_BUCKET}/{OUTPUT_BLOB_NAME}")
    try:
        # Convert the list of dictionaries to a JSON string
        # Ensure ensure_ascii=False if your text might contain non-ASCII characters
        # that you want preserved as-is in the JSON.
        json_data_string = json.dumps(data, indent=4, ensure_ascii=False)

        # Upload the JSON string to GCS using the helper function
        success = upload_string_to_gcs(OUTPUT_BUCKET, OUTPUT_BLOB_NAME, json_data_string)

        if success:
             logger.info(f"Successfully saved PDF data JSON to gs://{OUTPUT_BUCKET}/{OUTPUT_BLOB_NAME}")
             return True
        else:
             # Error should have been logged within upload_string_to_gcs
             logger.error("Failed to save PDF data to GCS.")
             return False
    except Exception as e:
        # Catch errors during json.dumps or any other unexpected issue
        logger.error(f"Error during JSON serialization or GCS upload for PDF data: {e}", exc_info=True)
        return False

def run_pdf_pipeline(limit=None):
    """Main function orchestrating the PDF processing pipeline using GCS."""
    logger.info("--- Starting PDF Processing Pipeline (GCS) ---")
    # Renamed original pdf_scraper function

    # Process PDFs, getting data from GCS
    extracted_data = process_pdfs_from_gcs(limit=limit)

    # Save the extracted data (if any) to GCS
    save_success = save_data_to_gcs(extracted_data)

    if save_success:
        # If saving was successful OR if there was no data to save
        logger.info("--- PDF Processing Pipeline (GCS) finished successfully! ---")
        print("PDF extraction and saving completed successfully!") # Keep user-facing print
        # return True # No longer just return True
    else:
        # If saving failed
        logger.error("--- PDF Processing Pipeline (GCS) failed during saving phase. ---")
        # raise ValueError("Failed to save PDF data JSON to GCS.") # RAISE EXCEPTION
        return False # Keep return False for now, let's fix permissions first. If permissions fix works, change this to raise ValueError.

# This block allows running the script directly from command line for testing
if __name__ == "__main__":
    # Example: Run with a limit of 2 PDFs for local testing
    # Prerequisites for local testing:
    # 1. gcs_utils.py must be in the same directory or accessible via PYTHONPATH.
    # 2. Run 'gcloud auth application-default login' in your terminal.
    # 3. Ensure source PDFs exist at gs://ragllm-454718-raw-data/source_pdf/
    # 4. Ensure you have permissions to write to gs://ragllm-454718-raw-data/raw_json_data/
    logger.info("Running pdfs.py directly for testing...")
    run_pdf_pipeline(limit=None) # Process all PDFs found in GCS source path
    # run_pdf_pipeline(limit=2) # Or test with a limit