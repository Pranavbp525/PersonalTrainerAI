# src/data_pipeline/gcs_utils.py
from google.cloud import storage
import tempfile
import os
import json
import logging

# Configure logging (you can adjust the level and format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache the client to avoid re-initialization
_storage_client = None

def _get_client():
    """Initializes and returns a GCS client, reusing if possible."""
    global _storage_client
    if _storage_client is None:
        try:
            # Client automatically uses credentials from the environment when run on GCP
            # (VM service account, Cloud Run service account, Composer service account).
            # For local testing, it uses Application Default Credentials (ADC)
            # set up via 'gcloud auth application-default login'.
            _storage_client = storage.Client()
            logger.debug("Initialized GCS client.")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    return _storage_client

def list_gcs_files(bucket_name, prefix=None):
    """Lists files (blobs) in a GCS bucket, optionally filtering by prefix."""
    client = _get_client()
    logger.debug(f"Listing blobs in gs://{bucket_name}/{prefix or ''}")
    try:
        blobs = client.list_blobs(bucket_name, prefix=prefix)
        return list(blobs) # Return as a list
    except Exception as e:
        logger.error(f"Error listing blobs in gs://{bucket_name}/{prefix or ''}: {e}")
        return []

def download_blob_to_temp(bucket_name, source_blob_name):
    """Downloads a blob to a named temporary file and returns the path."""
    client = _get_client()
    logger.debug(f"Attempting to download gs://{bucket_name}/{source_blob_name}")
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        if not blob.exists():
             logger.warning(f"Blob not found: gs://{bucket_name}/{source_blob_name}")
             return None

        # Create a temporary file with a relevant suffix if possible
        suffix = os.path.splitext(source_blob_name)[1]
        # Using NamedTemporaryFile ensures it's cleaned up if script exits unexpectedly in some cases
        # but we need delete=False to keep it available after the 'with' block if not using 'with'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.close() # Close the file handle immediately after creation

        blob.download_to_filename(temp_file.name)
        logger.info(f"Blob {source_blob_name} downloaded to temporary file: {temp_file.name}")
        return temp_file.name # Return the path to the temp file

    except Exception as e:
        logger.error(f"Failed to download gs://{bucket_name}/{source_blob_name}: {e}")
        # Attempt cleanup if temp file was created before error
        if 'temp_file' in locals() and temp_file and os.path.exists(temp_file.name):
             os.remove(temp_file.name)
        return None

def upload_string_to_gcs(bucket_name, destination_blob_name, data_string, content_type='application/json'):
    """Uploads a string data to the specified GCS blob."""
    client = _get_client()
    logger.debug(f"Attempting to upload data to gs://{bucket_name}/{destination_blob_name}")
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(data_string, content_type=content_type)
        logger.info(f"String data successfully uploaded to gs://{bucket_name}/{destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to gs://{bucket_name}/{destination_blob_name}: {e}")
        return False

def read_json_from_gcs(bucket_name, source_blob_name):
    """Reads and parses a JSON file from GCS."""
    client = _get_client()
    logger.debug(f"Attempting to read JSON from gs://{bucket_name}/{source_blob_name}")
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        if not blob.exists():
             logger.warning(f"JSON file not found in GCS: gs://{bucket_name}/{source_blob_name}")
             return None

        json_data_string = blob.download_as_string()
        data = json.loads(json_data_string)
        logger.info(f"Successfully read and parsed JSON from gs://{bucket_name}/{source_blob_name}")
        return data

    except json.JSONDecodeError as e:
         logger.error(f"Error decoding JSON from gs://{bucket_name}/{source_blob_name}: {e}")
         return None
    except Exception as e:
        logger.error(f"Failed to read JSON from gs://{bucket_name}/{source_blob_name}: {e}")
        return None

def cleanup_temp_file(file_path):
    """Safely removes a temporary file if it exists."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")
        except OSError as e:
            logger.error(f"Error removing temporary file {file_path}: {e}")