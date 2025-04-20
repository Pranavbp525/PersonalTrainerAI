# src/data_pipeline/vector_db.py
import os
import json
import logging
import time
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec, Index # Added Index type hint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import math

# Import GCS utility functions
try:
    from .gcs_utils import read_json_from_gcs
except ImportError:
    # Fallback for potential direct execution testing
    from gcs_utils import read_json_from_gcs


# Load environment variables (essential for PINECONE_API_KEY)
# Use the path expected inside the container based on docker-compose mount
dotenv_path = "/opt/airflow/app/.env"
if os.path.exists(dotenv_path):
    loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
    # Use logger only after basicConfig is set
    # logging.info(f"Attempted to load .env file from {dotenv_path}. Load successful: {loaded}")
else:
    # Use print or configure basic logging temporarily if needed before full setup
    print(f"WARNING: .env file not found at {dotenv_path}. Relying on environment variables already set.")


# Setup logger
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/vectordb.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Log .env loading result after logger is configured
logger.info(f"Attempted to load .env file from {dotenv_path}. Load successful: {loaded if 'loaded' in locals() else 'N/A (file not found)'}")
logger.info("Logging initialized in vector_db.py")

# --- Constants ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Ensure index name matches your .env or desired index
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot") # <<< USE CHATBOT INDEX
PINECONE_REGION = "us-east-1" # Or your specific region for ServerlessSpec
PINECONE_CLOUD = "aws" # Or your specific cloud for ServerlessSpec

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768 # Dimension for all-mpnet-base-v2
EMBEDDING_BATCH_SIZE = 32
PINECONE_UPSERT_BATCH_SIZE = 100

# GCS Configuration for reading PREPROCESSED data
PROCESSED_BUCKET = "ragllm-454718-processed-data"
INPUT_BLOB_PREFIX = "preprocessed_json_data/"
# Define the input files to process from GCS
INPUT_FILES_TO_PROCESS = {
    "pdf": f"{INPUT_BLOB_PREFIX}pdf_data.json",
    "ms": f"{INPUT_BLOB_PREFIX}ms_data.json",
    "articles": f"{INPUT_BLOB_PREFIX}blogs.json"
}

# **Step 1: Load JSON Data - Removed old local function, integrated GCS read below **

# **Step 2: Split Text into Chunks**
def split_text(data, source_type, chunk_size=400, chunk_overlap=50):
    """Splits text from loaded data into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " "], # Added newline separator
        length_function=len, # Use character length
        is_separator_regex=False,
    )
    chunked_data = []
    if not isinstance(data, list):
        logger.error(f"Expected a list of items for splitting (source: {source_type}), got {type(data)}")
        return chunked_data

    logger.info(f"Splitting text for {len(data)} items from source: {source_type}")
    items_skipped_no_text = 0
    items_skipped_not_dict = 0
    for item_index, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dictionary item #{item_index+1} in source {source_type}: {item}")
            items_skipped_not_dict +=1
            continue

        # *** Reading from 'text' field (ensure preprocessing output this key) ***
        text_to_split = item.get("text", "")
        # Extract other metadata fields (use defaults if missing)
        original_gcs_source = item.get("source", f"gs://{PROCESSED_BUCKET}/{source_type}_{item_index}") # Use GCS path if available
        title = item.get("title", "N/A")
        url = item.get("url", "N/A")

        if not text_to_split:
            logger.debug(f"Skipping item #{item_index+1} with empty text field: source={original_gcs_source}, title={title}")
            items_skipped_no_text += 1
            continue

        try:
            # Split the text from the 'text' field
            chunks = text_splitter.split_text(text_to_split)
            # Create a base ID using the source type and original item index
            base_id = f"{source_type}_{item_index}"

            for chunk_idx, chunk_text in enumerate(chunks):
                # Create a unique ID for each chunk
                chunk_id = f"{base_id}_chunk_{chunk_idx}"
                # Prepare chunk dictionary in the structure needed later
                chunked_data.append({
                    "id": chunk_id, # Use 'id' for the Pinecone vector ID
                    "text": chunk_text, # The actual text content of the chunk
                    # Metadata to be stored with the vector in Pinecone
                    "metadata": {
                         "original_source": original_gcs_source, # GCS path or original URL
                         "title": title,
                         "url": url,
                         "data_source_type": source_type, # e.g., 'pdf', 'ms', 'articles'
                         "chunk_index": chunk_idx,
                         "text": chunk_text # Also store full text in metadata for retrieval display
                     }
                })
        except Exception as e:
            logger.error(f"Error splitting text for item #{item_index+1} (source={original_gcs_source}, title={title}): {e}", exc_info=True)
            continue # Skip item on splitting error

    if items_skipped_no_text > 0: logger.warning(f"Skipped {items_skipped_no_text} items from {source_type} due to empty text field.")
    if items_skipped_not_dict > 0: logger.warning(f"Skipped {items_skipped_not_dict} items from {source_type} because they were not dictionaries.")
    logger.info(f"Generated {len(chunked_data)} chunks from source: {source_type}")
    return chunked_data


# **Step 3: Generate Embeddings (BATCHED)**
def generate_embeddings_batched(chunked_data, model_name=EMBEDDING_MODEL_NAME, batch_size=EMBEDDING_BATCH_SIZE):
    """Generates embeddings for text chunks IN BATCHES."""
    if not chunked_data:
        logger.warning("No chunked data provided for embedding generation.")
        return [], None # Return empty list and no model

    logger.info(f"Initializing embedding model: {model_name}")
    try:
        # Ensure device selection is appropriate for the execution environment (CPU likely on VM)
        # Consider adding cache_folder='/opt/airflow/embeddings_cache' or similar if space allows
        model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        logger.info("Embedding model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name}: {e}", exc_info=True)
        return [], None # Return empty list and no model if init fails

    # Extract the text content for embedding
    texts_to_embed = [item.get("text", "") for item in chunked_data]
    all_embeddings = []
    total_batches = math.ceil(len(texts_to_embed) / batch_size)

    logger.info(f"Generating embeddings for {len(texts_to_embed)} text chunks in {total_batches} batches of size {batch_size}...")

    try:
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Embeddings", total=total_batches):
            batch_texts = texts_to_embed[i:i + batch_size]
            # Filter out potential empty strings just in case, although split_text should avoid this
            batch_texts_non_empty = [text for text in batch_texts if isinstance(text, str) and text.strip()]

            if not batch_texts_non_empty:
                logger.warning(f"Batch {i//batch_size + 1} contained only empty strings or non-strings. Assigning None embeddings.")
                # Extend with None for the length of the original batch slice
                all_embeddings.extend([None] * len(batch_texts))
                continue

            # Generate embeddings for the valid texts in the batch
            batch_embeddings = model.embed_documents(batch_texts_non_empty)

            # Map results back to the original batch structure, including Nones for empty inputs
            embedding_iter = iter(batch_embeddings)
            batch_results = []
            for text in batch_texts:
                if isinstance(text, str) and text.strip():
                     try:
                          # Get the next embedding if the text was valid
                          batch_results.append(next(embedding_iter))
                     except StopIteration:
                          logger.error("Embedding model returned fewer results than expected for batch. Assigning None.")
                          batch_results.append(None) # Append None if iterator exhausted prematurely
                else:
                     # Append None for empty/invalid original text
                     batch_results.append(None)
            all_embeddings.extend(batch_results)

    except Exception as e:
        logger.error(f"Error during batch embedding generation: {e}", exc_info=True)
        # Fail completely if a batch fails mid-way for simplicity
        return [], model # Return empty data but keep model reference if needed

    logger.info(f"Finished embedding generation. Received {len(all_embeddings)} results (including Nones).")

    # Add embeddings back to the chunked data
    chunks_with_embeddings = []
    failed_count = 0
    if len(all_embeddings) != len(chunked_data):
         logger.error(f"CRITICAL: Mismatch between number of chunks ({len(chunked_data)}) and embedding results ({len(all_embeddings)}). Aborting combination.")
         # Consider returning empty list vs raising error based on desired behavior
         return [], model # Abort if lengths mismatch

    for i, chunk_item in enumerate(chunked_data):
        embedding_result = all_embeddings[i]
        # Check if embedding is a list and has the correct dimension
        if embedding_result is not None and isinstance(embedding_result, list) and len(embedding_result) == EMBEDDING_DIMENSION:
            chunk_item["embedding"] = embedding_result
            chunks_with_embeddings.append(chunk_item)
        else:
            # Log detailed warning if embedding is invalid
            embedding_info = f"type={type(embedding_result)}"
            if isinstance(embedding_result, list):
                 embedding_info += f", len={len(embedding_result)}"
            logger.warning(f"Embedding missing, None, or wrong dimension ({embedding_info}) for chunk ID {chunk_item.get('id')}. Skipping this chunk.")
            failed_count += 1

    if failed_count > 0:
         logger.warning(f"Total chunks skipped due to failed/invalid embeddings: {failed_count}")

    logger.info(f"Returning {len(chunks_with_embeddings)} chunks with valid embeddings.")
    return chunks_with_embeddings, model


# **Step 4: Store Chunks in Pinecone**
def store_in_pinecone(chunks_with_embeddings, pc_client: Pinecone, INDEX_NAME: str):
    """Stores chunked data with embeddings in the specified Pinecone index."""
    logger.info(f"Preparing to store {len(chunks_with_embeddings)} chunks with embeddings in Pinecone index '{INDEX_NAME}'...")
    if not chunks_with_embeddings:
        logger.warning("No valid chunks with embeddings provided for storage.")
        return True # Nothing to store is not a failure

    try:
        # --- Check/Connect to Index ---
        logger.info(f"Listing existing Pinecone indexes...")
        # --- !!! CORRECTED FIX APPLIED HERE !!! ---
        index_list_obj = pc_client.list_indexes() # Get the IndexList object first
        existing_index_names = index_list_obj.names # Access the 'names' attribute which IS the list
        # --- !!! END FIX !!! ---

        logger.info(f"Existing Pinecone index names found: {existing_index_names}") # Log the actual list now

        # Check if the target index exists in the retrieved list
        if INDEX_NAME not in existing_index_names:
            logger.error(f"Target Pinecone index '{INDEX_NAME}' does not exist in the list: {existing_index_names}! Please create it first.")
            return False # Fail if index doesn't exist

        logger.info(f"Connecting to Pinecone index '{INDEX_NAME}'...")
        index: Index = pc_client.Index(INDEX_NAME)

        # Verify connection by getting stats (Optional but good practice)
        try:
             stats = index.describe_index_stats()
             logger.info(f"Connected successfully. Index stats before upsert: {stats}")
        except Exception as desc_err:
             # Log warning but don't necessarily fail if stats check fails, proceed to upsert cautiously
             logger.warning(f"Could not get stats for index '{INDEX_NAME}'. Proceeding with upsert attempt... Error: {desc_err}")

        # --- Prepare vectors for upsert ---
        vectors_to_upsert = []
        skipped_chunks = 0
        for chunk in chunks_with_embeddings:
            # Validate required fields for upsert
            vector_id = chunk.get("id")
            embedding = chunk.get("embedding")
            metadata = chunk.get("metadata")

            # Check types robustly
            if isinstance(vector_id, str) and \
               isinstance(embedding, list) and len(embedding) == EMBEDDING_DIMENSION and \
               isinstance(metadata, dict):

                # Clean metadata: Keep only types Pinecone supports (str, bool, numbers, list of str)
                cleaned_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, bool, int, float)):
                        # Ensure strings are not excessively long if Pinecone has limits
                        if isinstance(v, str) and len(v.encode('utf-8')) > 40000: # Example limit, check Pinecone docs
                             logger.warning(f"Truncating long metadata string key '{k}' for vector '{vector_id}'")
                             cleaned_metadata[k] = v[:10000] + "... (truncated)" # Truncate long strings
                        else:
                             cleaned_metadata[k] = v
                    elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                         # Check list length and individual string lengths if needed
                         cleaned_metadata[k] = v # List of strings is okay
                    else:
                         # Convert other types to string as fallback, log warning
                         logger.warning(f"Converting unsupported metadata type ({type(v)}) to string for key '{k}' in vector '{vector_id}'")
                         cleaned_metadata[k] = str(v)

                vectors_to_upsert.append(
                    # Using Pinecone tuple format: (id, values, metadata)
                    (vector_id, embedding, cleaned_metadata)
                )
            else:
                logger.warning(f"Skipping invalid chunk during vector preparation (missing id, embedding, metadata, or wrong format): ID={vector_id}")
                skipped_chunks += 1

        if skipped_chunks > 0: logger.warning(f"Skipped {skipped_chunks} invalid chunks during vector preparation.")

        if not vectors_to_upsert:
            logger.warning("No valid vectors prepared for upserting after validation.")
            # Changed from error to warning, still return True as no operation failed
            return True # Return True as no storage operation needed/failed

        # --- Upsert in Batches ---
        total_vectors = len(vectors_to_upsert)
        batch_size = PINECONE_UPSERT_BATCH_SIZE
        num_batches = math.ceil(total_vectors / batch_size)
        logger.info(f"Upserting {total_vectors} vectors to index '{INDEX_NAME}' in {num_batches} batches of size {batch_size}...")
        total_upserted_count = 0
        storage_had_errors = False

        for i in tqdm(range(0, total_vectors, batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i:i + batch_size]
            if not batch: # Should not happen if vectors_to_upsert is not empty, but safety check
                logger.warning(f"Skipping empty batch at index {i}")
                continue
            try:
                upsert_response = index.upsert(vectors=batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}/{num_batches}. Response: {upsert_response}")
                # Check response for success indicators if available
                if upsert_response and hasattr(upsert_response, 'upserted_count') and upsert_response.upserted_count is not None:
                     total_upserted_count += upsert_response.upserted_count
                     if upsert_response.upserted_count != len(batch):
                          logger.warning(f"Batch {i//batch_size + 1} upsert count ({upsert_response.upserted_count}) differs from batch size ({len(batch)}).")
                else:
                     # If upserted_count is missing or None, we can't confirm success precisely
                     logger.warning(f"Pinecone upsert response for batch {i//batch_size + 1} did not report expected upserted count. Assuming batch size ({len(batch)}) were attempted.")
                     total_upserted_count += len(batch) # Assume attempted count

            except Exception as upsert_err:
                logger.error(f"Error upserting batch starting at index {i}: {upsert_err}", exc_info=True)
                storage_had_errors = True
                # Optional: Add logic here to collect failed IDs from 'batch' for retry

        logger.info(f"Finished upsert loop. Total reported/attempted upsert count by Pinecone: {total_upserted_count} (Target: {total_vectors})")
        # Log final index stats
        try:
            final_stats = index.describe_index_stats()
            logger.info(f"Final index stats after upsert: {final_stats}")
        except Exception as final_stat_err:
             logger.warning(f"Could not get final index stats: {final_stat_err}")

        # Return True if no errors occurred during upserting, False otherwise
        return not storage_had_errors

    except Exception as e:
        # Catch errors during index check or connection
        logger.error(f"A critical error occurred during the Pinecone storage process (e.g., connection, index check): {e}", exc_info=True)
        return False # Indicate overall storage failure


# **Step 5: Query Pinecone**
def query_pinecone(query: str, model, pc_client: Pinecone, INDEX_NAME: str):
    """Queries the Pinecone index with the given text query."""
    if not model:
        logger.error("Cannot query: Embedding model is not available.")
        return None
    if not pc_client:
        logger.error("Cannot query: Pinecone client is not available.")
        return None
    if not query:
         logger.warning("Cannot query: Empty query string provided.")
         return None

    logger.info(f"Attempting to query Pinecone index '{INDEX_NAME}' with: '{query}'")
    try:
        logger.debug("Generating query embedding...")
        query_embedding = model.embed_query(query)
        if not query_embedding:
             logger.error("Failed to generate query embedding.")
             return None
        logger.debug("Query embedding generated.")
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        return None

    try:
        logger.debug(f"Connecting to index '{INDEX_NAME}' for query...")
        index: Index = pc_client.Index(INDEX_NAME) # Get index object
        logger.debug(f"Querying index with top_k=3...")

        # Query the index
        result = index.query(
            vector=query_embedding,
            top_k=3, # Number of results to return
            include_metadata=True # Get the metadata we stored
            )
        logger.info("Pinecone query successful.")
        logger.info("\n--- Relevant Context Chunks Found ---")
        matches = result.get('matches', [])
        if not matches:
             logger.info("No relevant matches found in Pinecone.")
        else:
            for i, match in enumerate(matches):
                metadata = match.get('metadata', {})
                # Use get() with defaults for safer access
                text_chunk = metadata.get('text', 'N/A') # Get stored text chunk
                title = metadata.get('title', 'N/A')
                original_source = metadata.get('original_source', 'N/A')
                score = match.get('score', 'N/A')
                chunk_id = match.get('id', 'N/A')

                # Log retrieved information
                logger.info(f"Match {i+1}:")
                logger.info(f"  Score: {score:.4f}")
                logger.info(f"  ID: {chunk_id}")
                logger.info(f"  Source: {original_source}")
                logger.info(f"  Title: {title}")
                logger.info(f"  Text Chunk: {text_chunk[:200]}...") # Display preview
                logger.info("---")
        return result # Return the full result object

    except Exception as e:
        logger.error(f"Error querying Pinecone index '{INDEX_NAME}': {e}", exc_info=True)
        return None


# **Step 6: Main Execution Function for Airflow**
def run_chunk_embed_store_pipeline(**kwargs): # Add **kwargs to accept Airflow context
    """
    Main function orchestrating loading from GCS, chunking, embedding,
    and storage in Pinecone. Called by the Airflow DAG task.
    Raises exceptions on critical failure.
    """
    logger = logging.getLogger("airflow.task") # Use Airflow's task logger
    logger.info("--- Starting Chunk, Embed, Store Pipeline (GCS -> Pinecone) ---")
    start_time = time.time()

    # --- Initialize Pinecone Client ---
    if not PINECONE_API_KEY:
        logger.error("CRITICAL: PINECONE_API_KEY environment variable not found.")
        raise ValueError("PINECONE_API_KEY is required but not set.")
    try:
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc.list_indexes() # Simple check to confirm connection/auth
        logger.info("Pinecone client initialized and connection confirmed.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Pinecone client or confirm connection: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Pinecone: {e}") from e

    # --- Phase 1: Load and Chunk All Data from GCS ---
    logger.info("--- Phase 1: Loading and Chunking Data from GCS ---")
    load_chunk_start = time.time()
    all_chunks_to_process = []

    for source_type, input_blob_name in INPUT_FILES_TO_PROCESS.items():
        gcs_input_path = f"gs://{PROCESSED_BUCKET}/{input_blob_name}"
        logger.info(f"Processing source: {source_type} | File: {gcs_input_path}")
        data = read_json_from_gcs(PROCESSED_BUCKET, input_blob_name) # Use helper
        if data is not None and isinstance(data, list) and data:
            chunked_data = split_text(data, source_type=source_type) # Use helper
            if chunked_data:
                all_chunks_to_process.extend(chunked_data)
                logger.info(f"Successfully added {len(chunked_data)} chunks from source '{source_type}'.")
            else: logger.warning(f"No chunks generated after splitting for source '{source_type}'.")
        elif data is None: logger.warning(f"Skipping source '{source_type}' as file was not found or failed to load/parse from GCS.")
        elif not data: logger.info(f"Skipping source '{source_type}' as the input file from GCS was empty.")
        else: logger.warning(f"Skipping source '{source_type}' as loaded data was not a list ({type(data)}).")

    load_chunk_end = time.time()
    logger.info(f"--- Phase 1 Finished ({load_chunk_end - load_chunk_start:.2f} seconds) ---")
    logger.info(f"Total chunks generated across all sources: {len(all_chunks_to_process)}")

    if not all_chunks_to_process:
        logger.warning("No chunks generated from any source. Pipeline finished successfully but no data processed.")
        return # Exit successfully if no chunks

    # --- Phase 2: Generate Embeddings ---
    logger.info("--- Phase 2: Generating Embeddings ---")
    embedding_start = time.time()
    # generate_embeddings_batched returns (chunks_with_embeddings, embedding_model)
    chunks_with_embeddings, embedding_model = generate_embeddings_batched(all_chunks_to_process)
    embedding_end = time.time()
    logger.info(f"--- Phase 2 Finished ({embedding_end - embedding_start:.2f} seconds) ---")

    if not chunks_with_embeddings:
        logger.error("Embedding generation failed or yielded no results with valid embeddings.")
        raise RuntimeError("Embedding generation failed or yielded no valid embeddings.")

    # --- Phase 3: Store Embeddings in Pinecone ---
    logger.info("--- Phase 3: Storing in Pinecone ---")
    storage_start = time.time()
    # Pass the initialized client 'pc' and the correct INDEX_NAME
    storage_success = store_in_pinecone(chunks_with_embeddings, pc, INDEX_NAME)
    storage_end = time.time()
    logger.info(f"--- Phase 3 Finished ({storage_end - storage_start:.2f} seconds) ---")

    if not storage_success:
         logger.error("Pinecone storage process encountered critical errors.")
         raise RuntimeError("Pinecone storage process failed.")

    # --- Optional: Example Query ---
    if embedding_model: # Check if model loaded successfully
        try:
            logger.info("--- Running example Pinecone query ---")
            query_start = time.time()
            # Pass the initialized client 'pc' and the correct INDEX_NAME
            query_pinecone("How to improve pull-ups?", embedding_model, pc, INDEX_NAME)
            query_end = time.time()
            logger.info(f"--- Example Query Finished ({query_end - query_start:.2f} seconds) ---")
        except Exception as query_e:
             # Log error but don't fail the whole task just because query failed
             logger.error(f"Example query failed: {query_e}", exc_info=True)
    else:
         logger.warning("Skipping example query because embedding model was not loaded successfully.")

    # --- Final Status ---
    end_time = time.time()
    logger.info(f"--- Chunk, Embed, Store Pipeline finished successfully in {end_time - start_time:.2f} seconds ---")
    # If function reaches here without raising an exception, Airflow marks it as success


# --- Main Guard ---
# This part is only for direct script execution, not used by Airflow DAG
if __name__ == "__main__":
    logger.info("Running vector_db.py script directly for testing...")
    try:
        # Add **kwargs={} if the function requires it when called directly
        run_chunk_embed_store_pipeline()
        logger.info("Direct script execution finished successfully.")
    except Exception as e:
         logger.error(f"Direct script execution failed: {e}", exc_info=True)