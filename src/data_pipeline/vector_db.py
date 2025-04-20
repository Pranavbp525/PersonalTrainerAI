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

# --- GCS Utils Import ---
try:
    # This assumes gcs_utils.py is in the same directory or PYTHONPATH is correctly set
    from .gcs_utils import read_json_from_gcs
except ImportError:
    # Fallback for potential direct execution testing (less common in DAG context)
    print("WARNING: Could not perform relative import of gcs_utils. Attempting direct import.")
    from gcs_utils import read_json_from_gcs

# --- Setup logger FIRST ---
# Define log file path using __file__ (safer than name)
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/vectordb.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", # Added brackets for clarity
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Append mode
        logging.StreamHandler() # Also log to console (Airflow logs)
    ]
)
# Get the logger for the current module
logger = logging.getLogger(__name__)
logger.info("--- Logging initialized in vector_db.py ---")


# --- Load environment variables ---
# Use the path expected inside the container based on docker-compose mount
dotenv_path = "/opt/airflow/app/.env"
loaded_env = False # Flag to track if .env was loaded
if os.path.exists(dotenv_path):
    # Use override=True if you want .env to take precedence over system env vars
    loaded_env = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
    logger.info(f"Attempted to load .env file from {dotenv_path}. Load successful: {loaded_env}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying solely on environment variables already set.")

# --- Constants ---
# Retrieve values AFTER attempting to load .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Ensure index name matches your .env or desired index
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot") # Default to 'fitness-chatbot' if not set
# PINECONE_REGION and PINECONE_CLOUD are primarily for index creation, not connection usually
# PINECONE_REGION = "us-east-1"
# PINECONE_CLOUD = "aws"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768 # Dimension for all-mpnet-base-v2
EMBEDDING_BATCH_SIZE = 32 # Reduce if memory issues occur during embedding
PINECONE_UPSERT_BATCH_SIZE = 100 # Pinecone recommended batch size

# GCS Configuration for reading PREPROCESSED data
PROCESSED_BUCKET = os.getenv("PROCESSED_BUCKET", "ragllm-454718-processed-data") # Allow override via env
INPUT_BLOB_PREFIX = "preprocessed_json_data/"
# Define the input files to process from GCS
INPUT_FILES_TO_PROCESS = {
    "pdf": f"{INPUT_BLOB_PREFIX}pdf_data.json",
    "ms": f"{INPUT_BLOB_PREFIX}ms_data.json",
    "articles": f"{INPUT_BLOB_PREFIX}blogs.json"
}
logger.info(f"Target Pinecone Index: {INDEX_NAME}")
logger.info(f"Reading from GCS Bucket: {PROCESSED_BUCKET}")


# --- Step 2: Split Text into Chunks ---
def split_text(data, source_type, chunk_size=400, chunk_overlap=50):
    """Splits text from loaded data into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""], # Added ". " etc. and empty string as final fallback
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
        text_to_split = item.get("text") # Use get() for safer access
        # Extract other metadata fields (use defaults if missing)
        original_gcs_source = item.get("source", f"gs://{PROCESSED_BUCKET}/{source_type}_{item_index}") # Default GCS path
        title = item.get("title", "N/A")
        url = item.get("url", "N/A")

        # Ensure text_to_split is a non-empty string
        if not text_to_split or not isinstance(text_to_split, str) or not text_to_split.strip():
            logger.debug(f"Skipping item #{item_index+1} with missing, non-string, or empty text field: source={original_gcs_source}, title={title}")
            items_skipped_no_text += 1
            continue

        try:
            # Split the text from the 'text' field
            chunks = text_splitter.split_text(text_to_split)
            # Create a base ID using the source type and original item index
            # Ensure base_id is URL-safe if needed, but fine for Pinecone ID here
            base_id = f"{source_type}-{item_index}" # Use hyphen for readability

            for chunk_idx, chunk_text in enumerate(chunks):
                # Create a unique ID for each chunk
                chunk_id = f"{base_id}-chunk-{chunk_idx}"
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

    if items_skipped_no_text > 0: logger.warning(f"Skipped {items_skipped_no_text} items from {source_type} due to missing/empty text field.")
    if items_skipped_not_dict > 0: logger.warning(f"Skipped {items_skipped_not_dict} items from {source_type} because they were not dictionaries.")
    logger.info(f"Generated {len(chunked_data)} chunks from source: {source_type}")
    return chunked_data


# --- Step 3: Generate Embeddings (BATCHED) ---
def generate_embeddings_batched(chunked_data, model_name=EMBEDDING_MODEL_NAME, batch_size=EMBEDDING_BATCH_SIZE):
    """Generates embeddings for text chunks IN BATCHES."""
    if not chunked_data:
        logger.warning("No chunked data provided for embedding generation.")
        return [], None # Return empty list and no model

    logger.info(f"Initializing embedding model: {model_name}")
    try:
        # Ensure device selection is appropriate ('cpu' is safer in general container env)
        # Add cache_folder to potentially speed up loading on subsequent runs if volume persists
        # embedding_cache_dir = os.path.join(os.path.dirname(__file__), ".embedding_cache") # Example cache dir
        # os.makedirs(embedding_cache_dir, exist_ok=True)
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            # cache_folder=embedding_cache_dir # Uncomment if you want caching
            )
        logger.info("Embedding model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name}: {e}", exc_info=True)
        return [], None # Return empty list and no model if init fails

    # Extract the text content for embedding
    texts_to_embed = [item.get("text", "") for item in chunked_data]
    all_embeddings = []
    total_items = len(texts_to_embed)
    total_batches = math.ceil(total_items / batch_size)

    logger.info(f"Generating embeddings for {total_items} text chunks in {total_batches} batches of size {batch_size}...")

    try:
        # Use embed_documents directly for potential batching efficiency within the library
        all_embeddings = model.embed_documents(texts_to_embed, chunk_size=batch_size) # Use library's batching if available

        # Verify the length matches
        if len(all_embeddings) != total_items:
             logger.error(f"CRITICAL: Mismatch after embed_documents! Expected {total_items}, got {len(all_embeddings)}.")
             # Pad with Nones if length is less, though this indicates a deeper issue
             if len(all_embeddings) < total_items:
                 all_embeddings.extend([None] * (total_items - len(all_embeddings)))
             else: # Too many embeddings? Trim (also indicates issue)
                 all_embeddings = all_embeddings[:total_items]

    except Exception as e:
        logger.error(f"Error during embedding generation with embed_documents: {e}", exc_info=True)
        # Attempt to return partial results or fail completely
        # For simplicity, we fail completely if embed_documents raises an error
        return [], model # Return empty data but keep model reference if needed

    logger.info(f"Finished embedding generation. Received {len(all_embeddings)} embedding results.")

    # Add embeddings back to the chunked data, validating dimensions
    chunks_with_embeddings = []
    failed_count = 0
    for i, chunk_item in enumerate(chunked_data):
        embedding_result = all_embeddings[i]
        # Validate the embedding result
        if embedding_result is not None and isinstance(embedding_result, list) and len(embedding_result) == EMBEDDING_DIMENSION:
            chunk_item["embedding"] = embedding_result
            chunks_with_embeddings.append(chunk_item)
        else:
            embedding_info = f"type={type(embedding_result)}"
            if isinstance(embedding_result, list):
                 embedding_info += f", len={len(embedding_result)}"
            logger.warning(f"Embedding missing, None, or wrong dimension ({embedding_info}) for chunk ID {chunk_item.get('id', 'N/A')}. Skipping this chunk.")
            failed_count += 1

    if failed_count > 0:
         logger.warning(f"Total chunks skipped due to failed/invalid embeddings: {failed_count}")

    logger.info(f"Returning {len(chunks_with_embeddings)} chunks with valid embeddings.")
    return chunks_with_embeddings, model


# --- Step 4: Store Chunks in Pinecone ---
def store_in_pinecone(chunks_with_embeddings, pc_client: Pinecone, target_index_name: str):
    """Stores chunked data with embeddings in the specified Pinecone index."""
    logger.info(f"Preparing to store {len(chunks_with_embeddings)} chunks with embeddings in Pinecone index '{target_index_name}'...")
    if not chunks_with_embeddings:
        logger.warning("No valid chunks with embeddings provided for storage.")
        return True # Nothing to store is not a failure

    try:
        # --- Check if Index Exists ---
        logger.info(f"Listing existing Pinecone indexes to verify target index '{target_index_name}' exists...")
        # --- CORRECTED CODE TO GET INDEX NAMES ---
        index_list_obj = pc_client.list_indexes() # Get the IndexList object first
        existing_index_names = index_list_obj.names # Access the 'names' attribute which IS the list
        # --- END CORRECTION ---

        logger.info(f"Found existing Pinecone index names: {existing_index_names}") # Log the actual list

        # Check if the target index exists in the retrieved list
        if target_index_name not in existing_index_names:
            logger.error(f"Target Pinecone index '{target_index_name}' does not exist in the list: {existing_index_names}! Please create it first via the Pinecone console or API.")
            return False # Indicate failure: index does not exist

        # --- Connect to the Target Index ---
        logger.info(f"Connecting to Pinecone index '{target_index_name}'...")
        index: Index = pc_client.Index(target_index_name)

        # Verify connection by getting stats (Optional but good practice)
        try:
             stats = index.describe_index_stats()
             logger.info(f"Connected successfully. Index stats before upsert: {stats}")
        except Exception as desc_err:
             # Log warning but proceed cautiously if stats check fails
             logger.warning(f"Could not get stats for index '{target_index_name}'. Proceeding with upsert attempt... Error: {desc_err}")

        # --- Prepare vectors for upsert (validate and clean metadata) ---
        vectors_to_upsert = []
        skipped_chunks = 0
        logger.info("Preparing and validating vectors for Pinecone upsert...")
        for chunk in chunks_with_embeddings:
            vector_id = chunk.get("id")
            embedding = chunk.get("embedding")
            metadata = chunk.get("metadata")

            # Robust type and dimension check
            if isinstance(vector_id, str) and \
               isinstance(embedding, list) and len(embedding) == EMBEDDING_DIMENSION and \
               isinstance(metadata, dict):

                # Clean metadata: Keep only supported types, convert others to string
                cleaned_metadata = {}
                for k, v in metadata.items():
                    # Ensure key is also a string
                    key_str = str(k)
                    if isinstance(v, (str, bool, int, float)):
                        # Check string length (Pinecone has limits, e.g., ~40k bytes per value)
                        if isinstance(v, str) and len(v.encode('utf-8')) > 40000:
                             logger.warning(f"Truncating long metadata string key '{key_str}' for vector '{vector_id}'")
                             cleaned_metadata[key_str] = v[:10000] + "... (truncated)"
                        else:
                             cleaned_metadata[key_str] = v
                    elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                         # Add checks for list length or individual string lengths if needed
                         cleaned_metadata[key_str] = v # List of strings is supported
                    else:
                         # Convert other types to string as fallback and log
                         logger.debug(f"Converting unsupported metadata type ({type(v)}) to string for key '{key_str}' in vector '{vector_id}'")
                         try:
                              cleaned_metadata[key_str] = str(v)
                         except Exception:
                              logger.warning(f"Could not convert metadata key '{key_str}' to string. Skipping this metadata field for vector '{vector_id}'.")
                              cleaned_metadata[key_str] = "CONVERSION_ERROR" # Placeholder

                vectors_to_upsert.append(
                    # Using Pinecone tuple format: (id, vector_values, metadata_dict)
                    (vector_id, embedding, cleaned_metadata)
                )
            else:
                # Log detailed info about skipped chunk
                reason = "missing/invalid id" if not isinstance(vector_id, str) else \
                         "missing/invalid embedding" if not isinstance(embedding, list) else \
                         f"wrong embedding dimension (got {len(embedding)}, expected {EMBEDDING_DIMENSION})" if isinstance(embedding, list) else \
                         "missing/invalid metadata" if not isinstance(metadata, dict) else \
                         "unknown format issue"
                logger.warning(f"Skipping invalid chunk during vector preparation ({reason}): ID={vector_id}")
                skipped_chunks += 1

        if skipped_chunks > 0: logger.warning(f"Skipped {skipped_chunks} invalid chunks during vector preparation.")

        if not vectors_to_upsert:
            logger.warning("No valid vectors prepared for upserting after validation.")
            # Return True because no upsert operation needed to fail
            return True

        # --- Upsert in Batches ---
        total_vectors = len(vectors_to_upsert)
        batch_size = PINECONE_UPSERT_BATCH_SIZE
        num_batches = math.ceil(total_vectors / batch_size)
        logger.info(f"Upserting {total_vectors} vectors to index '{target_index_name}' in {num_batches} batches of size {batch_size}...")
        total_upserted_count = 0
        storage_had_errors = False

        for i in tqdm(range(0, total_vectors, batch_size), desc="Upserting to Pinecone", total=num_batches):
            batch = vectors_to_upsert[i:i + batch_size]
            if not batch: continue # Skip empty batches (shouldn't happen with ceil logic)
            try:
                upsert_response = index.upsert(vectors=batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}/{num_batches}. Response: {upsert_response}")
                # More robust check for upserted count
                current_upserted = getattr(upsert_response, 'upserted_count', 0)
                if current_upserted is None: current_upserted = 0 # Handle case where count is None

                total_upserted_count += current_upserted
                if current_upserted != len(batch):
                     logger.warning(f"Batch {i//batch_size + 1} upsert count ({current_upserted}) differs from batch size ({len(batch)}). May indicate partial success or issues.")
            except Exception as upsert_err:
                logger.error(f"Error upserting batch starting at index {i}: {upsert_err}", exc_info=True)
                storage_had_errors = True
                # Consider adding logic here to collect failed IDs from 'batch' for retry later

        logger.info(f"Finished upsert loop. Total reported upserted count by Pinecone: {total_upserted_count} (Vectors sent: {total_vectors})")
        # Log final index stats
        try:
            final_stats = index.describe_index_stats()
            logger.info(f"Final index stats after upsert: {final_stats}")
        except Exception as final_stat_err:
             logger.warning(f"Could not get final index stats: {final_stat_err}")

        # Return True only if NO errors occurred during the entire batch upsert process
        return not storage_had_errors

    except Exception as e:
        # Catch errors during index listing, connection, or other critical setup
        logger.error(f"A critical error occurred during the Pinecone storage setup/process: {e}", exc_info=True)
        return False # Indicate overall storage failure


# --- Step 5: Query Pinecone ---
def query_pinecone(query: str, model, pc_client: Pinecone, target_index_name: str):
    """Queries the Pinecone index with the given text query."""
    if not model:
        logger.error("Cannot query: Embedding model is not available.")
        return None
    if not pc_client:
        logger.error("Cannot query: Pinecone client is not available.")
        return None
    if not query or not isinstance(query, str) or not query.strip():
         logger.warning(f"Cannot query: Invalid query string provided ('{query}').")
         return None

    logger.info(f"Attempting to query Pinecone index '{target_index_name}' with query (preview): '{query[:100]}...'")
    try:
        logger.debug("Generating query embedding...")
        query_embedding = model.embed_query(query)
        if not query_embedding or not isinstance(query_embedding, list) or len(query_embedding) != EMBEDDING_DIMENSION:
             logger.error(f"Failed to generate valid query embedding (got type {type(query_embedding)}, len {len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}).")
             return None
        logger.debug("Query embedding generated successfully.")
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        return None

    try:
        logger.debug(f"Connecting to index '{target_index_name}' for query...")
        index: Index = pc_client.Index(target_index_name) # Get index object

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
                match_id = match.get('id', 'N/A')
                score = match.get('score', 'N/A')
                metadata = match.get('metadata', {})
                # Use get() with defaults for safer access
                text_chunk = metadata.get('text', 'N/A') # Get stored text chunk
                title = metadata.get('title', 'N/A')
                original_source = metadata.get('original_source', 'N/A')

                # Log retrieved information
                logger.info(f"Match {i+1}:")
                logger.info(f"  Score: {score:.4f}")
                logger.info(f"  ID: {match_id}")
                logger.info(f"  Source: {original_source}")
                logger.info(f"  Title: {title}")
                logger.info(f"  Text Chunk: {text_chunk[:200]}...") # Display preview
                logger.info("---")
        return result # Return the full result object

    except Exception as e:
        logger.error(f"Error querying Pinecone index '{target_index_name}': {e}", exc_info=True)
        return None


# --- Step 6: Main Execution Function for Airflow ---
# --- This is the function called by the PythonOperator in the DAG ---
def run_chunk_embed_store_pipeline(**kwargs): # Add **kwargs to accept Airflow context
    """
    Main function orchestrating loading from GCS, chunking, embedding,
    and storage in Pinecone. Called by the Airflow DAG task.
    Raises exceptions on critical failure.
    """
    # Use Airflow's task logger if available, otherwise use the module logger
    try:
        task_log = logging.getLogger("airflow.task")
        if not task_log.hasHandlers(): # Check if handlers are configured
             task_log = logger # Fallback to module logger if task logger not ready
    except Exception:
        task_log = logger # Fallback on any error getting task logger

    task_log.info("--- Starting Chunk, Embed, Store Pipeline (GCS -> Pinecone) ---")
    start_time = time.time()

    # --- Initialize Pinecone Client ---
    # API key is retrieved from env vars loaded at the top or passed by Airflow
    if not PINECONE_API_KEY:
        task_log.error("CRITICAL: PINECONE_API_KEY environment variable not found.")
        raise ValueError("PINECONE_API_KEY is required but not set.")
    try:
        task_log.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Simple check to confirm connection/auth works before proceeding
        pc.list_indexes()
        task_log.info("Pinecone client initialized and connection confirmed.")
    except Exception as e:
        task_log.error(f"CRITICAL: Failed to initialize Pinecone client or confirm connection: {e}", exc_info=True)
        # Raising RuntimeError will fail the Airflow task
        raise RuntimeError(f"Failed to initialize Pinecone: {e}") from e

    # --- Phase 1: Load and Chunk All Data from GCS ---
    task_log.info("--- Phase 1: Loading and Chunking Data from GCS ---")
    load_chunk_start = time.time()
    all_chunks_to_process = []

    for source_type, input_blob_name in INPUT_FILES_TO_PROCESS.items():
        gcs_input_path = f"gs://{PROCESSED_BUCKET}/{input_blob_name}"
        task_log.info(f"Processing source: {source_type} | File: {gcs_input_path}")
        # Use the helper function imported at the top
        data = read_json_from_gcs(PROCESSED_BUCKET, input_blob_name)
        if data is not None and isinstance(data, list) and data:
            # Use the helper function imported at the top
            chunked_data = split_text(data, source_type=source_type)
            if chunked_data:
                all_chunks_to_process.extend(chunked_data)
                task_log.info(f"Successfully added {len(chunked_data)} chunks from source '{source_type}'.")
            else: task_log.warning(f"No chunks generated after splitting for source '{source_type}'.")
        # Log reasons for skipping more clearly
        elif data is None: task_log.warning(f"Skipping source '{source_type}' as file was not found or failed to load/parse from GCS.")
        elif isinstance(data, list) and not data: task_log.info(f"Skipping source '{source_type}' as the input file from GCS was empty.")
        else: task_log.warning(f"Skipping source '{source_type}' as loaded data was not a list ({type(data)}).")

    load_chunk_end = time.time()
    task_log.info(f"--- Phase 1 Finished ({load_chunk_end - load_chunk_start:.2f} seconds) ---")
    task_log.info(f"Total chunks generated across all sources: {len(all_chunks_to_process)}")

    if not all_chunks_to_process:
        task_log.warning("No chunks generated from any source. Pipeline finished successfully but no data processed.")
        return # Exit successfully if no chunks, task succeeds

    # --- Phase 2: Generate Embeddings ---
    task_log.info("--- Phase 2: Generating Embeddings ---")
    embedding_start = time.time()
    # generate_embeddings_batched returns (chunks_with_embeddings, embedding_model)
    chunks_with_embeddings, embedding_model = generate_embeddings_batched(all_chunks_to_process)
    embedding_end = time.time()
    task_log.info(f"--- Phase 2 Finished ({embedding_end - embedding_start:.2f} seconds) ---")

    # Check if embedding failed or yielded no valid results
    if embedding_model is None or not chunks_with_embeddings:
        task_log.error("Embedding generation failed or yielded no results with valid embeddings.")
        raise RuntimeError("Embedding generation failed or yielded no valid embeddings.")

    # --- Phase 3: Store Embeddings in Pinecone ---
    task_log.info("--- Phase 3: Storing in Pinecone ---")
    storage_start = time.time()
    # Pass the initialized client 'pc' and the target INDEX_NAME
    storage_success = store_in_pinecone(chunks_with_embeddings, pc, INDEX_NAME)
    storage_end = time.time()
    task_log.info(f"--- Phase 3 Finished ({storage_end - storage_start:.2f} seconds) ---")

    # Check if storage process reported errors
    if not storage_success:
         task_log.error("Pinecone storage process encountered errors during upsert.")
         # Fail the Airflow task
         raise RuntimeError("Pinecone storage process failed during upsert.")

    # --- Optional: Example Query ---
    # Run query only if embedding model loaded AND storage succeeded
    if embedding_model:
        try:
            task_log.info("--- Running example Pinecone query ---")
            query_start = time.time()
            # Pass the initialized client 'pc' and the target INDEX_NAME
            query_pinecone("How to improve pull-ups?", embedding_model, pc, INDEX_NAME)
            query_end = time.time()
            task_log.info(f"--- Example Query Finished ({query_end - query_start:.2f} seconds) ---")
        except Exception as query_e:
             # Log error but don't fail the whole task just because query failed
             task_log.error(f"Example query failed: {query_e}", exc_info=True)
    else:
         task_log.warning("Skipping example query because embedding model was not loaded successfully earlier.")

    # --- Final Status ---
    end_time = time.time()
    task_log.info(f"--- Chunk, Embed, Store Pipeline finished successfully in {end_time - start_time:.2f} seconds ---")
    # If the function reaches here without raising an exception, Airflow marks the task as successful


# --- Main Guard ---
# This part is only for direct script execution (e.g., python src/data_pipeline/vector_db.py)
# It won't be executed when Airflow calls run_chunk_embed_store_pipeline
if __name__ == "__main__":
    logger.info("Running vector_db.py script directly for testing...")
    try:
        # Call the main pipeline function
        run_chunk_embed_store_pipeline()
        logger.info("Direct script execution finished successfully.")
    except Exception as e:
         # Log the error but don't crash the script if run directly
         logger.error(f"Direct script execution failed: {e}", exc_info=True)