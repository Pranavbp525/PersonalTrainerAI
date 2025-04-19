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
load_dotenv()

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
logger.info("Logging initialized in vector_db.py")

# --- Constants ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Ensure your Pinecone env matches where your index is (Serverless uses region/cloud)
# PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Less relevant for Serverless init
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-bot") # Your index name
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

# **Step 1: Load JSON Data - Now handled within main loop using GCS utils **
# (Removed the old load_json_data function)

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

        # *** IMPORTANT: Reading from 'text' field ***
        # *** Change to 'description' if your preprocessing outputs that key ***
        text_to_split = item.get("text", "")
        source = item.get("source", f"unknown_{source_type}") # Use GCS path or original source
        title = item.get("title", "N/A")
        url = item.get("url", "N/A")

        if not text_to_split:
            logger.debug(f"Skipping item #{item_index+1} with empty text field: source={source}, title={title}")
            items_skipped_no_text += 1
            continue

        try:
            # Split the text from the 'text' field
            chunks = text_splitter.split_text(text_to_split)
            base_id = f"{source_type}_{item_index}" # Create a base ID for the original document

            for chunk_idx, chunk_text in enumerate(chunks):
                # Create a unique ID for each chunk
                chunk_id = f"{base_id}_chunk_{chunk_idx}"
                chunked_data.append({
                    "id": chunk_id, # Use 'id' for the Pinecone vector ID
                    "text": chunk_text, # The actual text content of the chunk
                    # Metadata to be stored with the vector in Pinecone
                    "metadata": {
                         "original_source": source, # Where the data originally came from (URL/GCS path)
                         "title": title,
                         "url": url,
                         "data_source_type": source_type, # e.g., 'pdf', 'ms', 'articles'
                         "chunk_index": chunk_idx,
                         "text": chunk_text # Also store full text in metadata for retrieval display
                     }
                })
        except Exception as e:
            logger.error(f"Error splitting text for item #{item_index+1} (source={source}, title={title}): {e}", exc_info=True)
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
                all_embeddings.extend([None] * len(batch_texts)) # Add None for all items in original batch
                continue

            # Generate embeddings for the valid texts in the batch
            batch_embeddings = model.embed_documents(batch_texts_non_empty)

            # Map results back to the original batch structure, including Nones for empty inputs
            embedding_iter = iter(batch_embeddings)
            batch_results = []
            for text in batch_texts:
                if isinstance(text, str) and text.strip():
                     try:
                          batch_results.append(next(embedding_iter))
                     except StopIteration:
                          logger.error("Embedding model returned fewer results than expected for batch. Assigning None.")
                          batch_results.append(None)
                else:
                     batch_results.append(None) # Assign None for empty/invalid input text
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
         return [], model # Abort if lengths mismatch

    for i, chunk_item in enumerate(chunked_data):
        embedding_result = all_embeddings[i]
        if embedding_result is not None and isinstance(embedding_result, list) and len(embedding_result) == EMBEDDING_DIMENSION:
            chunk_item["embedding"] = embedding_result
            chunks_with_embeddings.append(chunk_item)
        else:
            logger.warning(f"Embedding missing, None, or wrong dimension ({len(embedding_result) if isinstance(embedding_result, list) else type(embedding_result)}) for chunk ID {chunk_item.get('id')}. Skipping this chunk.")
            failed_count += 1

    if failed_count > 0:
         logger.warning(f"Total chunks skipped due to failed/invalid embeddings: {failed_count}")

    logger.info(f"Returning {len(chunks_with_embeddings)} chunks with valid embeddings.")
    return chunks_with_embeddings, model


# **Step 4: Store Chunks in Pinecone**
def store_in_pinecone(chunks_with_embeddings, pc_client: Pinecone, index_name: str):
    """Stores chunked data with embeddings in the specified Pinecone index."""
    logger.info(f"Preparing to store {len(chunks_with_embeddings)} chunks with embeddings in Pinecone index '{index_name}'...")
    if not chunks_with_embeddings:
        logger.warning("No valid chunks with embeddings provided for storage.")
        return True # Nothing to store is not a failure

    try:
        # --- Check/Connect to Index --- (Moved connection logic here from main)
        existing_indexes = pc_client.list_indexes().names
        logger.info(f"Existing Pinecone indexes: {existing_indexes}")
        if index_name not in existing_indexes:
            logger.error(f"Target Pinecone index '{index_name}' does not exist! Please create it manually or add creation logic.")
            # Optional: Add creation logic here if desired
            # logger.info(f"Creating index '{index_name}'...")
            # pc_client.create_index(...)
            return False # Fail if index doesn't exist and isn't created

        logger.info(f"Connecting to Pinecone index '{index_name}'...")
        index = pc_client.Index(index_name)
        # Verify connection by getting stats
        try:
             stats = index.describe_index_stats()
             logger.info(f"Connected successfully. Index stats: {stats}")
        except Exception as desc_err:
             logger.error(f"Failed to get stats for index '{index_name}'. Connection issue? Error: {desc_err}")
             return False # Fail if cannot connect/describe

        # --- Prepare vectors for upsert ---
        vectors_to_upsert = []
        skipped_chunks = 0
        for chunk in chunks_with_embeddings:
            # Validate required fields for upsert
            vector_id = chunk.get("id")
            embedding = chunk.get("embedding")
            metadata = chunk.get("metadata")

            if isinstance(vector_id, str) and \
               isinstance(embedding, list) and len(embedding) == EMBEDDING_DIMENSION and \
               isinstance(metadata, dict):
                # Ensure metadata only contains compatible types (str, bool, number, list of strings)
                # Pinecone might error if metadata is too complex
                cleaned_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, bool, int, float))}
                # Re-add text to metadata if needed and not already there
                if 'text' not in cleaned_metadata and isinstance(chunk.get('text'), str) :
                    cleaned_metadata['text'] = chunk['text']

                vectors_to_upsert.append(
                    (vector_id, embedding, cleaned_metadata) # Pinecone tuple format
                )
            else:
                logger.warning(f"Skipping invalid chunk during vector preparation (missing id, embedding, metadata, or wrong format): ID={vector_id}")
                skipped_chunks += 1

        if skipped_chunks > 0: logger.warning(f"Skipped {skipped_chunks} invalid chunks during vector preparation.")

        if not vectors_to_upsert:
            logger.error("No valid vectors prepared for upserting after validation.")
            return False # Fail if nothing can be upserted

        # --- Upsert in Batches ---
        total_vectors = len(vectors_to_upsert)
        batch_size = PINECONE_UPSERT_BATCH_SIZE
        num_batches = math.ceil(total_vectors / batch_size)
        logger.info(f"Upserting {total_vectors} vectors to index '{index_name}' in {num_batches} batches of size {batch_size}...")
        total_upserted_count = 0

        for i in tqdm(range(0, total_vectors, batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                upsert_response = index.upsert(vectors=batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}/{num_batches}. Response: {upsert_response}")
                if upsert_response and upsert_response.upserted_count:
                     total_upserted_count += upsert_response.upserted_count
                else:
                     logger.warning(f"Pinecone upsert response for batch {i//batch_size + 1} did not report upserted count.")
            except Exception as upsert_err:
                # Log the error but continue to next batch (maybe some batches succeed)
                logger.error(f"Error upserting batch starting at index {i}: {upsert_err}", exc_info=True)
                # Consider adding failed batch IDs to a list for retry later if needed

        logger.info(f"Finished upserting vectors. Total reported upserted count by Pinecone: {total_upserted_count} (Target: {total_vectors})")
        # Log final index stats
        try:
            final_stats = index.describe_index_stats()
            logger.info(f"Final index stats after upsert: {final_stats}")
        except Exception as final_stat_err:
             logger.warning(f"Could not get final index stats: {final_stat_err}")

        # Consider success even if some batches failed, depends on requirements
        return True # Indicate storage process completed (maybe with errors)

    except Exception as e:
        logger.error(f"A critical error occurred during the Pinecone storage process: {e}", exc_info=True)
        return False # Indicate overall storage failure


# **Step 5: Query Pinecone**
def query_pinecone(query: str, model, pc_client: Pinecone, index_name: str):
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

    logger.info(f"Attempting to query Pinecone index '{index_name}' with: '{query}'")
    try:
        logger.debug("Generating query embedding...")
        query_embedding = model.embed_query(query)
        logger.debug("Query embedding generated.")
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        return None

    try:
        logger.debug(f"Connecting to index '{index_name}' for query...")
        index = pc_client.Index(index_name)
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
                text_chunk = metadata.get('text', 'N/A')
                title = metadata.get('title', 'N/A')
                original_source = metadata.get('original_source', 'N/A')
                score = match.get('score', 'N/A')
                chunk_id = match.get('id', 'N/A')

                logger.info(f"Match {i+1}:")
                logger.info(f"  Score: {score:.4f}")
                logger.info(f"  ID: {chunk_id}")
                logger.info(f"  Source: {original_source}")
                logger.info(f"  Title: {title}")
                logger.info(f"  Text Chunk: {text_chunk[:200]}...") # Display preview
                logger.info("---")
        return result # Return the full result object
    except Exception as e:
        logger.error(f"Error querying Pinecone index '{index_name}': {e}", exc_info=True)
        return None


# **Step 6: Main Execution Function for Airflow**
def run_chunk_embed_store_pipeline():
    """
    Main function orchestrating loading from GCS, chunking, embedding,
    and storage in Pinecone. Called by the Airflow DAG task.
    Returns True on success, False on critical failure.
    """
    logger.info("--- Starting Chunk, Embed, Store Pipeline (GCS -> Pinecone) ---")
    start_time = time.time()
    overall_success = True # Assume success initially

    # --- Initialize Pinecone Client ---
    if not PINECONE_API_KEY:
        logger.error("CRITICAL: PINECONE_API_KEY environment variable not found.")
        return False # Cannot proceed without API key
    try:
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Maybe add a simple check like list_indexes to confirm connection
        pc.list_indexes()
        logger.info("Pinecone client initialized and connection confirmed.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Pinecone client or confirm connection: {e}", exc_info=True)
        return False # Cannot proceed without Pinecone client

    # --- Phase 1: Load and Chunk All Data from GCS ---
    logger.info("--- Phase 1: Loading and Chunking Data from GCS ---")
    load_chunk_start = time.time()
    all_data_to_embed = []
    any_data_loaded = False

    for source_type, input_blob_name in INPUT_FILES_TO_PROCESS.items():
        gcs_input_path = f"gs://{PROCESSED_BUCKET}/{input_blob_name}"
        logger.info(f"Processing source: {source_type} | File: {gcs_input_path}")

        # Read preprocessed JSON from GCS
        data = read_json_from_gcs(PROCESSED_BUCKET, input_blob_name)

        if data is not None and isinstance(data, list):
            any_data_loaded = True
            # Split the text data
            chunked_data = split_text(data, source_type=source_type)
            if chunked_data:
                all_data_to_embed.extend(chunked_data)
                logger.info(f"Successfully added {len(chunked_data)} chunks from source '{source_type}'.")
            else:
                logger.warning(f"No chunks generated after splitting for source '{source_type}'.")
        elif data is None:
             logger.warning(f"Skipping source '{source_type}' as file was not found or failed to load/parse from GCS.")
             # Decide if this constitutes failure - potentially set overall_success = False
        else:
             logger.warning(f"Skipping source '{source_type}' as loaded data was not a list ({type(data)}).")


    load_chunk_end = time.time()
    logger.info(f"--- Phase 1 Finished ({load_chunk_end - load_chunk_start:.2f} seconds) ---")
    logger.info(f"Total chunks generated across all sources: {len(all_data_to_embed)}")

    if not any_data_loaded:
         logger.error("CRITICAL: No input data could be loaded from GCS. Stopping pipeline.")
         return False # Fail if no input was loaded at all
    if not all_data_to_embed:
        logger.warning("No chunks generated from loaded data. Pipeline will stop before embedding.")
        return True # Considered success as processing finished, just no output

    # --- Phase 2: Generate Embeddings ---
    logger.info("--- Phase 2: Generating Embeddings ---")
    embedding_start = time.time()
    chunks_with_embeddings, embedding_model = generate_embeddings_batched(all_data_to_embed)
    embedding_end = time.time()
    logger.info(f"--- Phase 2 Finished ({embedding_end - embedding_start:.2f} seconds) ---")

    if not chunks_with_embeddings:
        logger.error("Embedding generation failed or yielded no results. Cannot proceed to storage.")
        overall_success = False
        # We might still want to run the example query if the model loaded? No, model might be None.
        return False # Fail if no embeddings generated

    # --- Phase 3: Store Embeddings in Pinecone ---
    logger.info("--- Phase 3: Storing in Pinecone ---")
    storage_start = time.time()
    storage_success = store_in_pinecone(chunks_with_embeddings, pc, INDEX_NAME)
    storage_end = time.time()
    logger.info(f"--- Phase 3 Finished ({storage_end - storage_start:.2f} seconds) ---")

    if not storage_success:
         logger.error("Pinecone storage process encountered critical errors.")
         overall_success = False # Mark overall failure if storage had issues

    # --- Optional: Example Query ---
    # Run only if embedding model is available and storage process didn't critically fail early
    if embedding_model and overall_success:
        try:
            logger.info("--- Running example Pinecone query ---")
            query_start = time.time()
            query_pinecone("How to improve pull-ups?", embedding_model, pc, INDEX_NAME)
            query_end = time.time()
            logger.info(f"--- Example Query Finished ({query_end - query_start:.2f} seconds) ---")
        except Exception as query_e:
             logger.error(f"Example query failed: {query_e}", exc_info=True)
             # Don't necessarily mark overall failure for failed example query
    elif not embedding_model:
         logger.warning("Skipping example query because embedding model was not loaded successfully.")
    else:
         logger.warning("Skipping example query due to earlier critical failures.")


    # --- Final Status ---
    end_time = time.time()
    if overall_success:
        logger.info(f"--- Chunk, Embed, Store Pipeline finished successfully in {end_time - start_time:.2f} seconds ---")
        return True
    else:
         logger.error(f"--- Chunk, Embed, Store Pipeline finished with errors in {end_time - start_time:.2f} seconds ---")
         return False


# --- Main Guard ---
if __name__ == "__main__":
    logger.info("Running vector_db.py script directly for testing...")
    # Prerequisites for local testing:
    # 1. gcs_utils.py must be in the same directory or accessible via PYTHONPATH.
    # 2. Run 'gcloud auth application-default login' in your terminal.
    # 3. Ensure input files (e.g., pdf_data.json) exist in gs://ragllm-454718-processed-data/preprocessed_json_data/
    # 4. Ensure PINECONE_API_KEY is set in your environment (e.g., via a .env file loaded by load_dotenv()).
    # 5. Ensure Pinecone index 'fitness-bot' exists or add creation logic.
    run_chunk_embed_store_pipeline()
    logger.info("Direct script execution finished.")