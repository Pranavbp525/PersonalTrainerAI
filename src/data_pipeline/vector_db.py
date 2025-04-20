# src/data_pipeline/vector_db.py
import os
import json
import logging
import time
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, Index # Ensure Index is imported for type hints
# Use recommended langchain import if available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    logging.info("Imported HuggingFaceEmbeddings from langchain_huggingface")
except ImportError:
    logging.warning("Could not import from langchain_huggingface. Falling back to langchain_community.")
    # Ensure this is installed via _PIP_ADDITIONAL_REQUIREMENTS in worker if needed
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import math
import numpy as np # Keep numpy import if used elsewhere, otherwise optional

# --- GCS Utils Import ---
# Moved GCS import inside functions that need it or check availability globally
gcs_utils_available = False
try:
    from .gcs_utils import read_json_from_gcs
    gcs_utils_available = True
    logging.info("Successfully imported gcs_utils using relative path.")
except ImportError:
    try:
        from gcs_utils import read_json_from_gcs
        gcs_utils_available = True
        logging.warning("Could not perform relative import of gcs_utils. Falling back to direct import.")
    except ImportError:
        logging.error("CRITICAL: Could not import gcs_utils. Cannot read from GCS.")
        # Define dummy function so script doesn't crash immediately on import error
        # Execution will fail later when read_json_from_gcs is called if utils are unavailable.
        def read_json_from_gcs(*args, **kwargs):
             logging.error("gcs_utils not available, read_json_from_gcs called!")
             return None

# --- Setup logger FIRST ---
# Use __file__ to ensure correct path regardless of execution context
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/vectordb.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Configure root logger (will be used by modules unless they define their own)
# Airflow tasks might override this with their own handlers, which is fine.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Append mode for persistence
        logging.StreamHandler() # Log to console (stdout/stderr)
    ]
)
# Get the logger for the current module (__name__ resolves to 'src.data_pipeline.vector_db')
logger = logging.getLogger(__name__)
logger.info(f"--- Logging initialized in {__name__} ---")


# --- Load environment variables ---
# Use the path expected inside the container based on docker-compose mount
dotenv_path = "/opt/airflow/app/.env"
loaded_env = False # Flag to track if .env was loaded
if os.path.exists(dotenv_path):
    # Use override=True if you want .env to take precedence over system env vars
    # verbose=True logs the variables loaded to DEBUG level (if enabled)
    loaded_env = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=False)
    logger.info(f"Attempted to load .env file from {dotenv_path}. Load successful: {loaded_env}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying solely on environment variables already set.")

# --- Constants ---
# Retrieve values AFTER attempting to load .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Use the index name specified in your .env file
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot") # Default matches your .env

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768 # Dimension for all-mpnet-base-v2
EMBEDDING_BATCH_SIZE = 32 # Keep relatively small for CPU embedding
PINECONE_UPSERT_BATCH_SIZE = 100 # Max recommended by Pinecone

# GCS Configuration for reading PREPROCESSED data
PROCESSED_BUCKET = os.getenv("PROCESSED_BUCKET", "ragllm-454718-processed-data") # Allow override via env
INPUT_BLOB_PREFIX = "preprocessed_json_data/"
INPUT_FILES_TO_PROCESS = {
    "pdf": f"{INPUT_BLOB_PREFIX}pdf_data.json",
    "ms": f"{INPUT_BLOB_PREFIX}ms_data.json",
    "articles": f"{INPUT_BLOB_PREFIX}blogs.json"
}
logger.info(f"Target Pinecone Index: {INDEX_NAME}")
logger.info(f"Reading from GCS Bucket: {PROCESSED_BUCKET}")
if not PINECONE_API_KEY: logger.warning("PINECONE_API_KEY not found in environment!")


# --- Step 2: Split Text into Chunks ---
def split_text(data, source_type, chunk_size=400, chunk_overlap=50):
    """Splits text from loaded data into manageable chunks."""
    # (Using the existing robust version of this function)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunked_data = []
    if not isinstance(data, list): logger.error(f"Expected list for splitting (source: {source_type}), got {type(data)}"); return []

    logger.info(f"Splitting text for {len(data)} items from source: {source_type}")
    items_skipped_no_text, items_skipped_not_dict = 0, 0
    for item_index, item in enumerate(data):
        if not isinstance(item, dict): logger.warning(f"Skipping non-dict item #{item_index+1}"); items_skipped_not_dict +=1; continue
        text_to_split, title = item.get("text"), item.get("title", "N/A")
        original_gcs_source, url = item.get("source", f"gs://{PROCESSED_BUCKET}/{source_type}_{item_index}"), item.get("url", "N/A")
        if not text_to_split or not isinstance(text_to_split, str) or not text_to_split.strip(): logger.debug(f"Skipping item #{item_index+1} with empty text: {title}"); items_skipped_no_text += 1; continue
        try:
            chunks = text_splitter.split_text(text_to_split)
            base_id = f"{source_type}-{item_index}"
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{base_id}-chunk-{chunk_idx}"
                chunked_data.append({
                    "id": chunk_id, "text": chunk_text,
                    "metadata": {"original_source": original_gcs_source, "title": title, "url": url, "data_source_type": source_type, "chunk_index": chunk_idx, "text": chunk_text}
                })
        except Exception as e: logger.error(f"Error splitting text for item #{item_index+1} ({title}): {e}", exc_info=True); continue
    if items_skipped_no_text > 0: logger.warning(f"Skipped {items_skipped_no_text} items from {source_type} due to empty text.")
    if items_skipped_not_dict > 0: logger.warning(f"Skipped {items_skipped_not_dict} items from {source_type} (not dict).")
    logger.info(f"Generated {len(chunked_data)} chunks from source: {source_type}")
    return chunked_data


# --- Step 3: Generate Embeddings (BATCHED) ---
def generate_embeddings_batched(chunked_data, model_name=EMBEDDING_MODEL_NAME, batch_size=EMBEDDING_BATCH_SIZE):
    """Generates embeddings for text chunks IN BATCHES using HuggingFaceEmbeddings."""
    # (Using the corrected manual batching version)
    if not chunked_data: logger.warning("No data for embedding."); return [], None
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        logger.info("Embedding model initialized.")
    except Exception as e: logger.error(f"Failed to initialize embedding model: {e}", exc_info=True); return [], None

    all_embeddings = []
    total_items = len(chunked_data)
    total_batches = math.ceil(total_items / batch_size)
    logger.info(f"Generating embeddings for {total_items} chunks in {total_batches} batches...")

    try:
        for i in tqdm(range(0, total_items, batch_size), desc="Generating Embeddings", total=total_batches):
            batch_items = chunked_data[i:i + batch_size]
            batch_texts = [item.get("text", "") for item in batch_items]
            texts_for_embedding = [text for text in batch_texts if isinstance(text, str) and text.strip()]
            if not texts_for_embedding:
                logger.warning(f"Batch {i//batch_size + 1} empty. Assigning None embeddings.")
                all_embeddings.extend([None] * len(batch_texts)); continue
            # Embed only non-empty texts
            batch_embeddings = model.embed_documents(texts_for_embedding)
            # Map results back
            embedding_iter = iter(batch_embeddings)
            batch_results = [next(embedding_iter) if isinstance(text, str) and text.strip() else None for text in batch_texts]
            # Handle potential length mismatch (shouldn't happen often)
            if len(batch_results) != len(batch_texts): logger.error(f"Length mismatch in batch {i//batch_size + 1}!")
            all_embeddings.extend(batch_results)
    except Exception as e: logger.error(f"Error during embedding: {e}", exc_info=True); return [], model

    logger.info(f"Finished embedding. Received {len(all_embeddings)} results.")
    # Combine embeddings with chunks
    chunks_with_embeddings = []
    failed_count = 0
    for i, chunk_item in enumerate(chunked_data):
        emb = all_embeddings[i]
        if isinstance(emb, list) and len(emb) == EMBEDDING_DIMENSION:
            chunk_item["embedding"] = emb
            chunks_with_embeddings.append(chunk_item)
        else: logger.warning(f"Invalid embedding for chunk {chunk_item.get('id', 'N/A')}. Skipping."); failed_count += 1
    if failed_count > 0: logger.warning(f"Skipped {failed_count} chunks due to invalid embeddings.")
    logger.info(f"Returning {len(chunks_with_embeddings)} chunks with valid embeddings.")
    return chunks_with_embeddings, model


# --- Step 4: Store Chunks in Pinecone ---
def store_in_pinecone(chunks_with_embeddings, pc_client: Pinecone, target_index_name: str):
    """Stores chunked data with embeddings in the specified Pinecone index."""
    logger.info(f"Preparing to store {len(chunks_with_embeddings)} chunks in Pinecone index '{target_index_name}'...")
    if not chunks_with_embeddings: logger.warning("No valid chunks to store."); return True
    try:
        logger.info("Verifying target index exists...")
        # --- !!! THE CORRECTED CODE !!! ---
        index_list_obj = pc_client.list_indexes() # Get the IndexList object
        existing_index_names = index_list_obj.names # Access the '.names' list attribute
        # --- !!! END CORRECTION !!! ---
        logger.info(f"Found existing indexes: {existing_index_names}")
        if target_index_name not in existing_index_names: logger.error(f"Target index '{target_index_name}' not found!"); return False

        logger.info(f"Connecting to index '{target_index_name}'...")
        index: Index = pc_client.Index(target_index_name)
        try: logger.info(f"Index stats before upsert: {index.describe_index_stats()}")
        except Exception as desc_err: logger.warning(f"Could not get stats for index '{target_index_name}': {desc_err}")

        vectors_to_upsert = []
        skipped_count = 0
        logger.info("Preparing/validating vectors...")
        for chunk in chunks_with_embeddings:
            vec_id, emb, meta = chunk.get("id"), chunk.get("embedding"), chunk.get("metadata")
            if isinstance(vec_id, str) and isinstance(emb, list) and len(emb) == EMBEDDING_DIMENSION and isinstance(meta, dict):
                cleaned_meta = {str(k): (str(v)[:10000] + "..." if isinstance(v, str) and len(v.encode('utf-8')) > 40000 else v)
                                for k, v in meta.items()
                                if isinstance(v, (str, bool, int, float, list))}
                for k, v in cleaned_meta.items():
                    if isinstance(v, list): cleaned_meta[k] = [str(item) for item in v if isinstance(item, str)][:100] # Limit list length/items too
                vectors_to_upsert.append((vec_id, emb, cleaned_meta))
            else: logger.warning(f"Skipping invalid chunk: ID={vec_id}"); skipped_count += 1
        if skipped_count > 0: logger.warning(f"Skipped {skipped_count} invalid chunks.")
        if not vectors_to_upsert: logger.warning("No valid vectors prepared."); return True

        total_vectors, batch_size = len(vectors_to_upsert), PINECONE_UPSERT_BATCH_SIZE
        num_batches = math.ceil(total_vectors / batch_size)
        logger.info(f"Upserting {total_vectors} vectors in {num_batches} batches...")
        upsert_errors, total_upserted = 0, 0
        for i in tqdm(range(0, total_vectors, batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i : i + batch_size]
            if not batch: continue
            try:
                upsert_response = index.upsert(vectors=batch)
                count = getattr(upsert_response, 'upserted_count', 0) or 0
                total_upserted += count
                if count != len(batch): logger.warning(f"Batch {i//batch_size+1} upsert count ({count}) != size ({len(batch)}).")
            except Exception as upsert_err: logger.error(f"Error upserting batch {i//batch_size+1}: {upsert_err}", exc_info=True); upsert_errors += 1
        logger.info(f"Finished upserting. Total reported: {total_upserted}. Errors: {upsert_errors}.")
        try: logger.info(f"Final index stats: {index.describe_index_stats()}")
        except Exception as e: logger.warning(f"Could not get final stats: {e}")
        return upsert_errors == 0
    except Exception as e: logger.error(f"Critical error during Pinecone storage: {e}", exc_info=True); return False


# --- Step 5: Query Pinecone ---
def query_pinecone(query: str, model, pc_client: Pinecone, target_index_name: str):
    """Queries the Pinecone index with the given text query."""
    # --- (Keep this function as previously corrected) ---
    if not model: logger.error("Query fail: Embedding model missing."); return None
    if not pc_client: logger.error("Query fail: Pinecone client missing."); return None
    if not query or not isinstance(query, str) or not query.strip(): logger.warning(f"Invalid query: '{query}'."); return None
    logger.info(f"Querying index '{target_index_name}' for: '{query[:100]}...'")
    try:
        query_embedding = model.embed_query(query)
        if not isinstance(query_embedding, list) or len(query_embedding) != EMBEDDING_DIMENSION: logger.error("Failed to generate valid query embedding."); return None
    except Exception as e: logger.error(f"Query embed error: {e}", exc_info=True); return None
    try:
        index: Index = pc_client.Index(target_index_name)
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        logger.info("Query successful.")
        logger.info("\n--- Matches ---")
        matches = result.get('matches', [])
        if not matches: logger.info("No matches found.")
        else:
            for i, match in enumerate(matches):
                meta = match.get('metadata', {})
                logger.info(f"Match {i+1}: Score={match.get('score', 'N/A'):.4f} ID={match.get('id', 'N/A')}")
                logger.info(f"  Src: {meta.get('original_source', 'N/A')}, Title: {meta.get('title', 'N/A')}")
                logger.info(f"  Text: {meta.get('text', 'N/A')[:200]}...")
        return result
    except Exception as e: logger.error(f"Pinecone query error: {e}", exc_info=True); return None


# --- Step 6: Main Execution Function for Airflow ---
def run_chunk_embed_store_pipeline(**kwargs):
    """
    Main function called by Airflow. Orchestrates GCS load, chunking, embedding, Pinecone storage.
    Raises exceptions on critical failure.
    """
    # Use Airflow task logger if available
    try: task_log = logging.getLogger("airflow.task"); task_log.info("Using Airflow task logger.")
    except Exception: task_log = logger; task_log.warning("Using module logger as fallback.")

    task_log.info("--- Starting Chunk, Embed, Store Pipeline (GCS -> Pinecone) ---")
    start_time = time.time()

    # Validate GCS Utils
    if not gcs_utils_available: raise RuntimeError("GCS Utils failed to import, cannot read data.")
    # Validate Pinecone API Key
    if not PINECONE_API_KEY: task_log.error("CRITICAL: PINECONE_API_KEY missing."); raise ValueError("PINECONE_API_KEY required.")

    # Initialize Pinecone client
    try:
        task_log.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc.list_indexes() # Confirm connection
        task_log.info("Pinecone client initialized.")
    except Exception as e: task_log.error(f"CRITICAL: Failed Pinecone init: {e}", exc_info=True); raise RuntimeError("Pinecone init failed.") from e

    target_pinecone_index = INDEX_NAME # Use constant derived from env
    task_log.info(f"Targeting Pinecone index: '{target_pinecone_index}'")

    # Phase 1: Load and Chunk
    task_log.info("--- Phase 1: Loading and Chunking Data ---")
    load_chunk_start = time.time()
    all_chunks = []
    for source_type, blob_name in INPUT_FILES_TO_PROCESS.items():
        path = f"gs://{PROCESSED_BUCKET}/{blob_name}"
        task_log.info(f"Processing: {path}")
        data = read_json_from_gcs(PROCESSED_BUCKET, blob_name)
        if data is not None and isinstance(data, list) and data:
            chunks = split_text(data, source_type=source_type)
            if chunks: all_chunks.extend(chunks); task_log.info(f"Added {len(chunks)} chunks from '{source_type}'.")
            else: task_log.warning(f"No chunks from '{source_type}'.")
        else: task_log.warning(f"Skipping '{source_type}': Invalid/empty data from GCS.")
    task_log.info(f"--- Phase 1 Finished ({time.time() - load_chunk_start:.2f}s). Total chunks: {len(all_chunks)} ---")
    if not all_chunks: task_log.warning("No chunks generated."); return

    # Phase 2: Generate Embeddings
    task_log.info("--- Phase 2: Generating Embeddings ---")
    embed_start = time.time()
    chunks_with_embeds, embed_model = generate_embeddings_batched(all_chunks)
    task_log.info(f"--- Phase 2 Finished ({time.time() - embed_start:.2f}s). Valid embeddings: {len(chunks_with_embeds)} ---")
    if embed_model is None or not chunks_with_embeds: task_log.error("Embedding failed."); raise RuntimeError("Embedding failed.")

    # Phase 3: Store in Pinecone
    task_log.info("--- Phase 3: Storing in Pinecone ---")
    store_start = time.time()
    print("target pinecone index", target_pinecone_index)
    store_success = store_in_pinecone(chunks_with_embeds, pc, target_pinecone_index)
    task_log.info(f"--- Phase 3 Finished ({time.time() - store_start:.2f}s). Success: {store_success} ---")
    if not store_success: task_log.error("Pinecone storage failed."); raise RuntimeError("Pinecone storage failed.")

    # Phase 4: Example Query (Optional)
    task_log.info("--- Phase 4: Running Example Query ---")
    query_start = time.time()
    query_pinecone("How to improve pull-ups?", embed_model, pc, target_pinecone_index)
    task_log.info(f"--- Example Query Finished ({time.time() - query_start:.2f}s) ---")

    task_log.info(f"--- Pipeline finished successfully in {time.time() - start_time:.2f} seconds ---")


# --- Main Guard ---
if __name__ == "__main__":
    logger.info("Running vector_db.py script directly for testing...")
    run_chunk_embed_store_pipeline() # Call directly