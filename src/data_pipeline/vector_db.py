import os
import json
import logging
import time
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import math # Added for calculating batches

# Load environment variables
load_dotenv()

# Setup logger (ensure logs/ directory exists)
logger = logging.getLogger(__name__)
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/vectordb.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# --- Constants ---
INDEX_NAME = "fitness-bot"
REGION = "us-east-1" # Or your specific region
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32 # Adjust based on available memory (32 is often a good starting point)
PINECONE_UPSERT_BATCH_SIZE = 100 # Pinecone recommendation

json_files = {
    "pdf": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/pdf_data.json")),
    "ms": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/ms_data.json")),
    "articles": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/blogs.json"))
}

# **Step 1: Load JSON Data** - (Unchanged from your version)
def load_json_data(file_path):
    """Loads JSON data from a file."""
    if not os.path.exists(file_path):
        logger.error(f"JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.info(f"Successfully loaded JSON data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


# **Step 2: Split Text into Chunks** - (Unchanged from your version)
def split_text(data, chunk_size=400, chunk_overlap=50):
    """Splits text into manageable chunks while preserving context."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", "?", "!", " "],
    )
    chunked_data = []
    if not isinstance(data, list):
        logger.error(f"Expected a list of items for splitting, got {type(data)}")
        return chunked_data
    for item in data:
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dictionary item in data: {item}")
            continue
        text = item.get("description", "")
        source = item.get("source", "")
        title = item.get("title", "")
        url = item.get("url", "")
        if not text:
            logger.warning(f"Skipping item with empty description: source={source}, title={title}")
            continue
        try:
            chunks = text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error splitting text for item (source={source}, title={title}): {e}")
            continue
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "source": source, "title": title, "url": url,
                "chunk_id": f"{source}_{title}_{idx}", "chunk": chunk
            })
    logger.info(f"Total Chunks Created from input data: {len(chunked_data)}")
    return chunked_data


# **Step 3: Generate Embeddings (BATCHED)** - *** MODIFIED ***
def generate_embeddings_batched(chunked_data, model_name=EMBEDDING_MODEL_NAME, batch_size=EMBEDDING_BATCH_SIZE):
    """Generates embeddings for text chunks IN BATCHES."""
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        # Consider device='cuda' if GPU is available in the execution environment
        # You would need torch with CUDA support installed and nvidia drivers configured.
        # For CPU: device=None or device='cpu' is default.
        model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        logger.info("Embedding model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name}: {e}")
        return None, None

    texts_to_embed = [item.get("chunk", "") for item in chunked_data]
    all_embeddings = []
    total_batches = math.ceil(len(texts_to_embed) / batch_size)

    logger.info(f"Generating embeddings for {len(texts_to_embed)} texts in batches of {batch_size}...")

    try:
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Embeddings", total=total_batches):
            batch_texts = texts_to_embed[i:i + batch_size]
            # Filter out empty strings in the batch if any, though split_text should prevent this
            batch_texts_non_empty = [text for text in batch_texts if text]
            if not batch_texts_non_empty:
                # Handle case where a whole batch might be empty (unlikely)
                all_embeddings.extend([None] * len(batch_texts))
                continue

            # Generate embeddings for the non-empty texts in the batch
            batch_embeddings = model.embed_documents(batch_texts_non_empty)

            # Reconstruct the batch result including None for empty inputs
            embedding_iter = iter(batch_embeddings)
            batch_results = [next(embedding_iter) if text else None for text in batch_texts]
            all_embeddings.extend(batch_results)

            # Optional: Add a small sleep between batches if hitting CPU limits intensely
            # time.sleep(0.1)

    except Exception as e:
        logger.error(f"Error during batch embedding generation: {e}", exc_info=True)
        # Depending on error, you might want to return partial results or fail completely
        return None, None # Fail completely on error for simplicity

    logger.info(f"Generated {len(all_embeddings)} embedding results (including potential Nones).")

    # Add embeddings back to the chunked data, handling potential None values
    processed_chunks = []
    failed_count = 0
    for i, chunk_item in enumerate(chunked_data):
        if i < len(all_embeddings) and all_embeddings[i] is not None:
            chunk_item["embedding"] = all_embeddings[i]
            processed_chunks.append(chunk_item)
        else:
            logger.warning(f"Embedding failed or missing for chunk ID {chunk_item.get('chunk_id')}")
            failed_count += 1

    if failed_count > 0:
         logger.warning(f"Total chunks where embedding failed or was skipped: {failed_count}")

    logger.info(f"Returning {len(processed_chunks)} chunks with successful embeddings.")
    return processed_chunks, model


# **Step 4: Store Chunks in Pinecone** - (Unchanged from your updated version)
def store_in_pinecone(chunked_data, pc_client: Pinecone, index_name: str, region: str):
    """Stores chunked embeddings in Pinecone."""
    logger.info(f"Preparing to store {len(chunked_data)} chunks in Pinecone index '{index_name}'...")
    if not chunked_data:
        logger.warning("No chunked data with embeddings to store.")
        return
    try:
        existing_indexes = pc_client.list_indexes().names
        logger.info(f"Existing Pinecone indexes: {existing_indexes}")
        if index_name not in existing_indexes:
            logger.warning(f"Index '{index_name}' not found. Creating new index...")
            try:
                pc_client.create_index(
                    name=index_name, dimension=768, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=region)
                )
                logger.info("Waiting for new index to be ready...")
                while not pc_client.describe_index(index_name).status['ready']:
                    time.sleep(5)
                logger.info(f"Index '{index_name}' created and ready.")
            except Exception as create_err:
                logger.error(f"Failed to create Pinecone index '{index_name}': {create_err}")
                raise
        try:
            index = pc_client.Index(index_name)
            logger.info(f"Connected to Pinecone index '{index_name}'.")
        except Exception as connect_err:
            logger.error(f"Failed to connect to Pinecone index '{index_name}': {connect_err}")
            raise

        vectors_to_upsert = []
        for chunk in chunked_data:
             if chunk.get("embedding") and chunk.get("chunk_id"):
                  vectors_to_upsert.append(
                      (str(chunk["chunk_id"]), chunk["embedding"],
                       {"text": chunk.get("chunk", ""), "title": chunk.get("title", ""),
                        "source": chunk.get("source", ""), "url": chunk.get("url", "")})
                  )
             else:
                  logger.warning(f"Skipping invalid chunk during vector preparation: {chunk.get('chunk_id')}")
        if not vectors_to_upsert:
            logger.error("No valid vectors prepared for upserting.")
            return

        batch_size = PINECONE_UPSERT_BATCH_SIZE
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors in batches of {batch_size}...")
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
            except Exception as upsert_err:
                logger.error(f"Error upserting batch starting at index {i}: {upsert_err}")
        logger.info(f"Finished upserting vectors to Pinecone index '{index_name}'.")
        logger.info(f"Final index stats: {index.describe_index_stats()}") # Log stats after upsert
    except Exception as e:
        logger.error(f"An error occurred during Pinecone storage process: {e}")


# **Step 5: Query Pinecone** - (Unchanged from your updated version)
def query_pinecone(query: str, model, pc_client: Pinecone, index_name: str):
    """Retrieves relevant chunks from Pinecone using embeddings."""
    if not model: logger.error("Cannot query: Embedding model not available."); return None
    if not pc_client: logger.error("Cannot query: Pinecone client not available."); return None
    logger.info(f"Generating query embedding for: '{query}'")
    try: query_embedding = model.embed_query(query)
    except Exception as e: logger.error(f"Error generating query embedding: {e}"); return None
    logger.info(f"Querying Pinecone index '{index_name}'...")
    try:
        index = pc_client.Index(index_name)
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        logger.info("Pinecone query successful.")
        logger.info("\nRelevant Context Chunks Found:\n")
        for match in result.get('matches', []):
            metadata = match.get('metadata', {}); text = metadata.get('text', 'N/A')
            title = metadata.get('title', 'N/A'); source = metadata.get('source', 'N/A')
            score = match.get('score', 'N/A')
            logger.info(f"Score: {score:.4f} | Source: {source} | Title: {title}\nText: {text}\n---")
        return result
    except Exception as e: logger.error(f"Error querying Pinecone index '{index_name}': {e}"); return None


# **Step 6: Main Execution** - *** MODIFIED ***
def chunk_to_db():
    """
    Main function orchestrating loading, chunking, embedding, and storage.
    Called by the Airflow DAG task.
    """
    logger.info("--- Starting chunk_to_db process ---")
    start_time = time.time()

    # --- Initialize Pinecone Client ---
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not found.")
        raise ValueError("PINECONE_API_KEY is required but not set.")
    try:
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        logger.info("Pinecone client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
        raise

    all_data_to_embed = []
    model = None # Initialize embedding model variable

    # --- Phase 1: Load and Chunk All Data ---
    logger.info("--- Phase 1: Loading and Chunking Data ---")
    load_chunk_start = time.time()
    for key, json_path in json_files.items():
        logger.info(f"Processing source: {key} | File: {json_path}")
        data = load_json_data(json_path)
        if data:
            chunked_data = split_text(data)
            if chunked_data:
                all_data_to_embed.extend(chunked_data)
                logger.info(f"Added {len(chunked_data)} chunks from source '{key}'.")
            else: logger.warning(f"No chunks generated for source '{key}'.")
        else: logger.warning(f"Skipping source '{key}' due to data loading issues.")
    load_chunk_end = time.time()
    logger.info(f"--- Phase 1 Finished ({load_chunk_end - load_chunk_start:.2f} seconds) ---")
    logger.info(f"Total chunks to process: {len(all_data_to_embed)}")

    if not all_data_to_embed:
        logger.warning("No data loaded or chunked. Exiting.")
        return

    # --- Phase 2: Generate Embeddings in Batches ---
    logger.info("--- Phase 2: Generating Embeddings ---")
    embedding_start = time.time()
    # *** Call the BATCHED embedding function ***
    chunks_with_embeddings, model = generate_embeddings_batched(all_data_to_embed)
    embedding_end = time.time()
    logger.info(f"--- Phase 2 Finished ({embedding_end - embedding_start:.2f} seconds) ---")


    if not chunks_with_embeddings:
        logger.error("Embedding generation failed or yielded no results. Cannot proceed to storage.")
        return # Stop if embeddings failed

    # --- Phase 3: Store Embeddings in Pinecone ---
    logger.info("--- Phase 3: Storing in Pinecone ---")
    storage_start = time.time()
    store_in_pinecone(chunks_with_embeddings, pc, INDEX_NAME, REGION)
    storage_end = time.time()
    logger.info(f"--- Phase 3 Finished ({storage_end - storage_start:.2f} seconds) ---")


    # Optional: Query Pinecone Example after storage
    if model: # Ensure model is available
        logger.info("--- Running example Pinecone query ---")
        query_start = time.time()
        query_pinecone("How to improve pull-ups?", model, pc, INDEX_NAME)
        query_end = time.time()
        logger.info(f"--- Example Query Finished ({query_end - query_start:.2f} seconds) ---")

    end_time = time.time()
    logger.info(f"--- chunk_to_db process finished successfully in {end_time - start_time:.2f} seconds ---")


# --- Main Guard ---
if __name__ == "__main__":
    logger.info("Running vector_db.py script directly...")
    chunk_to_db()
    logger.info("Direct script execution finished.")