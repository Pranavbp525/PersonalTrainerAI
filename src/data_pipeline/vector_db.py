import os
import json
import logging
import time # Added for waiting
from dotenv import load_dotenv
from tqdm import tqdm
# Correct import for newer pinecone-client versions (now just 'pinecone')
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Ensure HuggingFaceEmbeddings is imported from community
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables - okay here for constants, but API keys read later
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/vectordb.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # Ensure log directory exists

if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


# --- Constants (Safe at module level) ---
INDEX_NAME = "fitness-bot"
# Define REGION here if needed by ServerlessSpec, or read from env inside function
REGION = "us-east-1" # Example region for AWS serverless - adjust if needed

# JSON File Paths (Inside data/raw_json_data/) - Ensure these paths are correct relative to execution
# Using relative paths from the script location might be fragile in Airflow. Consider absolute paths or env vars.
json_files = {
    "pdf": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/pdf_data.json")),
    "ms": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/ms_data.json")),
    "articles": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/blogs.json"))
}

# --- REMOVED Pinecone Initialization from Module Level ---
# API_KEY = os.getenv('PINECONE_API_KEY') # Read inside function now
# pc = Pinecone(api_key=API_KEY) # Initialize inside function now


# **Step 1: Load JSON Data**
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


# **Step 2: Split Text into Chunks**
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
        # Check if item is a dictionary before accessing keys
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

        # Split text into chunks
        try:
            chunks = text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error splitting text for item (source={source}, title={title}): {e}")
            continue

        # Store each chunk separately
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "source": source,
                "title": title,
                "url": url,
                "chunk_id": f"{source}_{title}_{idx}", # Made slightly more unique
                "chunk": chunk
            })

    logger.info(f"Total Chunks Created from input data: {len(chunked_data)}")
    return chunked_data


# **Step 3: Generate Embeddings**
def generate_embeddings(chunked_data, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Generates embeddings for each text chunk."""
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        model = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embedding model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name}: {e}")
        return None, None # Return None for model if failed

    embeddings_generated = 0
    embeddings_failed = 0
    for chunk_item in tqdm(chunked_data, desc="Generating Embeddings", unit="chunk"):
        try:
            # Ensure 'chunk' key exists and is not empty
            text_to_embed = chunk_item.get("chunk")
            if text_to_embed:
                chunk_item["embedding"] = model.embed_documents([text_to_embed])[0]
                embeddings_generated += 1
            else:
                logger.warning(f"Skipping embedding generation for empty chunk: ID={chunk_item.get('chunk_id')}")
                chunk_item["embedding"] = None # Mark as None if chunk was empty
                embeddings_failed +=1
        except Exception as e:
            logger.error(f"Error generating embedding for chunk ID {chunk_item.get('chunk_id')}: {e}")
            chunk_item["embedding"] = None # Mark as None on error
            embeddings_failed += 1

    if embeddings_failed > 0:
         logger.warning(f"Failed to generate embeddings for {embeddings_failed} chunks.")
    logger.info(f"Embeddings generated successfully for {embeddings_generated} chunks.")
    # Filter out chunks where embedding failed before returning
    chunked_data_with_embeddings = [c for c in chunked_data if c.get("embedding") is not None]
    return chunked_data_with_embeddings, model


# **Step 4: Store Chunks in Pinecone**
# MODIFIED: Accepts Pinecone client (pc_client) and index_name/region as arguments
def store_in_pinecone(chunked_data, pc_client: Pinecone, index_name: str, region: str):
    """Stores chunked embeddings in Pinecone."""
    logger.info(f"Preparing to store {len(chunked_data)} chunks in Pinecone index '{index_name}'...")
    if not chunked_data:
        logger.warning("No chunked data with embeddings to store.")
        return

    try:
        # List existing indexes
        existing_indexes = pc_client.list_indexes().names
        logger.info(f"Existing Pinecone indexes: {existing_indexes}")

        # Create index if it doesn't exist
        if index_name not in existing_indexes:
            logger.warning(f"Index '{index_name}' not found. Creating new index...")
            try:
                pc_client.create_index(
                    name=index_name,
                    dimension=768, # Assuming embedding dimension is 768
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=region) # Adjust cloud/region if needed
                )
                # Wait for index initialization
                logger.info("Waiting for new index to be ready...")
                while not pc_client.describe_index(index_name).status['ready']:
                    time.sleep(5)
                logger.info(f"Index '{index_name}' created and ready.")
            except Exception as create_err:
                logger.error(f"Failed to create Pinecone index '{index_name}': {create_err}")
                raise # Re-raise error if index creation fails

        # Connect to index
        try:
            index = pc_client.Index(index_name)
            logger.info(f"Connected to Pinecone index '{index_name}'.")
            # Optional: Log index stats
            # logger.info(index.describe_index_stats())
        except Exception as connect_err:
            logger.error(f"Failed to connect to Pinecone index '{index_name}': {connect_err}")
            raise # Re-raise error if connection fails

        # Prepare data for Pinecone (Ensure only valid chunks are included)
        vectors_to_upsert = []
        for chunk in chunked_data:
             # Double-check that embedding exists and chunk_id is valid
             if chunk.get("embedding") and chunk.get("chunk_id"):
                  vectors_to_upsert.append(
                      (
                          str(chunk["chunk_id"]), # Ensure ID is string
                          chunk["embedding"],
                          { # Metadata
                              "text": chunk.get("chunk", ""),
                              "title": chunk.get("title", ""),
                              "source": chunk.get("source", ""),
                              "url": chunk.get("url", "")
                          }
                      )
                  )
             else:
                  logger.warning(f"Skipping invalid chunk during vector preparation: {chunk.get('chunk_id')}")

        if not vectors_to_upsert:
            logger.error("No valid vectors prepared for upserting.")
            return

        # Upload in batches
        batch_size = 100 # Pinecone recommended batch size
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors in batches of {batch_size}...")
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
                # logger.debug(f"Upserted batch {i//batch_size + 1}") # Optional debug log
            except Exception as upsert_err:
                logger.error(f"Error upserting batch starting at index {i}: {upsert_err}")
                # Optionally add retry logic or skip the batch

        logger.info(f"Finished upserting vectors to Pinecone index '{index_name}'.")
        # Optional: Log final index stats
        # logger.info(index.describe_index_stats())

    except Exception as e:
        logger.error(f"An error occurred during Pinecone storage process: {e}")
        # Avoid raising here maybe, log error and continue if possible? Depends on desired flow.

# **Step 5: Query Pinecone**
# MODIFIED: Accepts Pinecone client (pc_client) and index_name as arguments
def query_pinecone(query: str, model, pc_client: Pinecone, index_name: str):
    """Retrieves relevant chunks from Pinecone using embeddings."""
    if not model:
        logger.error("Cannot query Pinecone: Embedding model not available.")
        return None
    if not pc_client:
        logger.error("Cannot query Pinecone: Pinecone client not available.")
        return None

    logger.info(f"Generating query embedding for: '{query}'")
    try:
        query_embedding = model.embed_query(query)
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return None

    logger.info(f"Querying Pinecone index '{index_name}'...")
    try:
        index = pc_client.Index(index_name)
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        logger.info("Pinecone query successful.")

        # Display results
        logger.info("\nRelevant Context Chunks Found:\n")
        for match in result.get('matches', []):
            metadata = match.get('metadata', {})
            text = metadata.get('text', 'N/A')
            title = metadata.get('title', 'N/A')
            source = metadata.get('source', 'N/A')
            score = match.get('score', 'N/A')
            logger.info(f"Score: {score:.4f} | Source: {source} | Title: {title}\nText: {text}\n---")
        return result

    except Exception as e:
        logger.error(f"Error querying Pinecone index '{index_name}': {e}")
        return None


# **Step 6: Main Execution** - Function called by Airflow
def chunk_to_db():
    """
    Main function orchestrating loading, chunking, embedding, and storage.
    Called by the Airflow DAG task.
    """
    logger.info("--- Starting chunk_to_db process ---")

    # --- Initialize Pinecone Client INSIDE the task function ---
    load_dotenv() # Load .env file within the task execution context
    api_key = os.getenv("PINECONE_API_KEY")
    # Read environment if needed by Pinecone ServerlessSpec or older init
    # environment = os.getenv("PINECONE_ENVIRONMENT")

    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not found.")
        raise ValueError("PINECONE_API_KEY is required but not set.")

    try:
        logger.info("Initializing Pinecone client...")
        # Use the correct initialization based on your library version
        pc = Pinecone(api_key=api_key)
        # Example for older init style if needed:
        # from pinecone import init as pinecone_init
        # pinecone_init(api_key=api_key, environment=environment) # Use if environment needed
        logger.info("Pinecone client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
        raise # Fail the task if client cannot be initialized

    all_chunked_data_processed = []
    model = None # Initialize model variable

    # Process each JSON file separately
    for key, json_path in json_files.items():
        logger.info(f"--- Processing source: {key} | File: {json_path} ---")

        data = load_json_data(json_path)
        if not data:
            logger.warning(f"Skipping source '{key}' due to data loading issues.")
            continue

        chunked_data = split_text(data)
        if not chunked_data:
            logger.warning(f"No chunks generated for source '{key}'.")
            continue

        # Generate Embeddings only if chunks exist
        # Reuse model if already initialized
        if model is None:
            logger.info("Initializing embedding model for the first time...")
            chunked_data_with_embeddings, model = generate_embeddings(chunked_data)
        else:
            logger.info("Reusing existing embedding model...")
            chunked_data_with_embeddings, _ = generate_embeddings(chunked_data, model_name=model.model_name) # Pass model name for consistency

        if not chunked_data_with_embeddings:
             logger.warning(f"No embeddings generated successfully for source '{key}'.")
             continue

        # Extend the main list
        all_chunked_data_processed.extend(chunked_data_with_embeddings)
        logger.info(f"Processed {len(chunked_data_with_embeddings)} chunks for source '{key}'.")

    # Store All Embeddings in Pinecone if any were processed
    if all_chunked_data_processed:
        logger.info(f"--- Starting Pinecone storage for {len(all_chunked_data_processed)} total chunks ---")
        store_in_pinecone(all_chunked_data_processed, pc, INDEX_NAME, REGION) # Pass pc client

        # Optional: Query Pinecone Example after storage
        logger.info("--- Running example Pinecone query ---")
        query_pinecone("How to improve pull-ups?", model, pc, INDEX_NAME) # Pass pc client
    else:
        logger.warning("No data was successfully processed and embedded from any source files.")

    logger.info("--- chunk_to_db process finished ---")

# --- Main Execution Guard (for potential direct script execution, though DAG calls chunk_to_db) ---
if __name__ == "__main__":
    logger.info("Running vector_db.py script directly...")
    chunk_to_db()
    logger.info("Direct script execution finished.")