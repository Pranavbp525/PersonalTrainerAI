import os
import json
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)  # Inherit global logger

if not logger.handlers:
    # Ensure logs are written to 'scraper.log' from pdfs.py
    file_handler = logging.FileHandler(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/vectordb.log")), mode='a')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


# Pinecone API key from .env
API_KEY = os.getenv('pinecone_api_key')
INDEX_NAME = "fitness-bot"
REGION = "us-east-1"

#  Initialize Pinecone
pc = Pinecone(api_key=API_KEY, environment=REGION)

# JSON File Paths (Inside data/raw_json_data/)
json_files = {
    "pdf": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/pdf_data.json")),
    "ms": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/ms_data.json")),
    "articles": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/blogs.json"))
}


# **Step 1: Load JSON Data**
def load_json_data(file_path):
    """Loads JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.info(f"Successfully loaded JSON data from {file_path}")
        return data
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
    for item in data:
        text = item.get("description", "")  # Use "description" field instead of "transcript"
        source = item.get("source", "")
        title = item.get("title", "")
        url = item.get("url", "")

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Store each chunk separately
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "source": source,
                "title": title,
                "url": url,
                "chunk_id": f"{source}_{idx}",  # Unique ID for each chunk
                "chunk": chunk
            })

    logger.info(f"Total Chunks Created: {len(chunked_data)}")
    return chunked_data


# **Step 3: Generate Embeddings**
def generate_embeddings(chunked_data, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Generates embeddings for each text chunk."""
    model = HuggingFaceEmbeddings(model_name=model_name)

    for chunk in tqdm(chunked_data, desc="Generating Embeddings", unit="chunk"):
        chunk["embedding"] = model.embed_documents([chunk["chunk"]])[0]

    logger.info("Chunked embeddings generated successfully!")
    return chunked_data, model


# **Step 4: Store Chunks in Pinecone**
def store_in_pinecone(chunked_data):
    """Stores chunked embeddings in Pinecone."""
    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=REGION)
        )

    # Connect to index
    index = pc.Index(INDEX_NAME)

    # Prepare data for Pinecone (Include `title` and `source` in metadata)
    vectors = [
        (
            chunk["chunk_id"],  # Unique ID
            chunk["embedding"],  # Embedding vector
            {
                "text": chunk["chunk"],  # Main chunk of text
                "title": chunk["title"],  # Include title
                "source": chunk["source"],  # Include source
                "url": chunk["url"]
            }
        )
        for chunk in chunked_data
    ]

    # Wait for index initialization
    while pc.describe_index(INDEX_NAME).status['ready'] is False:
        logger.info("Waiting for index to be ready...")
        time.sleep(5)

    # Upload in batches (to avoid rate limits)
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i+batch_size])

    logger.info("Chunked data uploaded to Pinecone successfully!")


# **Step 5: Query Pinecone**
def query_pinecone(query, model):
    """Retrieves relevant chunks from Pinecone using embeddings."""
    query_embedding = model.embed_query(query)

    # Search Pinecone
    index = pc.Index(INDEX_NAME)
    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Display results
    logger.info("\n Relevant Transcript Chunks:\n")
    for match in result['matches']:
        print(f"{match['metadata']}\n")


# **Step 6: Main Execution**
def chunk_to_db():
    all_chunked_data = []
    #model = None

    # Process each JSON file separately
    for key, json_path in json_files.items():
        logger.info(f"Processing JSON file: {json_path}")

        #  Load JSON Data
        data = load_json_data(json_path)
        if not data:
            continue  # Skip to the next file if data loading fails

        # Split Text into Chunks
        chunked_data = split_text(data)
        all_chunked_data.extend(chunked_data)  # Append to all data

    # Generate Embeddings for All Data
    if all_chunked_data:
        all_chunked_data, model = generate_embeddings(all_chunked_data)

        # Store Embeddings in Pinecone
        store_in_pinecone(all_chunked_data)

        # Query Pinecone Example
        query_pinecone("How to improve pull-ups?", model)
    else:
        logger.warning("No data was processed from JSON files.")

# model_name="sentence-transformers/all-mpnet-base-v2"
# model = HuggingFaceEmbeddings(model_name=model_name)
# query_pinecone("How to improve pull-ups?", model)