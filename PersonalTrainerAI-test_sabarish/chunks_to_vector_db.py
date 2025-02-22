import os
import torch
import json
import chromadb
from sentence_transformers import SentenceTransformer
import re
import multiprocessing

# Define directories
input_dir = "scraped_data/preprocessed_json_data"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_chatbot")

# Load embedding model (Sentence Transformers) with GPU if available
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

# List of JSON files to process
json_files = ["articles.json", "blogs.json", "pdf_data.json", "youtube_transcripts.json"]

# Enhanced Chunking function with duplicate sentence removal
def chunk_text(text, max_tokens=300, overlap=50):
    """Splits text into chunks while ensuring no duplicate sentences across chunks."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence boundaries
    seen_sentences = set()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_length = len(sentence.split())
        
        if sentence in seen_sentences:
            continue  # Skip duplicate sentences
        seen_sentences.add(sentence)
        
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Retain overlap for continuity
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function to process and store chunks
def process_file(file):
    file_path = os.path.join(input_dir, file)
    
    # Skip if file does not exist
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping {file}, file not found.")
        return

    # Load JSON data
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine the key to chunk
    text_key = "text" if "youtube" not in file else "transcript"

    # Process and store chunks
    embeddings_data = []
    for entry in data:
        if text_key in entry:
            chunks = chunk_text(entry[text_key])

            # Embed each chunk and prepare for bulk insertion
            for idx, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk).tolist()
                doc_id = f"{entry.get('url', 'doc')}_{idx}"  # Unique ID

                embeddings_data.append((doc_id, embedding, {"source": file, "original_id": entry.get("url", "unknown"), "text": chunk}))
    
    # Bulk insert into ChromaDB
    if embeddings_data:
        collection.add(
            ids=[doc[0] for doc in embeddings_data],
            embeddings=[doc[1] for doc in embeddings_data],
            metadatas=[doc[2] for doc in embeddings_data]
        )

# Use multiprocessing to speed up processing
if __name__ == "__main__":
    with multiprocessing.Pool(processes=min(4, len(json_files))) as pool:
        pool.map(process_file, json_files)

print("✅ Data chunked, embedded, and stored in ChromaDB!")
