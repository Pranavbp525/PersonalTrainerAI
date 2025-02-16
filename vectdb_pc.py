from pinecone import Pinecone, ServerlessSpec
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# 🔹 Step 1: Load JSON Data
json_file_path = r"data/preprocessed_json_data/youtube_transcripts.json"  # Replace with your actual file path

with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 🔹 Step 2: Initialize LangChain's Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Adjust based on model limit
    chunk_overlap=50,  # Overlapping to maintain context
    separators=["\n\n", ".", "?", "!", " "],  # Break at logical points
)

# 🔹 Step 3: Process and Chunk Transcripts
chunked_data = []
for item in data:
    #print(item)
    transcript = item.get("transcript", "")
    video_id = item.get("video_id", "")
    title = item.get("title", "")

    # Split transcript into smaller chunks
    chunks = text_splitter.split_text(transcript)

    # Store each chunk separately
    for idx, chunk in enumerate(chunks):
        chunked_data.append({
            "video_id": f"{video_id}_{idx}",  # Unique ID for each chunk
            "chunk": chunk,
            "title": title
        })

print(f"Total Chunks Created: {len(chunked_data)}")

# 🔹 Step 4: Generate Embeddings for Each Chunk
# Load embedding model
#model = SentenceTransformer("intfloat/multilingual-e5-large")
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Generate embeddings
for chunk in tqdm(chunked_data, desc="Generating Embeddings", unit="chunk"):
    chunk["embedding"] = model.embed_documents([chunk["chunk"]])[0]


print("Chunked embeddings generated successfully!")

# 🔹 Step 5: Store Chunked Embeddings in Pinecone
API_KEY = os.getenv('pinecone_api_key')
# Initialize Pinecone
pc = Pinecone(api_key= API_KEY, environment="us-east-1")  # Replace with your values
index_name = "fitness-chatbot"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws",
        region="us-east-1"))  # Change from 1024 → 768


# Connect to index
index = pc.Index(index_name)

# Prepare data for Pinecone
vectors = [(chunk["video_id"], chunk["embedding"], {"text": chunk["chunk"]}) for chunk in chunked_data]

# Upload in batches (to avoid rate limits)
batch_size = 50
for i in range(0, len(vectors), batch_size):
    index.upsert(vectors[i:i+batch_size])

print("Chunked data uploaded to Pinecone successfully!")

# 🔹 Step 6: Query Pinecone for Relevant Chunks
def query_pinecone(query):
    """Retrieve relevant chunks from Pinecone using embeddings."""
    query_embedding = model.embed_query(query)


    # Search Pinecone
    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Display results
    print("\n🔍 Relevant Transcript Chunks:\n")
    for match in result['matches']:
        print(f"{match['metadata']['text']}\n")

# Example query
query_pinecone("How to improve pull-ups?")