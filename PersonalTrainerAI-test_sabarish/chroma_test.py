import chromadb
from sentence_transformers import SentenceTransformer

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="fitness_chatbot")

# Load the exact same embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define query text - Enter ur query here
query_text = "how to build calf muscle?"
query_embedding = embedding_model.encode(query_text).tolist()

# Retrieve most similar results
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# Print results
print("\n🔍 **Query Results:**")
for idx, doc in enumerate(results["metadatas"][0]):
    print(f"\nResult {idx+1}:")
    print(f"🔹 **Source:** {doc['source']}")
    print(f"🔹 **Original ID:** {doc['original_id']}")
    print(f"🔹 **Text:** {doc['text']}...")  
