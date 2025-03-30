import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Load Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = "fitness-chatbot"

# Define keyword patterns for bias detection
GENDER_KEYWORDS = [
    r"\b(male|female|men|women|girls|boys)\b",
    r"\b(for\s+(men|women|ladies|guys))\b"
]
INTENSITY_KEYWORDS = [
    r"\b(hardcore|extreme|killer|fat[-\s]?burn(ing)?|no pain no gain|shred(ded)?)\b",
    r"\b(advanced only|not for beginners)\b"
]

def detect_bias_in_text(text):
    gender_matches = []
    intensity_matches = []

    for pattern in GENDER_KEYWORDS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        gender_matches.extend(matches)

    for pattern in INTENSITY_KEYWORDS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        intensity_matches.extend(matches)

    return {
        "gender_bias": list(set(gender_matches)),
        "intensity_bias": list(set(intensity_matches))
    }


def scan_pinecone_chunks():
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pc.Index(INDEX_NAME)

    print(f"Scanning index: {INDEX_NAME}")
    bias_report = []

    # Assuming you want to scan the whole index
    # Use fetch with pagination if large
    response = index.describe_index_stats()
    total_vectors = response.get("total_vector_count", 0)
    print(f"Total vectors: {total_vectors}")

    batch_size = 100
    offset = ""

    while True:
        results = index.query(
            vector=[0.0] * 768,
            top_k=batch_size,
            include_metadata=True,
            filter={},
            namespace=""
        )

        if not results.matches:
            break

        for match in results.matches:
            chunk_id = match.id
            metadata = match.metadata or {}
            text = metadata.get("text", "")
            bias = detect_bias_in_text(text)

            if bias["gender_bias"] or bias["intensity_bias"]:
                bias_report.append({
                    "id": chunk_id,
                    "text": text,
                    "gender_bias": bias["gender_bias"],
                    "intensity_bias": bias["intensity_bias"]
                })

        break  # Remove this to keep paginating when needed

    return bias_report


if __name__ == "__main__":
    report = scan_pinecone_chunks()
    if not report:
        print("✅ No bias found in chunks.")
    else:
        print(f"\n🚨 Bias detected in {len(report)} chunks:\n")
        for entry in report:
            print(f"🧠 Chunk ID: {entry['id']}")
            print(f"📌 Text: {entry['text'][:200]}...")
            print(f"   👥 Gender Bias: {entry['gender_bias']}")
            print(f"   🔥 Intensity Bias: {entry['intensity_bias']}\n")
