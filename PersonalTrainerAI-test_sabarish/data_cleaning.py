import os
import json
import re

# Define input and output directories
input_dir = "scraped_data/raw_json_data"
output_dir = "scraped_data/preprocessed_json_data"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of files to process
json_files = ["articles.json", "blogs.json", "pdf_data.json", "youtube_transcripts.json"]

# Function to clean text
def clean_text(text):
    """Removes special characters, excessive spaces, and normalizes text."""
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Keep only words, spaces, and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Process each JSON file
for file in json_files:
    input_file_path = os.path.join(input_dir, file)
    output_file_path = os.path.join(output_dir, file)
    
    # Skip if file does not exist
    if not os.path.exists(input_file_path):
        print(f"⚠️ Skipping {file}, file not found.")
        continue

    # Load JSON data
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine the key to clean
    text_key = "text" if "youtube" not in file else "transcript"

    # Clean text
    for entry in data:
        if text_key in entry:
            entry[text_key] = clean_text(entry[text_key])

    # Save cleaned data to the new output directory
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Preprocessed JSON files saved in 'scraped_data/preprocessed_json_data/'")
