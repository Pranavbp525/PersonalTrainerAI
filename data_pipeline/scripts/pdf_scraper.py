import os
import json
import PyPDF2
import unicodedata
import logging
import re

# ✅ Configure logging for each file, writing to the same file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),  # ✅ Logs go into the same file
        logging.StreamHandler()  # ✅ Also print logs to the console
    ]
)

logger = logging.getLogger(__name__)  # ✅ Logger for each file

# Define output directory
OUTPUT_DIR = "data/pdf_raw_json_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clean_text(text):
    """Normalize text encoding to remove special characters and excessive whitespace."""
    return re.sub(r'[\s●"“”]+', ' ', text).strip()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + " "
        logger.info(f"✅ Successfully extracted text from: {pdf_path}")  # ✅ Log success
    except Exception as e:
        logger.error(f"⚠️ Error reading {pdf_path}: {e}", exc_info=True)  # ✅ Log error with traceback
    return clean_text(text)

def scrape_pdfs(directory="data/source_pdf", max_pdfs=2):
    """Extract text from only 2 PDFs in the specified folder and format JSON output."""
    all_pdfs = []
    
    if not os.path.exists(directory):
        logger.warning(f"⚠️ Directory '{directory}' not found.")  # ✅ Log missing directory
        return all_pdfs
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")][:max_pdfs]  # ✅ Limit to 2 PDFs
    
    if not pdf_files:
        logger.warning(f"⚠️ No PDF files found in '{directory}'.")  # ✅ Log if no PDFs exist
        return all_pdfs

    logger.info(f"📌 Found {len(pdf_files)} PDFs. Extracting only {max_pdfs} for testing...")  # ✅ Log number of PDFs found

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        logger.info(f"📄 Processing PDF: {filename}")  # ✅ Log PDF processing start
        
        text = extract_text_from_pdf(file_path)
        if text:
            all_pdfs.append({
                "source": "Local PDF",
                "title": os.path.splitext(filename)[0],  # PDF title without extension
                "url": "N/A",  # No URL since it's local
                "description": text
            })
    
    logger.info(f"✅ PDF extraction completed. Extracted {len(all_pdfs)} PDFs.")  # ✅ Log completion
    return all_pdfs

def save_to_json(data, filename):
    """Save extracted data to a JSON file."""
    try:
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        logger.info(f"✅ Data saved to {output_path}")  # ✅ Log success
    except Exception as e:
        logger.error(f"⚠️ Error saving to {filename}: {e}", exc_info=True)  # ✅ Log failure

if __name__ == "__main__":
    logger.info("📌 Starting PDF extraction process (Limited to 2 PDFs)...")  # ✅ Log script start
    
    pdf_data = scrape_pdfs(max_pdfs=2)  # ✅ Limit to 2 PDFs
    
    if pdf_data:
        save_to_json(pdf_data, "pdf_data.json")
        logger.info("✅ PDF extraction and saving completed successfully!")  # ✅ Log final success
    else:
        logger.warning("⚠️ No PDFs extracted. No data saved.")  # ✅ Log if no PDFs were found
