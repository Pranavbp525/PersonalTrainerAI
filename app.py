from src.data_pipeline.ms_preprocess import ms_preprocessing
from src.data_pipeline.ms import ms_scraper
from src.data_pipeline.blogs import blog_scraper
from src.data_pipeline.pdfs import pdf_scraper
from src.data_pipeline.other_preprocesing import preprocess_json_other_files
from src.data_pipeline.vector_db import chunk_to_db

pdf_scraper = pdf_scraper()
print("PDF Scraping Completed")
blog_scraper()
print("Blog Scraping Completed")
ms_scraper()
print("MS Scraping Completed")
preprocess_json_other_files()
print("Other JSON Preprocessing Completed")
ms_preprocessing()
print("MS Preprocessing Completed")
chunk_to_db()
print("Chunking to DB Completed")