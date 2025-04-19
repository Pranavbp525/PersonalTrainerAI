# src/data_pipeline/blogs.py
import os
import json
import requests
import re
import logging
import time # Import time for delays
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Import GCS utility functions
try:
    from .gcs_utils import upload_string_to_gcs
except ImportError:
    # Fallback for potential direct execution testing
    from gcs_utils import upload_string_to_gcs

# ----------------------------------
# Logging Configuration
# ----------------------------------
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/scraper.log"))
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized in blogs.py")

# --- Define GCS paths ---
OUTPUT_BUCKET = "ragllm-454718-raw-data" # Or processed bucket if preferred for raw scrape
OUTPUT_BLOB_NAME = "raw_json_data/blogs.json" # Path within the OUTPUT_BUCKET

# Define URLs for blogs
# Note: M&S URLs often return 403, scraping them might be unreliable/blocked
BLOG_URLS = [
    # "https://www.muscleandstrength.com/articles/beginners-guide-to-zone-2-cardio", # Likely blocked
    # "https://www.muscleandstrength.com/articles/best-hiit-routines-gym-equipment", # Likely blocked
    "https://jeffnippard.com/blogs/news",
    "https://rpstrength.com/blogs/articles",
    "https://rpstrength.com/collections/guides", # This might be more of a product page
    "https://www.strongerbyscience.com/complete-strength-training-guide/",
    "https://www.strongerbyscience.com/how-to-squat/",
    "https://www.strongerbyscience.com/how-to-bench/",
    "https://www.strongerbyscience.com/how-to-deadlift/",
    "https://www.strongerbyscience.com/hypertrophy-range-fact-fiction/",
    "https://www.strongerbyscience.com/metabolic-adaptation/"
]

# --- Constants ---
REQUEST_TIMEOUT = 20 # Seconds
REQUEST_DELAY = 1.5 # Seconds between requests to the same domain (be respectful!)
MAX_RETRIES = 2 # Retries for non-403 errors
MAX_CRAWL_DEPTH = 2 # Limit recursion depth to avoid excessive scraping

# ----------------------------------
# Helper Functions
# ----------------------------------
def fetch_page_content(url):
    """
    Fetches page content using requests with User-Agent, timeout, and retries.
    Returns a BeautifulSoup object or None on failure.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    logger.debug(f"Fetching URL: {url}")
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Check content type to avoid parsing non-HTML content if possible
            if 'html' not in response.headers.get('Content-Type', '').lower():
                 logger.warning(f"Skipping non-HTML content type for {url}: {response.headers.get('Content-Type')}")
                 return None
            return BeautifulSoup(response.text, 'html.parser')

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error fetching {url} (Status: {e.response.status_code}): {e}")
            if e.response.status_code == 403:
                 logger.warning(f"Received 403 Forbidden for {url}. Site may be blocking scrapes. Aborting retries for this URL.")
                 return None # Don't retry 403
            # Don't retry other client errors (4xx) immediately
            if 400 <= e.response.status_code < 500 and e.response.status_code != 403:
                 logger.warning(f"Client error {e.response.status_code} fetching {url}. Aborting retries.")
                 return None
            # Retry server errors (5xx) after delay
            if attempt < MAX_RETRIES:
                 wait = REQUEST_DELAY * (2 ** attempt) # Exponential backoff
                 logger.info(f"Retrying {url} after {wait:.1f} sec (Attempt {attempt + 2}/{MAX_RETRIES + 1})...")
                 time.sleep(wait)
            else:
                 logger.error(f"Failed to fetch {url} after {MAX_RETRIES + 1} attempts due to HTTP error.")
                 return None # Max retries reached for server error

        except requests.exceptions.RequestException as e:
            # Includes connection errors, timeouts, etc.
            logger.error(f"Request Exception fetching {url} (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}", exc_info=True)
            if attempt < MAX_RETRIES:
                wait = REQUEST_DELAY * (2 ** attempt)
                logger.info(f"Retrying {url} after {wait:.1f} sec...")
                time.sleep(wait)
            else:
                 logger.error(f"Failed to fetch {url} after {MAX_RETRIES + 1} attempts due to RequestException.")
                 return None # Max retries reached

    return None # Should not be reached if loop completes, but safety return

def clean_text(text):
    """
    Removes special characters, escape sequences, and unnecessary whitespace.
    """
    # Keeping your original cleaning function
    if text:
        return re.sub(r'\s+', ' ', text).strip()
    return ""

def extract_main_content(soup):
    """
    Extracts the main content from an blog post by checking common selectors.
    """
    if not soup:
        return None
    # Prioritize common semantic tags for articles/main content
    selectors = [
        'article', 'main', '[role="main"]',
        '.entry-content', '.post-content', '.td-post-content', '.article-content',
        'div#content', 'div.content', 'section.content'
        ]
    content_block = None
    for selector in selectors:
        content_block = soup.select_one(selector)
        if content_block:
            logger.debug(f"Found main content block using selector: {selector}")
            return content_block
    logger.warning("Could not find specific main content block using common selectors, falling back to body.")
    return soup.body # Fallback to body if no specific container found

def scrape_text_from_url(url, base_domain, visited, all_data, current_depth=0):
    """
    Recursively scrapes text content from the given URL and follows relevant subpage links
    within the same base domain, up to a maximum depth.
    """
    if url in visited or current_depth > MAX_CRAWL_DEPTH:
        if url in visited:
             logger.debug(f"Skipping already visited URL: {url}")
        if current_depth > MAX_CRAWL_DEPTH:
             logger.debug(f"Skipping URL due to max depth ({MAX_CRAWL_DEPTH}): {url}")
        return

    # Add delay before processing each URL
    time.sleep(REQUEST_DELAY)

    logger.info(f"Scraping [Depth {current_depth}]: {url}")
    soup = fetch_page_content(url)
    if not soup:
        visited.add(url) # Mark as visited even if fetch failed to avoid retrying
        return

    visited.add(url)
    main_content = extract_main_content(soup)
    if not main_content:
        logger.warning(f"Could not extract main content block for {url}")
        return

    # Extract text from paragraphs within the main content
    paragraphs = main_content.find_all('p')
    text = clean_text(" ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)))
    title_tag = main_content.find('h1') or soup.find('title') # Prioritize H1 in content
    title = clean_text(title_tag.get_text()) if title_tag else urlparse(url).path

    # Log how many paragraphs were found for this URL
    logger.info(f"📄 Extracted {len(paragraphs)} paragraphs from {url} (Title: {title})")

    # Store data if text was found
    if text:
        all_data.append({
            "source": base_domain, # Store base domain as source category
            "title": title,
            "url": url,
            "text": text # Use 'text' key
        })
    else:
        logger.warning(f"No text extracted from paragraphs in main content of {url}")

    # Extract and follow subpage links within the same domain and depth limit
    if current_depth < MAX_CRAWL_DEPTH:
        links_found = 0
        for link in main_content.find_all('a', href=True):
            href = link['href']
            # Basic filtering: avoid javascript, mailto, anchors, PDFs etc.
            if href and not href.startswith(('#', 'javascript:', 'mailto:')) and not href.lower().endswith(('.pdf', '.jpg', '.png', '.zip')):
                sub_url = urljoin(url, href)
                # Check if the resolved URL belongs to the same base domain
                if urlparse(sub_url).netloc == base_domain:
                    links_found += 1
                    # Recursively call scrape function for the sub-URL
                    scrape_text_from_url(sub_url, base_domain, visited, all_data, current_depth + 1)
        logger.debug(f"Found {links_found} potential sub-links to follow from {url}")

def scrape_all_blogs():
    """
    Iterates through BLOG_URLS, scrapes each (including subpages up to MAX_CRAWL_DEPTH),
    and returns the collected data.
    """
    visited_global = set() # Keep track of visited URLs across all base URLs
    all_blogs_data = []

    for base_url in BLOG_URLS:
        logger.info(f"--- Starting scrape for base URL: {base_url} ---")
        # Extract base domain (e.g., 'jeffnippard.com') to constrain scraping
        base_domain = urlparse(base_url).netloc
        if not base_domain:
             logger.warning(f"Could not parse base domain for {base_url}, skipping.")
             continue

        scrape_text_from_url(base_url, base_domain, visited_global, all_blogs_data, current_depth=0)
        logger.info(f"--- Finished scrape for base URL: {base_url} ---")

    return all_blogs_data

def save_data_to_gcs(data):
    """Saves the extracted blog data list as JSON to the specified GCS location."""
    if not data:
        logger.warning("No extracted blog data to save to GCS.")
        return True # Success if there was nothing to save

    logger.info(f"Attempting to save extracted blog data to gs://{OUTPUT_BUCKET}/{OUTPUT_BLOB_NAME}")
    try:
        json_data_string = json.dumps(data, indent=4, ensure_ascii=False)
        success = upload_string_to_gcs(OUTPUT_BUCKET, OUTPUT_BLOB_NAME, json_data_string)

        if success:
             logger.info(f"Successfully saved blog data JSON ({len(data)} entries) to gs://{OUTPUT_BUCKET}/{OUTPUT_BLOB_NAME}")
             return True
        else:
             logger.error("Failed to save blog data to GCS.")
             return False
    except Exception as e:
        logger.error(f"Error during JSON serialization or GCS upload for blog data: {e}", exc_info=True)
        return False


# Main Execution Function for Airflow
def run_blog_pipeline():
    """Main function to run the blog scraping pipeline and save to GCS."""
    logger.info("--- Starting Blog Scraping Pipeline (GCS) ---")

    blogs_data = scrape_all_blogs()

    save_success = save_data_to_gcs(blogs_data)

    if save_success:
        logger.info("--- Blog Scraping Pipeline (GCS) finished successfully! ---")
        return True
    else:
        logger.error("--- Blog Scraping Pipeline (GCS) failed during saving phase. ---")
        return False

# Allow direct execution for testing
if __name__ == "__main__":
    logger.info("Running blogs.py directly for testing...")
    # Prerequisites for local testing:
    # 1. gcs_utils.py must be in the same directory or accessible via PYTHONPATH.
    # 2. Run 'gcloud auth application-default login' in your terminal.
    # 3. Ensure you have permissions to write to gs://ragllm-454718-raw-data/raw_json_data/
    run_blog_pipeline()