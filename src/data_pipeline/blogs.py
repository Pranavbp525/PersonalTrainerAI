import os
import json
import requests
import re
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ----------------------------------
# Logging Configuration
# ----------------------------------
#  Configure logging for each file, writing to the same file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/scraper.log"))),  # Logs go into the same file
        logging.StreamHandler()  # Also print logs to the console
    ]
)

logger = logging.getLogger(__name__)  # Logger for each file

# Define URLs for blogs
BLOG_URLS = [
    "https://www.muscleandstrength.com/articles/beginners-guide-to-zone-2-cardio",
    "https://www.muscleandstrength.com/articles/best-hiit-routines-gym-equipment",
    "https://jeffnippard.com/blogs/news",
    "https://rpstrength.com/blogs/articles",
    "https://rpstrength.com/collections/guides",
    "https://www.strongerbyscience.com/complete-strength-training-guide/",
    "https://www.strongerbyscience.com/how-to-squat/",
    "https://www.strongerbyscience.com/how-to-bench/",
    "https://www.strongerbyscience.com/how-to-deadlift/",
    "https://www.strongerbyscience.com/hypertrophy-range-fact-fiction/",
    "https://www.strongerbyscience.com/metabolic-adaptation/"
]

# ----------------------------------
# Helper Functions
# ----------------------------------
def fetch_page_content(url):
    """
    Fetch page content using requests and return a BeautifulSoup object.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}", exc_info=True)
        return None

def clean_text(text):
    """
    Removes special characters, escape sequences, and unnecessary whitespace.
    """
    return re.sub(r'[\s●"“”]+', ' ', text).strip()

def extract_main_content(soup):
    """
    Extracts the main content from an blog post by checking common selectors.
    """
    if not soup:
        return None
    for selector in ['main', 'article', 'div#content', 'div.content', 'div.entry-content', 'section']:
        content_block = soup.select_one(selector)
        if content_block:
            return content_block
    return soup  # fallback to entire soup if no specific selectors found

def scrape_text_from_url(url, visited, all_data):
    """
    Recursively scrapes text content from the given URL and follows subpage links.
    """
    if url in visited:
        return  # avoid duplicates / infinite loops

    soup = fetch_page_content(url)
    if not soup:
        return

    visited.add(url)
    main_content = extract_main_content(soup)
    if not main_content:
        logger.warning(f"Could not find a main content block for {url}")
        return

    paragraphs = main_content.find_all('p')
    text = clean_text(" ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)))
    title = main_content.find('h1') or soup.find('title')

    # Log how many paragraphs were found for this URL
    logger.info(f"📄 Extracted {len(paragraphs)} paragraphs from {url}")

    # Store data
    all_data.append({
        "source": url.split("//")[-1].split("/")[0],
        "title": title.get_text(strip=True) if title else url,
        "url": url,
        "description": text
    })

    # Extract subpage links
    for link in main_content.find_all('a', href=True):
        sub_url = urljoin(url, link['href'])
        if sub_url.startswith(url) and sub_url not in visited:
            scrape_text_from_url(sub_url, visited, all_data)

def scrape_web():
    """
    Scrape blogs (including subpages) and return the collected data.
    """
    visited = set()
    blogs_data = []

    # Scrape blog URLs
    for url in BLOG_URLS:
        logger.info(f"Scraping blog URL: {url}")
        scrape_text_from_url(url, visited, blogs_data)

    # Return combined or separate data depending on your needs
    return blogs_data

def save_to_json(data, filename):
    """
    Save extracted data to a JSON file.
    """
    try:
        output_path = filename

        # Make sure the directory exists
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving to {filename}: {e}", exc_info=True)


# Main Execution
def blog_scraper():

    logger.info("Starting web scraping...")

    blogs_data = scrape_web()

    if blogs_data:
        output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw_json_data/blogs.json"))
        save_to_json(blogs_data, output_file)
    else:
        logger.warning("No blog data extracted.")

    logger.info("Web scraping completed successfully!")
