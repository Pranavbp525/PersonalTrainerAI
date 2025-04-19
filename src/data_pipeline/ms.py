# src/data_pipeline/ms.py
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin, urlparse
import random
import os

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
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized in ms.py")

# --- Define GCS paths ---
OUTPUT_BUCKET = "ragllm-454718-raw-data" # Saving raw scrape output here
OUTPUT_BLOB_NAME = "raw_json_data/ms_data.json" # Consistent name

# --- Constants ---
REQUEST_TIMEOUT = 15 # Seconds (can be shorter if site responds quickly)
BASE_DELAY = 1.5 # Base seconds between requests
MAX_RETRIES = 2 # Retries for non-403/non-4xx errors
# Set a limit for testing, remove or increase for full scrape (beware of blocking)
MAX_WORKOUTS_TO_SCRAPE = 10 # Limit total workouts scraped

class WorkoutScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        # Extract base domain for reference
        self.base_domain = urlparse(base_url).netloc
        self.session = requests.Session()
        # Set a realistic User-Agent
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        })
        # Use a slightly randomized delay to appear less bot-like
        self.rate_limit_delay_range = (BASE_DELAY, BASE_DELAY + 1.5)

    def _get_delay(self):
        """Returns a random delay within the configured range."""
        return random.uniform(*self.rate_limit_delay_range)

    def fetch_html(self, url):
        """Fetch and parse HTML content with error handling, retries, and delay."""
        logger.debug(f"Fetching HTML from: {url}")
        # Add delay *before* making the request
        delay = self._get_delay()
        logger.debug(f"Waiting {delay:.2f} seconds before fetching {url}")
        time.sleep(delay)

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                if 'html' not in response.headers.get('Content-Type', '').lower():
                    logger.warning(f"Skipping non-HTML content type for {url}: {response.headers.get('Content-Type')}")
                    return None
                return BeautifulSoup(response.text, "html.parser")

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error fetching {url} (Status: {e.response.status_code}) (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                if e.response.status_code == 403:
                    logger.warning(f"Received 403 Forbidden for {url}. Muscle&Strength likely blocking. Aborting fetch for this URL.")
                    return None # Don't retry 403
                if 400 <= e.response.status_code < 500:
                     logger.warning(f"Client error {e.response.status_code} fetching {url}. Aborting retries.")
                     return None # Don't retry other 4xx client errors
                # Retry only server errors (5xx)
                if attempt < MAX_RETRIES:
                     retry_delay = self._get_delay() * (2 ** attempt) # Exponential backoff on retries
                     logger.info(f"Retrying {url} after {retry_delay:.1f} sec...")
                     time.sleep(retry_delay)
                else:
                     logger.error(f"Failed to fetch {url} after {MAX_RETRIES + 1} attempts due to HTTP error.")
                     return None # Max retries reached

            except requests.exceptions.RequestException as e:
                logger.error(f"Request Exception fetching {url} (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}", exc_info=True)
                if attempt < MAX_RETRIES:
                    retry_delay = self._get_delay() * (2 ** attempt)
                    logger.info(f"Retrying {url} after {retry_delay:.1f} sec...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch {url} after {MAX_RETRIES + 1} attempts due to RequestException.")
                    return None # Max retries reached

        return None # Should not be reached

    def extract_workout_links(self, soup):
        """Extracts workout links from the main page soup."""
        links = []
        # Finding the container for workout cells might need adjustment if class changes
        workout_container = soup.find("div", class_="grid-x grid-margin-x grid-margin-y")
        if not workout_container:
             logger.warning("Could not find main workout container div.")
             return []

        # Assuming each workout is in a cell div like before
        workout_divs = workout_container.find_all("div", class_="cell small-12 bp600-6", recursive=False) # Non-recursive might be safer
        logger.info(f"Found {len(workout_divs)} potential workout entry divs on page.")

        for div in workout_divs:
            try:
                # Find the link more specifically, maybe within a header or image
                link_tag = div.find("a", href=True) # Simplest approach first
                if link_tag and 'href' in link_tag.attrs:
                    link = urljoin(self.base_url, link_tag["href"])
                    # Basic check if it looks like a workout URL (adjust pattern if needed)
                    if '/workouts/' in link:
                         links.append(link)
                    else:
                         logger.debug(f"Skipping link not matching pattern: {link}")
                else:
                    logger.warning("Workout link tag not found within a div.")
            except Exception as e: # Catch broader errors during parsing
                logger.error(f"Error extracting a workout link: {e}", exc_info=True)

        logger.info(f"Extracted {len(links)} workout links from this page.")
        return links

    def extract_exercise_details(self, exercise_url):
        """Extracts description text from the individual exercise page."""
        logger.debug(f"Fetching exercise details from: {exercise_url}")
        soup = self.fetch_html(exercise_url)
        if not soup:
            logger.warning(f"Could not fetch exercise page {exercise_url}, returning default description.")
            return "No description available (fetch failed)."

        try:
            # Try to find the main content area for the exercise description
            # Selectors might need adjustment based on actual exercise page structure
            content_div = soup.find("div", class_="content clearfix") or \
                          soup.find("div", itemprop="description") or \
                          soup.find("div", class_="node-main-content") # Add more selectors if needed

            if content_div:
                # Attempt to remove known noisy sections *before* getting text
                selectors_to_remove = [
                    ".node-stats-block", ".grid-x.target-muscles", ".recommended-supplements",
                    ".video-container", ".related-exercises", ".addtoany_share_save_container"
                    ]
                for selector in selectors_to_remove:
                    for unwanted_element in content_div.select(selector):
                        unwanted_element.decompose()

                # Get text, using newline as separator for better readability potentially
                description_text = content_div.get_text(strip=True, separator="\n")
                # Basic cleaning (replace multiple newlines/spaces)
                description_text = re.sub(r'\n{2,}', '\n', description_text).strip()
                description_text = re.sub(r'\s{2,}', ' ', description_text).strip()

                if not description_text:
                     logger.warning(f"Extracted empty description from {exercise_url} after cleaning.")
                     return "No description available (empty after parse)."

                return description_text
            else:
                logger.warning(f"Could not find main content div for exercise description at {exercise_url}")
                return "No description available (content block not found)."

        except Exception as e:
            logger.error(f"Error extracting exercise description from {exercise_url}: {e}", exc_info=True)
            return "No description available (parsing error)."

    def extract_workout_detailed_info(self, url):
        """Extracts detailed workout information including exercises and their descriptions."""
        logger.info(f"Extracting workout details from: {url}")
        soup = self.fetch_html(url)
        if not soup:
            logger.error(f"Failed to fetch workout page {url}, cannot extract details.")
            return None # Return None to indicate failure

        workout_data = {
            "source_site": self.base_domain, # Base domain like 'muscleandstrength.com'
            "title": "No title found",
            "url": url,
            "summary": {},
            "description": "No description available.",
            "exercises": []
        }

        try:
            # --- Extract Title ---
            # Be more specific if possible, e.g., within a specific header div
            title_tag = soup.select_one("h1.page-title") or soup.find("h1") # Try common title tags
            if title_tag:
                workout_data["title"] = title_tag.get_text(strip=True)

            # --- Extract Workout Summary Stats ---
            summary_block = soup.find("div", class_="node-stats-block")
            if summary_block:
                list_items = summary_block.find_all("li")
                for item in list_items:
                    try:
                        label_tag = item.find("span")
                        if label_tag:
                             label = label_tag.text.strip().rstrip(':') # Clean label
                             value = item.text.replace(label_tag.text, "").strip()
                             if label and value: # Ensure both parts exist
                                workout_data["summary"][label] = value
                        else:
                             logger.debug(f"Could not find label span in summary item: {item.text}")
                    except Exception as item_e:
                         logger.warning(f"Error parsing summary item '{item.text}': {item_e}")
            else:
                 logger.warning(f"Could not find summary stats block for {url}")

            # --- Extract Main Workout Description & Exercises ---
            # Find the main content area first
            content_area = soup.find("div", class_="field field-name-body field-type-text-with-summary") or \
                           soup.find("div", class_="node-content") # Add other potential main content divs

            if content_area:
                # Extract Exercises FIRST (before potentially removing tables)
                # Assume exercises are in a table within the content area
                exercise_table = content_area.find("table") # Find the first table
                if exercise_table:
                    logger.debug(f"Found exercise table in content area for {url}")
                    # Look for links within table rows (more robust)
                    for row in exercise_table.find_all('tr'):
                         link_tag = row.find('a', href=True)
                         if link_tag:
                              exercise_name = link_tag.get_text(strip=True)
                              exercise_url = urljoin(self.base_url, link_tag["href"])
                              if exercise_name and "/exercises/" in exercise_url: # Basic validation
                                   # Extract the description from the linked exercise page
                                   exercise_description = self.extract_exercise_details(exercise_url)
                                   workout_data["exercises"].append({
                                        "exercise": exercise_name,
                                        "url": exercise_url,
                                        "description": exercise_description
                                   })
                              else:
                                   logger.debug(f"Skipping invalid exercise link/name in table: {link_tag.text}")

                    # Remove the table *after* extracting exercises to get clean description text
                    exercise_table.decompose()
                else:
                     logger.warning(f"Did not find an exercise table within the content area for {url}")

                # Extract the remaining text as the main description
                main_description_text = content_area.get_text(strip=True, separator=" ")
                # Clean the description text
                main_description_text = re.sub(r'\s+', ' ', main_description_text).strip()
                workout_data["description"] = main_description_text if main_description_text else "No description available."

            else:
                logger.warning(f"Could not find main content area for description/exercises at {url}")


            logger.info(f"Successfully extracted details for '{workout_data['title']}'. Found {len(workout_data['exercises'])} exercises.")
            return workout_data

        except Exception as e:
            logger.error(f"Critical error extracting detailed info from {url}: {e}", exc_info=True)
            # Return None or potentially partially filled dict based on needs
            return None # Indicate failure for this workout

    def get_next_page(self, soup):
        """Finds the URL for the next page in pagination."""
        if not soup: return None
        try:
            # Find the 'Next' link - selector might need adjustment
            next_button = soup.select_one('li.pager-next a[href]') # More specific selector
            if next_button:
                next_href = next_button['href']
                logger.debug(f"Found next page link: {next_href}")
                return urljoin(self.base_url, next_href) # Ensure absolute URL
            else:
                logger.debug("Next page link not found.")
                return None
        except Exception as e:
            logger.error(f"Error finding next page link: {e}", exc_info=True)
            return None

    def scrape_workouts(self, max_workouts=None):
        """
        Main function to scrape workout list pages and detail pages,
        up to a maximum number of workouts.
        """
        scraped_workouts = []
        current_page_url = self.base_url
        page_num = 1

        logger.info(f"Starting workout scraping from {self.base_url}. Max workouts: {max_workouts or 'Unlimited'}")

        while current_page_url:
            if max_workouts is not None and len(scraped_workouts) >= max_workouts:
                logger.info(f"Reached maximum workout limit ({max_workouts}). Stopping scrape.")
                break

            logger.info(f"--- Scraping List Page {page_num}: {current_page_url} ---")
            list_soup = self.fetch_html(current_page_url)
            if not list_soup:
                logger.error(f"Failed to fetch list page {page_num}, stopping scrape.")
                break # Stop if a list page fails

            workout_links = self.extract_workout_links(list_soup)
            if not workout_links:
                 logger.warning(f"No workout links found on page {page_num}. Checking for next page.")
                 # Still check for next page even if no links found here

            for link in workout_links:
                if max_workouts is not None and len(scraped_workouts) >= max_workouts:
                    logger.info(f"Reached maximum workout limit ({max_workouts}) during page processing.")
                    current_page_url = None # Break outer loop too
                    break # Stop processing links on this page

                detailed_info = self.extract_workout_detailed_info(link)
                if detailed_info: # Check if extraction succeeded
                    scraped_workouts.append(detailed_info)
                    logger.info(f"Successfully scraped workout {len(scraped_workouts)}: '{detailed_info.get('title', 'N/A')}'")
                else:
                     logger.warning(f"Failed to extract detailed info for workout link: {link}")
                # Delay is handled within fetch_html

            # Check for next page only if we haven't forced a break
            if current_page_url:
                 next_page_url = self.get_next_page(list_soup)
                 if next_page_url == current_page_url: # Avoid loop if next points to self
                      logger.warning("Next page URL is same as current URL. Stopping.")
                      current_page_url = None
                 else:
                      current_page_url = next_page_url
                 page_num += 1
                 if not current_page_url:
                      logger.info("No more pages found or next page link failed.")

        logger.info(f"--- Finished scraping. Total workouts collected: {len(scraped_workouts)} ---")
        return scraped_workouts

    def save_data_to_gcs(self, workouts):
        """Saves the scraped workout data list as JSON to the specified GCS location."""
        if not workouts:
            logger.warning("No M&S workout data scraped/extracted to save to GCS.")
            return True # Success if there was nothing to save

        logger.info(f"Attempting to save {len(workouts)} scraped workouts to gs://{OUTPUT_BUCKET}/{OUTPUT_BLOB_NAME}")

        # Structure the output slightly differently as per original save_to_json
        output_data = {
            "metadata": {
                "total_workouts_saved": len(workouts),
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "source_url": self.base_url
            },
            "workouts": workouts # List of workout dictionaries
        }

        try:
            json_data_string = json.dumps(output_data, indent=2, ensure_ascii=False)
            success = upload_string_to_gcs(OUTPUT_BUCKET, OUTPUT_BLOB_NAME, json_data_string)

            if success:
                 logger.info(f"Successfully saved M&S workout data JSON to gs://{OUTPUT_BUCKET}/{OUTPUT_BLOB_NAME}")
                 return True
            else:
                 logger.error("Failed to save M&S workout data to GCS (upload error).")
                 return False
        except Exception as e:
            logger.error(f"Error during JSON serialization or GCS upload for M&S data: {e}", exc_info=True)
            return False


# Main Execution Function for Airflow
def run_ms_pipeline(max_workouts=MAX_WORKOUTS_TO_SCRAPE):
    """Main function to run the M&S workout scraping pipeline and save to GCS."""
    logger.info("--- Starting Muscle & Strength Workout Scraping Pipeline (GCS) ---")
    start_url = "https://www.muscleandstrength.com/workouts/men" # Specific starting point

    try:
        scraper = WorkoutScraper(start_url)
        workouts_data = scraper.scrape_workouts(max_workouts=max_workouts)
        save_success = scraper.save_data_to_gcs(workouts_data)

        if save_success:
            logger.info("--- Muscle & Strength Pipeline (GCS) finished successfully! ---")
            return True
        else:
            logger.error("--- Muscle & Strength Pipeline (GCS) failed during saving phase. ---")
            return False
    except Exception as e:
        # Catch unexpected errors during scraper initialization or execution
        logger.error(f"Muscle & Strength pipeline failed with an unexpected error: {e}", exc_info=True)
        return False


# Allow direct execution for testing
if __name__ == "__main__":
    logger.info("Running ms.py directly for testing...")
    # Prerequisites for local testing:
    # 1. gcs_utils.py must be in the same directory or accessible via PYTHONPATH.
    # 2. Run 'gcloud auth application-default login' in your terminal.
    # 3. Ensure you have permissions to write to gs://ragllm-454718-raw-data/raw_json_data/
    run_ms_pipeline(max_workouts=5) # Test by scraping only 5 workouts