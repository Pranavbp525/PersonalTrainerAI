import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin
import random
import os
import urllib.request
import ssl
from dotenv import load_dotenv

load_dotenv()


#  Configure logging for each file, writing to the same file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/scraper.log"))),  # Logs go into the same file
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__)  
class WorkoutScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.exercise_cache = {}
        self.session = requests.Session()
        self.session.headers.update({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com",
        })
        self.rate_limit_delay = (1, 3)  

    def fetch_html(self, url):
        """Fetch and parse HTML content with error handling."""
        retries = 3
        bright_data_api_key = os.getenv('bright_data_api_key')
        proxy = f'http://brd-customer-hl_827d2605-zone-web_unlocker1:{bright_data_api_key}@brd.superproxy.io:33335'

        for attempt in range(retries):
            try:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({'https': proxy, 'http': proxy}),
                    urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
                )
                response = opener.open(url).read().decode()
                return BeautifulSoup(response, "html.parser")

            except Exception as e:
                logger.error(f"Error fetching {url} (Attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    delay = random.uniform(*self.rate_limit_delay)
                    logger.info(f"Retrying after {delay:.2f} sec...")
                    time.sleep(delay)

        return None

    def extract_workout_links(self, soup):
        """Extracts workout links from the main page."""
        links = []
        workout_divs = soup.find_all("div", class_="cell small-12 bp600-6")

        logger.info(f"Found {len(workout_divs)} workout entries")

        for div in workout_divs:
            try:
                link_tag = div.find("a", href=True)
                if link_tag:
                    link = urljoin(self.base_url, link_tag["href"])
                    links.append(link)
                else:
                    logger.warning("Workout link not found in div")
            except AttributeError as e:
                logger.error(f"Error extracting link: {str(e)}")

        logger.info(f"Extracted {len(links)} workout links")
        return links

    def extract_exercise_details(self, exercise_url):
        """Extracts only the description from the exercise page."""
        soup = self.fetch_html(exercise_url)
        if not soup:
            return "No description available."

        try:
            content_div = soup.find("div", class_="content clearfix")

            if content_div:
                # Removing unwanted divs
                for unwanted_class in ["node-stats-block", "grid-x target-muscles"]:
                    unwanted_div = content_div.find("div", class_=unwanted_class)
                    if unwanted_div:
                        unwanted_div.decompose()
    
                # Extracting cleaned text
                summary = content_div.get_text(strip=True, separator="\n")

                return summary
            return "No description available."
        except Exception as e:
            logger.error(f"Error extracting exercise description from {exercise_url}: {str(e)}")
            return "No description available."

    def extract_detailed_info(self, url):
        """Extract detailed workout information including exercises and their descriptions."""
        logger.info(f"Fetching workout details from: {url}")
        soup = self.fetch_html(url)
        if not soup:
            return {}

        try:
            #extracing title
            title_tag = soup.find("div", class_="node-header").find("h1")  # Find <h1> inside div
            title = title_tag.get_text(strip=True) if title_tag else "No title found"

            
            # Extracting workout summary
            summary_table = soup.find("div", class_="node-stats-block")
            summary = {}

            if summary_table:
                rows = summary_table.find_all("li")
                for row in rows:
                    row_label = row.find("span").text.strip()
                    value = row.text.replace(row_label, "").strip()
                    summary[row_label] = value

            # Extracting workout description
            description = soup.find("div", class_="field field-name-body field-type-text-with-summary field-label-hidden")
            if description:
                description_text = description.get_text(strip=True, separator=" ")

            # Extracting exercises from the same section (No Limit)
            exercises = []
            exercise_table = description.find("table") if description else None
            exercise_links = exercise_table.find_all("a", href=True) if exercise_table else []

            for link in exercise_links:
                exercise_name = link.get_text(strip=True)
                exercise_url = urljoin(self.base_url, link["href"])
                
                # Extract the description from the exercise page
                if exercise_url in self.exercise_cache:
                    logger.info(f"Using cached description for: {exercise_url}")
                    exercise_description = self.exercise_cache[exercise_url]
                else:
                    exercise_description = self.extract_exercise_details(exercise_url)
                    self.exercise_cache[exercise_url] = exercise_description

                exercises.append({
                    "exercise": exercise_name,
                    "url": exercise_url,
                    "description": exercise_description
                })


            # # Remove all <table> elements from the description
            # if description:
            #     for table in description.find_all("table"):
            #         table.decompose()  # Removes the table from the HTML

            #     description_text = description.get_text(strip=True)
            # else:
            #     description_text = "No description available."  


            logger.info(f"Scraped {len(exercises)} exercises from {url}")

            return {
                "source": self.base_url,
                "title": title,
                "url": url,
                "summary": summary,
                "description": description_text,
                "exercises": exercises,
            }

        except Exception as e:
            logger.error(f"Error extracting detailed info from {url}: {str(e)}")
            return {}

    def scrape_workouts(self):
        """Main function to scrape ONLY 5 workout tiles (not pages)."""
        workouts = []
        current_url = self.base_url
        page_count = 1
        workout_count = 0  # Counter for scraped workouts

        while current_url:
            logger.info(f"Scraping page {page_count}: {current_url}")

            soup = self.fetch_html(current_url)
            if not soup:
                break

            workout_links = self.extract_workout_links(soup)
            
            for link in workout_links:
                detailed_info = self.extract_detailed_info(link)
                if detailed_info:
                    workouts.append(detailed_info)
                    logger.info(f"Scraped workout {workout_count + 1} from: {link}")
                    workout_count += 1  # Increment count

            try:

                next_page = self.get_next_page(soup)
                print(next_page)
                if next_page:
                    current_url = urljoin(self.base_url, next_page) if next_page else None
                    page_count += 1  # Move to the next page
                else:
                    break
            except Exception as e:
                logger.error(f"Error getting next page: {str(e)}")
        logger.info(f"Finished scraping {workout_count} workouts.")
        return workouts

    def get_next_page(self, soup):
        # Find the 'Next' button in pagination
        try:
            next_button = soup.find('li', class_='pager-next').find('a')
            print(next_button)
            if next_button and 'href' in next_button.attrs:
                return next_button['href']
            return None
        except Exception as e:
            return None

    def save_to_json(self, workouts, filename):
        """Save scraped data to JSON file."""
        try:
            output_path = filename
            # Make sure the directory exists
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": {
                        "total_workouts": len(workouts),
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source_url": self.base_url
                    },
                    "workouts": workouts
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved {len(workouts)} workouts to {filename}")
        except IOError as e:
            logger.error(f"Error saving to JSON: {str(e)}")


def ms_scraper():
    url = "https://www.muscleandstrength.com/workouts/men"
    #output_file = "workouts.json"
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw_json_data/ms_data.json"))
    
    try:
        scraper = WorkoutScraper(url)
        workouts = scraper.scrape_workouts()
        scraper.save_to_json(workouts, output_file)
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
# ms_scraper()