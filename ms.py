import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WorkoutScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self.rate_limit_delay = (1, 3)  # Balanced delay (No blocking)

    def fetch_html(self, url):
        """Fetch and parse HTML content with error handling."""
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=5)  # Faster timeout (5 sec)
                response.raise_for_status()
                return BeautifulSoup(response.text, "html.parser")
            except requests.RequestException as e:
                logger.error(f"Error fetching {url} (Attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:  # Only delay if there are more retries left
                    delay = random.uniform(*self.rate_limit_delay)
                    logger.info(f"Retrying after {delay:.2f} sec...")
                    time.sleep(delay)
    
        return None  # Return None if all retries fail

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
                exercise_description = self.extract_exercise_details(exercise_url)

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
        max_workouts = 100  # Limit to 5 workout tiles only
        workout_count = 0  # Counter for scraped workouts

        while current_url and workout_count < max_workouts:
            logger.info(f"Scraping page {page_count}: {current_url}")

            soup = self.fetch_html(current_url)
            if not soup:
                break

            workout_links = self.extract_workout_links(soup)
            
            for link in workout_links:
                if workout_count >= max_workouts:
                    break  # Stop after 5 workouts

                detailed_info = self.extract_detailed_info(link)
                if detailed_info:
                    workouts.append(detailed_info)
                    logger.info(f"Scraped workout {workout_count + 1} from: {link}")
                    workout_count += 1  # Increment count

            # Stop if we have scraped 5 workouts
            if workout_count >= max_workouts:
                break

            try:
                next_page = self.get_next_page(soup)
                current_url = urljoin(self.base_url, next_page) if next_page else None
                page_count += 1  # Move to the next page
            except Exception as e:
                logger.error(f"Error getting next page: {str(e)}")
        logger.info(f"Finished scraping {workout_count} workouts.")
        return workouts



    def save_to_json(self, workouts, filename):
        """Save scraped data to JSON file."""
        try:
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

def main():
    base_url = ["https://www.muscleandstrength.com/workouts/men","https://www.muscleandstrength.com/workouts/women","https://www.muscleandstrength.com/workouts/muscle-building",
                "https://www.muscleandstrength.com/workouts/fat-loss", "https://www.muscleandstrength.com/workouts/strength", "https://www.muscleandstrength.com/workouts/abs",
                "https://www.muscleandstrength.com/workouts/full-body"]
    output_file = "workouts1.json"
    all_workouts = []
    try:
        for link in base_url:
            scraper = WorkoutScraper(link)
            workouts = scraper.scrape_workouts()
            all_workouts = all_workouts + workouts
            scraper.save_to_json(all_workouts, output_file)
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")

if __name__ == "__main__":
    main()