import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin
from requests.exceptions import RequestException
import random
from typing import Optional, Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkoutScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.rate_limit_delay = (1, 3)  # Random delay between 1 and 3 seconds

    def fetch_html(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML content with error handling."""
        try:
            time.sleep(random.uniform(*self.rate_limit_delay))
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_workout_info(self, workout_div) -> Optional[Dict]:
        """Extract basic workout information with error handling."""
        try:
            return {
                'title': workout_div.find('div', class_='node-title').text.strip(),
                'description': workout_div.find('div', class_='node-short-summary').text.strip(),
                'link': workout_div.find('a', class_='btn btn-blue')['href'],
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except AttributeError as e:
            logger.error(f"Error extracting workout info: {str(e)}")
            return None

    def extract_detailed_info(self, url: str) -> Dict:
        """Extract detailed workout information with error handling."""
        try:
            soup = self.fetch_html(url)
            if not soup:
                return {}

            detailed_info = {
                'equipment': [],
                'target_muscles': [],
                'workout_duration': None,
                'difficulty_level': None,
                'instructions': []
            }

            # Add your detailed parsing logic here
            # This is a placeholder for the actual implementation
            
            return detailed_info

        except Exception as e:
            logger.error(f"Error extracting detailed info from {url}: {str(e)}")
            return {}

    def get_next_page(self, soup: BeautifulSoup) -> Optional[str]:
        """Get next page URL with error handling."""
        try:
            next_link = soup.find('li', class_='pager-next')
            if next_link and next_link.find('a'):
                return next_link.find('a')['href']
            return None
        except Exception as e:
            logger.error(f"Error getting next page: {str(e)}")
            return None

    def scrape_workouts(self) -> List[Dict]:
        """Main scraping function with progress tracking."""
        workouts = []
        current_url = self.base_url
        page_count = 1

        while current_url:
            logger.info(f"Scraping page {page_count}: {current_url}")
            
            soup = self.fetch_html(current_url)
            if not soup:
                break

            workout_divs = soup.find_all('div', class_='cell small-12 bp600-6')
            for div in workout_divs:
                try:
                    workout = self.extract_workout_info(div)
                    if workout:
                        detailed_url = urljoin(self.base_url, workout['link'])
                        detailed_info = self.extract_detailed_info(detailed_url)
                        workout['detailed_info'] = detailed_info
                        workouts.append(workout)
                        logger.info(f"Scraped workout: {workout['title']}")
                except Exception as e:
                    pass

            next_page = self.get_next_page(soup)
            current_url = urljoin(self.base_url, next_page) if next_page else None
            page_count += 1

        return workouts

    def save_to_json(self, workouts: List[Dict], filename: str):
        """Save scraped data to JSON file with error handling."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'total_workouts': len(workouts),
                        'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'source_url': self.base_url
                    },
                    'workouts': workouts
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved {len(workouts)} workouts to {filename}")
        except IOError as e:
            logger.error(f"Error saving to JSON: {str(e)}")

def main():
    base_url = "https://www.muscleandstrength.com/workouts/men"
    output_file = "workouts.json"

    try:
        scraper = WorkoutScraper(base_url)
        workouts = scraper.scrape_workouts()
        scraper.save_to_json(workouts, output_file)
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")

if __name__ == "__main__":
    main()