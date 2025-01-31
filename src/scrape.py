import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin
import os

logging.basicConfig(level=logging.INFO)

def extract_text(url, depth=1):
    all_text = []

    def scrape_page(url, current_depth):
        if current_depth > depth:
            return

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            all_text.append(text)
            logging.info(f"Extracted text from {url}")

            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
            for link in links:
                scrape_page(link, current_depth + 1)

        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while fetching {url}: {e}")

    scrape_page(url, 1)

    return all_text

# Example usage
urls_to_scrape = [
    "https://www.bodybuilding.com/workout-plans",
    "https://www.menshealth.com/fitness/",
    "https://www.womenshealthmag.com/fitness/",
    "https://www.muscleandfitness.com/workout-routines/",
    "https://www.fitnessblender.com/videos",
    "https://www.acefitness.org/education-and-resources/lifestyle/exercise-library/",
    "https://www.nerdfitness.com/blog/category/workouts/",
    "https://www.verywellfit.com/workouts-4157099",
    "https://www.self.com/fitness",
    "https://greatist.com/fitness"
]

for url in urls_to_scrape:
    logging.info(f"Scraping {url}")
    texts = extract_text(url, depth=2)  # Adjust depth as needed
    if texts:
        filename = url.replace("https://", "").replace("/", "_") + ".txt"
        with open(filename, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n\n")
        logging.info(f"Successfully scraped {url} and saved to {filename}")
    else:
        logging.warning(f"Could not scrape any text from {url}")