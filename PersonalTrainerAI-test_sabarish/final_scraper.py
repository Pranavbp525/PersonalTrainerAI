import os
import json
import requests
import re
import PyPDF2
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Define output directory
OUTPUT_DIR = "scraped_data/raw_json_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# YouTube API setup
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Define URLs for Articles & Blogs
ARTICLE_URLS = [
    "https://www.health.harvard.edu/topics/diet-and-weight-loss",
    "https://www.health.harvard.edu/topics/exercise-and-fitness"
]

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

def fetch_page_content(url):
    """Fetch page content using requests and return BeautifulSoup object."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.text, 'html.parser') if response.status_code == 200 else None

def clean_text(text):
    """Removes special characters, escape sequences, and unnecessary whitespace."""
    return re.sub(r'[\s●"“”]+', ' ', text).strip()

def extract_main_content(soup):
    """Extracts the main content of an article or blog post."""
    for selector in ['main', 'article', 'div#content', 'div.content', 'div.entry-content', 'section']:
        content_block = soup.select_one(selector)
        if content_block:
            return content_block
    return soup

def scrape_text_from_url(url, visited, all_data):
    """Scrapes text content from a webpage and follows subpage links."""
    if url in visited:
        return
    soup = fetch_page_content(url)
    if not soup:
        return
    visited.add(url)
    main_content = extract_main_content(soup)
    paragraphs = main_content.find_all('p')
    text = clean_text(" ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)))

    # Store main page separately
    all_data.append({"source": url.split("//")[-1].split("/")[0], "url": url, "text": text})

    # Extract and store subpages separately
    for link in main_content.find_all('a', href=True):
        sub_url = urljoin(url, link['href'])
        if sub_url.startswith(url) and sub_url not in visited:
            scrape_text_from_url(sub_url, visited, all_data)

def scrape_web():
    """Scrape articles and blogs, including subpages."""
    visited = set()
    articles_data, blogs_data = [], []
    for url in ARTICLE_URLS:
        scrape_text_from_url(url, visited, articles_data)
    for url in BLOG_URLS:
        scrape_text_from_url(url, visited, blogs_data)
    save_to_json(articles_data, "articles.json")
    save_to_json(blogs_data, "blogs.json")

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
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
    return clean_text(text)

def scrape_pdfs(directory="PDFs"):
    """Extract text from all PDFs in the specified folder."""
    all_pdfs = []
    if not os.path.exists(directory):
        print(f"⚠️ Directory '{directory}' not found.")
        return all_pdfs
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(file_path)
            if text:
                all_pdfs.append({"source": "Local PDF", "title": os.path.splitext(filename)[0], "text": text})
    save_to_json(all_pdfs, "pdf_data.json")

def get_channel_videos(channel_name, days_back=365):
    """Get videos from a channel uploaded in the last X days with error handling."""
    try:
        search_response = youtube.search().list(q=channel_name, type='channel', part='id').execute()
        if not search_response['items']:
            return []
        channel_id = search_response['items'][0]['id']['channelId']
        search_params = {'channelId': channel_id, 'type': 'video', 'part': 'id,snippet', 'maxResults': 50}
        if days_back != -1:
            search_params['publishedAfter'] = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'

        videos = youtube.search().list(**search_params).execute()['items']
        return [{"video_id": v['id']['videoId'], "title": v['snippet']['title'], "published_at": v['snippet']['publishedAt']} for v in videos]
    
    except HttpError as e:
        print(f"❌ YouTube API error: {e}")
        return []

def get_transcript(video_id):
    """Get transcript for a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"An error occurred for video {video_id}: {str(e)}")
        return None

def scrape_youtube():
    """Scrape YouTube videos and transcripts."""
    channels = ["House of Hypertrophy", "Jeff Nippard", "Jeremy Ethier"]
    all_videos = []
    for channel in channels:
        videos = get_channel_videos(channel, days_back=-1)
        for video in videos:
            transcript = get_transcript(video['video_id'])
            if transcript:
                all_videos.append({"video_id": video['video_id'], "title": video['title'], "published_at": video['published_at'], "transcript": transcript})
    save_to_json(all_videos, "youtube_transcripts.json")

def save_to_json(data, filename):
    """Save extracted data to a JSON file."""
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"✅ Data saved to {filename}")

if __name__ == "__main__":
    print("📌 Starting web scraping...")
    scrape_web()
    print("📌 Scraping PDFs...")
    scrape_pdfs()
    print("📌 Scraping YouTube transcripts...")
    scrape_youtube()
    print("✅ All data scraping completed!")
