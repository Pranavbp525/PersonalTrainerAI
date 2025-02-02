import os
import json
from typing import List, Dict, Optional
from time import sleep
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# Configuration
API_KEY = os.getenv('youtube-api-key')
MAX_RESULTS_PER_PAGE = 50
RATE_LIMIT_DELAY = 1  # Delay in seconds between API calls
DATA_FOLDER = 'data'  # Folder to store JSON files

class YouTubeChannelTranscripts:
    def __init__(self, api_key: str):
        """Initialize the YouTube API client."""
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.setup_data_folder()

    def setup_data_folder(self):
        """Create data folder if it doesn't exist."""
        Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)

    def get_channel_id(self, channel_name: str) -> Optional[str]:
        """Get channel ID from channel name."""
        try:
            request = self.youtube.search().list(
                q=channel_name,
                type='channel',
                part='id',
                maxResults=1
            )
            response = request.execute()
            
            if response['items']:
                return response['items'][0]['id']['channelId']
            return None
            
        except HttpError as e:
            print(f"Error finding channel: {e}")
            return None

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get detailed information about a specific video."""
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                video = response['items'][0]
                return {
                    'id': video_id,
                    'title': video['snippet']['title'],
                    'description': video['snippet']['description'],
                    'published_at': video['snippet']['publishedAt'],
                    'channel_id': video['snippet']['channelId'],
                    'channel_title': video['snippet']['channelTitle'],
                    'duration': video['contentDetails']['duration'],
                    'view_count': video['statistics'].get('viewCount', 0),
                    'like_count': video['statistics'].get('likeCount', 0),
                    'comment_count': video['statistics'].get('commentCount', 0)
                }
            return None
            
        except HttpError as e:
            print(f"Error getting video details for {video_id}: {e}")
            return None

    def get_channel_videos(self, channel_id: str) -> List[Dict]:
        """Retrieve all videos from a channel."""
        videos = []
        next_page_token = None
        
        try:
            while True:
                request = self.youtube.search().list(
                    channelId=channel_id,
                    part='id,snippet',
                    order='date',
                    maxResults=MAX_RESULTS_PER_PAGE,
                    pageToken=next_page_token,
                    type='video'
                )
                response = request.execute()
                
                # Extract video information
                for item in response['items']:
                    video_id = item['id']['videoId']
                    video_details = self.get_video_details(video_id)
                    if video_details:
                        videos.append(video_details)
                
                # Check if there are more pages
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                sleep(RATE_LIMIT_DELAY)  # Rate limiting
                
            return videos
            
        except HttpError as e:
            print(f"Error retrieving videos: {e}")
            return videos

    def get_video_transcript(self, video_id: str) -> Optional[List[Dict]]:
        """Get transcript for a single video."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript_list
            
        except Exception as e:
            print(f"Error retrieving transcript for video {video_id}: {e}")
            return None

    def save_video_data(self, video_data: Dict, transcript: Optional[List[Dict]]):
        """Save video data and transcript to a JSON file."""
        video_id = video_data['id']
        
        # Create a clean filename
        clean_title = "".join(x for x in video_data['title'] if x.isalnum() or x in (' ', '-', '_'))
        filename = f"{video_id}_{clean_title[:50]}.json"
        filepath = os.path.join(DATA_FOLDER, filename)
        
        # Combine video data and transcript
        data = {
            'video_info': video_data,
            'transcript': transcript,
            'metadata': {
                'retrieved_at': datetime.now().isoformat(),
                'has_transcript': transcript is not None
            }
        }
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved data for video: {video_id}")
        except Exception as e:
            print(f"Error saving data for video {video_id}: {e}")

    def process_channel(self, channel_name: str):
        """Process all videos from a channel and save their data."""
        # Get channel ID
        channel_id = self.get_channel_id(channel_name)
        if not channel_id:
            print(f"Could not find channel: {channel_name}")
            return
            
        # Get all videos
        videos = self.get_channel_videos(channel_id)
        print(f"Found {len(videos)} videos")
        
        # Process each video
        for video in videos:
            video_id = video['id']
            transcript = self.get_video_transcript(video_id)
            self.save_video_data(video, transcript)
            sleep(RATE_LIMIT_DELAY)  # Rate limiting

def main():
    """Main execution function."""
    channel_name = "emi wong"
    processor = YouTubeChannelTranscripts(API_KEY)
    
    print(f"Starting to process channel: {channel_name}")
    processor.process_channel(channel_name)
    
    # Count processed files
    json_files = list(Path(DATA_FOLDER).glob('*.json'))
    print(f"\nProcessing complete:")
    print(f"Total JSON files created: {len(json_files)}")
    print(f"Data saved in: {os.path.abspath(DATA_FOLDER)}")

if __name__ == "__main__":
    main()