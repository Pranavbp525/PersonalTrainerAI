import os
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

DATA_FOLDER = 'data'
API_KEY = 'YOUR_API_KEY'

class YouTubeChannelTranscripts:
    def __init__(self, api_key: str):
        """Initialize the YouTube API client."""
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.setup_data_folder()

    def setup_data_folder(self):
        """Create data folder if it doesn't exist."""
        Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)

    def get_channel_id(self, channel_name: str):
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

def main():
    """Main execution function."""
    channel_name = "Chloe Ting"
    processor = YouTubeChannelTranscripts(API_KEY)
    
    channel_id = processor.get_channel_id(channel_name)
    if channel_id:
        print(f"Channel ID for '{channel_name}': {channel_id}")
    else:
        print(f"Channel '{channel_name}' not found.")

if __name__ == "__main__":
    main()
