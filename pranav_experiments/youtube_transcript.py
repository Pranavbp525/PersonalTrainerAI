

from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import json
import os

# YouTube Data API setup
API_KEY = 'AIzaSyA6VanJo1cXpcHNLjc2cEef2ZErAHB3k04'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_videos(channel_name, days_back=365):
    """Get videos from a channel uploaded in the last X days"""
    search_response = youtube.search().list(
        q=channel_name,
        type='channel',
        part='id'
    ).execute()
    
    if not search_response['items']:
        return []
        
    channel_id = search_response['items'][0]['id']['channelId']
    
    # Calculate date one week ago
    one_week_ago = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
    
    # Get videos from the channel
    search_response = youtube.search().list(
        channelId=channel_id,
        publishedAfter=one_week_ago,
        type='video',
        part='id,snippet',
        maxResults=50
    ).execute()
    
    videos = []
    for item in search_response['items']:
        videos.append({
            'video_id': item['id']['videoId'],
            'title': item['snippet']['title'],
            'published_at': item['snippet']['publishedAt']
        })
    
    return videos

def get_transcript(video_id):
    """Get transcript for a single video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"An error occurred for video {video_id}: {str(e)}")
        return None

def save_channel_transcripts(channels):
    """Process multiple channels and save their transcripts"""
    for channel in channels:
        print(f"Processing channel: {channel}")
        videos = get_channel_videos(channel)
        
        channel_data = []
        for video in videos:
            print(f"Processing video: {video['title']}")
            transcript = get_transcript(video['video_id'])
            if transcript:
                channel_data.append({
                    'video_id': video['video_id'],
                    'title': video['title'],
                    'published_at': video['published_at'],
                    'transcript': transcript
                })
        
        # Save as JSON
        if channel_data:
            filename = f"{channel.replace(' ', '_')}.json"
            with open(filename, 'w') as f:
                json.dump(channel_data, f, indent=2)
            print(f"Saved {len(channel_data)} videos to {filename}")

# Example usage
channels = ["House of Hypertrophy", "Jeff Nippard"]  # Add your channel names here
save_channel_transcripts(channels)
