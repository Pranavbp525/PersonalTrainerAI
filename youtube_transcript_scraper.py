from youtube_transcript_api import YouTubeTranscriptApi
import json

def fetch_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Could not retrieve transcript for video {video_id}: {e}")
        return None

# Fetch and save the transcript
if __name__ == "__main__":
    video_ids = ["2MoGxae-zyo"]
    for video_id in video_ids:
        transcript = fetch_youtube_transcript(video_id)
        if transcript:
            for entry in transcript:
                print(f"{entry['start']}: {entry['text']}")
