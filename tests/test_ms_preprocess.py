import pytest
import sys
sys.path.append('./src')
from data_pipeline.ms_preprocess import clean_text, extract_relevant_workout_data

def test_clean_text():
    """Test cleaning text to remove special characters."""
    assert clean_text("Hello, World! 123") == "Hello, World! 123"
    assert clean_text("café") == "café"  # Ensure Unicode normalization works

def test_extract_relevant_workout_data():
    """Test extracting relevant workout data from JSON."""
    json_data = {
        "workouts": [
            {
                "source": "Test Source",
                "title": "Test Workout",
                "url": "https://test.com/workout",
                "summary": {"Type": "Strength"},
                "description": "Workout description.",
                "exercises": [{"description": "Exercise details."}]
            }
        ]
    }
    extracted = extract_relevant_workout_data(json_data)
    assert len(extracted) == 1
    assert extracted[0]["source"] == "Test Source"
    assert "Workout description" in extracted[0]["description"]
