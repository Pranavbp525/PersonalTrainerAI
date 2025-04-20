import pytest
import os
import sys
import json
from pathlib import Path

from src.data_pipeline.ms_preprocess import (
    clean_text,
    format_summary,
    extract_relevant_workout_data,
)

@pytest.fixture
def mock_workout_data():
    return {
        "workouts": [
            {
                "source": "Muscle & Strength",
                "title": "Beginner Workout",
                "url": "https://example.com/workout1",
                "summary": {
                    "Type": "Full Body",
                    "Duration": "45 mins",
                    "Workout PDF": "https://example.com/pdf"
                },
                "description": "A great beginner routine.",
                "exercises": [
                    {
                        "exercise": "Push Up",
                        "description": "Do 3 sets of 10 reps."
                    },
                    {
                        "exercise": "Squat",
                        "description": "Bodyweight squats for 15 reps."
                    }
                ]
            }
        ]
    }

def test_clean_text():
    dirty_text = "Café"
    result = clean_text(dirty_text)
    assert result == "Cafe"

def test_format_summary():
    summary = {
        "Type": "Strength",
        "Workout PDF": "ignore this",
        "Duration": "30 mins"
    }
    formatted = format_summary(summary)
    assert "Type: Strength" in formatted
    assert "Duration: 30 mins" in formatted
    assert "Workout PDF" not in formatted

def test_extract_relevant_workout_data(mock_workout_data):
    extracted = extract_relevant_workout_data(mock_workout_data)

    assert len(extracted) == 1
    workout = extracted[0]
    assert workout["source"] == "Muscle & Strength"
    assert workout["title"] == "Beginner Workout"
    assert workout["url"] == "https://example.com/workout1"
    assert "Full Body" in workout["description"]
    assert "A great beginner routine." in workout["description"]
    assert "3 sets of 10 reps" in workout["description"]
    assert "Bodyweight squats" in workout["description"]
