import pytest
import sys
import os
sys.path.append('./src')
from data_pipeline.ms import WorkoutScraper
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
@pytest.fixture
def scraper():
    return WorkoutScraper("https://www.muscleandstrength.com/workouts/men")

def test_fetch_html(scraper):
    """Test fetching HTML content from a valid URL."""
    with patch("scripts.ms.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><body>Test</body></html>"

        soup = scraper.fetch_html("https://example.com")
        assert soup is not None
        assert "Test" in soup.text

def test_extract_workout_links(scraper):
    """Test extracting workout links from a mock page."""
    html = """
    <div class="cell small-12 bp600-6">
        <a href="/workout/test">Test Workout</a>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    links = scraper.extract_workout_links(soup)
    assert len(links) == 1
    assert links[0].endswith("/workout/test")

def test_extract_exercise_details(scraper):
    """Test extracting exercise details from a mock exercise page."""
    html = """<div class="content clearfix">Workout Details</div>"""
    with patch.object(scraper, 'fetch_html', return_value=BeautifulSoup(html, "html.parser")):
        details = scraper.extract_exercise_details("https://example.com/exercise")
        assert details == "Workout Details"

def test_extract_detailed_info(scraper):
    """Test extracting full workout details."""
    html = """
    <div class="node-header"><h1>Test Workout</h1></div>
    <div class="node-stats-block"><li><span>Type:</span> Strength</li></div>
    <div class="field field-name-body field-type-text-with-summary">
        <table><a href="/exercise/test">Test Exercise</a></table>
    </div>
    """
    with patch.object(scraper, 'fetch_html', return_value=BeautifulSoup(html, "html.parser")):
        with patch.object(scraper, 'extract_exercise_details', return_value="Exercise description"):
            details = scraper.extract_detailed_info("https://example.com/workout")
            assert details["title"] == "Test Workout"
            assert details["summary"]["Type:"] == "Strength"
            assert len(details["exercises"]) == 1
            assert details["exercises"][0]["exercise"] == "Test Exercise"
