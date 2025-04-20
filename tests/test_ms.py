import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from src.data_pipeline.ms import WorkoutScraper

@pytest.fixture
def scraper():
    """Fixture to create a WorkoutScraper instance."""
    return WorkoutScraper("https://www.muscleandstrength.com/workouts/men")

def test_fetch_html(scraper):
    """Test fetching HTML content from a valid URL."""
    with patch("urllib.request.OpenerDirector.open") as mock_open:
        mock_open.return_value.read.return_value.decode.return_value = "<html><body>Test</body></html>"
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
    <div class="node-stats-block">
        <ul>
            <li><span>Type:</span> Strength</li>
        </ul>
    </div>
    <div class="field field-name-body field-type-text-with-summary field-label-hidden">
        <p>Workout description here.</p>
        <table><tr><td><a href="/exercise/test">Test Exercise</a></td></tr></table>
    </div>
    """

    with patch.object(scraper, 'fetch_html', return_value=BeautifulSoup(html, "html.parser")):
        with patch.object(scraper, 'extract_exercise_details', return_value="Exercise description"):
            details = scraper.extract_detailed_info("https://example.com/workout")

            # Assertions
            assert details["title"] == "Test Workout"
            assert details["summary"]["Type:"] == "Strength"
            assert "Workout description here." in details["description"]
            assert len(details["exercises"]) == 1
            assert details["exercises"][0]["exercise"] == "Test Exercise"
            assert details["exercises"][0]["url"] == "https://www.muscleandstrength.com/exercise/test"
            assert details["exercises"][0]["description"] == "Exercise description"

def test_scrape_workouts(scraper):
    """Test scraping workouts with mocked HTML."""
    html = """
    <div class="cell small-12 bp600-6">
        <a href="/workout/test">Test Workout</a>
    </div>
    """
    with patch.object(scraper, 'fetch_html', return_value=BeautifulSoup(html, "html.parser")):
        with patch.object(scraper, 'extract_detailed_info', return_value={"title": "Test Workout"}):
            workouts = scraper.scrape_workouts()
            assert len(workouts) == 1
            assert workouts[0]["title"] == "Test Workout"