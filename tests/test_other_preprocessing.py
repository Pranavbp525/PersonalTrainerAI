import pytest
import json
import os
import sys
sys.path.append('./src')
from data_pipeline.other_preprocesing import preprocess_json_other_files, clean_text
from unittest.mock import patch, mock_open

# Sample raw JSON data
SAMPLE_JSON = [
    {"description": "Hello!! This is @a test..."},
    {"description": "Another text with   extra spaces!!"},
    {"description": "Special characters → removed?"}
]

# Expected cleaned JSON data
EXPECTED_JSON = [
    {"description": "Hello This is a test"},
    {"description": "Another text with extra spaces"},
    {"description": "Special characters removed"}
]

@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(SAMPLE_JSON))
@patch("json.dump")  # Mock json.dump to avoid writing files
@patch("os.path.exists", return_value=True)  # Mock path existence
def test_preprocess_json_files(mock_exists, mock_json_dump, mock_file):
    """Test that JSON preprocessing reads, cleans, and writes data correctly."""
    preprocess_json_files()

    # Ensure that json.dump was called with the expected cleaned data
    cleaned_data = mock_json_dump.call_args[0][0]
    assert cleaned_data == EXPECTED_JSON, "Processed data did not match expected output"

def test_clean_text():
    """Test text cleaning function to remove special characters & extra spaces."""
    assert clean_text("Hello!! This is @a test...") == "Hello This is a test"
    assert clean_text("Another    text   with   spaces") == "Another text with spaces"
    assert clean_text("Special characters → removed?") == "Special characters removed"
