import pytest
import sys
# sys.path.append('../src')
from src.data_pipeline.vector_db import split_text
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

def test_split_text():
    """Test the split_text function."""
    # Mock input data
    mock_data = [
        {
            "description": "This is a test description. It has multiple sentences. "
                           "This is to test the splitting functionality.",
            "source": "test_source",
            "title": "Test Title",
            "url": "http://example.com"
        }
    ]

    # Expected output
    expected_output = [
        {
            "source": "test_source",
            "title": "Test Title",
            "url": "http://example.com",
            "chunk_id": "Test Title_0",
            "chunk": "This is a test description"
        },
        {
            "source": "test_source",
            "title": "Test Title",
            "url": "http://example.com",
            "chunk_id": "Test Title_1",
            "chunk": ". It has multiple sentences"
        },
        {
            "source": "test_source",
            "title": "Test Title",
            "url": "http://example.com",
            "chunk_id": "Test Title_2",
            "chunk": ". This is to test the splitting functionality."
        }
    ]

    # Call the function
    result = split_text(mock_data, chunk_size=50, chunk_overlap=0)

    # Assertions
    assert len(result) == len(expected_output)
    for res, exp in zip(result, expected_output):
        assert res == exp
