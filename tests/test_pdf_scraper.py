import pytest
import sys
sys.path.append('./src')
from src.data_pipeline.pdfs import extract_text_from_pdf, clean_text
from unittest.mock import patch, mock_open, MagicMock

@patch("builtins.open", new_callable=mock_open, read_data=b"dummy data")
@patch("src.data_pipeline.pdfs.PyPDF2.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader, mock_file):
    """Test extracting text from a PDF file."""
    # Mock the behavior of PyPDF2.PdfReader
    mock_pdf_reader.return_value.pages = [
        MagicMock(extract_text=lambda: "Page 1 text"),
        MagicMock(extract_text=lambda: "Page 2 text")
    ]

    # Call the function
    result = extract_text_from_pdf("dummy.pdf")

    # Assertions
    assert result == "Page 1 text Page 2 text"
    mock_file.assert_called_once_with("dummy.pdf", "rb")
    mock_pdf_reader.assert_called_once()

def test_clean_text():
    """Test cleaning extracted PDF text."""
    assert clean_text("Hello  ● World!") == "Hello World!"
