import pytest
import sys
sys.path.append('./src')
from data_pipeline.pdfs import extract_text_from_pdf, clean_text
from unittest.mock import patch, MagicMock

@patch("scripts.pdf_scraper.PyPDF2.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader):
    """Test extracting text from a PDF file."""
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Sample PDF Text")]

    text = extract_text_from_pdf("test.pdf")
    assert text == "Sample PDF Text"

def test_clean_text():
    """Test cleaning extracted PDF text."""
    assert clean_text("Hello  ● World!") == "Hello World!"
