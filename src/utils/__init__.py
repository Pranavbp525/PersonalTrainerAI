# FILE: /web-scraping-project/web-scraping-project/src/utils/__init__.py
def is_valid_url(url):
    """Check if the URL is valid."""
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])