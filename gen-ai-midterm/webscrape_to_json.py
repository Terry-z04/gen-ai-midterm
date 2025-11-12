# ==========================
# webscrape_to_json.py
# Group 5 | Aneesha Dasari, Yumeng Li, Savita Moorthi, Huili Wang, Terry Zhuang
# Gen AI Principles | Utku Pamuksuz
# UChicago MS ADS
"""
This script crawls the University of Chicago's MS in Applied Data Science program website 
and extracts structured textual content for use in a Retrieval-Augmented Generation (RAG) system. 

Key Features:
- Recursively scrapes pages and internal links within the program domain.
- Preserves headings using Markdown-style formatting to maintain hierarchical structure.
- Captures **all page content**, deduplicates repeated text (header/footer/nav), ready for RAG.
- Stores page metadata (URL, parent URL, depth, last scraped) along with page content.
- Saves the final structured content as a JSON file suitable for downstream LLM/RAG processing.

Usage:
- Run the script directly to produce 'uchicago_msads_rag.json'.
"""
# ==========================

# ==========================
# Imports
# ==========================
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime

# ==========================
# Configuration
# ==========================
BASE_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
visited = set()  # track visited URLs to avoid loops
MAX_DEPTH = 2     # prevent excessive recursion

# ==========================
# Helper Functions
# ==========================
def get_soup(url):
    """Fetch a URL and return BeautifulSoup object."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_page_content(soup):
    """
    Extract all text from headings and paragraphs in order.
    - Keeps headings with Markdown-style prefixes.
    - Deduplicates repeated text.
    - Ignores nav, footer, header sections to reduce template repetition.
    """
    # Remove navigation, footer, and header sections to reduce template repetition
    for tag in soup.find_all(['nav', 'footer', 'header']):
        tag.decompose()

    content_parts = []
    seen_texts = set()

    for tag in soup.find_all(['h1','h2','h3','h4','h5','h6','p','li']):
        text = ' '.join(tag.get_text().split())  # clean text
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        if tag.name.startswith('h'):
            level = int(tag.name[1])
            prefix = '#' * level
            content_parts.append(f"{prefix} {text}")
        else:
            content_parts.append(text)

    return '\n\n'.join(content_parts)

def get_internal_links(soup, base_url):
    """Return internal links that belong to the site and have not been visited."""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href.split("#")[0])
        parsed = urlparse(full_url)
        if 'datascience.uchicago.edu' in parsed.netloc and full_url not in visited:
            links.add(full_url)
    return links

def build_node(url, depth=0, parent_url=None):
    """Recursively build JSON node for a page."""
    if url in visited or depth > MAX_DEPTH:
        return None

    soup = get_soup(url)
    if not soup:
        return None

    visited.add(url)
    node = {
        "title": soup.title.string.strip() if soup.title else url,
        "url": url,
        "parent_url": parent_url,
        "depth": depth,
        "last_scraped": datetime.utcnow().isoformat(),
        "content": extract_page_content(soup),
        "subsections": []
    }

    # Crawl internal links for subsections
    for link in get_internal_links(soup, url):
        child_node = build_node(link, depth=depth+1, parent_url=url)
        if child_node:
            node["subsections"].append(child_node)
    return node

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    data_tree = build_node(BASE_URL)

    output_file = "uchicago_msads_content_rag.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_tree, f, indent=2, ensure_ascii=False)

    print(f"Crawl complete. Data saved to: {output_file}")
    print(f"Total pages scraped: {len(visited)}")

    # Sample check: first page content snippet
    if data_tree:
        print("\n--- Sample Page Check ---")
        print(f"Title: {data_tree['title']}")
        print(f"URL: {data_tree['url']}")
        print(f"Content snippet:\n{data_tree['content'][:2000]}")