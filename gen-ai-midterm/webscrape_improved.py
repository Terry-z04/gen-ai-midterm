#!/usr/bin/env python3
"""
Improved Web Scraping Module with Firecrawl and LangChain Integration
Group 5 | UChicago MS-ADS RAG System

Features:
- Firecrawl API support for enhanced crawling
- Fallback to BeautifulSoup when Firecrawl unavailable
- LangChain RecursiveCharacterTextSplitter integration
- Better metadata structure and categorization
- Improved error handling and logging
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Import config for API keys
try:
    from config import Config
    FIRECRAWL_API_KEY = Config.FIRECRAWL_API_KEY
except:
    FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

# LangChain imports for chunking
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠ LangChain not installed. Chunking will use simple method.")
    LANGCHAIN_AVAILABLE = False

# Firecrawl imports
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    print("⚠ Firecrawl not installed. Will use BeautifulSoup fallback.")
    FIRECRAWL_AVAILABLE = False


class WebScraper:
    """Enhanced web scraper with multiple backend options"""
    
    def __init__(
        self,
        base_url: str,
        max_depth: int = 2,
        use_firecrawl: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited = set()
        self.use_firecrawl = use_firecrawl and FIRECRAWL_AVAILABLE and FIRECRAWL_API_KEY
        
        # Initialize Firecrawl if available
        if self.use_firecrawl:
            try:
                self.firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
                print("✓ Using Firecrawl for enhanced crawling")
            except Exception as e:
                print(f"⚠ Firecrawl initialization failed: {e}. Falling back to BeautifulSoup.")
                self.use_firecrawl = False
        
        # Initialize LangChain text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            print(f"✓ LangChain text splitter initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
        else:
            self.text_splitter = None
    
    def categorize_url(self, url: str) -> str:
        """Categorize URL based on path keywords"""
        url_lower = url.lower()
        if 'curriculum' in url_lower or 'courses' in url_lower:
            return 'curriculum'
        elif 'admission' in url_lower or 'apply' in url_lower:
            return 'admissions'
        elif 'faculty' in url_lower or 'staff' in url_lower:
            return 'faculty'
        elif 'career' in url_lower or 'outcomes' in url_lower:
            return 'careers'
        elif 'tuition' in url_lower or 'financial' in url_lower:
            return 'financial'
        else:
            return 'general'
    
    def scrape_with_firecrawl(self, url: str) -> Optional[Dict]:
        """Scrape using Firecrawl API"""
        try:
            # Firecrawl API uses different method names depending on version
            # Try the new API first
            try:
                result = self.firecrawl.scrape(url=url, formats=['markdown', 'html'])
            except:
                # Fallback to older API
                result = self.firecrawl.scrape_url(url, params={
                    'formats': ['markdown', 'html'],
                    'onlyMainContent': True
                })
            
            if result and 'markdown' in result:
                return {
                    'content': result['markdown'],
                    'title': result.get('metadata', {}).get('title', ''),
                    'description': result.get('metadata', {}).get('description', '')
                }
        except Exception as e:
            print(f"⚠ Firecrawl error for {url}: {e}")
        return None

    
    def scrape_with_beautifulsoup(self, url: str) -> Optional[Dict]:
        """Fallback scraping using BeautifulSoup"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove nav, footer, header
            for tag in soup.find_all(['nav', 'footer', 'header', 'script', 'style']):
                tag.decompose()
            
            # Extract title
            title = soup.title.string.strip() if soup.title else ''
            
            # Extract content
            content_parts = []
            seen_texts = set()
            
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = ' '.join(tag.get_text().split())
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                
                if tag.name.startswith('h'):
                    level = int(tag.name[1])
                    prefix = '#' * level
                    content_parts.append(f"{prefix} {text}")
                else:
                    content_parts.append(text)
            
            return {
                'content': '\n\n'.join(content_parts),
                'title': title,
                'description': ''
            }
        except Exception as e:
            print(f"⚠ BeautifulSoup error for {url}: {e}")
        return None
    
    def chunk_content(self, content: str) -> List[str]:
        """Chunk content using LangChain or simple method"""
        if self.text_splitter and LANGCHAIN_AVAILABLE:
            return self.text_splitter.split_text(content)
        else:
            # Simple chunking fallback
            chunk_size = 500
            words = content.split()
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunks.append(' '.join(words[i:i+chunk_size]))
            return chunks
    
    def get_internal_links(self, url: str, soup: BeautifulSoup = None) -> List[str]:
        """Extract internal links from a page"""
        if soup is None:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
            except:
                return []
        
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(url, href.split("#")[0])
            parsed = urlparse(full_url)
            
            # Only include links from the same domain
            if 'datascience.uchicago.edu' in parsed.netloc and full_url not in self.visited:
                links.add(full_url)
        
        return list(links)
    
    def scrape_page(self, url: str, depth: int = 0, parent_url: Optional[str] = None) -> Optional[Dict]:
        """Scrape a single page with metadata"""
        if url in self.visited or depth > self.max_depth:
            return None
        
        self.visited.add(url)
        print(f"{'  ' * depth}Scraping (depth {depth}): {url}")
        
        # Try Firecrawl first, fallback to BeautifulSoup
        if self.use_firecrawl:
            scraped_data = self.scrape_with_firecrawl(url)
        else:
            scraped_data = self.scrape_with_beautifulsoup(url)
        
        if not scraped_data or not scraped_data['content']:
            return None
        
        # Chunk the content
        chunks = self.chunk_content(scraped_data['content'])
        
        # Build page data
        page_data = {
            'url': url,
            'parent_url': parent_url,
            'title': scraped_data['title'],
            'description': scraped_data['description'],
            'category': self.categorize_url(url),
            'depth': depth,
            'crawl_date': datetime.utcnow().isoformat(),
            'content': scraped_data['content'],
            'chunks': chunks,
            'chunk_count': len(chunks),
            'subsections': []
        }
        
        # Crawl internal links
        internal_links = self.get_internal_links(url)
        for link in internal_links[:5]:  # Limit subsections to avoid explosion
            child_data = self.scrape_page(link, depth=depth+1, parent_url=url)
            if child_data:
                page_data['subsections'].append(child_data)
        
        return page_data
    
    def crawl(self) -> Dict:
        """Start crawling from base URL"""
        print(f"\n{'='*60}")
        print(f"Starting crawl from: {self.base_url}")
        print(f"Max depth: {self.max_depth}")
        print(f"Backend: {'Firecrawl' if self.use_firecrawl else 'BeautifulSoup'}")
        print(f"{'='*60}\n")
        
        data = self.scrape_page(self.base_url)
        
        print(f"\n{'='*60}")
        print(f"✓ Crawl complete!")
        print(f"  Total pages scraped: {len(self.visited)}")
        print(f"{'='*60}\n")
        
        return data
    
    def save_to_json(self, data: Dict, filename: str = "scraped_data.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Data saved to: {filename}")


def main():
    """Main execution"""
    BASE_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
    
    # Initialize scraper with Firecrawl support
    scraper = WebScraper(
        base_url=BASE_URL,
        max_depth=2,
        use_firecrawl=True,  # Set to False to force BeautifulSoup
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Crawl the website
    data = scraper.crawl()
    
    # Save results
    if data:
        scraper.save_to_json(data, "uchicago_msads_improved.json")
        
        # Print sample
        print("\n--- Sample Page Data ---")
        print(f"Title: {data.get('title', 'N/A')}")
        print(f"URL: {data.get('url', 'N/A')}")
        print(f"Category: {data.get('category', 'N/A')}")
        print(f"Chunks: {data.get('chunk_count', 0)}")
        print(f"\nFirst chunk preview:\n{data.get('chunks', [''])[0][:300]}...")


if __name__ == "__main__":
    main()
