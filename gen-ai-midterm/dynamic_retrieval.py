#!/usr/bin/env python3
"""
Dynamic Retrieval with FireCrawl Integration
Group 5 | UChicago MS-ADS RAG System

Features:
- On-demand web scraping with Firecrawl
- Dynamic content fetching based on queries
- Clean Markdown extraction and metadata
- Optional OpenAI embedding with caching
- LangChain tool integration
- Combines static (ChromaDB) + dynamic (Firecrawl) retrieval
"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime

# OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠ OpenAI not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# Firecrawl
try:
    from firecrawl import Firecrawl
    FIRECRAWL_AVAILABLE = True
except ImportError:
    print("⚠ Firecrawl not installed. Run: pip install firecrawl-py")
    FIRECRAWL_AVAILABLE = False

# LangChain (optional)
try:
    from langchain.tools import Tool
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠ LangChain not installed. Some features unavailable.")
    LANGCHAIN_AVAILABLE = False

# Config
try:
    from config import Config
    FIRECRAWL_API_KEY = Config.FIRECRAWL_API_KEY
    OPENAI_API_KEY = Config.OPENAI_API_KEY
except:
    FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class DynamicRetriever:
    """Dynamic retrieval with on-demand Firecrawl scraping"""
    
    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cache_dir: str = "./dynamic_cache",
        use_embedding: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize dynamic retriever
        
        Args:
            firecrawl_api_key: Firecrawl API key
            openai_api_key: OpenAI API key (for embeddings)
            cache_dir: Directory for caching scraped content
            use_embedding: Whether to embed scraped content
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
        """
        self.firecrawl_api_key = firecrawl_api_key or FIRECRAWL_API_KEY
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.cache_dir = cache_dir
        self.use_embedding = use_embedding
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize Firecrawl
        if not FIRECRAWL_AVAILABLE:
            raise ImportError("Firecrawl not installed. Run: pip install firecrawl-py")
        
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY not set")
        
        try:
            self.firecrawl = Firecrawl(api_key=self.firecrawl_api_key)
            print("✓ Firecrawl initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Firecrawl: {e}")
        
        # Initialize OpenAI if embedding enabled
        if self.use_embedding:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed. Run: pip install openai")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set for embeddings")
            
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            print("✓ OpenAI initialized for embeddings")
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            print(f"✓ Text splitter initialized (chunk_size={chunk_size})")
        else:
            self.text_splitter = None
            print("⚠ LangChain not available, using simple chunking")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, url: str) -> Optional[Dict]:
        """Load content from cache"""
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"✓ Loaded from cache: {url}")
                return cached_data
            except Exception as e:
                print(f"⚠ Cache load error: {e}")
        
        return None
    
    def _save_to_cache(self, url: str, data: Dict):
        """Save content to cache"""
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved to cache: {url}")
        except Exception as e:
            print(f"⚠ Cache save error: {e}")
    
    def scrape_with_tabs(
        self,
        url: str,
        tab_actions: List[Dict[str, Any]],
        formats: List[str] = ['markdown', 'html'],
        only_main_content: bool = False,
        max_age: Optional[int] = None,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Scrape URL with tab/accordion interactions using Firecrawl actions
        
        Args:
            url: URL to scrape
            tab_actions: List of actions to perform (click tabs, scroll, etc.)
                Each dict can have: {"click": "selector", "wait": 2000, "scroll": "down"}
            formats: Formats to extract
            only_main_content: Extract only main content
            max_age: Cache age in milliseconds
            use_cache: Whether to use cache
        
        Returns:
            Dict with content, metadata, and actions performed
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(url)
            if cached and cached.get('tab_actions_performed'):
                print(f"✓ Loaded from cache (with tab actions): {url}")
                return cached
        
        print(f"🔥 Scraping with Firecrawl + Tab Actions: {url}")
        print(f"   Actions to perform: {len(tab_actions)}")
        
        try:
            # Build actions list for Firecrawl
            actions = []
            
            # Initial wait for JS to load
            actions.append({"type": "wait", "milliseconds": 3000})
            
            # Process each tab action
            for i, action in enumerate(tab_actions, 1):
                print(f"   Action {i}: {action}")
                
                # Scroll action
                if "scroll" in action:
                    actions.append({
                        "type": "scroll",
                        "direction": action["scroll"]
                    })
                
                # Click action
                if "click" in action:
                    actions.append({
                        "type": "click",
                        "selector": action["click"]
                    })
                
                # Execute JavaScript action
                if "executeJavascript" in action:
                    actions.append({
                        "type": "executeJavascript",
                        "script": action["executeJavascript"]
                    })
                
                # Wait after action
                wait_ms = action.get("wait", 2000)
                actions.append({"type": "wait", "milliseconds": wait_ms})
            
            # Final wait to ensure all content loaded
            actions.append({"type": "wait", "milliseconds": 2000})
            
            # Call Firecrawl with actions
            result = self.firecrawl.scrape(
                url,
                formats=formats,
                only_main_content=only_main_content,
                wait_for=2000,
                actions=actions,
                max_age=max_age if max_age is not None else 0
            )
            
            if not result:
                print(f"✗ No result from Firecrawl")
                return None
            
            # Parse result
            if hasattr(result, "markdown"):
                markdown_content = getattr(result, "markdown", "") or ""
                html_content = (
                    getattr(result, "html", "")
                    or getattr(result, "rawHtml", "")
                    or ""
                )
                metadata = getattr(result, "metadata", {}) or {}
            elif isinstance(result, dict):
                data = result.get("data", result)
                markdown_content = data.get("markdown", "") or ""
                html_content = data.get("html", "") or data.get("rawHtml", "") or ""
                metadata = data.get("metadata", {}) or {}
            else:
                print(f"✗ Unexpected result type: {type(result)}")
                return None
            
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Build response
            data = {
                'url': url,
                'markdown': markdown_content,
                'html': html_content,
                'metadata': metadata,
                'tab_actions_performed': tab_actions,
                'num_actions': len(tab_actions),
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Save to cache
            if use_cache:
                self._save_to_cache(url, data)
            
            print(f"✓ Scraped with tabs successfully: {len(data['markdown'])} chars")
            print(f"   Tab actions performed: {len(tab_actions)}")
            return data
            
        except Exception as e:
            print(f"✗ Firecrawl error with tabs: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def scrape_url(
        self,
        url: str,
        use_cache: bool = True,
        formats: List[str] = ['markdown', 'html']
    ) -> Optional[Dict]:
        """
        Scrape URL with Firecrawl
        
        Args:
            url: URL to scrape
            use_cache: Whether to use cache
            formats: Formats to extract
        
        Returns:
            Dict with content, metadata, etc.
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(url)
            if cached:
                return cached
        
        # Scrape with Firecrawl
        print(f"🔥 Scraping with Firecrawl: {url}")
        try:
            # Firecrawl SDK - pass individual keyword arguments
            result = self.firecrawl.scrape(
                url,
                formats=formats,              # e.g. ['markdown', 'html']
                only_main_content=False,      # get full page, not just main content
                wait_for=5000,                # wait for JS to render
                max_age=0,                    # always fresh while debugging
            )
            
            if not result:
                print(f"✗ No result from Firecrawl")
                return None
            
            # Handle both object and dict responses
            if hasattr(result, "markdown"):
                # New-style Firecrawl object
                markdown_content = getattr(result, "markdown", "") or ""
                html_content = (
                    getattr(result, "html", "") 
                    or getattr(result, "rawHtml", "") 
                    or ""
                )
                metadata = getattr(result, "metadata", {}) or {}
            
            elif isinstance(result, dict):
                # Older / scrape_url style or raw dict
                data = result.get("data", result)  # some APIs wrap in "data"
                markdown_content = data.get("markdown", "") or ""
                html_content = data.get("html", "") or data.get("rawHtml", "") or ""
                metadata = data.get("metadata", {}) or {}
            
            else:
                print(f"✗ Unexpected result type: {type(result)}")
                return None
            
            # Ensure metadata is dict
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Check if we got meaningful content
            if len(markdown_content) < 200:
                print(f"⚠ Warning: Very short content ({len(markdown_content)} chars)")
                print(f"   Content preview: {markdown_content[:100]}...")
            
            # Extract data
            data = {
                'url': url,
                'markdown': markdown_content,
                'html': html_content,
                'metadata': metadata,
                'scraped_at': datetime.utcnow().isoformat()
            }

            
            # Save to cache
            if use_cache:
                self._save_to_cache(url, data)
            
            print(f"✓ Scraped successfully: {len(data['markdown'])} chars")
            return data
            
        except Exception as e:
            print(f"✗ Firecrawl error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def chunk_content(self, content: str) -> List[str]:
        """Chunk content into smaller pieces"""
        if not content:
            return []
        
        if self.text_splitter:
            return self.text_splitter.split_text(content)
        else:
            # Simple word-based chunking
            words = content.split()
            chunks = []
            chunk_size = 500
            for i in range(0, len(words), chunk_size):
                chunks.append(' '.join(words[i:i+chunk_size]))
            return chunks
    
    def embed_text(self, text: str, model: str = "text-embedding-ada-002") -> Optional[List[float]]:
        """Embed text using OpenAI"""
        if not self.use_embedding or not OPENAI_AVAILABLE:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠ Embedding error: {e}")
            return None
    
    def dynamic_retrieve(
        self,
        query: str,
        urls: List[str],
        use_cache: bool = True
    ) -> Dict:
        """
        Dynamically retrieve and process content from URLs
        
        Args:
            query: User query
            urls: List of URLs to scrape
            use_cache: Whether to use cache
        
        Returns:
            Dict with query, results, and metadata
        """
        print(f"\n{'='*60}")
        print(f"Dynamic Retrieval")
        print(f"Query: {query}")
        print(f"URLs to scrape: {len(urls)}")
        print(f"{'='*60}\n")
        
        results = []
        
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Processing: {url}")
            
            # Scrape URL
            scraped = self.scrape_url(url, use_cache=use_cache)
            if not scraped:
                continue
            
            # Extract content
            content = scraped.get('markdown', '')
            if not content:
                content = scraped.get('html', '')
            
            # Chunk content
            chunks = self.chunk_content(content)
            print(f"      Created {len(chunks)} chunks")
            
            # Optionally embed chunks
            if self.use_embedding:
                print(f"      Embedding chunks...")
                embeddings = []
                for chunk in chunks:
                    emb = self.embed_text(chunk)
                    if emb:
                        embeddings.append(emb)
                print(f"      Embedded {len(embeddings)} chunks")
            else:
                embeddings = None
            
            # Build result
            results.append({
                'url': url,
                'title': scraped.get('metadata', {}).get('title', url),
                'content': content,
                'chunks': chunks,
                'embeddings': embeddings,
                'metadata': scraped.get('metadata', {}),
                'scraped_at': scraped.get('scraped_at')
            })
        
        print(f"\n✓ Dynamic retrieval complete: {len(results)} URLs processed\n")
        
        return {
            'query': query,
            'num_urls': len(urls),
            'num_retrieved': len(results),
            'results': results
        }
    
    def create_langchain_tool(self) -> Optional[Any]:
        """Create LangChain Tool for dynamic retrieval"""
        if not LANGCHAIN_AVAILABLE:
            print("⚠ LangChain not available")
            return None
        
        def retrieve_urls(url_list: str) -> str:
            """Retrieve content from comma-separated URLs"""
            urls = [u.strip() for u in url_list.split(',')]
            response = self.dynamic_retrieve(
                query="LangChain tool query",
                urls=urls
            )
            
            # Format response
            output = []
            for result in response['results']:
                output.append(f"URL: {result['url']}")
                output.append(f"Title: {result['title']}")
                output.append(f"Content preview: {result['content'][:500]}...")
                output.append("-" * 40)
            
            return '\n'.join(output)
        
        tool = Tool(
            name="dynamic_firecrawl_retrieval",
            func=retrieve_urls,
            description="Dynamically scrape and retrieve content from URLs using Firecrawl. "
                       "Input should be comma-separated URLs."
        )
        
        print("✓ LangChain Tool created")
        return tool


def main():
    """Main execution"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Dynamic retrieval with Firecrawl")
    ap.add_argument("--urls", nargs='+', required=True, help="URLs to scrape")
    ap.add_argument("--query", default="Dynamic retrieval query", help="Query context")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache")
    ap.add_argument("--embed", action="store_true", help="Enable embedding")
    args = ap.parse_args()
    
    # Initialize retriever
    retriever = DynamicRetriever(use_embedding=args.embed)
    
    # Retrieve
    response = retriever.dynamic_retrieve(
        query=args.query,
        urls=args.urls,
        use_cache=not args.no_cache
    )
    
    # Print results
    print("="*60)
    print("RESULTS")
    print("="*60)
    for result in response['results']:
        print(f"\nURL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Chunks: {len(result['chunks'])}")
        print(f"Content preview:\n{result['content'][:500]}...")
        print("-"*60)


if __name__ == "__main__":
    main()
