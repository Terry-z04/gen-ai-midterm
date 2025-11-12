#!/usr/bin/env python3
"""
Sync Dynamic Data to Static Vector Store
Group 5 | UChicago MS-ADS RAG System

This script:
1. Scrapes URLs using enhanced Firecrawl configuration
2. Adds scraped content to ChromaDB static vector store
3. Updates both OpenAI and finetuned embedding stores

Usage:
    python sync_dynamic_to_static.py --urls URL1 URL2 ...
    python sync_dynamic_to_static.py --use-defaults  # Use default fallback URLs
    python sync_dynamic_to_static.py --clear-cache   # Clear cache and re-scrape
"""

import argparse
import json
from typing import List, Dict
from datetime import datetime
import chromadb
from chromadb.config import Settings

# Import our components
from dynamic_retrieval import DynamicRetriever
from config import Config

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠ OpenAI not available")
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ SentenceTransformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DynamicToStaticSync:
    """Sync dynamically retrieved data to static vector stores"""
    
    # Default URLs to scrape (matches fallback guardrail)
    DEFAULT_URLS = [
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/",
        "https://datascience.uchicago.edu/education/tuition-fees-aid/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/online-program/",
    ]
    
    def __init__(
        self,
        openai_db_path: str = "./chroma_db_openai",
        finetuned_db_path: str = "./chroma_db_finetuned",
        finetuned_model_path: str = "../finetune-embedding/exp_finetune"
    ):
        """
        Initialize sync system
        
        Args:
            openai_db_path: Path to OpenAI embedding ChromaDB
            finetuned_db_path: Path to finetuned embedding ChromaDB
            finetuned_model_path: Path to finetuned model
        """
        self.openai_db_path = openai_db_path
        self.finetuned_db_path = finetuned_db_path
        self.finetuned_model_path = finetuned_model_path
        
        print(f"\n{'='*70}")
        print("Dynamic to Static Sync Initializing")
        print(f"{'='*70}\n")
        
        # Initialize dynamic retriever
        self.retriever = DynamicRetriever(use_embedding=False)
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            print("✓ OpenAI client initialized")
        
        # Initialize finetuned model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.finetuned_model = SentenceTransformer(self.finetuned_model_path)
                print(f"✓ Finetuned model loaded from {self.finetuned_model_path}")
            except Exception as e:
                print(f"⚠ Could not load finetuned model: {e}")
                self.finetuned_model = None
        else:
            self.finetuned_model = None
    
    def scrape_urls(
        self,
        urls: List[str],
        use_cache: bool = False
    ) -> Dict:
        """
        Scrape URLs with enhanced configuration
        
        Args:
            urls: List of URLs to scrape
            use_cache: Whether to use cache
            
        Returns:
            Dict with scraped results
        """
        print(f"\n{'='*70}")
        print(f"Scraping {len(urls)} URLs")
        print(f"{'='*70}\n")
        
        response = self.retriever.dynamic_retrieve(
            query="Full content scrape for static store",
            urls=urls,
            use_cache=use_cache
        )
        
        return response
    
    def embed_with_openai(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Embed text using OpenAI"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available")
        
        response = self.openai_client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    
    def embed_with_finetuned(self, text: str) -> List[float]:
        """Embed text using finetuned model"""
        if not self.finetuned_model:
            raise ValueError("Finetuned model not loaded")
        
        return self.finetuned_model.encode(text, convert_to_tensor=False).tolist()
    
    def add_to_openai_store(
        self,
        scraped_data: Dict,
        collection_name: str = "uchicago_msads_docs"
    ) -> int:
        """
        Add scraped data to OpenAI embedding store
        
        Args:
            scraped_data: Data from scraping
            collection_name: ChromaDB collection name
            
        Returns:
            Number of chunks added
        """
        print(f"\n{'='*70}")
        print(f"Adding to OpenAI Embedding Store")
        print(f"{'='*70}\n")
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=self.openai_db_path)
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            print(f"✓ Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(name=collection_name)
            print(f"✓ Created new collection: {collection_name}")
        
        # Process each scraped URL
        total_added = 0
        for result in scraped_data['results']:
            url = result['url']
            chunks = result['chunks']
            
            print(f"\nProcessing: {url}")
            print(f"  Chunks: {len(chunks)}")
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = f"{url}_chunk_{i}_{datetime.now().timestamp()}"
                ids.append(chunk_id)
                
                # Embed chunk
                embedding = self.embed_with_openai(chunk)
                embeddings.append(embedding)
                
                # Metadata
                metadata = {
                    'url': url,
                    'chunk_index': i,
                    'scraped_at': result.get('scraped_at', ''),
                    'title': result.get('title', ''),
                    'source': 'dynamic_sync'
                }
                metadatas.append(metadata)
                
                # Document text
                documents.append(chunk)
            
            # Add to collection
            if ids:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                total_added += len(ids)
                print(f"  ✓ Added {len(ids)} chunks")
        
        print(f"\n✓ Total chunks added to OpenAI store: {total_added}")
        return total_added
    
    def add_to_finetuned_store(
        self,
        scraped_data: Dict,
        collection_name: str = "uchicago_msads_docs"
    ) -> int:
        """
        Add scraped data to finetuned embedding store
        
        Args:
            scraped_data: Data from scraping
            collection_name: ChromaDB collection name
            
        Returns:
            Number of chunks added
        """
        print(f"\n{'='*70}")
        print(f"Adding to Finetuned Embedding Store")
        print(f"{'='*70}\n")
        
        if not self.finetuned_model:
            print("⚠ Finetuned model not available, skipping")
            return 0
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=self.finetuned_db_path)
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            print(f"✓ Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(name=collection_name)
            print(f"✓ Created new collection: {collection_name}")
        
        # Process each scraped URL
        total_added = 0
        for result in scraped_data['results']:
            url = result['url']
            chunks = result['chunks']
            
            print(f"\nProcessing: {url}")
            print(f"  Chunks: {len(chunks)}")
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = f"{url}_chunk_{i}_{datetime.now().timestamp()}"
                ids.append(chunk_id)
                
                # Embed chunk
                embedding = self.embed_with_finetuned(chunk)
                embeddings.append(embedding)
                
                # Metadata
                metadata = {
                    'url': url,
                    'chunk_index': i,
                    'scraped_at': result.get('scraped_at', ''),
                    'title': result.get('title', ''),
                    'source': 'dynamic_sync'
                }
                metadatas.append(metadata)
                
                # Document text
                documents.append(chunk)
            
            # Add to collection
            if ids:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                total_added += len(ids)
                print(f"  ✓ Added {len(ids)} chunks")
        
        print(f"\n✓ Total chunks added to finetuned store: {total_added}")
        return total_added
    
    def sync(
        self,
        urls: List[str],
        use_cache: bool = False,
        update_openai: bool = True,
        update_finetuned: bool = True
    ) -> Dict:
        """
        Main sync operation
        
        Args:
            urls: URLs to scrape
            use_cache: Whether to use cache
            update_openai: Whether to update OpenAI store
            update_finetuned: Whether to update finetuned store
            
        Returns:
            Dict with sync results
        """
        print(f"\n{'='*70}")
        print("STARTING DYNAMIC TO STATIC SYNC")
        print(f"{'='*70}")
        print(f"URLs to process: {len(urls)}")
        print(f"Update OpenAI store: {update_openai}")
        print(f"Update finetuned store: {update_finetuned}")
        print(f"Use cache: {use_cache}")
        print(f"{'='*70}\n")
        
        # Step 1: Scrape URLs
        scraped_data = self.scrape_urls(urls, use_cache=use_cache)
        
        if not scraped_data.get('results'):
            print("✗ No data scraped, aborting sync")
            return {'success': False, 'error': 'No data scraped'}
        
        results = {
            'success': True,
            'urls_processed': scraped_data['num_retrieved'],
            'openai_chunks_added': 0,
            'finetuned_chunks_added': 0
        }
        
        # Step 2: Update OpenAI store
        if update_openai and OPENAI_AVAILABLE:
            try:
                openai_added = self.add_to_openai_store(scraped_data)
                results['openai_chunks_added'] = openai_added
            except Exception as e:
                print(f"✗ Error updating OpenAI store: {e}")
                results['openai_error'] = str(e)
        
        # Step 3: Update finetuned store
        if update_finetuned and self.finetuned_model:
            try:
                finetuned_added = self.add_to_finetuned_store(scraped_data)
                results['finetuned_chunks_added'] = finetuned_added
            except Exception as e:
                print(f"✗ Error updating finetuned store: {e}")
                results['finetuned_error'] = str(e)
        
        # Print summary
        print(f"\n{'='*70}")
        print("SYNC COMPLETE")
        print(f"{'='*70}")
        print(f"URLs processed: {results['urls_processed']}")
        print(f"OpenAI chunks added: {results['openai_chunks_added']}")
        print(f"Finetuned chunks added: {results['finetuned_chunks_added']}")
        print(f"{'='*70}\n")
        
        return results


def main():
    """Main execution"""
    ap = argparse.ArgumentParser(
        description="Sync dynamically retrieved data to static vector stores"
    )
    ap.add_argument(
        "--urls",
        nargs='+',
        help="URLs to scrape and add to vector stores"
    )
    ap.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use default fallback URLs"
    )
    ap.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache and force re-scrape"
    )
    ap.add_argument(
        "--openai-only",
        action="store_true",
        help="Only update OpenAI store"
    )
    ap.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Only update finetuned store"
    )
    args = ap.parse_args()
    
    # Determine URLs to process
    if args.use_defaults:
        urls = DynamicToStaticSync.DEFAULT_URLS
        print(f"Using default URLs ({len(urls)} URLs)")
    elif args.urls:
        urls = args.urls
        print(f"Using provided URLs ({len(urls)} URLs)")
    else:
        print("Error: Must provide --urls or --use-defaults")
        return 1
    
    # Initialize sync
    sync = DynamicToStaticSync()
    
    # Run sync
    results = sync.sync(
        urls=urls,
        use_cache=not args.clear_cache,
        update_openai=not args.finetuned_only,
        update_finetuned=not args.openai_only
    )
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
