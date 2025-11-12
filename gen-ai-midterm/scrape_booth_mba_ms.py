#!/usr/bin/env python3
"""
Scrape Chicago Booth MBA/MS Applied Data Science Joint Degree Page
Group 5 | UChicago MS-ADS RAG System

This script scrapes the Chicago Booth MBA/MS-ADS collaboration page and loads it
into the ChromaDB vector store for RAG retrieval.
"""

import json
import os
from datetime import datetime
from dynamic_retrieval import DynamicRetriever
from load_to_chromadb_openai import FinetunedChromaLoader


# Chicago Booth MBA/MS-ADS Joint Degree Page
BOOTH_MBA_MS_PAGE = {
    "url": "https://www.chicagobooth.edu/mba/joint-degree/mba-ms-applied-data-science?sc_lang=en",
    "title": "MBA/MS in Applied Data Science - Joint Degree - Chicago Booth",
    "category": "joint_program",
    "description": "Joint degree program combining Chicago Booth MBA with UChicago MS in Applied Data Science"
}


def scrape_booth_page():
    """Scrape the Chicago Booth MBA/MS-ADS joint degree page"""
    
    print("\n" + "="*80)
    print("SCRAPING CHICAGO BOOTH MBA/MS-ADS PAGE")
    print("="*80)
    
    print(f"\nTarget Page: {BOOTH_MBA_MS_PAGE['title']}")
    print(f"URL: {BOOTH_MBA_MS_PAGE['url']}")
    print(f"Category: {BOOTH_MBA_MS_PAGE['category']}")
    
    # Initialize dynamic retriever
    print("\nInitializing Firecrawl scraper...")
    retriever = DynamicRetriever()
    
    # Scrape the page
    print("\nScraping page with Firecrawl...")
    result = retriever.scrape_url(BOOTH_MBA_MS_PAGE['url'], use_cache=False)
    
    if not result:
        print("\n✗ Failed to scrape the page")
        return None
    
    # Display scraping results
    content_len = len(result['markdown'])
    print(f"\n✓ Successfully scraped!")
    print(f"  Content length: {content_len:,} characters")
    print(f"  Scraped at: {result['scraped_at']}")
    
    # Display metadata if available
    if result.get('metadata'):
        metadata = result['metadata']
        print(f"\nPage Metadata:")
        if metadata.get('title'):
            print(f"  Title: {metadata['title']}")
        if metadata.get('description'):
            print(f"  Description: {metadata['description']}")
        if metadata.get('language'):
            print(f"  Language: {metadata['language']}")
    
    # Create document structure
    document = {
        "title": BOOTH_MBA_MS_PAGE['title'],
        "url": BOOTH_MBA_MS_PAGE['url'],
        "parent_url": "",
        "category": BOOTH_MBA_MS_PAGE['category'],
        "description": BOOTH_MBA_MS_PAGE['description'],
        "depth": 0,
        "content": result['markdown'],
        "crawl_date": datetime.utcnow().isoformat(),
        "metadata": result.get('metadata', {}),
        "subsections": []
    }
    
    # Save individual JSON for inspection
    filename = 'chicago_booth_mba_ms_ads.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved scraped content to: {filename}")
    
    # Display content preview
    print(f"\nContent Preview (first 500 chars):")
    print("-" * 80)
    print(result['markdown'][:500])
    print("...")
    print("-" * 80)
    
    return document


def load_to_vector_store(document):
    """Load scraped document to ChromaDB"""
    
    print("\n" + "="*80)
    print("LOADING TO VECTOR STORE")
    print("="*80)
    
    # Configuration
    PERSIST_DIRECTORY = "./chroma_db_finetuned"
    COLLECTION_NAME = "uchicago_msads_finetuned"
    MODEL_PATH = "../finetune-embedding/exp_finetune"
    
    # Initialize loader
    print("\n1. Initializing ChromaDB loader with finetuned embeddings...")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Persist directory: {PERSIST_DIRECTORY}")
    print(f"   Embedding model: {MODEL_PATH}")
    
    loader = FinetunedChromaLoader(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model_path=MODEL_PATH,
        batch_size=50
    )
    
    # Get collection (don't drop - add to existing)
    print("\n2. Preparing ChromaDB collection...")
    loader.create_collection(drop_existing=False)
    
    initial_count = loader.collection.count()
    print(f"   Initial document count: {initial_count:,}")
    
    # Save temp JSON file for loading
    print("\n3. Preparing document for loading...")
    temp_file = 'temp_booth_mba_ms.json'
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    
    # Load to ChromaDB
    print("\n4. Loading document to ChromaDB...")
    print(f"   Document: {document['title']}")
    
    try:
        chunks = loader.prepare_data_for_insertion(temp_file)
        print(f"   ✓ Prepared {len(chunks)} chunks from document")
        
        loader.insert_data(chunks)
        print(f"   ✓ Successfully inserted {len(chunks)} chunks into vector store")
        
    except Exception as e:
        print(f"   ✗ Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Get final count
    final_count = loader.collection.count()
    chunks_added = final_count - initial_count
    
    print(f"\n✓ Loading complete!")
    print(f"  Initial count: {initial_count:,}")
    print(f"  Chunks added: {chunks_added:,}")
    print(f"  Final count: {final_count:,}")
    
    return loader


def verify_content(loader):
    """Verify the newly loaded content with test queries"""
    
    print("\n" + "="*80)
    print("VERIFYING LOADED CONTENT")
    print("="*80)
    
    # Test queries for MBA/MS joint degree program
    test_queries = [
        "What is the MBA/MS joint degree program?",
        "Tell me about the Chicago Booth MBA and MS Applied Data Science collaboration",
        "How long does the MBA/MS joint degree take?",
        "What are the requirements for the MBA/MS joint degree?",
        "What are the benefits of the MBA/MS in Applied Data Science program?"
    ]
    
    collection = loader.collection
    
    print("\nRunning test queries to verify content retrieval...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")
        
        # Get query embedding
        query_embedding = loader.get_embeddings([query])[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"   ✓ Found {len(results['documents'][0])} results")
            
            # Show first result snippet
            first_doc = results['documents'][0][0]
            snippet = first_doc[:200] + "..." if len(first_doc) > 200 else first_doc
            print(f"   Preview: {snippet}")
            
            # Show metadata if available
            if results.get('metadatas') and results['metadatas'][0]:
                metadata = results['metadatas'][0][0]
                if metadata.get('url'):
                    print(f"   Source URL: {metadata['url']}")
        else:
            print(f"   ✗ No results found")
        
        print()


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("CHICAGO BOOTH MBA/MS-ADS SCRAPER & VECTORIZER")
    print("="*80)
    print("\nThis script will:")
    print("  1. Scrape the Chicago Booth MBA/MS-ADS joint degree page")
    print("  2. Load the content into your ChromaDB vector store")
    print("  3. Verify the content is retrievable")
    print("="*80)
    
    # Step 1: Scrape the page
    print("\n" + "="*80)
    print("STEP 1: SCRAPING")
    print("="*80)
    
    document = scrape_booth_page()
    
    if not document:
        print("\n✗ Scraping failed. Exiting.")
        return
    
    # Step 2: Load to vector store
    print("\n" + "="*80)
    print("STEP 2: VECTORIZATION")
    print("="*80)
    
    loader = load_to_vector_store(document)
    
    if not loader:
        print("\n✗ Loading to vector store failed. Exiting.")
        return
    
    # Step 3: Verify
    print("\n" + "="*80)
    print("STEP 3: VERIFICATION")
    print("="*80)
    
    verify_content(loader)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ COMPLETE - CHICAGO BOOTH MBA/MS-ADS PAGE ADDED TO RAG SYSTEM")
    print("="*80)
    
    print(f"\n✓ Page scraped: {document['title']}")
    print(f"✓ URL: {document['url']}")
    print(f"✓ Content length: {len(document['content']):,} characters")
    print(f"✓ Total documents in collection: {loader.collection.count():,}")
    print(f"✓ Collection: {loader.collection_name}")
    
    print("\n💡 The Chicago Booth MBA/MS-ADS joint degree page is now available in your RAG system!")
    
    print("\n💡 Next steps:")
    print("   1. Test queries: python retrieve_from_chromadb_openai.py")
    print("   2. Run your app: python app.py")
    print("   3. Ask questions about the MBA/MS joint degree program!")
    
    print("\n💡 Example questions you can now ask:")
    print("   - What is the MBA/MS joint degree program?")
    print("   - How does the Chicago Booth MBA/MS collaboration work?")
    print("   - What are the benefits of the joint degree program?")
    print("   - What are the requirements for the MBA/MS program?")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
