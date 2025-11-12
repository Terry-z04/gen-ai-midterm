#!/usr/bin/env python3
"""
Batch Scrape Multiple Program Pages and Load to Vector Store
Scrapes pages with nested tabs/accordions and adds them to ChromaDB
"""

import json
from datetime import datetime
from dynamic_retrieval import DynamicRetriever
from load_to_chromadb_openai import FinetunedChromaLoader


# Pages to scrape (those with course tabs/accordions)
PAGES_TO_SCRAPE = [
    {
        "url": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",
        "title": "Course Progressions - Sample Schedules",
        "category": "curriculum"
    },
    # Add more pages here if needed
]


def scrape_all_programs():
    """Scrape all program pages with Firecrawl"""
    
    print("\n" + "="*80)
    print("BATCH SCRAPING PROGRAM PAGES")
    print("="*80)
    
    retriever = DynamicRetriever()
    scraped_data = []
    
    for i, page in enumerate(PAGES_TO_SCRAPE, 1):
        print(f"\n[{i}/{len(PAGES_TO_SCRAPE)}] Scraping: {page['title']}")
        print(f"  URL: {page['url']}")
        
        # Scrape with Firecrawl (automatically captures all tabs)
        result = retriever.scrape_url(page['url'], use_cache=False)
        
        if result:
            content_len = len(result['markdown'])
            print(f"  ✓ Scraped: {content_len:,} characters")
            
            # Create document structure
            document = {
                "title": page['title'],
                "url": page['url'],
                "parent_url": "",
                "category": page['category'],
                "depth": 0,
                "content": result['markdown'],
                "crawl_date": datetime.utcnow().isoformat(),
                "subsections": []
            }
            
            scraped_data.append(document)
            
            # Save individual JSON for inspection
            filename = page['url'].split('/')[-2] + '_program.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to: {filename}")
        else:
            print(f"  ✗ Failed to scrape")
    
    print(f"\n✓ Scraped {len(scraped_data)} pages successfully")
    return scraped_data


def load_to_vector_store(documents):
    """Load all scraped documents to ChromaDB"""
    
    print("\n" + "="*80)
    print("LOADING TO VECTOR STORE")
    print("="*80)
    
    # Configuration
    PERSIST_DIRECTORY = "./chroma_db_finetuned"
    COLLECTION_NAME = "uchicago_msads_finetuned"
    MODEL_PATH = "../finetune-embedding/exp_finetune"
    
    # Initialize loader
    print("\n1. Initializing loader with finetuned embeddings...")
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
    
    # Load each document
    print("\n3. Loading documents...")
    total_chunks_added = 0
    
    for i, doc in enumerate(documents, 1):
        print(f"\n   [{i}/{len(documents)}] Loading: {doc['title']}")
        
        # Save temp JSON file
        temp_file = f"temp_doc_{i}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        
        # Load to ChromaDB
        try:
            chunks = loader.prepare_data_for_insertion(temp_file)
            loader.insert_data(chunks)
            total_chunks_added += len(chunks)
            print(f"   ✓ Added {len(chunks)} chunks")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # Clean up temp file
        import os
        os.remove(temp_file)
    
    final_count = loader.collection.count()
    print(f"\n✓ Loading complete!")
    print(f"  Initial count: {initial_count:,}")
    print(f"  Chunks added: {total_chunks_added:,}")
    print(f"  Final count: {final_count:,}")
    
    return loader


def verify_new_content(loader):
    """Verify newly loaded content with test queries"""
    
    print("\n" + "="*80)
    print("VERIFYING NEW CONTENT")
    print("="*80)
    
    # Test queries for online program
    test_queries = [
        "What is the online program?",
        "Tell me about online program courses",
        "What are the differences between online and in-person programs?",
        "How does the online program work?",
        "What electives are available in the online program?"
    ]
    
    collection = loader.collection
    
    print("\nRunning test queries...\n")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")
        
        # Get query embedding
        query_embedding = loader.get_embeddings([query])[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if results['documents']:
            print(f"   ✓ Found {len(results['documents'][0])} results")
            first_doc = results['documents'][0][0]
            snippet = first_doc[:150] + "..." if len(first_doc) > 150 else first_doc
            print(f"   Preview: {snippet}")
        else:
            print(f"   ✗ No results found")
        print()


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("BATCH PROGRAM SCRAPER & LOADER")
    print("="*80)
    
    # Step 1: Scrape all pages
    print("\nStep 1: Scraping program pages...")
    documents = scrape_all_programs()
    
    if not documents:
        print("\n✗ No documents scraped. Exiting.")
        return
    
    # Step 2: Load to vector store
    print("\nStep 2: Loading to vector store...")
    loader = load_to_vector_store(documents)
    
    # Step 3: Verify
    print("\nStep 3: Verifying content...")
    verify_new_content(loader)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"✓ Scraped and loaded {len(documents)} program pages")
    print(f"✓ Total documents in collection: {loader.collection.count():,}")
    print(f"✓ Collection: {loader.collection_name}")
    print(f"✓ Ready for use in your RAG system!")
    print("="*80)
    
    print("\n💡 Pages added:")
    for doc in documents:
        print(f"   - {doc['title']}")
        print(f"     {doc['url']}")
    
    print("\n💡 Next steps:")
    print("   1. Test queries: python retrieve_from_chromadb_openai.py")
    print("   2. Run your app: python app.py")
    print("   3. Ask questions about both online and in-person programs!")


if __name__ == "__main__":
    main()
