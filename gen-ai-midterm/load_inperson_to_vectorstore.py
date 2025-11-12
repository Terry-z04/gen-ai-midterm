#!/usr/bin/env python3
"""
Load In-Person Program Content to ChromaDB Vector Store
Converts scraped markdown content to JSON and loads with finetuned embeddings
"""

import json
from datetime import datetime
from load_to_chromadb_openai import FinetunedChromaLoader


def create_json_from_scraped_content():
    """
    Convert scraped markdown content to JSON format expected by loader
    """
    
    # Read the scraped content
    with open('scraped_content_full.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create JSON structure
    document = {
        "title": "MS in Applied Data Science - In-Person Program",
        "url": "https://datascience.uchicago.edu/education/masters-programs/in-person-program/",
        "parent_url": "",
        "category": "academic_programs",
        "depth": 0,
        "content": content,
        "crawl_date": datetime.utcnow().isoformat(),
        "subsections": []
    }
    
    # Save to JSON file
    json_filename = 'in_person_program.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created JSON file: {json_filename}")
    print(f"  Content length: {len(content):,} characters")
    print(f"  URL: {document['url']}")
    
    return json_filename


def load_to_chromadb(json_file: str):
    """
    Load JSON content to ChromaDB with finetuned embeddings
    """
    
    print("\n" + "="*80)
    print("LOADING IN-PERSON PROGRAM TO VECTOR STORE")
    print("="*80)
    
    # Configuration
    PERSIST_DIRECTORY = "./chroma_db_finetuned"
    COLLECTION_NAME = "uchicago_msads_finetuned"
    MODEL_PATH = "../finetune-embedding/exp_finetune"
    
    # Initialize loader with finetuned embeddings
    print("\n1. Initializing loader with finetuned embeddings...")
    loader = FinetunedChromaLoader(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model_path=MODEL_PATH,
        batch_size=50  # Smaller batch size for large content
    )
    
    # Get or create collection (don't drop existing - we want to add to it)
    print("\n2. Preparing ChromaDB collection...")
    loader.create_collection(drop_existing=False)
    
    # Load and insert data
    print("\n3. Loading and embedding content...")
    loader.load_from_json(json_file)
    
    return loader


def verify_loaded_content(loader):
    """
    Verify the content was loaded correctly by running test queries
    """
    
    print("\n" + "="*80)
    print("VERIFYING LOADED CONTENT")
    print("="*80)
    
    # Test queries related to in-person program
    test_queries = [
        "What are the core courses in the in-person program?",
        "Tell me about Machine Learning courses",
        "What elective courses are available?",
        "What is the Data Engineering course about?",
        "Tell me about the Generative AI course"
    ]
    
    collection = loader.collection
    
    print("\nRunning test queries...\n")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")
        
        # Get query embedding
        query_embedding = loader.get_embeddings([query])[0]
        
        # Search collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if results['documents']:
            print(f"   ✓ Found {len(results['documents'][0])} results")
            # Show first result snippet
            first_doc = results['documents'][0][0]
            snippet = first_doc[:200] + "..." if len(first_doc) > 200 else first_doc
            print(f"   Preview: {snippet}")
        else:
            print(f"   ✗ No results found")
        print()


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("IN-PERSON PROGRAM VECTOR STORE LOADER")
    print("="*80)
    
    # Step 1: Create JSON from scraped content
    print("\nStep 1: Converting scraped content to JSON...")
    json_file = create_json_from_scraped_content()
    
    # Step 2: Load to ChromaDB
    print("\nStep 2: Loading to ChromaDB with finetuned embeddings...")
    loader = load_to_chromadb(json_file)
    
    # Step 3: Verify
    print("\nStep 3: Verifying content...")
    verify_loaded_content(loader)
    
    # Final summary
    count = loader.collection.count()
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"✓ In-person program content loaded successfully")
    print(f"✓ Total documents in collection: {count:,}")
    print(f"✓ Collection: {loader.collection_name}")
    print(f"✓ Embedding model: Fine-tuned sentence transformer")
    print(f"✓ Location: ./chroma_db_finetuned")
    print("="*80)
    
    print("\n💡 Next steps:")
    print("   1. Test queries with: python retrieve_from_chromadb_openai.py")
    print("   2. Use in your RAG system: The guardrail will now access complete course info!")
    print("   3. Run your app: python app.py")


if __name__ == "__main__":
    main()
