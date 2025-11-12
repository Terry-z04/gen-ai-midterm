"""
Generate Synthetic Dataset from Web-Scraped Data

This script generates a synthetic dataset of (query, relevant documents) pairs 
from web-scraped content without labelers by leveraging LLM.

This version is adapted to work with the hierarchical JSON structure from web scraping.
"""

import json
import re
import uuid
import random
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from openai import OpenAI as OpenAIClient


# =============================================================================
# Configuration
# =============================================================================

# Path to your web-scraped JSON file
WEBSCRAPE_DATA_PATH = '../gen-ai-midterm/uchicago_msads_content_rag.json'

# Output paths
TRAIN_CORPUS_FPATH = './data/train_corpus.json'
VAL_CORPUS_FPATH = './data/val_corpus.json'
TRAIN_QUERIES_FPATH = './data/train_queries.json'
TRAIN_RELEVANT_DOCS_FPATH = './data/train_relevant_docs.json'
VAL_QUERIES_FPATH = './data/val_queries.json'
VAL_RELEVANT_DOCS_FPATH = './data/val_relevant_docs.json'
TRAIN_DATASET_FPATH = './data/train_dataset.json'
VAL_DATASET_FPATH = './data/val_dataset.json'

# Parameters
CHUNK_SIZE = 1000  # Maximum characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
TRAIN_SPLIT_RATIO = 0.8  # 80% for training, 20% for validation
NUM_QUESTIONS_PER_CHUNK = 2  # Number of questions to generate per chunk


# =============================================================================
# Helper Functions
# =============================================================================

def extract_text_chunks_from_page(page_data: dict, parent_title: str = "") -> List[Tuple[str, dict]]:
    """
    Recursively extract text chunks from a page and its subsections.
    Returns list of (text, metadata) tuples.
    """
    chunks = []
    
    # Get current page info
    title = page_data.get('title', 'Untitled')
    url = page_data.get('url', '')
    content = page_data.get('content', '')
    
    # Create metadata
    metadata = {
        'title': title,
        'url': url,
        'parent_title': parent_title,
        'depth': page_data.get('depth', 0)
    }
    
    # Add current page content if it exists
    if content and len(content.strip()) > 0:
        chunks.append((content.strip(), metadata.copy()))
    
    # Recursively process subsections
    subsections = page_data.get('subsections', [])
    for subsection in subsections:
        sub_chunks = extract_text_chunks_from_page(subsection, parent_title=title)
        chunks.extend(sub_chunks)
    
    return chunks


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            last_para = text[start:end].rfind('\n\n')
            if last_para > chunk_size // 2:
                end = start + last_para
            else:
                # Look for sentence break
                last_period = text[start:end].rfind('. ')
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def load_and_process_webscrape_data(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> Dict[str, str]:
    """
    Load web-scraped JSON data and convert it into a corpus dictionary.
    Returns: {node_id: text_content}
    """
    print(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Extracting text chunks from pages...")
    raw_chunks = extract_text_chunks_from_page(data)
    print(f"Extracted {len(raw_chunks)} pages")
    
    # Create corpus with chunked text
    corpus = {}
    total_chunks = 0
    
    for text, metadata in raw_chunks:
        # Split large content into smaller chunks
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        for i, chunk in enumerate(text_chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            # Create unique ID
            chunk_id = f"{metadata['url']}#chunk_{i}" if metadata['url'] else str(uuid.uuid4())
            
            # Add context to chunk
            chunk_with_context = f"Title: {metadata['title']}\n\nContent: {chunk}"
            
            corpus[chunk_id] = chunk_with_context
            total_chunks += 1
    
    print(f"Created {total_chunks} text chunks")
    return corpus


def generate_queries(
    corpus,
    num_questions_per_chunk=2,
    prompt_template=None,
    verbose=False,
):
    """
    Automatically generate hypothetical questions that could be answered with
    doc in the corpus.
    """
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAIClient(api_key=api_key)

    prompt_template = prompt_template or """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge,
generate {num_questions_per_chunk} questions that could be answered using this context.

You are a prospective student researching the UChicago MS in Applied Data Science program.
Your task is to generate {num_questions_per_chunk} realistic questions that someone might ask
about the program based on this content.

The questions should be:
- Specific and answerable from the context
- Diverse in nature
- Natural language questions a real person might ask

Format: Return only the questions, one per line, without numbering.
"""

    queries = {}
    relevant_docs = {}
    
    for node_id, text in tqdm(corpus.items(), desc="Generating queries"):
        prompt = prompt_template.format(
            context_str=text, 
            num_questions_per_chunk=num_questions_per_chunk
        )
        
        try:
            # Use OpenAI API directly
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip().split("\n")
            
            # Clean up questions
            questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() 
                for question in result
            ]
            questions = [q for q in questions if len(q) > 10]  # Filter very short questions
            
            for question in questions:
                question_id = str(uuid.uuid4())
                queries[question_id] = question
                relevant_docs[question_id] = [node_id]
        
        except Exception as e:
            if verbose:
                print(f"Error generating queries for {node_id}: {e}")
            continue
    
    return queries, relevant_docs


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print("SYNTHETIC DATASET GENERATION FROM WEB-SCRAPED DATA")
    print("="*70)
    print()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Step 1: Load and process the corpus
    print("Step 1: Loading and processing web-scraped data...")
    full_corpus = load_and_process_webscrape_data(
        WEBSCRAPE_DATA_PATH, 
        chunk_size=CHUNK_SIZE, 
        overlap=CHUNK_OVERLAP
    )
    print()
    
    # Step 2: Split into train and validation sets
    print("Step 2: Splitting into train and validation sets...")
    corpus_items = list(full_corpus.items())
    random.shuffle(corpus_items)
    
    split_idx = int(len(corpus_items) * TRAIN_SPLIT_RATIO)
    train_corpus = dict(corpus_items[:split_idx])
    val_corpus = dict(corpus_items[split_idx:])
    
    print(f"Train corpus: {len(train_corpus)} chunks")
    print(f"Val corpus: {len(val_corpus)} chunks")
    print()
    
    # Step 3: Save corpus files
    print("Step 3: Saving corpus files...")
    with open(TRAIN_CORPUS_FPATH, 'w+') as f:
        json.dump(train_corpus, f, indent=2)
    
    with open(VAL_CORPUS_FPATH, 'w+') as f:
        json.dump(val_corpus, f, indent=2)
    
    print("Corpus files saved!")
    print()
    
    # Step 4: Generate queries for training set
    print("Step 4: Generating training queries...")
    train_queries, train_relevant_docs = generate_queries(
        train_corpus, 
        num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK,
        verbose=True
    )
    print(f"Generated {len(train_queries)} training queries")
    print()
    
    # Step 5: Generate queries for validation set
    print("Step 5: Generating validation queries...")
    val_queries, val_relevant_docs = generate_queries(
        val_corpus,
        num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK,
        verbose=True
    )
    print(f"Generated {len(val_queries)} validation queries")
    print()
    
    # Step 6: Save query files
    print("Step 6: Saving query files...")
    with open(TRAIN_QUERIES_FPATH, 'w+') as f:
        json.dump(train_queries, f, indent=2)
    
    with open(TRAIN_RELEVANT_DOCS_FPATH, 'w+') as f:
        json.dump(train_relevant_docs, f, indent=2)
    
    with open(VAL_QUERIES_FPATH, 'w+') as f:
        json.dump(val_queries, f, indent=2)
    
    with open(VAL_RELEVANT_DOCS_FPATH, 'w+') as f:
        json.dump(val_relevant_docs, f, indent=2)
    
    print("Query files saved!")
    print()
    
    # Step 7: Merge data into final datasets
    print("Step 7: Creating final datasets...")
    train_dataset = {
        'queries': train_queries,
        'corpus': train_corpus,
        'relevant_docs': train_relevant_docs,
    }
    
    val_dataset = {
        'queries': val_queries,
        'corpus': val_corpus,
        'relevant_docs': val_relevant_docs,
    }
    
    with open(TRAIN_DATASET_FPATH, 'w+') as f:
        json.dump(train_dataset, f, indent=2)
    
    with open(VAL_DATASET_FPATH, 'w+') as f:
        json.dump(val_dataset, f, indent=2)
    
    print()
    print("="*70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f"Training dataset: {len(train_queries)} queries, {len(train_corpus)} documents")
    print(f"Validation dataset: {len(val_queries)} queries, {len(val_corpus)} documents")
    print()
    print("Output files:")
    print(f"  - {TRAIN_DATASET_FPATH}")
    print(f"  - {VAL_DATASET_FPATH}")
    print("="*70)
    
    # Display sample queries
    print("\nSample Generated Queries:")
    print("-"*70)
    for i, (qid, query) in enumerate(list(train_queries.items())[:5]):
        print(f"\nQuery {i+1}: {query}")
        doc_id = train_relevant_docs[qid][0]
        print(f"Relevant Document ID: {doc_id[:80]}...")
    print("-"*70)


if __name__ == "__main__":
    main()
