#!/usr/bin/env python3
"""
ChromaDB Retrieval with Fine-tuned Sentence Transformer Embeddings
Group 5 | UChicago MS-ADS RAG System

Features:
- Query using fine-tuned sentence transformer model
- Metadata filtering by category
- Pretty output formatting
- Returns structured results for LLM integration
"""

import argparse
import sys
import os
from typing import Optional, List, Dict

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    sys.exit("chromadb not installed. Run: pip install chromadb")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    sys.exit("sentence-transformers not installed. Run: pip install sentence-transformers")

# Configuration
DEFAULT_DB_PATH = "chroma_db_finetuned"
DEFAULT_COLLECTION = "uchicago_msads_finetuned"
DEFAULT_MODEL_PATH = "../finetune-embedding/exp_finetune"


class FinetunedRetriever:
    """Retriever using fine-tuned sentence transformer embeddings"""
    
    def __init__(self, db_path: str, collection_name: str, embedding_model_path: str):
        """Initialize retriever"""
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Resolve model path
        if not os.path.isabs(embedding_model_path):
            # Make path relative to this script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            embedding_model_path = os.path.join(script_dir, embedding_model_path)
        
        if not os.path.exists(embedding_model_path):
            sys.exit(f"Model path does not exist: {embedding_model_path}")
        
        self.embedding_model_path = embedding_model_path
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=False))
        
        # Get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Connected to collection: {collection_name}")
            print(f"✓ Documents in collection: {self.collection.count()}")
        except Exception as e:
            sys.exit(f"Failed to get collection '{collection_name}': {e}")
        
        # Initialize fine-tuned sentence transformer model
        print(f"Loading fine-tuned model from: {embedding_model_path}")
        self.model = SentenceTransformer(embedding_model_path)
        print(f"✓ Fine-tuned model loaded successfully")
        print(f"✓ Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for query text using fine-tuned model"""
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")
    
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        max_chars: int = 900
    ) -> Dict:
        """
        Retrieve relevant documents for a question
        
        Args:
            question: Query text
            top_k: Number of results to return
            category_filter: Optional category filter (curriculum, admissions, faculty, etc.)
            max_chars: Max characters to display per result
            
        Returns:
            Dict with query and results
        """
        # Embed the query
        print(f"Embedding query with fine-tuned model...")
        query_embedding = self.embed_query(question)
        
        # Build where filter if category specified
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Extract results
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        
        # Pretty print
        self.pretty_print(question, docs, metas, dists, max_chars)
        
        # Return structured payload
        contexts = []
        for d, m, dist in zip(docs, metas, dists):
            contexts.append({
                "text": d,
                "source": m.get("url", "") if isinstance(m, dict) else "",
                "title": m.get("title", "") if isinstance(m, dict) else "",
                "category": m.get("category", "") if isinstance(m, dict) else "",
                "distance": dist,
                "metadata": m
            })
        
        return {
            "query": question,
            "top_k": top_k,
            "category_filter": category_filter,
            "results": contexts
        }
    
    def pretty_print(self, query: str, docs, metas, dists, max_chars: int = 900):
        """Pretty print results"""
        print("\n" + "="*60)
        print("QUERY")
        print("="*60)
        print(query)
        print("\n" + "="*60)
        print("TOP MATCHES")
        print("="*60 + "\n")
        
        if not docs:
            print("(no results)")
            return
        
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            # Extract metadata
            title = meta.get("title", "N/A") if isinstance(meta, dict) else "N/A"
            url = meta.get("url", "N/A") if isinstance(meta, dict) else "N/A"
            category = meta.get("category", "general") if isinstance(meta, dict) else "general"
            
            print(f"[{i}] Distance: {dist:.4f} | Category: {category}")
            print(f"    Title: {title}")
            print(f"    Source: {url}")
            
            # Print snippet
            snippet = (doc or "").strip()
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
            print(f"\n{snippet}\n")
            print("-" * 60 + "\n")


def list_collections(db_path: str):
    """List available collections"""
    client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=False))
    cols = client.list_collections()
    if not cols:
        print("No collections found.")
        return
    print("Available collections:")
    for c in cols:
        print(f" - {c.name}")


def interactive_mode(retriever: FinetunedRetriever, top_k: int, category_filter: Optional[str], max_chars: int):
    """Interactive query mode"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print(f"Collection: {retriever.collection_name}")
    print(f"Documents: {retriever.collection.count()}")
    print(f"Embedding: Fine-tuned model from {retriever.embedding_model_path}")
    if category_filter:
        print(f"Category filter: {category_filter}")
    print("\nPress Ctrl+C or Enter on empty line to exit.")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not question:
            print("\nExiting...")
            break
        
        try:
            retriever.retrieve(question, top_k=top_k, category_filter=category_filter, max_chars=max_chars)
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main execution"""
    ap = argparse.ArgumentParser(description="Semantic retrieval with fine-tuned sentence transformer embeddings")
    ap.add_argument("--db_path", default=DEFAULT_DB_PATH, help="Path to ChromaDB")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to fine-tuned model")
    ap.add_argument("--question", default=None, help="Query question")
    ap.add_argument("--top_k", type=int, default=5, help="Number of results")
    ap.add_argument("--category", default=None, help="Filter by category (curriculum, admissions, etc.)")
    ap.add_argument("--max_chars", type=int, default=900, help="Max chars per result")
    ap.add_argument("--list", action="store_true", help="List collections and exit")
    args = ap.parse_args()
    
    if args.list:
        list_collections(args.db_path)
        return
    
    # Initialize retriever
    retriever = FinetunedRetriever(args.db_path, args.collection, args.model_path)
    
    if not args.question:
        # Interactive mode
        interactive_mode(retriever, args.top_k, args.category, args.max_chars)
    else:
        # One-shot mode
        retriever.retrieve(args.question, top_k=args.top_k, category_filter=args.category, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
