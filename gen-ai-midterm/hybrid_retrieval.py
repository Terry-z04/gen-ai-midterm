#!/usr/bin/env python3
"""
Hybrid Retrieval System: Semantic (Dense) + Lexical (BM25)
Group 5 | UChicago MS-ADS RAG System

Features:
- Semantic search via ChromaDB (cosine similarity)
- Lexical search via BM25 (keyword matching)
- Score normalization and weighted fusion
- Reciprocal Rank Fusion (RRF) option
- Overlap boost for documents appearing in both results
"""

import argparse
import sys
from typing import List, Dict, Optional, Tuple
import numpy as np

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except:
    sys.exit("chromadb not installed. Run: pip install chromadb")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except:
    sys.exit("sentence-transformers not installed. Run: pip install sentence-transformers")

# BM25 for lexical search
try:
    from rank_bm25 import BM25Okapi
except:
    sys.exit("rank_bm25 not installed. Run: pip install rank-bm25")

# Config
try:
    from config import Config
except:
    Config = None


class HybridRetriever:
    """Hybrid retrieval combining semantic and lexical search"""
    
    # Query synonym dictionary for better keyword matching
    QUERY_SYNONYMS = {
        'core course': ['curriculum', 'required courses', 'core curriculum', 'required classes', 'core requirements'],
        'core courses': ['curriculum', 'required courses', 'core curriculum', 'required classes', 'core requirements'],
        'curriculum': ['core courses', 'required courses', 'course requirements', 'program structure'],
        'elective': ['elective courses', 'optional courses', 'course options'],
        'electives': ['elective courses', 'optional courses', 'course options'],
        'tuition': ['cost', 'fees', 'price', 'expenses', 'financial'],
        'cost': ['tuition', 'fees', 'price', 'expenses', 'financial'],
        'admission': ['admissions', 'application', 'apply', 'requirements', 'prerequisites'],
        'admissions': ['admission', 'application', 'apply', 'requirements', 'prerequisites'],
        'capstone': ['capstone project', 'final project', 'thesis'],
        'career': ['careers', 'employment', 'job', 'opportunities', 'outcomes'],
        'careers': ['career', 'employment', 'job', 'opportunities', 'outcomes'],
        'faculty': ['professors', 'instructors', 'teaching staff', 'teachers'],
        'deadline': ['deadlines', 'due date', 'application deadline', 'timeline'],
        'deadlines': ['deadline', 'due date', 'application deadline', 'timeline'],
    }
    
    def __init__(
        self,
        db_path: str = "./chroma_db_finetuned",
        collection_name: str = "uchicago_msads_finetuned",
        embedding_model: str = "../finetune-embedding/exp_finetune",
        semantic_weight: float = 0.8,
        lexical_weight: float = 0.2,
        use_rrf: bool = False,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever
        
        Args:
            db_path: Path to ChromaDB
            collection_name: Collection name
            embedding_model: Sentence transformer model (path or model name)
            semantic_weight: Weight for semantic search (0-1)
            lexical_weight: Weight for lexical search (0-1)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted fusion
            rrf_k: RRF constant (typically 60)
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=False))
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Connected to ChromaDB collection: {collection_name}")
            print(f"✓ Documents in collection: {self.collection.count()}")
        except Exception as e:
            sys.exit(f"Failed to get collection '{collection_name}': {e}")
        
        # Load all documents for BM25 indexing
        print("✓ Loading documents for BM25 indexing...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_documents = all_data["documents"]
        self.all_metadatas = all_data["metadatas"]
        self.all_ids = all_data["ids"]
        
        # Build BM25 index
        print("✓ Building BM25 index...")
        tokenized_corpus = [doc.lower().split() for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Load embedding model - check if it's a path or model name
        import os
        if not os.path.isabs(embedding_model):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(script_dir, embedding_model)
            if os.path.exists(potential_path):
                embedding_model = potential_path
                print(f"✓ Loading fine-tuned model from: {embedding_model}")
            else:
                print(f"✓ Loading embedding model: {embedding_model}")
        
        self.encoder = SentenceTransformer(embedding_model)
        
        print(f"\n{'='*60}")
        print("Hybrid Retriever Ready!")
        print(f"  Semantic weight: {semantic_weight}")
        print(f"  Lexical weight:  {lexical_weight}")
        print(f"  Fusion method:   {'RRF' if use_rrf else 'Weighted'}")
        print(f"{'='*60}\n")
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform semantic search using ChromaDB"""
        # Embed query
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0].tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        semantic_results = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            # Lower distance = higher similarity
            similarity = 1.0 / (1.0 + dist)
            
            semantic_results.append({
                "doc_id": f"semantic_{i}",
                "text": doc,
                "metadata": meta,
                "score": similarity,
                "distance": dist,
                "rank": i + 1
            })
        
        return semantic_results
    
    def expand_query_with_synonyms(self, query: str) -> str:
        """
        Expand query with synonyms for better keyword matching
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query with synonyms
        """
        query_lower = query.lower()
        expanded_terms = []
        
        # Check for multi-word phrases first
        for key_phrase, synonyms in self.QUERY_SYNONYMS.items():
            if key_phrase in query_lower:
                expanded_terms.extend(synonyms)
        
        # Check for individual words
        words = query_lower.split()
        for word in words:
            if word in self.QUERY_SYNONYMS:
                expanded_terms.extend(self.QUERY_SYNONYMS[word])
        
        # Combine original query with synonyms
        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms)
            return expanded_query
        
        return query
    
    def lexical_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform lexical search using BM25 with query synonym expansion"""
        # Expand query with synonyms
        expanded_query = self.expand_query_with_synonyms(query)
        
        # Tokenize expanded query
        tokenized_query = expanded_query.lower().split()
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Format results
        lexical_results = []
        for i, idx in enumerate(top_indices):
            lexical_results.append({
                "doc_id": f"lexical_{idx}",
                "text": self.all_documents[idx],
                "metadata": self.all_metadatas[idx],
                "score": float(bm25_scores[idx]),
                "rank": i + 1
            })
        
        return lexical_results
    
    def normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """Normalize scores to 0-1 range using min-max normalization"""
        if not results:
            return results
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score == 0:
            # All scores are the same
            for r in results:
                r["normalized_score"] = 1.0
        else:
            for r in results:
                r["normalized_score"] = (r["score"] - min_score) / (max_score - min_score)
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        lexical_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)
        Score = sum(1 / (k + rank)) for each result list
        """
        # Build mapping of document text to RRF scores
        doc_scores = {}
        doc_data = {}
        
        # Add semantic results
        for r in semantic_results:
            text = r["text"]
            rrf_score = 1.0 / (k + r["rank"])
            if text not in doc_scores:
                doc_scores[text] = 0.0
                doc_data[text] = r
            doc_scores[text] += rrf_score
            
            # Mark if in semantic results
            if text not in doc_data:
                doc_data[text] = r
            doc_data[text]["in_semantic"] = True
        
        # Add lexical results
        for r in lexical_results:
            text = r["text"]
            rrf_score = 1.0 / (k + r["rank"])
            if text not in doc_scores:
                doc_scores[text] = 0.0
                doc_data[text] = r
            doc_scores[text] += rrf_score
            
            # Mark if in lexical results
            if text not in doc_data:
                doc_data[text] = r
            doc_data[text]["in_lexical"] = True
        
        # Build final results
        merged_results = []
        for text, final_score in doc_scores.items():
            result = doc_data[text].copy()
            result["final_score"] = final_score
            result["in_semantic"] = result.get("in_semantic", False)
            result["in_lexical"] = result.get("in_lexical", False)
            result["overlap"] = result["in_semantic"] and result["in_lexical"]
            merged_results.append(result)
        
        # Sort by final score
        merged_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return merged_results
    
    def weighted_fusion(
        self,
        semantic_results: List[Dict],
        lexical_results: List[Dict],
        overlap_boost: float = 0.1
    ) -> List[Dict]:
        """
        Weighted fusion with overlap boost
        Score = semantic_weight * semantic_score + lexical_weight * lexical_score + overlap_boost
        """
        # Normalize scores
        semantic_results = self.normalize_scores(semantic_results)
        lexical_results = self.normalize_scores(lexical_results)
        
        # Build mapping
        doc_scores = {}
        doc_data = {}
        
        # Add semantic results
        for r in semantic_results:
            text = r["text"]
            if text not in doc_scores:
                doc_scores[text] = {"semantic": 0.0, "lexical": 0.0}
                doc_data[text] = r
            doc_scores[text]["semantic"] = r["normalized_score"]
            doc_data[text]["in_semantic"] = True
        
        # Add lexical results
        for r in lexical_results:
            text = r["text"]
            if text not in doc_scores:
                doc_scores[text] = {"semantic": 0.0, "lexical": 0.0}
                doc_data[text] = r
            doc_scores[text]["lexical"] = r["normalized_score"]
            
            if text not in doc_data:
                doc_data[text] = r
            doc_data[text]["in_lexical"] = True
        
        # Calculate final scores
        merged_results = []
        for text, scores in doc_scores.items():
            result = doc_data[text].copy()
            
            # Weighted fusion
            final_score = (
                self.semantic_weight * scores["semantic"] +
                self.lexical_weight * scores["lexical"]
            )
            
            # Overlap boost
            in_both = result.get("in_semantic", False) and result.get("in_lexical", False)
            if in_both:
                final_score += overlap_boost
            
            result["final_score"] = final_score
            result["semantic_score"] = scores["semantic"]
            result["lexical_score"] = scores["lexical"]
            result["overlap"] = in_both
            merged_results.append(result)
        
        # Sort by final score
        merged_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return merged_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_k: int = 10,
        lexical_k: int = 10,
        overlap_boost: float = 0.1
    ) -> Dict:
        """
        Hybrid retrieval combining semantic and lexical search
        
        Args:
            query: Query string
            top_k: Number of final results to return
            semantic_k: Number of semantic results to fetch
            lexical_k: Number of lexical results to fetch
            overlap_boost: Boost for documents in both result sets (weighted fusion only)
        
        Returns:
            Dict with query, results, and metadata
        """
        print(f"Query: {query}\n")
        
        # Semantic search
        print(f"[1/3] Performing semantic search (top-{semantic_k})...")
        semantic_results = self.semantic_search(query, top_k=semantic_k)
        print(f"      Found {len(semantic_results)} semantic results")
        
        # Lexical search
        print(f"[2/3] Performing lexical search (top-{lexical_k})...")
        lexical_results = self.lexical_search(query, top_k=lexical_k)
        print(f"      Found {len(lexical_results)} lexical results")
        
        # Merge results
        print(f"[3/3] Merging results using {'RRF' if self.use_rrf else 'weighted fusion'}...")
        if self.use_rrf:
            merged_results = self.reciprocal_rank_fusion(semantic_results, lexical_results, k=self.rrf_k)
        else:
            merged_results = self.weighted_fusion(semantic_results, lexical_results, overlap_boost=overlap_boost)
        
        # Get top-k
        final_results = merged_results[:top_k]
        
        print(f"      Final results: {len(final_results)}")
        overlaps = sum(1 for r in final_results if r.get("overlap", False))
        print(f"      Overlapping docs: {overlaps}\n")
        
        return {
            "query": query,
            "top_k": top_k,
            "fusion_method": "RRF" if self.use_rrf else "Weighted",
            "results": final_results
        }
    
    def pretty_print(self, response: Dict, max_chars: int = 400):
        """Pretty print hybrid retrieval results"""
        print("="*60)
        print("HYBRID RETRIEVAL RESULTS")
        print("="*60)
        print(f"Query: {response['query']}")
        print(f"Method: {response['fusion_method']}")
        print(f"Results: {len(response['results'])}\n")
        
        for i, result in enumerate(response["results"], 1):
            print(f"[{i}] Score: {result['final_score']:.4f}")
            
            if "semantic_score" in result:
                print(f"    Semantic: {result.get('semantic_score', 0):.4f} | "
                      f"Lexical: {result.get('lexical_score', 0):.4f}")
            
            if result.get("overlap"):
                print(f"    ⭐ OVERLAP - Found in both semantic and lexical results")
            
            meta = result.get("metadata", {})
            if isinstance(meta, dict):
                print(f"    URL: {meta.get('url', 'N/A')}")
                print(f"    Category: {meta.get('category', meta.get('doc_type', 'N/A'))}")
            
            snippet = result["text"][:max_chars]
            if len(result["text"]) > max_chars:
                snippet += "..."
            print(f"\n{snippet}\n")
            print("-"*60 + "\n")


def main():
    """Main execution"""
    ap = argparse.ArgumentParser(description="Hybrid retrieval: semantic + lexical")
    ap.add_argument("--db_path", default="./chroma_db_finetuned", help="Path to ChromaDB")
    ap.add_argument("--collection", default="uchicago_msads_finetuned", help="Collection name")
    ap.add_argument("--question", default=None, help="Query question")
    ap.add_argument("--top_k", type=int, default=5, help="Final number of results")
    ap.add_argument("--semantic_k", type=int, default=10, help="Semantic results to fetch")
    ap.add_argument("--lexical_k", type=int, default=10, help="Lexical results to fetch")
    ap.add_argument("--semantic_weight", type=float, default=0.6, help="Semantic weight")
    ap.add_argument("--lexical_weight", type=float, default=0.4, help="Lexical weight")
    ap.add_argument("--use_rrf", action="store_true", help="Use RRF instead of weighted fusion")
    ap.add_argument("--overlap_boost", type=float, default=0.1, help="Overlap boost (weighted only)")
    args = ap.parse_args()
    
    # Initialize retriever
    retriever = HybridRetriever(
        db_path=args.db_path,
        collection_name=args.collection,
        semantic_weight=args.semantic_weight,
        lexical_weight=args.lexical_weight,
        use_rrf=args.use_rrf
    )
    
    if not args.question:
        # Interactive mode
        print("Interactive mode. Press Ctrl+C or Enter on empty line to exit.\n")
        while True:
            try:
                question = input("Enter your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            
            if not question:
                print("\nExiting...")
                break
            
            response = retriever.retrieve(
                question,
                top_k=args.top_k,
                semantic_k=args.semantic_k,
                lexical_k=args.lexical_k,
                overlap_boost=args.overlap_boost
            )
            retriever.pretty_print(response)
    else:
        # One-shot mode
        response = retriever.retrieve(
            args.question,
            top_k=args.top_k,
            semantic_k=args.semantic_k,
            lexical_k=args.lexical_k,
            overlap_boost=args.overlap_boost
        )
        retriever.pretty_print(response)


if __name__ == "__main__":
    main()
