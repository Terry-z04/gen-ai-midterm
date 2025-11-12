#!/usr/bin/env python3
"""
Confidence Scoring & Intelligent Routing
Group 5 | UChicago MS-ADS RAG System

Features:
- Confidence scoring based on semantic similarity, lexical match, and freshness
- Intelligent routing: static retrieval vs dynamic Firecrawl
- Threshold-based decision making
- Combined scoring with configurable weights
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import our retrieval systems
try:
    from hybrid_retrieval import HybridRetriever
    HYBRID_AVAILABLE = True
except ImportError:
    print("⚠ hybrid_retrieval not available")
    HYBRID_AVAILABLE = False

try:
    from dynamic_retrieval import DynamicRetriever
    DYNAMIC_AVAILABLE = True
except ImportError:
    print("⚠ dynamic_retrieval not available")
    DYNAMIC_AVAILABLE = False


class ConfidenceRouter:
    """
    Intelligent routing based on confidence scores
    Decides between static and dynamic retrieval
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.3,
        recency_weight: float = 0.1,
        confidence_threshold: float = 0.3,
        freshness_days: int = 30,
        db_path: str = "./chroma_db_finetuned",
        collection_name: str = "uchicago_msads_finetuned"
    ):
        """
        Initialize confidence router
        
        Args:
            semantic_weight: Weight for semantic similarity score
            lexical_weight: Weight for lexical match score
            recency_weight: Weight for content freshness score
            confidence_threshold: Threshold to trigger dynamic retrieval
            freshness_days: Days before content considered stale
            db_path: Path to ChromaDB
            collection_name: Collection name
        """
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.recency_weight = recency_weight
        self.confidence_threshold = confidence_threshold
        self.freshness_days = freshness_days
        
        # Initialize hybrid retriever for static search with updated weights
        if not HYBRID_AVAILABLE:
            raise ImportError("hybrid_retrieval not available")
        
        self.hybrid_retriever = HybridRetriever(
            db_path=db_path,
            collection_name=collection_name,
            semantic_weight=0.8,
            lexical_weight=0.2
        )
        
        # Initialize dynamic retriever (lazy loaded)
        self._dynamic_retriever = None
        
        print(f"\n{'='*60}")
        print("Confidence Router Initialized")
        print(f"  Semantic weight:  {semantic_weight}")
        print(f"  Lexical weight:   {lexical_weight}")
        print(f"  Recency weight:   {recency_weight}")
        print(f"  Threshold:        {confidence_threshold}")
        print(f"  Freshness:        {freshness_days} days")
        print(f"{'='*60}\n")
    
    @property
    def dynamic_retriever(self):
        """Lazy load dynamic retriever"""
        if self._dynamic_retriever is None and DYNAMIC_AVAILABLE:
            self._dynamic_retriever = DynamicRetriever()
        return self._dynamic_retriever
    
    def score_semantic_similarity(self, results: List[Dict]) -> float:
        """
        Score semantic similarity from retrieval results
        Uses max cosine similarity (min distance) from top results
        
        Args:
            results: Results from hybrid retriever
            
        Returns:
            Semantic similarity score (0-1)
        """
        if not results:
            return 0.0
        
        # Get best semantic score from results
        # Lower distance = higher similarity in ChromaDB
        best_score = 0.0
        for result in results:
            # Try to get distance or score
            distance = result.get('distance', float('inf'))
            
            # Convert distance to similarity
            # Using 1 / (1 + distance) formula
            similarity = 1.0 / (1.0 + distance)
            best_score = max(best_score, similarity)
        
        return min(best_score, 1.0)  # Cap at 1.0
    
    def score_lexical_match(self, query: str, results: List[Dict]) -> float:
        """
        Score lexical match using TF-IDF term overlap
        
        Args:
            query: User query
            results: Results from hybrid retriever
            
        Returns:
            Lexical match score (0-1)
        """
        if not results:
            return 0.0
        
        # Tokenize query
        query_terms = set(query.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Calculate term overlap with top results
        best_overlap = 0.0
        for result in results:
            text = result.get('text', '').lower()
            text_terms = set(text.split())
            
            if not text_terms:
                continue
            
            # Calculate Jaccard similarity
            intersection = query_terms & text_terms
            union = query_terms | text_terms
            
            if union:
                overlap = len(intersection) / len(union)
                best_overlap = max(best_overlap, overlap)
        
        return min(best_overlap, 1.0)
    
    def score_content_freshness(self, results: List[Dict]) -> float:
        """
        Score content freshness based on crawl_date
        
        Args:
            results: Results from hybrid retriever
            
        Returns:
            Freshness score (0-1, 1 = very fresh)
        """
        if not results:
            return 0.0
        
        now = datetime.utcnow()
        freshness_threshold = timedelta(days=self.freshness_days)
        
        # Get most recent crawl date from results
        most_recent = None
        for result in results:
            metadata = result.get('metadata', {})
            if isinstance(metadata, dict):
                # Try different date field names
                crawl_date_str = metadata.get('crawl_date') or metadata.get('last_scraped') or metadata.get('scraped_at')
                
                if crawl_date_str:
                    try:
                        # Parse ISO format datetime
                        crawl_date = datetime.fromisoformat(crawl_date_str.replace('Z', '+00:00'))
                        if most_recent is None or crawl_date > most_recent:
                            most_recent = crawl_date
                    except:
                        continue
        
        if most_recent is None:
            # No date info, assume medium freshness
            return 0.5
        
        # Calculate freshness score
        age = now - most_recent
        if age <= timedelta(days=0):
            # Future date or today
            return 1.0
        elif age >= freshness_threshold:
            # Too old
            return 0.0
        else:
            # Linear decay
            return 1.0 - (age / freshness_threshold)
    
    def calculate_confidence(
        self,
        query: str,
        results: List[Dict]
    ) -> Dict:
        """
        Calculate overall confidence score
        
        Args:
            query: User query
            results: Results from retrieval
            
        Returns:
            Dict with scores and confidence
        """
        # Calculate individual scores
        semantic_score = self.score_semantic_similarity(results)
        lexical_score = self.score_lexical_match(query, results)
        freshness_score = self.score_content_freshness(results)
        
        # Combined confidence score
        confidence = (
            self.semantic_weight * semantic_score +
            self.lexical_weight * lexical_score +
            self.recency_weight * freshness_score
        )
        
        return {
            'semantic_score': semantic_score,
            'lexical_score': lexical_score,
            'freshness_score': freshness_score,
            'confidence': confidence,
            'meets_threshold': confidence >= self.confidence_threshold
        }
    
    def route_query(
        self,
        query: str,
        top_k: int = 5,
        fallback_urls: Optional[List[str]] = None
    ) -> Dict:
        """
        Main routing logic: decide between static and dynamic retrieval
        
        Args:
            query: User query
            top_k: Number of results to return
            fallback_urls: URLs to scrape if confidence is low
            
        Returns:
            Dict with results, scores, and routing decision
        """
        print(f"\n{'='*60}")
        print(f"CONFIDENCE ROUTING")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Try static hybrid retrieval
        print("[1/3] Attempting static retrieval...")
        static_response = self.hybrid_retriever.retrieve(
            query,
            top_k=top_k,
            semantic_k=10,
            lexical_k=10
        )
        
        static_results = static_response.get('results', [])
        
        # Step 2: Calculate confidence
        print("[2/3] Calculating confidence scores...")
        scores = self.calculate_confidence(query, static_results)
        
        print(f"\n  Semantic score:  {scores['semantic_score']:.3f}")
        print(f"  Lexical score:   {scores['lexical_score']:.3f}")
        print(f"  Freshness score: {scores['freshness_score']:.3f}")
        print(f"  → Confidence:    {scores['confidence']:.3f}")
        print(f"  → Threshold:     {self.confidence_threshold}")
        
        # Step 3: Routing decision
        if scores['meets_threshold']:
            print(f"\n✓ Confidence ABOVE threshold → Using static retrieval\n")
            
            return {
                'query': query,
                'routing': 'static',
                'scores': scores,
                'results': static_results,
                'num_results': len(static_results),
                'dynamic_triggered': False
            }
        else:
            print(f"\n⚠ Confidence BELOW threshold → Triggering dynamic retrieval\n")
            
            # Trigger dynamic retrieval if URLs provided
            if fallback_urls and DYNAMIC_AVAILABLE:
                print("[3/3] Executing dynamic retrieval...")
                dynamic_response = self.dynamic_retriever.dynamic_retrieve(
                    query=query,
                    urls=fallback_urls
                )
                
                # Combine static and dynamic results
                combined_results = static_results + dynamic_response.get('results', [])
                
                return {
                    'query': query,
                    'routing': 'dynamic',
                    'scores': scores,
                    'static_results': static_results,
                    'dynamic_results': dynamic_response.get('results', []),
                    'results': combined_results[:top_k],
                    'num_results': len(combined_results),
                    'dynamic_triggered': True
                }
            else:
                print("⚠ No fallback URLs provided, returning static results\n")
                
                return {
                    'query': query,
                    'routing': 'static_fallback',
                    'scores': scores,
                    'results': static_results,
                    'num_results': len(static_results),
                    'dynamic_triggered': False,
                    'warning': 'Low confidence but no dynamic URLs provided'
                }
    
    def pretty_print(self, response: Dict):
        """Pretty print routing response"""
        print("="*60)
        print("ROUTING RESULTS")
        print("="*60)
        print(f"Query: {response['query']}")
        print(f"Routing: {response['routing'].upper()}")
        print(f"Confidence: {response['scores']['confidence']:.3f}")
        print(f"Dynamic triggered: {response['dynamic_triggered']}")
        print(f"Total results: {response['num_results']}\n")
        
        if response.get('warning'):
            print(f"⚠ Warning: {response['warning']}\n")
        
        # Print top results
        for i, result in enumerate(response['results'][:5], 1):
            print(f"[{i}] {result.get('title', 'N/A')[:60]}")
            if 'final_score' in result:
                print(f"    Score: {result['final_score']:.3f}")
            print(f"    {result.get('text', '')[:200]}...")
            print()


def main():
    """Main execution"""
    ap = argparse.ArgumentParser(description="Confidence-based routing")
    ap.add_argument("--question", required=True, help="Query question")
    ap.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    ap.add_argument("--fallback-urls", nargs='+', help="URLs to scrape if confidence low")
    ap.add_argument("--top_k", type=int, default=5, help="Number of results")
    args = ap.parse_args()
    
    # Initialize router
    router = ConfidenceRouter(
        confidence_threshold=args.threshold
    )
    
    # Route query
    response = router.route_query(
        query=args.question,
        top_k=args.top_k,
        fallback_urls=args.fallback_urls
    )
    
    # Print results
    router.pretty_print(response)


if __name__ == "__main__":
    main()
