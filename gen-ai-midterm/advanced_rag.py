#!/usr/bin/env python3
"""
Advanced RAG with RAPTOR, HyDE, and RAG Fusion
Group 5 | UChicago MS-ADS RAG System

Techniques:
1. HyDE (Hypothetical Document Embeddings): Generate hypothetical answer → embed → retrieve
2. RAG Fusion: Multiple query variations → retrieve → rank fusion
3. RAPTOR: Hierarchical summarization (if needed for long docs)
4. Combined approach for best results
"""

import argparse
from typing import List, Dict, Optional
from openai import OpenAI

# Config
try:
    from config import Config
    OPENAI_API_KEY = Config.OPENAI_API_KEY
except:
    import os
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Import existing systems
try:
    from hybrid_retrieval import HybridRetriever
    HYBRID_AVAILABLE = True
except:
    print("⚠ hybrid_retrieval not available")
    HYBRID_AVAILABLE = False

try:
    from qa_generator import QAGenerator
    QA_AVAILABLE = True
except:
    print("⚠ qa_generator not available")
    QA_AVAILABLE = False


class AdvancedRAG:
    """Advanced RAG with HyDE, RAG Fusion, and RAPTOR"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "./chroma_db_finetuned",
        collection_name: str = "uchicago_msads_finetuned"  # Updated to use fine-tuned embeddings
    ):
        """Initialize advanced RAG system with fine-tuned embeddings"""
        self.api_key = api_key or OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize retriever with updated weights (semantic 0.8, lexical 0.2)
        if HYBRID_AVAILABLE:
            self.retriever = HybridRetriever(
                db_path=db_path,
                collection_name=collection_name,
                semantic_weight=0.8,
                lexical_weight=0.2
            )
        else:
            raise ImportError("HybridRetriever required")
        
        # Initialize QA generator
        if QA_AVAILABLE:
            self.qa_gen = QAGenerator(api_key=self.api_key, model="gpt-4o-mini")
        
        print(f"\n{'='*60}")
        print("Advanced RAG System Initialized")
        print("  Techniques: HyDE, RAG Fusion, Multi-Query")
        print(f"{'='*60}\n")
    
    def hyde_retrieval(
        self,
        question: str,
        top_k: int = 10
    ) -> Dict:
        """
        HyDE: Generate hypothetical answer, embed it, retrieve similar docs
        
        Args:
            question: User question
            top_k: Number of results
            
        Returns:
            Dict with results and metadata
        """
        print(f"\n[HyDE] Generating hypothetical answer...")
        
        # Generate hypothetical answer
        hyde_prompt = f"""Generate a detailed, factual answer to this question about UChicago's MS in Applied Data Science program:

Question: {question}

Write a comprehensive answer (2-3 paragraphs) as if you were an expert, including specific details about the program."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": hyde_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        hypothetical_answer = response.choices[0].message.content
        print(f"[HyDE] Hypothetical answer generated ({len(hypothetical_answer)} chars)")
        
        # Use hypothetical answer as query
        results = self.retriever.retrieve(
            query=hypothetical_answer,
            top_k=top_k,
            semantic_k=top_k * 2,
            lexical_k=top_k * 2
        )
        
        return {
            'method': 'HyDE',
            'hypothetical_answer': hypothetical_answer,
            'results': results['results']
        }
    
    def generate_query_variations(
        self,
        question: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate query variations for RAG Fusion
        
        Args:
            question: Original question
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        print(f"\n[RAG Fusion] Generating {num_variations} query variations...")
        
        prompt = f"""Generate {num_variations} different ways to ask the following question about UChicago's MS in Applied Data Science program. Each variation should:
- Focus on different aspects
- Use different phrasing
- Maintain the same intent

Original question: {question}

Provide only the {num_variations} variations, one per line, without numbering."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=200
        )
        
        variations_text = response.choices[0].message.content
        variations = [line.strip() for line in variations_text.strip().split('\n') if line.strip()]
        
        # Add original question
        all_queries = [question] + variations[:num_variations]
        
        for i, q in enumerate(all_queries):
            print(f"  {i+1}. {q}")
        
        return all_queries
    
    def rag_fusion(
        self,
        question: str,
        num_variations: int = 3,
        top_k_per_query: int = 5,
        final_top_k: int = 10
    ) -> Dict:
        """
        RAG Fusion: Multiple query variations + reciprocal rank fusion
        
        Args:
            question: Original question
            num_variations: Number of query variations
            top_k_per_query: Results per query
            final_top_k: Final number of results
            
        Returns:
            Dict with fused results
        """
        # Generate variations
        queries = self.generate_query_variations(question, num_variations)
        
        # Retrieve for each query
        print(f"\n[RAG Fusion] Retrieving for {len(queries)} queries...")
        all_results = []
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] Retrieving...")
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k_per_query,
                semantic_k=top_k_per_query * 2,
                lexical_k=top_k_per_query * 2
            )
            all_results.append(results['results'])
        
        # Reciprocal Rank Fusion
        print(f"\n[RAG Fusion] Applying reciprocal rank fusion...")
        doc_scores = {}
        doc_data = {}
        k = 60  # RRF constant
        
        for query_idx, results in enumerate(all_results):
            for rank, result in enumerate(results, 1):
                text = result.get('text', '')
                
                # Use text as key (simple deduplication)
                if text not in doc_scores:
                    doc_scores[text] = 0.0
                    doc_data[text] = result
                
                # RRF score
                doc_scores[text] += 1.0 / (k + rank)
        
        # Sort by fusion score
        fused_results = []
        for text, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_data[text].copy()
            result['fusion_score'] = score
            fused_results.append(result)
        
        return {
            'method': 'RAG_Fusion',
            'num_queries': len(queries),
            'queries': queries,
            'results': fused_results[:final_top_k]
        }
    
    def detect_curriculum_query(self, question: str) -> bool:
        """Detect if question is asking about curriculum, courses, or program structure"""
        curriculum_keywords = [
            'course', 'courses', 'curriculum', 'class', 'classes',
            'core', 'elective', 'requirement', 'requirements',
            'take', 'study', 'learn', 'teach', 'taught',
            'quarter', 'semester', 'program structure'
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in curriculum_keywords)
    
    def rerank_for_courses(self, results: List[Dict], question: str) -> List[Dict]:
        """
        Re-rank results to boost chunks containing course names for curriculum queries
        """
        # Course name indicators
        course_indicators = [
            'Letter Grade', 'Core', 'Elective', 
            'Machine Learning', 'Statistical', 'Data Engineering',
            'Time Series', 'Leadership and Consulting',
            'Capstone', 'Thesis', 'quarter', 'units'
        ]
        
        for result in results:
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            # Boost score if contains multiple course indicators
            boost = 0.0
            indicator_count = sum(1 for indicator in course_indicators if indicator in text)
            
            if indicator_count > 0:
                boost = indicator_count * 0.3  # Boost per indicator
                
            # Extra boost for "Core" + course name patterns
            if 'Core' in text and any(name in text for name in ['Machine Learning', 'Statistical', 'Leadership', 'Time Series', 'Data Engineering']):
                boost += 0.5
                
            # Update final score
            result['final_score'] = result.get('final_score', 0.0) + boost
            result['course_boost'] = boost
        
        return results
    
    def multi_method_retrieval(
        self,
        question: str,
        top_k: int = 10
    ) -> Dict:
        """
        Combined: Use multiple methods and merge results
        
        Args:
            question: User question
            top_k: Final number of results
            
        Returns:
            Dict with best results from all methods
        """
        print(f"\n{'='*60}")
        print(f"ADVANCED RAG - MULTI-METHOD RETRIEVAL")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Detect curriculum query and adjust retrieval
        is_curriculum_query = self.detect_curriculum_query(question)
        retrieve_k = top_k * 3 if is_curriculum_query else top_k  # Get more chunks for curriculum queries
        
        if is_curriculum_query:
            print(f"\n[Detected curriculum query - retrieving {retrieve_k} chunks for re-ranking]")
        
        # Method 1: Standard hybrid retrieval
        print(f"\n[1/3] Standard Hybrid Retrieval...")
        standard = self.retriever.retrieve(
            query=question,
            top_k=retrieve_k,
            semantic_k=retrieve_k * 2,
            lexical_k=retrieve_k * 2
        )
        
        # Method 2: HyDE
        print(f"\n[2/3] HyDE Retrieval...")
        hyde = self.hyde_retrieval(question, top_k=retrieve_k)
        
        # Method 3: RAG Fusion
        print(f"\n[3/3] RAG Fusion...")
        fusion = self.rag_fusion(question, num_variations=2, top_k_per_query=10 if is_curriculum_query else 5, final_top_k=retrieve_k)
        
        # Combine and deduplicate
        print(f"\n[Final] Combining results from all methods...")
        all_docs = {}
        
        # Add standard results
        for result in standard['results']:
            text = result.get('text', '')
            if text and text not in all_docs:
                result['methods'] = ['standard']
                all_docs[text] = result
            elif text in all_docs:
                all_docs[text]['methods'].append('standard')
        
        # Add HyDE results
        for result in hyde['results']:
            text = result.get('text', '')
            if text and text not in all_docs:
                result['methods'] = ['hyde']
                all_docs[text] = result
            elif text in all_docs:
                all_docs[text]['methods'].append('hyde')
        
        # Add Fusion results
        for result in fusion['results']:
            text = result.get('text', '')
            if text and text not in all_docs:
                result['methods'] = ['fusion']
                result['final_score'] = result.get('fusion_score', 0.0)
                all_docs[text] = result
            elif text in all_docs:
                all_docs[text]['methods'].append('fusion')
                all_docs[text]['final_score'] = max(
                    all_docs[text].get('final_score', 0.0),
                    result.get('fusion_score', 0.0)
                )
        
        # Boost documents found by multiple methods
        for doc in all_docs.values():
            method_count = len(set(doc.get('methods', [])))
            doc['method_count'] = method_count
            doc['final_score'] = doc.get('final_score', 0.0) + (method_count * 0.2)
        
        # Convert to list for re-ranking
        results_list = list(all_docs.values())
        
        # Apply course-specific re-ranking if needed
        if is_curriculum_query:
            print(f"\n[Applying course-specific re-ranking...]")
            results_list = self.rerank_for_courses(results_list, question)
        
        # Sort by final score and method count
        final_results = sorted(
            results_list,
            key=lambda x: (x.get('final_score', 0.0), x.get('method_count', 0)),
            reverse=True
        )[:top_k]
        
        print(f"\nCombined {len(all_docs)} unique documents")
        print(f"Top {top_k} selected\n")
        
        return {
            'method': 'Multi-Method (Standard + HyDE + RAG Fusion)',
            'question': question,
            'results': final_results,
            'hyde_hypothetical': hyde['hypothetical_answer'],
            'fusion_queries': fusion['queries']
        }
    
    def answer_with_advanced_rag(
        self,
        question: str,
        top_k: int = 12,  # Increased default from 8 to 12
        model: str = "gpt-4o-mini"
    ) -> Dict:


        """
        Full pipeline: Advanced retrieval + answer generation
        
        Args:
            question: User question
            top_k: Number of results to use
            model: Model name
            
        Returns:
            Dict with answer and metadata
        """
        # Retrieve using advanced methods
        retrieval_response = self.multi_method_retrieval(question, top_k=top_k)
        
        # Format context
        context_parts = ["## Retrieved Context (from multiple methods)\n"]
        
        for i, result in enumerate(retrieval_response['results'], 1):
            text = result.get('text', '').strip()
            url = result.get('metadata', {}).get('url', result.get('url', ''))
            methods = result.get('methods', [])
            method_count = result.get('method_count', 1)
            
            context_parts.append(f"### Source {i} (Found by: {', '.join(methods)}, Quality: {'★' * method_count})")
            if url:
                context_parts.append(f"URL: {url}")
            context_parts.append(f"{text}\n")
        
        context = '\n'.join(context_parts)
        
        # Generate answer
        if QA_AVAILABLE:
            self.qa_gen.model = model
            answer_response = self.qa_gen.generate_answer(question, context)
            
            return {
                **answer_response,
                'retrieval': {
                    'method': retrieval_response['method'],
                    'num_results': len(retrieval_response['results']),
                    'documents': retrieval_response['results'],  # Add actual documents for evaluation
                    'hyde_hypothetical': retrieval_response['hyde_hypothetical'],
                    'fusion_queries': retrieval_response['fusion_queries']
                }
            }
        else:
            return {
                'question': question,
                'context': context,
                'retrieval': retrieval_response
            }
    
    def pretty_print(self, response: Dict):
        """Pretty print response"""
        print("\n" + "="*60)
        print("ADVANCED RAG RESULTS")
        print("="*60)
        print(f"Q: {response['question']}\n")
        print(f"A: {response['answer']}\n")
        print("="*60)
        print("METADATA")
        print("="*60)
        print(f"Model: {response.get('model', 'N/A')}")
        print(f"Tokens: {response.get('tokens', {})}")
        print(f"Retrieval Method: {response.get('retrieval', {}).get('method', 'N/A')}")
        print(f"Documents Used: {response.get('retrieval', {}).get('num_results', 0)}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    ap = argparse.ArgumentParser(description="Advanced RAG with HyDE + RAG Fusion")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name")
    ap.add_argument("--top_k", type=int, default=8, help="Number of results")
    args = ap.parse_args()
    
    # Initialize advanced RAG
    rag = AdvancedRAG()
    
    # Run pipeline
    response = rag.answer_with_advanced_rag(
        question=args.question,
        top_k=args.top_k,
        model=args.model
    )
    
    # Print results
    rag.pretty_print(response)


if __name__ == "__main__":
    main()
