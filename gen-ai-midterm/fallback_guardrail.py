#!/usr/bin/env python3
"""
Fallback Guardrail with Firecrawl Integration
Group 5 | UChicago MS-ADS RAG System

Features:
- Detects when LLM doesn't have information in knowledge base
- Automatically triggers Firecrawl for live data retrieval
- Regenerates answer with fresh context
- Configurable fallback URLs
- Logging and monitoring
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import components
try:
    from dynamic_retrieval import DynamicRetriever
    DYNAMIC_AVAILABLE = True
except ImportError:
    print("⚠ dynamic_retrieval not available")
    DYNAMIC_AVAILABLE = False

try:
    from qa_generator import QAGenerator
    QA_AVAILABLE = True
except ImportError:
    print("⚠ qa_generator not available")
    QA_AVAILABLE = False

try:
    from intelligent_search_agent import IntelligentSearchAgent
    INTELLIGENT_AGENT_AVAILABLE = True
except ImportError:
    print("⚠ intelligent_search_agent not available")
    INTELLIGENT_AGENT_AVAILABLE = False


class FallbackGuardrail:
    """
    Guardrail that detects knowledge gaps and triggers live data retrieval
    """
    
    # Patterns that indicate the model doesn't have information
    UNCERTAINTY_PATTERNS = [
        r"i don'?t have that information",
        r"not (in|available in) (my|the) knowledge base",
        r"i (cannot|can't) (find|locate|provide) (that|this) information",
        r"information (is )?not available",
        r"(sorry|unfortunately).{0,50}(don'?t|cannot|can't) (find|provide)",
        r"no information (about|on|regarding)",
        r"not enough (information|context|data)",
        r"the (context|information) doesn'?t (contain|include|mention)",
        r"based on.{0,30}(no|limited) information",
        r"i'?m unable to (answer|provide|find)",
    ]
    
    # Default fallback URLs for UChicago MS-ADS program
    # Updated to include course-specific pages for better curriculum information
    DEFAULT_FALLBACK_URLS = [
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",  # Course curriculum details
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/",
        "https://datascience.uchicago.edu/education/tuition-fees-aid/",
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/",  # In-person program details
        "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/online-program/",  # Online program details
    ]
    
    def __init__(
        self,
        qa_generator: Optional[QAGenerator] = None,
        dynamic_retriever: Optional[DynamicRetriever] = None,
        intelligent_agent: Optional['IntelligentSearchAgent'] = None,
        fallback_urls: Optional[List[str]] = None,
        enable_logging: bool = True,
        use_ai_url_selection: bool = True
    ):
        """
        Initialize fallback guardrail
        
        Args:
            qa_generator: QA generator instance
            dynamic_retriever: Dynamic retriever instance  
            intelligent_agent: Intelligent search agent for AI-powered URL selection
            fallback_urls: List of URLs to scrape when knowledge gap detected
            enable_logging: Enable logging
            use_ai_url_selection: Use AI to intelligently select URLs
        """
        self.qa_generator = qa_generator
        self.dynamic_retriever = dynamic_retriever
        self.intelligent_agent = intelligent_agent
        self.fallback_urls = fallback_urls or self.DEFAULT_FALLBACK_URLS
        self.enable_logging = enable_logging
        self.use_ai_url_selection = use_ai_url_selection
        
        # Compile uncertainty patterns
        self.uncertainty_regex = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.UNCERTAINTY_PATTERNS
        ]
        
        # Initialize components if not provided
        if self.qa_generator is None and QA_AVAILABLE:
            self.qa_generator = QAGenerator(model="gpt-4o-mini")
            if self.enable_logging:
                print("✓ QA Generator initialized")
        
        if self.dynamic_retriever is None and DYNAMIC_AVAILABLE:
            self.dynamic_retriever = DynamicRetriever()
            if self.enable_logging:
                print("✓ Dynamic Retriever initialized")
        
        # Initialize intelligent agent if AI URL selection enabled
        if self.use_ai_url_selection and self.intelligent_agent is None and INTELLIGENT_AGENT_AVAILABLE:
            self.intelligent_agent = IntelligentSearchAgent(model="gpt-4o-mini")
            if self.enable_logging:
                print("✓ Intelligent Search Agent initialized (AI-powered URL selection)")
        
        if self.enable_logging:
            print(f"\n{'='*60}")
            print("Fallback Guardrail Initialized")
            print(f"  AI URL Selection: {self.use_ai_url_selection}")
            print(f"  Fallback URLs: {len(self.fallback_urls)}")
            print(f"  Uncertainty patterns: {len(self.uncertainty_regex)}")
            print(f"{'='*60}\n")
    
    def detect_uncertainty(self, answer: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if answer indicates missing information
        
        Args:
            answer: Generated answer text
            
        Returns:
            Tuple of (is_uncertain, matched_pattern)
        """
        answer_lower = answer.lower()
        
        for pattern_regex in self.uncertainty_regex:
            match = pattern_regex.search(answer_lower)
            if match:
                return True, match.group(0)
        
        return False, None
    
    def get_relevant_urls(
        self,
        question: str,
        all_urls: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select relevant URLs based on question keywords
        
        Args:
            question: User question
            all_urls: List of all available URLs
            
        Returns:
            List of relevant URLs
        """
        if all_urls is None:
            all_urls = self.fallback_urls
        
        question_lower = question.lower()
        
        # Keyword to URL mapping
        url_keywords = {
            'curriculum': ['curriculum', 'course', 'courses', 'classes', 'elective', 'core', 'required', 'program structure', 'syllabus'],
            'admissions': ['admission', 'admissions', 'apply', 'application', 'requirement', 'prerequisite', 'deadline', 'deadlines'],
            'tuition-aid': ['tuition', 'cost', 'fee', 'scholarship', 'financial', 'aid', 'funding', 'price'],
        }
        
        # Find matching URLs
        relevant_urls = []
        for url in all_urls:
            url_lower = url.lower()
            
            # Check if any keywords match
            for category, keywords in url_keywords.items():
                if category in url_lower:
                    # Check if question contains related keywords
                    if any(keyword in question_lower for keyword in keywords):
                        relevant_urls.append(url)
                        break
        
        # If no specific match, return all URLs
        if not relevant_urls:
            relevant_urls = all_urls[:2]  # Take first 2 as default
        
        return relevant_urls
    
    def fetch_live_data(
        self,
        question: str,
        urls: Optional[List[str]] = None
    ) -> Dict:
        """
        Fetch live data using Firecrawl with AI-powered URL selection
        
        Args:
            question: User question
            urls: URLs to scrape (if None, will use AI or keyword-based selection)
            
        Returns:
            Dict with scraped results
        """
        if not DYNAMIC_AVAILABLE:
            return {'error': 'Dynamic retrieval not available'}
        
        if urls is None:
            # Use AI-powered URL selection if available
            if self.use_ai_url_selection and self.intelligent_agent:
                if self.enable_logging:
                    print(f"\n🤖 Using AI to select relevant URLs for: '{question}'\n")
                
                search_result = self.intelligent_agent.intelligent_search(
                    query=question,
                    max_urls=3
                )
                urls = search_result['selected_urls']
                
                if self.enable_logging:
                    print(f"   AI Keywords: {', '.join(search_result.get('keywords', []))}")
            else:
                # Fall back to keyword-based selection
                urls = self.get_relevant_urls(question)
        
        if self.enable_logging:
            print(f"\n🔥 Triggering live data retrieval...")
            print(f"   URLs to scrape: {len(urls)} pages\n")
        
        # Fetch with dynamic retriever
        response = self.dynamic_retriever.dynamic_retrieve(
            query=question,
            urls=urls,
            use_cache=True
        )
        
        return response
    
    def format_dynamic_context(self, dynamic_response: Dict) -> str:
        """
        Format dynamic retrieval results as context
        
        Args:
            dynamic_response: Response from dynamic retriever
            
        Returns:
            Formatted context string
        """
        if not dynamic_response.get('results'):
            return "No live data retrieved."
        
        lines = ["## IMPORTANT: Live Retrieved Information (Freshly Scraped from Official Website)\n"]
        lines.append("The following information was just retrieved from the official UChicago MS-ADS website.")
        lines.append("Use this information to answer the user's question.\n")
        
        for i, result in enumerate(dynamic_response['results'], 1):
            url = result.get('url', '')
            title = result.get('title', 'Unknown')
            content = result.get('content', '')
            
            # Take more chunks for better coverage (increased from 3 to 10)
            chunks = result.get('chunks', [])[:10]  # First 10 chunks
            
            lines.append(f"### Source {i}: {title}")
            lines.append(f"URL: {url}")
            lines.append(f"Scraped: {result.get('scraped_at', 'Unknown')}\n")
            
            if chunks:
                for j, chunk in enumerate(chunks, 1):
                    lines.append(f"**Chunk {j}:**")
                    lines.append(f"{chunk}\n")
            else:
                # Use raw content (increased from 1000 to 3000 chars)
                lines.append(f"{content[:3000]}...\n")
        
        lines.append("\n## Instructions:")
        lines.append("Based on the live information above, provide a helpful and accurate answer.")
        lines.append("If you find relevant information, use it to answer the question completely.")
        
        return '\n'.join(lines)
    
    def answer_with_fallback(
        self,
        question: str,
        initial_context: Optional[str] = None,
        initial_results: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Main guardrail logic: Generate answer with automatic fallback
        
        Args:
            question: User question
            initial_context: Initial retrieved context (optional)
            initial_results: Initial retrieval results (optional)
            
        Returns:
            Dict with answer, metadata, and fallback info
        """
        if self.enable_logging:
            print(f"\n{'='*60}")
            print(f"FALLBACK GUARDRAIL")
            print(f"Question: {question}")
            print(f"{'='*60}\n")
        
        # Step 1: Generate initial answer
        if self.enable_logging:
            print("[1/3] Generating initial answer...")
        
        if initial_context is None and initial_results:
            initial_context = self.qa_generator.format_context(initial_results)
        elif initial_context is None:
            initial_context = "No context provided."
        
        initial_response = self.qa_generator.generate_answer(
            question=question,
            context=initial_context
        )
        
        initial_answer = initial_response['answer']
        
        # Step 2: Check for uncertainty
        if self.enable_logging:
            print("[2/3] Checking for knowledge gaps...")
        
        is_uncertain, matched_pattern = self.detect_uncertainty(initial_answer)
        
        if not is_uncertain:
            if self.enable_logging:
                print("✓ Answer looks confident, no fallback needed\n")
            
            return {
                **initial_response,
                'fallback_triggered': False,
                'initial_answer': initial_answer,
                'confidence': 'high'
            }
        
        # Step 3: Trigger fallback
        if self.enable_logging:
            print(f"⚠ Uncertainty detected: '{matched_pattern}'")
            print("[3/3] Triggering live data fallback...\n")
        
        # Fetch live data
        dynamic_response = self.fetch_live_data(question)
        
        if dynamic_response.get('error') or not dynamic_response.get('results'):
            if self.enable_logging:
                print("✗ Live data retrieval failed, returning initial answer\n")
            
            return {
                **initial_response,
                'fallback_triggered': True,
                'fallback_success': False,
                'initial_answer': initial_answer,
                'error': 'Live retrieval failed'
            }
        
        # Format new context
        live_context = self.format_dynamic_context(dynamic_response)
        
        # Regenerate answer with live context using fallback prompt
        if self.enable_logging:
            print("🔄 Regenerating answer with live data (using fallback prompt)...\n")
        
        final_response = self.qa_generator.generate_answer(
            question=question,
            context=live_context,
            use_fallback_prompt=True  # Use fallback prompt for live data
        )
        
        final_answer = final_response['answer']
        
        # Check if new answer is still uncertain
        is_still_uncertain, _ = self.detect_uncertainty(final_answer)
        
        if self.enable_logging:
            if is_still_uncertain:
                print("⚠ Answer still uncertain after fallback")
            else:
                print("✓ Successfully generated answer with live data!\n")
        
        return {
            **final_response,
            'fallback_triggered': True,
            'fallback_success': not is_still_uncertain,
            'initial_answer': initial_answer,
            'live_sources': [r['url'] for r in dynamic_response['results']],
            'live_data_scraped': dynamic_response['num_retrieved'],
            'uncertainty_pattern': matched_pattern,
            'confidence': 'medium' if is_still_uncertain else 'high_with_live_data'
        }
    
    def pretty_print(self, response: Dict):
        """Pretty print guardrail response"""
        print("\n" + "="*60)
        print("GUARDRAIL RESULTS")
        print("="*60)
        print(f"Q: {response['question']}\n")
        
        if response.get('fallback_triggered'):
            print("⚠ FALLBACK TRIGGERED")
            print(f"   Initial answer was uncertain")
            print(f"   Pattern: {response.get('uncertainty_pattern', 'N/A')}\n")
            
            if response.get('fallback_success'):
                print("✓ Successfully retrieved live data!")
                print(f"   Sources: {response.get('live_data_scraped', 0)} URLs")
                print(f"   URLs: {response.get('live_sources', [])}\n")
            else:
                print("✗ Fallback failed\n")
        
        print(f"A: {response['answer']}\n")
        print("="*60)
        print("METADATA")
        print("="*60)
        print(f"Confidence: {response.get('confidence', 'unknown')}")
        print(f"Fallback triggered: {response.get('fallback_triggered', False)}")
        print(f"Tokens: {response.get('tokens', {})}")
        print("="*60 + "\n")


def main():
    """Main execution for testing"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Test fallback guardrail")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--urls", nargs='+', help="Custom fallback URLs")
    args = ap.parse_args()
    
    # Initialize guardrail
    guardrail = FallbackGuardrail(
        fallback_urls=args.urls if args.urls else None
    )
    
    # Test with question
    response = guardrail.answer_with_fallback(
        question=args.question
    )
    
    # Print results
    guardrail.pretty_print(response)


if __name__ == "__main__":
    main()
