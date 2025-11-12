#!/usr/bin/env python3
"""
GPT-4 QA Generation with Grounding
Group 5 | UChicago MS-ADS RAG System

Features:
- System prompt enforcing grounding in retrieved content
- Clean Markdown context formatting
- Token limit management (< 8k)
- Optional streaming output
- Integration with confidence router
"""

import argparse
import tiktoken
from typing import List, Dict, Optional

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠ OpenAI not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# LangChain (optional)
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.chains import LLMChain
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except Exception as e:
    # Catch all errors including pydantic compatibility issues
    print(f"⚠ LangChain not available ({type(e).__name__}). Using direct OpenAI API.")
    LANGCHAIN_AVAILABLE = False

# Config
try:
    from config import Config
    OPENAI_API_KEY = Config.OPENAI_API_KEY
except:
    import os
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Import router (defer to avoid import chain issues)
ROUTER_AVAILABLE = False
ConfidenceRouter = None

def _load_router():
    """Lazy load router to avoid import chain issues"""
    global ROUTER_AVAILABLE, ConfidenceRouter
    if not ROUTER_AVAILABLE and ConfidenceRouter is None:
        try:
            from confidence_router import ConfidenceRouter as CR
            ConfidenceRouter = CR
            ROUTER_AVAILABLE = True
        except Exception as e:
            print(f"⚠ confidence_router not available ({type(e).__name__})")
            ROUTER_AVAILABLE = False
    return ROUTER_AVAILABLE


# System prompt
SYSTEM_PROMPT = """You are a helpful admissions assistant for the University of Chicago's MS in Applied Data Science program.

CRITICAL RULES:
1. Ground ALL answers STRICTLY in the provided context
2. If information is NOT in the context, say "I don't have that information in my knowledge base"
3. DO NOT make up facts or information
4. Cite sources when available (mention URLs)
5. Be specific and accurate - use program language where possible
6. Keep answers concise but complete
7. Provide links to the program page for sources.

Your role is to help prospective students understand the program using ONLY the information provided in the context below."""

# System prompt for fallback/live data scenarios
SYSTEM_PROMPT_FALLBACK = """You are a helpful admissions assistant for the University of Chicago's MS in Applied Data Science program.

CRITICAL RULES:
1. The context below contains LIVE, freshly scraped information from the official UChicago website
2. Use this live information to answer the user's question accurately and completely
3. Ground ALL answers STRICTLY in the provided context
4. DO NOT make up facts beyond what's in the context
5. Cite the source URLs when providing information
6. Be specific and accurate - use exact program details from the context
7. Keep answers helpful and informative

Your role is to help prospective students using the LIVE information that was just retrieved from the official website."""




class QAGenerator:
    """GPT-4 QA Generator with grounding in retrieved content"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_context_tokens: int = 6000,
        max_response_tokens: int = 1000,
        temperature: float = 0.3,
        use_streaming: bool = False
    ):
        """
        Initialize QA generator
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4-turbo-preview, etc.)
            max_context_tokens: Max tokens for context
            max_response_tokens: Max tokens for response
            temperature: Temperature for generation
            use_streaming: Enable streaming output
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.use_streaming = use_streaming
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key)
            print(f"✓ OpenAI client initialized (model: {model})")
        else:
            raise ImportError("OpenAI not installed")
        
        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print(f"✓ QA Generator ready")
        print(f"  Max context tokens: {max_context_tokens}")
        print(f"  Max response tokens: {max_response_tokens}")
        print(f"  Streaming: {use_streaming}\n")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def format_context(
        self,
        results: List[Dict],
        format_type: str = "markdown"
    ) -> str:
        """
        Format retrieved results as clean context
        
        Args:
            results: Retrieved results
            format_type: "markdown" or "paragraph"
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        if format_type == "markdown":
            # Markdown bullet list format
            lines = ["## Retrieved Context\n"]
            
            for i, result in enumerate(results, 1):
                text = result.get('text', result.get('content', '')).strip()
                url = result.get('url', '')
                
                if url:
                    lines.append(f"### Source {i}: {url}\n")
                else:
                    lines.append(f"### Source {i}\n")
                
                lines.append(f"{text}\n")
            
            return '\n'.join(lines)
        
        else:
            # Paragraph format
            paragraphs = []
            for i, result in enumerate(results, 1):
                text = result.get('text', result.get('content', '')).strip()
                url = result.get('url', '')
                
                if url:
                    paragraphs.append(f"[Source {i} from {url}]: {text}")
                else:
                    paragraphs.append(f"[Source {i}]: {text}")
            
            return '\n\n'.join(paragraphs)
    
    def truncate_context(
        self,
        context: str,
        question: str
    ) -> str:
        """
        Truncate context to fit within token limits
        
        Args:
            context: Full context
            question: User question
            
        Returns:
            Truncated context
        """
        system_tokens = self.count_tokens(SYSTEM_PROMPT)
        question_tokens = self.count_tokens(question)
        
        # Reserve tokens for system, question, and response
        available_tokens = (
            self.max_context_tokens - 
            system_tokens - 
            question_tokens - 
            100  # Safety buffer
        )
        
        context_tokens = self.count_tokens(context)
        
        if context_tokens <= available_tokens:
            return context
        
        # Truncate context
        print(f"⚠ Context too long ({context_tokens} tokens), truncating to {available_tokens}...")
        
        # Simple truncation by characters (approximate)
        ratio = available_tokens / context_tokens
        target_chars = int(len(context) * ratio * 0.9)  # 90% to be safe
        
        return context[:target_chars] + "\n\n[Context truncated due to length...]"
    
    def generate_answer(
        self,
        question: str,
        context: str,
        use_fallback_prompt: bool = False
    ) -> Dict:
        """
        Generate answer using GPT-4
        
        Args:
            question: User question
            context: Retrieved context
            use_fallback_prompt: Use fallback system prompt (for live data)
            
        Returns:
            Dict with answer and metadata
        """
        # Truncate context if needed
        context = self.truncate_context(context, question)
        
        # Select appropriate system prompt
        system_prompt = SYSTEM_PROMPT_FALLBACK if use_fallback_prompt else SYSTEM_PROMPT
        
        # Build prompt
        if use_fallback_prompt:
            user_prompt = f"""Context:
{context}

Question: {question}

Please provide a helpful, accurate answer using the live information provided in the context above."""
        else:
            user_prompt = f"""Context:
{context}

Question: {question}

Please provide a helpful, accurate answer based ONLY on the context above. If the answer is not in the context, say so."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Count tokens
        total_prompt_tokens = sum(self.count_tokens(msg["content"]) for msg in messages)
        print(f"Prompt tokens: {total_prompt_tokens}")
        
        if total_prompt_tokens > self.max_context_tokens:
            print(f"⚠ Warning: Prompt exceeds limit ({total_prompt_tokens} > {self.max_context_tokens})")
        
        # Generate answer
        if self.use_streaming:
            print("\n" + "="*60)
            print("ANSWER (streaming):")
            print("="*60 + "\n")
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_response_tokens,
                stream=True
            )
            
            answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)
                    answer += content
            
            print("\n\n" + "="*60 + "\n")
            
            return {
                'question': question,
                'answer': answer,
                'streaming': True,
                'tokens': {
                    'prompt': total_prompt_tokens,
                    'completion': self.count_tokens(answer)
                }
            }
        
        else:
            # Non-streaming
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_response_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                'question': question,
                'answer': answer,
                'streaming': False,
                'tokens': {
                    'prompt': response.usage.prompt_tokens,
                    'completion': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                },
                'model': self.model,
                'finish_reason': response.choices[0].finish_reason
            }
    
    def answer_with_retrieval(
        self,
        question: str,
        router: Optional[ConfidenceRouter] = None,
        fallback_urls: Optional[List[str]] = None,
        format_type: str = "markdown"
    ) -> Dict:
        """
        Full pipeline: retrieve context and generate answer
        
        Args:
            question: User question
            router: Confidence router (if None, will create one)
            fallback_urls: URLs for dynamic retrieval
            format_type: Context format
            
        Returns:
            Dict with answer and all metadata
        """
        print(f"\n{'='*60}")
        print(f"QA PIPELINE")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Initialize router if needed
        if router is None:
            _load_router()  # Load router if not already loaded
            if ROUTER_AVAILABLE and ConfidenceRouter is not None:
                router = ConfidenceRouter()
        
        # Retrieve context
        print("[1/2] Retrieving context...")
        if router:
            retrieval_response = router.route_query(
                question,
                top_k=5,
                fallback_urls=fallback_urls
            )
            results = retrieval_response['results']
            routing_info = {
                'routing': retrieval_response['routing'],
                'confidence': retrieval_response['scores']['confidence'],
                'dynamic_triggered': retrieval_response['dynamic_triggered']
            }
        else:
            # Fallback to simple retrieval
            print("⚠ No router available, skipping retrieval")
            results = []
            routing_info = {'routing': 'none'}
        
        # Format context
        context = self.format_context(results, format_type=format_type)
        
        # Generate answer
        print("[2/2] Generating answer...")
        answer_response = self.generate_answer(question, context)
        
        # Combine results
        return {
            **answer_response,
            'retrieval': {
                'num_results': len(results),
                **routing_info
            },
            'context': context
        }
    
    def pretty_print(self, response: Dict):
        """Pretty print QA response"""
        print("\n" + "="*60)
        print("QUESTION & ANSWER")
        print("="*60)
        print(f"Q: {response['question']}\n")
        print(f"A: {response['answer']}\n")
        print("="*60)
        print("METADATA")
        print("="*60)
        print(f"Model: {response.get('model', 'N/A')}")
        print(f"Tokens: {response.get('tokens', {})}")
        print(f"Retrieval: {response.get('retrieval', {})}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    ap = argparse.ArgumentParser(description="GPT-4 QA with grounding")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--model", default="gpt-4-turbo-preview", help="Model name")
    ap.add_argument("--stream", action="store_true", help="Enable streaming")
    ap.add_argument("--fallback-urls", nargs='+', help="URLs for dynamic retrieval")
    ap.add_argument("--no-retrieval", action="store_true", help="Skip retrieval, test QA only")
    args = ap.parse_args()
    
    # Initialize QA generator
    qa = QAGenerator(
        model=args.model,
        use_streaming=args.stream
    )
    
    if args.no_retrieval:
        # Test mode: manual context
        context = "The MS in Applied Data Science program requires a bachelor's degree and programming skills."
        response = qa.generate_answer(args.question, context)
    else:
        # Full pipeline with retrieval
        response = qa.answer_with_retrieval(
            question=args.question,
            fallback_urls=args.fallback_urls
        )
    
    # Print results
    if not args.stream:
        qa.pretty_print(response)


if __name__ == "__main__":
    main()
