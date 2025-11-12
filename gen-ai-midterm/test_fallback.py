#!/usr/bin/env python3
"""
Test script to verify fallback guardrail with live web crawl
"""

import sys
from fallback_guardrail import FallbackGuardrail

def test_fallback():
    """Test fallback mechanism"""
    
    print("\n" + "="*70)
    print("TESTING FALLBACK GUARDRAIL WITH LIVE WEB CRAWL")
    print("="*70)
    
    # Initialize guardrail
    print("\n[1] Initializing Fallback Guardrail...")
    try:
        guardrail = FallbackGuardrail(enable_logging=True)
        print("✓ Guardrail initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize guardrail: {e}")
        return False
    
    # Test question that should trigger fallback (not in knowledge base)
    test_question = "What are the core courses in the MS in Applied Data Science program?"
    
    print(f"\n[2] Testing with question: '{test_question}'")
    print("    This question should trigger fallback since it's likely not in the static KB")
    
    try:
        # Run answer with fallback
        response = guardrail.answer_with_fallback(
            question=test_question,
            initial_context="No initial context provided."
        )
        
        print("\n[3] Analyzing Response...")
        print(f"    Fallback triggered: {response.get('fallback_triggered', False)}")
        print(f"    Fallback success: {response.get('fallback_success', False)}")
        print(f"    Confidence: {response.get('confidence', 'unknown')}")
        
        if response.get('fallback_triggered'):
            print(f"\n    Initial answer (before fallback):")
            print(f"    {response.get('initial_answer', 'N/A')[:200]}...")
            
            if response.get('live_sources'):
                print(f"\n    Live sources scraped:")
                for url in response.get('live_sources', []):
                    print(f"      - {url}")
                print(f"    Total: {response.get('live_data_scraped', 0)} URLs")
            
            print(f"\n    Final answer (after fallback):")
            print(f"    {response.get('answer', 'N/A')[:200]}...")
        else:
            print(f"\n    Answer (no fallback needed):")
            print(f"    {response.get('answer', 'N/A')[:200]}...")
        
        print("\n[4] Full Response Details:")
        guardrail.pretty_print(response)
        
        # Check if answer uses fallback information
        print("\n[5] Verification:")
        answer = response.get('answer', '')
        
        if response.get('fallback_triggered'):
            # Check if the answer seems to use new information
            has_live_info = any([
                'retrieved' in answer.lower(),
                'based on the live' in answer.lower(),
                'from the website' in answer.lower(),
                len(answer) > 100  # Substantial answer
            ])
            
            if has_live_info and not any([
                "i don't have" in answer.lower(),
                "not available" in answer.lower(),
                "not in my knowledge base" in answer.lower()
            ]):
                print("✓ SUCCESS: Answer appears to use fallback information")
                return True
            else:
                print("⚠ WARNING: Fallback triggered but answer still uncertain")
                print(f"    This might indicate an issue with using fallback content")
                return False
        else:
            print("✓ Fallback was not triggered (answer confident from static KB)")
            return True
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_retrieval():
    """Test dynamic retrieval directly"""
    
    print("\n" + "="*70)
    print("TESTING DYNAMIC RETRIEVAL (FIRECRAWL) DIRECTLY")
    print("="*70)
    
    try:
        from dynamic_retrieval import DynamicRetriever
        
        print("\n[1] Initializing Dynamic Retriever...")
        retriever = DynamicRetriever(use_embedding=False)
        print("✓ Retriever initialized")
        
        test_urls = [
            "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
        ]
        
        print(f"\n[2] Testing scraping of: {test_urls[0]}")
        
        response = retriever.dynamic_retrieve(
            query="Test query for courses",
            urls=test_urls,
            use_cache=True
        )
        
        print(f"\n[3] Results:")
        print(f"    URLs requested: {response['num_urls']}")
        print(f"    URLs retrieved: {response['num_retrieved']}")
        
        if response['results']:
            result = response['results'][0]
            print(f"\n    First result:")
            print(f"      URL: {result['url']}")
            print(f"      Title: {result.get('title', 'N/A')}")
            print(f"      Content length: {len(result.get('content', ''))} chars")
            print(f"      Chunks: {len(result.get('chunks', []))}")
            print(f"      Content preview: {result.get('content', '')[:300]}...")
            
            if len(result.get('content', '')) > 200:
                print("\n✓ SUCCESS: Web scraping working correctly")
                return True
            else:
                print("\n⚠ WARNING: Content seems too short")
                return False
        else:
            print("\n✗ FAILED: No results retrieved")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FALLBACK GUARDRAIL TEST SUITE")
    print("="*70)
    
    # Test 1: Dynamic retrieval
    print("\n\nTEST 1: Dynamic Retrieval")
    test1_passed = test_dynamic_retrieval()
    
    # Test 2: Full fallback pipeline
    print("\n\nTEST 2: Full Fallback Pipeline")
    test2_passed = test_fallback()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Test 1 (Dynamic Retrieval): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (Fallback Pipeline): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("="*70 + "\n")
    
    sys.exit(0 if (test1_passed and test2_passed) else 1)
