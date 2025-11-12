#!/usr/bin/env python3
"""
Test Tab Scraping for Dynamic Content
Demonstrates how to scrape pages with tabs/accordions like the in-person program page
"""

from dynamic_retrieval import DynamicRetriever


def test_in_person_page_tabs():
    """Test scraping the in-person program page with tab clicks"""
    
    print("="*60)
    print("TEST: Scraping In-Person Program Page with Tabs")
    print("="*60)
    
    # Initialize retriever
    retriever = DynamicRetriever()
    
    # URL with course information in tabs
    url = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/"
    
    # Define tab actions to reveal course content
    # These are example selectors - may need adjustment based on actual page structure
    tab_actions = [
        # Click core courses tab
        {
            "click": "button[data-tab='core-courses']",
            "wait": 2000
        },
        # Click electives tab  
        {
            "click": "button[data-tab='electives']",
            "wait": 2000
        },
        # Alternative: if tabs are links
        {
            "click": "a[href='#curriculum']",
            "wait": 2000
        },
        # Scroll to ensure all content visible
        {
            "scroll": "down",
            "wait": 1000
        }
    ]
    
    print(f"\nURL: {url}")
    print(f"Tab actions: {len(tab_actions)}")
    
    # Scrape with tab interactions
    result = retriever.scrape_with_tabs(
        url=url,
        tab_actions=tab_actions,
        use_cache=False  # Disable cache for testing
    )
    
    if result:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"✓ Successfully scraped with tabs!")
        print(f"  Content length: {len(result['markdown'])} characters")
        print(f"  Actions performed: {result['num_actions']}")
        print(f"  Scraped at: {result['scraped_at']}")
        
        print(f"\n  Content preview (first 1000 chars):")
        print("-" * 60)
        print(result['markdown'][:1000])
        print("-" * 60)
        
        # Check if we got course information
        content_lower = result['markdown'].lower()
        course_keywords = ['core course', 'elective', 'adsp', 'data science', 'python', 'machine learning']
        found_keywords = [kw for kw in course_keywords if kw in content_lower]
        
        print(f"\n  Course-related keywords found: {len(found_keywords)}/{len(course_keywords)}")
        print(f"  Keywords: {found_keywords}")
        
        return result
    else:
        print("\n✗ Failed to scrape with tabs")
        return None


def test_simple_page_tabs():
    """Test with simpler tab actions (more generic)"""
    
    print("\n" + "="*60)
    print("TEST: Scraping with Generic Tab Actions")
    print("="*60)
    
    retriever = DynamicRetriever()
    
    url = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/"
    
    # More generic tab actions
    tab_actions = [
        # Wait for page to fully load
        {"wait": 3000},
        # Scroll down to reveal content
        {"scroll": "down", "wait": 2000},
        # Click any visible tabs (will try common selectors)
        {"click": ".tab-button", "wait": 2000},
        {"click": ".accordion-header", "wait": 2000},
    ]
    
    result = retriever.scrape_with_tabs(
        url=url,
        tab_actions=tab_actions,
        use_cache=False
    )
    
    if result:
        print(f"\n✓ Scraped: {len(result['markdown'])} chars")
        print(f"  Check if content includes 'Core Courses': {'Core Courses' in result['markdown']}")
        print(f"  Check if content includes 'ADSP': {'ADSP' in result['markdown']}")
        return result
    else:
        print("\n✗ Failed")
        return None


def compare_with_without_tabs():
    """Compare scraping with and without tab actions"""
    
    print("\n" + "="*60)
    print("COMPARISON: With vs Without Tab Actions")
    print("="*60)
    
    retriever = DynamicRetriever()
    url = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/"
    
    # Scrape without tabs
    print("\n1. Scraping WITHOUT tab actions...")
    result_no_tabs = retriever.scrape_url(url, use_cache=False)
    
    # Scrape with tabs
    print("\n2. Scraping WITH tab actions...")
    tab_actions = [
        {"scroll": "down", "wait": 2000},
        {"scroll": "down", "wait": 2000},
    ]
    result_with_tabs = retriever.scrape_with_tabs(
        url=url,
        tab_actions=tab_actions,
        use_cache=False
    )
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if result_no_tabs and result_with_tabs:
        len_no_tabs = len(result_no_tabs['markdown'])
        len_with_tabs = len(result_with_tabs['markdown'])
        
        print(f"Without tabs: {len_no_tabs:,} characters")
        print(f"With tabs:    {len_with_tabs:,} characters")
        print(f"Difference:   {len_with_tabs - len_no_tabs:,} characters ({((len_with_tabs/len_no_tabs - 1) * 100):.1f}% more)")
        
        # Check for course keywords
        keywords = ['core course', 'elective', 'requirement', 'curriculum']
        print("\nKeyword presence:")
        for kw in keywords:
            in_no_tabs = kw.lower() in result_no_tabs['markdown'].lower()
            in_with_tabs = kw.lower() in result_with_tabs['markdown'].lower()
            print(f"  '{kw}': No tabs={in_no_tabs}, With tabs={in_with_tabs}")
    else:
        print("✗ One or both scraping attempts failed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DYNAMIC TAB SCRAPING TEST SUITE")
    print("="*80)
    
    # Test 1: Specific tab actions for in-person page
    test_in_person_page_tabs()
    
    # Test 2: Generic tab actions
    # test_simple_page_tabs()
    
    # Test 3: Comparison
    # compare_with_without_tabs()
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
