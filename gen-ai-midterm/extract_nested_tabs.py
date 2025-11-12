#!/usr/bin/env python3
"""
Extract Nested Tab Content from UChicago Data Science Pages
Solves the problem of scraping content hidden in tabs and accordions

This script:
1. Clicks through all main tabs (NONCREDIT, CORE, ELECTIVE COURSES)
2. Expands all accordion items within each tab
3. Extracts complete content after all interactions
4. Saves to vector store
"""

from dynamic_retrieval import DynamicRetriever
import json


def extract_in_person_curriculum():
    """
    Extract curriculum from in-person program page with nested tabs/accordions
    
    Strategy:
    1. Use JavaScript to click all tabs and expand all accordions at once
    2. This avoids timing issues with sequential clicks
    3. Capture content after everything is expanded
    """
    
    print("="*80)
    print("EXTRACTING NESTED TAB CONTENT: IN-PERSON PROGRAM")
    print("="*80)
    
    retriever = DynamicRetriever()
    url = "https://datascience.uchicago.edu/education/masters-programs/in-person-program/"
    
    # Define comprehensive tab actions
    # We'll use JavaScript to expand everything at once
    tab_actions = [
        # Initial wait for page load
        {"wait": 3000},
        
        # Scroll to curriculum section
        {"scroll": "down", "wait": 1000},
        {"scroll": "down", "wait": 1000},
        
        # JavaScript to click all tabs and expand all accordions
        {
            "executeJavascript": """
                (function() {
                    // Function to click all tabs
                    function clickAllTabs() {
                        // Find all tab buttons/links
                        const tabSelectors = [
                            'a[href*="noncredit"]',
                            'a[href*="core"]', 
                            'a[href*="elective"]',
                            '[data-tabs*="noncredit"]',
                            '[data-tabs*="core"]',
                            '[data-tabs*="elective"]',
                            '.tabs-title a',
                            '[role="tab"]'
                        ];
                        
                        tabSelectors.forEach(selector => {
                            const tabs = document.querySelectorAll(selector);
                            tabs.forEach(tab => {
                                try {
                                    tab.click();
                                } catch(e) {}
                            });
                        });
                    }
                    
                    // Function to expand all accordions
                    function expandAllAccordions() {
                        // Find accordion headers - multiple possible selectors
                        const accordionSelectors = [
                            '.accordion_item',
                            '.accordion-item', 
                            '[class*="accordion"]',
                            '.list-accordion',
                            '[data-accordion-item]'
                        ];
                        
                        accordionSelectors.forEach(selector => {
                            const items = document.querySelectorAll(selector);
                            items.forEach(item => {
                                // Try to find the clickable element
                                const clickables = [
                                    item.querySelector('.accordion_indicator'),
                                    item.querySelector('[aria-expanded]'),
                                    item.querySelector('a'),
                                    item
                                ];
                                
                                clickables.forEach(clickable => {
                                    if (clickable) {
                                        try {
                                            // Check if not already expanded
                                            const isExpanded = clickable.getAttribute('aria-expanded') === 'true';
                                            const hasActiveClass = clickable.classList.contains('is-active');
                                            
                                            if (!isExpanded && !hasActiveClass) {
                                                clickable.click();
                                            }
                                        } catch(e) {}
                                    }
                                });
                            });
                        });
                    }
                    
                    // Execute clicks
                    clickAllTabs();
                    setTimeout(() => {
                        expandAllAccordions();
                    }, 1000);
                })();
            """,
            "wait": 3000
        },
        
        # Scroll through content to ensure everything loads
        {"scroll": "down", "wait": 1500},
        {"scroll": "down", "wait": 1500},
        {"scroll": "up", "wait": 1000},
        {"scroll": "down", "wait": 1500},
        
        # Final wait for all content to be visible
        {"wait": 2000}
    ]
    
    print(f"\nURL: {url}")
    print(f"Actions planned: {len(tab_actions)}")
    print("\nScraping with tab/accordion expansion...")
    
    # Scrape with actions
    result = retriever.scrape_with_tabs(
        url=url,
        tab_actions=tab_actions,
        use_cache=False,  # Disable cache during testing
        formats=['markdown', 'html']
    )
    
    if result:
        print("\n" + "="*80)
        print("SUCCESS - Content Extracted")
        print("="*80)
        print(f"Total content length: {len(result['markdown']):,} characters")
        print(f"Actions performed: {result['num_actions']}")
        
        # Analyze what we captured
        content = result['markdown'].lower()
        
        # Check for course-related keywords
        keywords = {
            'course names': ['time series', 'machine learning', 'statistical models', 
                           'data engineering', 'leadership', 'capstone'],
            'course codes': ['adsp', 'data'],
            'requirements': ['required', 'elective', 'core course'],
            'descriptions': ['description', 'learn', 'student', 'will']
        }
        
        print("\n" + "-"*80)
        print("CONTENT ANALYSIS")
        print("-"*80)
        
        for category, terms in keywords.items():
            found = sum(1 for term in terms if term in content)
            print(f"{category:20s}: {found}/{len(terms)} terms found")
        
        # Show a sample
        print("\n" + "-"*80)
        print("CONTENT PREVIEW (first 2000 chars)")
        print("-"*80)
        print(result['markdown'][:2000])
        print("...")
        
        # Save to file for inspection
        output_file = "scraped_in_person_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'url': result['url'],
                'markdown': result['markdown'],
                'metadata': result.get('metadata', {}),
                'scraped_at': result['scraped_at'],
                'content_length': len(result['markdown'])
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Full content saved to: {output_file}")
        
        return result
    else:
        print("\n✗ Failed to extract content")
        return None


def extract_with_simpler_approach():
    """
    Alternative: Simpler approach using only scrolling and waits
    Sometimes Firecrawl captures dynamic content if we give it enough time
    """
    
    print("\n" + "="*80)
    print("TRYING SIMPLER APPROACH: Scroll and Wait")
    print("="*80)
    
    retriever = DynamicRetriever()
    url = "https://datascience.uchicago.edu/education/masters-programs/in-person-program/"
    
    # Simple actions - just scroll and wait extensively
    tab_actions = [
        {"wait": 5000},  # Long initial wait
        {"scroll": "down", "wait": 2000},
        {"scroll": "down", "wait": 2000},
        {"scroll": "down", "wait": 2000},
        {"scroll": "down", "wait": 2000},
        {"scroll": "down", "wait": 2000},
        {"wait": 3000},  # Final wait
    ]
    
    result = retriever.scrape_with_tabs(
        url=url,
        tab_actions=tab_actions,
        use_cache=False,
        formats=['markdown']
    )
    
    if result:
        print(f"\n✓ Content length: {len(result['markdown']):,} chars")
        print(f"Preview: {result['markdown'][:500]}...")
        return result
    else:
        print("\n✗ Failed")
        return None


def compare_approaches():
    """Compare different scraping approaches"""
    
    print("\n" + "="*80)
    print("COMPARING SCRAPING APPROACHES")
    print("="*80)
    
    retriever = DynamicRetriever()
    url = "https://datascience.uchicago.edu/education/masters-programs/in-person-program/"
    
    results = {}
    
    # Approach 1: No special actions
    print("\n1. Basic scrape (no actions)...")
    basic = retriever.scrape_url(url, use_cache=False)
    if basic:
        results['basic'] = len(basic['markdown'])
        print(f"   Length: {results['basic']:,} chars")
    
    # Approach 2: Simple scroll and wait
    print("\n2. With scrolling...")
    scroll_result = retriever.scrape_with_tabs(
        url=url,
        tab_actions=[
            {"wait": 3000},
            {"scroll": "down", "wait": 2000},
            {"scroll": "down", "wait": 2000},
        ],
        use_cache=False
    )
    if scroll_result:
        results['scroll'] = len(scroll_result['markdown'])
        print(f"   Length: {results['scroll']:,} chars")
    
    # Approach 3: JavaScript execution
    print("\n3. With JavaScript (expand all)...")
    js_result = extract_in_person_curriculum()
    if js_result:
        results['javascript'] = len(js_result['markdown'])
        print(f"   Length: {results['javascript']:,} chars")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    for approach, length in results.items():
        print(f"{approach:15s}: {length:,} characters")
    
    if results:
        best = max(results.items(), key=lambda x: x[1])
        print(f"\n✓ Best approach: {best[0]} ({best[1]:,} chars)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "compare":
            compare_approaches()
        elif mode == "simple":
            extract_with_simpler_approach()
        else:
            extract_in_person_curriculum()
    else:
        # Default: use the comprehensive approach
        extract_in_person_curriculum()
