#!/usr/bin/env python3
"""
Intelligent Search Agent for UChicago MS-ADS
Uses GPT-4 to intelligently decide which pages to crawl based on user queries

Group 5 | UChicago MS-ADS RAG System
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI

# Config
try:
    from config import Config
    OPENAI_API_KEY = Config.OPENAI_API_KEY
except:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class IntelligentSearchAgent:
    """AI-powered search agent that intelligently selects URLs to crawl"""
    
    # Comprehensive list of UChicago MS-ADS related URLs with descriptions
    KNOWN_URLS = {
        "main": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
        "in_person": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/",  # IMPORTANT: Has COURSE INFO in tabs!
        "online": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/online-program/",
        "curriculum": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/curriculum/",
        "course_progressions": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",  # Detailed course list
        "admissions": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/admissions/",
        "faqs": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
        "tuition": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/tuition-financial-aid/",
        "faculty": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faculty-instructors/",
        "career": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/career-outcomes/",
        "capstone": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/capstone/",
        "student_life": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/student-experience/",
    }
    
    # Page content descriptions for AI agent
    URL_DESCRIPTIONS = {
        "in_person": "Contains detailed COURSE INFORMATION in tabs/accordions - core courses, electives, requirements",
        "course_progressions": "Contains course progressions, curriculum details, and program structure",
        "curriculum": "Overview of curriculum structure and requirements",
        "admissions": "Admission requirements, deadlines, application process",
        "faqs": "Frequently asked questions about program, courses, admissions",
        "tuition": "Tuition costs, financial aid, scholarships",
        "main": "Program overview, formats (in-person/online), general information",
        "online": "Online program details and format",
        "faculty": "Faculty members and instructors",
        "career": "Career outcomes and placement statistics",
        "capstone": "Capstone project information",
        "student_life": "Student experience and community",
    }
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize intelligent search agent
        
        Args:
            openai_api_key: OpenAI API key
            model: GPT model to use for intelligence
        """
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.model = model
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        print(f"✓ Intelligent Search Agent initialized (model: {model})")
    
    def select_urls_for_query(
        self,
        query: str,
        max_urls: int = 3
    ) -> List[str]:
        """
        Use GPT-4 to intelligently select which URLs to crawl for a given query
        
        Args:
            query: User's question
            max_urls: Maximum number of URLs to return
        
        Returns:
            List of selected URLs
        """
        # Create prompt for GPT-4 with descriptions
        urls_with_descriptions = "\n".join([
            f"- {key}: {self.URL_DESCRIPTIONS.get(key, 'General information')}"
            for key in self.KNOWN_URLS.keys()
        ])
        
        prompt = f"""You are an intelligent search agent for the University of Chicago MS in Applied Data Science (MS-ADS) program.

Given a user's question, select the MOST RELEVANT pages to crawl to answer their question.

Available pages and their content:
{urls_with_descriptions}

User Question: "{query}"

IMPORTANT NOTES:
- For questions about COURSES, COURSEWORK, CORE COURSES, or ELECTIVES: ALWAYS include "in_person" (it has detailed course info in tabs)
- For questions about CURRICULUM or PROGRAM STRUCTURE: Include "course_progressions" and "in_person"
- For questions about ADMISSIONS: Include "admissions" and "faqs"
- The "in_person" page contains the most comprehensive course information even though it needs live web crawling

Instructions:
1. Analyze what information the user needs
2. Select {max_urls} most relevant pages (return page keys like "in_person", "course_progressions", "faqs", etc.)
3. Prioritize pages with the most relevant content
4. Return ONLY the page keys as a comma-separated list (e.g., "in_person,course_progressions,faqs")

Your response (just the keys):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at selecting relevant information sources for academic program queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            # Parse response
            selected_keys = response.choices[0].message.content.strip()
            selected_keys = [k.strip() for k in selected_keys.split(',')]
            
            # Convert keys to URLs
            selected_urls = []
            for key in selected_keys[:max_urls]:
                if key in self.KNOWN_URLS:
                    selected_urls.append(self.KNOWN_URLS[key])
            
            # If no valid URLs, use defaults
            if not selected_urls:
                selected_urls = [
                    self.KNOWN_URLS["main"],
                    self.KNOWN_URLS["faqs"]
                ]
            
            print(f"🤖 AI Selected URLs for query: '{query}'")
            for i, url in enumerate(selected_urls, 1):
                # Find the key for this URL
                url_key = [k for k, v in self.KNOWN_URLS.items() if v == url][0]
                print(f"   {i}. {url_key}: {url}")
            
            return selected_urls
            
        except Exception as e:
            print(f"⚠ Error in AI URL selection: {e}")
            # Fallback to default URLs
            return [
                self.KNOWN_URLS["main"],
                self.KNOWN_URLS["faqs"]
            ]
    
    def generate_search_keywords(self, query: str) -> List[str]:
        """
        Generate search keywords from user query using GPT-4
        
        Args:
            query: User's question
        
        Returns:
            List of search keywords
        """
        prompt = f"""Extract 3-5 key search terms from this question about the UChicago MS in Applied Data Science program.

Question: "{query}"

Return ONLY the keywords as a comma-separated list (e.g., "admissions, deadlines, requirements"):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract key search terms from questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            keywords = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords.split(',')]
            
            print(f"🔍 AI Generated Keywords: {', '.join(keywords)}")
            return keywords
            
        except Exception as e:
            print(f"⚠ Error generating keywords: {e}")
            return []
    
    def is_valid_uchicago_url(self, url: str) -> bool:
        """
        Check if URL is within UChicago MS-ADS domain
        
        Args:
            url: URL to check
        
        Returns:
            True if valid, False otherwise
        """
        allowed_domains = [
            "datascience.uchicago.edu",
        ]
        
        return any(domain in url for domain in allowed_domains)
    
    def intelligent_search(
        self,
        query: str,
        max_urls: int = 3
    ) -> Dict:
        """
        Perform intelligent search: generate keywords and select URLs
        
        Args:
            query: User's question
            max_urls: Maximum URLs to select
        
        Returns:
            Dict with keywords and selected URLs
        """
        print(f"\n{'='*60}")
        print(f"🤖 INTELLIGENT SEARCH AGENT")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # Generate search keywords
        keywords = self.generate_search_keywords(query)
        
        # Select relevant URLs
        urls = self.select_urls_for_query(query, max_urls=max_urls)
        
        # Verify all URLs are valid
        urls = [url for url in urls if self.is_valid_uchicago_url(url)]
        
        result = {
            "query": query,
            "keywords": keywords,
            "selected_urls": urls,
            "num_urls": len(urls)
        }
        
        print(f"\n✓ Intelligent search complete")
        print(f"  Keywords: {len(keywords)}")
        print(f"  URLs selected: {len(urls)}\n")
        
        return result


def main():
    """Test the intelligent search agent"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Intelligent Search Agent")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--max-urls", type=int, default=3, help="Max URLs to select")
    args = ap.parse_args()
    
    # Initialize agent
    agent = IntelligentSearchAgent()
    
    # Perform intelligent search
    result = agent.intelligent_search(args.query, max_urls=args.max_urls)
    
    # Print results
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Query: {result['query']}")
    print(f"\nKeywords: {', '.join(result['keywords'])}")
    print(f"\nSelected URLs ({result['num_urls']}):")
    for i, url in enumerate(result['selected_urls'], 1):
        print(f"  {i}. {url}")


if __name__ == "__main__":
    main()
