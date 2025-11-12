# Fallback Guardrail with Firecrawl Integration

## Overview
The Fallback Guardrail automatically detects when the LLM doesn't have information in the knowledge base and triggers Firecrawl to fetch live data from the web, then regenerates the answer with the fresh context.

## How It Works

### 1. Uncertainty Detection
The system uses regex patterns to detect when an answer indicates missing information:

**Detected Patterns:**
- "I don't have that information"
- "Not in my knowledge base"
- "I cannot find that information"
- "Information is not available"
- "Sorry, I can't provide..."
- "No information about..."
- "Not enough information"
- "The context doesn't contain..."
- And more...

### 2. Automatic Fallback Flow

```
User Query
    ↓
Initial Answer Generation (from knowledge base)
    ↓
Uncertainty Detection
    ↓
If uncertain → Trigger Firecrawl
    ↓
Fetch Live Data from relevant URLs
    ↓
Regenerate Answer with fresh context
    ↓
Return updated answer with metadata
```

### 3. Smart URL Selection
The system automatically selects relevant URLs based on question keywords:

- **Curriculum questions** → scrapes curriculum page
- **Admission questions** → scrapes admissions page
- **Tuition/cost questions** → scrapes tuition page
- **General questions** → scrapes main program pages

## Features

✅ **Automatic Detection**: No manual intervention needed  
✅ **Smart URL Routing**: Selects relevant pages to scrape  
✅ **Caching**: Caches scraped content to reduce API calls  
✅ **Transparent**: Returns both initial and final answers  
✅ **Confidence Scoring**: Indicates answer confidence level  
✅ **Metadata Rich**: Provides sources, patterns matched, etc.

## API Usage

### Endpoint
```
POST /api/query/guardrail
```

### Request
```json
{
  "question": "What are the new course offerings for 2025?"
}
```

### Response (No Fallback Needed)
```json
{
  "success": true,
  "question": "What are the core courses?",
  "answer": "The core courses include...",
  "fallback_triggered": false,
  "confidence": "high",
  "tokens": {...},
  "timestamp": "2025-11-11T17:00:00.000Z"
}
```

### Response (Fallback Triggered)
```json
{
  "success": true,
  "question": "What are the new course offerings for 2025?",
  "answer": "Based on the latest information from the website...",
  "fallback_triggered": true,
  "fallback_success": true,
  "confidence": "high_with_live_data",
  "initial_answer": "I don't have that information in my knowledge base",
  "live_sources": [
    "https://datascience.uchicago.edu/education/masters-program/curriculum/"
  ],
  "uncertainty_pattern": "i don't have that information",
  "tokens": {...},
  "timestamp": "2025-11-11T17:00:00.000Z"
}
```

## Programmatic Usage

### Python Example
```python
from fallback_guardrail import FallbackGuardrail

# Initialize
guardrail = FallbackGuardrail()

# Query with automatic fallback
response = guardrail.answer_with_fallback(
    question="What are the new admission requirements for 2025?"
)

# Check if fallback was triggered
if response['fallback_triggered']:
    print(f"✓ Fallback triggered: {response['uncertainty_pattern']}")
    print(f"✓ Live sources: {response['live_sources']}")
    
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']}")
```

### Command Line Testing
```bash
# Test the guardrail
cd gen-ai-midterm
python fallback_guardrail.py --question "What are the new courses for 2025?"

# With custom URLs
python fallback_guardrail.py \
  --question "What scholarships are available?" \
  --urls https://datascience.uchicago.edu/education/masters-program/tuition-aid/
```

## Configuration

### Default Fallback URLs
The system uses these default URLs for UChicago MS-ADS:
```python
DEFAULT_FALLBACK_URLS = [
    "https://datascience.uchicago.edu/education/masters-program/curriculum/",
    "https://datascience.uchicago.edu/education/masters-program/admissions/",
    "https://datascience.uchicago.edu/education/masters-program/tuition-aid/",
    "https://datascience.uchicago.edu/education/masters-program/",
]
```

### Custom URLs
```python
# Custom fallback URLs
guardrail = FallbackGuardrail(
    fallback_urls=[
        "https://example.com/page1",
        "https://example.com/page2"
    ]
)
```

### Adding New Uncertainty Patterns
Edit `fallback_guardrail.py`:
```python
UNCERTAINTY_PATTERNS = [
    # ... existing patterns ...
    r"your new pattern here",
    r"another pattern",
]
```

## Integration with Existing Systems

### With QA Generator
```python
from qa_generator import QAGenerator
from fallback_guardrail import FallbackGuardrail

qa = QAGenerator()
guardrail = FallbackGuardrail(qa_generator=qa)

# Get initial results
initial_response = qa.answer_with_retrieval(question)

# Apply guardrail
final_response = guardrail.answer_with_fallback(
    question=question,
    initial_results=initial_response['retrieval']['results']
)
```

### With Advanced RAG
```python
from advanced_rag import AdvancedRAG
from fallback_guardrail import FallbackGuardrail

rag = AdvancedRAG()
guardrail = FallbackGuardrail()

# First try advanced RAG
rag_response = rag.answer_with_advanced_rag(question)

# If needed, apply guardrail
if uncertainty_detected(rag_response['answer']):
    final_response = guardrail.fetch_live_data(question)
```

## Environment Variables Required

```bash
# Required for Firecrawl
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Required for answer generation
OPENAI_API_KEY=your_openai_api_key
```

## Benefits

### 1. **Always Up-to-Date**
- Automatically fetches latest information when knowledge base is outdated
- No manual updates needed

### 2. **Improved User Experience**
- Users get answers even for very recent information
- Transparent about data sources

### 3. **Cost Effective**
- Only triggers Firecrawl when needed
- Caches results to minimize API calls

### 4. **Reliable**
- Graceful fallback if live retrieval fails
- Returns initial answer with warning

## Monitoring & Logging

### Enable Logging
```python
guardrail = FallbackGuardrail(enable_logging=True)
```

### Log Output Example
```
============================================================
FALLBACK GUARDRAIL
Question: What are the new courses for 2025?
============================================================

[1/3] Generating initial answer...
Prompt tokens: 584
[2/3] Checking for knowledge gaps...
⚠ Uncertainty detected: 'i don't have that information'
[3/3] Triggering live data fallback...

🔥 Triggering live data retrieval...
   URLs to scrape: ['https://...']

🔥 Scraping with Firecrawl: https://...
✓ Scraped successfully: 5234 chars
✓ Saved to cache: https://...
✓ Created 11 chunks

🔄 Regenerating answer with live data...

✓ Successfully generated answer with live data!
```

## Best Practices

### 1. URL Selection
- Keep fallback URLs focused and relevant
- Order URLs by importance
- Limit to 2-3 URLs to reduce latency

### 2. Caching
- Enable caching for development (`use_cache=True`)
- Clear cache periodically for production
- Cache directory: `./dynamic_cache/`

### 3. Error Handling
- Always check `fallback_success` in response
- Provide user feedback when fallback fails
- Log failures for monitoring

### 4. Performance
- Firecrawl typically takes 2-5 seconds per URL
- Use async processing for multiple URLs
- Consider timeout limits

## Troubleshooting

### Fallback Not Triggering
**Check:**
1. Uncertainty patterns match your responses
2. Initial answer contains uncertainty phrases
3. Logging enabled to see detection process

### Firecrawl Errors
**Check:**
1. `FIRECRAWL_API_KEY` is set correctly
2. URLs are accessible and valid
3. Firecrawl API quota not exceeded

### Cache Issues
**Solutions:**
```bash
# Clear cache
rm -rf gen-ai-midterm/dynamic_cache/*.json

# Disable cache for testing
guardrail.fetch_live_data(question, use_cache=False)
```

## Examples

### Example 1: Recent News Query
```
Q: "What new partnerships did UChicago announce this month?"

Initial: "I don't have that information in my knowledge base"
         ↓ (Fallback triggered)
Live Scraping → Fetch latest news
         ↓
Final: "According to the latest updates from the website..."
```

### Example 2: Outdated Information
```
Q: "What is the current tuition for 2025?"

Initial: (Provides 2024 tuition from knowledge base)
         ↓ (If year mismatch detected)
Live Scraping → Fetch current tuition page
         ↓
Final: "The current tuition for 2025 is..."
```

## Files Modified

1. **`fallback_guardrail.py`** - New guardrail module
2. **`app.py`** - Added `/api/query/guardrail` endpoint
3. **`FALLBACK_GUARDRAIL_GUIDE.md`** - This documentation

## Next Steps

1. **Test the system** with various queries
2. **Monitor fallback rates** to optimize patterns
3. **Expand URL coverage** for more topics
4. **Fine-tune uncertainty detection** based on real usage
5. **Add telemetry** for fallback analytics

## Support

For issues or questions:
- Check logs with `enable_logging=True`
- Test individual components separately
- Verify API keys are set correctly
- Review Firecrawl documentation: https://docs.firecrawl.dev/

---

**Built by Group 5 | UChicago MS-ADS RAG System**
