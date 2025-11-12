# Query Synonym and Weight Update Summary

## Overview
Updated the hybrid retrieval system to include query synonym expansion for better keyword matching and adjusted the semantic/lexical weights as requested.

## Changes Made

### 1. Query Synonym Dictionary (`hybrid_retrieval.py`)
Added a comprehensive synonym dictionary for common query terms:

```python
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
```

**How it works:**
- When a user asks about "core courses", the system automatically expands the query to include synonyms like "curriculum", "required courses", etc.
- This improves keyword matching by capturing related terms the user might not have explicitly mentioned.

### 2. Query Expansion Method (`hybrid_retrieval.py`)
Added `expand_query_with_synonyms()` method:
- Checks for multi-word phrases first (e.g., "core course")
- Then checks individual words
- Combines original query with relevant synonyms
- Used in lexical search to improve BM25 keyword matching

### 3. Weight Updates

#### Updated Default Weights
Changed from **semantic 0.6 / lexical 0.4** to **semantic 0.8 / lexical 0.2**

**Files Updated:**
1. `hybrid_retrieval.py` - Default parameters in `__init__()`:
   - `semantic_weight: float = 0.8` (was 0.6)
   - `lexical_weight: float = 0.2` (was 0.4)

2. `advanced_rag.py` - HybridRetriever initialization:
   - Added explicit `semantic_weight=0.8` and `lexical_weight=0.2`

3. `confidence_router.py` - HybridRetriever initialization:
   - Added explicit `semantic_weight=0.8` and `lexical_weight=0.2`

## Impact

### Improved Query Handling
**Example Query:** "What are the core courses?"
- **Before:** Searched only for "core" and "courses"
- **After:** Searches for "core", "courses", AND "curriculum", "required courses", "core curriculum", "required classes", "core requirements"

### Weight Adjustment Benefits
- **Semantic weight increased to 0.8:** Prioritizes meaning and context over exact keyword matches
- **Lexical weight reduced to 0.2:** Still captures keyword matches but with less influence
- **Better for conceptual queries:** Users asking in different ways will get more relevant results
- **Synonym expansion compensates:** Even with lower lexical weight, expanded synonyms ensure keyword coverage

## Usage

### Using the Updated System

```bash
# Test with synonym expansion
python hybrid_retrieval.py --question "What are the core courses?" --top_k 5

# The system will automatically:
# 1. Expand "core courses" with synonyms (curriculum, required courses, etc.)
# 2. Use 0.8 weight for semantic search
# 3. Use 0.2 weight for lexical search (with expanded query)
```

### Programmatic Usage

```python
from hybrid_retrieval import HybridRetriever

# Initialize with new defaults (0.8 semantic, 0.2 lexical)
retriever = HybridRetriever()

# Query with automatic synonym expansion
results = retriever.retrieve("What are the core courses?", top_k=5)

# Synonyms are automatically applied to lexical search
```

## Configuration

The weights can still be customized if needed:

```python
# Custom weights
retriever = HybridRetriever(
    semantic_weight=0.7,  # Adjust as needed
    lexical_weight=0.3    # Adjust as needed
)
```

## Files Modified

1. `gen-ai-midterm/hybrid_retrieval.py` - Core changes
2. `gen-ai-midterm/advanced_rag.py` - Weight updates
3. `gen-ai-midterm/confidence_router.py` - Weight updates

## Testing Recommendations

1. **Test synonym expansion:**
   - Query: "What are the core courses?"
   - Should find documents about curriculum, requirements, etc.

2. **Test weight impact:**
   - Semantic queries should rank higher
   - Results should be more contextually relevant

3. **Compare results:**
   - Try same queries with old weights (--semantic_weight 0.6 --lexical_weight 0.4)
   - Compare with new defaults

## Notes

- Synonyms can be easily extended by adding to the `QUERY_SYNONYMS` dictionary
- The synonym expansion happens automatically for all lexical searches
- Semantic search remains unchanged (uses embeddings, not keywords)
- The weighted fusion combines both semantic and lexical scores with the new weights
