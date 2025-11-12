# UChicago MS-ADS RAG System
**Group 5 | Generative AI Principles & Applications**

An advanced Retrieval-Augmented Generation (RAG) system for the University of Chicago's MS in Applied Data Science program, featuring hybrid retrieval, dynamic web scraping, fallback guardrails, and comprehensive monitoring.

---

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Python Files Documentation](#python-files-documentation)
   - [Core Application](#core-application)
   - [RAG System Components](#rag-system-components)
   - [Data Collection & Scraping](#data-collection--scraping)
   - [Vector Database Management](#vector-database-management)
   - [Evaluation & Monitoring](#evaluation--monitoring)
   - [Testing Files](#testing-files)
   - [Utility Files](#utility-files)
3. [File Relationships](#file-relationships)
4. [Setup & Installation](#setup--installation)

---

## System Overview

This RAG system provides intelligent question-answering capabilities for prospective students interested in the UChicago MS-ADS program. It combines multiple advanced techniques:

- **Hybrid Retrieval**: Semantic search (ChromaDB) + Lexical search (BM25)
- **Advanced RAG**: HyDE, RAG Fusion, and multi-method retrieval
- **Dynamic Retrieval**: Real-time web scraping with Firecrawl
- **Fallback Guardrails**: Automatic detection of knowledge gaps with live data fallback
- **Fine-tuned Embeddings**: Custom embedding model trained on program-specific data
- **Comprehensive Monitoring**: Logging, cost tracking, and performance metrics

---

## Python Files Documentation

### Core Application

#### `app.py`
**Purpose**: Flask web application - main entry point for the RAG system

**Description**: 
Production-ready Flask web application serving as the main interface for the RAG system. Provides REST API endpoints for querying the system with both standard and streaming responses.

**Key Features**:
- Multiple query endpoints (`/api/query`, `/api/query/stream`, `/api/query/guardrail`)
- Lazy loading of system components for efficient resource usage
- Automatic fallback guardrail integration
- Health check endpoint for AWS/GCP deployment
- CORS enabled for cross-origin requests
- Support for both advanced RAG and basic retrieval methods
- Streaming response capability for real-time answer generation
- Monitoring and dashboard integration

**Dependencies**: `advanced_rag.py`, `qa_generator.py`, `fallback_guardrail.py`, `config.py`, `monitoring.py`

**API Endpoints**:
- `GET /` - Main web interface
- `GET /health` - Health check for load balancers
- `POST /api/query` - Standard query with optional fallback
- `POST /api/query/stream` - Streaming query responses
- `POST /api/query/guardrail` - Query with explicit guardrail
- `GET /api/methods` - Available retrieval methods
- `GET /api/dashboard` - Monitoring dashboard data

**Usage**: 
```bash
python app.py  # Development server
gunicorn -c gunicorn_config.py app:app  # Production
```

---

#### `config.py`
**Purpose**: Centralized configuration management

**Description**:
Centralized configuration file that loads environment variables and provides a unified interface for accessing system settings across all modules.

**Key Features**:
- Loads environment variables from `.env` file using python-dotenv
- Provides configuration for API keys (OpenAI, Firecrawl)
- LangSmith observability configuration for tracing
- ChromaDB database paths and collection names
- Model configuration (embedding models, LLM settings)
- Configuration validation to ensure required keys are present
- Pretty-print configuration display (with masked API keys)

**Configuration Classes**:
- `Config` - Main configuration class with class methods for validation and setup

**Key Methods**:
- `Config.validate()` - Validates required configuration
- `Config.setup_langsmith()` - Enables LangSmith tracing
- `Config.display_config()` - Displays current configuration

**Used By**: All modules that need API keys or configuration settings

**Environment Variables Required**:
- `OPENAI_API_KEY` - OpenAI API key
- `FIRECRAWL_API_KEY` - Firecrawl API key (optional)
- `LANGCHAIN_API_KEY` - LangChain/LangSmith API key (optional)
- `LANGCHAIN_PROJECT` - LangSmith project name (optional)

---

### RAG System Components

#### `advanced_rag.py`
**Purpose**: Advanced RAG implementation with HyDE and RAG Fusion

**Description**:
Implements state-of-the-art RAG techniques including HyDE (Hypothetical Document Embeddings), RAG Fusion, and multi-method retrieval for improved answer quality.

**Key Features**:
- **HyDE**: Generates hypothetical answers and uses them for retrieval
- **RAG Fusion**: Creates multiple query variations and uses reciprocal rank fusion
- **Multi-Method Retrieval**: Combines standard, HyDE, and fusion methods
- **Curriculum Query Detection**: Special handling for course/curriculum questions
- **Course-Specific Re-ranking**: Boosts chunks containing course information
- **Method Consensus**: Prioritizes documents found by multiple methods

**Main Classes**:
- `AdvancedRAG` - Main class implementing advanced retrieval techniques

**Key Methods**:
- `hyde_retrieval()` - HyDE-based retrieval
- `generate_query_variations()` - Generate query variations for RAG Fusion
- `rag_fusion()` - RAG Fusion with reciprocal rank fusion
- `multi_method_retrieval()` - Combined multi-method approach
- `answer_with_advanced_rag()` - Complete pipeline with answer generation
- `detect_curriculum_query()` - Detects curriculum-related questions
- `rerank_for_courses()` - Re-ranks results for curriculum queries

**Dependencies**: `hybrid_retrieval.py`, `qa_generator.py`, `config.py`

**Used By**: `app.py`, `evaluation.py`

**Usage**:
```bash
python advanced_rag.py --question "What are the core courses?" --top_k 10
```

---

#### `qa_generator.py`
**Purpose**: Question-answering with GPT-4 and grounding in retrieved content

**Description**:
GPT-4 powered question-answering system with strict grounding in retrieved content. Ensures answers are based only on provided context to prevent hallucinations.

**Key Features**:
- Strict grounding in retrieved context
- System prompts enforce citation and accuracy
- Token counting and management (< 8k context limit)
- Optional streaming output for real-time responses
- Context truncation with safety buffers
- Markdown formatting for clean context presentation
- Separate prompts for static and live data scenarios
- Integration with confidence router

**Main Classes**:
- `QAGenerator` - Main QA generation class

**Key Methods**:
- `generate_answer()` - Generate answer from context
- `format_context()` - Format retrieved results as context
- `truncate_context()` - Truncate context to fit token limits
- `count_tokens()` - Count tokens using tiktoken
- `answer_with_retrieval()` - Full pipeline with retrieval

**Dependencies**: `config.py`, `confidence_router.py` (optional)

**Used By**: `app.py`, `advanced_rag.py`, `fallback_guardrail.py`

**Usage**:
```bash
python qa_generator.py --question "What are the admission requirements?" --model gpt-4o-mini
```

---

#### `hybrid_retrieval.py`
**Purpose**: Hybrid retrieval combining semantic and lexical search

**Description**:
Combines semantic search (via ChromaDB with fine-tuned embeddings) and lexical search (via BM25) for robust retrieval. Uses weighted fusion or reciprocal rank fusion to merge results.

**Key Features**:
- Semantic search using sentence transformers and ChromaDB
- Lexical search using BM25 for keyword matching
- Query synonym expansion for better keyword coverage
- Weighted score fusion with overlap boost
- Reciprocal Rank Fusion (RRF) option
- Score normalization (min-max)
- Deduplication of results
- Fine-tuned embedding model support

**Main Classes**:
- `HybridRetriever` - Main hybrid retrieval class

**Key Methods**:
- `semantic_search()` - Semantic search via ChromaDB
- `lexical_search()` - Lexical search via BM25
- `expand_query_with_synonyms()` - Expand query with synonyms
- `weighted_fusion()` - Weighted score fusion
- `reciprocal_rank_fusion()` - RRF-based fusion
- `retrieve()` - Complete hybrid retrieval pipeline

**Dependencies**: ChromaDB, sentence-transformers, rank-bm25

**Used By**: `advanced_rag.py`, `confidence_router.py`

**Configuration**:
- Default: 80% semantic, 20% lexical
- Supports custom weights and RRF mode

**Usage**:
```bash
python hybrid_retrieval.py --question "Tell me about the program" --top_k 5
```

---

#### `fallback_guardrail.py`
**Purpose**: Automatic fallback to live data when knowledge gaps detected

**Description**:
Intelligent guardrail system that detects when the LLM indicates it lacks information and automatically triggers live web scraping to fetch current data from the official website.

**Key Features**:
- Pattern-based uncertainty detection (regex patterns)
- Automatic Firecrawl activation when knowledge gaps detected
- AI-powered URL selection for targeted scraping
- Keyword-based URL relevance filtering
- Live data context formatting with special prompts
- Regenerates answers with fresh scraped content
- Configurable fallback URLs
- Logging and monitoring support

**Main Classes**:
- `FallbackGuardrail` - Main guardrail class

**Key Methods**:
- `detect_uncertainty()` - Detect uncertainty patterns in answers
- `fetch_live_data()` - Fetch live data via Firecrawl
- `get_relevant_urls()` - Select relevant URLs based on query
- `format_dynamic_context()` - Format live data as context
- `answer_with_fallback()` - Complete guardrail pipeline

**Uncertainty Patterns Detected**:
- "I don't have that information"
- "Not available in my knowledge base"
- "Cannot find that information"
- "Sorry, I'm unable to..."

**Dependencies**: `dynamic_retrieval.py`, `qa_generator.py`, `intelligent_search_agent.py`

**Used By**: `app.py`

**Usage**:
```bash
python fallback_guardrail.py --question "What's the new application deadline?"
```

---

#### `dynamic_retrieval.py`
**Purpose**: On-demand web scraping with Firecrawl integration

**Description**:
Dynamic content retrieval system using Firecrawl API for on-demand web scraping. Supports JavaScript rendering, tab interactions, and caching for efficiency.

**Key Features**:
- Firecrawl integration for reliable web scraping
- JavaScript rendering support (waits for dynamic content)
- Tab and accordion interaction capabilities
- Content caching with MD5 hashing
- Markdown and HTML extraction
- Text chunking for RAG processing
- Optional OpenAI embedding integration
- LangChain tool wrapper support

**Main Classes**:
- `DynamicRetriever` - Main dynamic retrieval class

**Key Methods**:
- `scrape_url()` - Scrape single URL
- `scrape_with_tabs()` - Scrape with tab/accordion interactions
- `chunk_content()` - Chunk content for RAG
- `embed_text()` - Embed text with OpenAI (optional)
- `dynamic_retrieve()` - Retrieve and process multiple URLs
- `create_langchain_tool()` - Create LangChain tool wrapper

**Dependencies**: `firecrawl-py`, OpenAI (optional), LangChain (optional), `config.py`

**Used By**: `fallback_guardrail.py`, `batch_scrape_programs.py`, `extract_nested_tabs.py`

**Usage**:
```bash
python dynamic_retrieval.py --urls https://datascience.uchicago.edu/... --query "program info"
```

---

#### `intelligent_search_agent.py`
**Purpose**: AI-powered URL selection for targeted web scraping

**Description**:
AI-powered agent that intelligently selects the most relevant URLs to scrape based on the user's query. Uses GPT to extract keywords and match them against available URLs.

**Key Features**:
- GPT-powered keyword extraction from queries
- Intelligent URL ranking based on keyword matches
- Configurable URL pool for different program pages
- Returns top-N most relevant URLs for scraping
- Supports custom URL lists

**Main Classes**:
- `IntelligentSearchAgent` - AI-powered URL selection agent

**Key Methods**:
- `extract_keywords()` - Extract keywords from query using GPT
- `rank_urls()` - Rank URLs by relevance to keywords
- `intelligent_search()` - Complete intelligent URL selection pipeline

**Dependencies**: OpenAI, `config.py`

**Used By**: `fallback_guardrail.py`

**Usage**:
```bash
python intelligent_search_agent.py --query "What are the core courses?"
```

---

#### `confidence_router.py`
**Purpose**: Confidence-based routing between static and dynamic retrieval

**Description**:
Routes queries between static (ChromaDB) and dynamic (Firecrawl) retrieval based on confidence scores and relevance metrics. Ensures optimal data source selection.

**Key Features**:
- Confidence scoring for retrieval results
- Automatic routing between static and dynamic sources
- Threshold-based decision making
- Fallback URL management
- Integration with hybrid retrieval

**Main Classes**:
- `ConfidenceRouter` - Confidence-based routing system

**Key Methods**:
- `calculate_confidence()` - Calculate confidence scores
- `route_query()` - Route query to appropriate source
- `should_use_dynamic()` - Determine if dynamic retrieval needed

**Dependencies**: `hybrid_retrieval.py`, `dynamic_retrieval.py`

**Used By**: `qa_generator.py`

---

### Data Collection & Scraping

#### `webscrape_to_json.py`
**Purpose**: Initial web scraping of UChicago MS-ADS website

**Description**:
Basic web scraper using BeautifulSoup to recursively crawl the MS-ADS program website. Extracts structured content while preserving heading hierarchy and deduplicating repeated elements.

**Key Features**:
- Recursive crawling with depth limits (MAX_DEPTH=2)
- Deduplication of repeated text (nav, header, footer removal)
- Markdown-style heading preservation (# ## ###)
- Internal link following within datascience.uchicago.edu domain
- Hierarchical JSON output with parent-child relationships
- Metadata tracking (URLs, depth, timestamps)

**Output**: `uchicago_msads_content_rag.json`

**Dependencies**: requests, BeautifulSoup4

**Used By**: `load_to_chromadb.py`

**Usage**:
```bash
python webscrape_to_json.py
```

---

#### `batch_scrape_programs.py`
**Purpose**: Batch scraping of multiple program pages

**Description**:
Batch scraper using Firecrawl to scrape multiple program-related pages in parallel. More robust than basic BeautifulSoup scraping with JavaScript rendering support.

**Key Features**:
- Batch scraping of predefined URL lists
- Firecrawl API integration for reliability
- JavaScript rendering for dynamic content
- Progress tracking and error handling
- JSON output for each URL
- Configurable wait times and formats

**Dependencies**: `dynamic_retrieval.py`, `config.py`

**Output**: Multiple JSON files (e.g., `course-progressions_program.json`, `in_person_program.json`)

**Usage**:
```bash
python batch_scrape_programs.py
```

---

#### `extract_nested_tabs.py`
**Purpose**: Extract content from nested tabs and accordions

**Description**:
Specialized scraper for extracting content hidden in tabs and accordion elements using Firecrawl's action capabilities. Essential for capturing curriculum and course details.

**Key Features**:
- Tab/accordion interaction simulation
- Click actions on tab selectors
- Scroll actions for lazy-loaded content
- JavaScript execution for dynamic interactions
- Wait times between actions for content loading
- Comprehensive content extraction after all interactions

**Main Focus**: Course progression pages with tabbed curricula

**Dependencies**: `dynamic_retrieval.py`

**Output**: Extracted content from hidden tab sections

**Usage**:
```bash
python extract_nested_tabs.py
```

---

#### `scrape_booth_mba_ms.py`
**Purpose**: Scrape specific Booth MBA/MS program pages

**Description**:
Specialized scraper for University of Chicago Booth School MBA and MS-ADS comparison pages. Extracts program comparisons and cross-references.

**Key Features**:
- Targeted scraping of Booth-specific pages
- Program comparison data extraction
- Cross-program reference handling

**Dependencies**: `dynamic_retrieval.py`

**Output**: `chicago_booth_mba_ms_ads.json`

**Usage**:
```bash
python scrape_booth_mba_ms.py
```

---

#### `webscrape_improved.py`
**Purpose**: Enhanced web scraping with improved content extraction

**Description**:
Improved version of the basic web scraper with better content extraction, error handling, and support for more complex page structures.

**Key Features**:
- Enhanced content extraction algorithms
- Better handling of edge cases
- Improved error recovery
- More robust parsing logic

**Dependencies**: requests, BeautifulSoup4

**Used By**: Testing and development

---

### Vector Database Management

#### `load_to_chromadb.py` (deprecated)
**Purpose**: Load scraped content into ChromaDB with fine-tuned embeddings

**Description**:
Data loader that processes scraped JSON files and loads content into ChromaDB vector database using fine-tuned sentence transformer embeddings for optimal domain-specific retrieval.

**Key Features**:
- Loads from hierarchical JSON (output of `webscrape_to_json.py`)
- Flattens nested document structures
- Text chunking with overlap (512 chars, 50 char overlap)
- Sentence boundary-aware chunking
- Fine-tuned embedding model integration (`../finetune-embedding/exp_finetune`)
- Batch insertion for efficiency
- Metadata preservation (URLs, titles, depth, parent URLs)
- Persistent ChromaDB storage

**Main Classes**:
- `UCMSADSChromaLoader` - Main loader class

**Key Methods**:
- `create_collection()` - Create/reset ChromaDB collection
- `chunk_text()` - Chunk text with overlap
- `flatten_documents()` - Flatten nested JSON structure
- `prepare_data_for_insertion()` - Prepare chunked data
- `insert_data()` - Batch insert into ChromaDB
- `load_from_json()` - Complete pipeline

**Output**: ChromaDB collection at `./chroma_db_finetuned`

**Dependencies**: ChromaDB, sentence-transformers

**Usage**:
```bash
python load_to_chromadb.py
```

---

#### `load_to_chromadb_openai.py`
**Purpose**: Load content into ChromaDB with fine-tuned sentence transformer embeddings

**Description**:
**Note: Despite the "openai" in the filename, this file uses fine-tuned sentence transformer embeddings, NOT OpenAI embeddings.** This is the primary loader for the production system using domain-specific fine-tuned embeddings from `../finetune-embedding/exp_finetune`.

**Key Features**:
- Fine-tuned sentence transformer model integration
- Enhanced metadata structure with categories
- Text chunking with overlap (512 chars, 50 char overlap)
- Sentence boundary-aware chunking
- Batch processing for efficient embedding generation
- Progress tracking and error handling
- Persistent ChromaDB storage

**Main Classes**:
- `FinetunedChromaLoader` - Loader using fine-tuned embeddings

**Key Methods**:
- `create_collection()` - Create/reset collection
- `get_embeddings()` - Generate embeddings using fine-tuned model
- `chunk_text()` - Chunk with overlap
- `flatten_documents()` - Flatten nested JSON
- `prepare_data_for_insertion()` - Prepare chunked data
- `insert_data()` - Batch insert with embeddings
- `load_from_json()` - Complete pipeline

**Output**: ChromaDB collection at `./chroma_db_finetuned`

**Dependencies**: ChromaDB, sentence-transformers, fine-tuned model

**Usage**:
```bash
python load_to_chromadb_openai.py
```

---

#### `retrieve_from_chromadb.py` (deprecated)
**Purpose**: Retrieve content from ChromaDB - deprecated version

**Description**:
Basic retrieval script (deprecated). Use `retrieve_from_chromadb_openai.py` instead.

**Status**: Deprecated - maintained for backward compatibility

---

#### `retrieve_from_chromadb_openai.py`
**Purpose**: Retrieve content from ChromaDB using fine-tuned sentence transformer embeddings

**Description**:
**Note: Despite the "openai" in the filename, this file uses fine-tuned sentence transformer embeddings, NOT OpenAI embeddings.** This is the primary retrieval script for querying the production ChromaDB with domain-specific embeddings.

**Key Features**:
- Query embedding with fine-tuned sentence transformer model
- Similarity search using cosine distance
- Category-based filtering (curriculum, admissions, faculty, etc.)
- Interactive query mode for testing
- Pretty formatted output with metadata
- Returns structured results for LLM integration
- Model path resolution and validation

**Main Classes**:
- `FinetunedRetriever` - Retriever using fine-tuned embeddings

**Key Methods**:
- `embed_query()` - Embed query with fine-tuned model
- `retrieve()` - Perform similarity search
- `pretty_print()` - Format and display results

**Dependencies**: ChromaDB, sentence-transformers, fine-tuned model

**Used By**: Testing, development, and standalone queries

**Usage**:
```bash
# Interactive mode
python retrieve_from_chromadb_openai.py

# One-shot query
python retrieve_from_chromadb_openai.py --question "Tell me about the program" --top_k 5

# With category filter
python retrieve_from_chromadb_openai.py --question "What are the core courses?" --category curriculum
```

---

#### `load_inperson_to_vectorstore.py`
**Purpose**: Load in-person program content to vector store

**Description**:
Specialized loader for in-person program-specific content. Ensures comprehensive coverage of in-person program details that may differ from online program.

**Key Features**:
- Loads in-person program JSON files
- Same chunking and embedding strategy
- Separate or merged collection support
- Handles in-person specific metadata

**Input**: `in_person_program.json`

**Dependencies**: ChromaDB, sentence-transformers

**Usage**:
```bash
python load_inperson_to_vectorstore.py
```

---

### Evaluation & Monitoring

#### `evaluation.py`
**Purpose**: Comprehensive evaluation framework for RAG system

**Description**:
Evaluation framework for assessing RAG system performance using metrics like retrieval accuracy (Recall@K, MRR, NDCG), answer quality (ROUGE, semantic similarity), and end-to-end pipeline evaluation.

**Key Features**:
- Retrieval evaluation metrics (Recall@K, MRR, NDCG)
- Answer quality evaluation (ROUGE-L, semantic similarity)
- End-to-end pipeline evaluation
- Support for validation and test datasets (train.jsonl, val.jsonl)
- JSON output for evaluation results with timestamps
- Comparison between different RAG methods (basic vs advanced)
- Batch evaluation for efficiency
- Detailed per-query analysis

**Main Classes**:
- `RAGEvaluator` - Main evaluation class

**Key Methods**:
- `evaluate_retrieval()` - Evaluate retrieval performance
- `evaluate_answers()` - Evaluate answer quality
- `evaluate_pipeline()` - End-to-end evaluation
- `save_results()` - Save evaluation results to JSON

**Evaluation Metrics & Formulas**:

The evaluation framework implements comprehensive metrics for both retrieval and generation components, following industry standards from Confident AI, DeepEval, Evidently AI, LangChain, and Pinecone.

**📊 RETRIEVAL COMPONENT METRICS:**

1. **Contextual Relevancy** - How relevant the retrieved documents are to the user query
   ```
   Query_Words = words_in_query - stop_words
   Doc_Words = words_in_doc - stop_words
   Relevancy_per_doc = |Query_Words ∩ Doc_Words| / |Query_Words|
   Contextual_Relevancy = Σ(Relevancy_per_doc) / num_docs
   ```
   - Measures: Word overlap between query and retrieved documents
   - Range: [0, 1], higher is better
   - Reference: Confident AI, Evidently AI

2. **Contextual Recall** - How much necessary information is included in retrieved set
   ```
   Expected_Words = words_in_expected_answer - stop_words
   Retrieved_Words = words_in_all_retrieved_docs - stop_words
   Contextual_Recall = |Expected_Words ∩ Retrieved_Words| / |Expected_Words|
   ```
   - Measures: Coverage of expected answer information in retrieved docs
   - Range: [0, 1], higher is better
   - Reference: DeepEval, Evidently AI

3. **Contextual Precision** - How many retrieved items are actually useful (vs noise)
   ```
   Answer_Words = words_in_generated_answer - stop_words
   Useful_Docs = count of docs with overlap with Answer_Words
   Contextual_Precision = Useful_Docs / Total_Retrieved_Docs
   ```
   - Measures: Proportion of retrieved docs that contributed to answer
   - Range: [0, 1], higher is better
   - Reference: DeepEval

4. **Mean Reciprocal Rank (MRR)** - Average of reciprocal ranks of first relevant document
   ```
   MRR = 1 / rank_of_first_relevant_doc
   ```
   where relevance is defined as >10% word overlap with expected answer
   - Measures: How quickly users find relevant results
   - Range: [0, 1], higher is better
   - Reference: Pinecone

5. **Precision@K** - Proportion of retrieved documents that are relevant
   ```
   Precision@K = |Relevant_Docs_in_top_K| / K
   ```
   - Measures: Accuracy of top-K retrieved results
   - Range: [0, 1], higher is better
   - Reference: Pinecone

6. **Recall@K** - Proportion of relevant documents retrieved in top-K results
   ```
   Recall@K = |Relevant_Docs_in_top_K| / |Total_Relevant_Docs|
   ```
   - Measures: Coverage of relevant documents in top-K
   - Range: [0, 1], higher is better
   - Reference: Pinecone

**✨ GENERATION COMPONENT METRICS:**

7. **Answer Relevance** - How well the output answers the user's question
   ```
   Query_Words = words_in_query - stop_words
   Answer_Words = words_in_answer - stop_words
   Answer_Relevance = |Query_Words ∩ Answer_Words| / |Query_Words|
   ```
   - Measures: Whether answer directly addresses the query
   - Range: [0, 1], higher is better
   - Reference: LangChain

8. **Faithfulness (Groundedness)** - Does the model stick to retrieved context (no hallucination)?
   ```
   Answer_Words = words_in_answer - stop_words
   Context_Words = words_in_all_retrieved_docs - stop_words
   Faithfulness = |Answer_Words ∩ Context_Words| / |Answer_Words|
   ```
   - Measures: Proportion of answer grounded in retrieved context
   - High score (>0.7) means low hallucination risk
   - Range: [0, 1], higher is better
   - Reference: Confident AI, arXiv:2305.14251

9. **Correctness/Accuracy** - How correct is the output against ground truth?
   ```
   Generated_Words = words_in_generated_answer - stop_words
   Expected_Words = words_in_expected_answer - stop_words
   Correctness = |Generated_Words ∩ Expected_Words| / |Expected_Words|
   Accuracy = |Generated_Words ∩ Expected_Words| / |Generated_Words|
   ```
   - Correctness: Ratio of correct words to total expected
   - Accuracy: Ratio of correct words to total generated
   - Range: [0, 1], higher is better
   - Reference: Evidently AI

10. **Clarity & Coherence** - Answer quality through readability heuristics
    ```
    Complete_Sentences = sentences with ≥3 words
    Clarity_Score = Complete_Sentences / Total_Sentences
    Transitions_Used = count of transition words (however, therefore, etc.)
    Coherence_Score = Transitions_Used / Total_Sentences
    ```
    - Measures: Sentence structure and logical flow
    - Range: [0, 1], higher is better

11. **F1 Score** - Harmonic mean of precision and recall for word overlap
    ```
    Precision = |Correct_Words| / |Generated_Words|
    Recall = |Correct_Words| / |Expected_Words|
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    ```
    - Measures: Balance between precision and recall
    - Range: [0, 1], higher is better

**📈 AGGREGATE METRICS:**
- Success Rate: Proportion of successful queries
- Average Latency: Mean query processing time
- Average Token Usage: Mean tokens per query
- Method-specific comparisons (Basic vs Advanced RAG)

**Note**: All word-based metrics exclude common stop words ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by')

**Output**: `evaluation_results/` directory with timestamped JSON files

**Dependencies**: `advanced_rag.py`, `qa_generator.py`, `hybrid_retrieval.py`

**Usage**:
```bash
python evaluation.py --dataset val.jsonl --method advanced --output evaluation_results/
```

---

#### `monitoring.py`
**Purpose**: Logging, cost tracking, and performance monitoring

**Description**:
Comprehensive monitoring system that tracks all aspects of RAG system operation including queries, API costs, performance metrics, and automated alerting for anomalies.

**Key Features**:
- Query and response logging (JSONL format in `logs/queries.jsonl`)
- API cost tracking for OpenAI and Firecrawl (`logs/costs.jsonl`)
- Performance metrics (latency, success rate, token usage, confidence scores)
- Automated alerting system for errors and anomalies (`logs/alerts.jsonl`)
- Dashboard data aggregation for visualization
- Daily and monthly cost reports
- P95 latency tracking
- Routing distribution analysis
- Success rate monitoring

**Main Classes**:
- `MonitoringSystem` - Main monitoring orchestrator
- `QueryLogger` - Query/response logging
- `CostTracker` - API cost tracking with pricing models
- `MetricsTracker` - Performance metrics calculation
- `AlertSystem` - Automated alerting for issues

**Key Methods**:
- `log_query_response()` - Log complete query-response cycle
- `get_dashboard_data()` - Get aggregated monitoring dashboard data
- `print_dashboard()` - Print dashboard to console
- `log_openai_cost()` - Track OpenAI API costs
- `log_firecrawl_cost()` - Track Firecrawl API costs

**Pricing Tracked**:
- GPT-4o-mini: $0.15/$0.60 per 1M tokens (input/output)
- GPT-4o: $2.50/$10.00 per 1M tokens (input/output)
- Firecrawl: $0.002 per page

**Output**: `logs/` directory with queries.jsonl, costs.jsonl, alerts.jsonl

**Used By**: `app.py` for production monitoring

**Usage**:
```bash
python monitoring.py  # View current dashboard
```

---

### Testing Files

#### `test_fallback.py`
**Purpose**: Test fallback guardrail functionality

**Description**:
Test script for validating fallback guardrail system including uncertainty detection patterns, Firecrawl integration, and answer regeneration with live data.

**Key Features**:
- Tests all uncertainty detection patterns (10+ regex patterns)
- Validates Firecrawl integration and scraping
- Tests AI-powered and keyword-based URL selection logic
- Verifies answer regeneration with live data
- End-to-end fallback pipeline testing
- Mock testing for unit tests
- Integration testing with real API calls

**Test Coverage**:
- Uncertainty pattern detection
- Dynamic retrieval triggering
- URL ranking and selection
- Live data formatting
- Answer comparison (before/after fallback)

**Dependencies**: `fallback_guardrail.py`, `dynamic_retrieval.py`

**Usage**:
```bash
python test_fallback.py
```

---

#### `test_program_retrieval.py`
**Purpose**: Test program-specific retrieval quality

**Description**:
Test script for validating retrieval quality on common program-specific query types (admissions, curriculum, costs, deadlines, faculty, etc.) to ensure comprehensive coverage.

**Key Features**:
- Tests common query categories (admissions, curriculum, tuition, etc.)
- Validates retrieval relevance and accuracy
- Tests curriculum query handling and course-specific re-ranking
- Verifies metadata preservation and URL tracking
- Interactive testing mode for manual validation
- Automated test suite for CI/CD
- Performance benchmarking

**Test Query Categories**:
- Admissions requirements
- Curriculum and courses
- Tuition and financial aid
- Program format (online/in-person)
- Faculty and instructors
- Application deadlines

**Dependencies**: `hybrid_retrieval.py`, `advanced_rag.py`

**Usage**:
```bash
python test_program_retrieval.py
```

---

#### `test_tab_scraping.py`
**Purpose**: Test tab/accordion scraping functionality

**Description**:
Test script for validating tab and accordion interaction capabilities during web scraping with Firecrawl actions to ensure all hidden content is captured.

**Key Features**:
- Tests Firecrawl action sequences (click, scroll, wait)
- Validates tab interaction and accordion expansion
- Tests wait times and timing between actions
- Verifies content extraction completeness after interactions
- Debug mode for troubleshooting scraping issues
- Visual verification helpers
- Action replay capabilities

**Test Cases**:
- Course progression tabs
- FAQ accordions
- Program comparison tabs
- Dynamic content loading

**Dependencies**: `dynamic_retrieval.py`, `extract_nested_tabs.py`

**Usage**:
```bash
python test_tab_scraping.py --debug  # With debug output
```

---

### Utility Files

#### `sync_dynamic_to_static.py`
**Purpose**: Synchronize dynamically scraped content to static vector store

**Description**:
Utility for synchronizing content from dynamic cache (Firecrawl results stored in `dynamic_cache/`) back into the static ChromaDB vector store for future offline use and improved performance.

**Key Features**:
- Reads from `dynamic_cache/` directory (cached Firecrawl results)
- Processes cached JSON files with Markdown content
- Chunks and embeds new content using fine-tuned model
- Updates ChromaDB collection with new documents
- Deduplication by URL to avoid duplicates
- Metadata synchronization and preservation
- Batch processing for efficiency
- Progress tracking and logging

**Workflow**:
1. Scan dynamic_cache directory for new cached pages
2. Extract and chunk content
3. Generate embeddings with fine-tuned model
4. Insert into ChromaDB with metadata
5. Mark synced files to avoid re-processing

**Dependencies**: `dynamic_retrieval.py`, `load_to_chromadb_openai.py`, ChromaDB

**Usage**:
```bash
python sync_dynamic_to_static.py --cache-dir dynamic_cache --update
```

---

#### `gunicorn_config.py`
**Purpose**: Production WSGI server configuration for deployment

**Description**:
Gunicorn WSGI server configuration optimized for the RAG system's resource requirements and performance characteristics for production deployment on AWS/GCP.

**Key Configuration**:
- **Workers**: 4 (balanced for I/O and CPU)
- **Worker class**: sync (standard synchronous workers)
- **Timeout**: 120 seconds (accommodates long RAG queries with fallback)
- **Bind**: 0.0.0.0:5000 (accessible from all interfaces for cloud deployment)
- **Access logging**: Enabled for request tracking
- **Error logging**: Enabled with detailed error messages
- **Keepalive**: 5 seconds
- **Max requests**: 1000 (worker recycling for memory management)

**Production Features**:
- Graceful worker restarts
- Request timeout handling for long-running queries
- Memory leak prevention via worker recycling
- Logging integration with system logs

**Used With**: `app.py` Flask application

**Usage**:
```bash
gunicorn -c gunicorn_config.py app:app
```

---

#### `finetune_sft.py` (depricated)
**Purpose**: Supervised fine-tuning (SFT) utilities and configuration

**Description**:
Configuration and utilities for supervised fine-tuning of language models on program-specific QA pairs. Prepares datasets and manages fine-tuning workflows.

**Key Features**:
- JSONL dataset preparation and validation
- Training/validation split management
- Fine-tuning hyperparameter configuration
- Model export and versioning utilities
- Integration with OpenAI fine-tuning API
- Training progress monitoring
- Evaluation metrics tracking

**Dataset Format** (JSONL):
```json
{"messages": [
  {"role": "system", "content": "You are a UChicago MS-ADS assistant..."},
  {"role": "user", "content": "What are the core courses?"},
  {"role": "assistant", "content": "The core courses include..."}
]}
```

**Input Files**: 
- `train.jsonl` - Training dataset
- `val.jsonl` - Validation dataset

**Dependencies**: OpenAI API, validation utilities

**Usage**:
```bash
python finetune_sft.py --train train.jsonl --val val.jsonl --model gpt-4o-mini
```

---

## File Relationships

(To be updated with detailed relationship diagram...)

---

## Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
python app.py
```

For production deployment, see [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md)

---

*Last Updated: November 12, 2025*

**README Status**: Complete with comprehensive descriptions of all 26 Python files including:
- Full file purposes and descriptions
- Key features and functionalities
- Dependencies and file relationships
- Usage examples and CLI commands
- Complete evaluation metrics with mathematical formulas
- References to industry standards (Confident AI, DeepEval, Evidently AI, LangChain, Pinecone)
