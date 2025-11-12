**Checklist: Hybrid RAG System for UChicago MS-ADS Webpage**

---

### 1. Environment & Tool Setup
- [x] Create Python virtual environment (Python 3.10+)
  - created, environment called genaiEnv
- [ ] Install required libraries:
  - `langchain`
  - `chromadb`
  - `openai`
  - `firecrawl`
  - `streamlit` or `flask`
  - `beautifulsoup4`, `requests`, `scikit-learn`, `schedule` (for optional components)
- [ ] Set API keys: `OPENAI_API_KEY`, `FIRECRAWL_API_KEY`
- [ ] Initialize LangChain and LangSmith for observability

---

### 📂 2. Static Website Crawling & Chunking
- [ ] Check current framework and see if any improvements is needed
- [x] Use Firecrawl in `crawl` mode on MS-ADS homepage
- [x] Clean extracted content (remove nav/footers, extract sections)
- [x] Structure metadata: `source_url`, `section_title`, `crawl_date`, `category`
- [x] Chunk using LangChain RecursiveCharacterTextSplitter (~300-500 tokens with overlap)

---

### 3. Embedding & Vector Store (OpenAI Model)
- [ ] Use OpenAI `text-embedding-ada-002` for all chunks
- [ ] Batch embed all content
- [x] Initialize persistent ChromaDB vector store
- [ ] We already have ChromaDB vector store, however, it currently uses sentence transformer, instead of openAI
- [ ] Store documents with metadata and persist the DB

---

### 4. Hybrid Retrieval (Dense + Lexical)
- [ ] Implement semantic search using Chroma (cosine similarity)
- [ ] Implement lexical search (BM25 via `rank_bm25`)
- [ ] Normalize scores and apply weighted fusion or RRF
- [ ] Merge top-k semantic + top-m lexical results with overlap boost
- [ ] Sort final ranked documents and include in retrieval response

---

### 5. FireCrawl Integration (Dynamic Retrieval)
- [ ] Add Firecrawl tool (LangChain tool or direct API)
- [ ] Prepare scrape mode & optional search-based dynamic discovery
- [ ] Construct scraping prompts from query
- [ ] Fetch clean Markdown and metadata
- [ ] On-demand embedding with OpenAI (optional cache)
- [ ] Dynamically insert into current prompt context
- [ ] Add Firecrawl MCP server tool for fast structured search:
- [ ] Install MCP server: npx -y firecrawl-mcp
- [ ] Configure .env or runtime with FIRECRAWL_API_KEY
- [ ] Launch and wrap in LangChain Tool using query → MCP API call

---

### 6. Confidence Scoring & Routing
- [ ] Score embedding similarity (max cosine from Chroma)
- [ ] Score lexical match (TF/IDF term overlap)
- [ ] Score content freshness from `crawl_date`
- [ ] Normalize and combine (e.g., 0.6*semantic + 0.3*lexical + 0.1*recency)
- [ ] If score < threshold (e.g. 0.5), trigger Firecrawl

---

### 7. GPT-4.1 QA Generation
- [ ] Use system prompt instructing grounding in retrieved content only
- [ ] Format context: clean, Markdown bullet list or paragraph + question
- [ ] Customize our `LLMChain`
- [ ] Limit total prompt length (< 8k tokens)
- [ ] Optionally stream token output

---

### 8. Frontend (Streamlit or Flask)
- [ ] Build interface: text input, answer area
- [ ] Show source metadata (page, section)
- [ ] Call full RAG pipeline from UI
- [ ] Display streamed or full answer
- [ ] Handle errors gracefully, add 2 to 3 retires as fallback

---
### 9. Logging, Evaluation & Monitoring
- [ ] Log query, top docs, scores, route taken, GPT answer
- [ ] Track OpenAI and FireCrawl usage/cost
- [ ] Build evaluation set (QA pairs) for test runs
- [ ] Track metrics: retrieval precision, latency, confidence correlation
- [ ] Configure alerting on failure or downtime

