# Synthetic Dataset Generation from Web-Scraped Data

This guide explains how to generate a synthetic training dataset for embedding fine-tuning using your web-scraped UChicago MS-ADS content.

## Overview

The `generate_dataset_webscrape.py` script:
1. Loads your hierarchical web-scraped JSON data
2. Chunks the content into manageable pieces
3. Generates synthetic queries using GPT-3.5-turbo
4. Creates training and validation datasets for embedding fine-tuning

## Prerequisites

### 1. Install Dependencies

```bash
cd finetune-embedding
pip install -r requirements.txt
```

The required packages are:
- `llama-index==0.8.5.post2` (for LLM integration)
- `sentence-transformers==2.2.2` (for embedding models)
- `tqdm` (for progress bars)

### 2. Set Up OpenAI API Key

The script uses GPT-3.5-turbo to generate synthetic queries. Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file in the `finetune-embedding` directory:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Verify Web-Scraped Data

Ensure your web-scraped data exists at:
```
../gen-ai-midterm/uchicago_msads_content_rag.json
```

The JSON should have this structure:
```json
{
  "title": "Page Title",
  "url": "https://...",
  "content": "Page content...",
  "subsections": [...]
}
```

## Usage

### Option 1: Generate Full Dataset (Recommended)

Run the complete pipeline to generate synthetic queries:

```bash
python generate_dataset_webscrape.py
```

This will:
- Load and chunk your web-scraped content
- Split into 80% training / 20% validation
- Generate 2 questions per chunk using GPT-3.5
- Save all datasets to the `data/` directory

**Expected output files:**
- `data/train_corpus.json` - Training document chunks
- `data/train_queries.json` - Generated training questions
- `data/train_relevant_docs.json` - Question-to-document mappings
- `data/train_dataset.json` - Complete training dataset
- `data/val_corpus.json` - Validation document chunks
- `data/val_queries.json` - Generated validation questions
- `data/val_relevant_docs.json` - Question-to-document mappings (val)
- `data/val_dataset.json` - Complete validation dataset

### Option 2: Test Data Loading First

Before running the full pipeline (which makes API calls), test that data loading works:

```bash
python test_data_loading.py
```

This validates:
- Web-scraped JSON file exists and is readable
- Text extraction from hierarchical structure works
- Chunking algorithm functions correctly
- No API calls are made (free to run)

## Configuration

Edit the configuration variables in `generate_dataset_webscrape.py`:

```python
# Path to your web-scraped JSON file
WEBSCRAPE_DATA_PATH = '../gen-ai-midterm/uchicago_msads_content_rag.json'

# Chunking parameters
CHUNK_SIZE = 1000  # Maximum characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Dataset split
TRAIN_SPLIT_RATIO = 0.8  # 80% train, 20% validation

# Query generation
NUM_QUESTIONS_PER_CHUNK = 2  # Questions per chunk
```

## Cost Estimation

The script uses OpenAI's GPT-3.5-turbo API:
- **Model**: gpt-3.5-turbo
- **Cost**: ~$0.0005 per 1K tokens (input) + ~$0.0015 per 1K tokens (output)
- **Typical cost**: If you have 500 chunks, generating 2 questions per chunk costs approximately $1-3

To reduce costs:
- Decrease `NUM_QUESTIONS_PER_CHUNK`
- Increase `CHUNK_SIZE` to create fewer chunks
- Process a subset of your data first

## Output Format

The generated datasets follow this structure:

```json
{
  "queries": {
    "query_id_1": "What are the admission requirements?",
    "query_id_2": "How long is the program?"
  },
  "corpus": {
    "doc_id_1": "Title: Admissions\n\nContent: The program requires...",
    "doc_id_2": "Title: Program Duration\n\nContent: Students complete..."
  },
  "relevant_docs": {
    "query_id_1": ["doc_id_1"],
    "query_id_2": ["doc_id_2"]
  }
}
```

## Next Steps

After generating the synthetic dataset:

1. **Fine-tune embeddings**: Use `finetune.ipynb` to train a custom embedding model
2. **Evaluate performance**: Use `evaluate.ipynb` to compare base vs fine-tuned models
3. **Integration**: Use the fine-tuned model in your RAG system

## Troubleshooting

### ModuleNotFoundError: No module named 'llama_index'

Install the requirements:
```bash
pip install -r requirements.txt
```

### OpenAI API Key Error

Set your API key:
```bash
export OPENAI_API_KEY='your-key'
```

### File Not Found Error

Verify the path to your web-scraped JSON:
```bash
ls -la ../gen-ai-midterm/uchicago_msads_content_rag.json
```

### Low Quality Questions

Adjust the prompt template in the `generate_queries()` function to better suit your domain.

### Too Expensive

- Reduce `NUM_QUESTIONS_PER_CHUNK` from 2 to 1
- Increase `CHUNK_SIZE` to create fewer chunks
- Test on a small subset first

## Comparison with Original

The original `generate_dataset.ipynb` was designed for PDF documents. This version:

✅ Works with hierarchical web-scraped JSON  
✅ Preserves page titles and metadata  
✅ Handles nested subsections recursively  
✅ Generates domain-specific questions (UChicago MS-ADS)  
✅ Available as both `.py` script and `.ipynb` notebook  

## Files

- `generate_dataset_webscrape.py` - Main script (Python)
- `generate_dataset_webscrape.ipynb` - Notebook version (Jupyter)
- `test_data_loading.py` - Test script (no API calls)
- `README_WEBSCRAPE.md` - This file

## Example Run

```bash
$ python generate_dataset_webscrape.py

======================================================================
SYNTHETIC DATASET GENERATION FROM WEB-SCRAPED DATA
======================================================================

Step 1: Loading and processing web-scraped data...
Loading data from ../gen-ai-midterm/uchicago_msads_content_rag.json
Extracting text chunks from pages...
Extracted 145 pages
Created 487 text chunks

Step 2: Splitting into train and validation sets...
Train corpus: 389 chunks
Val corpus: 98 chunks

Step 3: Saving corpus files...
Corpus files saved!

Step 4: Generating training queries...
Generating queries: 100%|████████████| 389/389 [05:23<00:00,  1.20it/s]
Generated 756 training queries

Step 5: Generating validation queries...
Generating queries: 100%|████████████| 98/98 [01:21<00:00,  1.21it/s]
Generated 189 validation queries

...

✅ DATASET GENERATION COMPLETE!
Training dataset: 756 queries, 389 documents
Validation dataset: 189 queries, 98 documents
```

## Support

For issues or questions, refer to:
- Original fine-tuning tutorial: `finetune.ipynb`
- LlamaIndex docs: https://docs.llamaindex.ai/
- Sentence Transformers: https://www.sbert.net/
