# Embedding Fine-Tuning Guide

Complete guide for fine-tuning embedding models on your web-scraped data.

## 📋 Overview

This folder contains tools for:
1. **Generating synthetic datasets** from web-scraped content
2. **Fine-tuning embedding models** on the synthetic data
3. **Evaluating and comparing** model performance

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (for dataset generation)
export OPENAI_API_KEY='your-key-here'
```

### Complete Workflow

```bash
# Step 1: Generate synthetic dataset (~10-15 minutes)
python generate_dataset_webscrape.py

# Step 2: Fine-tune embedding model (~5-10 minutes)
python finetune.py

# Step 3: Evaluate and compare models (~5 minutes)
python evaluate.py
```

---

## 📝 Detailed Instructions

### Step 1: Generate Synthetic Dataset

```bash
python generate_dataset_webscrape.py
```

**What it does:**
- Loads your web-scraped data from `../gen-ai-midterm/uchicago_msads_content_rag.json`
- Chunks text into manageable pieces (1000 chars with 200 overlap)
- Uses GPT-3.5 to generate 2 questions per chunk
- Splits into train (80%) and validation (20%) sets
- Saves 8 JSON files to `data/` directory

**Output:**
```
data/
├── train_corpus.json           # Training text chunks
├── train_queries.json          # Training questions
├── train_relevant_docs.json    # Query-to-doc mappings
├── train_dataset.json          # Complete training set
├── val_corpus.json            # Validation text chunks
├── val_queries.json           # Validation questions
├── val_relevant_docs.json     # Query-to-doc mappings
└── val_dataset.json           # Complete validation set
```

**Configuration:**
Edit constants in `generate_dataset_webscrape.py`:
- `CHUNK_SIZE = 1000` - Characters per chunk
- `CHUNK_OVERLAP = 200` - Overlap between chunks
- `NUM_QUESTIONS_PER_CHUNK = 2` - Questions per chunk
- `TRAIN_SPLIT_RATIO = 0.8` - Train/val split

**Estimated:**
- Time: 10-15 minutes
- Cost: $2-4 USD
- Output: ~2,856 synthetic questions

---

### Step 2: Fine-tune Embedding Model

```bash
# Basic usage
python finetune.py

# With custom parameters
python finetune.py \
    --model_id BAAI/bge-small-en \
    --epochs 5 \
    --batch_size 32 \
    --output_path exp_finetune
```

**What it does:**
- Loads pre-trained `BAAI/bge-small-en` model
- Creates (query, relevant_doc) training pairs
- Fine-tunes using MultipleNegativesRankingLoss
- Evaluates on validation set every 50 steps
- Saves fine-tuned model to `exp_finetune/`

**Arguments:**
```
--model_id           Base model (default: BAAI/bge-small-en)
--train_dataset      Training data (default: ./data/train_dataset.json)
--val_dataset        Validation data (default: ./data/val_dataset.json)
--epochs             Training epochs (default: 2, try 5-10 for better results)
--batch_size         Batch size (default: 10, try 32-64 if you have GPU)
--output_path        Save directory (default: exp_finetune)
--evaluation_steps   Eval frequency (default: 50)
```

**Output:**
```
exp_finetune/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── vocab.txt
└── ... (model files)
```

**Estimated:**
- Time: 5-10 minutes (CPU), 2-3 minutes (GPU)
- Memory: ~2GB RAM
- Disk: ~500MB

**Training Tips:**
- **More epochs = better results** (try 5-10)
- **Larger batch size = better loss** (32-64 if GPU available)
- Monitor validation metrics during training
- Model auto-saves the best checkpoint

---

### Step 3: Evaluate Models

```bash
# Basic usage
python evaluate.py

# Custom parameters
python evaluate.py \
    --base_model BAAI/bge-small-en \
    --finetuned_model exp_finetune \
    --top_k 5
```

**What it does:**
- Evaluates base model (BAAI/bge-small-en)
- Evaluates fine-tuned model
- Compares performance with multiple metrics:
  - **Hit Rate**: % queries where relevant doc is in top-k
  - **Accuracy@k**: Accuracy at different k values
  - **Precision, Recall, F1**: Retrieval metrics
  - **NDCG, MRR**: Ranking metrics
- Saves detailed results to `results/`

**Arguments:**
```
--base_model        Base model ID (default: BAAI/bge-small-en)
--finetuned_model   Fine-tuned model path (default: exp_finetune)
--val_dataset       Validation data (default: ./data/val_dataset.json)
--top_k            Top-k for hit rate (default: 5)
--results_dir      Results directory (default: results)
```

**Output:**
```
results/
├── Information-Retrieval_evaluation_base_results.csv
└── Information-Retrieval_evaluation_finetuned_results.csv
```

**Expected Results:**
```
Hit Rate Comparison:
           Model  Hit Rate  Hits  Total
      Base Model    0.6500   372    572
Fine-tuned Model    0.8500   486    572

Improvement: +30.77%
```

**Metrics Explained:**
- **Hit Rate**: Simple metric - is relevant doc in top-k?
- **Accuracy@k**: Precision at different cutoffs (1, 3, 5, 10)
- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality)
- **MRR**: Mean Reciprocal Rank (average position of correct answer)

---

## 🔧 Advanced Configuration

### Custom Dataset Path

Edit `generate_dataset_webscrape.py`:
```python
WEBSCRAPE_DATA_PATH = '/path/to/your/data.json'
```

### Different Base Model

Try other models from HuggingFace:
```bash
# Smaller, faster
python finetune.py --model_id sentence-transformers/all-MiniLM-L6-v2

# Larger, more accurate
python finetune.py --model_id BAAI/bge-base-en
```

### Production Training Settings

For better results:
```bash
python finetune.py \
    --epochs 10 \
    --batch_size 64 \
    --evaluation_steps 25
```

---

## 📊 Expected Performance

Based on typical results:

| Model | Hit Rate@5 | Improvement |
|-------|------------|-------------|
| Base (BAAI/bge-small-en) | ~65% | - |
| Fine-tuned | ~85% | +30% |

Fine-tuning typically shows:
- ✅ **20-30% improvement** in retrieval accuracy
- ✅ **Better ranking** of relevant documents
- ✅ **Domain-specific understanding** of UChicago MSADS content
- ✅ **Approaching proprietary models** like OpenAI embeddings

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-...'
```

### "Dataset not found"
Run dataset generation first:
```bash
python generate_dataset_webscrape.py
```

### "Out of memory during training"
Reduce batch size:
```bash
python finetune.py --batch_size 4
```

### "Training is slow"
- Use GPU if available (auto-detected by PyTorch)
- Reduce epochs or batch size for faster iteration
- Try a smaller base model

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 Files Overview

### Python Scripts
- `generate_dataset_webscrape.py` - Generate synthetic datasets
- `finetune.py` - Fine-tune embedding model
- `evaluate.py` - Evaluate and compare models
- `test_data_loading.py` - Test data loading (optional)

### Jupyter Notebooks (Reference)
- `generate_dataset.ipynb` - Original notebook version
- `finetune.ipynb` - Reference fine-tuning notebook
- `evaluate.ipynb` - Reference evaluation notebook

### Documentation
- `README_WEBSCRAPE.md` - Dataset generation details
- `FINETUNING_GUIDE.md` - This file

---

## 🎯 Integration with RAG System

After fine-tuning, use your model in the main RAG system:

### Option 1: Update ChromaDB
```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('./finetune-embedding/exp_finetune')

# Use with ChromaDB
import chromadb
from chromadb.utils import embedding_functions

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='./finetune-embedding/exp_finetune'
)
```

### Option 2: Update Your Retriever
```python
# In your RAG code
embedding_model = SentenceTransformer('finetune-embedding/exp_finetune')
query_embedding = embedding_model.encode(query)
```

---

## 💡 Tips for Best Results

1. **More training data = better model**
   - Generate more questions per chunk
   - Use larger web-scraped dataset

2. **Tune hyperparameters**
   - Try epochs: 5, 10, 15
   - Try batch_size: 16, 32, 64

3. **Monitor validation metrics**
   - Watch for overfitting
   - Use best checkpoint

4. **Experiment with base models**
   - Different models have different strengths
   - Smaller = faster, Larger = more accurate

5. **Quality over quantity**
   - Better synthetic questions → better fine-tuning
   - Review generated questions occasionally

---

## 📖 References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html)
- [BAAI/bge-small-en Model](https://huggingface.co/BAAI/bge-small-en)

---

## ✅ Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Set OpenAI API key
- [ ] Generate synthetic dataset (`generate_dataset_webscrape.py`)
- [ ] Fine-tune model (`finetune.py`)
- [ ] Evaluate models (`evaluate.py`)
- [ ] Review results in `results/` directory
- [ ] Integrate fine-tuned model into RAG system

---

**Questions?** Check the individual script docstrings or run with `--help`:
```bash
python finetune.py --help
python evaluate.py --help
```
