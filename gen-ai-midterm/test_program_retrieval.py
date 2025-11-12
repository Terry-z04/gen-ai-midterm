#!/usr/bin/env python3
"""
Test Program Retrieval using Finetune Training Data
Evaluates retrieval performance with metrics like Recall@K, MRR@K, NDCG@K

Based on the evaluation shown in your image with metrics like:
- recall@10: 1.0
- mrr@10: 0.8742947003683324
- ndcg@10: 0.8742947003683666
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path

# Import retrieval systems
try:
    from retrieve_from_chromadb_openai import FinetunedRetriever
    RETRIEVER_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Finetuned retrieval not available: {e}")
    RETRIEVER_AVAILABLE = False


def load_evaluation_data(data_dir: str = "../finetune-embedding/data"):
    """Load queries, corpus, and relevant docs from finetune data"""
    data_path = Path(data_dir)
    
    # Load train data
    with open(data_path / "train_queries.json", 'r') as f:
        train_queries = json.load(f)
    
    with open(data_path / "train_relevant_docs.json", 'r') as f:
        train_relevant = json.load(f)
    
    with open(data_path / "train_corpus.json", 'r') as f:
        train_corpus = json.load(f)
    
    # Load validation data
    with open(data_path / "val_queries.json", 'r') as f:
        val_queries = json.load(f)
    
    with open(data_path / "val_relevant_docs.json", 'r') as f:
        val_relevant = json.load(f)
    
    with open(data_path / "val_corpus.json", 'r') as f:
        val_corpus = json.load(f)
    
    return {
        "train": {
            "queries": train_queries,
            "relevant": train_relevant,
            "corpus": train_corpus
        },
        "val": {
            "queries": val_queries,
            "relevant": val_relevant,
            "corpus": val_corpus
        }
    }


def build_gold_from_instance(instance: dict) -> set:
    """Build gold standard set from instance"""
    gold = set()
    for e in instance.get("evidence", []):
        doc_id = to_doc_id(e.get("source", ""))
        page = e.get("page", None)
        if page is not None:
            gold.add((doc_id, int(page)))
    return gold


def to_doc_id(source: str) -> str:
    """Extract document ID from source URL"""
    # For the finetune data, the doc IDs are URLs with #chunk_X
    # We want to extract just the base URL part
    if "#chunk_" in source:
        return source.split("#chunk_")[0]
    return source


def extract_doc_id_from_metadata(doc_metadata: dict) -> str:
    """Extract doc ID from retrieved document metadata"""
    source = doc_metadata.get("source", "")
    return to_doc_id(source)


def is_hit(doc_id: str, gold_pairs: set) -> bool:
    """Check if retrieved document is relevant"""
    # For the training data format, gold_pairs contains full URLs with #chunk_X
    # We need to check if any gold doc matches this doc_id
    for gold_doc in gold_pairs:
        if isinstance(gold_doc, str):
            # Remove chunk part for comparison
            gold_base = to_doc_id(gold_doc)
            if doc_id == gold_base or gold_doc == doc_id:
                return True
        elif isinstance(gold_doc, tuple):
            # (doc_id, page) format
            if doc_id == gold_doc[0]:
                return True
    return False


def rrf_fuse(retrieval_lists: List[List], k: int = 60) -> List:
    """Reciprocal Rank Fusion"""
    fused = defaultdict(float)
    for retrieval_list in retrieval_lists:
        for i, r in enumerate(retrieval_list, start=1):
            if r == i:
                mrr = 1.0 / i
                break
    
    # Calculate RRF scores
    for retrieval_list in retrieval_lists:
        for i, doc in enumerate(retrieval_list, start=1):
            fused[doc] += 1.0 / (k + i)
    
    # Sort by score
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def dcg(relevances: List[int]) -> float:
    """Discounted Cumulative Gain"""
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def calculate_metrics(instances: List[dict], k: int = 10) -> Dict:
    """
    Calculate retrieval metrics:
    - Recall@K: Proportion of relevant docs retrieved in top K
    - MRR@K: Mean Reciprocal Rank (position of first relevant doc)
    - NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
    - Precision@K: Proportion of retrieved docs that are relevant
    """
    recall_list = []
    mrr_list = []
    ndcg_list = []
    precision_list = []
    
    for inst in instances:
        gold_pairs = inst.get("gold_pairs", set())
        if not gold_pairs:
            continue
        
        retrieved_docs = inst.get("retrieved_docs", [])
        
        # Calculate hits for each position
        hits = [1 if is_hit(d, gold_pairs) else 0 for d in retrieved_docs[:k]]
        
        # Recall@K: How many relevant docs were retrieved
        recall = sum(hits) / len(gold_pairs) if gold_pairs else 0.0
        recall_list.append(recall)
        
        # MRR@K: Position of first relevant doc
        mrr = 0.0
        for i, hit in enumerate(hits, 1):
            if hit:
                mrr = 1.0 / i
                break
        mrr_list.append(mrr)
        
        # NDCG@K: Normalized DCG
        dcg_k = dcg(hits)
        
        # Ideal DCG (all relevant docs at top)
        ideal_hits = sorted(hits, reverse=True)
        idcg_k = dcg(ideal_hits)
        
        ndcg = dcg_k / idcg_k if idcg_k > 0 else 0.0
        ndcg_list.append(ndcg)
        
        # Precision@K
        precision = sum(hits) / k if k > 0 else 0.0
        precision_list.append(precision)
    
    return {
        f"recall@{k}": np.mean(recall_list) if recall_list else 0.0,
        f"mrr@{k}": np.mean(mrr_list) if mrr_list else 0.0,
        f"ndcg@{k}": np.mean(ndcg_list) if ndcg_list else 0.0,
        f"precision@{k}": np.mean(precision_list) if precision_list else 0.0,
        "n": len(recall_list)
    }


def eval_instance(
    query: str,
    gold_docs: List[str],
    retriever: 'FinetunedRetriever',
    k: int = 10
) -> Dict:
    """Evaluate a single query instance"""
    # Convert gold docs to set
    gold_pairs = set(gold_docs)
    
    # Retrieve documents
    try:
        if not retriever:
            return {
                "query": query,
                "gold_pairs": gold_pairs,
                "retrieved_docs": [],
                "num_gold": len(gold_pairs),
                "num_retrieved": 0,
                "error": "No retrieval system available"
            }
        
        # Use retriever to get results
        results = retriever.retrieve(query, top_k=k, max_chars=0)  # Suppress printing
        
        # Extract document IDs from retrieved results
        retrieved_docs = []
        for result_item in results.get("results", []):
            if isinstance(result_item, dict):
                doc_id = result_item.get("source", "")
                retrieved_docs.append(doc_id)
        
        return {
            "query": query,
            "gold_pairs": gold_pairs,
            "retrieved_docs": retrieved_docs,
            "num_gold": len(gold_pairs),
            "num_retrieved": len(retrieved_docs)
        }
    except Exception as e:
        return {
            "query": query,
            "gold_pairs": gold_pairs,
            "retrieved_docs": [],
            "num_gold": len(gold_pairs),
            "num_retrieved": 0,
            "error": str(e)
        }


def evaluate_part1(
    test_questions: Dict[str, str],
    relevant_docs: Dict[str, List[str]],
    k: int = 10
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate retrieval performance on a set of queries
    
    Args:
        test_questions: Dict mapping query_id -> query text
        relevant_docs: Dict mapping query_id -> list of relevant doc IDs
        k: Top K documents to retrieve
    
    Returns:
        Tuple of (individual results, aggregated metrics)
    """
    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION")
    print(f"System: Finetuned Embeddings")
    print(f"Queries: {len(test_questions)}")
    print(f"K: {k}")
    print(f"{'='*60}\n")
    
    # Initialize retriever
    if not RETRIEVER_AVAILABLE:
        print("⚠ Retriever not available!")
        return [], {"n": 0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0, f"precision@{k}": 0.0}
    
    try:
        retriever = FinetunedRetriever(
            db_path="chroma_db_finetuned",
            collection_name="uchicago_msads_finetuned",
            embedding_model_path="../finetune-embedding/exp_finetune"
        )
    except Exception as e:
        print(f"⚠ Failed to initialize retriever: {e}")
        return [], {"n": 0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0, f"precision@{k}": 0.0}
    
    rows = []
    
    for idx, (query_id, query_text) in enumerate(test_questions.items(), 1):
        if query_id not in relevant_docs:
            print(f"⚠ No gold docs for query: {query_id}")
            continue
        
        gold_docs = relevant_docs[query_id]
        
        print(f"[{idx}/{len(test_questions)}] Query: {query_text[:60]}...")
        
        # Evaluate this instance
        result = eval_instance(
            query=query_text,
            gold_docs=gold_docs,
            retriever=retriever,
            k=k
        )
        
        rows.append(result)
        
        # Show progress
        if "error" not in result:
            hits = sum(1 for doc in result["retrieved_docs"][:k] 
                      if is_hit(doc, result["gold_pairs"]))
            print(f"  ✓ Retrieved {hits}/{result['num_gold']} relevant docs\n")
    
    # Calculate aggregate metrics
    agg = calculate_metrics(rows, k=k)
    
    return rows, agg


def print_results(part1_rows, part1_agg, k=10):
    """Print evaluation results in a format similar to the image"""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Total Queries Evaluated: {part1_agg['n']}")
    print(f"\nMetrics @ K={k}:")
    print(f"  recall@{k}:    {part1_agg[f'recall@{k}']:.4f}")
    print(f"  mrr@{k}:       {part1_agg[f'mrr@{k}']:.16f}")
    print(f"  ndcg@{k}:      {part1_agg[f'ndcg@{k}']:.16f}")
    print(f"  precision@{k}: {part1_agg[f'precision@{k}']:.4f}")
    
    # Also display as percentages
    print(f"\nMetrics (as percentages):")
    print(f"  Recall@{k}:    {part1_agg[f'recall@{k}']*100:.1f}%")
    print(f"  MRR@{k}:       {part1_agg[f'mrr@{k}']*100:.1f}%")
    print(f"  NDCG@{k}:      {part1_agg[f'ndcg@{k}']*100:.1f}%")
    print(f"  Precision@{k}: {part1_agg[f'precision@{k}']*100:.1f}%")
    
    print(f"\n{'='*60}\n")
    
    # Create dict format similar to image output
    output_dict = {
        'n': part1_agg['n'],
        f'recall@{k}': part1_agg[f'recall@{k}'],
        f'mrr@{k}': part1_agg[f'mrr@{k}'],
        f'ndcg@{k}': part1_agg[f'ndcg@{k}'],
        f'precision@{k}': part1_agg[f'precision@{k}']
    }
    
    print("Results Dictionary:")
    print(output_dict)
    print()


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test program retrieval with finetune data")
    parser.add_argument("--dataset", default="train", choices=["train", "val"], 
                       help="Dataset to use (train or val)")
    parser.add_argument("--k", type=int, default=10, 
                       help="Top K documents to retrieve")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of queries to test")
    parser.add_argument("--openai", action="store_true", 
                       help="Use OpenAI embeddings instead of finetuned")
    args = parser.parse_args()
    
    # Load data
    print("Loading evaluation data...")
    data = load_evaluation_data()
    
    dataset = data[args.dataset]
    queries = dataset["queries"]
    relevant = dataset["relevant"]
    
    # Limit queries if specified
    if args.limit:
        query_ids = list(queries.keys())[:args.limit]
        queries = {qid: queries[qid] for qid in query_ids}
        relevant = {qid: relevant[qid] for qid in query_ids if qid in relevant}
    
    print(f"✓ Loaded {len(queries)} queries from {args.dataset} dataset")
    
    # Run evaluation
    part1_rows, part1_agg = evaluate_part1(
        test_questions=queries,
        relevant_docs=relevant,
        k=args.k
    )
    
    # Print results
    print_results(part1_rows, part1_agg, k=args.k)
    
    # Save results to file
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    system_name = "openai" if args.openai else "finetuned"
    results_file = results_dir / f"retrieval_eval_{args.dataset}_{system_name}_k{args.k}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "dataset": args.dataset,
            "system": system_name,
            "k": args.k,
            "num_queries": len(queries),
            "metrics": part1_agg,
            "individual_results": [
                {
                    "query": r["query"],
                    "num_gold": r["num_gold"],
                    "num_retrieved": r["num_retrieved"]
                }
                for r in part1_rows
            ]
        }, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
