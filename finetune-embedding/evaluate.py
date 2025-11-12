"""
Evaluate Embedding Models

This script evaluates and compares different embedding models:
1. Base model (e.g., BAAI/bge-small-en)
2. Fine-tuned model
3. Optionally: OpenAI embeddings

Usage:
    python evaluate.py --base_model BAAI/bge-small-en --finetuned_model exp_finetune
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def evaluate_with_hit_rate(dataset, model, top_k=5, model_name="model"):
    """
    Evaluate model using simple hit rate metric.
    
    Args:
        dataset: Dictionary containing corpus, queries, and relevant_docs
        model: SentenceTransformer model
        top_k: Number of top results to retrieve
        model_name: Name for display
        
    Returns:
        List of evaluation results
    """
    print(f"\nEvaluating {model_name} with hit rate metric (top_k={top_k})...")
    
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']
    
    # Encode corpus
    print("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())
    corpus_embeddings = model.encode(corpus_texts, show_progress_bar=True)
    
    # Evaluate queries
    eval_results = []
    hits = 0
    
    print("Evaluating queries...")
    for query_id, query in tqdm(queries.items(), desc=f"Queries ({model_name})"):
        # Encode query
        query_embedding = model.encode([query])[0]
        
        # Compute similarities
        from sentence_transformers.util import cos_sim
        similarities = cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = similarities.argsort(descending=True)[:top_k]
        retrieved_ids = [corpus_ids[idx] for idx in top_k_indices.tolist()]
        
        # Check if relevant doc is in top-k
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids
        
        if is_hit:
            hits += 1
        
        eval_result = {
            'query_id': query_id,
            'query': query,
            'is_hit': is_hit,
            'retrieved': retrieved_ids,
            'expected': expected_id,
        }
        eval_results.append(eval_result)
    
    hit_rate = hits / len(queries)
    print(f"✓ Hit Rate for {model_name}: {hit_rate:.4f} ({hits}/{len(queries)})")
    
    return eval_results, hit_rate


def evaluate_with_st_evaluator(dataset, model_path, name, output_dir='results'):
    """
    Evaluate model using SentenceTransformers InformationRetrievalEvaluator.
    
    This provides comprehensive metrics: precision, recall, ndcg, mrr, etc.
    
    Args:
        dataset: Dictionary containing corpus, queries, and relevant_docs
        model_path: Path to model or HuggingFace model ID
        name: Name for the evaluation (used in output files)
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating with SentenceTransformers evaluator: {name}")
    
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']
    
    # Create evaluator
    evaluator = InformationRetrievalEvaluator(
        queries, 
        corpus, 
        relevant_docs, 
        name=name
    )
    
    # Load model
    model = SentenceTransformer(model_path)
    
    # Run evaluation
    results = evaluator(model, output_path=output_dir)
    
    print(f"✓ Evaluation complete for {name}")
    print(f"  Results saved to: {output_dir}/")
    
    return results


def compare_results(results_dict):
    """
    Compare results from different models.
    
    Args:
        results_dict: Dictionary mapping model names to (eval_results, hit_rate) tuples
    """
    print("\n" + "="*70)
    print("COMPARISON OF MODELS")
    print("="*70)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, (eval_results, hit_rate) in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Hit Rate': f"{hit_rate:.4f}",
            'Hits': sum(1 for r in eval_results if r['is_hit']),
            'Total': len(eval_results)
        })
    
    df = pd.DataFrame(comparison_data)
    print("\nHit Rate Comparison:")
    print(df.to_string(index=False))
    print()
    
    # Show improvement
    if len(comparison_data) >= 2:
        base_hit_rate = float(comparison_data[0]['Hit Rate'])
        finetuned_hit_rate = float(comparison_data[-1]['Hit Rate'])
        improvement = ((finetuned_hit_rate - base_hit_rate) / base_hit_rate) * 100
        print(f"Improvement: {improvement:+.2f}%")
        print()


def load_st_results(results_dir, model_names):
    """Load and compare SentenceTransformers evaluation results."""
    print("\n" + "="*70)
    print("SENTENCETRANSFORMERS EVALUATION METRICS")
    print("="*70)
    
    all_results = []
    
    for name in model_names:
        result_file = Path(results_dir) / f"Information-Retrieval_evaluation_{name}_results.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            df['model'] = name
            all_results.append(df)
            print(f"✓ Loaded results for {name}")
    
    if all_results:
        df_combined = pd.concat(all_results, ignore_index=True)
        df_pivot = df_combined.set_index('model')
        
        print("\nDetailed Metrics:")
        print(df_pivot.to_string())
        print()
        
        # Highlight key metrics
        print("Key Metrics Summary:")
        key_metrics = ['cos_sim-Accuracy@1', 'cos_sim-Accuracy@3', 'cos_sim-Accuracy@5', 'cos_sim-Accuracy@10']
        available_metrics = [m for m in key_metrics if m in df_pivot.columns]
        if available_metrics:
            print(df_pivot[available_metrics].to_string())
        
        return df_pivot
    
    return None


def main(args):
    """Main evaluation function."""
    
    print("="*70)
    print("EMBEDDING MODEL EVALUATION")
    print("="*70)
    print()
    
    # Configuration
    print("Configuration:")
    print(f"  Base Model: {args.base_model}")
    print(f"  Fine-tuned Model: {args.finetuned_model}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Results Directory: {args.results_dir}")
    print()
    
    # Create results directory
    Path(args.results_dir).mkdir(exist_ok=True)
    
    # Load validation dataset
    val_dataset = load_dataset(args.val_dataset)
    print(f"Loaded validation dataset:")
    print(f"  Queries: {len(val_dataset['queries'])}")
    print(f"  Corpus: {len(val_dataset['corpus'])}")
    print()
    
    # Evaluate base model
    print("="*70)
    print("EVALUATING BASE MODEL")
    print("="*70)
    
    print(f"Loading base model: {args.base_model}")
    base_model = SentenceTransformer(args.base_model)
    
    base_results, base_hit_rate = evaluate_with_hit_rate(
        val_dataset, 
        base_model, 
        top_k=args.top_k,
        model_name="Base Model"
    )
    
    # Evaluate with ST evaluator
    evaluate_with_st_evaluator(
        val_dataset,
        args.base_model,
        name='base',
        output_dir=args.results_dir
    )
    
    # Evaluate fine-tuned model
    print("\n" + "="*70)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*70)
    
    if Path(args.finetuned_model).exists():
        print(f"Loading fine-tuned model: {args.finetuned_model}")
        finetuned_model = SentenceTransformer(args.finetuned_model)
        
        finetuned_results, finetuned_hit_rate = evaluate_with_hit_rate(
            val_dataset,
            finetuned_model,
            top_k=args.top_k,
            model_name="Fine-tuned Model"
        )
        
        # Evaluate with ST evaluator
        evaluate_with_st_evaluator(
            val_dataset,
            args.finetuned_model,
            name='finetuned',
            output_dir=args.results_dir
        )
        
        # Compare results
        compare_results({
            'Base Model': (base_results, base_hit_rate),
            'Fine-tuned Model': (finetuned_results, finetuned_hit_rate),
        })
        
        # Load and display ST results
        load_st_results(args.results_dir, ['base', 'finetuned'])
        
    else:
        print(f"ERROR: Fine-tuned model not found at: {args.finetuned_model}")
        print("Please run finetune.py first")
        return
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.results_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare embedding models"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="BAAI/bge-small-en",
        help="Base model ID (HuggingFace)"
    )
    
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="exp_finetune",
        help="Path to fine-tuned model directory"
    )
    
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="./data/val_dataset.json",
        help="Path to validation dataset"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results for hit rate calculation"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Validate dataset exists
    if not Path(args.val_dataset).exists():
        print(f"ERROR: Validation dataset not found: {args.val_dataset}")
        print("Please run generate_dataset_webscrape.py first")
        exit(1)
    
    main(args)
