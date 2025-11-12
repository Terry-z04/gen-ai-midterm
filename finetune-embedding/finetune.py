"""
Fine-tune Embedding Model

This script fine-tunes an open-source sentence-transformers embedding model 
on synthetically generated dataset from web-scraped content.

Usage:
    python finetune.py --model_id BAAI/bge-small-en --epochs 2 --batch_size 10
"""

import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def prepare_training_data(dataset, batch_size=10):
    """
    Prepare training data loader from dataset.
    
    Args:
        dataset: Dictionary containing corpus, queries, and relevant_docs
        batch_size: Batch size for training
        
    Returns:
        DataLoader with InputExample objects
    """
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']
    
    # Create training examples (query, relevant_document) pairs
    examples = []
    for query_id, query in queries.items():
        node_id = relevant_docs[query_id][0]
        text = corpus[node_id]
        example = InputExample(texts=[query, text])
        examples.append(example)
    
    print(f"Created {len(examples)} training examples")
    
    # Create data loader
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    return loader


def create_evaluator(dataset):
    """
    Create evaluator for monitoring training progress.
    
    Args:
        dataset: Validation dataset
        
    Returns:
        InformationRetrievalEvaluator object
    """
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']
    
    evaluator = InformationRetrievalEvaluator(
        queries, 
        corpus, 
        relevant_docs,
        name='validation'
    )
    return evaluator


def main(args):
    """Main fine-tuning function."""
    
    print("="*70)
    print("EMBEDDING MODEL FINE-TUNING")
    print("="*70)
    print()
    
    # Configuration
    print("Configuration:")
    print(f"  Model ID: {args.model_id}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Output Path: {args.output_path}")
    print()
    
    # Load datasets
    train_dataset = load_dataset(args.train_dataset)
    val_dataset = load_dataset(args.val_dataset)
    print()
    
    # Load pre-trained model
    print(f"Loading pre-trained model: {args.model_id}")
    model = SentenceTransformer(args.model_id)
    print(f"Model loaded successfully")
    print(f"  Model type: {type(model)}")
    print(f"  Max sequence length: {model.max_seq_length}")
    print()
    
    # Prepare training data
    print("Preparing training data...")
    train_loader = prepare_training_data(train_dataset, batch_size=args.batch_size)
    print(f"  Training batches: {len(train_loader)}")
    print()
    
    # Define loss function
    print("Setting up loss function...")
    # MultipleNegativesRankingLoss samples n-1 negative docs per batch
    loss = losses.MultipleNegativesRankingLoss(model)
    print("  Using: MultipleNegativesRankingLoss")
    print("  This loss works great for retrieval with positive pairs")
    print()
    
    # Create evaluator
    print("Setting up evaluator...")
    evaluator = create_evaluator(val_dataset)
    print("  Evaluator created for validation monitoring")
    print()
    
    # Calculate warmup steps (10% of total steps)
    warmup_steps = int(len(train_loader) * args.epochs * 0.1)
    print(f"Training configuration:")
    print(f"  Total steps: {len(train_loader) * args.epochs}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Evaluation frequency: every {args.evaluation_steps} steps")
    print()
    
    # Train the model
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print()
    
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_path,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
        save_best_model=True,
    )
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Fine-tuned model saved to: {args.output_path}")
    print()
    print("Next steps:")
    print("  1. Run evaluate.py to compare model performance")
    print("  2. Use the fine-tuned model in your RAG system")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model on synthetic dataset"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="BAAI/bge-small-en",
        help="HuggingFace model ID for base embedding model"
    )
    
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="./data/train_dataset.json",
        help="Path to training dataset JSON file"
    )
    
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="./data/val_dataset.json",
        help="Path to validation dataset JSON file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Training batch size (default: 10, increase for better performance)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="exp_finetune",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=50,
        help="Evaluate model every N steps"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.train_dataset).exists():
        print(f"ERROR: Training dataset not found: {args.train_dataset}")
        print("Please run generate_dataset_webscrape.py first")
        exit(1)
    
    if not Path(args.val_dataset).exists():
        print(f"ERROR: Validation dataset not found: {args.val_dataset}")
        print("Please run generate_dataset_webscrape.py first")
        exit(1)
    
    main(args)
