#!/usr/bin/env python3
"""
Evaluation Framework for RAG System
Group 5 | UChicago MS-ADS RAG System

Uses real QA pairs from train.jsonl and val.jsonl for evaluation
"""

import json
import time
from typing import Dict, List, Optional
from pathlib import Path

# Import RAG components
try:
    from advanced_rag import AdvancedRAG
    from qa_generator import QAGenerator
    SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠ RAG system not available")
    SYSTEM_AVAILABLE = False


def load_qa_pairs_from_jsonl(file_path: str) -> List[Dict]:
    """Load QA pairs from JSONL file"""
    qa_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            qa_pairs.append({
                "question": entry["input"],
                "expected_answer": entry["output"],
                "instruction": entry["instruction"]
            })
    
    return qa_pairs


class EvaluationFramework:
    """Framework for evaluating RAG system performance"""
    
    def __init__(
        self,
        train_file: str = "train.jsonl",
        val_file: str = "val.jsonl",
        results_dir: str = "./evaluation_results"
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load QA pairs from files
        self.train_pairs = self._load_qa_pairs(train_file)
        self.val_pairs = self._load_qa_pairs(val_file)
        
        print(f"✓ Loaded {len(self.train_pairs)} training QA pairs")
        print(f"✓ Loaded {len(self.val_pairs)} validation QA pairs")
        
        if SYSTEM_AVAILABLE:
            self.rag = AdvancedRAG()
            self.qa = QAGenerator(model="gpt-4o-mini")
        else:
            self.rag = None
            self.qa = None
    
    def _load_qa_pairs(self, file_path: str) -> List[Dict]:
        """Load QA pairs from JSONL file"""
        if not Path(file_path).exists():
            print(f"⚠ File not found: {file_path}")
            return []
        
        return load_qa_pairs_from_jsonl(file_path)
    
    def calculate_answer_similarity(
        self,
        generated_answer: str,
        expected_answer: str
    ) -> Dict:
        """
        Calculate similarity between generated and expected answers
        
        Simple overlap-based similarity:
        - Word overlap
        - Key phrase presence
        """
        gen_words = set(generated_answer.lower().split())
        exp_words = set(expected_answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        gen_words -= stop_words
        exp_words -= stop_words
        
        if not exp_words:
            return {"word_overlap": 0.0, "precision": 0.0, "recall": 0.0}
        
        overlap = gen_words & exp_words
        precision = len(overlap) / len(gen_words) if gen_words else 0
        recall = len(overlap) / len(exp_words) if exp_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "word_overlap": len(overlap),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    # ========== RETRIEVAL COMPONENT METRICS ==========
    
    def calculate_contextual_relevancy(
        self,
        retrieved_docs: List[Dict],
        query: str
    ) -> Dict:
        """
        Contextual Relevancy: How relevant the retrieved documents/chunks are to the user query.
        
        References:
        - Confident AI: https://docs.confident-ai.com
        - Evidently AI: https://docs.evidentlyai.com
        
        Measures word overlap and semantic similarity between query and retrieved documents.
        """
        if not retrieved_docs:
            return {"relevancy_score": 0.0, "avg_overlap": 0.0}
        
        query_words = set(query.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        
        relevancy_scores = []
        for doc in retrieved_docs:
            doc_text = doc.get('text', doc.get('content', ''))
            doc_words = set(doc_text.lower().split())
            doc_words -= stop_words
            
            if doc_words:
                overlap = len(query_words & doc_words)
                relevancy = overlap / len(query_words) if query_words else 0
                relevancy_scores.append(relevancy)
        
        return {
            "relevancy_score": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            "avg_overlap": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            "num_relevant_docs": sum(1 for score in relevancy_scores if score > 0.1)
        }
    
    def calculate_contextual_recall(
        self,
        retrieved_docs: List[Dict],
        expected_answer: str
    ) -> Dict:
        """
        Contextual Recall: How much of the necessary information is actually included in the retrieved set.
        
        References:
        - DeepEval: https://docs.confident-ai.com/docs/metrics-contextual-recall
        - Evidently AI: https://docs.evidentlyai.com
        
        Measures whether the retrieved documents contain the information needed to answer the question.
        """
        if not retrieved_docs:
            return {"recall_score": 0.0, "coverage": 0.0}
        
        expected_words = set(expected_answer.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        expected_words -= stop_words
        
        # Combine all retrieved document texts
        all_retrieved_text = ' '.join([
            doc.get('text', doc.get('content', ''))
            for doc in retrieved_docs
        ])
        retrieved_words = set(all_retrieved_text.lower().split())
        retrieved_words -= stop_words
        
        # Calculate how many expected answer words are in retrieved docs
        covered_words = expected_words & retrieved_words
        recall_score = len(covered_words) / len(expected_words) if expected_words else 0
        
        return {
            "recall_score": recall_score,
            "coverage": len(covered_words) / len(expected_words) if expected_words else 0,
            "covered_words": len(covered_words),
            "total_expected_words": len(expected_words)
        }
    
    def calculate_contextual_precision(
        self,
        retrieved_docs: List[Dict],
        generated_answer: str
    ) -> Dict:
        """
        Contextual Precision: How many of the retrieved items are actually useful (vs irrelevant or noise).
        
        References:
        - DeepEval: https://docs.confident-ai.com/docs/metrics-contextual-precision
        
        Measures whether the retrieved documents were actually used in generating the answer.
        """
        if not retrieved_docs:
            return {"precision_score": 0.0, "useful_docs": 0}
        
        answer_words = set(generated_answer.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        answer_words -= stop_words
        
        useful_docs = 0
        for doc in retrieved_docs:
            doc_text = doc.get('text', doc.get('content', ''))
            doc_words = set(doc_text.lower().split())
            doc_words -= stop_words
            
            # Check if this document contributed to the answer
            overlap = len(answer_words & doc_words)
            if overlap > 0:
                useful_docs += 1
        
        precision_score = useful_docs / len(retrieved_docs)
        
        return {
            "precision_score": precision_score,
            "useful_docs": useful_docs,
            "total_docs": len(retrieved_docs),
            "noise_docs": len(retrieved_docs) - useful_docs
        }
    
    def calculate_ranking_metrics(
        self,
        retrieved_docs: List[Dict],
        expected_answer: str,
        k_values: List[int] = None
    ) -> Dict:
        """
        Ranking metrics: Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
        
        References:
        - Pinecone: https://www.pinecone.io/learn/offline-evaluation/
        
        Reflects not just what you retrieved but the order.
        """
        if k_values is None:
            k_values = [1, 3, 5]
        
        if not retrieved_docs:
            return {
                "mrr": 0.0,
                **{f"precision_at_{k}": 0.0 for k in k_values},
                **{f"recall_at_{k}": 0.0 for k in k_values}
            }
        
        expected_words = set(expected_answer.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        expected_words -= stop_words
        
        # Calculate relevance for each document
        doc_relevances = []
        first_relevant_rank = None
        
        for idx, doc in enumerate(retrieved_docs, 1):
            doc_text = doc.get('text', doc.get('content', ''))
            doc_words = set(doc_text.lower().split())
            doc_words -= stop_words
            
            overlap = len(expected_words & doc_words)
            is_relevant = overlap > len(expected_words) * 0.1  # At least 10% overlap
            doc_relevances.append(is_relevant)
            
            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = idx
        
        # Calculate MRR
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        # Calculate Precision@K and Recall@K
        results = {"mrr": mrr}
        total_relevant = sum(doc_relevances)
        
        for k in k_values:
            relevant_at_k = sum(doc_relevances[:k])
            results[f"precision_at_{k}"] = relevant_at_k / k if k <= len(retrieved_docs) else 0.0
            results[f"recall_at_{k}"] = relevant_at_k / total_relevant if total_relevant > 0 else 0.0
        
        return results
    
    # ========== GENERATION COMPONENT METRICS ==========
    
    def calculate_answer_relevance(
        self,
        generated_answer: str,
        query: str
    ) -> Dict:
        """
        Answer Relevance: How well does the output answer the user's question (given the context/retrieval)?
        
        References:
        - LangChain Docs: https://python.langchain.com/docs/guides/evaluation/
        
        Measures whether the answer directly addresses the query.
        """
        query_words = set(query.lower().split())
        answer_words = set(generated_answer.lower().split())
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        answer_words -= stop_words
        
        # Calculate word overlap
        overlap = query_words & answer_words
        relevance_score = len(overlap) / len(query_words) if query_words else 0
        
        # Check for question words addressed
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which'}
        query_question_words = query_words & question_words
        
        return {
            "relevance_score": relevance_score,
            "word_overlap": len(overlap),
            "addresses_question": len(query_question_words) > 0,
            "answer_length": len(generated_answer.split())
        }
    
    def calculate_faithfulness(
        self,
        generated_answer: str,
        retrieved_docs: List[Dict]
    ) -> Dict:
        """
        Faithfulness (Groundedness): Does the model stick to the retrieved context 
        (does not hallucinate or add unsupported facts)?
        
        References:
        - Confident AI: https://docs.confident-ai.com/docs/metrics-faithfulness
        - arXiv: https://arxiv.org/abs/2305.14251
        
        Ensures the model uses the context appropriately without hallucination.
        """
        if not retrieved_docs:
            return {"faithfulness_score": 0.0, "grounded": False}
        
        answer_words = set(generated_answer.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        answer_words -= stop_words
        
        # Combine all retrieved document texts
        all_context_text = ' '.join([
            doc.get('text', doc.get('content', ''))
            for doc in retrieved_docs
        ])
        context_words = set(all_context_text.lower().split())
        context_words -= stop_words
        
        # Calculate how many answer words are grounded in context
        grounded_words = answer_words & context_words
        faithfulness_score = len(grounded_words) / len(answer_words) if answer_words else 0
        
        # High faithfulness means most of the answer comes from context
        is_grounded = faithfulness_score > 0.7
        
        return {
            "faithfulness_score": faithfulness_score,
            "grounded": is_grounded,
            "grounded_words": len(grounded_words),
            "total_answer_words": len(answer_words),
            "unsupported_words": len(answer_words) - len(grounded_words)
        }
    
    def calculate_correctness(
        self,
        generated_answer: str,
        expected_answer: str
    ) -> Dict:
        """
        Correctness / Accuracy: If you have a ground-truth answer, how correct is the output?
        
        References:
        - Evidently AI: https://docs.evidentlyai.com
        
        Measures accuracy against ground truth when available.
        """
        # This is similar to answer similarity but focuses on correctness
        gen_words = set(generated_answer.lower().split())
        exp_words = set(expected_answer.lower().split())
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        gen_words -= stop_words
        exp_words -= stop_words
        
        if not exp_words:
            return {"correctness_score": 0.0, "accuracy": 0.0}
        
        # Calculate overlap
        correct_words = gen_words & exp_words
        incorrect_words = gen_words - exp_words
        
        # Correctness: ratio of correct to total expected
        correctness_score = len(correct_words) / len(exp_words)
        
        # Accuracy: ratio of correct to total generated
        accuracy = len(correct_words) / len(gen_words) if gen_words else 0
        
        return {
            "correctness_score": correctness_score,
            "accuracy": accuracy,
            "correct_words": len(correct_words),
            "incorrect_words": len(incorrect_words),
            "missing_words": len(exp_words - gen_words)
        }
    
    def calculate_clarity_coherence(
        self,
        generated_answer: str
    ) -> Dict:
        """
        Clarity, coherence, relevance to prompt (less strictly measurable but important)
        
        Basic heuristics for answer quality:
        - Sentence structure
        - Length appropriateness
        - Readability indicators
        """
        sentences = [s.strip() for s in generated_answer.split('.') if s.strip()]
        words = generated_answer.split()
        
        # Calculate basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Check for complete sentences (should end with punctuation)
        complete_sentences = sum(1 for s in sentences if len(s.split()) >= 3)
        
        # Coherence: presence of transition words
        transition_words = {'however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                           'consequently', 'meanwhile', 'thus', 'hence', 'also', 'first', 
                           'second', 'finally', 'in addition', 'for example'}
        answer_lower = generated_answer.lower()
        transitions_used = sum(1 for word in transition_words if word in answer_lower)
        
        return {
            "clarity_score": min(1.0, complete_sentences / len(sentences)) if sentences else 0,
            "coherence_score": min(1.0, transitions_used / max(1, len(sentences))),
            "avg_sentence_length": avg_sentence_length,
            "num_sentences": len(sentences),
            "num_words": len(words),
            "transitions_used": transitions_used
        }
    
    def run_single_test(
        self,
        qa_pair: Dict,
        method: str = "advanced"
    ) -> Dict:
        """
        Run a single test case
        
        Args:
            qa_pair: Dictionary with question and expected_answer
            method: Retrieval method to use
            
        Returns:
            Dict with test results
        """
        if not SYSTEM_AVAILABLE:
            return {"error": "RAG system not available"}
        
        question = qa_pair["question"]
        expected_answer = qa_pair["expected_answer"]
        
        print(f"\n[TEST] {question[:80]}...")
        
        # Start timer
        start_time = time.time()
        
        try:
            # Run retrieval and QA
            if method == "advanced":
                response = self.rag.answer_with_advanced_rag(
                    question=question,
                    top_k=8,
                    model="gpt-4o-mini"
                )
            else:
                response = self.qa.answer_with_retrieval(question=question)
            
            latency = time.time() - start_time
            
            # Extract results
            answer = response.get('answer', '')
            retrieved_docs = response.get('retrieval', {}).get('documents', [])
            
            # Calculate similarity metrics
            similarity = self.calculate_answer_similarity(answer, expected_answer)
            
            # Calculate RETRIEVAL metrics
            contextual_relevancy = self.calculate_contextual_relevancy(retrieved_docs, question)
            contextual_recall = self.calculate_contextual_recall(retrieved_docs, expected_answer)
            contextual_precision = self.calculate_contextual_precision(retrieved_docs, answer)
            ranking_metrics = self.calculate_ranking_metrics(retrieved_docs, expected_answer)
            
            # Calculate GENERATION metrics
            answer_relevance = self.calculate_answer_relevance(answer, question)
            faithfulness = self.calculate_faithfulness(answer, retrieved_docs)
            correctness = self.calculate_correctness(answer, expected_answer)
            clarity_coherence = self.calculate_clarity_coherence(answer)
            
            result = {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "method": method,
                "similarity_metrics": similarity,
                "retrieval_metrics": {
                    "contextual_relevancy": contextual_relevancy,
                    "contextual_recall": contextual_recall,
                    "contextual_precision": contextual_precision,
                    "ranking_metrics": ranking_metrics
                },
                "generation_metrics": {
                    "answer_relevance": answer_relevance,
                    "faithfulness": faithfulness,
                    "correctness": correctness,
                    "clarity_coherence": clarity_coherence
                },
                "latency_seconds": latency,
                "tokens": response.get('tokens', {}),
                "retrieval_method": response.get('retrieval', {}).get('method', method),
                "num_retrieved_docs": len(retrieved_docs),
                "success": True,
                "error": None
            }
            
            print(f"✓ F1 Score: {similarity['f1_score']:.2%}")
            print(f"✓ Faithfulness: {faithfulness['faithfulness_score']:.2%}")
            print(f"✓ Contextual Relevancy: {contextual_relevancy['relevancy_score']:.2%}")
            print(f"✓ Latency: {latency:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return {
                "question": question,
                "expected_answer": expected_answer,
                "method": method,
                "success": False,
                "error": str(e),
                "latency_seconds": time.time() - start_time
            }
    
    def run_evaluation(
        self,
        dataset: str = "val",
        method: str = "advanced",
        save_results: bool = True
    ) -> Dict:
        """
        Run evaluation on dataset
        
        Args:
            dataset: "train" or "val"
            method: Retrieval method to use
            save_results: Whether to save results to file
            
        Returns:
            Dict with aggregate results
        """
        test_pairs = self.train_pairs if dataset == "train" else self.val_pairs
        
        if not test_pairs:
            print(f"⚠ No test pairs found for dataset: {dataset}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"RUNNING EVALUATION")
        print(f"Dataset: {dataset}")
        print(f"Method: {method}")
        print(f"Test cases: {len(test_pairs)}")
        print(f"{'='*60}")
        
        results = []
        
        for i, qa_pair in enumerate(test_pairs, 1):
            print(f"\n[{i}/{len(test_pairs)}]", end=" ")
            result = self.run_single_test(qa_pair, method=method)
            results.append(result)
            
            # Pause between requests to avoid rate limits
            time.sleep(1)
        
        # Calculate aggregate metrics
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if successful:
            avg_f1 = sum(r['similarity_metrics']['f1_score'] for r in successful) / len(successful)
            avg_precision = sum(r['similarity_metrics']['precision'] for r in successful) / len(successful)
            avg_recall = sum(r['similarity_metrics']['recall'] for r in successful) / len(successful)
            avg_latency = sum(r['latency_seconds'] for r in successful) / len(successful)
            avg_tokens = sum(r['tokens'].get('total', 0) for r in successful) / len(successful)
            
            # Aggregate RETRIEVAL metrics
            avg_contextual_relevancy = sum(
                r['retrieval_metrics']['contextual_relevancy']['relevancy_score'] 
                for r in successful
            ) / len(successful)
            avg_contextual_recall = sum(
                r['retrieval_metrics']['contextual_recall']['recall_score'] 
                for r in successful
            ) / len(successful)
            avg_contextual_precision = sum(
                r['retrieval_metrics']['contextual_precision']['precision_score'] 
                for r in successful
            ) / len(successful)
            avg_mrr = sum(
                r['retrieval_metrics']['ranking_metrics']['mrr'] 
                for r in successful
            ) / len(successful)
            
            # Aggregate GENERATION metrics
            avg_answer_relevance = sum(
                r['generation_metrics']['answer_relevance']['relevance_score'] 
                for r in successful
            ) / len(successful)
            avg_faithfulness = sum(
                r['generation_metrics']['faithfulness']['faithfulness_score'] 
                for r in successful
            ) / len(successful)
            avg_correctness = sum(
                r['generation_metrics']['correctness']['correctness_score'] 
                for r in successful
            ) / len(successful)
            avg_clarity = sum(
                r['generation_metrics']['clarity_coherence']['clarity_score'] 
                for r in successful
            ) / len(successful)
        else:
            avg_f1 = avg_precision = avg_recall = avg_latency = avg_tokens = 0
            avg_contextual_relevancy = avg_contextual_recall = avg_contextual_precision = avg_mrr = 0
            avg_answer_relevance = avg_faithfulness = avg_correctness = avg_clarity = 0
        
        aggregate_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "method": method,
            "total_tests": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_f1_score": avg_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_latency_seconds": avg_latency,
            "avg_tokens": avg_tokens,
            "retrieval_metrics_avg": {
                "contextual_relevancy": avg_contextual_relevancy,
                "contextual_recall": avg_contextual_recall,
                "contextual_precision": avg_contextual_precision,
                "mrr": avg_mrr
            },
            "generation_metrics_avg": {
                "answer_relevance": avg_answer_relevance,
                "faithfulness": avg_faithfulness,
                "correctness": avg_correctness,
                "clarity": avg_clarity
            },
            "individual_results": results
        }
        
        # Save results
        if save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"eval_{dataset}_{method}_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(aggregate_results, f, indent=2)
            print(f"\n✓ Results saved to: {results_file}")
        
        # Print summary
        self.print_summary(aggregate_results)
        
        return aggregate_results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {results['dataset']}")
        print(f"Method: {results['method']}")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        
        print(f"\n📊 Basic Performance Metrics:")
        print(f"  Avg F1 Score: {results['avg_f1_score']:.1%}")
        print(f"  Avg Precision: {results['avg_precision']:.1%}")
        print(f"  Avg Recall: {results['avg_recall']:.1%}")
        print(f"  Avg Latency: {results['avg_latency_seconds']:.2f}s")
        print(f"  Avg Tokens: {results['avg_tokens']:.0f}")
        
        # Display RETRIEVAL metrics
        if 'retrieval_metrics_avg' in results:
            ret_metrics = results['retrieval_metrics_avg']
            print(f"\n🔍 Retrieval Component Metrics:")
            print(f"  Contextual Relevancy: {ret_metrics['contextual_relevancy']:.1%}")
            print(f"  Contextual Recall: {ret_metrics['contextual_recall']:.1%}")
            print(f"  Contextual Precision: {ret_metrics['contextual_precision']:.1%}")
            print(f"  MRR (Mean Reciprocal Rank): {ret_metrics['mrr']:.3f}")
        
        # Display GENERATION metrics
        if 'generation_metrics_avg' in results:
            gen_metrics = results['generation_metrics_avg']
            print(f"\n✨ Generation Component Metrics:")
            print(f"  Answer Relevance: {gen_metrics['answer_relevance']:.1%}")
            print(f"  Faithfulness (Groundedness): {gen_metrics['faithfulness']:.1%}")
            print(f"  Correctness: {gen_metrics['correctness']:.1%}")
            print(f"  Clarity: {gen_metrics['clarity']:.1%}")
        
        print(f"\n{'='*60}\n")
    
    def compare_methods(
        self,
        dataset: str = "val",
        methods: List[str] = None
    ) -> Dict:
        """
        Compare different retrieval methods
        
        Args:
            dataset: "train" or "val"
            methods: List of methods to compare
            
        Returns:
            Dict with comparison results
        """
        if methods is None:
            methods = ["basic", "advanced"]
        
        print(f"\n{'='*60}")
        print(f"METHOD COMPARISON")
        print(f"Dataset: {dataset}")
        print(f"Methods: {', '.join(methods)}")
        print(f"{'='*60}")
        
        all_results = {}
        
        for method in methods:
            print(f"\n\nTesting method: {method.upper()}")
            results = self.run_evaluation(dataset=dataset, method=method, save_results=True)
            all_results[method] = results
            time.sleep(2)  # Pause between methods
        
        # Print comparison
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for method in methods:
            r = all_results.get(method, {})
            print(f"\n{method.upper()}:")
            print(f"  Success Rate: {r.get('success_rate', 0):.1%}")
            print(f"  F1 Score: {r.get('avg_f1_score', 0):.1%}")
            print(f"  Precision: {r.get('avg_precision', 0):.1%}")
            print(f"  Recall: {r.get('avg_recall', 0):.1%}")
            print(f"  Avg Latency: {r.get('avg_latency_seconds', 0):.2f}s")
        
        print(f"\n{'='*60}\n")
        
        return all_results


def main():
    """Run evaluation"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Evaluate RAG system")
    ap.add_argument("--dataset", default="val", choices=["train", "val"], help="Dataset to use")
    ap.add_argument("--method", default="advanced", choices=["basic", "advanced"], help="Method to test")
    ap.add_argument("--compare", action="store_true", help="Compare all methods")
    args = ap.parse_args()
    
    evaluator = EvaluationFramework()
    
    if args.compare:
        evaluator.compare_methods(dataset=args.dataset)
    else:
        evaluator.run_evaluation(dataset=args.dataset, method=args.method)


if __name__ == "__main__":
    main()
