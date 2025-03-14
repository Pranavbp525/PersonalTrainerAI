"""
Main script to test and compare RAG implementations for PersonalTrainerAI

This script runs the evaluation framework to compare different RAG implementations
and selects the best approach based on evaluation metrics.
"""

import os
import logging
import json
from typing import Dict, Any
import argparse
from dotenv import load_dotenv

# Import RAG implementations and evaluator
from rag_model.naive_rag import NaiveRAG
from rag_model.advanced_rag import AdvancedRAG
from rag_model.modular_rag import ModularRAG
from rag_model.rag_evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """
    Main function to run RAG comparison and evaluation.
    """
    parser = argparse.ArgumentParser(description='Compare RAG implementations for PersonalTrainerAI')
    parser.add_argument('--test-queries', type=str, help='Path to test queries JSON file')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--generate-queries', type=int, default=0, help='Number of test queries to generate')
    parser.add_argument('--implementations', type=str, default='all', 
                        help='Comma-separated list of implementations to test (naive,advanced,modular)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(test_queries_path=args.test_queries)
    
    # Generate additional test queries if requested
    if args.generate_queries > 0:
        evaluator.generate_test_queries(args.generate_queries)
        evaluator.save_test_queries(os.path.join(args.output_dir, "test_queries.json"))
    
    # Determine which implementations to test
    implementations_to_test = []
    if args.implementations.lower() == 'all':
        implementations_to_test = ['naive', 'advanced', 'modular']
    else:
        implementations_to_test = [impl.strip().lower() for impl in args.implementations.split(',')]
    
    # Run evaluation for each implementation
    results = {}
    for implementation in implementations_to_test:
        if implementation in evaluator.rag_implementations:
            logger.info(f"Evaluating {implementation} RAG implementation")
            results[implementation] = evaluator.evaluate_implementation(implementation)
        else:
            logger.warning(f"Implementation {implementation} not found, skipping")
    
    # Determine best implementation
    best_implementation = None
    best_score = -1
    
    for implementation_name, implementation_results in results.items():
        if "average_scores" in implementation_results and "overall" in implementation_results["average_scores"]:
            overall_score = implementation_results["average_scores"]["overall"]
            if overall_score > best_score:
                best_score = overall_score
                best_implementation = implementation_name
    
    # Create comparison results
    comparison_results = {
        "comparison": results,
        "best_implementation": best_implementation,
        "best_score": best_score
    }
    
    # Save evaluation results
    evaluator.save_evaluation_results(
        comparison_results, 
        os.path.join(args.output_dir, "evaluation_results.json")
    )
    
    # Print summary
    print("\n" + "="*50)
    print("RAG IMPLEMENTATION COMPARISON RESULTS")
    print("="*50)
    
    if best_implementation:
        print(f"\nBest implementation: {best_implementation.upper()}")
        print(f"Best overall score: {best_score:.2f}/10.0")
        print("\nDetailed scores by implementation:")
        
        for implementation, implementation_results in results.items():
            if "average_scores" in implementation_results:
                print(f"\n{implementation.upper()} RAG:")
                for metric, score in implementation_results["average_scores"].items():
                    print(f"  {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
    else:
        print("\nNo valid comparison results available.")
    
    print("\nDetailed evaluation results saved to:", os.path.join(args.output_dir, "evaluation_results.json"))
    print("="*50)

if __name__ == "__main__":
    main()
