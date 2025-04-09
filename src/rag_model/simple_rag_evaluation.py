"""
Simple RAG Evaluation Script for PersonalTrainerAI

This script evaluates and compares the performance of different RAG implementations:
- Advanced RAG
- Modular RAG
- RAPTOR RAG

Usage:
    python simple_rag_evaluation.py --output-dir results
"""

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test queries for evaluation
TEST_QUERIES = [
    "How much protein should I consume daily for muscle growth?",
    "What are the best exercises for core strength?",
    "How do I create a balanced workout routine for beginners?",
    "What's the optimal rest period between strength training sessions?",
    "How can I improve my running endurance?",
    "What should I eat before and after a workout?",
    "How do I prevent muscle soreness after exercise?",
    "What are the benefits of high-intensity interval training?",
    "How can I track my fitness progress effectively?",
    "What's the proper form for a deadlift?"
]

class SimpleRAGEvaluator:
    """
    A simple evaluator for comparing different RAG implementations.
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        test_queries: List[str] = None,
        evaluation_llm_model: str = "gpt-4"
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            test_queries: List of test queries to evaluate
            evaluation_llm_model: LLM model to use for evaluation
        """
        self.output_dir = output_dir
        self.test_queries = test_queries or TEST_QUERIES
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluation LLM
        self.evaluation_llm = ChatOpenAI(
            model_name=evaluation_llm_model,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Import RAG implementations directly in the method to avoid import errors
        self.rag_implementations = {}
        self._initialize_rag_implementations()
        
        # Evaluation metrics
        self.metrics = [
            "relevance",
            "factual_accuracy",
            "completeness",
            "hallucination",
            "relationship_awareness",
            "reasoning_quality"
        ]
        
        logger.info(f"SimpleRAGEvaluator initialized with {len(self.test_queries)} test queries")
    
    def _initialize_rag_implementations(self):
        """Initialize RAG implementations with proper imports."""
        try:
            # Import Advanced RAG
            from src.rag_model.advanced_rag import AdvancedRAG
            self.rag_implementations["advanced"] = AdvancedRAG()
            logger.info("Advanced RAG implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Advanced RAG: {e}")
        
        try:
            # Import Modular RAG
            from src.rag_model.modular_rag import ModularRAG
            self.rag_implementations["modular"] = ModularRAG()
            logger.info("Modular RAG implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Modular RAG: {e}")
        
        try:
            # Import RAPTOR RAG
            from src.rag_model.raptor_rag import RaptorRAG
            self.rag_implementations["raptor"] = RaptorRAG()
            logger.info("RAPTOR RAG implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAPTOR RAG: {e}")
    
    def evaluate_response(self, query: str, response: str) -> Dict[str, float]:
        """
        Evaluate a response based on multiple metrics.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            Dictionary of evaluation scores
        """
        # Prompt for evaluation
        evaluation_prompt = f"""
        You are an expert evaluator of RAG (Retrieval Augmented Generation) systems for fitness knowledge.
        
        Please evaluate the following response to a fitness-related query based on these metrics:
        
        1. Relevance (1-10): How well the response addresses the user's query
        2. Factual Accuracy (1-10): Whether the information provided is correct
        3. Completeness (1-10): Whether the response covers all important aspects
        4. Hallucination (1-10, higher is better): Whether the response avoids containing information not supported by evidence
        5. Relationship Awareness (1-10): How well the response demonstrates understanding of relationships between fitness concepts
        6. Reasoning Quality (1-10): The quality of reasoning demonstrated in the response
        
        Query: {query}
        
        Response: {response}
        
        Provide your evaluation as a JSON object with the metrics as keys and scores (1-10) as values.
        Include a brief explanation for each score.
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(evaluation_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract scores from evaluation
        try:
            # Try to parse JSON directly
            scores = json.loads(evaluation_text)
            
            # Ensure all metrics are present
            for metric in self.metrics:
                if metric not in scores:
                    scores[metric] = 5.0  # Default score
            
            # Extract only numeric scores
            scores = {k: float(v) if isinstance(v, (int, float)) else 5.0 
                     for k, v in scores.items() if k in self.metrics}
            
        except json.JSONDecodeError:
            # Fallback: extract scores using simple parsing
            scores = {}
            for metric in self.metrics:
                score_match = f"{metric}: (\\d+)"
                import re
                match = re.search(score_match, evaluation_text, re.IGNORECASE)
                if match:
                    scores[metric] = float(match.group(1))
                else:
                    scores[metric] = 5.0  # Default score
        
        # Calculate overall score (average of all metrics)
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def evaluate_implementation(self, implementation_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific RAG implementation.
        
        Args:
            implementation_name: Name of the RAG implementation to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {implementation_name} RAG implementation")
        
        if implementation_name not in self.rag_implementations:
            logger.error(f"Unknown implementation: {implementation_name}")
            return {"error": f"Unknown implementation: {implementation_name}"}
        
        rag = self.rag_implementations[implementation_name]
        results = []
        total_response_time = 0
        
        for i, query in enumerate(self.test_queries):
            logger.info(f"Processing query {i+1}/{len(self.test_queries)}: {query[:50]}...")
            
            # Measure response time
            start_time = time.time()
            response = rag.answer_question(query)
            end_time = time.time()
            response_time = end_time - start_time
            total_response_time += response_time
            
            # Evaluate response
            evaluation = self.evaluate_response(query, response)
            
            # Store result
            results.append({
                "query": query,
                "response": response,
                "evaluation": evaluation,
                "response_time": response_time
            })
            
            logger.info(f"Query {i+1} completed. Overall score: {evaluation['overall']:.2f}/10.0")
        
        # Calculate average scores
        average_scores = {}
        for metric in self.metrics + ["overall"]:
            average_scores[metric] = sum(r["evaluation"][metric] for r in results) / len(results)
        
        # Calculate average response time
        average_response_time = total_response_time / len(self.test_queries)
        
        return {
            "implementation": implementation_name,
            "results": results,
            "average_scores": average_scores,
            "average_response_time": average_response_time
        }
    
    def compare_implementations(self) -> Dict[str, Any]:
        """
        Compare all RAG implementations.
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting RAG implementation comparison")
        
        results = {}
        for implementation_name in self.rag_implementations:
            results[implementation_name] = self.evaluate_implementation(implementation_name)
        
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
        
        # Save results
        self.save_results(comparison_results)
        
        # Generate comparison charts
        self.generate_comparison_charts(results)
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
        """
        output_file = os.path.join(self.output_dir, "evaluation_results.json")
        
        # Convert results to JSON-serializable format
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
    
    def generate_comparison_charts(self, results: Dict[str, Any]) -> None:
        """
        Generate comparison charts for visualization.
        
        Args:
            results: Evaluation results to visualize
        """
        # Extract data for charts
        implementations = list(results.keys())
        metrics_data = {metric: [] for metric in self.metrics + ["overall"]}
        response_times = []
        
        for implementation in implementations:
            impl_results = results[implementation]
            if "average_scores" in impl_results:
                for metric in self.metrics + ["overall"]:
                    metrics_data[metric].append(impl_results["average_scores"][metric])
            
            if "average_response_time" in impl_results:
                response_times.append(impl_results["average_response_time"])
            else:
                response_times.append(0)
        
        # Create metrics comparison chart
        plt.figure(figsize=(12, 8))
        x = range(len(implementations))
        width = 0.1
        offset = -0.3
        
        for metric in self.metrics + ["overall"]:
            plt.bar([i + offset for i in x], metrics_data[metric], width=width, label=metric.replace("_", " ").title())
            offset += width
        
        plt.xlabel("RAG Implementation")
        plt.ylabel("Score (1-10)")
        plt.title("RAG Implementation Comparison by Metric")
        plt.xticks(x, implementations)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "metrics_comparison.png"))
        
        # Create response time comparison chart
        plt.figure(figsize=(10, 6))
        plt.bar(implementations, response_times)
        plt.xlabel("RAG Implementation")
        plt.ylabel("Average Response Time (seconds)")
        plt.title("RAG Implementation Comparison by Response Time")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "response_time_comparison.png"))
        
        logger.info(f"Comparison charts saved to {self.output_dir}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.
        
        Args:
            results: Evaluation results to summarize
        """
        print("\n" + "="*50)
        print("RAG IMPLEMENTATION COMPARISON RESULTS")
        print("="*50)
        
        if "best_implementation" in results and results["best_implementation"]:
            print(f"\nBest implementation: {results['best_implementation'].upper()}")
            print(f"Best overall score: {results['best_score']:.2f}/10.0")
            
            print("\nDetailed scores by implementation:")
            for implementation, implementation_results in results["comparison"].items():
                if "average_scores" in implementation_results:
                    print(f"\n{implementation.upper()} RAG:")
                    for metric, score in implementation_results["average_scores"].items():
                        print(f"  {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
                    print(f"  Response Time: {implementation_results.get('average_response_time', 0):.2f}s")
        else:
            print("\nNo valid comparison results available.")
        
        print("\nDetailed evaluation results saved to:", os.path.join(self.output_dir, "evaluation_results.json"))
        print("Comparison charts saved to:", self.output_dir)
        print("="*50)


def main():
    """Main function to run the RAG evaluation."""
    parser = argparse.ArgumentParser(description="Simple RAG Evaluation for PersonalTrainerAI")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4", help="LLM model to use for evaluation")
    parser.add_argument("--implementation", type=str, choices=["advanced", "modular", "raptor", "all"], 
                        default="all", help="RAG implementation to evaluate")
    parser.add_argument("--num-queries", type=int, default=10, 
                        help="Number of test queries to evaluate (max 10)")
    args = parser.parse_args()
    
    # Limit number of queries if specified
    test_queries = TEST_QUERIES[:min(args.num_queries, len(TEST_QUERIES))]
    
    # Initialize evaluator
    evaluator = SimpleRAGEvaluator(
        output_dir=args.output_dir,
        test_queries=test_queries,
        evaluation_llm_model=args.evaluation_model
    )
    
    # Run evaluation
    if args.implementation == "all":
        results = evaluator.compare_implementations()
        evaluator.print_summary(results)
    else:
        results = evaluator.evaluate_implementation(args.implementation)
        print(f"\nEvaluation results for {args.implementation.upper()} RAG:")
        for metric, score in results["average_scores"].items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
        print(f"  Response Time: {results.get('average_response_time', 0):.2f}s")
        
        # Save individual results
        individual_results = {
            "comparison": {args.implementation: results},
            "best_implementation": args.implementation,
            "best_score": results["average_scores"]["overall"]
        }
        evaluator.save_results(individual_results)
        print(f"\nDetailed evaluation results saved to: {os.path.join(args.output_dir, 'evaluation_results.json')}")


if __name__ == "__main__":
    main()
