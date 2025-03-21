"""
RAG Evaluation Module for PersonalTrainerAI

This module provides a simple framework for evaluating and comparing RAG implementations.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import RAG implementations
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG
from .raptor_rag import RaptorRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SimpleRAGEvaluator:
    """
    Simple evaluator for comparing different RAG implementations.
    """
    
    def __init__(
        self,
        llm_model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            llm_model_name: Name of the language model to use for evaluation
        """
        # Load environment variables
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not self.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")
        
        # Initialize LLM for evaluation
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.evaluator_llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_key=self.OPENAI_API_KEY)
        
        # Define evaluation prompt template
        self.evaluation_template = PromptTemplate(
            input_variables=["question", "answer"],
            template="""
            Evaluate this answer to the fitness-related question on a scale of 1-10.
            
            Question: {question}
            
            Answer: {answer}
            
            Consider:
            - Relevance: Does the answer directly address the question?
            - Accuracy: Is the information correct and up-to-date?
            - Completeness: Does the answer cover all aspects of the question?
            - Clarity: Is the answer well-structured and easy to understand?
            
            Provide only a numerical score from 1-10, nothing else.
            """
        )
        
        # Create LLM chain for evaluation
        self.evaluation_chain = LLMChain(llm=self.evaluator_llm, prompt=self.evaluation_template)
        
        logger.info("RAG evaluator initialized successfully")
    
    def generate_test_queries(self, num_queries: int = 5) -> List[Dict[str, str]]:
        """
        Generate test queries for evaluation.
        
        Args:
            num_queries: Number of test queries to generate
            
        Returns:
            List of test query dictionaries with 'query' fields
        """
        logger.info(f"Generating {num_queries} test queries")
        
        # Define query generation prompt
        query_gen_template = PromptTemplate(
            input_variables=["num_queries"],
            template="""
            Generate {num_queries} diverse and realistic questions about fitness, exercise, nutrition, and personal training.
            
            Format your response as a JSON array of objects, each with a 'query' field.
            
            Example:
            [
                {{
                    "query": "What's the best way to increase my bench press max?"
                }},
                ...
            ]
            """
        )
        
        query_gen_chain = LLMChain(llm=self.evaluator_llm, prompt=query_gen_template)
        
        # Generate queries
        response = query_gen_chain.run(num_queries=num_queries)
        
        try:
            # Parse JSON response
            test_queries = json.loads(response)
            logger.info(f"Successfully generated {len(test_queries)} test queries")
            return test_queries
        except json.JSONDecodeError:
            logger.error("Failed to parse generated queries as JSON")
            # Fallback to a few predefined queries
            return [
                {"query": "What's a good beginner workout routine?"},
                {"query": "How much protein should I consume daily?"},
                {"query": "What are the best exercises for core strength?"},
                {"query": "How do I improve my running endurance?"},
                {"query": "What's the difference between HIIT and steady-state cardio?"}
            ]
    
    def evaluate_response(
        self, 
        question: str, 
        answer: str
    ) -> float:
        """
        Evaluate a single response.
        
        Args:
            question: The question that was asked
            answer: The answer to evaluate
            
        Returns:
            Evaluation score (1-10)
        """
        logger.info(f"Evaluating response for question: {question[:50]}...")
        
        # Evaluate answer
        score_text = self.evaluation_chain.run(question=question, answer=answer)
        
        try:
            # Parse score
            score = float(score_text.strip())
            return score
        except ValueError:
            logger.warning(f"Failed to parse evaluation score: {score_text}")
            return 5.0  # Default middle score
    
    def compare_implementations(
        self,
        test_queries: Optional[List[Dict[str, str]]] = None,
        num_queries: int = 3
    ) -> Dict[str, Any]:
        """
        Compare the three RAG implementations using test queries.
        
        Args:
            test_queries: List of test queries, generated if None
            num_queries: Number of test queries to generate if test_queries is None
            
        Returns:
            Dictionary of comparison results
        """
        logger.info("Comparing RAG implementations: AdvancedRAG, ModularRAG, RaptorRAG")
        
        # Generate test queries if not provided
        if test_queries is None:
            test_queries = self.generate_test_queries(num_queries)
        
        # Initialize RAG implementations
        implementations = {
            "advanced": AdvancedRAG(),
            "modular": ModularRAG(),
            "raptor": RaptorRAG()
        }
        
        # Run evaluations
        scores = {impl: [] for impl in implementations}
        response_times = {impl: [] for impl in implementations}
        all_responses = {impl: {} for impl in implementations}
        
        for i, query_data in enumerate(test_queries):
            query = query_data["query"]
            
            logger.info(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            for impl_name, rag in implementations.items():
                try:
                    # Measure response time
                    start_time = time.time()
                    response = rag.answer_question(query)
                    end_time = time.time()
                    
                    # Store response time
                    response_time = end_time - start_time
                    response_times[impl_name].append(response_time)
                    
                    # Store response
                    all_responses[impl_name][query] = response
                    
                    # Evaluate response
                    score = self.evaluate_response(query, response)
                    scores[impl_name].append(score)
                    
                    logger.info(f"{impl_name} RAG: Score = {score:.2f}, Time = {response_time:.2f}s")
                except Exception as e:
                    logger.error(f"Error evaluating {impl_name} RAG: {e}")
        
        # Calculate average scores and response times
        avg_scores = {impl: np.mean(s) if s else 0 for impl, s in scores.items()}
        avg_times = {impl: np.mean(t) if t else 0 for impl, t in response_times.items()}
        
        # Determine best implementation
        if avg_scores:
            best_impl = max(avg_scores.items(), key=lambda x: x[1])[0]
        else:
            best_impl = None
        
        # Determine fastest implementation
        if avg_times:
            fastest_impl = min(avg_times.items(), key=lambda x: x[1])[0]
        else:
            fastest_impl = None
        
        return {
            "avg_scores": avg_scores,
            "avg_times": avg_times,
            "best_implementation": best_impl,
            "fastest_implementation": fastest_impl,
            "all_responses": all_responses,
            "test_queries": test_queries
        }
    
    def generate_comparison_report(self, results: Dict[str, Any], output_dir: str = "results") -> str:
        """
        Generate a simple comparison report from evaluation results.
        
        Args:
            results: Results from compare_implementations
            output_dir: Directory to save report and visualizations
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating comparison report")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract results
        avg_scores = results["avg_scores"]
        avg_times = results["avg_times"]
        best_impl = results["best_implementation"]
        fastest_impl = results["fastest_implementation"]
        test_queries = results["test_queries"]
        all_responses = results["all_responses"]
        
        # Generate report markdown
        report = "# RAG Implementation Comparison Report\n\n"
        
        # Summary
        report += "## Summary\n\n"
        report += f"- **Number of test queries:** {len(test_queries)}\n"
        report += f"- **Best implementation:** {best_impl}\n"
        report += f"- **Fastest implementation:** {fastest_impl}\n\n"
        
        # Results table
        report += "## Performance Metrics\n\n"
        report += "| Implementation | Average Score | Average Response Time (s) |\n"
        report += "|" + "-|" * 3 + "\n"
        
        for impl in avg_scores.keys():
            report += f"| {impl} | {avg_scores[impl]:.2f} | {avg_times[impl]:.2f} |\n"
        
        # Sample responses
        report += "\n## Sample Responses\n\n"
        
        # Select a few queries to show as examples
        sample_queries = test_queries[:2] if len(test_queries) > 2 else test_queries
        
        for i, query_data in enumerate(sample_queries):
            query = query_data["query"]
            
            report += f"### Query {i+1}: {query}\n\n"
            
            for impl in avg_scores.keys():
                if query in all_responses.get(impl, {}):
                    response = all_responses[impl][query]
                    report += f"**{impl.capitalize()} RAG Response:**\n\n{response}\n\n"
        
        # Save report
        report_path = os.path.join(output_dir, "comparison_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Generate bar chart
        plt.figure(figsize=(10, 6))
        implementations = list(avg_scores.keys())
        scores = list(avg_scores.values())
        
        plt.bar(implementations, scores, color="skyblue")
        plt.title("RAG Implementation Comparison")
        plt.xlabel("Implementation")
        plt.ylabel("Average Score (1-10)")
        plt.ylim(0, 10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(scores):
            plt.text(i, v + 0.1, f"{v:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_chart.png"))
        plt.close()
        
        logger.info(f"Report saved to {report_path}")
        return report_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare RAG implementations")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--num-queries", type=int, default=3, help="Number of test queries to generate")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SimpleRAGEvaluator()
    
    # Compare implementations
    results = evaluator.compare_implementations(num_queries=args.num_queries)
    
    # Generate report
    evaluator.generate_comparison_report(results, args.output_dir)
    
    # Print summary
    print("\nEvaluation complete!")
    print(f"Best implementation: {results['best_implementation']}")
    print(f"Results saved to {args.output_dir}/")
