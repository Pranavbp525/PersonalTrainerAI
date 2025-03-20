"""
RAG Evaluation Framework for PersonalTrainerAI

This module implements evaluation metrics and methods for comparing
different RAG implementations.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import RAG implementations
from .naive_rag import NaiveRAG
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG
from .graph_rag import GraphRAG
from .raptor_rag import RAPTORRAG


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGEvaluator:
    """
    Evaluation framework for comparing different RAG implementations.
    
    This class provides methods to evaluate and compare the performance of
    different RAG implementations using various metrics.
    """
    
    def __init__(
        self,
        evaluator_model_name: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            evaluator_model_name: Name of the LLM to use for evaluation
            temperature: Temperature parameter for the evaluation LLM
            max_tokens: Maximum tokens for LLM response
        """
        # Initialize evaluation LLM (using a more capable model for evaluation)
        self.evaluator_llm = OpenAI(
            model_name=evaluator_model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize RAG implementations
        self.rag_implementations = {
            "naive": NaiveRAG(),
            "advanced": AdvancedRAG(),
            "modular": ModularRAG(),
            "graph": GraphRAG(),
            "raptor": RaptorRAG()
        }
        
        # Define evaluation metrics
        self.metrics = [
            "relevance",
            "factual_accuracy",
            "completeness",
            "hallucination",
            "relationship_awareness",
            "reasoning_quality"
        ]
        
        # Set up evaluation prompts
        self._setup_evaluation_prompts()

    def _setup_evaluation_prompts(self):
        """Set up prompts for different evaluation metrics."""
        self.evaluation_prompts = {
            "relevance": PromptTemplate(
                input_variables=["query", "response"],
                template="""
                Evaluate the relevance of the response to the user's query on a scale of 1-10.
                A score of 10 means the response directly and completely addresses the query.
                A score of 1 means the response is completely irrelevant to the query.
                
                User query: {query}
                
                Response: {response}
                
                Relevance score (1-10):
                """
            ),
            "factual_accuracy": PromptTemplate(
                input_variables=["query", "response", "ground_truth"],
                template="""
                Evaluate the factual accuracy of the response on a scale of 1-10.
                A score of 10 means all facts in the response are correct according to the ground truth.
                A score of 1 means the response contains many factual errors.
                
                User query: {query}
                
                Response: {response}
                
                Ground truth information: {ground_truth}
                
                Factual accuracy score (1-10):
                """
            ),
            "completeness": PromptTemplate(
                input_variables=["query", "response", "ground_truth"],
                template="""
                Evaluate the completeness of the response on a scale of 1-10.
                A score of 10 means the response covers all important aspects of the query.
                A score of 1 means the response is missing critical information.
                
                User query: {query}
                
                Response: {response}
                
                Ground truth information: {ground_truth}
                
                Completeness score (1-10):
                """
            ),
            "hallucination": PromptTemplate(
                input_variables=["query", "response", "ground_truth"],
                template="""
                Evaluate the response for hallucination on a scale of 1-10.
                A score of 10 means the response contains no information not supported by the ground truth.
                A score of 1 means the response contains many hallucinated facts.
                
                User query: {query}
                
                Response: {response}
                
                Ground truth information: {ground_truth}
                
                Hallucination score (1-10, where 10 means NO hallucination):
                """
            ),
            "relationship_awareness": PromptTemplate(
                input_variables=["query", "response", "ground_truth"],
                template="""
                Evaluate how well the response demonstrates understanding of relationships between fitness concepts on a scale of 1-10.
                A score of 10 means the response shows excellent understanding of how different fitness concepts relate to each other.
                A score of 1 means the response treats concepts in isolation with no understanding of their relationships.
                
                User query: {query}
                
                Response: {response}
                
                Ground truth information: {ground_truth}
                
                Relationship awareness score (1-10):
                """
            ),
            "reasoning_quality": PromptTemplate(
                input_variables=["query", "response"],
                template="""
                Evaluate the quality of reasoning in the response on a scale of 1-10.
                A score of 10 means the response demonstrates excellent logical reasoning and explains concepts clearly.
                A score of 1 means the response shows poor reasoning with logical errors or unclear explanations.
                
                User query: {query}
                
                Response: {response}
                
                Reasoning quality score (1-10):
                """
            )
        }
        
        # Create evaluation chains
        self.evaluation_chains = {
            metric: LLMChain(
                llm=self.evaluator_llm,
                prompt=prompt
            )
            for metric, prompt in self.evaluation_prompts.items()
        }

    def evaluate_response(
        self,
        query: str,
        response: str,
        ground_truth: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a response using specified metrics.
        
        Args:
            query: User query
            response: Generated response to evaluate
            ground_truth: Ground truth information for factual evaluation
            metrics: List of metrics to evaluate (defaults to all)
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = self.metrics
        
        scores = {}
        
        for metric in metrics:
            if metric not in self.evaluation_chains:
                logger.warning(f"Unknown metric: {metric}")
                continue
            
            try:
                # Run evaluation chain
                if metric in ["relevance", "reasoning_quality"]:
                    # These metrics don't need ground truth
                    result = self.evaluation_chains[metric].run(
                        query=query,
                        response=response
                    )
                else:
                    # These metrics need ground truth
                    result = self.evaluation_chains[metric].run(
                        query=query,
                        response=response,
                        ground_truth=ground_truth
                    )
                
                # Extract numeric score
                try:
                    score = float(result.strip())
                    scores[metric] = score
                except ValueError:
                    logger.error(f"Could not parse score for {metric}: {result}")
                    scores[metric] = 0.0
            except Exception as e:
                logger.error(f"Error evaluating {metric}: {e}")
                scores[metric] = 0.0
        
        return scores

    def evaluate_implementation(
        self,
        implementation_name: str,
        test_queries: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a specific RAG implementation using test queries.
        
        Args:
            implementation_name: Name of the RAG implementation to evaluate
            test_queries: List of test queries with ground truth
            metrics: List of metrics to evaluate (defaults to all)
            
        Returns:
            Evaluation results
        """
        if implementation_name not in self.rag_implementations:
            raise ValueError(f"Unknown RAG implementation: {implementation_name}")
        
        if metrics is None:
            metrics = self.metrics
        
        rag_implementation = self.rag_implementations[implementation_name]
        
        results = {
            "implementation": implementation_name,
            "queries": [],
            "average_scores": {metric: 0.0 for metric in metrics}
        }
        
        for query_data in test_queries:
            query = query_data["query"]
            ground_truth = query_data.get("ground_truth", "")
            
            # Generate response using the RAG implementation
            response = rag_implementation.answer_question(query)
            
            # Evaluate the response
            scores = self.evaluate_response(
                query=query,
                response=response,
                ground_truth=ground_truth,
                metrics=metrics
            )
            
            # Store results for this query
            query_result = {
                "query": query,
                "response": response,
                "scores": scores
            }
            
            results["queries"].append(query_result)
            
            # Update average scores
            for metric in metrics:
                results["average_scores"][metric] += scores.get(metric, 0.0)
        
        # Calculate final averages
        num_queries = len(test_queries)
        if num_queries > 0:
            for metric in metrics:
                results["average_scores"][metric] /= num_queries
        
        return results

    def compare_implementations(
        self,
        test_queries: List[Dict[str, Any]],
        implementations: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple RAG implementations using test queries.
        
        Args:
            test_queries: List of test queries with ground truth
            implementations: List of implementation names to compare (defaults to all)
            metrics: List of metrics to evaluate (defaults to all)
            
        Returns:
            Comparison results
        """
        if implementations is None:
            implementations = list(self.rag_implementations.keys())
        
        if metrics is None:
            metrics = self.metrics
        
        comparison_results = {
            "implementations": implementations,
            "metrics": metrics,
            "results": {}
        }
        
        for implementation in implementations:
            try:
                results = self.evaluate_implementation(
                    implementation_name=implementation,
                    test_queries=test_queries,
                    metrics=metrics
                )
                
                comparison_results["results"][implementation] = results
            except Exception as e:
                logger.error(f"Error evaluating {implementation}: {e}")
                comparison_results["results"][implementation] = {
                    "error": str(e)
                }
        
        # Determine best implementation for each metric
        best_implementations = {}
        
        for metric in metrics:
            best_score = -1
            best_impl = None
            
            for implementation, results in comparison_results["results"].items():
                if "average_scores" in results:
                    score = results["average_scores"].get(metric, 0.0)
                    if score > best_score:
                        best_score = score
                        best_impl = implementation
            
            if best_impl:
                best_implementations[metric] = {
                    "implementation": best_impl,
                    "score": best_score
                }
        
        comparison_results["best_implementations"] = best_implementations
        
        # Calculate overall best implementation
        overall_scores = {}
        
        for implementation in implementations:
            if implementation in comparison_results["results"]:
                results = comparison_results["results"][implementation]
                if "average_scores" in results:
                    overall_scores[implementation] = sum(
                        results["average_scores"].get(metric, 0.0)
                        for metric in metrics
                    ) / len(metrics)
        
        if overall_scores:
            best_implementation = max(overall_scores.items(), key=lambda x: x[1])
            comparison_results["overall_best"] = {
                "implementation": best_implementation[0],
                "score": best_implementation[1]
            }
        
        return comparison_results

    def generate_test_queries(
        self,
        num_queries: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate test queries for evaluation.
        
        Args:
            num_queries: Number of test queries to generate
            categories: List of query categories to include
            
        Returns:
            List of test queries with ground truth
        """
        if categories is None:
            categories = [
                "workout_plan",
                "exercise_technique",
                "nutrition",
                "fitness_goal",
                "equipment",
                "recovery"
            ]
        
        # Prompt for generating test queries
        query_gen_prompt = PromptTemplate(
            input_variables=["category", "num_queries"],
            template="""
            Generate {num_queries} realistic user queries about fitness in the category: {category}.
            For each query, also provide ground truth information that would be needed to evaluate responses.
            
            Format your response as a JSON array of objects, each with "query" and "ground_truth" fields.
            
            Example:
            [
                {{
                    "query": "What's a good beginner workout routine for building muscle?",
                    "ground_truth": "A good beginner workout routine for building muscle should include compound exercises like squats, deadlifts, bench press, rows, and overhead press. It should be performed 3-4 times per week with progressive overload. Rest periods of 48 hours between working the same muscle groups are recommended. Beginners should start with lighter weights to learn proper form before increasing intensity."
                }}
            ]
            
            JSON:
            """
        )
        
        query_gen_chain = LLMChain(
            llm=self.evaluator_llm,
            prompt=query_gen_prompt
        )
        
        # Generate queries for each category
        all_queries = []
        queries_per_category = num_queries // len(categories)
        
        for category in categories:
            try:
                result = query_gen_chain.run(
                    category=category,
                    num_queries=queries_per_category
                )
                
                # Parse JSON result
                category_queries = json.loads(result)
                
                # Add category to each query
                for query_data in category_queries:
                    query_data["category"] = category
                
                all_queries.extend(category_queries)
            except Exception as e:
                logger.error(f"Error generating queries for {category}: {e}")
        
        return all_queries

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results
            output_file: Path to save results
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def generate_report(self, comparison_results: Dict[str, Any], output_file: str):
        """
        Generate a human-readable report from comparison results.
        
        Args:
            comparison_results: Results from compare_implementations
            output_file: Path to save report
        """
        try:
            with open(output_file, 'w') as f:
                f.write("# RAG Implementation Comparison Report\n\n")
                
                # Overall best implementation
                if "overall_best" in comparison_results:
                    best = comparison_results["overall_best"]
                    f.write(f"## Overall Best Implementation: {best['implementation']}\n")
                    f.write(f"Overall Score: {best['score']:.2f}/10\n\n")
                
                # Best implementation for each metric
                f.write("## Best Implementation by Metric\n\n")
                
                for metric, data in comparison_results.get("best_implementations", {}).items():
                    f.write(f"- **{metric.replace('_', ' ').title()}**: {data['implementation']} (Score: {data['score']:.2f}/10)\n")
                
                f.write("\n## Average Scores by Implementation\n\n")
                
                # Table header
                metrics = comparison_results.get("metrics", [])
                f.write("| Implementation | " + " | ".join(m.replace('_', ' ').title() for m in metrics) + " | Overall |\n")
                f.write("|---------------|" + "|".join(["-" * (len(m.replace('_', ' ').title()) + 2) for m in metrics]) + "|--------|\n")
                
                # Table rows
                for implementation in comparison_results.get("implementations", []):
                    results = comparison_results.get("results", {}).get(implementation, {})
                    
                    if "average_scores" in results:
                        scores = results["average_scores"]
                        overall = sum(scores.values()) / len(scores)
                        
                        row = f"| {implementation} | "
                        row += " | ".join(f"{scores.get(m, 0.0):.2f}" for m in metrics)
                        row += f" | {overall:.2f} |\n"
                        
                        f.write(row)
                
                f.write("\n## Detailed Results\n\n")
                
                # Detailed results for each implementation
                for implementation in comparison_results.get("implementations", []):
                    results = comparison_results.get("results", {}).get(implementation, {})
                    
                    if "error" in results:
                        f.write(f"### {implementation}\n\n")
                        f.write(f"Error: {results['error']}\n\n")
                        continue
                    
                    if "queries" not in results:
                        continue
                    
                    f.write(f"### {implementation}\n\n")
                    
                    for i, query_result in enumerate(results.get("queries", [])):
                        f.write(f"#### Query {i+1}: {query_result.get('query')}\n\n")
                        f.write(f"**Response:**\n\n{query_result.get('response')}\n\n")
                        f.write("**Scores:**\n\n")
                        
                        for metric, score in query_result.get("scores", {}).items():
                            f.write(f"- {metric.replace('_', ' ').title()}: {score:.2f}/10\n")
                        
                        f.write("\n")
            
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def visualize_results(self, comparison_results: Dict[str, Any], output_dir: str):
        """
        Generate visualizations of comparison results.
        
        Args:
            comparison_results: Results from compare_implementations
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract data for visualization
            implementations = comparison_results.get("implementations", [])
            metrics = comparison_results.get("metrics", [])
            
            # Create DataFrame for scores
            data = []
            
            for implementation in implementations:
                results = comparison_results.get("results", {}).get(implementation, {})
                
                if "average_scores" in results:
                    scores = results["average_scores"]
                    row = {"Implementation": implementation}
                    
                    for metric in metrics:
                        row[metric.replace('_', ' ').title()] = scores.get(metric, 0.0)
                    
                    data.append(row)
            
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("No data available for visualization")
                return
            
            # Set Implementation as index
            df.set_index("Implementation", inplace=True)
            
            # Bar chart for each metric
            plt.figure(figsize=(12, 8))
            df.plot(kind='bar', figsize=(12, 8))
            plt.title("RAG Implementation Comparison")
            plt.xlabel("Implementation")
            plt.ylabel("Score (0-10)")
            plt.ylim(0, 10)
            plt.legend(title="Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "comparison_bar_chart.png"))
            
            # Radar chart
            plt.figure(figsize=(10, 10))
            
            # Number of metrics
            N = len(metrics)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot for each implementation
            ax = plt.subplot(111, polar=True)
            
            for implementation in df.index:
                values = df.loc[implementation].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=implementation)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels
            plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics])
            
            # Set y-axis limits
            plt.ylim(0, 10)
            
            plt.title("RAG Implementation Comparison")
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.savefig(os.path.join(output_dir, "comparison_radar_chart.png"))
            
            # Heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(df.values, cmap='YlGn', aspect='auto', vmin=0, vmax=10)
            plt.colorbar(label='Score (0-10)')
            plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
            plt.yticks(range(len(df.index)), df.index)
            plt.title("RAG Implementation Comparison Heatmap")
            
            # Add text annotations
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha='center', va='center', color='black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "comparison_heatmap.png"))
            
            logger.info(f"Visualizations saved to {output_dir}")
        except ImportError:
            logger.warning("Matplotlib or pandas not installed. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")


def main():
    """Run RAG evaluation as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare RAG implementations")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--implementations", nargs="+", default=None, help="RAG implementations to evaluate")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to evaluate")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of test queries to generate")
    parser.add_argument("--test-queries-file", help="JSON file with test queries")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Get test queries
    if args.test_queries_file:
        try:
            with open(args.test_queries_file, 'r') as f:
                test_queries = json.load(f)
        except Exception as e:
            logger.error(f"Error loading test queries: {e}")
            test_queries = evaluator.generate_test_queries(num_queries=args.num_queries)
    else:
        test_queries = evaluator.generate_test_queries(num_queries=args.num_queries)
    
    # Save test queries
    with open(os.path.join(args.output_dir, "test_queries.json"), 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    # Compare implementations
    comparison_results = evaluator.compare_implementations(
        test_queries=test_queries,
        implementations=args.implementations,
        metrics=args.metrics
    )
    
    # Save results
    evaluator.save_results(
        results=comparison_results,
        output_file=os.path.join(args.output_dir, "comparison_results.json")
    )
    
    # Generate report
    evaluator.generate_report(
        comparison_results=comparison_results,
        output_file=os.path.join(args.output_dir, "comparison_report.md")
    )
    
    # Generate visualizations
    evaluator.visualize_results(
        comparison_results=comparison_results,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
