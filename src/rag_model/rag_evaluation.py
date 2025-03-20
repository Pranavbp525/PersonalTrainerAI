"""
RAG Evaluation Module for PersonalTrainerAI

This module provides tools for evaluating and comparing different RAG implementations.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import RAG implementations
from .naive_rag import NaiveRAG
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG
from .graph_rag import GraphRAG
from .raptor_rag import RaptorRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGEvaluator:
    """
    Evaluator for comparing different RAG implementations.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            embedding_model_name: Name of the embedding model to use for semantic similarity
            llm_model_name: Name of the language model to use for evaluation
        """
        # Load environment variables
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not self.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")
        
        # Initialize embedding model for semantic similarity
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize LLM for evaluation
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.evaluator_llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_key=self.OPENAI_API_KEY)
        
        # Define evaluation prompt templates
        self.relevance_template = PromptTemplate(
            input_variables=["question", "answer"],
            template="""
            Evaluate the relevance of this answer to the question on a scale of 1-10.
            
            Question: {question}
            
            Answer: {answer}
            
            A score of 1 means completely irrelevant, and 10 means perfectly relevant.
            Consider whether the answer directly addresses what was asked.
            
            Provide only a numerical score from 1-10, nothing else.
            """
        )
        
        self.completeness_template = PromptTemplate(
            input_variables=["question", "answer"],
            template="""
            Evaluate the completeness of this answer to the question on a scale of 1-10.
            
            Question: {question}
            
            Answer: {answer}
            
            A score of 1 means extremely incomplete, and 10 means fully comprehensive.
            Consider whether the answer covers all aspects of the question.
            
            Provide only a numerical score from 1-10, nothing else.
            """
        )
        
        self.accuracy_template = PromptTemplate(
            input_variables=["question", "answer", "ground_truth"],
            template="""
            Evaluate the accuracy of this answer compared to the ground truth on a scale of 1-10.
            
            Question: {question}
            
            Answer to evaluate: {answer}
            
            Ground truth: {ground_truth}
            
            A score of 1 means completely inaccurate, and 10 means perfectly accurate.
            Consider factual correctness and alignment with the ground truth.
            
            Provide only a numerical score from 1-10, nothing else.
            """
        )
        
        self.coherence_template = PromptTemplate(
            input_variables=["answer"],
            template="""
            Evaluate the coherence and clarity of this answer on a scale of 1-10.
            
            Answer: {answer}
            
            A score of 1 means completely incoherent, and 10 means perfectly clear and well-structured.
            Consider logical flow, organization, and readability.
            
            Provide only a numerical score from 1-10, nothing else.
            """
        )
        
        # Create LLM chains for evaluation
        self.relevance_chain = LLMChain(llm=self.evaluator_llm, prompt=self.relevance_template)
        self.completeness_chain = LLMChain(llm=self.evaluator_llm, prompt=self.completeness_template)
        self.accuracy_chain = LLMChain(llm=self.evaluator_llm, prompt=self.accuracy_template)
        self.coherence_chain = LLMChain(llm=self.evaluator_llm, prompt=self.coherence_template)
        
        logger.info("RAG evaluator initialized successfully")
    
    def generate_test_queries(self, num_queries: int = 10) -> List[Dict[str, str]]:
        """
        Generate test queries for evaluation.
        
        Args:
            num_queries: Number of test queries to generate
            
        Returns:
            List of test query dictionaries with 'query' and 'ground_truth' fields
        """
        logger.info(f"Generating {num_queries} test queries")
        
        # Define query generation prompt
        query_gen_template = PromptTemplate(
            input_variables=["domain", "num_queries"],
            template="""
            Generate {num_queries} diverse and realistic questions about fitness, exercise, nutrition, and personal training.
            
            For each question, also provide a comprehensive ground truth answer that would be expected from a knowledgeable fitness professional.
            
            Format your response as a JSON array of objects, each with 'query' and 'ground_truth' fields.
            
            Example:
            [
                {{
                    "query": "What's the best way to increase my bench press max?",
                    "ground_truth": "To increase your bench press max, focus on progressive overload by gradually increasing weight, implement proper form with scapular retraction and leg drive, incorporate accessory exercises like close-grip bench and tricep extensions, ensure adequate recovery with 48-72 hours between chest workouts, optimize nutrition with sufficient protein intake, and consider periodization to cycle between higher volume and higher intensity phases."
                }},
                ...
            ]
            """
        )
        
        query_gen_chain = LLMChain(llm=self.evaluator_llm, prompt=query_gen_template)
        
        # Generate queries
        response = query_gen_chain.run(domain="fitness", num_queries=num_queries)
        
        try:
            # Parse JSON response
            test_queries = json.loads(response)
            logger.info(f"Successfully generated {len(test_queries)} test queries")
            return test_queries
        except json.JSONDecodeError:
            logger.error("Failed to parse generated queries as JSON")
            # Fallback to a few predefined queries
            return [
                {
                    "query": "What's a good beginner workout routine?",
                    "ground_truth": "A good beginner workout routine should include compound exercises like squats, push-ups, and rows, start with 2-3 sessions per week, focus on proper form, and gradually increase intensity."
                },
                {
                    "query": "How much protein should I consume daily?",
                    "ground_truth": "The general recommendation is 0.8-1g of protein per pound of bodyweight for active individuals, with higher amounts (1-1.2g/lb) beneficial for muscle building."
                }
            ]
    
    def evaluate_response(
        self, 
        question: str, 
        answer: str, 
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single response using multiple metrics.
        
        Args:
            question: The question that was asked
            answer: The answer to evaluate
            ground_truth: Optional ground truth answer for accuracy evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating response for question: {question[:50]}...")
        
        # Initialize results
        results = {}
        
        # Evaluate relevance
        relevance_score = self.relevance_chain.run(question=question, answer=answer)
        try:
            results["relevance"] = float(relevance_score.strip())
        except ValueError:
            logger.warning(f"Failed to parse relevance score: {relevance_score}")
            results["relevance"] = 5.0
        
        # Evaluate completeness
        completeness_score = self.completeness_chain.run(question=question, answer=answer)
        try:
            results["completeness"] = float(completeness_score.strip())
        except ValueError:
            logger.warning(f"Failed to parse completeness score: {completeness_score}")
            results["completeness"] = 5.0
        
        # Evaluate coherence
        coherence_score = self.coherence_chain.run(answer=answer)
        try:
            results["coherence"] = float(coherence_score.strip())
        except ValueError:
            logger.warning(f"Failed to parse coherence score: {coherence_score}")
            results["coherence"] = 5.0
        
        # Evaluate accuracy if ground truth is provided
        if ground_truth:
            accuracy_score = self.accuracy_chain.run(
                question=question, 
                answer=answer, 
                ground_truth=ground_truth
            )
            try:
                results["accuracy"] = float(accuracy_score.strip())
            except ValueError:
                logger.warning(f"Failed to parse accuracy score: {accuracy_score}")
                results["accuracy"] = 5.0
            
            # Calculate semantic similarity
            try:
                answer_embedding = self.embedding_model.embed_query(answer)
                truth_embedding = self.embedding_model.embed_query(ground_truth)
                similarity = cosine_similarity([answer_embedding], [truth_embedding])[0][0]
                results["semantic_similarity"] = float(similarity)
            except Exception as e:
                logger.warning(f"Failed to calculate semantic similarity: {e}")
                results["semantic_similarity"] = 0.5
        
        # Calculate overall score
        if ground_truth:
            results["overall"] = (
                results["relevance"] * 0.25 +
                results["completeness"] * 0.25 +
                results["coherence"] * 0.2 +
                results["accuracy"] * 0.2 +
                results["semantic_similarity"] * 10 * 0.1  # Scale up similarity to 0-10 range
            )
        else:
            results["overall"] = (
                results["relevance"] * 0.4 +
                results["completeness"] * 0.4 +
                results["coherence"] * 0.2
            )
        
        return results
    
    def compare_implementations(
        self,
        test_queries: Optional[List[Dict[str, str]]] = None,
        implementations: List[str] = ["naive", "advanced", "modular", "graph", "raptor"],
        num_queries: int = 10
    ) -> Dict[str, Any]:
        """
        Compare different RAG implementations using test queries.
        
        Args:
            test_queries: List of test queries with ground truth, generated if None
            implementations: List of implementation names to compare
            num_queries: Number of test queries to generate if test_queries is None
            
        Returns:
            Dictionary of comparison results
        """
        logger.info(f"Comparing RAG implementations: {implementations}")
        
        # Generate test queries if not provided
        if test_queries is None:
            test_queries = self.generate_test_queries(num_queries)
        
        # Initialize RAG implementations
        rag_implementations = {}
        for impl_name in implementations:
            try:
                if impl_name == "naive":
                    rag_implementations[impl_name] = NaiveRAG()
                elif impl_name == "advanced":
                    rag_implementations[impl_name] = AdvancedRAG()
                elif impl_name == "modular":
                    rag_implementations[impl_name] = ModularRAG()
                elif impl_name == "graph":
                    rag_implementations[impl_name] = GraphRAG()
                elif impl_name == "raptor":
                    rag_implementations[impl_name] = RaptorRAG()
                else:
                    logger.warning(f"Unknown implementation: {impl_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {impl_name} RAG: {e}")
        
        # Run evaluations
        results = {impl: [] for impl in rag_implementations}
        response_times = {impl: [] for impl in rag_implementations}
        all_responses = {impl: {} for impl in rag_implementations}
        
        for i, query_data in enumerate(test_queries):
            query = query_data["query"]
            ground_truth = query_data.get("ground_truth")
            
            logger.info(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            for impl_name, rag in rag_implementations.items():
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
                    evaluation = self.evaluate_response(query, response, ground_truth)
                    evaluation["response_time"] = response_time
                    results[impl_name].append(evaluation)
                    
                    logger.info(f"{impl_name} RAG: Overall score = {evaluation['overall']:.2f}, Time = {response_time:.2f}s")
                except Exception as e:
                    logger.error(f"Error evaluating {impl_name} RAG: {e}")
        
        # Aggregate results
        aggregated_results = {}
        for impl_name, impl_results in results.items():
            if impl_results:
                aggregated_results[impl_name] = {
                    "overall": np.mean([r["overall"] for r in impl_results]),
                    "relevance": np.mean([r["relevance"] for r in impl_results]),
                    "completeness": np.mean([r["completeness"] for r in impl_results]),
                    "coherence": np.mean([r["coherence"] for r in impl_results]),
                    "response_time": np.mean(response_times[impl_name])
                }
                
                if "accuracy" in impl_results[0]:
                    aggregated_results[impl_name]["accuracy"] = np.mean([r["accuracy"] for r in impl_results])
                
                if "semantic_similarity" in impl_results[0]:
                    aggregated_results[impl_name]["semantic_similarity"] = np.mean([r["semantic_similarity"] for r in impl_results])
        
        # Determine best implementation for each metric
        best_implementations = {}
        for metric in ["overall", "relevance", "completeness", "coherence", "accuracy", "semantic_similarity", "response_time"]:
            if metric in next(iter(aggregated_results.values()), {}):
                if metric == "response_time":
                    # Lower is better for response time
                    best_impl = min(aggregated_results.items(), key=lambda x: x[1].get(metric, float('inf')))[0]
                else:
                    # Higher is better for other metrics
                    best_impl = max(aggregated_results.items(), key=lambda x: x[1].get(metric, 0))[0]
                best_implementations[metric] = best_impl
        
        # Determine overall best implementation
        if aggregated_results:
            overall_best = max(aggregated_results.items(), key=lambda x: x[1]["overall"])[0]
        else:
            overall_best = None
        
        return {
            "detailed_results": results,
            "aggregated_results": aggregated_results,
            "best_implementations": best_implementations,
            "overall_best": overall_best,
            "all_responses": all_responses,
            "test_queries": test_queries
        }
    
    def generate_comparison_report(self, results: Dict[str, Any], output_dir: str = "results") -> str:
        """
        Generate a detailed comparison report from evaluation results.
        
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
        aggregated_results = results["aggregated_results"]
        best_implementations = results["best_implementations"]
        overall_best = results["overall_best"]
        test_queries = results["test_queries"]
        all_responses = results["all_responses"]
        
        # Generate report markdown
        report = "# RAG Implementation Comparison Report\n\n"
        
        # Summary
        report += "## Summary\n\n"
        report += f"- **Number of test queries:** {len(test_queries)}\n"
        report += f"- **Implementations compared:** {', '.join(aggregated_results.keys())}\n"
        report += f"- **Overall best implementation:** {overall_best}\n\n"
        
        # Detailed results table
        report += "## Performance Metrics\n\n"
        report += "| Implementation | Overall | Relevance | Completeness | Coherence |"
        
        if "accuracy" in next(iter(aggregated_results.values()), {}):
            report += " Accuracy |"
        
        if "semantic_similarity" in next(iter(aggregated_results.values()), {}):
            report += " Semantic Similarity |"
        
        report += " Response Time (s) |\n"
        report += "|" + "-|" * (5 + ("accuracy" in next(iter(aggregated_results.values()), {})) + 
                              ("semantic_similarity" in next(iter(aggregated_results.values()), {})) + 1) + "\n"
        
        for impl, metrics in aggregated_results.items():
            report += f"| {impl} | {metrics['overall']:.2f} | {metrics['relevance']:.2f} | {metrics['completeness']:.2f} | {metrics['coherence']:.2f} |"
            
            if "accuracy" in metrics:
                report += f" {metrics['accuracy']:.2f} |"
            
            if "semantic_similarity" in metrics:
                report += f" {metrics['semantic_similarity']:.2f} |"
            
            report += f" {metrics['response_time']:.2f} |\n"
        
        # Best implementations by metric
        report += "\n## Best Implementation by Metric\n\n"
        for metric, impl in best_implementations.items():
            if metric == "response_time":
                report += f"- **Fastest ({metric}):** {impl}\n"
            else:
                report += f"- **Best {metric}:** {impl}\n"
        
        # Sample responses
        report += "\n## Sample Responses\n\n"
        
        # Select a few queries to show as examples
        sample_queries = test_queries[:3] if len(test_queries) > 3 else test_queries
        
        for i, query_data in enumerate(sample_queries):
            query = query_data["query"]
            ground_truth = query_data.get("ground_truth", "Not provided")
            
            report += f"### Query {i+1}: {query}\n\n"
            report += f"**Ground Truth:** {ground_truth}\n\n"
            
            for impl in aggregated_results.keys():
                if query in all_responses.get(impl, {}):
                    response = all_responses[impl][query]
                    report += f"**{impl.capitalize()} RAG Response:**\n\n{response}\n\n"
        
        # Save report
        report_path = os.path.join(output_dir, "comparison_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Generate visualizations
        self._generate_visualizations(aggregated_results, output_dir)
        
        # Save raw results
        results_path = os.path.join(output_dir, "comparison_results.json")
        with open(results_path, "w") as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_results = {
                "aggregated_results": {
                    impl: {k: float(v) for k, v in metrics.items()}
                    for impl, metrics in aggregated_results.items()
                },
                "best_implementations": best_implementations,
                "overall_best": overall_best,
                "test_queries": test_queries
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def _generate_visualizations(self, aggregated_results: Dict[str, Dict[str, float]], output_dir: str) -> None:
        """
        Generate visualizations from aggregated results.
        
        Args:
            aggregated_results: Aggregated evaluation results
            output_dir: Directory to save visualizations
        """
        logger.info("Generating visualizations")
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(aggregated_results).T
        
        # Bar chart of overall scores
        plt.figure(figsize=(10, 6))
        ax = df["overall"].plot(kind="bar", color="skyblue")
        plt.title("Overall Performance by RAG Implementation")
        plt.xlabel("Implementation")
        plt.ylabel("Overall Score (0-10)")
        plt.ylim(0, 10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(df["overall"]):
            ax.text(i, v + 0.1, f"{v:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_bar_chart.png"))
        plt.close()
        
        # Radar chart of different metrics
        metrics = [col for col in df.columns if col != "response_time"]
        
        # Prepare data for radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Plot for each implementation
        for impl, row in df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=impl)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        plt.xticks(angles[:-1], metrics)
        
        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        
        plt.title("RAG Implementation Comparison by Metric")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_radar_chart.png"))
        plt.close()
        
        # Heatmap of all metrics
        plt.figure(figsize=(12, 8))
        
        # Normalize response time for better visualization (lower is better)
        if "response_time" in df.columns:
            max_time = df["response_time"].max()
            df["response_time_normalized"] = 10 * (1 - df["response_time"] / max_time)
            heatmap_df = df.drop(columns=["response_time"])
            heatmap_df = heatmap_df.rename(columns={"response_time_normalized": "speed"})
        else:
            heatmap_df = df.copy()
        
        plt.imshow(heatmap_df.values, cmap="YlGnBu", aspect="auto")
        plt.colorbar(label="Score (0-10)")
        
        # Add labels
        plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45)
        plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
        
        # Add values in cells
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                plt.text(j, i, f"{heatmap_df.iloc[i, j]:.2f}", 
                         ha="center", va="center", 
                         color="black" if heatmap_df.iloc[i, j] > 5 else "white")
        
        plt.title("Performance Heatmap by Implementation and Metric")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_heatmap.png"))
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare RAG implementations")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--implementations", type=str, nargs="+", 
                        default=["naive", "advanced", "modular", "graph", "raptor"],
                        help="RAG implementations to compare")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of test queries to generate")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Compare implementations
    results = evaluator.compare_implementations(
        implementations=args.implementations,
        num_queries=args.num_queries
    )
    
    # Generate report
    evaluator.generate_comparison_report(results, args.output_dir)
    
    # Print summary
    print("\nEvaluation complete!")
    print(f"Overall best implementation: {results['overall_best']}")
    print(f"Results saved to {args.output_dir}/")
