"""
Updated RAG Implementation Comparison Script for PersonalTrainerAI

This script compares all RAG implementations, including Graph RAG and RAPTOR RAG,
to determine which performs best for fitness knowledge retrieval.
"""

import os
import logging
import json
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

# Import RAG implementations
from rag_model.naive_rag import NaiveRAG
from rag_model.advanced_rag import AdvancedRAG
from rag_model.modular_rag import ModularRAG
from rag_model.graph_rag import GraphRAG
from rag_model.raptor_rag import RAPTORRAG
from rag_model.rag_evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def compare_rag_implementations(
    output_dir: str = "results",
    test_queries_path: str = None,
    generate_queries: int = 0,
    graph_path: str = "fitness_knowledge_graph.json",
    build_graph: bool = False
) -> Dict[str, Any]:
    """
    Compare all RAG implementations, including Graph RAG and RAPTOR RAG.
    
    Args:
        output_dir: Directory to save results
        test_queries_path: Path to test queries JSON file
        generate_queries: Number of test queries to generate
        graph_path: Path to knowledge graph for Graph RAG
        build_graph: Whether to build the knowledge graph
        
    Returns:
        Dictionary containing comparison results
    """
    logger.info("Starting RAG implementation comparison")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        test_queries_path=test_queries_path,
        graph_path=graph_path if not build_graph else None
    )
    
    # Generate additional test queries if requested
    if generate_queries > 0:
        evaluator.generate_test_queries(generate_queries)
        evaluator.save_test_queries(os.path.join(output_dir, "test_queries.json"))
    
    # Evaluate all implementations
    results = {}
    for implementation_name in evaluator.rag_implementations:
        logger.info(f"Evaluating {implementation_name} RAG implementation")
        results[implementation_name] = evaluator.evaluate_implementation(implementation_name)
    
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
        os.path.join(output_dir, "evaluation_results.json")
    )
    
    # Generate comparison charts
    evaluator.generate_comparison_charts(results, output_dir)
    
    # Generate detailed comparison report
    generate_comparison_report(results, best_implementation, best_score, output_dir)
    
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
    
    print("\nDetailed evaluation results saved to:", os.path.join(output_dir, "evaluation_results.json"))
    print("Comparison charts saved to:", output_dir)
    print("Detailed comparison report saved to:", os.path.join(output_dir, "comparison_report.md"))
    print("="*50)
    
    return comparison_results

def generate_comparison_report(
    results: Dict[str, Any],
    best_implementation: str,
    best_score: float,
    output_dir: str
) -> None:
    """
    Generate a detailed comparison report in Markdown format.
    
    Args:
        results: Evaluation results
        best_implementation: Name of the best implementation
        best_score: Score of the best implementation
        output_dir: Directory to save the report
    """
    logger.info("Generating detailed comparison report")
    
    report = f"""# RAG Implementation Comparison Report

## Overview

This report compares different RAG (Retrieval Augmented Generation) implementations for the PersonalTrainerAI project,
including the newly added Graph RAG and RAPTOR RAG architectures.

## Best Implementation

**{best_implementation.upper()} RAG** achieved the highest overall score of **{best_score:.2f}/10.0**.

## Implementation Comparison

| Implementation | Overall Score | Relevance | Factual Accuracy | Completeness | Hallucination | Relationship Awareness | Reasoning Quality | Avg Response Time |
|----------------|--------------|-----------|------------------|--------------|--------------|------------------------|-------------------|-------------------|
"""
    
    # Add rows for each implementation
    for implementation_name, implementation_results in results.items():
        if "average_scores" not in implementation_results:
            continue
            
        avg_scores = implementation_results["average_scores"]
        response_time = implementation_results.get("average_response_time", 0)
        
        overall = avg_scores.get("overall", 0)
        relevance = avg_scores.get("relevance", 0)
        factual_accuracy = avg_scores.get("factual_accuracy", 0)
        completeness = avg_scores.get("completeness", 0)
        hallucination = avg_scores.get("hallucination", 0)
        relationship_awareness = avg_scores.get("relationship_awareness", 0)
        reasoning_quality = avg_scores.get("reasoning_quality", 0)
        
        report += f"| {implementation_name.upper()} | {overall:.2f} | {relevance:.2f} | {factual_accuracy:.2f} | {completeness:.2f} | {hallucination:.2f} | {relationship_awareness:.2f} | {reasoning_quality:.2f} | {response_time:.2f}s |\n"
    
    # Add performance by query complexity
    report += """
## Performance by Query Complexity

Different RAG implementations perform differently based on query complexity:

| Implementation | Low Complexity | Medium Complexity | High Complexity |
|----------------|---------------|------------------|----------------|
"""
    
    # Add rows for each implementation
    for implementation_name, implementation_results in results.items():
        if "average_scores_by_complexity" not in implementation_results:
            continue
            
        scores_by_complexity = implementation_results["average_scores_by_complexity"]
        
        low = scores_by_complexity.get("low", 0)
        medium = scores_by_complexity.get("medium", 0)
        high = scores_by_complexity.get("high", 0)
        
        report += f"| {implementation_name.upper()} | {low:.2f} | {medium:.2f} | {high:.2f} |\n"
    
    # Add performance by query category
    report += """
## Performance by Query Category

Performance across different fitness-related query categories:

| Implementation | Exercise Technique | Workout Planning | Nutrition Advice | Progress Tracking | Injury Prevention | Exercise Science | Complex Planning |
|----------------|-------------------|-----------------|-----------------|------------------|------------------|-----------------|-----------------|
"""
    
    # Add rows for each implementation
    for implementation_name, implementation_results in results.items():
        if "average_scores_by_category" not in implementation_results:
            continue
            
        scores_by_category = implementation_results["average_scores_by_category"]
        
        exercise_technique = scores_by_category.get("exercise_technique", 0)
        workout_planning = scores_by_category.get("workout_planning", 0)
        nutrition_advice = scores_by_category.get("nutrition_advice", 0)
        progress_tracking = scores_by_category.get("progress_tracking", 0)
        injury_prevention = scores_by_category.get("injury_prevention", 0)
        exercise_science = scores_by_category.get("exercise_science", 0)
        complex_planning = scores_by_category.get("complex_planning", 0)
        
        report += f"| {implementation_name.upper()} | {exercise_technique:.2f} | {workout_planning:.2f} | {nutrition_advice:.2f} | {progress_tracking:.2f} | {injury_prevention:.2f} | {exercise_science:.2f} | {complex_planning:.2f} |\n"
    
    # Add implementation details
    report += """
## Implementation Details

### Naive RAG
- Basic vector similarity search
- Direct document retrieval
- Simple prompt construction

### Advanced RAG
- Query expansion using LLM
- Sentence-window retrieval for better context
- Re-ranking of retrieved documents
- Dynamic context window based on relevance
- Structured prompt engineering

### Modular RAG
- Query classification
- Specialized retrievers for different fitness topics
- Template-based responses

### Graph RAG
- Knowledge graph construction from fitness documents
- Graph-based retrieval using node relationships
- Path-aware context augmentation
- Relationship-enhanced prompting
- Multi-hop reasoning for complex queries

### RAPTOR RAG
- Query planning and decomposition
- Iterative, multi-step retrieval
- Reasoning over retrieved information
- Self-reflection and refinement
- Structured response synthesis

## Recommendations

"""
    
    # Add recommendations based on results
    if best_implementation == "naive":
        report += """
For the PersonalTrainerAI project, the Naive RAG implementation performed best overall. This suggests that:
- The fitness domain knowledge may be straightforward enough that simple retrieval works well
- The vector embeddings effectively capture the semantic meaning of fitness queries
- More complex implementations may be introducing unnecessary complexity

Consider using the Naive RAG implementation for production, but continue testing with real user queries to validate this choice.
"""
    elif best_implementation == "advanced":
        report += """
For the PersonalTrainerAI project, the Advanced RAG implementation performed best overall. This suggests that:
- The additional context from sentence-window retrieval improves answer quality
- Re-ranking helps prioritize the most relevant information
- Query expansion helps capture different aspects of fitness queries

The Advanced RAG implementation provides a good balance between complexity and performance and is recommended for production use.
"""
    elif best_implementation == "modular":
        report += """
For the PersonalTrainerAI project, the Modular RAG implementation performed best overall. This suggests that:
- Different fitness topics benefit from specialized retrieval approaches
- Template-based responses provide better structure for fitness advice
- Query classification effectively routes queries to the right processing pipeline

The Modular RAG implementation is recommended for production use, with ongoing refinement of the templates and classification logic.
"""
    elif best_implementation == "graph":
        report += """
For the PersonalTrainerAI project, the Graph RAG implementation performed best overall. This suggests that:
- The relationships between fitness concepts are important for providing comprehensive answers
- The knowledge graph effectively captures the interconnected nature of fitness knowledge
- Path-aware context augmentation provides valuable additional context

The Graph RAG implementation is recommended for production use, with ongoing refinement of the knowledge graph.
"""
    elif best_implementation == "raptor":
        report += """
For the PersonalTrainerAI project, the RAPTOR RAG implementation performed best overall. This suggests that:
- Complex fitness queries benefit from multi-step reasoning
- Breaking down queries into sub-questions improves retrieval quality
- Iterative retrieval and reasoning leads to more comprehensive answers

The RAPTOR RAG implementation is recommended for production use, particularly for complex queries that require reasoning across multiple fitness concepts.
"""
    
    # Add conclusion
    report += """
## Conclusion

The comparison of different RAG implementations shows that [key insights from the comparison]. 

For the PersonalTrainerAI project, we recommend using the best-performing implementation while considering the trade-offs between performance and computational requirements.

Regular evaluation with real user queries should be conducted to ensure the chosen implementation continues to meet user needs as the fitness knowledge base grows and evolves.
"""
    
    # Save report
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved comparison report to {report_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare RAG Implementations for PersonalTrainerAI')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--test-queries', type=str, help='Path to test queries JSON file')
    parser.add_argument('--generate-queries', type=int, default=0, help='Number of test queries to generate')
    parser.add_argument('--graph-path', type=str, default='fitness_knowledge_graph.json', help='Path to knowledge graph for Graph RAG')
    parser.add_argument('--build-graph', action='store_true', help='Whether to build the knowledge graph')
    
    args = parser.parse_args()
    
    # Compare RAG implementations
    compare_rag_implementations(
        output_dir=args.output_dir,
        test_queries_path=args.test_queries,
        generate_queries=args.generate_queries,
        graph_path=args.graph_path,
        build_graph=args.build_graph
    )
