"""
Test script for MLflow RAG evaluation integration

This script tests the MLflow integration with a simplified RAG evaluation
to verify that everything is working correctly.
"""

import os
import sys
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MLflow tracker
from mlflow_rag_tracker import MLflowRAGTracker, start_mlflow_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_mlflow_integration")

def generate_mock_evaluation_results(implementation_name):
    """
    Generate mock evaluation results for testing.
    
    Args:
        implementation_name: Name of the RAG implementation
    
    Returns:
        Dictionary of mock evaluation results
    """
    # Generate random scores based on implementation name for consistent results
    np.random.seed(hash(implementation_name) % 2**32)
    
    # Generate RAGAS metrics
    ragas_metrics = {
        "faithfulness": np.random.uniform(0.7, 0.95),
        "answer_relevancy": np.random.uniform(0.7, 0.95),
        "context_relevancy": np.random.uniform(0.7, 0.95),
        "context_precision": np.random.uniform(0.7, 0.95)
    }
    
    # Generate custom metrics
    custom_metrics = {
        "fitness_domain_accuracy": np.random.uniform(0.7, 0.95),
        "scientific_correctness": np.random.uniform(0.7, 0.95),
        "practical_applicability": np.random.uniform(0.7, 0.95),
        "safety_consideration": np.random.uniform(0.7, 0.95)
    }
    
    # Generate retrieval metrics
    retrieval_metrics = {
        "retrieval_precision": np.random.uniform(0.7, 0.95),
        "retrieval_recall": np.random.uniform(0.7, 0.95)
    }
    
    # Generate human evaluation metrics
    human_eval_metrics = {
        "answer_completeness": np.random.uniform(0.7, 0.95),
        "answer_conciseness": np.random.uniform(0.7, 0.95),
        "answer_helpfulness": np.random.uniform(0.7, 0.95)
    }
    
    # Calculate overall score
    all_metrics = list(ragas_metrics.values()) + list(custom_metrics.values()) + \
                 list(retrieval_metrics.values()) + list(human_eval_metrics.values())
    overall_score = sum(all_metrics) / len(all_metrics)
    
    # Create a sample figure
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(ragas_metrics.keys())
    values = list(ragas_metrics.values())
    
    ax.bar(categories, values)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{implementation_name} RAG - RAGAS Metrics")
    ax.set_ylabel("Score")
    
    # Return the results
    return {
        "ragas_metrics": ragas_metrics,
        "custom_metrics": custom_metrics,
        "retrieval_metrics": retrieval_metrics,
        "human_eval_metrics": human_eval_metrics,
        "overall_score": overall_score,
        "figures": {
            "ragas_metrics_bar": fig
        }
    }
    plt.close(fig)

def test_mlflow_integration():
    """Test the MLflow integration with mock evaluation results."""
    # Create test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
    os.makedirs(test_dir, exist_ok=True)
    
    # Start MLflow server
    logger.info("Starting MLflow server...")
    mlflow_pid = start_mlflow_server(
        backend_store_uri=f"file://{os.path.abspath(os.path.join(test_dir, 'mlruns'))}"
    )
    if mlflow_pid is None:
        logger.error("Failed to start MLflow server. Please check the configuration.")
        return
    logger.info(f"MLflow server started with PID: {mlflow_pid}")
    
    # Initialize MLflow tracker
    tracker = MLflowRAGTracker(
    experiment_name="test_rag_evaluation",
    tracking_uri="http://localhost:5000",
    artifact_location=f"file://{os.path.abspath(os.path.join(test_dir, 'artifacts'))}"
    )
    
    # Test RAG implementations
    implementations = ["advanced", "modular", "raptor"]
    
    # Dictionary to store all evaluation results
    all_results = {}
    
    # Evaluate each RAG implementation
    for implementation_name in implementations:
        logger.info(f"Testing MLflow tracking for {implementation_name} RAG implementation")
        
        # Generate mock evaluation results
        results = generate_mock_evaluation_results(implementation_name)
        all_results[implementation_name] = results
        
        # Mock parameters
        parameters = {
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-3.5-turbo",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 5,
            "temperature": 0.7
        }
        
        # Log the results to MLflow
        tracker.log_rag_evaluation_results(
            results=results,
            implementation_name=implementation_name,
            parameters=parameters
        )
        
        logger.info(f"MLflow tracking for {implementation_name} RAG implementation completed")
    
    # Save all results to a JSON file
    with open(os.path.join(test_dir, "test_results.json"), "w") as f:
        # Convert numpy values to float for JSON serialization
        serializable_results = {}
        for impl, res in all_results.items():
            serializable_results[impl] = {
                k: v if k != "figures" else "figure_object" 
                for k, v in res.items()
            }
            
            # Convert numpy values to float
            for category in ["ragas_metrics", "custom_metrics", "retrieval_metrics", "human_eval_metrics"]:
                if category in serializable_results[impl]:
                    serializable_results[impl][category] = {
                        k: float(v) for k, v in serializable_results[impl][category].items()
                    }
            
            if "overall_score" in serializable_results[impl]:
                serializable_results[impl]["overall_score"] = float(serializable_results[impl]["overall_score"])
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Test results saved to {os.path.join(test_dir, 'test_results.json')}")
    logger.info("MLflow integration test completed successfully")
    logger.info("MLflow UI available at: http://localhost:5000")
    logger.info("Press Ctrl+C to stop the MLflow server")
    
    try:
        # Keep the script running to keep the server alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping MLflow server...")

if __name__ == "__main__":
    test_mlflow_integration()
