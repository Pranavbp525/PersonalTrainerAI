"""
RAG Evaluation with MLflow Tracking for PersonalTrainerAI

This script demonstrates how to use MLflow to track RAG evaluation experiments.
It integrates with the existing advanced_rag_evaluation.py framework.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import MLflow tracker
from mlflow_rag_tracker import MLflowRAGTracker, start_mlflow_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_evaluation_mlflow")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG Evaluation with MLflow Tracking")
    
    parser.add_argument(
        "--rag-implementations",
        type=str,
        nargs="+",
        default=["advanced", "modular", "raptor"],
        help="RAG implementations to evaluate (default: advanced modular raptor)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results)"
    )
    
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: http://localhost:5000)"
    )
    
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="rag_evaluation",
        help="MLflow experiment name (default: rag_evaluation)"
    )
    
    parser.add_argument(
        "--start-mlflow-server",
        action="store_true",
        help="Start a local MLflow server"
    )
    
    parser.add_argument(
        "--test-queries",
        type=str,
        default=None,
        help="Path to JSON file with test queries (default: use built-in queries)"
    )
    
    return parser.parse_args()

def import_rag_implementation(implementation_name):
    """
    Dynamically import a RAG implementation class.
    
    Args:
        implementation_name: Name of the RAG implementation
    
    Returns:
        The imported RAG implementation class
    """
    try:
        if implementation_name == "advanced":
            from src.rag_model.advanced_rag import AdvancedRAG
            return AdvancedRAG
        elif implementation_name == "modular":
            from src.rag_model.modular_rag import ModularRAG
            return ModularRAG
        elif implementation_name == "raptor":
            from src.rag_model.raptor_rag import RaptorRAG
            return RaptorRAG
        else:
            raise ImportError(f"Unknown RAG implementation: {implementation_name}")
    except ImportError as e:
        logger.error(f"Failed to import {implementation_name} RAG implementation: {e}")
        return None

def run_evaluation_with_mlflow(args):
    """
    Run RAG evaluation with MLflow tracking.
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start MLflow server if requested
    if args.start_mlflow_server:
        mlflow_pid = start_mlflow_server(
            backend_store_uri=os.path.join(args.output_dir, "mlruns")
        )
        logger.info(f"MLflow server started with PID: {mlflow_pid}")
        logger.info(f"MLflow UI available at: {args.mlflow_tracking_uri}")
    
    # Initialize MLflow tracker
    tracker = MLflowRAGTracker(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_location=os.path.join(args.output_dir, "artifacts")
    )
    
    # Load test queries if provided
    test_queries = None
    if args.test_queries:
        try:
            with open(args.test_queries, 'r') as f:
                test_queries = json.load(f)
            logger.info(f"Loaded {len(test_queries)} test queries from {args.test_queries}")
        except Exception as e:
            logger.error(f"Failed to load test queries: {e}")
    
    # Import the advanced RAG evaluation module
    try:
        from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator
        logger.info("Successfully imported AdvancedRAGEvaluator")
    except ImportError as e:
        logger.error(f"Failed to import AdvancedRAGEvaluator: {e}")
        logger.error("Make sure you're running this script from the project root")
        return
    
    # Initialize the evaluator
    evaluator = AdvancedRAGEvaluator(
        output_dir=args.output_dir,
        test_queries=test_queries
    )
    
    # Dictionary to store all evaluation results
    all_results = {}
    
    # Evaluate each RAG implementation
    for implementation_name in args.rag_implementations:
        logger.info(f"Evaluating {implementation_name} RAG implementation")
        
        # Import the RAG implementation
        rag_class = import_rag_implementation(implementation_name)
        if rag_class is None:
            logger.warning(f"Skipping {implementation_name} RAG implementation")
            continue
        
        # Initialize the RAG implementation
        try:
            rag_instance = rag_class()
            
            # Extract parameters from the RAG implementation
            parameters = {
                "embedding_model": getattr(rag_instance, "embedding_model", "unknown"),
                "llm_model": getattr(rag_instance, "llm_model", "unknown"),
                "chunk_size": getattr(rag_instance, "chunk_size", 0),
                "chunk_overlap": getattr(rag_instance, "chunk_overlap", 0),
                "retrieval_k": getattr(rag_instance, "retrieval_k", 0),
                "temperature": getattr(rag_instance, "temperature", 0.0)
            }
            
            # Evaluate the RAG implementation
            results = evaluator.evaluate_implementation(rag_instance)
            
            # Store the results
            all_results[implementation_name] = results
            
            # Log the results to MLflow
            tracker.log_rag_evaluation_results(
                results=results,
                implementation_name=implementation_name,
                parameters=parameters
            )
            
            logger.info(f"Evaluation of {implementation_name} RAG implementation completed")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {implementation_name} RAG implementation: {e}")
    
    # Create comparison visualizations
    if len(all_results) > 1:
        logger.info("Creating comparison visualizations")
        
        # Start a new MLflow run for comparisons
        tracker.start_run(run_name="rag_implementations_comparison")
        
        try:
            # Create a comparison dataframe
            comparison_data = []
            for impl_name, results in all_results.items():
                row = {"implementation": impl_name}
                
                # Add overall score if available
                if "overall_score" in results:
                    row["overall_score"] = results["overall_score"]
                
                # Add RAGAS metrics
                if "ragas_metrics" in results:
                    for metric, value in results["ragas_metrics"].items():
                        row[f"ragas_{metric}"] = value
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save comparison dataframe
            comparison_csv = os.path.join(args.output_dir, "rag_comparison.csv")
            comparison_df.to_csv(comparison_csv, index=False)
            tracker.log_artifact(comparison_csv)
            
            # Create bar chart for overall scores
            if "overall_score" in comparison_df.columns:
                plt.figure(figsize=(10, 6))
                ax = comparison_df.plot.bar(x="implementation", y="overall_score", rot=0)
                ax.set_title("Overall RAG Performance Comparison")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1.0)
                
                # Save the figure
                overall_score_fig = os.path.join(args.output_dir, "overall_score_comparison.png")
                plt.tight_layout()
                plt.savefig(overall_score_fig)
                tracker.log_artifact(overall_score_fig)
            
            # Create radar chart for RAGAS metrics
            ragas_cols = [col for col in comparison_df.columns if col.startswith("ragas_")]
            if ragas_cols:
                from matplotlib.path import Path as MplPath
                from matplotlib.spines import Spine
                from matplotlib.transforms import Affine2D
                
                def radar_factory(num_vars, frame='circle'):
                    """Create a radar chart with `num_vars` axes."""
                    # Calculate evenly-spaced axis angles
                    theta = 2*3.1415926*np.linspace(0, 1-1./num_vars, num_vars)
                    
                    # Create figure
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='polar')
                    ax.set_theta_offset(3.1415926/2)
                    ax.set_theta_direction(-1)
                    
                    # Draw axis lines
                    ax.set_thetagrids(np.degrees(theta), labels=[col.replace("ragas_", "") for col in ragas_cols])
                    
                    return fig, ax
                
                # Create radar chart
                import numpy as np
                fig, ax = radar_factory(len(ragas_cols))
                
                # Plot each implementation
                for i, row in comparison_df.iterrows():
                    values = [row[col] for col in ragas_cols]
                    values += values[:1]  # Close the loop
                    angles = np.linspace(0, 2*3.1415926, len(values))
                    ax.plot(angles, values, linewidth=2, label=row["implementation"])
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_ylim(0, 1)
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                # Save the figure
                radar_fig = os.path.join(args.output_dir, "ragas_metrics_comparison.png")
                plt.tight_layout()
                plt.savefig(radar_fig)
                tracker.log_artifact(radar_fig)
            
            logger.info("Comparison visualizations created")
            
        except Exception as e:
            logger.error(f"Failed to create comparison visualizations: {e}")
        
        # End the comparison run
        tracker.end_run()
    
    logger.info("RAG evaluation with MLflow tracking completed")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"MLflow UI available at: {args.mlflow_tracking_uri}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluation_with_mlflow(args)
