"""
MLflow RAG Tracker for PersonalTrainerAI

This module implements MLflow tracking for RAG evaluation experiments.
It integrates with the existing evaluation framework in advanced_rag_evaluation.py.
"""

import os
import sys
import mlflow
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the metrics configuration
from mlflow_rag_metrics import RAG_METRICS, ALL_METRICS, EXPERIMENT_PARAMS, EXPERIMENT_TAGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlflow_rag_tracker")

class MLflowRAGTracker:
    """
    MLflow tracker for RAG evaluation experiments.
    
    This class provides methods to track RAG evaluation metrics, parameters,
    and artifacts in MLflow. It integrates with the existing evaluation framework.
    """
    
    def __init__(
        self,
        experiment_name: str = "rag_evaluation",
        tracking_uri: str = "http://localhost:5000",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize the MLflow RAG tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI of the MLflow tracking server
            artifact_location: Location to store artifacts (default: None, uses MLflow default)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.active_run = None
        
        # Set up MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        # Set the tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.info(f"Creating new experiment: {self.experiment_name}")
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
        
        self.experiment_id = experiment_id
    
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name of the run (default: None, uses timestamp)
            tags: Additional tags for the run (default: None)
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"rag_evaluation_{timestamp}"
        
        # Combine default tags with provided tags
        all_tags = EXPERIMENT_TAGS.copy()
        if tags:
            all_tags.update(tags)
        
        logger.info(f"Starting MLflow run: {run_name}")
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=all_tags
        )
        return self.active_run
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the active MLflow run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.active_run is None:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        logger.info(f"Logging {len(params)} parameters")
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the active MLflow run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step value for the metrics (default: None)
        """
        if self.active_run is None:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        logger.info(f"Logging {len(metrics)} metrics")
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str) -> None:
        """
        Log an artifact to the active MLflow run.
        
        Args:
            local_path: Path to the local file to log
        """
        if self.active_run is None:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        logger.info(f"Logging artifact: {local_path}")
        mlflow.log_artifact(local_path)
    
    def log_figure(self, figure, artifact_path: str) -> None:
        """
        Log a matplotlib figure to the active MLflow run.
        
        Args:
            figure: Matplotlib figure to log
            artifact_path: Path within the artifact directory to log the figure
        """
        if self.active_run is None:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        logger.info(f"Logging figure to {artifact_path}")
        mlflow.log_figure(figure, artifact_path)
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str) -> None:
        """
        Log a dictionary as a JSON artifact to the active MLflow run.
        
        Args:
            dictionary: Dictionary to log
            artifact_path: Path within the artifact directory to log the dictionary
        """
        if self.active_run is None:
            logger.warning("No active run. Starting a new run.")
            self.start_run()
        
        logger.info(f"Logging dictionary to {artifact_path}")
        mlflow.log_dict(dictionary, artifact_path)
    
    def log_rag_evaluation_results(
        self, 
        results: Dict[str, Any], 
        implementation_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log RAG evaluation results to MLflow.
        
        Args:
            results: Dictionary of evaluation results
            implementation_name: Name of the RAG implementation
            parameters: Additional parameters to log (default: None)
        """
        # Start a run with the implementation name
        self.start_run(run_name=f"{implementation_name}_evaluation")
        
        # Log implementation name as a parameter
        self.log_parameters({"rag_implementation": implementation_name})
        
        # Log additional parameters if provided
        if parameters:
            self.log_parameters(parameters)
        
        # Extract and log metrics
        metrics = {}
        
        # Process each metric category
        for category, metric_names in RAG_METRICS.items():
            if category in results:
                category_results = results[category]
                for metric_name in metric_names:
                    if metric_name in category_results:
                        metrics[f"{category}.{metric_name}"] = category_results[metric_name]
        
        # Log overall metrics if available
        if "overall_score" in results:
            metrics["overall_score"] = results["overall_score"]
        
        # Log the metrics
        self.log_metrics(metrics)
        
        # Create a copy of results without non-serializable entries
        results_for_logging = results.copy()
        if "figures" in results_for_logging:
            results_for_logging.pop("figures")

        # Log the full results as a JSON artifact
        self.log_dict(results_for_logging, "evaluation_results.json")
        
        # Log any figures if available
        if "figures" in results:
            for fig_name, fig in results["figures"].items():
                self.log_figure(fig, f"figures/{fig_name}.png")
        
        # End the run
        self.end_run()
    
    def end_run(self) -> None:
        """End the active MLflow run."""
        if self.active_run:
            logger.info("Ending MLflow run")
            mlflow.end_run()
            self.active_run = None
        else:
            logger.warning("No active run to end")

def start_mlflow_server(host="127.0.0.1", port=5000, backend_store_uri="./mlruns"):
    """
    Start a local MLflow tracking server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        backend_store_uri: URI for the backend store
    
    Returns:
        Process ID of the MLflow server
    """
    import subprocess
    import atexit
    
    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri
    ]
    
    logger.info(f"Starting MLflow server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    
    # Register a function to terminate the server on exit
    def cleanup():
        logger.info("Terminating MLflow server")
        process.terminate()
        process.wait()
    
    atexit.register(cleanup)
    
    return process.pid

if __name__ == "__main__":
    # Example usage
    print("Starting MLflow server...")
    pid = start_mlflow_server()
    print(f"MLflow server started with PID: {pid}")
    print("MLflow UI available at: http://localhost:5000")
    
    # Keep the script running to keep the server alive
    try:
        print("Press Ctrl+C to stop the server")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping MLflow server...")
