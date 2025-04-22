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
import json # Ensure json is imported

# Import the metrics configuration (Assuming this file exists and is relevant)
# If mlflow_rag_metrics.py is not essential for the tracker class itself, consider removing
try:
    from .mlflow_rag_metrics import RAG_METRICS, ALL_METRICS, EXPERIMENT_PARAMS, EXPERIMENT_TAGS
except ImportError:
    logger = logging.getLogger("mlflow_rag_tracker")
    logger.warning("Could not import mlflow_rag_metrics. Using default tags.")
    EXPERIMENT_TAGS = {"project": "PersonalTrainerAI"} # Provide a default

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlflow_rag_tracker")

class MLflowRAGTracker:
    """
    MLflow tracker for RAG and Agent evaluation experiments.

    This class provides methods to track evaluation metrics, parameters,
    and artifacts in MLflow.
    """

    def __init__(
        self,
        experiment_name: str = "rag_evaluation",
        tracking_uri: str = "http://localhost:5000",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize the MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI of the MLflow tracking server
            artifact_location: Location to store artifacts (default: None, uses MLflow default)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.active_run = None
        self.experiment_id = None # Initialize experiment_id

        # Set up MLflow
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        if not self.tracking_uri:
            logger.warning("MLflow tracking URI is not set. Using default local tracking.")
            self.tracking_uri = mlflow.get_tracking_uri() # Use default or already set env var

        # Set the tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")

        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.info(f"Creating new experiment: {self.experiment_name}")
            try:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.artifact_location
                )
            except Exception as e:
                logger.error(f"Failed to create MLflow experiment '{self.experiment_name}': {e}", exc_info=True)
                raise # Re-raise the error - cannot proceed without experiment
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
        if self.active_run:
            logger.warning(f"An MLflow run (ID: {self.active_run.info.run_id}) is already active. Ending it before starting a new one.")
            self.end_run()

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"evaluation_{timestamp}"

        # Combine default tags with provided tags
        all_tags = EXPERIMENT_TAGS.copy() if 'EXPERIMENT_TAGS' in globals() else {}
        if tags:
            all_tags.update(tags)

        if not self.experiment_id:
             logger.error("Experiment ID is not set. Cannot start run.")
             raise ValueError("MLflow experiment not properly initialized.")


        logger.info(f"Starting MLflow run: {run_name} in experiment ID: {self.experiment_id}")
        try:
            self.active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=all_tags
            )
            logger.info(f"MLflow run started. Run ID: {self.active_run.info.run_id}")
            return self.active_run
        except Exception as e:
            logger.error(f"Failed to start MLflow run '{run_name}': {e}", exc_info=True)
            self.active_run = None # Ensure active_run is None if start fails
            raise

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the active MLflow run.

        Args:
            params: Dictionary of parameters to log
        """
        if self.active_run is None:
            logger.warning("No active run. Cannot log parameters. Please start a run first.")
            # Optional: Automatically start a run? Decided against it for clarity.
            # logger.warning("No active run. Starting a default run.")
            # self.start_run()
            return # Exit if no active run

        try:
            # Sanitize parameters: MLflow limits param value length
            sanitized_params = {}
            for key, value in params.items():
                str_value = str(value)
                if len(str_value) > 500: # MLflow UI might truncate anyway, but DB limit is higher
                    logger.warning(f"Parameter '{key}' value is too long ({len(str_value)} chars), truncating to 500.")
                    sanitized_params[key] = str_value[:500] + "..."
                else:
                    sanitized_params[key] = str_value # Ensure it's a string

            logger.info(f"Logging {len(sanitized_params)} parameters for run ID: {self.active_run.info.run_id}")
            mlflow.log_params(sanitized_params)
            logger.debug(f"Logged parameters: {sanitized_params}")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}", exc_info=True)


    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the active MLflow run. Converts non-float metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Step value for the metrics (default: None)
        """
        if self.active_run is None:
            logger.warning("No active run. Cannot log metrics. Please start a run first.")
            return

        # Attempt to convert metrics to floats
        float_metrics = {}
        for key, value in metrics.items():
            try:
                float_metrics[key] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert metric '{key}' with value '{value}' (type: {type(value)}) to float. Skipping.")

        if not float_metrics:
            logger.warning("No valid metrics could be converted to float for logging.")
            return

        try:
            logger.info(f"Logging {len(float_metrics)} metrics for run ID: {self.active_run.info.run_id}")
            mlflow.log_metrics(float_metrics, step=step)
            logger.debug(f"Logged metrics: {float_metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}", exc_info=True)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file or directory) to the active MLflow run.

        Args:
            local_path: Path to the local file or directory to log
            artifact_path: Optional destination path within the run's artifact URI.
        """
        if self.active_run is None:
            logger.warning("No active run. Cannot log artifact. Please start a run first.")
            return

        if not os.path.exists(local_path):
             logger.error(f"Cannot log artifact. Local path does not exist: {local_path}")
             return

        try:
            logger.info(f"Logging artifact from '{local_path}' to destination '{artifact_path or '.'}' for run ID: {self.active_run.info.run_id}")
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact '{local_path}': {e}", exc_info=True)


    def log_figure(self, figure, artifact_path: str) -> None:
        """
        Log a matplotlib or plotly figure to the active MLflow run.

        Args:
            figure: Matplotlib or Plotly figure object to log
            artifact_path: Path within the artifact directory to log the figure (e.g., "plots/my_figure.png")
        """
        if self.active_run is None:
            logger.warning("No active run. Cannot log figure. Please start a run first.")
            return

        try:
            logger.info(f"Logging figure to '{artifact_path}' for run ID: {self.active_run.info.run_id}")
            mlflow.log_figure(figure, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log figure '{artifact_path}': {e}", exc_info=True)

    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str) -> None:
        """
        Log a dictionary as a JSON artifact to the active MLflow run.

        Args:
            dictionary: Dictionary to log
            artifact_path: Path within the artifact directory to log the dictionary (e.g., "data/results.json")
        """
        if self.active_run is None:
            logger.warning("No active run. Cannot log dictionary. Please start a run first.")
            return

        try:
            logger.info(f"Logging dictionary to '{artifact_path}' for run ID: {self.active_run.info.run_id}")
            mlflow.log_dict(dictionary, artifact_path)
        except Exception as e:
            # Try logging with default=str for non-serializable types
            try:
                logger.warning(f"Initial attempt to log dict failed: {e}. Retrying with default=str serialization.")
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_file:
                    json.dump(dictionary, tmp_file, indent=2, default=str) # Use default=str
                    tmp_path = tmp_file.name
                mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
                os.remove(tmp_path) # Clean up temp file
                logger.info(f"Successfully logged dictionary to '{artifact_path}' using default=str fallback.")
            except Exception as e2:
                logger.error(f"Failed to log dictionary '{artifact_path}' even with fallback: {e2}", exc_info=True)


    def log_rag_evaluation_results(
        self,
        results: Dict[str, Any],
        implementation_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log RAG evaluation results to MLflow. (Handles run start/end)

        Args:
            results: Dictionary of evaluation results (potentially nested)
            implementation_name: Name of the RAG implementation
            parameters: Additional parameters specific to this RAG run
        """
        # Start a run with the implementation name
        run_name = f"{implementation_name}_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_run(run_name=run_name, tags={"evaluation_type": "rag", "rag_implementation": implementation_name})

        try:
            # Log implementation name as a parameter
            base_params = {"rag_implementation": implementation_name}
            if parameters:
                # Ensure keys don't clash and update
                parameters = {k: v for k, v in parameters.items() if k != "rag_implementation"}
                base_params.update(parameters)
            self.log_parameters(base_params)

            # Extract and log metrics - Adapt based on the actual structure of 'results'
            metrics_to_log = {}

            # Example: If results contain an 'average_scores' dictionary
            if isinstance(results.get("average_scores"), dict):
                for metric_name, score in results["average_scores"].items():
                    metrics_to_log[f"avg_{metric_name}"] = score
                logger.info("Logging metrics from 'average_scores' dict.")

            # Example: If results contain RAGAS metrics directly
            elif isinstance(results.get("ragas_metrics"), dict):
                 for metric_name, score in results["ragas_metrics"].items():
                     metrics_to_log[f"ragas_{metric_name}"] = score
                 logger.info("Logging metrics from 'ragas_metrics' dict.")

            # Example: If top-level keys in results are metrics
            else:
                 for key, value in results.items():
                      if isinstance(value, (int, float)):
                           metrics_to_log[key] = value
                 if metrics_to_log:
                    logger.info("Logging top-level numeric values as metrics.")


            # Log overall score if explicitly present
            if "overall_score" in results and isinstance(results["overall_score"], (int, float)):
                metrics_to_log["overall_score"] = results["overall_score"]
            elif "overall" in metrics_to_log: # Handle case from AdvancedRAGEvaluator
                 metrics_to_log["overall_score"] = metrics_to_log["overall"]


            if metrics_to_log:
                self.log_metrics(metrics_to_log)
            else:
                logger.warning(f"Could not extract standard metrics from results structure for {implementation_name}.")


            # Log the full results dictionary (excluding figures potentially)
            results_for_logging = results.copy()
            figures = results_for_logging.pop("figures", None) # Remove figures if they exist

             # Try logging the main results dict
            self.log_dict(results_for_logging, "evaluation_results.json")


            # Log any figures if available
            if isinstance(figures, dict):
                for fig_name, fig in figures.items():
                    self.log_figure(fig, f"figures/{fig_name}.png")

        except Exception as e:
             logger.error(f"Error during RAG evaluation logging for run '{run_name}': {e}", exc_info=True)
             # Ensure run ends even if logging fails mid-way
             self.end_run()
             raise # Re-raise the exception after attempting to end the run
        else:
            # End the run successfully
            self.end_run()
            logger.info(f"Successfully logged RAG evaluation run '{run_name}' to MLflow.")


    # <<< ADDED METHOD START >>>
    def log_agent_evaluation_results(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        judgements: Optional[List[Dict[str, Any]]] = None,
        judgements_artifact_name: str = "llm_run_judgements.json"
    ) -> None:
        """
        Logs the results of an agent evaluation run (based on LangSmith feedback) to MLflow.

        Args:
            run_name (str): A descriptive name for this specific MLflow run.
            parameters (dict): Dictionary of parameters used for the agent evaluation setup.
            metrics (dict): Dictionary of summary metrics (e.g., average score).
            judgements (list, optional): List of individual judgement dicts ({score, reasoning})
                                         to be logged as an artifact. Defaults to None.
            judgements_artifact_name (str, optional): Filename for the judgements artifact.
                                                      Defaults to "llm_run_judgements.json".
        """
        # Start a new run for this agent evaluation
        self.start_run(run_name=run_name, tags={"evaluation_type": "agent"}) # Add a tag

        try:
            # Log Parameters
            self.log_parameters(parameters)

            # Log Metrics
            self.log_metrics(metrics)

            # Log Judgements Artifact if provided
            if judgements:
                logger.info(f"Logging {len(judgements)} individual judgements to artifact '{judgements_artifact_name}'")
                # Ensure judgements is serializable before logging
                serializable_judgements = []
                for j in judgements:
                     if isinstance(j, dict):
                          serializable_judgements.append({k: str(v) for k, v in j.items()}) # Convert values to str just in case
                     else:
                          logger.warning(f"Skipping non-dict item in judgements list: {j}")

                if serializable_judgements:
                     self.log_dict(
                         dictionary={"judgements": serializable_judgements}, # Wrap list in a dict for log_dict
                         artifact_path=judgements_artifact_name
                     )
                else:
                     logger.warning("No valid judgements found to log after serialization attempt.")

            else:
                logger.info("No individual judgements provided to log as artifact.")

        except Exception as e:
            logger.error(f"Error during agent evaluation logging for run '{run_name}': {e}", exc_info=True)
            # Ensure run ends even if logging fails mid-way
            self.end_run()
            raise # Re-raise the exception after attempting to end the run
        else:
            # End the run successfully
            self.end_run()
            logger.info(f"Successfully logged agent evaluation run '{run_name}' to MLflow.")
    # <<< ADDED METHOD END >>>


    def end_run(self) -> None:
        """End the active MLflow run."""
        if self.active_run:
            run_id = self.active_run.info.run_id
            logger.info(f"Ending MLflow run ID: {run_id}")
            try:
                mlflow.end_run()
                self.active_run = None
            except Exception as e:
                logger.error(f"Failed to end MLflow run {run_id}: {e}", exc_info=True)
                # Consider setting self.active_run to None even if end fails
                self.active_run = None
        else:
            logger.debug("No active MLflow run to end.")


# Utility function - Keep outside the class
def start_mlflow_server(host="127.0.0.1", port=5000, backend_store_uri="mlruns", artifacts_destination="mlartifacts"):
    """
    Start a local MLflow tracking server.

    Args:
        host: Host to bind to
        port: Port to bind to
        backend_store_uri: URI for the backend store (default: ./mlruns relative to cwd)
        artifacts_destination: Location for artifacts (default: ./mlartifacts relative to cwd)

    Returns:
        Process ID of the MLflow server
    """
    import subprocess
    import atexit

    # Ensure directories exist
    os.makedirs(backend_store_uri, exist_ok=True)
    os.makedirs(artifacts_destination, exist_ok=True)

    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", f"sqlite:///{os.path.abspath(backend_store_uri)}/mlflow.db", # Use file-based sqlite
        "--default-artifact-root", os.path.abspath(artifacts_destination)
    ]

    logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")
    try:
        # Start the process without capturing output streams immediately
        process = subprocess.Popen(cmd) # stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Avoid capturing streams if causing issues

        logger.info(f"MLflow server started with PID: {process.pid}")

        # Register a function to terminate the server on exit
        def cleanup():
            logger.info(f"Terminating MLflow server (PID: {process.pid})...")
            try:
                process.terminate()
                process.wait(timeout=5) # Wait for graceful termination
                logger.info("MLflow server terminated.")
            except subprocess.TimeoutExpired:
                logger.warning("MLflow server did not terminate gracefully, killing...")
                process.kill()
                logger.info("MLflow server killed.")
            except Exception as kill_err:
                 logger.error(f"Error during MLflow server cleanup: {kill_err}")

        atexit.register(cleanup)
        return process.pid

    except FileNotFoundError:
         logger.error("Failed to start MLflow server: 'mlflow' command not found. Is MLflow installed and in PATH?")
         return None
    except Exception as e:
        logger.error(f"Failed to start MLflow server: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Example usage for starting the server
    print("Starting MLflow server...")
    pid = start_mlflow_server()

    if pid:
        print(f"MLflow server started with PID: {pid}")
        print("MLflow UI should be available at: http://localhost:5000")
        print("The server runs in the background.")
        print("This script will now exit, but the server will continue.")
        print("Press Ctrl+C in the terminal where the server process *actually* runs if needed, or terminate the PID.")
        # Keep the main script running if you want to block here instead
        # try:
        #     print("Press Ctrl+C here to stop THIS script (server termination handled by atexit).")
        #     import time
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     print("\nScript interrupted.")
    else:
        print("Failed to start MLflow server.")