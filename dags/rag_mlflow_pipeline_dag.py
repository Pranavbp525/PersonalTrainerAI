# dags/rag_evaluation_dag.py

from __future__ import annotations

import pendulum
import os
import logging
import traceback
from datetime import timedelta

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

# --- Configuration ---
# Determine project root relative to the DAG file location within Airflow
# This assumes dags/ is one level below the project root mapped in docker-compose
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "rag_evaluation_output")

# Set up basic logging for the DAG file itself
log = logging.getLogger(__name__)

# Add src directory to Python path for worker imports
# Note: Setting PYTHONPATH in Docker environment is preferred
import sys
if SRC_DIR not in sys.path:
    log.info(f"Adding {SRC_DIR} to sys.path for DAG definition")
    sys.path.insert(0, SRC_DIR)

# --- Define Task Functions ---

def run_advanced_rag_evaluation_task(**context):
    """
    Runs the advanced RAG evaluation script's comparison functionality.
    MLflow tracking is handled internally by the AdvancedRAGEvaluator/MLflowRAGTracker.
    """
    log.info("Starting RAG Evaluation Task...")
    log.info(f"Using Project Root: {PROJECT_ROOT}")
    log.info(f"Output Directory: {OUTPUT_DIR}")

    # --- Environment Checks ---
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # Add checks for other keys if needed (e.g., PINECONE_API_KEY)
    # pinecone_api_key = os.getenv('PINECONE_API_KEY')
    # pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

    missing_vars = []
    if not mlflow_tracking_uri:
        missing_vars.append("MLFLOW_TRACKING_URI")
    if not openai_api_key:
        missing_vars.append("OPENAI_API_KEY")
    # if not pinecone_api_key: missing_vars.append("PINECONE_API_KEY")
    # if not pinecone_env: missing_vars.append("PINECONE_ENVIRONMENT")

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        log.error(error_msg)
        raise ValueError(error_msg)
    else:
        log.info("Required environment variables (MLFLOW_TRACKING_URI, OPENAI_API_KEY) are set.")

    # --- Ensure Output Directory Exists ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log.info(f"Ensured output directory exists: {OUTPUT_DIR}")
    except OSError as e:
        log.error(f"Could not create output directory {OUTPUT_DIR}: {e}")
        raise

    # --- Execute Evaluation ---
    try:
        # Import the class *inside* the task function
        # This ensures it runs in the worker's context with the correct path
        from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator

        log.info("Initializing AdvancedRAGEvaluator...")
        # Instantiate the evaluator
        # Pass parameters if needed, otherwise rely on its defaults
        evaluator = AdvancedRAGEvaluator(
            output_dir=OUTPUT_DIR,
            # test_queries can use default
            # evaluation_llm_model can use default or get from env var/Airflow variable
            # embedding_model can use default or get from env var/Airflow variable
        )

        log.info("Running comparison of RAG implementations...")
        # Call the comparison method which should handle evaluations and logging
        comparison_results = evaluator.compare_implementations()

        log.info("RAG Implementation Comparison Finished.")
        # Optionally log key results to Airflow logs
        if comparison_results.get("best_implementation"):
             log.info(f"Best Implementation determined: {comparison_results['best_implementation']}")
             log.info(f"Best Overall Score: {comparison_results.get('best_score', 'N/A'):.2f}")
        else:
             log.warning("Could not determine best implementation from results.")

        # Print summary to Airflow logs
        evaluator.print_summary(comparison_results)

        return True # Indicate success

    except ImportError as e:
        log.error(f"Import Error during RAG evaluation: {e}")
        log.error("Ensure 'src' directory is in PYTHONPATH in the Airflow worker environment.")
        log.error(f"Current sys.path: {sys.path}")
        raise
    except Exception as e:
        log.error(f"An error occurred during RAG evaluation: {e}")
        log.error(traceback.format_exc()) # Log full traceback for debugging
        raise

# --- Define DAG ---
with DAG(
    dag_id='rag_evaluation_pipeline',
    default_args={
        'owner': 'vinyasnaidu',
        'start_date': pendulum.datetime(2024, 1, 1, tz="UTC"),
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=2),
        'catchup': False,
        'email_on_failure': False, # Set as needed
        'email_on_retry': False,
    },
    description='Runs RAG evaluation using MLflow after data pipeline completion.',
    schedule=None,
    tags=['rag', 'evaluation', 'mlflow'],
) as dag:

    # Task 1: Wait for the Data_pipeline_dag
    wait_for_data_pipeline = ExternalTaskSensor(
        task_id='wait_for_data_pipeline_completion',
        external_dag_id='Data_pipeline_dag', # Match the DAG ID of your data pipeline
        external_task_id=None, # Wait for whole DAG
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='poke',
        timeout=60 * 60 * 2, # 2 hours timeout
        poke_interval=60, # Check every 1 minute
    )

    # Task 2: Run RAG Evaluation Comparison
    run_evaluation_comparison = PythonOperator(
        task_id='run_rag_evaluation_comparison',
        python_callable=run_advanced_rag_evaluation_task,
        # provide_context=True # Uncomment if you need access to Airflow context vars like {{ ds }}
    )

    # Define task dependencies
    wait_for_data_pipeline >> run_evaluation_comparison