# dags/model_evaluation_pipeline_dag.py

from __future__ import annotations
import pendulum
import os
import logging
import traceback
from datetime import timedelta
import sys # Keep sys import

# Import necessary Airflow classes FIRST
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
# from airflow.operators.empty import EmptyOperator # No longer needed

# --- Configuration ---
# Use the project root defined by the docker-compose volume mount
PROJECT_ROOT = "/opt/airflow/app" # <<< Correct project root inside container
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# --- Logging Setup ---
# Use standard Airflow logging configuration provided by task handlers
log = logging.getLogger(__name__)

# --- Add src directory to Python path ---
# Ensure this path is correct based on docker-compose volume mount
if SRC_DIR not in sys.path:
    log.info(f"Adding {SRC_DIR} to sys.path for DAG definition/worker.")
    sys.path.insert(0, SRC_DIR)


# --- REMOVED Top-Level Imports of Task Functions ---

# --- Wrapper function for RAG Evaluation Task ---
def run_rag_eval_wrapper(**context):
    """
    Sets up and runs the Advanced RAG Evaluator.
    Imports happen inside the task execution on the worker.
    Relies on environment variables being set via docker-compose (.env mount).
    """
    # Import necessary module INSIDE the function
    # Ensure the path is correct relative to SRC_DIR
    from rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator

    task_log = logging.getLogger("airflow.task") # Get task logger
    task_log.info("--- Starting RAG Evaluation Task ---")

    required_vars = ["MLFLOW_TRACKING_URI", "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"] # Added PINECONE_INDEX_NAME
    missing_vars = [k for k in required_vars if not os.getenv(k)]
    if missing_vars:
        error_msg = f"RAG Eval Task Failed: Missing required environment variables: {', '.join(missing_vars)}."
        task_log.error(error_msg)
        raise ValueError(error_msg)

    task_log.info("RAG Eval Task: Required environment variables appear to be present.")
    task_log.info(f"Using MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    task_log.info(f"Targeting Pinecone Index: {os.getenv('PINECONE_INDEX_NAME')}")


    try:
        temp_output_dir = "/tmp/rag_eval_output" # Define a temporary directory within the container
        os.makedirs(temp_output_dir, exist_ok=True)
        task_log.info(f"Using temp output directory: {temp_output_dir}")

        task_log.info("Initializing AdvancedRAGEvaluator...")
        # Ensure AdvancedRAGEvaluator uses env vars correctly or accepts them
        evaluator = AdvancedRAGEvaluator(output_dir=temp_output_dir)

        task_log.info("Running comparison of RAG implementations...")
        # Ensure compare_implementations uses env vars for Pinecone/MLflow/OpenAI keys+index
        comparison_results = evaluator.compare_implementations()
        task_log.info(f"RAG Implementation Comparison Finished. Results: {comparison_results}") # Log results

        # Optional: Add more robust result checking if needed
        if comparison_results is None:
             task_log.warning("RAG evaluation comparison returned None.")
        elif isinstance(comparison_results, dict) and comparison_results.get("status") == "error":
             task_log.error(f"RAG evaluation failed with error: {comparison_results.get('message', 'Unknown error')}")
             raise RuntimeError(f"RAG Evaluation failed: {comparison_results.get('message', 'Unknown error')}")

        task_log.info("--- RAG Evaluation Task Completed Successfully ---")

    except Exception as e:
        task_log.error(f"An error occurred during RAG evaluation task: {e}", exc_info=True)
        raise # Re-raise exception to fail the Airflow task

# --- Wrapper function for Agent Evaluation Task ---
def run_agent_eval_wrapper(**context):
    """
    Runs the Agent evaluation script.
    Imports happen inside the task execution on the worker.
    Relies on environment variables being set via docker-compose (.env mount).
    """
    # Import necessary module INSIDE the function
    # Ensure the path is correct relative to SRC_DIR
    from chatbot.agent_eval.eval import evaluate_agent

    task_log = logging.getLogger("airflow.task") # Get task logger
    task_log.info("--- Starting Agent Evaluation Task ---")

    required_vars = ["MLFLOW_TRACKING_URI", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "OPENAI_API_KEY"] # Add others if needed
    missing_vars = [k for k in required_vars if not os.getenv(k)]
    if missing_vars:
        error_msg = f"Agent Eval Task Failed: Missing required environment variables: {', '.join(missing_vars)}."
        task_log.error(error_msg)
        raise ValueError(error_msg)

    task_log.info("Agent Eval Task: Required environment variables appear to be present.")
    task_log.info(f"Using MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    task_log.info(f"Using LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")

    try:
        task_log.info("Running Agent evaluation via evaluate_agent()...")
        # Ensure evaluate_agent uses env vars correctly
        evaluate_agent() # Assumes this function raises errors on failure
        task_log.info("--- Agent Evaluation Task Completed Successfully ---")

    except Exception as e:
        task_log.error(f"An error occurred during Agent evaluation task: {e}", exc_info=True)
        raise # Re-raise exception to fail the Airflow task


# --- Define DAG ---
with DAG(
    dag_id='model_evaluation_pipeline_dag', # Keep the specific ID
    default_args={
        'owner': 'Vinyas',
        'start_date': pendulum.datetime(2024, 4, 18, tz="UTC"), # Keep start date in the past
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=3),
        'email_on_failure': False,
        'email_on_retry': False,
    },
    # --- MODIFIED ---
    description='Runs scheduled RAG and Agent evaluations, logging results to MLflow.',
    schedule='@daily', # <<< CHANGED: Set a schedule (e.g., daily at midnight UTC)
    catchup=False, # <<< IMPORTANT: Ensures it only runs for the latest interval on startup
    is_paused_upon_creation=False, # <<< ADDED: Starts the DAG in unpaused state
    # --- END MODIFIED ---
    tags=['evaluation', 'rag', 'agent', 'mlflow', 'scheduled'], # Added scheduled tag
    template_searchpath=PROJECT_ROOT
) as dag:

    # Task 1: Run RAG Evaluation Comparison
    rag_evaluation_task = PythonOperator(
        task_id='run_rag_evaluation',
        python_callable=run_rag_eval_wrapper, # Call the wrapper
        execution_timeout=timedelta(hours=1),
    )

    # Task 2: Run Agent Evaluation
    agent_evaluation_task = PythonOperator(
        task_id='run_agent_evaluation',
        python_callable=run_agent_eval_wrapper, # Call the wrapper
        execution_timeout=timedelta(hours=1),
    )

    # Define task dependencies: Run evaluations in parallel (assuming independence)
    # No lines needed here if they run in parallel