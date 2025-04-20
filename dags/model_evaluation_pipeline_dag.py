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
# from airflow.operators.bash import BashOperator # Removed if no bash needed

# --- Configuration ---
# Use the project root defined by the docker-compose volume mount
PROJECT_ROOT = "/opt/airflow/app" # <<< Correct project root inside container
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
# Output directory inside container (optional, only needed if scripts write locally before GCS/MLflow)
# EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "airflow_eval_output")

# --- Logging Setup ---
log = logging.getLogger(__name__)
# No need to setLevel here, Airflow task handler configures logging

# --- Add src directory to Python path ---
# Ensure this path is correct based on docker-compose volume mount
if SRC_DIR not in sys.path:
    log.info(f"Adding {SRC_DIR} to sys.path for DAG definition/worker.")
    sys.path.insert(0, SRC_DIR)

# --- Import Task Functions ---
# Import the main execution functions directly from the refactored scripts
try:
    # Import the class for RAG eval
    from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator
    # Import the function for Agent eval
    from src.chatbot.agent_eval.eval import evaluate_agent
    IMPORT_SUCCESS = True
    log.info("Successfully imported evaluation functions/classes.")
except ImportError as e:
    log.error(f"Failed to import evaluation functions: {e}", exc_info=True)
    IMPORT_SUCCESS = False
    # Define dummy functions if import fails to allow DAG parsing
    def run_rag_eval_wrapper(**kwargs): raise ImportError("advanced_rag_evaluation module/class not found")
    def run_agent_eval_wrapper(**kwargs): raise ImportError("agent_eval.eval module/function not found")


# --- Wrapper function for RAG Evaluation Task ---
def run_rag_eval_wrapper(**context):
    """
    Sets up and runs the Advanced RAG Evaluator.
    Relies on environment variables being set via docker-compose (.env mount).
    """
    task_log = logging.getLogger("airflow.task")
    task_log.info("--- Starting RAG Evaluation Task ---")

    # Check for essential environment variables needed by the evaluator or RAG models
    required_vars = ["MLFLOW_TRACKING_URI", "OPENAI_API_KEY", "PINECONE_API_KEY"] # Add others if needed
    missing_vars = [k for k in required_vars if not os.getenv(k)]
    if missing_vars:
        error_msg = f"RAG Eval Task Failed: Missing required environment variables: {', '.join(missing_vars)}."
        task_log.error(error_msg)
        raise ValueError(error_msg) # Fail task if critical env vars missing

    task_log.info("RAG Eval Task: Required environment variables appear to be present.")
    task_log.info(f"Using MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")

    try:
        # Define a temporary output directory inside the container if needed by evaluator internals
        # Otherwise, this might not be needed if all output goes direct to GCS/MLflow
        temp_output_dir = "/tmp/rag_eval_output"
        os.makedirs(temp_output_dir, exist_ok=True)
        task_log.info(f"Ensured temp output directory exists: {temp_output_dir}")

        task_log.info("Initializing AdvancedRAGEvaluator...")
        # Pass the temporary dir if the class needs it, otherwise remove output_dir arg
        evaluator = AdvancedRAGEvaluator(output_dir=temp_output_dir)

        task_log.info("Running comparison of RAG implementations...")
        # The compare_implementations method now handles MLflow logging and GCS saving
        comparison_results = evaluator.compare_implementations()
        task_log.info("RAG Implementation Comparison Finished.")

        # Add a final check or summary log if needed
        if not comparison_results or "error" in comparison_results:
             task_log.warning(f"RAG evaluation completed but reported an error or no results: {comparison_results}")
             # Decide if this should fail the task
             # raise RuntimeError("RAG Evaluation script reported an error or no results.")

        task_log.info("--- RAG Evaluation Task Completed ---")
        # No return needed - success implied by lack of exceptions

    except Exception as e:
        task_log.error(f"An error occurred during RAG evaluation task: {e}", exc_info=True)
        raise # Re-raise exception to fail the Airflow task

# --- Wrapper function for Agent Evaluation Task ---
def run_agent_eval_wrapper(**context):
    """
    Runs the Agent evaluation script.
    Relies on environment variables being set via docker-compose (.env mount).
    """
    task_log = logging.getLogger("airflow.task")
    task_log.info("--- Starting Agent Evaluation Task ---")

    # Check for essential environment variables
    required_vars = ["MLFLOW_TRACKING_URI", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "OPENAI_API_KEY"] # Add others if needed
    missing_vars = [k for k in required_vars if not os.getenv(k)]
    if missing_vars:
        error_msg = f"Agent Eval Task Failed: Missing required environment variables: {', '.join(missing_vars)}."
        task_log.error(error_msg)
        raise ValueError(error_msg) # Fail task

    task_log.info("Agent Eval Task: Required environment variables appear to be present.")
    task_log.info(f"Using MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    task_log.info(f"Using LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")

    try:
        task_log.info("Running Agent evaluation via evaluate_agent()...")
        # Directly call the imported function
        evaluate_agent() # The function now handles MLflow logging internally and raises errors
        task_log.info("--- Agent Evaluation Task Completed ---")
        # No return needed - success implied by lack of exceptions

    except Exception as e:
        task_log.error(f"An error occurred during Agent evaluation task: {e}", exc_info=True)
        raise # Re-raise exception to fail the Airflow task


# --- Define DAG ---
with DAG(
    dag_id='model_evaluation_pipeline_dag', # Use the specific ID
    default_args={
        'owner': 'Vinyas', # Set consistent owner
        'start_date': pendulum.datetime(2024, 4, 18, tz="UTC"), # Align start date
        'depends_on_past': False,
        'retries': 1, # Adjust retries if needed for flaky evals
        'retry_delay': timedelta(minutes=3),
        'catchup': False,
        'email_on_failure': False, # Set to True and add email if needed
        'email_on_retry': False,
    },
    description='Runs RAG and Agent evaluations, logging results to MLflow. Triggered by Data_pipeline_dag.',
    schedule=None, # This DAG is triggered
    tags=['evaluation', 'rag', 'agent', 'mlflow'],
    template_searchpath=PROJECT_ROOT # Use updated PROJECT_ROOT
) as dag:

    if not IMPORT_SUCCESS:
        # If imports failed, create a single dummy task to indicate the error
        import_error_task = BashOperator(
            task_id='import_error',
            bash_command='echo "CRITICAL: Failed to import evaluation functions. Check Airflow logs and sys.path configuration." && exit 1',
        )
    else:
        # Task 1: Run RAG Evaluation Comparison
        rag_evaluation_task = PythonOperator(
            task_id='run_rag_evaluation',
            python_callable=run_rag_eval_wrapper, # Call the wrapper
            execution_timeout=timedelta(hours=1), # Add timeout for safety
        )

        # Task 2: Run Agent Evaluation
        agent_evaluation_task = PythonOperator(
            task_id='run_agent_evaluation',
            python_callable=run_agent_eval_wrapper, # Call the wrapper
            execution_timeout=timedelta(hours=1), # Add timeout for safety
        )

        # Define task dependencies: Run evaluations in parallel
        # If agent eval depends on RAG eval completing, set dependency:
        # rag_evaluation_task >> agent_evaluation_task
        # If they can run independently, no dependency line is needed between them.
        # Assuming they can run in parallel for now.