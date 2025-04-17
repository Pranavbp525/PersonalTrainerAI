# dags/model_evaluation_pipeline_dag.py

from __future__ import annotations
import pendulum
import os
import logging
import traceback
from datetime import timedelta
from airflow.utils.state import DagRunState
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

# --- Configuration ---
# Determine project root relative to the DAG file location within Airflow
# Assumes dags/ is one level below the project root mapped in docker-compose as /opt/airflow/dags
# and src/ is mapped as /opt/airflow/src
# Airflow workers run tasks often with /opt/airflow as the CWD
PROJECT_ROOT = "/opt/airflow" # Base directory inside the container
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
# Define output dir relative to the mapped project root if needed outside MLflow
# Or rely solely on MLflow artifacts. Let's define it for consistency for now.
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "airflow_eval_output")

# Set up basic logging for the DAG file parsing itself
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Add src directory to Python path *if needed* by the worker for imports.
# Often, imports like `from src.module...` work if CWD is /opt/airflow
# or if PYTHONPATH includes /opt/airflow
# Let's keep your pattern for now, but it might be redundant.
import sys
if SRC_DIR not in sys.path:
    log.info(f"Adding {SRC_DIR} to sys.path for DAG definition/worker.")
    sys.path.insert(0, SRC_DIR)

# --- Task Function for RAG Evaluation ---

def run_rag_evaluation_task(**context):
    """
    Airflow Task: Runs the advanced RAG evaluation comparison.
    Relies on internal MLflow logging within AdvancedRAGEvaluator.
    """
    task_log = logging.getLogger("airflow.task") # Use Airflow's task logger
    task_log.info("--- Starting RAG Evaluation Task ---")
    task_log.info(f"Using PROJECT_ROOT (inside container): {PROJECT_ROOT}")
    task_log.info(f"Using SRC_DIR (inside container): {SRC_DIR}")
    task_log.info(f"Defining output directory (if used by script): {EVAL_OUTPUT_DIR}")

    # --- Environment Variable Checks ---
    required_vars = {
        "MLFLOW_TRACKING_URI": os.getenv('MLFLOW_TRACKING_URI'),
        "LANGSMITH_API_KEY": os.getenv('LANGSMITH_API_KEY'),
        "LANGSMITH_PROJECT": os.getenv('LANGSMITH_PROJECT'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY') # Switched check to OpenAI key
        # "DEEPSEEK_API_KEY": os.getenv('DEEPSEEK_API_KEY') # Or your judge LLM key name
    }
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        error_msg = f"RAG Eval Task Failed: Missing required environment variables: {', '.join(missing_vars)}. Ensure .env file is mounted and loaded."
        task_log.error(error_msg)
        raise ValueError(error_msg)
    else:
        task_log.info("RAG Eval Task: Required environment variables are present.")
        # Log the MLflow URI being used by the task
        task_log.info(f"MLFLOW_TRACKING_URI detected: {required_vars['MLFLOW_TRACKING_URI']}")


    # --- Ensure Output Directory Exists (Optional but good practice) ---
    try:
        os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
        task_log.info(f"Ensured output directory exists: {EVAL_OUTPUT_DIR}")
    except OSError as e:
        task_log.error(f"Could not create output directory {EVAL_OUTPUT_DIR}: {e}")
        raise # Fail the task if directory creation fails

    # --- Execute RAG Evaluation ---
    try:
        # Import the evaluator class *inside* the task function
        from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator
        # Ensure dotenv is loaded if the script relies on it (it should)
        from dotenv import load_dotenv
        env_path = os.path.join(PROJECT_ROOT, ".env") # Path inside container
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
            task_log.info(f"Loaded .env file from {env_path}")
        else:
             task_log.warning(f".env file not found at {env_path}. Relying on pre-set environment variables.")


        task_log.info("Initializing AdvancedRAGEvaluator...")
        evaluator = AdvancedRAGEvaluator(
            output_dir=EVAL_OUTPUT_DIR,
            # Rely on defaults or env vars loaded within the class for models
        )

        task_log.info("Running comparison of RAG implementations via evaluator.compare_implementations()...")
        # This method should contain the logic to evaluate *and* log to MLflow
        comparison_results = evaluator.compare_implementations()

        task_log.info("RAG Implementation Comparison Finished.")
        if comparison_results and comparison_results.get("best_implementation"):
             task_log.info(f"Best RAG Implementation determined: {comparison_results['best_implementation']}")
             task_log.info(f"Best Overall Score: {comparison_results.get('best_score', 'N/A'):.2f}")
             # Log summary to Airflow logs as well
             evaluator.print_summary(comparison_results) # Print summary to task log
        else:
             task_log.warning("Could not determine best RAG implementation from results, or results were empty.")

        task_log.info("--- RAG Evaluation Task Successfully Completed ---")
        return True # Indicate success to Airflow

    except ImportError as e:
        task_log.error(f"Import Error during RAG evaluation task: {e}")
        task_log.error("Ensure 'src' directory is available and dependencies are installed in the worker environment.")
        task_log.error(f"Current sys.path: {sys.path}")
        raise
    except Exception as e:
        task_log.error(f"An error occurred during RAG evaluation task: {e}")
        task_log.error(traceback.format_exc()) # Log full traceback
        raise


# --- Task Function for Agent Evaluation ---

def run_agent_evaluation_task(**context):
    """
    Airflow Task: Runs the agent evaluation script.
    Relies on internal MLflow logging within the evaluate_agent function.
    """
    task_log = logging.getLogger("airflow.task") # Use Airflow's task logger
    task_log.info("--- Starting Agent Evaluation Task ---")
    task_log.info(f"Using PROJECT_ROOT (inside container): {PROJECT_ROOT}")
    task_log.info(f"Using SRC_DIR (inside container): {SRC_DIR}")

    # --- Environment Variable Checks ---
    required_vars = {
        "MLFLOW_TRACKING_URI": os.getenv('MLFLOW_TRACKING_URI'),
        "LANGSMITH_API_KEY": os.getenv('LANGSMITH_API_KEY'),
        "LANGSMITH_PROJECT": os.getenv('LANGSMITH_PROJECT'),
        "DEEPSEEK_API_KEY": os.getenv('DEEPSEEK_API_KEY') # Or your judge LLM key name
    }
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        error_msg = f"Agent Eval Task Failed: Missing required environment variables: {', '.join(missing_vars)}. Ensure .env file is mounted and loaded."
        task_log.error(error_msg)
        raise ValueError(error_msg)
    else:
        task_log.info("Agent Eval Task: Required environment variables are present.")
        task_log.info(f"MLFLOW_TRACKING_URI detected: {required_vars['MLFLOW_TRACKING_URI']}")
        task_log.info(f"LANGSMITH_PROJECT detected: {required_vars['LANGSMITH_PROJECT']}")


    # --- Execute Agent Evaluation ---
    try:
        # Import the main evaluation function *inside* the task
        from src.chatbot.agent_eval.eval import evaluate_agent
        # Ensure dotenv is loaded if the script relies on it (it should)
        from dotenv import load_dotenv
        env_path = os.path.join(PROJECT_ROOT, ".env") # Path inside container
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
            task_log.info(f"Loaded .env file from {env_path}")
        else:
             task_log.warning(f".env file not found at {env_path}. Relying on pre-set environment variables.")


        task_log.info("Running Agent evaluation via evaluate_agent()...")
        # Call the function which performs LangSmith fetching, judging, and MLflow logging
        average_accuracy = evaluate_agent() # Use defaults defined within the script for now

        if average_accuracy is not None:
            task_log.info(f"Agent evaluation completed. Average Accuracy reported: {average_accuracy:.2f}")
        else:
            task_log.warning("Agent evaluation finished, but no average accuracy score was returned/calculated.")

        task_log.info("--- Agent Evaluation Task Successfully Completed ---")
        return True # Indicate success to Airflow

    except ImportError as e:
        task_log.error(f"Import Error during Agent evaluation task: {e}")
        task_log.error("Ensure 'src' directory is available and dependencies are installed in the worker environment.")
        task_log.error(f"Current sys.path: {sys.path}")
        raise
    except Exception as e:
        task_log.error(f"An error occurred during Agent evaluation task: {e}")
        task_log.error(traceback.format_exc()) # Log full traceback
        raise


# --- Define DAG ---
with DAG(
    dag_id='model_evaluation_pipeline', # Renamed DAG ID
    default_args={
        'owner': 'Your Name / Team', # Change owner
        'start_date': pendulum.datetime(2024, 1, 1, tz="UTC"), # Adjust start date
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=3),
        'catchup': False,
        'email_on_failure': False, # Configure as needed
        'email_on_retry': False,
    },
    description='Runs RAG and Agent evaluations using MLflow after data pipeline completion.',
    schedule=None, # Trigger manually or after the data pipeline
    tags=['evaluation', 'rag', 'agent', 'mlflow'],
    template_searchpath=PROJECT_ROOT # Allows referencing files relative to project root if needed in templates
) as dag:

    # Task 1: Wait for the Data_pipeline_dag to succeed
    wait_for_data_pipeline = ExternalTaskSensor(
        task_id='wait_for_data_pipeline_completion',
        external_dag_id='Data_pipeline_dag', # *** MAKE SURE THIS MATCHES ***
        external_task_id=None, # Wait for the entire DAG run to succeed
        allowed_states=[DagRunState.SUCCESS], # Use DagRunState enum
        failed_states=[DagRunState.FAILED],    # Use DagRunState enum
        mode='poke',
        timeout=60 * 60 * 3,
        poke_interval=120,
    )

    # Task 2: Run RAG Evaluation Comparison
    rag_evaluation_task = PythonOperator(
        task_id='run_rag_evaluation',
        python_callable=run_rag_evaluation_task,
    )

    # Task 3: Run Agent Evaluation
    agent_evaluation_task = PythonOperator(
        task_id='run_agent_evaluation',
        python_callable=run_agent_evaluation_task,
    )

    # Define task dependencies: Wait, then run evaluations in parallel
    wait_for_data_pipeline >> [rag_evaluation_task, agent_evaluation_task]